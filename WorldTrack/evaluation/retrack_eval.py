"""
Retrack evaluation engine.

For one trained checkpoint:
  1. run the network once per camera-removal condition (all cameras, every combination of 1..max_drop removed
     cameras) on the test split and cache the decoded BEV detections, their scores and (optionally) the
     multi-view re-ID descriptors;
  2. score the cached detections offline under every evaluated configuration:
       S0 baseline        threshold 0.5, TrackTacular tracker (track buffer 5)
       S1 threshold       tuned detection threshold, TrackTacular tracker
       S2 byte            tuned threshold, ByteTrack-style two-pass association, longer track buffer
       S3 byte+reid       S2 + camera-dropout-aware appearance cost
     plus (with --grids) the full threshold sweep and the Stage 2 / Stage 3 parameter grids.

Detection: MODA, MODP, precision, recall (0.5 m). Tracking: MOTA, IDF1, MT, ML, IDs, Frag (1 m), with raw counts.

usage (from WorldTrack/):
    python -m evaluation.retrack_eval --ckpt model.ckpt --data_dir /content/data/Wildtrack --out results/eval/x
"""
import argparse
import inspect
import itertools
import json
import os
import os.path as osp
import shutil
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from evaluation import mot_bev  # noqa: F401  (numpy 2 shim for motmetrics)
import motmetrics as mm
from tracking.multitracker import build_tracker
from utils import decode

THRESHOLDS = [round(t, 2) for t in np.arange(0.20, 0.76, 0.05)]
MOD_DIST = 20  # 0.5 m in 2.5 cm grid cells
MOT_SCALE = 0.025  # grid cell -> metres; tracking match threshold is 1 m

# a-priori settings of the Retrack stages (the grids show the alternatives)
S2_DEFAULTS = dict(tracker='byte', low_thres=0.1, track_buffer=10, low_match_thresh=50.)
S3_DEFAULTS = dict(S2_DEFAULTS, use_reid=True, reid_lambda=0.98, reid_thresh=0.6)
S2_GRID = [dict(S2_DEFAULTS, low_thres=lo, track_buffer=buf) for lo in (0.1, 0.2, 0.3) for buf in (5, 10, 20)]
S3_GRID = [dict(S2_DEFAULTS, use_reid=True, reid_lambda=lam, reid_thresh=th)
           for lam in (0.5, 0.9, 0.98) for th in (0.4, 0.6, 0.8)]


# ------------------------------------------------------------------ model / data
def load_model(ckpt_path, device):
    from world_track import WorldTrackModel
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # LightningCLI also stores entries such as `_instantiator` that are not constructor arguments
    accepted = inspect.signature(WorldTrackModel.__init__).parameters
    hparams = {k: v for k, v in ckpt.get('hyper_parameters', {}).items() if k in accepted}
    model = WorldTrackModel(**hparams)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    assert not missing and not unexpected, f'checkpoint mismatch: missing={missing}, unexpected={unexpected}'
    return model.to(device).eval(), hparams


def load_test_items(data_dir, model, num_workers=4):
    """Loads the whole test split once (images kept as uint8, which is lossless for 8-bit images)."""
    from torch.utils.data import DataLoader
    from datasets.pedestrian_dataset import PedestrianDataset
    from datasets.wildtrack_dataset import Wildtrack
    from datasets.multiviewx_dataset import MultiviewX

    name = osp.basename(osp.normpath(data_dir)).lower()
    base = Wildtrack(data_dir) if 'wildtrack' in name else MultiviewX(data_dir)
    dataset = PedestrianDataset(base, is_train=False, resolution=model.resolution, bounds=model.bounds)
    items = []
    for item, _ in DataLoader(dataset, batch_size=1, num_workers=num_workers):
        item['img'] = (item['img'] * 255).round().to(torch.uint8)
        items.append(item)
    return items


def reset_temporal_cache(model):
    model.temporal_cache = None
    model.temporal_cache_frames = -2 * torch.ones(model.max_cache, dtype=torch.long)


@torch.no_grad()
def run_inference(model, items, drop_cams=(), reid=None, reid_min_score=0.1):
    """Runs the network on the test split with `drop_cams` switched off and decodes the BEV detections."""
    device = model.device
    reset_temporal_cache(model)
    frames = []
    for item in items:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
        batch['img'] = batch['img'].float() / 255
        S = batch['img'].shape[1]
        cam_mask = None
        if drop_cams:
            cam_mask = torch.ones(1, S, device=device)
            cam_mask[:, list(drop_cams)] = 0
            batch['cam_mask'] = cam_mask
        output = model(batch)

        xy_e, xy_prev_e, scores_e, _, _, _ = decode.decoder(
            output['instance_center'].sigmoid(), output['instance_offset'], output['instance_size'],
            rz_e=output['instance_rot'], K=model.max_detections)
        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = model.vox_util.Mem2Ref(mem_xyz, model.Y, model.Z, model.X)[..., :2]
        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy_prev = model.vox_util.Mem2Ref(mem_xyz_prev, model.Y, model.Z, model.X)[..., :2]

        scores = scores_e[0]
        feats = None
        if reid is not None:
            keep = scores >= reid_min_score
            feats = np.full((len(scores), reid.model.feature_dim), np.nan, np.float32)
            if keep.any():
                feats[keep.cpu().numpy()] = reid(batch['img'][0], batch['intrinsic'][0], batch['extrinsic'][0],
                                                 batch['ref_T_global'][0], ref_xy[0][keep],
                                                 None if cam_mask is None else cam_mask[0])[0]
        grid_gt = item['grid_gt'][0]
        frames.append({
            'frame': int(item['frame'][0]),
            'seq': int(item['sequence_num'][0]),
            'gt': grid_gt[grid_gt.sum(1) != 0].numpy().astype(np.float64),  # x, y, pid
            # float32 as in WorldTrackModel.test_step, so the baseline configuration reproduces it exactly
            'xy': ref_xy[0].cpu().numpy(),
            'xy_prev': ref_xy_prev[0].cpu().numpy(),
            'score': scores.cpu().numpy(),
            'feat': feats,
        })
    return frames


# ------------------------------------------------------------------ metrics
def detection_metrics(frames, thr, td=MOD_DIST):
    """CLEAR MOD (as evaluation/CLEAR_MOD_HUN.py) computed over every ground-truth frame."""
    tp = fp = fn = n_gt = 0
    dist_sum = 0.0
    for f in frames:
        gt = f['gt'][:, :2]
        det = f['xy'][f['score'] > thr].astype(np.float64)
        n_gt += len(gt)
        matched = 0
        if len(gt) and len(det):
            dist = np.linalg.norm(gt[:, None] - det[None], axis=2)
            cost = dist.copy()
            cost[cost > td] = 1e6
            r, c = linear_sum_assignment(cost)
            ok = dist[r, c] < td
            matched = int(ok.sum())
            dist_sum += float(np.sum(1 - dist[r, c][ok] / td))
        tp += matched
        fp += len(det) - matched
        fn += len(gt) - matched
    return {
        'moda': (1 - (fn + fp) / max(n_gt, 1)) * 100,
        'modp': dist_sum / tp * 100 if tp else 0.0,
        'precision': tp / (tp + fp) * 100 if tp + fp else 0.0,
        'recall': tp / max(n_gt, 1) * 100,
        'det_tp': tp, 'det_fp': fp, 'det_fn': fn, 'num_gt': n_gt,
    }


TRACK_KEYS = ['mota', 'motp', 'idf1', 'idp', 'idr', 'mostly_tracked', 'partially_tracked', 'mostly_lost',
              'num_switches', 'num_fragmentations', 'num_misses', 'num_false_positives', 'num_unique_objects',
              'num_objects', 'num_predictions']


def tracking_metrics(frames, thr, tracker_kwargs, scale=MOT_SCALE):
    """Runs a tracker over the cached detections and computes CLEAR MOT / ID metrics (as on_test_epoch_end)."""
    kwargs = dict(tracker_kwargs)
    tracker = build_tracker(conf_thres=thr, **kwargs)
    use_reid = kwargs.get('use_reid', False)
    acc = mm.MOTAccumulator()
    for f in frames:
        if use_reid:
            stracks = tracker.update(f['xy'], f['xy_prev'], f['score'], feats=f['feat'])
        else:
            stracks = tracker.update(torch.from_numpy(f['xy']), torch.from_numpy(f['xy_prev']),
                                     torch.from_numpy(f['score']))
        hyp_ids = [s.track_id for s in stracks]
        hyp_xy = np.array([np.asarray(s.xy, dtype=np.float64) for s in stracks]).reshape(-1, 2)
        gt = f['gt']
        C = np.sqrt(mm.distances.norm2squared_matrix(gt[:, :2] * scale, hyp_xy * scale, max_d2=1))
        acc.update(gt[:, 2].astype(int).tolist(), hyp_ids, C, frameid=f['frame'])

    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=TRACK_KEYS, name='all').iloc[0].to_dict()
    n = max(s['num_unique_objects'], 1)
    return {
        'mota': s['mota'] * 100, 'idf1': s['idf1'] * 100, 'idp': s['idp'] * 100, 'idr': s['idr'] * 100,
        'motp': (1 - s['motp']) * 100 if np.isfinite(s['motp']) else 0.0,
        'mt': s['mostly_tracked'] / n * 100, 'pt': s['partially_tracked'] / n * 100, 'ml': s['mostly_lost'] / n * 100,
        'id_switches': int(s['num_switches']), 'fragmentations': int(s['num_fragmentations']),
        'trk_misses': int(s['num_misses']), 'trk_fp': int(s['num_false_positives']),
        'num_unique_objects': int(s['num_unique_objects']),
    }


def tracker_name(kw):
    if kw['tracker'] == 'baseline':
        return f"baseline(buf={kw.get('track_buffer', 5)})"
    name = f"byte(low={kw['low_thres']},buf={kw['track_buffer']})"
    if kw.get('use_reid'):
        name += f"+reid(lam={kw['reid_lambda']},th={kw['reid_thresh']})"
    return name


def evaluate_config(frames, method, thr, tracker_kwargs):
    row = {'method': method, 'thr': thr, 'tracker': tracker_name(tracker_kwargs)}
    row.update(detection_metrics(frames, thr))
    row.update(tracking_metrics(frames, thr, tracker_kwargs))
    return row


def pick_threshold(sweep_rows):
    """Threshold with the best MODA (ties -> closest to the 0.5 default)."""
    best = max(sweep_rows, key=lambda r: (round(r['moda'], 6), -abs(r['thr'] - 0.5)))
    return best['thr']


def camera_conditions(num_cams, max_drop, max_combos=None, seed=0):
    rng = np.random.default_rng(seed)
    conds = []
    for k in range(0, max_drop + 1):
        combos = list(itertools.combinations(range(num_cams), k))
        if max_combos is not None and len(combos) > max_combos:
            combos = [combos[i] for i in sorted(rng.choice(len(combos), max_combos, replace=False))]
        conds += combos
    return conds


# ------------------------------------------------------------------ driver
def evaluate_checkpoint(ckpt, data_dir, out_dir, tag, variant='', seed=-1, max_drop=3, max_combos=None,
                        grids=False, use_reid=True, fixed_thr=None, device=None, num_workers=4,
                        keep_cache=False):
    os.makedirs(out_dir, exist_ok=True)
    done_path = osp.join(out_dir, 'metrics.csv')
    if osp.exists(done_path):
        print(f'[{tag}] already evaluated -> {done_path}')
        return pd.read_csv(done_path)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, hparams = load_model(ckpt, device)
    print(f'[{tag}] {ckpt}\n  model={hparams.get("model_name")} fusion={hparams.get("fusion", "concat")} '
          f'coverage={hparams.get("coverage_map", False)} cam_dropout={hparams.get("cam_dropout_prob", 0.0)}')
    items = load_test_items(data_dir, model, num_workers=num_workers)
    num_cams = items[0]['img'].shape[1]

    reid = None
    if use_reid:
        from tracking.reid import BEVReIDExtractor
        reid = BEVReIDExtractor(person_height=0.9 * (model.bounds[5] - model.bounds[4]), device=device,
                                z_sign=hparams.get('z_sign', 1))

    cache_dir = osp.join(out_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    conds = camera_conditions(num_cams, max_drop, max_combos)
    rows = []
    tuned_thr = fixed_thr
    t0 = time.time()
    for ci, drop in enumerate(conds):
        cache = osp.join(cache_dir, 'drop_' + ('-'.join(map(str, drop)) or 'none') + '.pt')
        if osp.exists(cache):
            frames = torch.load(cache, weights_only=False)
        else:
            frames = run_inference(model, items, drop, reid)
            torch.save(frames, cache)

        base = {'tag': tag, 'variant': variant, 'seed': seed, 'k': len(drop),
                'dropped': '-'.join(str(c + 1) for c in drop) or 'none'}
        cond_rows = []
        if len(drop) == 0:
            sweep = [evaluate_config(frames, 'sweep', thr, dict(tracker='baseline', track_buffer=5))
                     for thr in THRESHOLDS]
            cond_rows += sweep
            if tuned_thr is None:
                tuned_thr = pick_threshold(sweep)
                print(f'[{tag}] tuned detection threshold: {tuned_thr}')
        cond_rows.append(evaluate_config(frames, 'S0_baseline', 0.5, dict(tracker='baseline', track_buffer=5)))
        cond_rows.append(evaluate_config(frames, 'S1_threshold', tuned_thr, dict(tracker='baseline', track_buffer=5)))
        cond_rows.append(evaluate_config(frames, 'S2_byte', tuned_thr, S2_DEFAULTS))
        if reid is not None:
            cond_rows.append(evaluate_config(frames, 'S3_byte_reid', tuned_thr, S3_DEFAULTS))
        if grids:
            cond_rows += [evaluate_config(frames, 'grid_S2', tuned_thr, kw) for kw in S2_GRID]
            if reid is not None:
                cond_rows += [evaluate_config(frames, 'grid_S3', tuned_thr, kw) for kw in S3_GRID]
        rows += [dict(base, **r) for r in cond_rows]
        s0 = [r for r in cond_rows if r['method'] == 'S0_baseline'][0]
        print(f'[{tag}] {ci + 1}/{len(conds)} dropped={base["dropped"]:>8s}  MODA {s0["moda"]:6.2f}  '
              f'MOTA {s0["mota"]:6.2f}  IDF1 {s0["idf1"]:6.2f}  ({time.time() - t0:.0f}s)')

    df = pd.DataFrame(rows)
    df.to_csv(done_path + '.part', index=False)
    os.replace(done_path + '.part', done_path)
    with open(osp.join(out_dir, 'info.json'), 'w') as f:
        json.dump({'ckpt': ckpt, 'tuned_thr': tuned_thr, 'hparams': {k: str(v) for k, v in hparams.items()}}, f,
                  indent=1)
    if not keep_cache:  # the per-condition caches only serve to resume an interrupted evaluation
        shutil.rmtree(cache_dir, ignore_errors=True)
    return df


@torch.no_grad()
def lightning_test(ckpt, data_dir, log_dir):
    """Runs the unmodified Lightning `test` path (TrackTacular's own evaluation) for a sanity comparison."""
    import lightning as pl
    from datasets.pedestrian_datamodule import PedestrianDataModule
    model, _ = load_model(ckpt, 'cuda' if torch.cuda.is_available() else 'cpu')
    dm = PedestrianDataModule(data_dir=data_dir, batch_size=1, num_workers=4, resolution=model.resolution,
                              bounds=model.bounds)
    logger = pl.pytorch.loggers.CSVLogger(log_dir, name='lightning_test')
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1, logger=logger,
                         enable_progress_bar=False)
    res = trainer.test(model, datamodule=dm, verbose=False)
    return {k: float(v) for k, v in res[0].items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tag', default='model')
    p.add_argument('--variant', default='')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--max_drop', type=int, default=3)
    p.add_argument('--max_combos', type=int, default=None)
    p.add_argument('--grids', action='store_true')
    p.add_argument('--no_reid', action='store_true')
    p.add_argument('--thr', type=float, default=None, help='fixed tuned threshold instead of the MODA-best one')
    p.add_argument('--keep_cache', action='store_true', help='keep the cached detections / re-ID descriptors')
    args = p.parse_args()
    torch.set_float32_matmul_precision('medium')
    evaluate_checkpoint(args.ckpt, args.data_dir, args.out, args.tag, args.variant, args.seed, args.max_drop,
                        args.max_combos, args.grids, not args.no_reid, args.thr, keep_cache=args.keep_cache)
