"""
Evaluation of one checkpoint (TrackTacular baseline or R2) on the test split, with cameras switched off.

For every camera condition (all cameras, then every combination of 1..max_drop removed cameras):
  detection  MODA, MODP, precision, recall, TP / FP / FN at threshold 0.5 and 0.5 m (as evaluation/mod.py)
  tracking   TrackTacular's tracker (tracking/multitracker.py, buffer 5) on the same detections:
             MOTA, IDF1, MT, ML, ID switches at 1 m (as evaluation/mot_bev.py)

Also used by the Colab visual tools (colab/viz.py): load_model, load_test_items, run_inference, match_frame.

usage (from WorldTrack/):
    python -m evaluation.r2_eval --ckpt last.ckpt --data_dir /content/data/Wildtrack --out <results>/eval/x
"""
import argparse
import inspect
import itertools
import json
import os
import os.path as osp
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from evaluation import mot_bev  # noqa: F401  (numpy 2 shim for motmetrics)
import motmetrics as mm
from tracking.multitracker import JDETracker
from utils import decode

THR = 0.5
MOD_DIST = 20  # 0.5 m in 2.5 cm grid cells (both datasets)
MOT_SCALE = 0.025  # grid cell -> metres; the tracking match threshold is 1 m
TRACK_BUFFER = 5

# hparams of research variants that this branch does not implement: loading them would silently drop modules
UNSUPPORTED = {
    'da3': 'R1 (Depth Anything 3 features)',
    'track_queries': 'R4 (track queries)',
    'vgf': 'volumetric-guided fusion (methodology 2)',
    'ray_cond': 'ray conditioning (methodology 2)',
    'fg_background': 'foreground gate (methodology 4)',
    'coverage_map': 'coverage map (Retrack stage 4)',
}


class CheckpointNotSupported(ValueError):
    pass


def read_hparams(ckpt_path):
    """Hyper-parameters stored in a Lightning checkpoint, without loading the weights into memory."""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001  (older formats cannot be memory-mapped)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return dict(ckpt.get('hyper_parameters', {})), {k: ckpt.get(k) for k in ('epoch', 'global_step')}


def variant_of(hparams):
    """'B0' (TrackTacular), 'R2', or a description of the unsupported research variant."""
    bad = [name for key, name in UNSUPPORTED.items() if hparams.get(key)]
    if hparams.get('fusion', 'concat') != 'concat':
        bad.append(f"fusion={hparams['fusion']}")
    if hparams.get('model_name', 'segnet') != 'segnet':
        bad.append(f"model={hparams['model_name']}")
    if bad:
        return 'unsupported: ' + ', '.join(bad)
    return 'R2' if hparams.get('sparse_queries') else 'B0'


def dataset_of(hparams):
    res = tuple(hparams.get('resolution', ()))
    return {(120, 4, 360): 'wildtrack', (160, 2, 250): 'multiviewx'}.get(res, f'unknown{res}')


def load_model(ckpt_path, device):
    """Builds WorldTrackModel from the checkpoint's hyper-parameters and loads the weights (strict)."""
    from world_track import WorldTrackModel
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp_all = dict(ckpt.get('hyper_parameters', {}))
    variant = variant_of(hp_all)
    if variant.startswith('unsupported'):
        raise CheckpointNotSupported(f'{ckpt_path}: {variant}. This branch has the TrackTacular baseline and R2 only; '
                                     'use the branch that trained it.')
    # LightningCLI also stores entries such as `_instantiator`, and older branches stored extra options
    accepted = inspect.signature(WorldTrackModel.__init__).parameters
    hparams = {k: v for k, v in hp_all.items() if k in accepted}
    model = WorldTrackModel(**hparams)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if missing or unexpected:
        raise CheckpointNotSupported(f'{ckpt_path}: weights do not match this code '
                                     f'(missing {missing[:5]}, unexpected {unexpected[:5]})')
    model.eval().to(device)
    model.use_temporal_cache = bool(hparams.get('use_temporal_cache', False))
    return model, hp_all


def load_test_items(data_dir, model, num_workers=2):
    """The whole test split (last 10 % of the frames), images kept as uint8."""
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
    return items, dataset


def reset_temporal_cache(model):
    model.temporal_cache = None
    model.temporal_cache_frames = -2 * torch.ones(model.max_cache, dtype=torch.long)


def to_ref(model, xy_mem):
    """B,K,2 BEV mem coordinates -> reference (2.5 cm grid) coordinates"""
    xyz = torch.cat([xy_mem, torch.zeros_like(xy_mem[..., :1])], dim=-1)
    return model.vox_util.Mem2Ref(xyz, model.Y, model.Z, model.X)[..., :2]


@torch.no_grad()
def run_inference(model, items, drop_cams=(), keep_maps=False):
    """Runs the network on every test frame with `drop_cams` switched off and decodes the BEV detections.
    keep_maps: also keep the BEV heatmap (and, for R2, the proposals and the queries of every layer) per frame."""
    device = model.device
    reset_temporal_cache(model)
    frames = []
    for item in items:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
        batch['img'] = batch['img'].float() / 255
        S = batch['img'].shape[1]
        if drop_cams:
            cam_mask = torch.ones(1, S, device=device)
            cam_mask[:, list(drop_cams)] = 0
            batch['cam_mask'] = cam_mask
        output = model(batch)
        xy_e, xy_prev_e, scores_e, _, _, _ = decode.decoder(
            output['instance_center'].sigmoid(), output['instance_offset'], output['instance_size'],
            rz_e=output['instance_rot'], K=model.max_detections)
        grid_gt = item['grid_gt'][0]
        f = {
            'frame': int(item['frame'][0]),
            'seq': int(item['sequence_num'][0]),
            'gt': grid_gt[grid_gt.sum(1) != 0].numpy().astype(np.float64),  # x, y, pid (reference coordinates)
            'xy': to_ref(model, xy_e)[0].cpu().numpy(),  # float32, as WorldTrackModel.test_step
            'xy_prev': to_ref(model, xy_prev_e)[0].cpu().numpy(),
            'score': scores_e[0].cpu().numpy(),
        }
        if keep_maps:
            heat = output['proposal_center'] if 'proposal_center' in output else output['instance_center']
            f['heat'] = heat[0, 0].sigmoid().float().cpu().numpy()  # Y,X
            if 'query_layers' in output:
                f['queries'] = [{'xy': ly['xy'][0].float().cpu().numpy(),  # mem coordinates
                                 'score': ly['logit'][0].sigmoid().float().cpu().numpy()}
                                for ly in output['query_layers']]
                f['anchors'] = output['query_anchor'][0].float().cpu().numpy() + 0.5
        frames.append(f)
    return frames


# ------------------------------------------------------------------ metrics
def match_frame(gt_xy, det_xy, td=MOD_DIST):
    """Hungarian matching within td (as evaluation/CLEAR_MOD_HUN.py). Returns (gt index, det index) pairs and
    the matched distances."""
    if not len(gt_xy) or not len(det_xy):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    dist = np.linalg.norm(gt_xy[:, None] - det_xy[None], axis=2)
    cost = dist.copy()
    cost[cost > td] = 1e6
    r, c = linear_sum_assignment(cost)
    ok = dist[r, c] < td
    return r[ok], c[ok], dist[r, c][ok]


def detection_metrics(frames, thr=THR, td=MOD_DIST):
    tp = fp = fn = n_gt = 0
    dist_sum = 0.0
    for f in frames:
        gt = f['gt'][:, :2]
        det = f['xy'][f['score'] > thr].astype(np.float64)
        r, _, d = match_frame(gt, det, td)
        n_gt += len(gt)
        tp += len(r)
        fp += len(det) - len(r)
        fn += len(gt) - len(r)
        dist_sum += float(np.sum(1 - d / td))
    return {
        'moda': (1 - (fn + fp) / max(n_gt, 1)) * 100,
        'modp': dist_sum / tp * 100 if tp else 0.0,
        'precision': tp / (tp + fp) * 100 if tp + fp else 0.0,
        'recall': tp / max(n_gt, 1) * 100,
        'det_tp': tp, 'det_fp': fp, 'det_fn': fn, 'num_gt': n_gt,
    }


def run_tracker(frames, thr=THR, track_buffer=TRACK_BUFFER):
    """TrackTacular's tracker over the cached detections. Returns per frame [(track_id, xy), ...]."""
    tracker = JDETracker(conf_thres=thr, track_buffer=track_buffer)
    hyps = []
    for f in frames:
        stracks = tracker.update(torch.from_numpy(f['xy']), torch.from_numpy(f['xy_prev']),
                                 torch.from_numpy(f['score']))
        hyps.append([(int(s.track_id), np.asarray(s.xy, dtype=np.float64)) for s in stracks])
    return hyps


def tracking_metrics(frames, hyps, scale=MOT_SCALE):
    """CLEAR MOT / ID metrics, plus the motmetrics accumulator (its mot_events list the ID switches)."""
    acc = mm.MOTAccumulator()
    for f, hyp in zip(frames, hyps):
        hyp_xy = np.array([h[1] for h in hyp]).reshape(-1, 2)
        gt = f['gt']
        C = np.sqrt(mm.distances.norm2squared_matrix(gt[:, :2] * scale, hyp_xy * scale, max_d2=1))
        acc.update(gt[:, 2].astype(int).tolist(), [h[0] for h in hyp], C, frameid=f['frame'])
    keys = ['mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches', 'num_unique_objects']
    s = mm.metrics.create().compute(acc, metrics=keys, name='all').iloc[0].to_dict()
    n = max(s['num_unique_objects'], 1)
    return {
        'mota': s['mota'] * 100, 'idf1': s['idf1'] * 100,
        'motp': (1 - s['motp']) * 100 if np.isfinite(s['motp']) else 0.0,
        'mt': s['mostly_tracked'] / n * 100, 'ml': s['mostly_lost'] / n * 100,
        'id_switches': int(s['num_switches']),
    }, acc


def camera_conditions(num_cams, max_drop):
    conds = []
    for k in range(0, max_drop + 1):
        conds += list(itertools.combinations(range(num_cams), k))
    return conds


# ------------------------------------------------------------------ CLI
def main():
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max_drop', type=int, default=3)
    p.add_argument('--quick', action='store_true', help='3 frames, all cameras only (smoke test)')
    p.add_argument('--num_workers', type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, hp = load_model(args.ckpt, device)
    items, _ = load_test_items(args.data_dir, model, num_workers=args.num_workers)
    if args.quick:
        items = items[:3]
    S = items[0]['img'].shape[1]
    conds = [()] if args.quick else camera_conditions(S, args.max_drop)
    print(f'{dataset_of(hp)} {variant_of(hp)}: {len(items)} frames, {S} cameras, {len(conds)} camera conditions',
          flush=True)

    rows, t0 = [], time.time()
    for drop in conds:
        frames = run_inference(model, items, drop)
        hyps = run_tracker(frames)
        trk, _ = tracking_metrics(frames, hyps)
        row = {'drop': '-'.join(str(c) for c in drop) or 'none', 'k': len(drop), 'num_cams': S}
        row.update(detection_metrics(frames))
        row.update(trk)
        rows.append(row)
        if not drop:
            torch.save({'frames': frames, 'hyps': hyps}, osp.join(args.out, 'all_cameras_frames.pt'))
            print(f'  all cameras: MODA {row["moda"]:.2f}  MOTA {row["mota"]:.2f}  IDF1 {row["idf1"]:.2f}  '
                  f'(FP {row["det_fp"]}, FN {row["det_fn"]}, ID switches {row["id_switches"]})', flush=True)
    df = pd.DataFrame(rows)
    times = []
    reset_temporal_cache(model)
    for item in items[:12]:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
        batch['img'] = batch['img'].float() / 255
        if device == 'cuda':
            torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            model(batch)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t)
    summary = {
        'ckpt': args.ckpt, 'dataset': dataset_of(hp), 'variant': variant_of(hp), 'seed': hp.get('seed'),
        'params_M': sum(p.numel() for p in model.parameters()) / 1e6,
        'ms_per_frame': float(np.median(times[2:] if len(times) > 4 else times)) * 1000,
        'eval_seconds': time.time() - t0,
        'all_cameras': rows[0],
        'mean_by_k': df.groupby('k')[['moda', 'mota', 'idf1', 'det_fp', 'det_fn']].mean().round(3)
        .to_dict(orient='index'),
    }
    json.dump(summary, open(osp.join(args.out, 'summary.json'), 'w'), indent=1, default=float)
    if not args.quick:
        df.to_csv(osp.join(args.out, 'metrics.csv'), index=False)  # written last: marks the evaluation as done
    print(f'  {len(conds)} conditions in {time.time() - t0:.0f}s; {summary["params_M"]:.2f} M parameters, '
          f'{summary["ms_per_frame"]:.0f} ms/frame', flush=True)


if __name__ == '__main__':
    main()
