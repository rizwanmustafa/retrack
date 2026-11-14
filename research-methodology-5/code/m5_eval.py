"""
Research methodology 5 evaluation of one checkpoint on the 40 test frames of WILDTRACK or MultiviewX.

For every camera condition (all cameras, then every combination of 1..max_drop removed cameras):
  detection  MODA, MODP, precision, recall at threshold 0.5 (0.5 m), as evaluation/retrack_eval.py
  tracking   the variant's own tracker: R4 variants use tracking/query_tracker.py online (the network gets the live
             tracks every frame); the others use TrackTacular's tracker (buffer 5) on the same detections.
             `base_*` columns: TrackTacular's tracker on the detections of every variant, so R4's gain can be
             separated from detection changes.
Also: trainable / frozen parameter counts and the forward time per frame (all cameras, batch 1).

usage (any directory):
    python research-methodology-5/code/m5_eval.py --ckpt last.ckpt --data_dir /content/data/Wildtrack \
        --out <results>/m5/eval/wildtrack_R124_s80085 --dataset wildtrack --variant R124 --seed 80085
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
WORLDTRACK = osp.normpath(osp.join(HERE, '..', '..', 'WorldTrack'))
sys.path.insert(0, WORLDTRACK)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from evaluation.retrack_eval import (load_model, load_test_items, run_inference, reset_temporal_cache,  # noqa: E402
                                     detection_metrics, tracking_metrics, hyps_metrics, camera_conditions)
from tracking.query_tracker import QueryTracker  # noqa: E402
from utils import decode  # noqa: E402

THR = 0.5
BASE_TRACKER = dict(tracker='baseline', track_buffer=5)


def to_mem(model, xy_ref, device):
    xyz = torch.zeros(1, len(xy_ref), 3, device=device)
    xyz[0, :, :2] = torch.as_tensor(xy_ref, dtype=torch.float32, device=device)
    return model.vox_util.Ref2Mem(xyz, model.Y, model.Z, model.X)[..., :2]


def to_ref(model, xy_mem):
    xyz = torch.cat([xy_mem, torch.zeros_like(xy_mem[..., :1])], dim=-1)
    return model.vox_util.Mem2Ref(xyz, model.Y, model.Z, model.X)[..., :2]


@torch.no_grad()
def run_online(model, items, drop_cams=()):
    """R4: frame by frame with the tracker in the loop. Returns (frames as run_inference, hypotheses per frame)."""
    device = model.device
    reset_temporal_cache(model)
    tracker = QueryTracker(conf_thres=THR, track_buffer=5)
    frames, hyps = [], []
    for item in items:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
        batch['img'] = batch['img'].float() / 255
        S = batch['img'].shape[1]
        if drop_cams:
            cam_mask = torch.ones(1, S, device=device)
            cam_mask[:, list(drop_cams)] = 0
            batch['cam_mask'] = cam_mask
        pos = tracker.query_positions()
        if len(pos):
            track_xy = to_mem(model, pos, device)
            output = model(batch, track_xy, torch.ones(track_xy.shape[:2], dtype=torch.bool, device=device))
            last = output['track_layers'][-1]
            pred = to_ref(model, last['xy'])[0].cpu().numpy()
            presence = last['logit'][0].sigmoid().cpu().numpy()
        else:
            output = model(batch)
            pred, presence = np.zeros((0, 2)), np.zeros(0)
        xy_e, xy_prev_e, scores_e, _, _, _ = decode.decoder(
            output['instance_center'].sigmoid(), output['instance_offset'], output['instance_size'],
            rz_e=output['instance_rot'], K=model.max_detections)
        ref_xy = to_ref(model, xy_e)[0].cpu().numpy()
        ref_prev = to_ref(model, xy_prev_e)[0].cpu().numpy()
        scores = scores_e[0].cpu().numpy()
        grid_gt = item['grid_gt'][0]
        frames.append({'frame': int(item['frame'][0]), 'seq': int(item['sequence_num'][0]),
                       'gt': grid_gt[grid_gt.sum(1) != 0].numpy().astype(np.float64),
                       'xy': ref_xy, 'xy_prev': ref_prev, 'score': scores, 'feat': None})
        hyps.append(tracker.update(ref_xy, scores, pred, presence))
    return frames, hyps


@torch.no_grad()
def time_forward(model, items, n=20):
    device = model.device
    reset_temporal_cache(model)
    ts = []
    for item in items[:n]:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
        batch['img'] = batch['img'].float() / 255
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        model(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return float(np.median(ts[2:] if len(ts) > 4 else ts)) * 1000


def param_counts(model):
    trainable = sum(p.numel() for p in model.parameters())
    frozen = 0
    bb = getattr(model.model, 'da3_backbone', None)
    if bb is not None and bb.net is not None:
        frozen = sum(p.numel() for p in bb.net.parameters())
    return trainable, frozen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--max_drop', type=int, default=3)
    p.add_argument('--quick', action='store_true', help='3 frames, all cameras only (preflight)')
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, hp = load_model(args.ckpt, device)
    use_q = bool(hp.get('track_queries', False))
    items = load_test_items(args.data_dir, model, num_workers=args.num_workers)
    S = items[0]['img'].shape[1]
    conds = [()] if args.quick else camera_conditions(S, args.max_drop)
    if args.quick:
        items = items[:3]
    print(f'{args.dataset} {args.variant} s{args.seed}: {len(items)} frames, {S} cameras, {len(conds)} conditions, '
          f'tracker {"query (R4)" if use_q else "TrackTacular"}', flush=True)

    rows, keep = [], {}
    t0 = time.time()
    for drop in conds:
        if use_q:
            frames, hyps = run_online(model, items, drop)
            trk = hyps_metrics(frames, hyps)
        else:
            frames = run_inference(model, items, drop)
            trk = None
        base = tracking_metrics(frames, THR, BASE_TRACKER)
        trk = trk or base
        row = {'dataset': args.dataset, 'variant': args.variant, 'seed': args.seed,
               'drop': '-'.join(str(c) for c in drop) or 'none', 'k': len(drop), 'num_cams': S}
        row.update(detection_metrics(frames, THR))
        row.update(trk)
        row.update({f'base_{k}': v for k, v in base.items() if k in ('mota', 'idf1', 'id_switches')})
        rows.append(row)
        if not drop:
            keep = {'frames': frames, 'hyps': hyps if use_q else None}
            print(f'  all cameras: MODA {row["moda"]:.2f}  MOTA {row["mota"]:.2f}  IDF1 {row["idf1"]:.2f}  '
                  f'(TrackTacular tracker: MOTA {row["base_mota"]:.2f}  IDF1 {row["base_idf1"]:.2f})', flush=True)
    df = pd.DataFrame(rows)
    trainable, frozen = param_counts(model)
    summary = {
        'dataset': args.dataset, 'variant': args.variant, 'seed': args.seed, 'ckpt': args.ckpt,
        'trainable_params': trainable, 'frozen_params': frozen,
        'ms_per_frame': time_forward(model, items), 'eval_seconds': time.time() - t0,
        'hparams': {k: hp.get(k) for k in ('da3', 'da3_repo', 'sparse_queries', 'num_queries', 'query_layers',
                                           'track_queries', 'fusion', 'num_cameras', 'resolution')},
        'all_cameras': rows[0],
        'mean_by_k': df.groupby('k')[['moda', 'mota', 'idf1']].mean().round(3).to_dict(orient='index'),
    }
    torch.save(keep, osp.join(args.out, 'all_cameras_frames.pt'))
    json.dump(summary, open(osp.join(args.out, 'summary.json'), 'w'), indent=1, default=float)
    if not args.quick:
        df.to_csv(osp.join(args.out, 'metrics.csv'), index=False)  # written last: marks the evaluation as done
    print(f'  {len(conds)} conditions in {time.time() - t0:.0f}s; trainable {trainable / 1e6:.2f} M, '
          f'frozen {frozen / 1e6:.2f} M, {summary["ms_per_frame"]:.0f} ms/frame', flush=True)


if __name__ == '__main__':
    main()
