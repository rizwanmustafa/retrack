"""
Robustness and mirror test-time augmentation for research methodology 3 (research-methodology-2/METHODOLOGY.md §9).

Every condition is scored with the paper protocol (threshold 0.5, TrackTacular's tracker; cell thr0.5|tt of arl_eval):
  clean        the plain model, as arl_eval
  tta          center heatmap averaged over the four mirrors of the reference frame (identity, x, y, xy); offsets
               and tracking offsets from the identity pass
  calib_low    Gaussian extrinsic noise per camera, rotation 0.5 deg and translation 0.05 m per axis (std),
  calib_high   rotation 2.0 deg and translation 0.2 m: the magnitudes of ATLASFusion Tab. 5. 3 draws each, the same
               draws for every checkpoint, so checkpoints are compared on identical perturbations
  drop1        one camera switched off, each of the 7
  drop2        two cameras switched off, the 7 cyclic pairs (0,1) ... (6,0)

    python -m evaluation.robust_eval --spec spec.json --data_dir /content/data/Wildtrack --out_dir robust/
    spec.json: [{"tag": "A2_s1", "variant": "A2", "seed": 1, "ckpt": ".../last.ckpt"}, ...]
Writes <out_dir>/<tag>.csv per checkpoint (skipped when it exists, so a re-run continues).
"""
import argparse
import json
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch

from evaluation.retrack_eval import load_model, load_test_items, reset_temporal_cache, detection_metrics, \
    tracking_metrics
from utils import decode

TT = dict(tracker='baseline', track_buffer=5)  # TrackTacular's tracker, as arl_eval's tt cells
MIRRORS = ((-1, 1), (1, -1), (-1, -1))
CALIB = {'calib_low': (0.5, 0.05), 'calib_high': (2.0, 0.2)}  # (rotation deg, translation m), ATLASFusion Tab. 5
WORLD_UNIT_M = 0.01  # WILDTRACK world coordinates are in cm (datasets/wildtrack_dataset.py: 2.5 cm grid cells)


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def calib_noise(num_cams, rot_deg, trans_m, seed):
    """S,4,4 perturbations applied in each camera frame: cams_T_global' = N cams_T_global."""
    rng = np.random.default_rng(seed)
    N = np.tile(np.eye(4), (num_cams, 1, 1))
    for v in range(num_cams):
        N[v, :3, :3] = rodrigues(rng.normal(0, np.deg2rad(rot_deg), 3))
        N[v, :3, 3] = rng.normal(0, trans_m / WORLD_UNIT_M, 3)
    return torch.tensor(N, dtype=torch.float32)


def mirror(sx, sy, bounds, device):
    """Reference-frame mirror about the grid centre (as PedestrianDataset.sample_scene_transform)."""
    xmin, xmax, ymin, ymax = bounds[:4]
    T = torch.eye(4, device=device)
    T[0, 0], T[1, 1] = sx, sy
    T[0, 3] = (xmin + xmax) / 2 * (1 - sx)
    T[1, 3] = (ymin + ymax) / 2 * (1 - sy)
    return T


@torch.no_grad()
def infer(model, items, noise=None, drop_cams=(), tta=False):
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
        if noise is not None:
            batch['extrinsic'] = torch.matmul(noise.to(device).unsqueeze(0), batch['extrinsic'])
        out = model(batch)
        center = out['instance_center'].sigmoid()  # B,1,Y,X
        if tta:
            # voxel i is centred at xmin + (i + 1/2) * size, so a mirror about the grid centre maps cell i to
            # cell X-1-i and torch.flip undoes it exactly
            acc = center.clone()
            for sx, sy in MIRRORS:
                mirrored = dict(batch, ref_T_global=torch.matmul(mirror(sx, sy, model.bounds, device),
                                                                 batch['ref_T_global']))
                c = model(mirrored)['instance_center'].sigmoid()
                acc += torch.flip(c, [d for d, s in ((-1, sx), (-2, sy)) if s < 0])
            center = acc / (1 + len(MIRRORS))
        xy_e, xy_prev_e, scores_e, _, _, _ = decode.decoder(
            center, out['instance_offset'], out['instance_size'], rz_e=out['instance_rot'], K=model.max_detections)
        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = model.vox_util.Mem2Ref(mem_xyz, model.Y, model.Z, model.X)[..., :2]
        ref_xy_prev = model.vox_util.Mem2Ref(mem_xyz_prev, model.Y, model.Z, model.X)[..., :2]
        grid_gt = item['grid_gt'][0]
        frames.append({
            'frame': int(item['frame'][0]),
            'gt': grid_gt[grid_gt.sum(1) != 0].numpy().astype(np.float64),
            'xy': ref_xy[0].cpu().numpy(),
            'xy_prev': ref_xy_prev[0].cpu().numpy(),
            'score': scores_e[0].cpu().numpy(),
        })
    return frames


def score(frames):
    row = detection_metrics(frames, 0.5)
    row.update(tracking_metrics(frames, 0.5, TT))
    return {k: row[k] for k in ('moda', 'modp', 'precision', 'recall', 'det_fn', 'det_fp', 'mota', 'idf1',
                                'id_switches', 'trk_misses', 'trk_fp')}


def conditions(num_cams, quick=False):
    """[(condition, draw, kwargs)]; quick = one of each, for the preflight."""
    out = [('clean', 0, {}), ('tta', 0, dict(tta=True))]
    draws = 1 if quick else 3
    for name, (r, t) in CALIB.items():
        out += [(name, k, dict(noise=calib_noise(num_cams, r, t, seed=1000 * (r > 1) + k))) for k in range(draws)]
    singles = [(v,) for v in range(num_cams)]
    pairs = [(v, (v + 1) % num_cams) for v in range(num_cams)]
    out += [('drop1', i, dict(drop_cams=c)) for i, c in enumerate(singles[:1] if quick else singles)]
    out += [('drop2', i, dict(drop_cams=c)) for i, c in enumerate(pairs[:1] if quick else pairs)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--quick', action='store_true', help='3 test frames and one draw per condition (preflight)')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    items, items_key = None, None
    for job in json.load(open(args.spec)):
        out = osp.join(args.out_dir, f"{job['tag']}.csv")
        if osp.exists(out):
            continue
        t0 = time.time()
        model, _ = load_model(job['ckpt'], device)
        key = (tuple(model.resolution), tuple(model.bounds))
        if items is None or key != items_key:
            items, items_key = load_test_items(args.data_dir, model, num_workers=args.num_workers), key
            if args.quick:
                items = items[:3]
        rows = []
        for cond, draw, kw in conditions(items[0]['img'].shape[1], args.quick):
            rows.append(dict(tag=job['tag'], variant=job['variant'], seed=job['seed'], condition=cond, draw=draw,
                             **score(infer(model, items, **kw))))
        pd.DataFrame(rows).to_csv(out + '.part', index=False)
        os.replace(out + '.part', out)
        clean = rows[0]
        print(f"[{job['tag']}] {len(rows)} conditions in {time.time() - t0:.0f}s; clean MODA {clean['moda']:.2f} "
              f"MOTA {clean['mota']:.2f} IDF1 {clean['idf1']:.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
