"""
Robustness pilot (evaluation only, no training): how do trained models degrade under realistic camera faults
other than a camera going dark?

  drop      the camera sends nothing (reference: what the model already handles)
  rot       calibration drift: the camera is knocked by `angle` degrees about a random axis, but the model still
            uses the original calibration (one camera at a time, and all cameras at once)
  lag       the camera's stream is `lag` annotated frames behind the others (WILDTRACK: 0.5 s per frame)
  freeze    the camera is stuck on the first test frame for the whole sequence

Single-camera faults are applied to each camera in turn and averaged. For every condition the script reports
detection (MODA, recall, precision) and tracking (MOTA, IDF1, ID switches) with the TrackTacular tracker at
threshold 0.5 (S0) and with the ByteTrack-style tracker (S2), and counts how often a fault is worse than simply
dropping that camera.

    python -m evaluation.perturb_eval --results /content/drive/MyDrive/retrack_results/full \
        --data_dir /content/data/Wildtrack
"""
import argparse
import glob
import json
import math
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch

from evaluation.retrack_eval import (S2_DEFAULTS, detection_metrics, load_model, load_test_items, run_inference,
                                     tracking_metrics)

ANGLES = (0.5, 1.0, 2.0)  # degrees
ALL_CAM_ANGLES = (0.5, 1.0)
LAGS = (1, 2)  # annotated frames (0.5 s each on WILDTRACK)


def rotation(axis, angle_deg):
    """4x4 rotation about a unit axis (Rodrigues)."""
    a = torch.as_tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    t = math.radians(angle_deg)
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float32)
    R = torch.eye(4)
    R[:3, :3] = torch.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
    return R


def make_conditions(num_cams, cams, seed=0):
    """List of (kind, magnitude, cameras, perturbation spec)."""
    rng = np.random.default_rng(seed)
    axes = {c: rng.normal(size=3) for c in range(num_cams)}
    conds = [('clean', 0, (), None)]
    for c in cams:
        conds.append(('drop', 0, (c,), None))
        conds += [('rot', a, (c,), {c: rotation(axes[c], a)}) for a in ANGLES]
        conds += [('lag', l, (c,), l) for l in LAGS]
        conds.append(('freeze', 0, (c,), None))
    conds += [('rot_all', a, tuple(range(num_cams)), {c: rotation(axes[c], a) for c in range(num_cams)})
              for a in ALL_CAM_ANGLES]
    return conds


def make_perturb(kind, cams, spec, items):
    if kind in ('clean', 'drop'):
        return None

    def perturb(index, batch):
        dev = batch['img'].device
        batch['extrinsic'] = batch['extrinsic'].clone()  # never modify the cached test items
        for c in cams:
            if kind in ('rot', 'rot_all'):
                batch['extrinsic'][:, c] = spec[c].to(dev) @ batch['extrinsic'][:, c]
            elif kind == 'lag':
                src = items[max(index - spec, 0)]
                batch['img'][:, c] = src['img'][:, c].to(dev).float() / 255
            elif kind == 'freeze':
                batch['img'][:, c] = items[0]['img'][:, c].to(dev).float() / 255
    return perturb


def evaluate_model(ckpt, tag, variant, data_dir, out_dir, cams=None, max_frames=None):
    out = osp.join(out_dir, f'{tag}.csv')
    if osp.exists(out):
        print(f'[{tag}] already done -> {out}')
        return pd.read_csv(out)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(ckpt, device)
    items = load_test_items(data_dir, model)
    if max_frames:
        items = items[:max_frames]
    num_cams = items[0]['img'].shape[1]
    cams = list(range(num_cams)) if cams is None else cams
    rows, t0 = [], time.time()
    for kind, mag, cs, spec in make_conditions(num_cams, cams):
        frames = run_inference(model, items, drop_cams=cs if kind == 'drop' else (),
                               perturb=make_perturb(kind, cs, spec, items))
        row = {'tag': tag, 'variant': variant, 'kind': kind, 'magnitude': mag,
               'cameras': '-'.join(str(c + 1) for c in cs) or 'none'}
        det = detection_metrics(frames, 0.5)
        row.update({k: det[k] for k in ('moda', 'modp', 'recall', 'precision', 'det_fn', 'det_fp')})
        for name, kw in [('S0', dict(tracker='baseline', track_buffer=5)), ('S2', S2_DEFAULTS)]:
            trk = tracking_metrics(frames, 0.5, kw)
            row.update({f'{name}_{k}': trk[k] for k in ('mota', 'idf1', 'id_switches', 'fragmentations')})
        rows.append(row)
        print(f'[{tag}] {kind:7s} {mag:<4} cams={row["cameras"]:>13s}  MODA {row["moda"]:6.2f}  '
              f'MOTA {row["S0_mota"]:6.2f}  IDF1 {row["S0_idf1"]:6.2f}  ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def summarise(df):
    """Mean over cameras per (variant, fault) and the fraction of cameras where the fault is worse than dropping."""
    clean = df[df.kind == 'clean'].set_index('variant')
    drop = df[df.kind == 'drop'].set_index(['variant', 'cameras'])['moda']
    lines = []
    for variant, g in df.groupby('variant', sort=False):
        c = clean.loc[variant]
        lines.append(f'\n### {variant}  (clean: MODA {c.moda:.1f}, MOTA {c.S0_mota:.1f}, IDF1 {c.S0_idf1:.1f})\n')
        lines.append('| fault | MODA | ΔMODA | recall | precision | MOTA (S0) | IDF1 (S0) | MOTA (S2) | IDF1 (S2) '
                     '| worse than dropping the camera |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for (kind, mag), h in g[g.kind != 'clean'].groupby(['kind', 'magnitude'], sort=False):
            label = {'drop': 'camera dropped', 'rot': f'1 camera rotated {mag}°', 'rot_all': f'all cameras rotated {mag}°',
                     'lag': f'1 camera {int(mag)} frame(s) late', 'freeze': '1 camera frozen'}[kind]
            worse = ''
            if kind in ('rot', 'lag', 'freeze'):
                n = sum(r.moda < drop.loc[(variant, r.cameras)] for r in h.itertuples())
                worse = f'{n}/{len(h)}'
            lines.append(f'| {label} | {h.moda.mean():.1f} | {h.moda.mean() - c.moda:+.1f} | {h.recall.mean():.1f} '
                         f'| {h.precision.mean():.1f} | {h.S0_mota.mean():.1f} | {h.S0_idf1.mean():.1f} '
                         f'| {h.S2_mota.mean():.1f} | {h.S2_idf1.mean():.1f} | {worse} |')
    return '\n'.join(lines)


def find_runs(results, seed):
    runs = []
    for p in sorted(glob.glob(osp.join(results, 'runs', '*', 'DONE.json'))):
        info = json.load(open(p))
        if seed is None or int(info['seed']) == seed:
            runs.append((info['ckpt'], f"{info['variant']}_s{info['seed']}", info['variant']))
    return runs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', help='pipeline results folder: evaluates every finished run of --seed')
    p.add_argument('--seed', type=int, default=80085)
    p.add_argument('--ckpt', action='append', default=[], help='evaluate this checkpoint (repeatable)')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', default=None, help='default: <results>/perturb')
    p.add_argument('--variants', nargs='+', default=None, help='only these variants (default: all finished)')
    p.add_argument('--cams', type=int, nargs='+', default=None, help='0-based cameras to perturb (default: all)')
    p.add_argument('--max_frames', type=int, default=None, help='debug: only the first N test frames')
    args = p.parse_args()
    torch.set_float32_matmul_precision('medium')

    runs = [(c, osp.basename(osp.dirname(c)) or 'model', 'model') for c in args.ckpt]
    if args.results:
        runs += find_runs(args.results, args.seed)
    if args.variants:
        runs = [r for r in runs if r[2] in args.variants]
    assert runs, 'no checkpoints found'
    out_dir = args.out or osp.join(args.results or '.', 'perturb')
    # the baseline first, so its numbers are available as early as possible
    runs.sort(key=lambda r: r[2] != 'concat')
    print('evaluating:', [r[1] for r in runs])

    dfs = [evaluate_model(ckpt, tag, variant, args.data_dir, out_dir, args.cams, args.max_frames)
           for ckpt, tag, variant in runs]
    text = '# Robustness pilot: camera faults other than a dropped camera\n' \
           'Threshold 0.5. S0 = TrackTacular tracker, S2 = ByteTrack-style tracker. Single-camera faults are ' \
           'averaged over the cameras; "worse than dropping" counts cameras for which the fault gives lower MODA ' \
           'than switching that camera off.\n' + summarise(pd.concat(dfs, ignore_index=True))
    with open(osp.join(out_dir, 'SUMMARY.md'), 'w') as f:
        f.write(text)
    print('\n' + text + f'\n\nwritten to {out_dir}/SUMMARY.md')
