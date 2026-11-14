"""
Why are people missed when cameras are removed? (research-methodology-5/OPEN_FAILURES.md, item 2b)

Evaluation only, on an existing checkpoint, for every camera condition (all cameras, then every combination of 1 to
--max_drop removed cameras):

  sweep.csv    detection FP / FN / MODA at thresholds 0.10 .. 0.70. If the best threshold falls as cameras are
               removed, the extra misses come from scores that dropped (under-confidence), not from missing evidence.
  people.csv   one row per ground-truth person per frame per condition:
                 hit at 0.5, best detection score within 0.5 m (no threshold), cameras that see the person
                 (annotated boxes) in total and among the working cameras, distance to the nearest other person.
  summary.txt  the tables printed below, for each k (number of removed cameras):
                 misses by best score (<0.1 none, 0.1-0.3 weak, 0.3-0.5 just below threshold)
                 misses by number of working cameras that see the person (0, 1, 2, 3, 4, 5+)
                 best threshold, and MODA at 0.5 against MODA at the best threshold
                 median best score of the people found (does the score of everyone drop?)

Matching as the protocol: 0.5 m, Hungarian, threshold 0.5. The `best` score ignores the matching, so it tells whether
the network put any response near the person.

usage (repo root):
    python research-methodology-5/code/m5_diagnose.py --ckpt <last.ckpt> --data_dir /content/data/Wildtrack \
        --out <results>/m5/diagnose/wildtrack_B0_s80085
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.normpath(osp.join(HERE, '..', '..', 'WorldTrack')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from evaluation.retrack_eval import (MOD_DIST, load_model, load_test_items, run_inference,  # noqa: E402
                                     detection_metrics, camera_conditions)

THR = 0.5
THRS = np.round(np.arange(0.10, 0.701, 0.05), 2)
CELL_M = 0.025  # reference grid cell (both datasets)
CAM_BINS = [(0, 0, '0'), (1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 4, '4'), (5, 99, '5+')]
SCORE_BINS = [(0.0, 0.1, '<0.1'), (0.1, 0.3, '0.1-0.3'), (0.3, 0.5, '0.3-0.5')]


def visibility(data_dir, frame_step, frames):
    """{(frame index, personID): set of cameras whose annotation has a box for the person}"""
    folder = osp.join(data_dir, 'annotations_positions')
    files = {int(f.split('.')[0]): osp.join(folder, f) for f in os.listdir(folder) if f.endswith('.json')}
    vis = {}
    for fr in frames:
        with open(files[fr * frame_step]) as f:
            for p in json.load(f):
                vis[(fr, int(p['personID']))] = {i for i, v in enumerate(p['views'])
                                                 if not (v['xmin'] == -1 and v['xmax'] == -1 and
                                                         v['ymin'] == -1 and v['ymax'] == -1)}
    return vis


def people_rows(frames, vis, drop):
    rows = []
    dropped = set(drop)
    for f in frames:
        gt = f['gt']
        det = f['xy'][f['score'] > THR].astype(np.float64)
        hit = np.zeros(len(gt), bool)
        if len(gt) and len(det):
            d = np.linalg.norm(gt[:, None, :2] - det[None], axis=2)
            c = d.copy()
            c[c > MOD_DIST] = 1e6
            r, k = linear_sum_assignment(c)
            hit[r[d[r, k] < MOD_DIST]] = True
        for i, g in enumerate(gt):
            near = np.linalg.norm(f['xy'] - g[:2], axis=1) < MOD_DIST
            cams = vis.get((f['frame'], int(g[2])), set())
            others = np.delete(gt[:, :2], i, axis=0)
            rows.append({'frame': f['frame'], 'pid': int(g[2]), 'hit': bool(hit[i]),
                         'best': float(f['score'][near].max(initial=0.0)),
                         'cams_total': len(cams), 'cams_working': len(cams - dropped),
                         'nn_m': float(np.linalg.norm(others - g[:2], axis=1).min() * CELL_M) if len(others) else 99.0})
    return rows


def summarise(sweep, people, out_txt):
    lines = []
    p = lambda s='': (print(s, flush=True), lines.append(s))  # noqa: E731
    for k, pk in people.groupby('k'):
        n_cond = pk['drop'].nunique()
        miss = pk[~pk.hit]
        sk = sweep[sweep.k == k].groupby('thr')[['det_fp', 'det_fn', 'moda']].mean()
        best_thr = sk['moda'].idxmax()
        p(f'k = {k} cameras removed ({n_cond} conditions, means per condition)')
        p(f'  FP {sk.loc[THR, "det_fp"]:.1f}  FN {sk.loc[THR, "det_fn"]:.1f}  MODA {sk.loc[THR, "moda"]:.2f} at 0.5;'
          f'  best threshold {best_thr:.2f}: MODA {sk.loc[best_thr, "moda"]:.2f}'
          f' (FP {sk.loc[best_thr, "det_fp"]:.1f}, FN {sk.loc[best_thr, "det_fn"]:.1f})')
        p(f'  median best score of people found: {pk[pk.hit].best.median():.3f}')
        p('  misses by best score within 0.5 m: ' + ', '.join(
            f'{name} {((miss.best >= lo) & (miss.best < hi)).sum() / n_cond:.1f}' for lo, hi, name in SCORE_BINS))
        p('  missed / total by working cameras that see the person: ' + ', '.join(
            f'{name}: {(~pk.hit & pk.cams_working.between(lo, hi)).sum() / n_cond:.1f}'
            f'/{pk.cams_working.between(lo, hi).sum() / n_cond:.1f}' for lo, hi, name in CAM_BINS))
        p('  miss rate by working cameras: ' + ', '.join(
            f'{name}: {100 * (~pk.hit & pk.cams_working.between(lo, hi)).sum() / max(pk.cams_working.between(lo, hi).sum(), 1):.1f}%'
            for lo, hi, name in CAM_BINS))
        p()
    open(out_txt, 'w').write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_drop', type=int, default=3)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.ckpt, device)
    items = load_test_items(args.data_dir, model, num_workers=args.num_workers)
    S = items[0]['img'].shape[1]
    frame_step = 5 if 'wildtrack' in osp.basename(osp.normpath(args.data_dir)).lower() else 1
    vis = visibility(args.data_dir, frame_step, [int(it['frame'][0]) for it in items])
    conds = camera_conditions(S, args.max_drop)
    print(f'{args.ckpt}\n{len(items)} frames, {S} cameras, {len(conds)} conditions', flush=True)

    sweep, people = [], []
    t0 = time.time()
    for drop in conds:
        frames = run_inference(model, items, drop)
        tag = '-'.join(str(c) for c in drop) or 'none'
        for thr in THRS:
            sweep.append({'drop': tag, 'k': len(drop), 'thr': float(thr), **detection_metrics(frames, thr)})
        for r in people_rows(frames, vis, drop):
            people.append({'drop': tag, 'k': len(drop), **r})
    print(f'{len(conds)} conditions in {time.time() - t0:.0f}s\n', flush=True)

    sweep, people = pd.DataFrame(sweep), pd.DataFrame(people)
    sweep.to_csv(osp.join(args.out, 'sweep.csv'), index=False)
    people.to_csv(osp.join(args.out, 'people.csv'), index=False)
    summarise(sweep, people, osp.join(args.out, 'summary.txt'))


if __name__ == '__main__':
    main()
