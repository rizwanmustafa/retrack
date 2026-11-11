"""
Where do the models miss people? Reads saved test detections (arl_eval's detections_test.pt) and splits every
ground-truth person of the 40 test frames by: number of cameras that see them (WILDTRACK annotations), distance to the
edge of the annotated area, distance to the nearest other person, the best detection score within 0.5 m, and whether
the miss is a single frame (detected in both neighbouring frames). Threshold 0.5 and 0.5 m matching, as the protocol.

    python research-methodology-4/code/miss_analysis.py --data_dir /content/data/Wildtrack --out <results>/m4/misses.csv \
        --eval <results>/m4/eval <results>/m2/eval /content/drive/MyDrive/retrack_results_8nov/arl_full/eval
"""
import argparse
import glob
import json
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.normpath(osp.join(HERE, '..', '..', 'WorldTrack')))
from evaluation.retrack_eval import MOD_DIST  # noqa: E402

CELL_M = 0.025
GRID = (1440, 480)  # WILDTRACK grid, x and y in 2.5 cm cells
FRAME_STEP = 5
BINS = {
    'cams': [(1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 4, '4'), (5, 99, '5+')],
    'border_m': [(0, 1, '<1 m'), (1, 1e9, '>=1 m')],
    'nn_m': [(0, 0.75, '<0.75 m'), (0.75, 1.5, '0.75-1.5 m'), (1.5, 1e9, '>=1.5 m')],
}


def camera_counts(data_dir, raw_frame):
    with open(osp.join(data_dir, 'annotations_positions', f'{raw_frame:08d}.json')) as f:
        peds = json.load(f)
    return {p['personID']: sum(v['xmin'] != -1 or v['xmax'] != -1 for v in p['views']) for p in peds}


def people(frames, data_dir, thr=0.5):
    out, missed = [], defaultdict(set)
    for f in frames:
        gt, det = f['gt'], f['xy'][f['score'] > thr]
        hit = np.zeros(len(gt), bool)
        if len(gt) and len(det):
            d = np.linalg.norm(gt[:, None, :2] - det[None], axis=2)
            c = d.copy()
            c[c > MOD_DIST] = 1e6
            r, k = linear_sum_assignment(c)
            hit[r[d[r, k] < MOD_DIST]] = True
        cams = camera_counts(data_dir, int(f['frame']) * FRAME_STEP)
        for i, g in enumerate(gt):
            others = np.delete(gt[:, :2], i, axis=0)
            near = np.linalg.norm(f['xy'] - g[:2], axis=1) < MOD_DIST
            out.append(dict(frame=int(f['frame']), pid=int(g[2]), hit=bool(hit[i]), cams=cams.get(int(g[2]), 0),
                            border_m=min(g[0], GRID[0] - g[0], g[1], GRID[1] - g[1]) * CELL_M,
                            nn_m=np.linalg.norm(others - g[:2], axis=1).min() * CELL_M if len(others) else 1e9,
                            best=float(f['score'][near].max(initial=0))))
            if not hit[i]:
                missed[int(g[2])].add(int(f['frame']))
    for p in out:
        p['single'] = not p['hit'] and not ({p['frame'] - 1, p['frame'] + 1} & missed[p['pid']])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--eval', nargs='+', required=True, help='folders holding <tag>/detections_test.pt')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = []
    for folder in args.eval:
        for path in sorted(glob.glob(osp.join(folder, '*_test_s*', 'detections_test.pt'))):
            tag = osp.basename(osp.dirname(path))
            variant, seed = tag.split('_test_s')
            ppl = people(torch.load(path, weights_only=False), args.data_dir)
            miss = [p for p in ppl if not p['hit']]
            row = dict(source=folder, variant=variant, seed=int(seed), gt=len(ppl), misses=len(miss),
                       single_frame=sum(p['single'] for p in miss),
                       score_lt_0_1=sum(p['best'] < 0.1 for p in miss),
                       score_0_1_to_0_3=sum(0.1 <= p['best'] < 0.3 for p in miss),
                       score_0_3_to_0_5=sum(0.3 <= p['best'] < 0.5 for p in miss))
            for key, bins in BINS.items():
                for lo, hi, name in bins:
                    sub = [p for p in ppl if lo <= p[key] <= hi] if key == 'cams' else \
                        [p for p in ppl if lo <= p[key] < hi]
                    row[f'{key} {name} missed'] = sum(not p['hit'] for p in sub)
                    row[f'{key} {name} total'] = len(sub)
            rows.append(row)
            print(f"{variant} s{seed}: {row['misses']} misses; by cameras " + ', '.join(
                f"{n}: {row[f'cams {n} missed']}/{row[f'cams {n} total']}" for _, _, n in BINS['cams']) +
                f"; near-threshold (0.3-0.5) {row['score_0_3_to_0_5']}; single-frame {row['single_frame']}",
                flush=True)
    if rows:
        keys = list(rows[0])
        with open(args.out, 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r[k]) for k in keys) + '\n')
        print(f'-> {args.out}')


if __name__ == '__main__':
    main()
