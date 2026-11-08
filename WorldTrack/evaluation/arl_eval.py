"""
ARL evaluation engine (research_methodology_attempt_1/METHODOLOGY.md, §4.2-§5).

For one trained checkpoint:
  1. run the network once on a split (all cameras) and cache the decoded BEV detections {xy, xy_prev, score} and,
     optionally, the multi-view OSNet descriptors (Retrack Stage 3);
  2. tune the parameters of every post-hoc component on a tuning split, one component at a time, each in the
     context of the components before it:
        B1   detection threshold (best MODA)
        C1   ByteTrack-style online BEV tracker (low threshold, track buffer)
        R    appearance in C1 and C2 (OSNet weight / threshold)
        C2   offline tracklet linking (window, no-link cost, gap penalty)
        C2b  detection-level min-cost flow (alternative to C1 + C2)
        C3   Gaussian-smoothed interpolation (gap, length scale, noise)
     tracking objective: (MOTA + IDF1) / 2;
  3. score every on/off combination of the post-hoc switches with those fixed parameters, so that each addition can
     be judged by switching only it (evaluation/arl_report.py does that).

Roles:
  --role val    validation model (trained on frames 0-323): tune on frames 324-359, write params.json  (clean)
  --role test   final model: cache the test frames, tune on them (oracle, labelled) and, with --clean_params,
                also score them with the parameters chosen on the validation model (clean)

usage (from WorldTrack/):
    python -m evaluation.arl_eval --role test --ckpt last.ckpt --data_dir /content/data/Wildtrack --out eval/x \
        --clean_params eval/x_val/params.json
"""
import argparse
import itertools
import json
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch

from evaluation.retrack_eval import (load_model, load_test_items, run_inference, detection_metrics,
                                     tracking_metrics, hyps_metrics, pick_threshold)
from tracking import offline

SPLITS = {'test': (0.9, 1.0), 'val': (0.81, 0.9)}
# B1 sweep, wider than Retrack's 0.20-0.75: in the quick run the best threshold of a strongly augmented model was
# the lowest one tried
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.76, 0.05)]

# ------------------------------------------------------------------ parameter grids (METHODOLOGY.md §6)
BYTE_DEFAULT = dict(tracker='byte', low_thres=0.1, track_buffer=10, low_match_thresh=50.)
BYTE_GRID = [dict(BYTE_DEFAULT, low_thres=lo, track_buffer=buf) for lo in (0.1, 0.2, 0.3) for buf in (5, 10)]
REID_DEFAULT = dict(use_reid=True, reid_lambda=0.98, reid_thresh=0.6)
REID_GRID = [dict(REID_DEFAULT, reid_lambda=lam, reid_thresh=th) for lam in (0.9, 0.98) for th in (0.4, 0.6, 0.8)]
# C2 must reach further than C1 to add anything: the online trackers already re-attach a lost track within their
# track buffer (5-10 frames) and match gate (75 cells = 1.875 m, tracking/multitracker.py). The quick run showed that
# a 1 m gate and gaps of <= 10 frames found no link at all, so the grid covers longer gaps and wider gates.
LINK_DEFAULT = dict(max_gap=10, gate=2.0, gate_growth=0.5, w_m=1.0, w_t=0.1, w_a=0.0, c0=1.0)
LINK_GRID = [dict(LINK_DEFAULT, max_gap=g, gate=gt, c0=c0, w_t=wt) for g in (10, 20, 40) for gt in (2.0, 3.0)
             for c0 in (0.5, 1.0, 2.0) for wt in (0.0, 0.1)]
LINK_APP = (0.25, 0.5, 1.0)
# same gate as the online trackers (1.875 m); the quick run showed a 1 m gate fragments tracks
FLOW_DEFAULT = dict(low_thr=0.1, max_skip=2, gate=1.875, gate_growth=0.5, c_b=1.0, w_t=0.2)
FLOW_GRID = [dict(FLOW_DEFAULT, low_thr=lo, max_skip=k, gate=gt, c_b=cb) for lo in (0.1, 0.3) for k in (1, 3)
             for gt in (1.875, 2.5) for cb in (0.5, 1.0, 2.0)]
GSI_DEFAULT = dict(max_gap=2, tau=10.0, noise=0.01)
GSI_GRID = [dict(max_gap=g, tau=0.0, noise=0.01) for g in (1, 2, 3)] + \
           [dict(max_gap=g, tau=tau, noise=nz) for g in (1, 2, 3) for tau in (3.0, 10.0) for nz in (0.01, 0.04)]

# association modes of the factorial. tt = TrackTacular's own online tracker (Retrack S0/S1)
ASSOC = ['tt', 'tt+link', 'byte', 'byte+link', 'byte_r', 'byte_r+link', 'flow']


def objective(row):
    return (row['mota'] + row['idf1']) / 2


def switches(cell):
    a = cell['assoc']
    return {'B1': int(cell['B1']), 'C1': int(a.startswith('byte')), 'R': int(a.startswith('byte_r')),
            'C2': int('link' in a), 'C2b': int(a == 'flow'), 'C3': int(cell['C3'])}


def cell_name(cell):
    return f"{'thr*' if cell['B1'] else 'thr0.5'}|{cell['assoc']}{'|gsi' if cell['C3'] else ''}"


def all_cells(with_reid):
    for b1, assoc, c3 in itertools.product((0, 1), ASSOC, (0, 1)):
        if assoc.startswith('byte_r') and not with_reid:
            continue
        yield {'B1': b1, 'assoc': assoc, 'C3': c3}


# ------------------------------------------------------------------ one configuration
def run_cell(frames, cell, P):
    thr = P['thr'] if cell['B1'] else 0.5
    assoc = cell['assoc']
    link = 'link' in assoc
    reid = assoc.startswith('byte_r')
    row = detection_metrics(frames, thr)
    if assoc == 'flow':
        tracks = offline.flow_tracks(frames, thr=thr, **P['flow'])
    else:
        if assoc.startswith('tt'):
            kw = dict(tracker='baseline', track_buffer=5)
        else:
            kw = dict(P['byte'], **(P['reid'] if reid else {}))
        if not link and not cell['C3']:  # plain online tracking: exactly the Retrack / TrackTacular evaluation
            row.update(tracking_metrics(frames, thr, kw))
            return row
        tracks = offline.online_tracklets(frames, thr, kw, use_feats=reid)
        if link:
            lp = dict(P['link_r'] if reid else P['link'])
            c0 = lp.pop('c0')
            tracks = offline.link_tracklets(tracks, c0=c0, **lp)
    if cell['C3']:
        tracks = [offline.gsi(tr, **P['gsi']) for tr in tracks]
    row.update(hyps_metrics(frames, offline.tracks_to_hyps(tracks, len(frames))))
    return row


# ------------------------------------------------------------------ tuning
def tune(frames, with_reid, log=print):
    """Greedy, one component at a time, each with the components before it switched on."""
    P = {'byte': dict(BYTE_DEFAULT), 'reid': dict(REID_DEFAULT), 'link': dict(LINK_DEFAULT),
         'link_r': dict(LINK_DEFAULT, w_a=0.5), 'flow': dict(FLOW_DEFAULT), 'gsi': dict(GSI_DEFAULT)}
    sweep = [dict(thr=t, **detection_metrics(frames, t)) for t in THRESHOLDS]
    P['thr'] = pick_threshold(sweep)
    P['with_reid'] = bool(with_reid)
    log(f'  B1 threshold {P["thr"]} (MODA {max(r["moda"] for r in sweep):.2f})')

    def best(key, grid, cell):
        scored = []
        for g in grid:
            P[key] = g
            scored.append((objective(run_cell(frames, cell, P)), g))
        score, g = max(scored, key=lambda s: s[0])  # ties -> first (= closest to the a-priori default)
        P[key] = g
        log(f'  {key:6s} {g}  ((MOTA+IDF1)/2 = {score:.2f})')

    best('byte', BYTE_GRID, {'B1': 1, 'assoc': 'byte', 'C3': 0})
    if with_reid:
        best('reid', REID_GRID, {'B1': 1, 'assoc': 'byte_r', 'C3': 0})
    best('link', LINK_GRID, {'B1': 1, 'assoc': 'byte+link', 'C3': 0})
    if with_reid:
        best('link_r', [dict(P['link'], w_a=a) for a in LINK_APP], {'B1': 1, 'assoc': 'byte_r+link', 'C3': 0})
    best('flow', FLOW_GRID, {'B1': 1, 'assoc': 'flow', 'C3': 0})
    best('gsi', GSI_GRID, {'B1': 1, 'assoc': 'byte+link', 'C3': 1})
    return P, sweep


def score_all(frames, P, with_reid, base):
    rows = []
    for cell in all_cells(with_reid):
        r = run_cell(frames, cell, P)
        rows.append(dict(base, cell=cell_name(cell), assoc=cell['assoc'], thr=P['thr'] if cell['B1'] else 0.5,
                         **switches(cell), **r))
    return rows


# ------------------------------------------------------------------ driver
def cached_frames(ckpt, data_dir, split, out_dir, use_reid, num_workers=4):
    path = osp.join(out_dir, f'detections_{split}.pt')
    if osp.exists(path):
        return torch.load(path, weights_only=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, hparams = load_model(ckpt, device)
    print(f'  model={hparams.get("model_name")} temporal_cache={hparams.get("use_temporal_cache")} '
          f'fusion={hparams.get("fusion", "concat")} cam_dropout={hparams.get("cam_dropout_prob", 0.0)}')
    items = load_test_items(data_dir, model, num_workers=num_workers, frame_range=SPLITS[split])
    reid = None
    if use_reid:
        from tracking.reid import BEVReIDExtractor
        reid = BEVReIDExtractor(person_height=0.9 * (model.bounds[5] - model.bounds[4]), device=device,
                                z_sign=hparams.get('z_sign', 1))
    frames = run_inference(model, items, (), reid)
    torch.save(frames, path + '.part')
    os.replace(path + '.part', path)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return frames


def evaluate(role, ckpt, data_dir, out_dir, tag, variant, seed, clean_params=None, use_reid=True):
    os.makedirs(out_dir, exist_ok=True)
    done = osp.join(out_dir, 'params.json' if role == 'val' else 'metrics.csv')
    json.dump({'tag': tag, 'variant': variant, 'seed': seed, 'role': role, 'ckpt': ckpt,
               'clean_params': clean_params, 'metrics': done}, open(osp.join(out_dir, 'info.json'), 'w'), indent=1)
    if osp.exists(done):
        print(f'[{tag}] already evaluated -> {done}')
        return
    t0 = time.time()
    split = 'val' if role == 'val' else 'test'
    print(f'[{tag}] {role}: {ckpt} on the {split} frames')
    frames = cached_frames(ckpt, data_dir, split, out_dir, use_reid)
    with_reid = use_reid and frames[0]['feat'] is not None
    base = {'tag': tag, 'variant': variant, 'seed': seed}

    print(f'[{tag}] tuning on the {split} frames')
    P, sweep = tune(frames, with_reid)
    pd.DataFrame([dict(base, **r) for r in sweep]).to_csv(osp.join(out_dir, f'threshold_sweep_{split}.csv'),
                                                          index=False)
    if role == 'val':
        rows = score_all(frames, P, with_reid, dict(base, protocol='val_tuning'))
        pd.DataFrame(rows).to_csv(osp.join(out_dir, 'metrics_val.csv'), index=False)
        json.dump(P, open(done + '.part', 'w'), indent=1)
        os.replace(done + '.part', done)
        print(f'[{tag}] clean parameters -> {done} ({time.time() - t0:.0f}s)')
        return

    json.dump(P, open(osp.join(out_dir, 'params_oracle.json'), 'w'), indent=1)
    rows = score_all(frames, P, with_reid, dict(base, protocol='oracle'))
    if clean_params:
        Pc = json.load(open(clean_params))
        rows += score_all(frames, Pc, with_reid and Pc.get('with_reid', False), dict(base, protocol='clean'))
    df = pd.DataFrame(rows)
    df.to_csv(done + '.part', index=False)
    os.replace(done + '.part', done)
    for prot in df.protocol.unique():
        d = df[df.protocol == prot].set_index('cell')
        for c in ('thr0.5|tt', 'thr*|byte+link|gsi'):
            if c in d.index:
                r = d.loc[c]
                print(f'[{tag}] {prot:6s} {c:20s} MODA {r.moda:6.2f}  MOTA {r.mota:6.2f}  IDF1 {r.idf1:6.2f}')
    print(f'[{tag}] done ({time.time() - t0:.0f}s) -> {done}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--role', choices=['val', 'test'], required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tag', default='model')
    p.add_argument('--variant', default='')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--clean_params', default=None, help='params.json written by --role val (clean protocol)')
    p.add_argument('--no_reid', action='store_true')
    args = p.parse_args()
    torch.set_float32_matmul_precision('medium')
    evaluate(args.role, args.ckpt, args.data_dir, args.out, args.tag, args.variant, args.seed, args.clean_params,
             not args.no_reid)
