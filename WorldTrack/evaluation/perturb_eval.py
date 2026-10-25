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

--mode calib_sweep repeats the calibration error with several random draws per setting:
  fixed_angle       every camera rotated by exactly `angle` degrees about its own random axis
  roll_pitch_gauss  every camera gets Gaussian roll and pitch noise with standard deviation `sigma` degrees
                    (the extrinsic-perturbation protocol of BEVHeight-style robustness evaluations)
  k_cameras         a random subset of k cameras rotated by 1 degree (k = 1..all), as in NCGR
It also reports how far the error moves ground points (median, world units: cm on WILDTRACK).

    python -m evaluation.perturb_eval --results /content/drive/MyDrive/retrack_results/full \
        --data_dir /content/data/Wildtrack [--mode calib_sweep]
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
SWEEP_ANGLES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)  # degrees, every camera
SWEEP_SIGMAS = (0.25, 0.5, 0.75, 1.0)  # degrees, roll/pitch standard deviation
K_CAMERAS_ANGLE = 1.0  # degrees, for the "how many cameras are miscalibrated" curve


def rotation(axis, angle_deg):
    """4x4 rotation about a unit axis (Rodrigues)."""
    a = torch.as_tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    t = math.radians(angle_deg)
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float32)
    R = torch.eye(4)
    R[:3, :3] = torch.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
    return R


def roll_pitch_noise(rng, sigma_deg):
    """Gaussian pitch (about the camera x axis) and roll (about the optical axis) in the camera frame."""
    pitch, roll = rng.normal(0.0, sigma_deg, 2)
    return rotation([1, 0, 0], pitch) @ rotation([0, 0, 1], roll)


def sweep_conditions(num_cams, draws, seed=0):
    """(protocol, magnitude, draw, {camera: 4x4 rotation}) for the all-camera calibration sweep."""
    rng = np.random.default_rng(seed)
    conds = [('clean', 0.0, 0, None)]
    for angle in SWEEP_ANGLES:
        conds += [('fixed_angle', angle, d, {c: rotation(rng.normal(size=3), angle) for c in range(num_cams)})
                  for d in range(draws)]
    for sigma in SWEEP_SIGMAS:
        conds += [('roll_pitch_gauss', sigma, d, {c: roll_pitch_noise(rng, sigma) for c in range(num_cams)})
                  for d in range(draws)]
    # k of the cameras (random subset) rotated by K_CAMERAS_ANGLE, as in NCGR's 1..N-camera evaluation
    for k in range(1, num_cams + 1):
        for d in range(draws):
            cams = sorted(rng.choice(num_cams, k, replace=False).tolist())
            conds.append(('k_cameras', k, d, {c: rotation(rng.normal(size=3), K_CAMERAS_ANGLE) for c in cams}))
    return conds


def ground_displacement(items, spec, frame_step=4, original_width=1920):
    """
    How far a wrong calibration moves ground points, as (median ground shift in world units, median reprojection
    error in pixels at the original image resolution). Ground-truth positions are projected into each camera with
    the true calibration and back-projected to the ground with the perturbed one; the pixel error is the distance
    between projecting with the true and with the perturbed calibration (the usual calibration-quality measure, so it
    can be compared with WILDTRACK's own calibration error from evaluation/diagnose.py).
    """
    dists, pixels = [], []
    for item in items[::frame_step]:
        gt = item['grid_gt'][0]
        gt = gt[gt.sum(1) != 0][:, :2].float()
        if len(gt) == 0:
            continue
        ref = torch.cat([gt, torch.zeros(len(gt), 1), torch.ones(len(gt), 1)], 1)
        world = (torch.inverse(item['ref_T_global'][0].float()) @ ref.T)  # 4,N, ground at z=0
        ground = torch.stack([world[0], world[1], torch.ones(world.shape[1])])  # 3,N
        H_img, W_img = item['img'].shape[-2:]
        for c, R in spec.items():
            K = item['intrinsic'][0, c, :3, :3].float()
            E = item['extrinsic'][0, c].float()
            H = K @ E[:3][:, [0, 1, 3]]
            H_wrong = K @ (R @ E)[:3][:, [0, 1, 3]]
            pix = H @ ground
            u = pix[:2] / pix[2].clamp(min=1e-9)
            visible = (pix[2] > 0) & (u[0] >= 0) & (u[0] < W_img) & (u[1] >= 0) & (u[1] < H_img)
            back = torch.linalg.solve(H_wrong, pix)
            ok = visible & (back[2].abs() > 1e-9)
            if ok.any():
                xy = back[:2, ok] / back[2, ok]
                dists.append((xy - ground[:2, ok]).norm(dim=0))
            pix_wrong = H_wrong @ ground
            ok = visible & (pix_wrong[2] > 0)
            if ok.any():
                u_wrong = pix_wrong[:2, ok] / pix_wrong[2, ok]
                pixels.append((u_wrong - u[:, ok]).norm(dim=0) * (original_width / W_img))
    med = lambda xs: float(torch.cat(xs).median()) if xs else float('nan')
    return med(dists), med(pixels)


def condition_metrics(frames):
    row = {}
    det = detection_metrics(frames, 0.5)
    row.update({k: det[k] for k in ('moda', 'modp', 'recall', 'precision', 'det_fn', 'det_fp')})
    for name, kw in [('S0', dict(tracker='baseline', track_buffer=5)), ('S2', S2_DEFAULTS)]:
        trk = tracking_metrics(frames, 0.5, kw)
        row.update({f'{name}_{k}': trk[k] for k in ('mota', 'idf1', 'id_switches', 'fragmentations')})
    return row


def calib_sweep_model(ckpt, tag, variant, data_dir, out_dir, draws=5, max_frames=None):
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
    rows, t0 = [], time.time()
    for protocol, mag, draw, spec in sweep_conditions(num_cams, draws):
        perturb = None if spec is None else make_perturb('rot_all', list(spec), spec, items)
        shift, px = (0.0, 0.0) if spec is None else ground_displacement(items, spec)
        row = {'tag': tag, 'variant': variant, 'protocol': protocol, 'magnitude': mag, 'draw': draw,
               'ground_shift': shift, 'pixel_error': px}
        row.update(condition_metrics(run_inference(model, items, perturb=perturb)))
        rows.append(row)
        print(f'[{tag}] {protocol:16s} {mag:<5} draw {draw}  shift {row["ground_shift"]:5.1f} cm '
              f'{row["pixel_error"]:5.1f} px  '
              f'MODA {row["moda"]:6.2f}  recall {row["recall"]:6.2f}  MOTA {row["S0_mota"]:6.2f}  '
              f'({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def summarise_sweep(df):
    """
    Per (variant, protocol, magnitude): MODA averaged over draws and seeds. ΔMODA is against each seed's own clean
    score. "±" is the std over random draws for a single seed, and the std over seeds when there are several.
    """
    lines = []
    for variant, g in df.groupby('variant', sort=False):
        clean = g[g.protocol == 'clean'].set_index('tag')
        n = len(clean)
        lines.append(f'\n### {variant}  ({n} seed{"s" if n > 1 else ""}; clean: MODA {clean.moda.mean():.1f}, '
                     f'MOTA {clean.S0_mota.mean():.1f}, IDF1 {clean.S0_idf1.mean():.1f})\n')
        lines.append(f'| protocol | magnitude | reprojection error (px) | ground shift | MODA (mean ± std over '
                     f'{"seeds" if n > 1 else "draws"}) | ΔMODA | recall | precision | MOTA (S0) | IDF1 (S0) '
                     f'| MOTA (S2) | IDF1 (S2) |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for (protocol, mag), h in g[g.protocol != 'clean'].groupby(['protocol', 'magnitude'], sort=False):
            name, value = {'fixed_angle': ('fixed angle, all cameras', f'{mag:g}°'),
                           'roll_pitch_gauss': ('roll/pitch Gaussian, all cameras', f'σ = {mag:g}°'),
                           'k_cameras': (f'{K_CAMERAS_ANGLE:g}° on k cameras', f'k = {int(mag)}')}[protocol]
            per_seed = h.groupby('tag').moda.mean()
            spread = per_seed.std() if n > 1 else h.moda.std()
            delta = (per_seed - clean.moda).dropna().mean()
            px = h.pixel_error.median() if 'pixel_error' in h else float('nan')
            lines.append(f'| {name} | {value} | {px:.0f} | {h.ground_shift.median():.0f} | {per_seed.mean():.1f} '
                         f'± {spread:.1f} | {delta:+.1f} | {h.recall.mean():.1f} | {h.precision.mean():.1f} '
                         f'| {h.S0_mota.mean():.1f} | {h.S0_idf1.mean():.1f} | {h.S2_mota.mean():.1f} '
                         f'| {h.S2_idf1.mean():.1f} |')
    return '\n'.join(lines)


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
        row.update(condition_metrics(frames))
        rows.append(row)
        print(f'[{tag}] {kind:7s} {mag:<4} cams={row["cameras"]:>13s}  MODA {row["moda"]:6.2f}  '
              f'MOTA {row["S0_mota"]:6.2f}  IDF1 {row["S0_idf1"]:6.2f}  ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def summarise(df):
    """
    Mean over cameras (and seeds) per (variant, fault). ΔMODA is taken against each seed's own clean score; "worse
    than dropping" counts (seed, camera) pairs where the fault gives lower MODA than switching that camera off.
    """
    drop = df[df.kind == 'drop'].set_index(['tag', 'cameras'])['moda']
    lines = []
    for variant, g in df.groupby('variant', sort=False):
        clean = g[g.kind == 'clean'].set_index('tag')
        n = len(clean)
        lines.append(f'\n### {variant}  ({n} seed{"s" if n > 1 else ""}; clean: MODA {clean.moda.mean():.1f}, '
                     f'MOTA {clean.S0_mota.mean():.1f}, IDF1 {clean.S0_idf1.mean():.1f})\n')
        lines.append('| fault | MODA | ΔMODA | recall | precision | MOTA (S0) | IDF1 (S0) | MOTA (S2) | IDF1 (S2) '
                     '| worse than dropping the camera |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for (kind, mag), h in g[g.kind != 'clean'].groupby(['kind', 'magnitude'], sort=False):
            label = {'drop': 'camera dropped', 'rot': f'1 camera rotated {mag}°', 'rot_all': f'all cameras rotated {mag}°',
                     'lag': f'1 camera {int(mag)} frame(s) late', 'freeze': '1 camera frozen'}[kind]
            delta = (h.groupby('tag').moda.mean() - clean.moda).dropna().mean()
            worse = ''
            if kind in ('rot', 'lag', 'freeze'):
                n_worse = sum(r.moda < drop.loc[(r.tag, r.cameras)] for r in h.itertuples())
                worse = f'{n_worse}/{len(h)}'
            lines.append(f'| {label} | {h.moda.mean():.1f} | {delta:+.1f} | {h.recall.mean():.1f} '
                         f'| {h.precision.mean():.1f} | {h.S0_mota.mean():.1f} | {h.S0_idf1.mean():.1f} '
                         f'| {h.S2_mota.mean():.1f} | {h.S2_idf1.mean():.1f} | {worse} |')
    return '\n'.join(lines)


def find_runs(results, seeds):
    runs = []
    for p in sorted(glob.glob(osp.join(results, 'runs', '*', 'DONE.json'))):
        info = json.load(open(p))
        if not seeds or int(info['seed']) in seeds:
            runs.append((info['ckpt'], f"{info['variant']}_s{info['seed']}", info['variant']))
    return runs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', help='pipeline results folder: evaluates every finished run of --seed')
    p.add_argument('--seed', type=int, nargs='+', default=[80085], help='training seeds to evaluate')
    p.add_argument('--ckpt', action='append', default=[], help='evaluate this checkpoint (repeatable)')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', default=None, help='default: <results>/perturb')
    p.add_argument('--variants', nargs='+', default=None, help='only these variants (default: all finished)')
    p.add_argument('--cams', type=int, nargs='+', default=None, help='0-based cameras to perturb (default: all)')
    p.add_argument('--max_frames', type=int, default=None, help='debug: only the first N test frames')
    p.add_argument('--mode', choices=['faults', 'calib_sweep'], default='faults')
    p.add_argument('--draws', type=int, default=5, help='calib_sweep: random draws per setting')
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

    if args.mode == 'calib_sweep':
        out_dir = osp.join(out_dir, 'calib_sweep')
        dfs = [calib_sweep_model(ckpt, tag, variant, args.data_dir, out_dir, args.draws, args.max_frames)
               for ckpt, tag, variant in runs]
        text = '# Calibration sweep: every camera miscalibrated\n' \
               f'Threshold 0.5. {args.draws} random draws per setting. The model keeps the original calibration ' \
               'while the cameras are rotated. "Reprojection error" = median pixel error the wrong calibration ' \
               'causes at full resolution (1920x1080), comparable with WILDTRACK\'s own calibration error of ' \
               '2-17 px per camera (evaluation/diagnose.py). "Ground shift" = median distance a ground point moves ' \
               'when projected into a camera and back with the wrong calibration (cm on WILDTRACK). ' \
               'S0 = TrackTacular tracker, S2 = ByteTrack-style tracker.\n' + \
               summarise_sweep(pd.concat(dfs, ignore_index=True))
        with open(osp.join(out_dir, 'CALIB_SWEEP.md'), 'w') as f:
            f.write(text)
        print('\n' + text + f'\n\nwritten to {out_dir}/CALIB_SWEEP.md')
        raise SystemExit
    dfs = [evaluate_model(ckpt, tag, variant, args.data_dir, out_dir, args.cams, args.max_frames)
           for ckpt, tag, variant in runs]
    text = '# Robustness pilot: camera faults other than a dropped camera\n' \
           'Threshold 0.5. S0 = TrackTacular tracker, S2 = ByteTrack-style tracker. Single-camera faults are ' \
           'averaged over the cameras; "worse than dropping" counts cameras for which the fault gives lower MODA ' \
           'than switching that camera off.\n' + summarise(pd.concat(dfs, ignore_index=True))
    with open(osp.join(out_dir, 'SUMMARY.md'), 'w') as f:
        f.write(text)
    print('\n' + text + f'\n\nwritten to {out_dir}/SUMMARY.md')
