"""
ARL pipeline (research_methodology_attempt_1/METHODOLOGY.md): does every addition improve MODA / MOTA / IDF1 on
WILDTRACK over the retrained TrackTacular baseline?

Training-time additions are switched on and off by training one variant per combination:
    A1 3D random occlusion   A2 scene (BEV) augmentation   A3 temporal BEV history   A5 weight averaging
    variants are named by their additions, e.g. base, A1, A1A2A3, A1A2A3A5
    presets: core = base, A1, A2, A3, A1A2A3 (add-one + full)
             loo  = A2A3, A1A3, A1A2       (leave-one-out from the full method)
             avg  = A1A2A3A5              (A5 on top of the full method)
             all  = core + loo + avg      (default for --mode full)
Post-hoc additions (B1 threshold, C1 online tracker, R appearance, C2 linking, C2b flow, C3 interpolation) are
switched on the cached detections of every trained model, in every on/off combination (evaluation/arl_eval.py).

For every variant:
  * one validation model (first seed) trained on frames 0-323 -> parameters tuned on frames 324-359 (clean)
  * one model per seed trained on frames 0-359 -> scored on frames 360-399 with the clean parameters and,
    labelled as oracle, with parameters tuned on the test frames
The report (evaluation/arl_report.py) gives a keep/drop verdict per addition and copies the best checkpoints.

Everything lives under one folder, <results>/arl_<mode>/ (put it on Google Drive):
    runs/<variant>_<split>_s<seed>/   training logs, config, last.ckpt (and averaged.ckpt for A5), DONE.json
    eval/<variant>_<role>_s<seed>/     cached detections, tuned parameters, metrics.csv, info.json
    report/                            REPORT.md, ladder.csv, verdicts.csv, all_cells.csv, additions.png
    best/                              best_model.ckpt (+ params, model card), per_variant/<variant>.ckpt
    for_claude.zip                     everything except checkpoints, to hand over for analysis
    env.json, pipeline.log, MANIFEST.md
Every step is skipped when its output exists and interrupted training resumes from last.ckpt, so after a Colab
disconnect the same command continues where it stopped.

    python arl_pipeline.py --results /content/drive/MyDrive/retrack_results --mode quick   # smoke run
    python arl_pipeline.py --results /content/drive/MyDrive/retrack_results --mode full    # the study
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
import shlex
import subprocess
import sys
import time
import zipfile

from retrack_pipeline import HERE, SEEDS, log, run, prepare_wildtrack

PRESETS = {'core': ['base', 'A1', 'A2', 'A3', 'A1A2A3'], 'loo': ['A2A3', 'A1A3', 'A1A2'], 'avg': ['A1A2A3A5']}
PRESETS['all'] = PRESETS['core'] + PRESETS['loo'] + PRESETS['avg']
QUICK_VARIANTS = ['base', 'A1A2A3']


def components(variant):
    comps = set(re.findall(r'A\d', variant))
    assert variant == 'base' or comps, f'unknown variant {variant}'
    assert comps <= {'A1', 'A2', 'A3', 'A5'}, f'unknown addition in {variant}'
    return comps


def expand(names):
    out = []
    for n in names:
        for v in PRESETS.get(n, [n]):
            v = ''.join(sorted(components(v))) or 'base'
            if v not in out:
                out.append(v)
    return out


# ------------------------------------------------------------------ training
def train(variant, split, seed, data_dir, results, args):
    comps = components(variant)
    run_dir = osp.join(results, 'runs', f'{variant}_{split}_s{seed}')
    done = osp.join(run_dir, 'DONE.json')
    if osp.exists(done):
        return json.load(open(done))['ckpt']
    os.makedirs(run_dir, exist_ok=True)
    schedule = 'configs/t_arl_avg.yml' if 'A5' in comps else 'configs/t_retrack.yml'
    cmd = [sys.executable, 'world_track.py', 'fit',
           '-c', schedule, '-c', 'configs/d_wildtrack.yml', '-c', 'configs/m_segnet.yml', '-c', 'configs/v_concat.yml',
           '--model.use_temporal_cache', 'true' if 'A3' in comps else 'false',
           '--data.init_args.data_dir', data_dir,
           '--data.init_args.split', split,
           '--data.init_args.occlusion_prob', str(args.occ_p if 'A1' in comps else 0.0),
           '--data.init_args.occlusion_width_ratio', str(args.occ_alpha),
           '--data.init_args.scene_aug_prob', str(args.scene_p if 'A2' in comps else 0.0),
           '--data.init_args.scene_aug_mode', args.scene_mode,
           '--data.init_args.batch_size', str(args.batch_size),
           '--data.init_args.num_workers', str(args.num_workers),
           '--trainer.accumulate_grad_batches', str(args.accumulate),
           '--trainer.max_epochs', str(args.epochs),
           '--trainer.precision', args.precision,
           '--trainer.default_root_dir', run_dir,
           '--seed_everything', str(seed)] + shlex.split(args.train_args)
    previous = sorted(glob.glob(osp.join(run_dir, 'lightning_logs', '*', 'checkpoints', 'last.ckpt')),
                      key=osp.getmtime)
    if previous:
        print(f'resuming from {previous[-1]}')
        cmd += ['--ckpt_path', previous[-1]]
    log(f'Training {variant} on the {split} split (seed {seed}, {args.epochs} epochs) -> {run_dir}')
    t0 = time.time()
    run(cmd)
    name = 'averaged.ckpt' if 'A5' in comps else 'last.ckpt'
    ckpt = sorted(glob.glob(osp.join(run_dir, 'lightning_logs', '*', 'checkpoints', name)), key=osp.getmtime)[-1]
    json.dump({'ckpt': ckpt, 'ckpt_rel': osp.relpath(ckpt, results), 'variant': variant, 'additions': sorted(comps),
               'split': split, 'seed': seed, 'epochs': args.epochs,
               'train_minutes_last_session': round((time.time() - t0) / 60, 1)}, open(done, 'w'), indent=1)
    return ckpt


# ------------------------------------------------------------------ evaluation / report / bundle
def evaluate(role, ckpt, variant, seed, data_dir, results, args, clean_params=None):
    tag = f'{variant}_{role}_s{seed}'
    out = osp.join(results, 'eval', tag)
    if osp.exists(osp.join(out, 'params.json' if role == 'val' else 'metrics.csv')):
        return out
    log(f'Evaluating {tag}')
    cmd = [sys.executable, '-m', 'evaluation.arl_eval', '--role', role, '--ckpt', ckpt, '--data_dir', data_dir,
           '--out', out, '--tag', tag, '--variant', variant, '--seed', str(seed)]
    if clean_params:
        cmd += ['--clean_params', clean_params]
    if args.no_reid:
        cmd += ['--no_reid']
    run(cmd)
    return out


def report_and_bundle(results, final=False):
    r = subprocess.run([sys.executable, '-m', 'evaluation.arl_report', '--results', results], cwd=HERE,
                       stdout=None if final else subprocess.DEVNULL)
    if final and r.returncode != 0:
        raise RuntimeError('report failed')
    bundle(results)


def bundle(results):
    """for_claude.zip: the whole results folder except checkpoints (all CSVs, cached detections, parameters,
    training logs and configs, report, pipeline log), small enough to share for analysis."""
    write_manifest(results)
    path = osp.join(results, 'for_claude.zip')
    with zipfile.ZipFile(path + '.part', 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(results):
            for f in files:
                full = osp.join(root, f)
                if f.endswith(('.ckpt', '.part', '.zip')):
                    continue
                z.write(full, osp.relpath(full, results))
    os.replace(path + '.part', path)


def write_manifest(results):
    lines = ['# ARL results folder', '', 'Produced by `WorldTrack/arl_pipeline.py`.', '']
    runs = sorted(glob.glob(osp.join(results, 'runs', '*', 'DONE.json')))
    evals = sorted(glob.glob(osp.join(results, 'eval', '*', 'info.json')))
    lines += [f'- trained models: {len(runs)}', f'- evaluations: {len(evals)}', '']
    for p in runs:
        d = json.load(open(p))
        lines.append(f'- run `{osp.relpath(osp.dirname(p), results)}`: variant {d["variant"]}, split {d["split"]}, '
                     f'seed {d["seed"]}, {d["epochs"]} epochs, checkpoint `{d.get("ckpt_rel", d["ckpt"])}`')
    open(osp.join(results, 'MANIFEST.md'), 'w').write('\n'.join(lines) + '\n')


def write_env(results, args):
    env = {'args': vars(args), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        env['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
        env['git_branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=HERE,
                                                    text=True).strip()
    except Exception:
        pass
    try:
        import torch
        env['torch'] = torch.__version__
        env['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    json.dump(env, open(osp.join(results, f'env_{time.strftime("%Y%m%d_%H%M%S")}.json'), 'w'), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='parent folder (Google Drive); outputs go to <results>/arl_<mode>')
    p.add_argument('--data_root', default='/content/data')
    p.add_argument('--mode', choices=['full', 'quick'], default='full')
    p.add_argument('--variants', nargs='+', default=None,
                   help=f'variant names or presets {list(PRESETS)} (full: all, quick: {QUICK_VARIANTS})')
    p.add_argument('--seeds', type=int, default=None, help='training seeds per variant (full: 3, quick: 1)')
    p.add_argument('--epochs', type=int, default=None, help='full: 30 as TrackTacular, quick: 2')
    p.add_argument('--occ_p', type=float, default=1.0, help='A1 occlusion probability (3DROM best: 1.0)')
    p.add_argument('--occ_alpha', type=float, default=0.4, help='A1 occluder width / height (not given by 3DROM)')
    p.add_argument('--scene_p', type=float, default=0.5, help='A2 probability (MVAug best: 0.5)')
    p.add_argument('--scene_mode', default='flip', choices=['flip', 'flip_affine'])
    p.add_argument('--no_reid', action='store_true', help='skip the OSNet appearance switch R')
    p.add_argument('--skip_val', action='store_true',
                   help='diagnostics only: no validation models, so no clean protocol (oracle only), half the time')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--precision', default='32-true', help="'32-true' as the original; 'bf16-mixed' is faster")
    p.add_argument('--train_args', default='', help='extra arguments for `world_track.py fit`')
    args = p.parse_args()

    quick = args.mode == 'quick'
    args.variants = expand(args.variants or (QUICK_VARIANTS if quick else ['all']))
    args.seeds = args.seeds or (1 if quick else 3)
    args.epochs = args.epochs or (2 if quick else 30)
    seeds = SEEDS[:args.seeds]
    results = osp.join(args.results, f'arl_{args.mode}')
    os.makedirs(results, exist_ok=True)
    write_env(results, args)
    n_runs = len(args.variants) * (len(seeds) + (0 if args.skip_val else 1))
    print(f'ARL pipeline: mode={args.mode} variants={args.variants} seeds={seeds} epochs={args.epochs} '
          f'-> {results}\n{n_runs} training runs ({len(args.variants)} variants x ({len(seeds)} seeds'
          f'{"" if args.skip_val else " + 1 validation model"})); about 25 min each on an A100 at 30 epochs')

    log('Data')
    data_dir = prepare_wildtrack(args.data_root)

    # one seed of every variant first, so that a complete comparison exists as early as possible
    clean = {}
    for si, seed in enumerate(seeds):
        for variant in args.variants:
            if si == 0 and not args.skip_val:
                vck = train(variant, 'val', seed, data_dir, results, args)
                vout = evaluate('val', vck, variant, seed, data_dir, results, args)
                clean[variant] = osp.join(vout, 'params.json')
            ckpt = train(variant, 'standard', seed, data_dir, results, args)
            evaluate('test', ckpt, variant, seed, data_dir, results, args, clean_params=clean.get(variant))
            report_and_bundle(results)

    log('Report')
    report_and_bundle(results, final=True)
    print(f'\nREPORT: {osp.join(results, "report", "REPORT.md")}\nbest checkpoints: {osp.join(results, "best")}\n'
          f'to share for analysis: {osp.join(results, "for_claude.zip")}')


if __name__ == '__main__':
    main()
