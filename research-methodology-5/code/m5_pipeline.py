"""
Research methodology 5: every combination of R1 (DA3 features), R2 (sparse queries) and R4 (track queries), trained
and evaluated on WILDTRACK and MultiviewX.

    python research-methodology-5/code/m5_pipeline.py --results /content/drive/MyDrive/retrack_results_m5

Steps (each one is skipped when its output already exists, so re-running continues where it stopped):
  1. preflight: every variant trains 4 batches on WILDTRACK, is reloaded the way the evaluation loads it and is
     evaluated on 3 test frames; B0 and R124 also on MultiviewX (catches crashes in minutes, not hours)
  2. training: TrackTacular's schedule (configs/t_retrack.yml): 30 epochs on WILDTRACK, 50 on MultiviewX
     (configs/d_multiviewx.yml), batch 2 x 8 accumulation, last checkpoint; up to --parallel runs at once
  3. evaluation: m5_eval.py on the 40 test frames, all cameras and every camera-removal combination (k <= 3)
  4. report: m5_report.py -> <results>/m5/REPORT.md, and <results>/m5/for_claude.zip (everything but checkpoints)

Variants (R3 was dropped): the full 2^3 factorial.
  B0    retrained TrackTacular (segnet, concat fusion, no history) in the current code
  R1    + frozen Depth Anything 3 multi-view features
  R2    + sparse ground-anchored queries instead of the dense BEV decoder
  R4    + track queries (the tracker's tracks are a network input)
  R12, R14, R24, R124   the combinations; R124 is the proposed architecture
Everything else (augmentation, schedule, fusion, no camera dropout, no temporal history) is the same in all variants.
"""
import argparse
import glob
import json
import os
import os.path as osp
import shlex
import subprocess
import sys
import time
import zipfile

HERE = osp.dirname(osp.abspath(__file__))
WORLDTRACK = osp.normpath(osp.join(HERE, '..', '..', 'WorldTrack'))

VARIANTS = {name: dict(da3='1' in name[1:], sparse='2' in name[1:], track='4' in name[1:])
            for name in ('B0', 'R1', 'R2', 'R4', 'R12', 'R14', 'R24', 'R124')}
VARIANTS['B0'] = dict(da3=False, sparse=False, track=False)
ORDER = ['B0', 'R124', 'R1', 'R2', 'R4', 'R12', 'R14', 'R24']  # baseline and full model first
DATASETS = {
    'wildtrack': dict(config='configs/d_wildtrack.yml', folder='Wildtrack', epochs=30),
    'multiviewx': dict(config='configs/d_multiviewx.yml', folder='MultiviewX', epochs=50),
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def flag(v):
    return str(bool(v)).lower()


def train_cmd(variant, dataset, seed, run_dir, args, epochs, extra=()):
    v, d = VARIANTS[variant], DATASETS[dataset]
    return [sys.executable, 'world_track.py', 'fit',
            '-c', 'configs/t_retrack.yml', '-c', d['config'], '-c', 'configs/m_segnet.yml',
            '-c', 'configs/v_concat.yml',
            '--model.use_temporal_cache', 'false',
            '--model.da3', flag(v['da3']),
            '--model.da3_repo', args.da3_repo,
            '--model.sparse_queries', flag(v['sparse']),
            '--model.track_queries', flag(v['track']),
            '--data.init_args.data_dir', osp.join(args.data_root, d['folder']),
            '--data.init_args.split', 'standard',
            '--data.init_args.occlusion_prob', '0.0',
            '--data.init_args.scene_aug_prob', '0.0',
            '--data.init_args.batch_size', str(args.batch_size),
            '--data.init_args.num_workers', str(args.num_workers),
            '--trainer.accumulate_grad_batches', str(args.accumulate),
            '--trainer.max_epochs', str(epochs),
            '--trainer.precision', args.precision,
            '--trainer.default_root_dir', run_dir,
            '--seed_everything', str(seed)] + list(extra) + shlex.split(args.train_args)


def eval_cmd(ckpt, out, variant, dataset, seed, args, quick=False):
    return [sys.executable, osp.join(HERE, 'm5_eval.py'), '--ckpt', ckpt,
            '--data_dir', osp.join(args.data_root, DATASETS[dataset]['folder']), '--out', out,
            '--dataset', dataset, '--variant', variant, '--seed', str(seed),
            '--num_workers', str(min(args.num_workers, 4))] + (['--quick'] if quick else [])


def last_ckpt(run_dir):
    found = sorted(glob.glob(osp.join(run_dir, 'lightning_logs', '*', 'checkpoints', 'last.ckpt')), key=osp.getmtime)
    return found[-1] if found else None


def run(cmd, log_path=None):
    print('$ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=WORLDTRACK, check=True)
        return None
    return subprocess.Popen(cmd, cwd=WORLDTRACK, stdout=open(log_path, 'a'), stderr=subprocess.STDOUT)


def preflight(jobs, root, args):
    checks = [(v, 'wildtrack') for v in ORDER if any(j[0] == v for j in jobs)]
    checks += [(v, 'multiviewx') for v in ('B0', 'R124') if any(j[0] == v and j[1] == 'multiviewx' for j in jobs)]
    for variant, dataset in checks:
        out = osp.join(root, 'preflight', f'{dataset}_{variant}')
        if osp.exists(osp.join(out, 'PASSED')):
            continue
        log(f'Preflight {dataset} {variant}: 4 training batches, checkpoint reload, evaluation on 3 frames')
        run(train_cmd(variant, dataset, 0, osp.join(out, 'run'), args, 1,
                      ['--trainer.limit_train_batches', '4', '--trainer.limit_val_batches', '2',
                       '--trainer.accumulate_grad_batches', '1']))
        run(eval_cmd(last_ckpt(osp.join(out, 'run')), osp.join(out, 'eval'), variant, dataset, 0, args, quick=True))
        open(osp.join(out, 'PASSED'), 'w').close()
    log('Preflight passed')


def finish(job, run_dir, args, ckpts):
    variant, dataset, seed = job
    ckpt = last_ckpt(run_dir)
    json.dump({'ckpt': ckpt, 'variant': variant, 'dataset': dataset, 'seed': seed,
               'epochs': args.epochs.get(dataset), **VARIANTS[variant]},
              open(osp.join(run_dir, 'DONE.json'), 'w'), indent=1)
    ckpts[job] = ckpt
    log(f'Trained {dataset} {variant} seed {seed} -> {ckpt}')


def evaluate(job, ckpt, root, args):
    variant, dataset, seed = job
    out = osp.join(root, 'eval', f'{dataset}_{variant}_s{seed}')
    if osp.exists(osp.join(out, 'metrics.csv')):
        return
    log(f'Evaluating {dataset} {variant} seed {seed}')
    run(eval_cmd(ckpt, out, variant, dataset, seed, args))


def train_and_eval(jobs, root, args):
    """Up to args.parallel trainings at once; each finished training is evaluated right away (sequentially)."""
    ckpts, pending, running = {}, [], []
    for job in jobs:
        variant, dataset, seed = job
        run_dir = osp.join(root, 'runs', f'{dataset}_{variant}_s{seed}')
        if osp.exists(osp.join(run_dir, 'DONE.json')):
            ckpts[job] = json.load(open(osp.join(run_dir, 'DONE.json')))['ckpt']
            evaluate(job, ckpts[job], root, args)
        else:
            pending.append((job, run_dir))
    while pending or running:
        while pending and len(running) < args.parallel:
            job, run_dir = pending.pop(0)
            variant, dataset, seed = job
            os.makedirs(run_dir, exist_ok=True)
            extra = ['--ckpt_path', last_ckpt(run_dir)] if last_ckpt(run_dir) else []
            epochs = args.epochs[dataset]
            log(f'Training {dataset} {variant} seed {seed} ({epochs} epochs){" - resuming" if extra else ""}')
            proc = run(train_cmd(variant, dataset, seed, run_dir, args, epochs, extra), osp.join(run_dir, 'train.log'))
            running.append((proc, job, run_dir, time.time()))
        for entry in list(running):
            proc, job, run_dir, t0 = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            if proc.returncode != 0:
                raise RuntimeError(f'training {job} failed, see {run_dir}/train.log')
            finish(job, run_dir, args, ckpts)
            evaluate(job, ckpts[job], root, args)
        if running:
            time.sleep(60)
            log('running: ' + ', '.join(f'{d} {v} s{s} ({(time.time() - t0) / 60:.0f} min)'
                                        for _, (v, d, s), _, t0 in running))
    return ckpts


def bundle(root):
    path = osp.join(root, 'for_claude.zip')
    with zipfile.ZipFile(path + '.part', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(osp.join(root, '**', '*'), recursive=True):
            if osp.isfile(f) and not f.endswith(('.ckpt', '.zip', '.part', '.pt')):
                z.write(f, osp.relpath(f, root))
    os.replace(path + '.part', path)
    log(f'Bundle for analysis -> {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='parent folder (Google Drive); outputs go to <results>/m5')
    p.add_argument('--data_root', default='/content/data', help='holds Wildtrack/ and MultiviewX/')
    p.add_argument('--datasets', nargs='+', default=['wildtrack', 'multiviewx'], choices=list(DATASETS))
    p.add_argument('--variants', nargs='+', default=ORDER, choices=ORDER)
    p.add_argument('--seeds', nargs='+', type=int, default=[80085],
                   help='80085 is the seed of the earlier baselines; add seeds for confirmation runs')
    p.add_argument('--parallel', type=int, default=2, help='trainings at once on one GPU')
    p.add_argument('--epochs_wildtrack', type=int, default=DATASETS['wildtrack']['epochs'])
    p.add_argument('--epochs_multiviewx', type=int, default=DATASETS['multiviewx']['epochs'])
    p.add_argument('--da3_repo', default='depth-anything/DA3-BASE',
                   help='DA3-BASE (Apache 2.0, 0.12 B); DA3-LARGE is CC BY-NC 4.0')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=5)
    p.add_argument('--precision', default='32-true', help='as the earlier baselines')
    p.add_argument('--train_args', default='', help='extra arguments for `world_track.py fit`')
    p.add_argument('--skip_preflight', action='store_true')
    args = p.parse_args()
    args.epochs = {'wildtrack': args.epochs_wildtrack, 'multiviewx': args.epochs_multiviewx}

    for d in args.datasets:
        folder = osp.join(args.data_root, DATASETS[d]['folder'])
        assert all(osp.isdir(osp.join(folder, s)) for s in ('Image_subsets', 'annotations_positions', 'calibrations')), \
            f'{d} not found at {folder}'
    jobs = [(v, d, s) for s in args.seeds for v in ORDER if v in args.variants for d in args.datasets]
    root = osp.join(args.results, 'm5')
    os.makedirs(root, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=WORLDTRACK, capture_output=True, text=True).stdout.strip()
    json.dump(dict(vars(args), git_commit=commit, time=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(osp.join(root, f'env_{time.strftime("%Y%m%d_%H%M%S")}.json'), 'w'), indent=1)
    log(f'Research methodology 5: {len(jobs)} training(s), datasets {args.datasets}, seeds {args.seeds}, '
        f'parallel={args.parallel}, commit {commit[:7]}')

    if not args.skip_preflight:
        preflight(jobs, root, args)
    train_and_eval(jobs, root, args)
    subprocess.run([sys.executable, osp.join(HERE, 'm5_report.py'), '--results', root], check=True)
    bundle(root)


if __name__ == '__main__':
    main()
