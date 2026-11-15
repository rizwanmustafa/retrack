"""
Trains and evaluates the TrackTacular baseline (B0) and R2 (sparse person queries) on WILDTRACK and MultiviewX.

    python colab/pipeline.py --results /content/drive/MyDrive/retrack_results_r2 --seeds 80085 1 2

Every step is skipped when its output exists, so re-running the same command continues where it stopped:
  1. preflight   each variant trains 4 batches, is reloaded the way the evaluation loads it and evaluated on 3 frames
  2. training    TrackTacular's schedule: batch 2 x 8 accumulation, 30 epochs (WILDTRACK) / 50 (MultiviewX), no
                 temporal history, last checkpoint; up to --parallel runs at once; an interrupted run resumes
  3. evaluation  evaluation/r2_eval.py: all cameras and every combination of up to 3 removed cameras
  4. report      colab/report.py -> <results>/REPORT.md + figures, and <results>/for_claude.zip (no checkpoints)

--reuse <folder> takes finished runs from an earlier results folder (same layout: runs/<dataset>_<variant>_s<seed>)
instead of training them again, e.g. the B0 and R2 runs of research methodology 5 (retrack_results_m5/m5). They are
re-evaluated with this branch's evaluation.

Layout of <results>:
    runs/<dataset>_<variant>_s<seed>/   lightning_logs/version_*/checkpoints/last.ckpt, train.log, DONE.json
    eval/<dataset>_<variant>_s<seed>/   metrics.csv (one row per camera condition), summary.json, all_cameras_frames.pt
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
WORLDTRACK = osp.normpath(osp.join(HERE, '..', 'WorldTrack'))

VARIANTS = {'B0': [], 'R2': ['-c', 'configs/m_r2.yml']}
DATASETS = {
    'wildtrack': dict(config='configs/d_wildtrack.yml', folder='Wildtrack', epochs=30),
    'multiviewx': dict(config='configs/d_multiviewx.yml', folder='MultiviewX', epochs=50),
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def tag(variant, dataset, seed):
    return f'{dataset}_{variant}_s{seed}'


def train_cmd(variant, dataset, seed, run_dir, args, epochs, extra=()):
    d = DATASETS[dataset]
    return [sys.executable, 'world_track.py', 'fit',
            '-c', 'configs/t_colab.yml', '-c', d['config'], '-c', 'configs/m_segnet.yml', *VARIANTS[variant],
            '--model.use_temporal_cache', 'false',
            '--data.init_args.data_dir', osp.join(args.data_root, d['folder']),
            '--data.init_args.batch_size', str(args.batch_size),
            '--data.init_args.num_workers', str(args.num_workers),
            '--trainer.accumulate_grad_batches', str(args.accumulate),
            '--trainer.max_epochs', str(epochs),
            '--trainer.precision', args.precision,
            '--trainer.default_root_dir', run_dir,
            '--seed_everything', str(seed)] + list(extra) + shlex.split(args.train_args)


def eval_cmd(ckpt, out, dataset, args, quick=False):
    return [sys.executable, '-m', 'evaluation.r2_eval', '--ckpt', ckpt,
            '--data_dir', osp.join(args.data_root, DATASETS[dataset]['folder']), '--out', out,
            '--max_drop', str(args.max_drop), '--num_workers', str(min(args.num_workers, 4))] + \
        (['--quick'] if quick else [])


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
    for variant, dataset in sorted({(v, d) for v, d, _ in jobs}):
        out = osp.join(root, 'preflight', tag(variant, dataset, 0))
        if osp.exists(osp.join(out, 'PASSED')):
            continue
        log(f'Preflight {dataset} {variant}: 4 training batches, checkpoint reload, evaluation on 3 frames')
        run(train_cmd(variant, dataset, 0, osp.join(out, 'run'), args, 1,
                      ['--trainer.limit_train_batches', '4', '--trainer.limit_val_batches', '2',
                       '--trainer.accumulate_grad_batches', '1']))
        run(eval_cmd(last_ckpt(osp.join(out, 'run')), osp.join(out, 'eval'), dataset, args, quick=True))
        open(osp.join(out, 'PASSED'), 'w').close()
    log('Preflight passed')


def reuse(job, run_dir, args):
    """Marks the run as done with the checkpoint of an earlier results folder (no copy, no training)."""
    if not args.reuse:
        return False
    src = osp.join(args.reuse, 'runs', tag(*job))
    if not osp.exists(osp.join(src, 'DONE.json')):
        return False
    ckpt = json.load(open(osp.join(src, 'DONE.json'))).get('ckpt') or last_ckpt(src)
    if not ckpt or not osp.exists(ckpt):
        return False
    os.makedirs(run_dir, exist_ok=True)
    json.dump({'ckpt': ckpt, 'variant': job[0], 'dataset': job[1], 'seed': job[2], 'reused_from': src},
              open(osp.join(run_dir, 'DONE.json'), 'w'), indent=1)
    log(f'Reusing {tag(*job)} from {src}')
    return True


def evaluate(job, ckpt, root, args):
    out = osp.join(root, 'eval', tag(*job))
    if osp.exists(osp.join(out, 'metrics.csv')):
        return
    log(f'Evaluating {tag(*job)}')
    run(eval_cmd(ckpt, out, job[1], args))


def train_and_eval(jobs, root, args):
    """Up to args.parallel trainings at once; each finished training is evaluated right away."""
    pending, running = [], []
    for job in jobs:
        run_dir = osp.join(root, 'runs', tag(*job))
        if osp.exists(osp.join(run_dir, 'DONE.json')) or reuse(job, run_dir, args):
            evaluate(job, json.load(open(osp.join(run_dir, 'DONE.json')))['ckpt'], root, args)
        else:
            pending.append((job, run_dir))
    while pending or running:
        while pending and len(running) < args.parallel:
            job, run_dir = pending.pop(0)
            variant, dataset, seed = job
            os.makedirs(run_dir, exist_ok=True)
            resume = last_ckpt(run_dir)
            extra = ['--ckpt_path', resume] if resume else []
            epochs = args.epochs[dataset]
            log(f'Training {tag(*job)} ({epochs} epochs){" - resuming" if resume else ""}; log: {run_dir}/train.log')
            proc = run(train_cmd(variant, dataset, seed, run_dir, args, epochs, extra), osp.join(run_dir, 'train.log'))
            running.append((proc, job, run_dir, time.time()))
        for entry in list(running):
            proc, job, run_dir, t0 = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            if proc.returncode != 0:
                os.system(f'tail -n 30 "{run_dir}/train.log"')
                raise RuntimeError(f'training {tag(*job)} failed, see {run_dir}/train.log')
            ckpt = last_ckpt(run_dir)
            json.dump({'ckpt': ckpt, 'variant': job[0], 'dataset': job[1], 'seed': job[2],
                       'minutes': round((time.time() - t0) / 60, 1)},
                      open(osp.join(run_dir, 'DONE.json'), 'w'), indent=1)
            log(f'Trained {tag(*job)} in {(time.time() - t0) / 60:.0f} min -> {ckpt}')
            evaluate(job, ckpt, root, args)
        if running:
            time.sleep(60)
            log('running: ' + ', '.join(f'{tag(*j)} ({(time.time() - t0) / 60:.0f} min)' for _, j, _, t0 in running))


def bundle(root):
    path = osp.join(root, 'for_claude.zip')
    with zipfile.ZipFile(path + '.part', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(osp.join(root, '**', '*'), recursive=True):
            if osp.isfile(f) and not f.endswith(('.ckpt', '.zip', '.part', '.mp4', '.gif')):
                z.write(f, osp.relpath(f, root))
    os.replace(path + '.part', path)
    log(f'Bundle for analysis -> {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='output folder (on Google Drive, so it survives the runtime)')
    p.add_argument('--data_root', default='/content/data', help='holds Wildtrack/ and MultiviewX/')
    p.add_argument('--datasets', nargs='+', default=['wildtrack', 'multiviewx'], choices=list(DATASETS))
    p.add_argument('--variants', nargs='+', default=['B0', 'R2'], choices=list(VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int, default=[80085])
    p.add_argument('--reuse', default='', help='earlier results folder whose finished runs are taken over')
    p.add_argument('--parallel', type=int, default=2, help='trainings at once on one GPU (2 for a 40+ GB GPU)')
    p.add_argument('--epochs_wildtrack', type=int, default=DATASETS['wildtrack']['epochs'])
    p.add_argument('--epochs_multiviewx', type=int, default=DATASETS['multiviewx']['epochs'])
    p.add_argument('--max_drop', type=int, default=3, help='evaluate with up to this many cameras removed')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=5)
    p.add_argument('--precision', default='32-true')
    p.add_argument('--train_args', default='', help='extra arguments for `world_track.py fit`')
    p.add_argument('--skip_preflight', action='store_true')
    args = p.parse_args()
    args.epochs = {'wildtrack': args.epochs_wildtrack, 'multiviewx': args.epochs_multiviewx}
    for d in args.datasets:
        folder = osp.join(args.data_root, DATASETS[d]['folder'])
        assert all(osp.isdir(osp.join(folder, s)) for s in ('Image_subsets', 'annotations_positions', 'calibrations')), \
            f'{d} not found at {folder}'
    jobs = [(v, d, s) for s in args.seeds for v in args.variants for d in args.datasets]
    root = args.results
    os.makedirs(root, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=WORLDTRACK, capture_output=True, text=True).stdout.strip()
    json.dump(dict(vars(args), git_commit=commit, time=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(osp.join(root, f'env_{time.strftime("%Y%m%d_%H%M%S")}.json'), 'w'), indent=1)
    log(f'{len(jobs)} run(s): variants {args.variants}, datasets {args.datasets}, seeds {args.seeds}, '
        f'parallel={args.parallel}, commit {commit[:7]}')
    if not args.skip_preflight:
        preflight(jobs, root, args)
    train_and_eval(jobs, root, args)
    subprocess.run([sys.executable, osp.join(HERE, 'report.py'), '--results', root], check=True)
    bundle(root)


if __name__ == '__main__':
    main()
