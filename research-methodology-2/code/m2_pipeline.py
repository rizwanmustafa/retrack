"""
Research methodology 2: train, evaluate and report in about one hour on an A100.

    python research-methodology-2/code/m2_pipeline.py --results /content/drive/MyDrive/retrack_results_m2

Steps (each one is skipped when its output already exists, so re-running continues where it stopped):
  1. preflight: every requested variant trains for a few batches and is evaluated once (catches crashes in
     about 3 minutes instead of after a 35-minute run)
  2. training: 30 epochs, same schedule as the attempt-1 baselines (configs/t_retrack.yml)
  3. evaluation: evaluation/arl_eval.py on the 40 test frames (cached detections + all post-hoc cells)
  4. report: m2_report.py -> <results>/m2/REPORT.md, and <results>/m2/for_claude.zip (everything but checkpoints)

Variants (see METHODOLOGY.md):
  M2   scene augmentation + VGF + ray conditioning + view-diversity gate prior   (the proposed method)
  M2c  same with GRACE's coverage prior instead of ours                          (is our prior worth it?)
  A2f  scene augmentation alone, with the batch-order fix                        (is VGF worth it?)
  VD   VGF + rays + view-diversity prior, no scene augmentation                  (is scene augmentation worth it?)
  M2cx M2c trained with view mixing (attempt 1's batch order, on purpose)        (does the "bug" add to VGF?)
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

VARIANTS = {
    'M2': dict(scene=True, vgf=True, ray=True, prior='diversity', mix=False),
    'M2c': dict(scene=True, vgf=True, ray=True, prior='coverage', mix=False),
    'A2f': dict(scene=True, vgf=False, ray=False, prior='diversity', mix=False),
    'VD': dict(scene=False, vgf=True, ray=True, prior='diversity', mix=False),
    'M2cx': dict(scene=True, vgf=True, ray=True, prior='coverage', mix=True),
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def train_cmd(variant, seed, run_dir, data_dir, args, epochs, extra=()):
    v = VARIANTS[variant]
    return [sys.executable, 'world_track.py', 'fit',
            '-c', 'configs/t_retrack.yml', '-c', 'configs/d_wildtrack.yml', '-c', 'configs/m_segnet.yml',
            '-c', 'configs/v_concat.yml',
            '--model.use_temporal_cache', 'false',
            '--model.vgf', str(v['vgf']).lower(),
            '--model.ray_cond', str(v['ray']).lower(),
            '--model.vgf_prior', v['prior'],
            '--model.view_mix', str(v['mix']).lower(),
            '--data.init_args.data_dir', data_dir,
            '--data.init_args.split', 'standard',
            '--data.init_args.occlusion_prob', '0.0',
            '--data.init_args.scene_aug_prob', str(args.scene_p if v['scene'] else 0.0),
            '--data.init_args.scene_aug_mode', 'flip',
            '--data.init_args.batch_size', str(args.batch_size),
            '--data.init_args.num_workers', str(args.num_workers),
            '--trainer.accumulate_grad_batches', str(args.accumulate),
            '--trainer.max_epochs', str(epochs),
            '--trainer.precision', args.precision,
            '--trainer.default_root_dir', run_dir,
            '--seed_everything', str(seed)] + list(extra) + shlex.split(args.train_args)


def last_ckpt(run_dir):
    found = sorted(glob.glob(osp.join(run_dir, 'lightning_logs', '*', 'checkpoints', 'last.ckpt')), key=osp.getmtime)
    return found[-1] if found else None


def eval_cmd(ckpt, out, tag, variant, seed, data_dir):
    return [sys.executable, '-m', 'evaluation.arl_eval', '--role', 'test', '--ckpt', ckpt, '--data_dir', data_dir,
            '--out', out, '--tag', tag, '--variant', variant, '--seed', str(seed), '--no_reid']


def run(cmd, log_path=None):
    print('$ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=WORLDTRACK, check=True)
        return None
    return subprocess.Popen(cmd, cwd=WORLDTRACK, stdout=open(log_path, 'a'), stderr=subprocess.STDOUT)


PREFLIGHT_CHECK = '''
import sys, torch
from evaluation.retrack_eval import load_model, load_test_items, run_inference
ckpt, data_dir, vgf, ray, prior = sys.argv[1:6]
model, hp = load_model(ckpt, 'cuda' if torch.cuda.is_available() else 'cpu')
got = (str(hp.get('vgf', False)).lower(), str(hp.get('ray_cond', False)).lower(), hp.get('vgf_prior', 'diversity'))
assert got == (vgf, ray, prior), f'checkpoint flags {got} != requested {(vgf, ray, prior)}'
with torch.no_grad():
    frames = run_inference(model, load_test_items(data_dir, model, num_workers=2)[:3])
print(f'preflight: checkpoint flags {got} ok, inference on {len(frames)} test frames ok')
'''


def preflight(variants, root, data_dir, args):
    """Per variant: 4 training batches (+2 validation batches), then reload the checkpoint the way the evaluation
    does and run inference on 3 test frames. Catches shape, flag and loading errors in a few minutes."""
    for variant in variants:
        out = osp.join(root, 'preflight', variant)
        if osp.exists(osp.join(out, 'PASSED')):
            continue
        log(f'Preflight {variant}: 4 training batches, checkpoint reload, inference')
        run(train_cmd(variant, 0, osp.join(out, 'run'), data_dir, args, 1,
                      ['--trainer.limit_train_batches', '4', '--trainer.limit_val_batches', '2',
                       '--trainer.accumulate_grad_batches', '1']))
        v = VARIANTS[variant]
        run([sys.executable, '-c', PREFLIGHT_CHECK, last_ckpt(osp.join(out, 'run')), data_dir,
             str(v['vgf']).lower(), str(v['ray']).lower(), v['prior']])
        open(osp.join(out, 'PASSED'), 'w').close()
    log('Preflight passed')


def train_all(jobs, root, data_dir, args):
    """jobs: [(variant, seed)]. Runs up to args.parallel trainings at once; returns {(variant, seed): ckpt}."""
    ckpts, pending, running = {}, [], []
    for variant, seed in jobs:
        run_dir = osp.join(root, 'runs', f'{variant}_s{seed}')
        done = osp.join(run_dir, 'DONE.json')
        if osp.exists(done):
            ckpts[(variant, seed)] = json.load(open(done))['ckpt']
        else:
            pending.append((variant, seed, run_dir, done))
    while pending or running:
        while pending and len(running) < args.parallel:
            variant, seed, run_dir, done = pending.pop(0)
            os.makedirs(run_dir, exist_ok=True)
            extra = ['--ckpt_path', last_ckpt(run_dir)] if last_ckpt(run_dir) else []
            log(f'Training {variant} seed {seed} ({args.epochs} epochs){" - resuming" if extra else ""} -> {run_dir}')
            log_path = osp.join(run_dir, 'train.log') if args.parallel > 1 else None
            proc = run(train_cmd(variant, seed, run_dir, data_dir, args, args.epochs, extra), log_path)
            if proc is None:  # sequential: already finished
                finish(variant, seed, run_dir, done, args, ckpts)
            else:
                running.append((proc, variant, seed, run_dir, done, time.time()))
        for job in list(running):
            proc, variant, seed, run_dir, done, t0 = job
            if proc.poll() is None:
                continue
            running.remove(job)
            if proc.returncode != 0:
                raise RuntimeError(f'training {variant} seed {seed} failed, see {run_dir}/train.log')
            finish(variant, seed, run_dir, done, args, ckpts)
        if running:
            time.sleep(60)
            log('running: ' + ', '.join(f'{v} s{s} ({(time.time() - t0) / 60:.0f} min)'
                                        for _, v, s, _, _, t0 in running))
    return ckpts


def finish(variant, seed, run_dir, done, args, ckpts):
    ckpt = last_ckpt(run_dir)
    json.dump({'ckpt': ckpt, 'variant': variant, 'seed': seed, 'epochs': args.epochs, **VARIANTS[variant]},
              open(done, 'w'), indent=1)
    ckpts[(variant, seed)] = ckpt
    log(f'Trained {variant} seed {seed} -> {ckpt}')


def bundle(root):
    path = osp.join(root, 'for_claude.zip')
    with zipfile.ZipFile(path + '.part', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(osp.join(root, '**', '*'), recursive=True):
            if osp.isfile(f) and not f.endswith(('.ckpt', '.zip', '.part')):
                z.write(f, osp.relpath(f, root))
    os.replace(path + '.part', path)
    log(f'Bundle for analysis -> {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='parent folder (Google Drive); outputs go to <results>/m2')
    p.add_argument('--data_root', default='/content/data')
    p.add_argument('--runs', nargs='+', default=['M2:80085'],
                   help='variant:seed pairs. 80085 is the seed attempt 1 used for its first baseline run')
    p.add_argument('--parallel', type=int, default=1, help='trainings at once on one GPU (untested above 1)')
    p.add_argument('--epochs', type=int, default=30, help='30 as TrackTacular and the attempt-1 baselines')
    p.add_argument('--scene_p', type=float, default=0.5, help='scene augmentation probability (MVAug Tab. 4)')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--precision', default='32-true', help='as the attempt-1 baselines')
    p.add_argument('--train_args', default='', help='extra arguments for `world_track.py fit`')
    p.add_argument('--skip_preflight', action='store_true')
    args = p.parse_args()

    jobs = []
    for r in args.runs:
        variant, seed = r.split(':')
        assert variant in VARIANTS, f'unknown variant {variant}, expected one of {list(VARIANTS)}'
        jobs.append((variant, int(seed)))
    root = osp.join(args.results, 'm2')
    data_dir = osp.join(args.data_root, 'Wildtrack')
    os.makedirs(root, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=WORLDTRACK, capture_output=True, text=True).stdout.strip()
    json.dump(dict(vars(args), git_commit=commit, time=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(osp.join(root, f'env_{time.strftime("%Y%m%d_%H%M%S")}.json'), 'w'), indent=1)
    log(f'Research methodology 2: {len(jobs)} training(s) {args.runs}, parallel={args.parallel}, commit {commit[:7]}')

    if not args.skip_preflight:
        preflight(sorted({v for v, _ in jobs}), root, data_dir, args)
    ckpts = train_all(jobs, root, data_dir, args)
    for (variant, seed), ckpt in ckpts.items():
        tag = f'{variant}_test_s{seed}'
        out = osp.join(root, 'eval', tag)
        if not osp.exists(osp.join(out, 'metrics.csv')):
            log(f'Evaluating {tag}')
            run(eval_cmd(ckpt, out, tag, variant, seed, data_dir))
    subprocess.run([sys.executable, osp.join(HERE, 'm2_report.py'), '--results', root], check=True)
    bundle(root)


if __name__ == '__main__':
    main()
