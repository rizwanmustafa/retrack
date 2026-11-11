"""
Research methodology 2: train, evaluate and report in about one hour on an A100.

    python research-methodology-2/code/m2_pipeline.py --results /content/drive/MyDrive/retrack_results_m2

Steps (each one is skipped when its output already exists, so re-running continues where it stopped):
  1. preflight: every requested variant trains for a few batches and is evaluated once (catches crashes in
     about 3 minutes instead of after a 35-minute run)
  2. training: 30 epochs, same schedule as the attempt-1 baselines (configs/t_retrack.yml)
  3. evaluation: evaluation/arl_eval.py on the 40 test frames (cached detections + all post-hoc cells)
  4. robustness (methodology 3): evaluation/robust_eval.py on every trained checkpoint and on attempt 1's base and A2
     checkpoints (clean, mirror test-time augmentation, calibration noise, camera failures)
  5. report: m2_report.py -> <results>/m2/REPORT.md, and <results>/m2/for_claude.zip (everything but checkpoints)

Variants (see METHODOLOGY.md):
  M2   scene augmentation + VGF + ray conditioning + view-diversity gate prior   (the proposed method)
  M2c  same with GRACE's coverage prior instead of ours                          (is our prior worth it?)
  A2f  scene augmentation alone, with the batch-order fix                        (is VGF worth it?)
  VD   VGF + rays + view-diversity prior, no scene augmentation                  (is scene augmentation worth it?)
  M2cx M2c trained with view mixing (attempt 1's batch order, on purpose)        (does the "bug" add to VGF?)
  A2x  SA-VM: scene augmentation + view mixing of cameras 1, 3, 5 (= attempt-1 A2, on purpose)
  RVM  scene augmentation + random view mixing (each camera with probability 3/7)  (methodology 3)
  CD   scene augmentation + camera dropout instead of view mixing                 (control: is mixing just dropout?)
Research methodology 4 (research-methodology-4/METHODOLOGY.md, needs --fg_background):
  FGx  SA-VM (A2x) + foreground-gated lifting
  FG0  foreground-gated lifting alone, no scene augmentation, no mixing
  E0c  retrained TrackTacular in the current code (no scene augmentation, no mixing)
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

_V = dict(scene=True, vgf=False, ray=False, prior='diversity', mix=False, mixr=0.0, cdrop=0.0, fg=False)
VARIANTS = {
    'M2': dict(_V, vgf=True, ray=True),
    'M2c': dict(_V, vgf=True, ray=True, prior='coverage'),
    'A2f': dict(_V),
    'VD': dict(_V, scene=False, vgf=True, ray=True),
    'M2cx': dict(_V, vgf=True, ray=True, prior='coverage', mix=True),
    'A2x': dict(_V, mix=True),
    'RVM': dict(_V, mixr=round(3 / 7, 4)),
    # dropout matched to view mixing: 2/3 of the samples lose 1-3 cameras, as 2/3 of the batches misplace 3 cameras
    'CD': dict(_V, cdrop=0.67),
    'FGx': dict(_V, mix=True, fg=True),
    'FG0': dict(_V, scene=False, fg=True),
    'E0c': dict(_V, scene=False),
}
ATTEMPT1 = {'base': 'E0', 'A2': 'A2'}  # attempt-1 runs evaluated for robustness: folder variant -> report variant


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
            '--model.view_mix_random', str(v['mixr']),
            '--model.cam_dropout_prob', str(v['cdrop']),
            '--model.cam_dropout_max', '3',
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
            '--seed_everything', str(seed)] + (['--model.fg_background', args.fg_background] if v['fg'] else []) \
        + list(extra) + shlex.split(args.train_args)


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
        if not args.skip_robust and not osp.exists(osp.join(root, 'preflight', 'robust_quick')):
            spec = osp.join(out, 'robust_spec.json')
            json.dump([dict(tag=f'preflight_{variant}', variant=variant, seed=0,
                            ckpt=last_ckpt(osp.join(out, 'run')))], open(spec, 'w'))
            run([sys.executable, '-m', 'evaluation.robust_eval', '--spec', spec, '--data_dir', data_dir,
                 '--out_dir', osp.join(root, 'preflight', 'robust_quick'), '--quick'])
        open(osp.join(out, 'PASSED'), 'w').close()
    log('Preflight passed')


def robust_jobs(root, ckpts, attempt1):
    """Every checkpoint trained in <root>/runs (this session or earlier) plus attempt 1's base and A2."""
    jobs = {}
    for done in glob.glob(osp.join(root, 'runs', '*', 'DONE.json')):
        d = json.load(open(done))
        jobs[f"{d['variant']}_s{d['seed']}"] = dict(variant=d['variant'], seed=d['seed'], ckpt=d['ckpt'])
    for (variant, seed), ckpt in ckpts.items():
        jobs[f'{variant}_s{seed}'] = dict(variant=variant, seed=seed, ckpt=ckpt)
    for folder, variant in ATTEMPT1.items():
        for ckpt in glob.glob(osp.join(attempt1, 'runs', f'{folder}_standard_s*', 'lightning_logs', '*',
                                       'checkpoints', 'last.ckpt')):
            seed = int(ckpt.split(f'{folder}_standard_s')[1].split(os.sep)[0])
            jobs[f'attempt1_{variant}_s{seed}'] = dict(variant=f'attempt1_{variant}', seed=seed, ckpt=ckpt)
    return [dict(tag=t, **j) for t, j in sorted(jobs.items()) if j['ckpt'] and osp.exists(j['ckpt'])]


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
    p.add_argument('--results', required=True, help='parent folder (Google Drive); outputs go to <results>/<subdir>')
    p.add_argument('--subdir', default='m2', help='m4 for research methodology 4')
    p.add_argument('--fg_background', default='', help='fg_background.npz (research-methodology-4), for FG* variants')
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
    p.add_argument('--skip_robust', action='store_true', help='skip the methodology-3 robustness evaluation')
    p.add_argument('--attempt1', default='/content/drive/MyDrive/retrack_results_8nov/arl_full',
                   help="attempt 1's results folder; its base and A2 checkpoints get the robustness evaluation")
    args = p.parse_args()

    jobs = []
    for r in args.runs:
        variant, seed = r.split(':')
        assert variant in VARIANTS, f'unknown variant {variant}, expected one of {list(VARIANTS)}'
        jobs.append((variant, int(seed)))
        assert not VARIANTS[variant]['fg'] or osp.exists(args.fg_background), \
            f'{variant} needs --fg_background (research-methodology-4/code/fg_background.py)'
    root = osp.join(args.results, args.subdir)
    data_dir = osp.join(args.data_root, 'Wildtrack')
    os.makedirs(root, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=WORLDTRACK, capture_output=True, text=True).stdout.strip()
    json.dump(dict(vars(args), git_commit=commit, time=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(osp.join(root, f'env_{time.strftime("%Y%m%d_%H%M%S")}.json'), 'w'), indent=1)
    log(f'Research methodology pipeline ({args.subdir}): {len(jobs)} training(s) {args.runs}, parallel={args.parallel}, commit {commit[:7]}')

    if not args.skip_preflight:
        preflight(sorted({v for v, _ in jobs}), root, data_dir, args)
    ckpts = train_all(jobs, root, data_dir, args)
    for (variant, seed), ckpt in ckpts.items():
        tag = f'{variant}_test_s{seed}'
        out = osp.join(root, 'eval', tag)
        if not osp.exists(osp.join(out, 'metrics.csv')):
            log(f'Evaluating {tag}')
            run(eval_cmd(ckpt, out, tag, variant, seed, data_dir))
    if not args.skip_robust:
        jobs = robust_jobs(root, ckpts, args.attempt1)
        log(f'Robustness evaluation of {len(jobs)} checkpoint(s) '
            f'({sum(j["variant"].startswith("attempt1") for j in jobs)} from attempt 1, {args.attempt1})')
        spec = osp.join(root, 'robust_spec.json')
        json.dump(jobs, open(spec, 'w'), indent=1)
        run([sys.executable, '-m', 'evaluation.robust_eval', '--spec', spec, '--data_dir', data_dir,
             '--out_dir', osp.join(root, 'robust')])
    subprocess.run([sys.executable, osp.join(HERE, 'm2_report.py'), '--results', root], check=True)
    bundle(root)


if __name__ == '__main__':
    main()
