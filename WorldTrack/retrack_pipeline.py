"""
Retrack end-to-end pipeline: TrackTacular baseline vs Retrack on WILDTRACK.

  1. prepares WILDTRACK (Kaggle mirror) under --data_root
  2. trains the TrackTacular baseline (concatenation fusion, first seed) and checks that the Retrack evaluation
     engine reproduces TrackTacular's own `test` numbers on it. The released WILDTRACK checkpoints do not work
     with the public code (TrackTacular issues #2 and #7: MODA ~0), so the baseline is retrained.
  3. evaluates Stages 1-3 (threshold, ByteTrack-style association, re-ID) on the trained baseline without
     retraining, with all cameras and with cameras removed, including the parameter grids
  4. trains the other Stage 4 fusion variants x seeds with the baseline schedule and evaluates each one
  5. writes the comparison report (results/report/REPORT.md + CSVs + plots)

Everything is written under --results (e.g. Google Drive) and every step is skipped if its output already
exists, so after a disconnect the same command resumes where it stopped (interrupted training resumes from
last.ckpt).

    python retrack_pipeline.py --results /content/drive/MyDrive/retrack_results               # full study
    python retrack_pipeline.py --results /content/drive/MyDrive/retrack_results --mode quick  # ~1 h smoke run
Outputs go to <results>/<mode>/.
"""
import argparse
import glob
import json
import os
import os.path as osp
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

HERE = osp.dirname(osp.abspath(__file__))
VARIANTS = ['concat', 'concat_drop', 'masked_drop', 'masked_cov_drop', 'weighted_cov_drop']
# Approach 1 (calibration-noise training), run on request: --variants concat_calib masked_calib
EXTRA_VARIANTS = ['concat_calib', 'masked_calib']
SEEDS = [80085, 1, 2, 3, 4]  # 80085 is the repo's / official checkpoint's seed
ZOO_URL = 'https://github.com/tteepe/TrackTacular/releases/download/v1/wild_{model}.zip'
KAGGLE_WILDTRACK = 'aryashah2k/large-scale-multicamera-detection-dataset'


def log(msg):
    print(f'\n[{time.strftime("%H:%M:%S")}] ===== {msg}', flush=True)


def run(cmd, **kw):
    print('$ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE, **kw)


# ------------------------------------------------------------------ data
def prepare_wildtrack(data_root):
    dst = osp.join(data_root, 'Wildtrack')
    needed = ['Image_subsets', 'annotations_positions', 'calibrations']
    if all(osp.exists(osp.join(dst, n)) for n in needed):
        print(f'WILDTRACK ready at {dst}')
        return dst
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_WILDTRACK)
    found = {n: [] for n in needed}
    for dirpath, dirnames, _ in os.walk(path):
        for d in dirnames:
            if d in found:
                found[d].append(osp.join(dirpath, d))
    for n, v in found.items():
        assert v, f'"{n}" not found in the Kaggle download at {path}'
    os.makedirs(dst, exist_ok=True)
    for n in ['annotations_positions', 'calibrations']:
        if not osp.exists(osp.join(dst, n)):
            shutil.copytree(found[n][0], osp.join(dst, n))
    if not osp.exists(osp.join(dst, 'Image_subsets')):
        os.symlink(found['Image_subsets'][0], osp.join(dst, 'Image_subsets'))
    cams = sorted(os.listdir(osp.join(dst, 'Image_subsets')))
    n_ann = len(os.listdir(osp.join(dst, 'annotations_positions')))
    print(f'WILDTRACK: cameras {cams}, {n_ann} annotated frames -> {dst}')
    assert len(cams) >= 7 and n_ann == 400, 'unexpected WILDTRACK layout'
    return dst


def official_checkpoint(model, results):
    out = osp.join(results, 'model_zoo', f'wild_{model}')
    ckpts = glob.glob(osp.join(out, '**', '*.ckpt'), recursive=True)
    if not ckpts:
        os.makedirs(out, exist_ok=True)
        zip_path = out + '.zip'
        print(f'downloading official TrackTacular wild_{model} checkpoint')
        urllib.request.urlretrieve(ZOO_URL.format(model=model), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
        os.remove(zip_path)
        ckpts = glob.glob(osp.join(out, '**', '*.ckpt'), recursive=True)
    return sorted(ckpts)[-1]


# ------------------------------------------------------------------ steps
def sanity_check(ckpt, data_dir, results, tag):
    path = osp.join(results, 'sanity_lightning_test.json')
    if osp.exists(path) and json.load(open(path)).get('tag') == tag:
        return
    log(f'Sanity check: TrackTacular\'s own Lightning test on {tag}')
    code = ('import json, sys, torch; torch.set_float32_matmul_precision("medium"); '
            'from evaluation.retrack_eval import lightning_test; '
            f'res = lightning_test({ckpt!r}, {data_dir!r}, {osp.join(results, "logs")!r}); '
            f'res["tag"] = {tag!r}; '
            f'json.dump(res, open({path!r}, "w"), indent=1); print(res)')
    run([sys.executable, '-c', code])


def evaluate(ckpt, data_dir, results, tag, variant, seed, args, grids=False):
    out = osp.join(results, 'eval', tag)
    if osp.exists(osp.join(out, 'metrics.csv')):
        return
    log(f'Evaluating {tag}')
    cmd = [sys.executable, '-m', 'evaluation.retrack_eval', '--ckpt', ckpt, '--data_dir', data_dir, '--out', out,
           '--tag', tag, '--variant', variant, '--seed', str(seed), '--max_drop', str(args.max_drop)]
    if args.max_combos:
        cmd += ['--max_combos', str(args.max_combos)]
    if grids:
        cmd += ['--grids']
    run(cmd)
    report(results)


def train(variant, seed, data_dir, results, args):
    run_dir = osp.join(results, 'runs', f'{args.model}_{variant}_s{seed}')
    done = osp.join(run_dir, 'DONE.json')
    if osp.exists(done):
        return json.load(open(done))['ckpt']
    os.makedirs(run_dir, exist_ok=True)
    cmd = [sys.executable, 'world_track.py', 'fit',
           '-c', 'configs/t_retrack.yml', '-c', 'configs/d_wildtrack.yml', '-c', f'configs/m_{args.model}.yml',
           '-c', f'configs/v_{variant}.yml',
           '--data.init_args.data_dir', data_dir,
           '--data.init_args.batch_size', str(args.batch_size),
           '--data.init_args.num_workers', str(args.num_workers),
           '--trainer.accumulate_grad_batches', str(args.accumulate),
           '--trainer.max_epochs', str(args.epochs),
           '--trainer.precision', args.precision,
           '--trainer.default_root_dir', run_dir,
           '--seed_everything', str(seed)] + shlex.split(args.train_args)
    if not args.progress_bar:
        # the live progress bar floods notebook output; one line per epoch is printed instead
        cmd += ['--trainer.enable_progress_bar', 'false']
    previous = sorted(glob.glob(osp.join(run_dir, 'lightning_logs', '*', 'checkpoints', 'last.ckpt')),
                      key=osp.getmtime)
    if previous:
        print(f'resuming from {previous[-1]}')
        cmd += ['--ckpt_path', previous[-1]]
    log(f'Training {variant} (seed {seed}, {args.epochs} epochs) -> {run_dir}')
    t0 = time.time()
    run(cmd)
    ckpt = sorted(glob.glob(osp.join(run_dir, 'lightning_logs', '*', 'checkpoints', 'last.ckpt')),
                  key=osp.getmtime)[-1]
    json.dump({'ckpt': ckpt, 'variant': variant, 'seed': seed, 'epochs': args.epochs,
               'train_minutes_last_session': round((time.time() - t0) / 60, 1)}, open(done, 'w'), indent=1)
    return ckpt


def report(results):
    subprocess.run([sys.executable, '-m', 'evaluation.retrack_report', '--results', results], cwd=HERE,
                   stdout=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True, help='output folder (put it on Google Drive to survive disconnects)')
    p.add_argument('--data_root', default='/content/data')
    p.add_argument('--mode', choices=['full', 'quick'], default='full')
    p.add_argument('--model', default='segnet', choices=['segnet', 'mvdet', 'liftnet'],
                   help='lifting method (segnet = bilinear sampling, the best TrackTacular model on WILDTRACK)')
    p.add_argument('--seeds', type=int, default=None, help='training seeds per fusion variant (full: 3, quick: 1)')
    p.add_argument('--epochs', type=int, default=None, help='full: 30 as in the paper, quick: 2')
    p.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS + EXTRA_VARIANTS)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--precision', default='32-true', help="'32-true' as the original; 'bf16-mixed' is faster")
    p.add_argument('--max_drop', type=int, default=None, help='evaluate up to this many removed cameras')
    p.add_argument('--max_combos', type=int, default=None, help='sample at most this many combinations per k')
    p.add_argument('--official', action='store_true',
                   help='also evaluate the released TrackTacular checkpoint (known not to work, see issue #7)')
    p.add_argument('--train_args', default='', help='extra arguments for `world_track.py fit`')
    p.add_argument('--progress_bar', action='store_true', help='show Lightning\'s live progress bar while training')
    args = p.parse_args()

    quick = args.mode == 'quick'
    args.seeds = args.seeds or (1 if quick else 3)
    args.epochs = args.epochs or (2 if quick else 30)
    args.max_drop = args.max_drop if args.max_drop is not None else (1 if quick else 3)
    seeds = SEEDS[:args.seeds]
    args.results = osp.join(args.results, args.mode)  # quick and full runs never share outputs
    os.makedirs(args.results, exist_ok=True)
    json.dump(vars(args), open(osp.join(args.results, f'pipeline_args_{time.strftime("%Y%m%d_%H%M%S")}.json'), 'w'),
              indent=1)
    print(f'Retrack pipeline: mode={args.mode} model={args.model} variants={args.variants} seeds={seeds} '
          f'epochs={args.epochs} max_drop={args.max_drop} -> {args.results}')

    log('Data')
    data_dir = prepare_wildtrack(args.data_root)

    if args.official:
        log('Released TrackTacular checkpoint (reference only)')
        official = official_checkpoint(args.model, args.results)
        evaluate(official, data_dir, args.results, 'official', 'official', -1, args)

    # The first model trained (the concat baseline, first seed) is the trained TrackTacular model that
    # Stages 1-3 are applied to: it gets the sanity check and the parameter grids.
    # One full seed of every variant runs first, so a complete comparison exists as early as possible.
    variants = ['concat'] + [v for v in args.variants if v != 'concat'] if 'concat' in args.variants \
        else args.variants
    for seed in seeds:
        for variant in variants:
            ckpt = train(variant, seed, data_dir, args.results, args)
            tag = f'{variant}_s{seed}'
            primary = seed == seeds[0] and variant == 'concat'
            if primary:
                sanity_check(ckpt, data_dir, args.results, tag)
            evaluate(ckpt, data_dir, args.results, tag, variant, seed, args, grids=primary)

    log('Report')
    subprocess.run([sys.executable, '-m', 'evaluation.retrack_report', '--results', args.results], cwd=HERE,
                   check=True)


if __name__ == '__main__':
    main()
