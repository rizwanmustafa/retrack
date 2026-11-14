"""
Colab: miss diagnosis on the existing B0 checkpoints (evaluation only, no training; about 10 minutes on any GPU).
Run from a notebook cell (see RUN.md):

    %run -i /content/retrack/research-methodology-5/m5_diagnose_colab.py

Writes {RESULTS}/m5/diagnose/<dataset>_<variant>_s<seed>/ (sweep.csv, people.csv, summary.txt) and
{RESULTS}/m5/diagnose.zip to share.
"""
import glob
import os
import subprocess

g = globals()
RESULTS = g.get('RESULTS', '/content/drive/MyDrive/retrack_results_m5')
WILDTRACK_TAR = g.get('WILDTRACK_TAR', '/content/drive/MyDrive/datasets/wildtrack.tar')
MULTIVIEWX_TAR = g.get('MULTIVIEWX_TAR', '/content/drive/MyDrive/datasets/multiviewx.tar')
RUNS = g.get('RUNS', 'wildtrack_B0_s80085 multiviewx_B0_s80085')  # folders under {RESULTS}/m5/runs
BRANCH = g.get('BRANCH', 'feat/research-methodology-5')
REPO_DIR = '/content/retrack'
DATA = '/content/data'
OUT = f'{RESULTS}/m5/diagnose'
FOLDER = {'wildtrack': 'Wildtrack', 'multiviewx': 'MultiviewX'}
TAR = {'wildtrack': WILDTRACK_TAR, 'multiviewx': MULTIVIEWX_TAR}


def sh(cmd):
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive  # noqa: E402

drive.mount('/content/drive')
if not os.path.exists(REPO_DIR):
    sh(f'git clone -q -b {BRANCH} https://github.com/rizwanmustafa/retrack.git {REPO_DIR}')
sh(f'git -C {REPO_DIR} log --oneline -1')
sh(f'pip install -q -r {REPO_DIR}/requirements-colab.txt')
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

for run in RUNS.split():
    dataset = run.split('_')[0]
    folder = f'{DATA}/{FOLDER[dataset]}'
    if not os.path.isdir(f'{folder}/annotations_positions'):
        sh(f'tar -xf {TAR[dataset]} -C {DATA}')
    ckpts = sorted(glob.glob(f'{RESULTS}/m5/runs/{run}/lightning_logs/version_*/checkpoints/last.ckpt'))
    assert ckpts, f'no checkpoint for {run}'
    sh(f'cd {REPO_DIR} && PYTHONUNBUFFERED=1 python research-methodology-5/code/m5_diagnose.py '
       f'--ckpt {ckpts[-1]} --data_dir {folder} --out {OUT}/{run}')

sh(f'cd {RESULTS}/m5 && rm -f diagnose.zip && zip -qr diagnose.zip diagnose && ls -la diagnose.zip')
print(f'Share {RESULTS}/m5/diagnose.zip (or paste the summary tables printed above).')
