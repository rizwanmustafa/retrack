"""
Colab cell 1: train and evaluate B0 (TrackTacular) and R2, then write REPORT.md.

    !rm -rf /content/retrack && git clone -q -b feat/r2-sparse-queries https://github.com/rizwanmustafa/retrack.git /content/retrack
    %run -i /content/retrack/colab/train_eval.py

Settings: define any of these in the cell before the %run line to change them.
    RESULTS   output folder on Drive                      default /content/drive/MyDrive/retrack_results_r2
    DATASETS  'wildtrack multiviewx'
    VARIANTS  'B0 R2'
    SEEDS     '80085 1 2'                                  3 seeds, so differences can be told from noise
    REUSE     earlier results folder with finished runs     default: the methodology 5 folder, if it exists
    PARALLEL  trainings at once                           2 for a 40+ GB GPU (A100, G4/RTX PRO 6000), else 1
    EXTRA     extra pipeline.py arguments, e.g. '--max_drop 2'
Running it again continues where it stopped (finished trainings and evaluations are skipped).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '/content/retrack/colab')
from common import REPO, DATA, sh, mount_drive, pip_install, ensure_dataset  # noqa: E402

g = globals()
RESULTS = g.get('RESULTS', '/content/drive/MyDrive/retrack_results_r2')
DATASETS = g.get('DATASETS', 'wildtrack multiviewx')
VARIANTS = g.get('VARIANTS', 'B0 R2')
SEEDS = g.get('SEEDS', '80085 1 2')
REUSE = g.get('REUSE', '/content/drive/MyDrive/retrack_results_m5/m5')
PARALLEL = g.get('PARALLEL', None)
EXTRA = g.get('EXTRA', '')

mount_drive()
sh(f'git -C {REPO} log --oneline -1')
for name in DATASETS.split():
    ensure_dataset(name)
pip_install()

if PARALLEL is None:
    import torch
    mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else 0
    PARALLEL = 2 if mem >= 38 else 1
    print(f'GPU memory {mem:.0f} GiB -> PARALLEL = {PARALLEL}')
sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
os.makedirs(RESULTS, exist_ok=True)
reuse = f'--reuse {REUSE}' if REUSE and os.path.isdir(REUSE) else ''
sh(f'cd {REPO} && PYTHONUNBUFFERED=1 python colab/pipeline.py --results {RESULTS} --data_root {DATA} '
   f'--datasets {DATASETS} --variants {VARIANTS} --seeds {SEEDS} --parallel {PARALLEL} {reuse} {EXTRA} '
   f'2>&1 | tee -a {RESULTS}/pipeline.log; exit ${{PIPESTATUS[0]}}')

from IPython.display import Markdown, display  # noqa: E402

from IPython.display import Image  # noqa: E402

display(Markdown('\n'.join(l for l in open(f'{RESULTS}/REPORT.md').read().splitlines() if not l.startswith('!['))))
for f in sorted(os.listdir(f'{RESULTS}/figures')):
    display(Image(f'{RESULTS}/figures/{f}'))
print(f'Share {RESULTS}/for_claude.zip for the analysis.')
