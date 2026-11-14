# Research methodology 5 on Colab: R1 / R2 / R4, all 8 combinations, WILDTRACK and MultiviewX.
# Runtime > Change runtime type > A100 GPU. Run through `%run -i` (see RUN.md); any setting below can be set before.
# Outputs (checkpoints, logs, REPORT.md, for_claude.zip) go to Google Drive; re-running continues where it stopped.
import os
import subprocess

g = globals()
RESULTS = g.get('RESULTS', '/content/drive/MyDrive/retrack_results_m5')  # outputs go to {RESULTS}/m5
WILDTRACK_TAR = g.get('WILDTRACK_TAR', '/content/drive/MyDrive/datasets/wildtrack.tar')
MULTIVIEWX_TAR = g.get('MULTIVIEWX_TAR', '/content/drive/MyDrive/datasets/multiviewx.tar')
DATASETS = g.get('DATASETS', 'wildtrack multiviewx')
VARIANTS = g.get('VARIANTS', 'B0 R124 R1 R2 R4 R12 R14 R24')
SEEDS = g.get('SEEDS', '80085')
PARALLEL = g.get('PARALLEL', 2)
EXTRA = g.get('EXTRA', '')  # extra m5_pipeline.py arguments
BRANCH = g.get('BRANCH', 'feat/research-methodology-5')
REPO_DIR = '/content/retrack'
DA3_DIR = '/content/da3'
DA3_COMMIT = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'  # as models/da3_branch.py
DATA = '/content/data'
OUT = f'{RESULTS}/m5'


def sh(cmd):
    """Runs a shell command and streams its output into the notebook (Colab does not show the output of
    subprocess.run), raising on failure."""
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive  # noqa: E402

drive.mount('/content/drive')

# 1. code
if not os.path.exists(REPO_DIR):
    sh(f'git clone -q -b {BRANCH} https://github.com/rizwanmustafa/retrack.git {REPO_DIR}')
sh(f'git -C {REPO_DIR} log --oneline -1')


# 2. datasets: this session -> Drive copy (-> Kaggle for WILDTRACK only; MultiviewX has no automatic source)
def ready(folder):
    return all(os.path.isdir(f'{DATA}/{folder}/{d}') for d in ('Image_subsets', 'annotations_positions', 'calibrations'))


os.makedirs(DATA, exist_ok=True)
if 'wildtrack' in DATASETS:
    if not ready('Wildtrack') and WILDTRACK_TAR and os.path.exists(WILDTRACK_TAR):
        print('restoring WILDTRACK from Drive ...')
        sh(f'tar -xf {WILDTRACK_TAR} -C {DATA}')
    if not ready('Wildtrack'):
        print('downloading WILDTRACK from Kaggle ...')
        sh('pip -q install kagglehub')
        sh(f'cd {REPO_DIR}/WorldTrack && python -c "from retrack_pipeline import prepare_wildtrack; '
           f'prepare_wildtrack(\'{DATA}\')"')
        if WILDTRACK_TAR:
            os.makedirs(os.path.dirname(WILDTRACK_TAR), exist_ok=True)
            sh(f'cd {DATA} && tar -chf {WILDTRACK_TAR}.part Wildtrack && mv {WILDTRACK_TAR}.part {WILDTRACK_TAR}')
    assert ready('Wildtrack'), 'WILDTRACK is not ready'
if 'multiviewx' in DATASETS:
    if not ready('MultiviewX'):
        assert os.path.exists(MULTIVIEWX_TAR), (
            f'{MULTIVIEWX_TAR} not found. MultiviewX is only on OneDrive (https://github.com/hou-yz/MultiviewX): '
            'make multiviewx.tar with a top-level MultiviewX/ folder and upload it to MyDrive/datasets/')
        print('restoring MultiviewX from Drive ...')
        sh(f'tar -xf {MULTIVIEWX_TAR} -C {DATA}')
    assert ready('MultiviewX'), f'MultiviewX is not ready: {MULTIVIEWX_TAR} must contain MultiviewX/Image_subsets ...'
    n_img = sum(len(fs) for _, _, fs in os.walk(f'{DATA}/MultiviewX/Image_subsets'))
    print(f'MultiviewX: {n_img} images')

# 3. dependencies: the repo's, plus the Depth Anything 3 backbone code (pinned commit, used from source: its own
#    package pins numpy<2 and pulls open3d/xformers, none of which the backbone needs)
sh(f'pip install -q -r {REPO_DIR}/requirements-colab.txt omegaconf addict einops safetensors huggingface_hub')
if not os.path.exists(f'{DA3_DIR}/.git'):
    sh(f'git clone -q https://github.com/ByteDance-Seed/Depth-Anything-3.git {DA3_DIR}')
sh(f'git -C {DA3_DIR} fetch -q --depth 1 origin {DA3_COMMIT} || true; git -C {DA3_DIR} checkout -q {DA3_COMMIT}')
os.environ['DA3_SRC'] = f'{DA3_DIR}/src'
os.environ['PYTHONPATH'] = f'{DA3_DIR}/src' + (':' + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
os.environ['DA3_LOG_LEVEL'] = 'WARN'
# download the DA3 weights once, before parallel trainings start
sh('python -c "from huggingface_hub import hf_hub_download as d; '
   '[d(\'depth-anything/DA3-BASE\', f) for f in (\'config.json\', \'model.safetensors\')]"')
sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
os.makedirs(OUT, exist_ok=True)

# 4. preflight, training, evaluation, report
sh(f'cd {REPO_DIR} && PYTHONUNBUFFERED=1 python research-methodology-5/code/m5_pipeline.py --results {RESULTS} '
   f'--data_root {DATA} --datasets {DATASETS} --variants {VARIANTS} --seeds {SEEDS} --parallel {PARALLEL} {EXTRA} '
   f'2>&1 | tee -a {OUT}/pipeline.log; exit ${{PIPESTATUS[0]}}')

from IPython.display import Markdown, display  # noqa: E402

display(Markdown(open(f'{OUT}/REPORT.md').read()))
print(f'Share {OUT}/for_claude.zip for the analysis.')
