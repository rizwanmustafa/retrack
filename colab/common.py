"""
Shared Colab helpers: shell commands with live output, Google Drive, datasets, Python packages.

Datasets are restored into /content/data (fast local disk) from tar files on Drive:
    MyDrive/datasets/wildtrack.tar    top-level folder Wildtrack/   (downloaded from Kaggle and saved there if missing)
    MyDrive/datasets/multiviewx.tar   top-level folder MultiviewX/  (only on OneDrive: https://github.com/hou-yz/MultiviewX)
"""
import os
import os.path as osp
import shutil
import subprocess

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORLDTRACK = osp.join(REPO, 'WorldTrack')
DATA = '/content/data'
DRIVE = '/content/drive/MyDrive'
TARS = {'wildtrack': f'{DRIVE}/datasets/wildtrack.tar', 'multiviewx': f'{DRIVE}/datasets/multiviewx.tar'}
FOLDERS = {'wildtrack': 'Wildtrack', 'multiviewx': 'MultiviewX'}
KAGGLE_WILDTRACK = 'aryashah2k/large-scale-multicamera-detection-dataset'
NEEDED = ('Image_subsets', 'annotations_positions', 'calibrations')


def sh(cmd, cwd=None):
    """Runs a shell command and streams its output into the notebook (Colab does not show the output of
    subprocess.run); raises on failure."""
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive():
    if in_colab() and not osp.isdir(DRIVE):
        from google.colab import drive
        drive.mount('/content/drive')


def pip_install():
    """Packages Colab does not ship (it already has torch, torchvision, numpy, scipy, pandas, matplotlib, opencv)."""
    try:
        import kornia, lightning, motmetrics, lap, timm, efficientnet_pytorch  # noqa: F401,E401
    except ImportError:
        sh(f'pip install -q -r {REPO}/requirements-colab.txt')


def dataset_ready(folder):
    return all(osp.isdir(osp.join(folder, d)) for d in NEEDED)


def ensure_dataset(name, data_root=DATA, tar=None):
    """Path of the dataset folder, restoring it from Drive (or Kaggle, for WILDTRACK) when it is not there yet."""
    folder = osp.join(data_root, FOLDERS[name])
    if dataset_ready(folder):
        return folder
    os.makedirs(data_root, exist_ok=True)
    tar = tar or TARS[name]
    if osp.exists(tar):
        print(f'restoring {FOLDERS[name]} from {tar} ...', flush=True)
        sh(f'tar -xf "{tar}" -C "{data_root}"')
    elif name == 'wildtrack':
        print('downloading WILDTRACK from Kaggle ...', flush=True)
        prepare_wildtrack(data_root)
        if osp.isdir(osp.dirname(tar)) or osp.isdir(DRIVE):
            os.makedirs(osp.dirname(tar), exist_ok=True)
            sh(f'cd "{data_root}" && tar -chf "{tar}.part" Wildtrack && mv "{tar}.part" "{tar}"')
    else:
        raise FileNotFoundError(
            f'{tar} not found. MultiviewX is only on OneDrive (https://github.com/hou-yz/MultiviewX): make '
            'multiviewx.tar with a top-level MultiviewX/ folder and upload it to MyDrive/datasets/')
    assert dataset_ready(folder), f'{folder} must contain {", ".join(NEEDED)}'
    return folder


def prepare_wildtrack(data_root):
    import kagglehub
    dst = osp.join(data_root, 'Wildtrack')
    path = kagglehub.dataset_download(KAGGLE_WILDTRACK)
    found = {n: [] for n in NEEDED}
    for dirpath, dirnames, _ in os.walk(path):
        for d in dirnames:
            if d in found:
                found[d].append(osp.join(dirpath, d))
    for n, v in found.items():
        assert v, f'"{n}" not found in the Kaggle download at {path}'
    os.makedirs(dst, exist_ok=True)
    for n in ('annotations_positions', 'calibrations'):
        if not osp.exists(osp.join(dst, n)):
            shutil.copytree(found[n][0], osp.join(dst, n))
    if not osp.exists(osp.join(dst, 'Image_subsets')):
        os.symlink(found['Image_subsets'][0], osp.join(dst, 'Image_subsets'))
    return dst
