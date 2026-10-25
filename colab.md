Claude
/
tracktacular_kaggle_setup.md
tracktacular_kaggle_setup.md
MD
tracktacular_context_handoff.md
tracktacular_context_handoff.md
MD

I am creating a new notebook on Colab. Give me cells for Colab. Final version
Resolving library version conflicts for a stable Colab setup

Below is the Colab version of your Kaggle setup. The Kaggle version was tested end to end, but this Colab version is not, so check the outputs of Cells 5 and 8 before you train.

These are the changes from the Kaggle version:

    Paths: everything moves from /kaggle/working to /content.
    Dataset path: kagglehub downloads to a writable cache folder on Colab. data_dir still points at /content/data, so nothing else changes.
    Setuptools repair: removed, because the openmim problem it fixed only happened on Kaggle.
    Restart: the install cell restarts the session for you.
    OpenCV pin: OpenCV is pinned below 4.12 because Colab’s preinstalled OpenCV needs numpy 2 and would crash once numpy is downgraded to 1.x.
    Saving: Colab deletes /content when the session ends, so there’s an optional cell that saves your training results to Google Drive.

Before you start: go to Runtime → Change runtime type → T4 GPU.
Cell 1 — clone
python

import os
if not os.path.exists('/content/TrackTacular'):
    !git clone https://github.com/tteepe/TrackTacular.git /content/TrackTacular
%cd /content/TrackTacular/WorldTrack
!nvidia-smi --query-gpu=name,memory.total --format=csv

Cell 2 — requirements file

%%writefile must be the first line of its own cell.
python

%%writefile /content/req_colab.txt
kornia
matplotlib
opencv-python<4.12
opencv-python-headless<4.12
lightning[extra]
scipy
einops
nuscenes-devkit
numpy<2.0
lap
motmetrics
Pillow
pytorch_metric_learning
efficientnet_pytorch
timm
imgaug

Cell 3 — install and restart
python

!pip install -r /content/req_colab.txt
# Dependency-conflict warnings about other Colab packages are harmless. Only "Failed to build" matters.

# numpy<2.0 only takes effect after a restart. This line kills the session on purpose;
# Colab will say it crashed/restarted. That's expected — continue from Cell 4.
import os; os.kill(os.getpid(), 9)

Cell 4 — patch the repo (4 fixes, safe to re-run)
python

%cd /content/TrackTacular/WorldTrack

# 1. models/__init__.py — bevformernet is the only file needing mmcv
!sed -i 's/^from models.bevformernet import Bevformernet/# from models.bevformernet import Bevformernet  # needs mmcv/' models/__init__.py

# 2. world_track.py — imports Bevformernet by name
!sed -i 's/^from models import Segnet, MVDet, Liftnet, Bevformernet/from models import Segnet, MVDet, Liftnet  # Bevformernet needs mmcv/' world_track.py

# 3. datasets/__init__.py — imports two datamodules never committed to the repo
!sed -i 's/^from \.v2x_datamodule import V2XDataModule/# &  # module not in repo/' datasets/__init__.py
!sed -i 's/^from \.tum_traffic_datamodule import TUMTrafDataModule/# &  # module not in repo/' datasets/__init__.py

# 4. datasets/sampler.py — torch>=2.2 removed Sampler.__init__(data_source)
!sed -i 's/^        super().__init__(data_source)/        super().__init__()  # torch>=2.2 removed data_source arg/' datasets/sampler.py

!head -4 models/__init__.py
!sed -n '7p' world_track.py
!cat datasets/__init__.py
!sed -n '8p' datasets/sampler.py

Cell 5 — verify environment
python

%cd /content/TrackTacular/WorldTrack
import torch, numpy, cv2
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("numpy", numpy.__version__, "| cv2", cv2.__version__)
import models
print("models OK:", [n for n in dir(models) if not n.startswith('_')])

Expected: cuda True, numpy 1.26.x, cv2 4.11.x, and ['Liftnet', 'MVDet', 'Segnet', ...]. The torch version will be whatever Colab ships, which is fine.
Cell 6 — dataset
python

import kagglehub, os, shutil

# If this fails with 401/403: add KAGGLE_USERNAME and KAGGLE_KEY in Colab Secrets (key icon, left panel), then uncomment:
# from google.colab import userdata
# os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
# os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

path = kagglehub.dataset_download("priankamukhopadhyay/multiviewx")
print("Downloaded to:", path)

# The mirror double-nests the real data and bundles an unrelated EPFL-RLC labels tree
SRC = os.path.join(path, 'MultiviewX', 'MultiviewX')
DST = '/content/data/MultiviewX'
os.makedirs(DST, exist_ok=True)

for name in ['annotations_positions', 'calibrations']:
    dst = os.path.join(DST, name)
    if not os.path.exists(dst):
        shutil.copytree(os.path.join(SRC, name), dst)

img_dst = os.path.join(DST, 'Image_subsets')
if not os.path.exists(img_dst):
    os.symlink(os.path.join(SRC, 'Image_subsets'), img_dst)

print('cams:', sorted(os.listdir(img_dst)))
print('frames:', sorted(os.listdir(os.path.join(img_dst, 'C1')))[:3])
print('annotations:', len(os.listdir(os.path.join(DST, 'annotations_positions'))))

Expected: cams C1–C6, frames 0000.png …, 400 annotations.
Cell 7 — data config
python

%%writefile /content/TrackTacular/WorldTrack/configs/d_multiviewx_colab.yml
data:
  class_path: datasets.PedestrianDataModule
  init_args:
    data_dir: "/content/data/MultiviewX"
    batch_size: 1
    num_workers: 2
model:
  resolution: [160, 2, 250]  # Y,Z,X
  bounds: [0, 1000, 0, 640, 0, 2]  # xmin,xmax,ymin,ymax,zmin,zmax
  depth: [32, 0, -20]
  z_sign: -1
  num_cameras: 6
trainer:
  max_epochs: 50

Cell 8 — smoke test
python

%cd /content/TrackTacular/WorldTrack
from datasets.pedestrian_datamodule import PedestrianDataModule

dm = PedestrianDataModule(
    data_dir='/content/data/MultiviewX',
    batch_size=1, num_workers=0,
    resolution=[160, 2, 250], bounds=[0, 1000, 0, 640, 0, 2],
)
dm.setup('fit')
print('train:', len(dm.data_train), '| val:', len(dm.data_val))
item, target = next(iter(dm.train_dataloader()))
print({k: tuple(v.shape) for k, v in item.items()})

Expected: train: 360 | val: 40.
Cell 9 — train
python

%cd /content/TrackTacular/WorldTrack
!python world_track.py fit \
    -c configs/t_fit.yml \
    -c configs/d_multiviewx_colab.yml \
    -c configs/m_mvdet.yml \
    --trainer.max_epochs 2 \
    --trainer.devices 1 \
    --data.init_args.batch_size 1 \
    --data.init_args.num_workers 2

This runs 2 epochs first to prove everything works. Once it does, change the setting to --trainer.max_epochs 50, which should take about 3 h 45 min on a T4. Free Colab disconnects when idle, so keep the tab active during the long run.

To swap models, replace m_mvdet.yml with m_segnet.yml or m_liftnet.yml.
Cell 10 — test the latest run
python

%cd /content/TrackTacular/WorldTrack
import glob, os
run = sorted(glob.glob('lightning_logs/version_*'), key=os.path.getmtime)[-1]
ckpt = sorted(glob.glob(f'{run}/checkpoints/*.ckpt'), key=os.path.getmtime)[-1]
print('run:', run, '| ckpt:', ckpt)
!python world_track.py test -c {run}/config.yaml --ckpt_path {ckpt}

Cell 11 — save to Google Drive (optional)
python

from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/tracktacular
!cp -r /content/TrackTacular/WorldTrack/lightning_logs /content/drive/MyDrive/tracktacular/
!ls /content/drive/MyDrive/tracktacular/lightning_logs

To resume in a new session, re-run Cells 1–8. Then add --ckpt_path /content/drive/MyDrive/tracktacular/lightning_logs/version_N/checkpoints/last.ckpt to the Cell 9 command.

/content/TrackTacular/WorldTrack
Seed set to 80085
/usr/local/lib/python3.13/dist-packages/jsonargparse/_typehints.py:3065: JsonargparseWarning: 
    Dump of a value that does not round-trip: a tuple is serialized as list and, since the type is not validated,
    parsing it back gives a list. Value: (160, 2, 250)

  warning(
/usr/local/lib/python3.13/dist-packages/jsonargparse/_typehints.py:3065: JsonargparseWarning: 
    Dump of a value that does not round-trip: a tuple is serialized as list and, since the type is not validated,
    parsing it back gives a list. Value: (0, 1000, 0, 640, 0, 2)

  warning(
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100% 44.7M/44.7M [00:00<00:00, 237MB/s]
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-09-26 11:10:53.526094: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-09-26 11:10:53.588417: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.13/dist-packages/lightning/pytorch/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name           ┃ Type      ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ center_loss_fn │ FocalLoss │      0 │ train │     0 │
│ 1 │ model          │ MVDet     │ 11.3 M │ train │     0 │
└───┴────────────────┴───────────┴────────┴───────┴───────┘
Trainable params: 11.3 M                                                        
Non-trainable params: 0                                                         
Total params: 11.3 M                                                            
Total estimated model params size (MB): 45.139                                  
Modules in train mode: 155                                                      
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/tensorboard/compat/__init__.py", line 42, in tf
    from tensorboard.compat import notf  # noqa: F401
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.13/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/lightning/fabric/loggers/tensorboard.py", line 213, in log_metrics
    self.experiment.add_scalar(k, v, step)
    ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning/fabric/loggers/logger.py", line 118, in experiment
    return fn(self)
  File "/usr/local/lib/python3.13/dist-packages/lightning/fabric/loggers/tensorboard.py", line 191, in experiment
    from torch.utils.tensorboard import SummaryWriter
  File "/usr/local/lib/python3.13/dist-packages/torch/utils/tensorboard/__init__.py", line 12, in <module>
    from .writer import FileWriter, SummaryWriter
  File "/usr/local/lib/python3.13/dist-packages/torch/utils/tensorboard/writer.py", line 19, in <module>
    from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt
  File "/usr/local/lib/python3.13/dist-packages/torch/utils/tensorboard/_embedding.py", line 10, in <module>
    _HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")
                              ^^^^^
  File "/usr/local/lib/python3.13/dist-packages/tensorboard/lazy.py", line 65, in __getattr__
    return getattr(load_once(self), attr_name)
                   ~~~~~~~~~^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/tensorboard/lazy.py", line 97, in wrapper
    cache[arg] = f(arg)
                 ~^^^^^
  File "/usr/local/lib/python3.13/dist-packages/tensorboard/lazy.py", line 50, in load_once
    module = load_fn()
  File "/usr/local/lib/python3.13/dist-packages/tensorboard/compat/__init__.py", line 45, in tf
    import tensorflow
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/__init__.py", line 55, in <module>
    from tensorflow._api.v2 import compat
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/_api/v2/compat/__init__.py", line 8, in <module>
    from tensorflow._api.v2.compat import v1
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/_api/v2/compat/v1/__init__.py", line 30, in <module>
    from tensorflow._api.v2.compat.v1 import compat
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/_api/v2/compat/v1/compat/__init__.py", line 8, in <module>
    from tensorflow._api.v2.compat.v1.compat import v1
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/_api/v2/compat/v1/compat/v1/__init__.py", line 47, in <module>
    from tensorflow._api.v2.compat.v1 import lite
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/_api/v2/compat/v1/lite/__init__.py", line 9, in <module>
    from tensorflow._api.v2.compat.v1.lite import experimental
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/_api/v2/compat/v1/lite/experimental/__init__.py", line 8, in <module>
    from tensorflow._api.v2.compat.v1.lite.experimental import authoring
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/_api/v2/compat/v1/lite/experimental/authoring/__init__.py", line 8, in <module>
    from tensorflow.lite.python.authoring.authoring import compatible # line: 263
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/lite/python/authoring/authoring.py", line 42, in <module>
    from tensorflow.lite.python import convert
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/lite/python/convert.py", line 31, in <module>
    from tensorflow.lite.python import util
  File "/usr/local/lib/python3.13/dist-packages/tensorflow/lite/python/util.py", line 53, in <module>
    from jax import jit as _jit
  File "/usr/local/lib/python3.13/dist-packages/jax/__init__.py", line 37, in <module>
    import jax.core as _core
  File "/usr/local/lib/python3.13/dist-packages/jax/core.py", line 18, in <module>
    import jax._src.core as _src_core
  File "/usr/local/lib/python3.13/dist-packages/jax/_src/core.py", line 40, in <module>
    from jax._src import dtypes
  File "/usr/local/lib/python3.13/dist-packages/jax/_src/dtypes.py", line 610, in <module>
    string_dtype = np.dtypes.StringDType()
                   ^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'. Did you mean: 'StrDType'?

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/content/TrackTacular/WorldTrack/world_track.py", line 358, in <module>
    cli = MyLightningCLI(WorldTrackModel)
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/cli.py", line 421, in __init__
    self._run_subcommand(self.subcommand)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/cli.py", line 759, in _run_subcommand
    fn(**fn_kwargs)
    ~~^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        self,
        ^^^^^
    ...<6 lines>...
        weights_only,
        ^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/trainer/call.py", line 49, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/trainer/trainer.py", line 630, in _fit_impl
    self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/trainer/trainer.py", line 1062, in _run
    _log_hyperparams(self)
    ~~~~~~~~~~~~~~~~^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/loggers/utilities.py", line 100, in _log_hyperparams
    logger.log_hyperparams(hparams_initial)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning_utilities/core/rank_zero.py", line 42, in wrapped_fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/lightning/pytorch/loggers/tensorboard.py", line 182, in log_hyperparams
    return super().log_hyperparams(params=params, metrics=metrics, step=step)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning_utilities/core/rank_zero.py", line 42, in wrapped_fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/lightning/fabric/loggers/tensorboard.py", line 251, in log_hyperparams
    self.log_metrics(metrics, step)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lightning_utilities/core/rank_zero.py", line 42, in wrapped_fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/lightning/fabric/loggers/tensorboard.py", line 216, in log_metrics
    raise ValueError(
        f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
    ) from ex
ValueError: 

Using Nvidia a100 on colab pro plus

Diagnosing a TensorBoard import fallback in Colab.
Diagnosing a TensorBoard import fallback in Colab.

Colab’s preinstalled TensorFlow is what breaks, not your setup. When Lightning starts its TensorBoard logger, TensorBoard imports TensorFlow, TensorFlow imports JAX, and JAX needs numpy 2. Your environment correctly has numpy 1.26 (the repo needs it), so JAX crashes on import. TrackTacular doesn’t use TensorFlow or JAX at all. If TensorFlow is removed, TensorBoard falls back to its built-in stand-in and logging works.
Fix — run this once, then re-run Cell 9
python

!pip uninstall -y tensorflow tensorflow-cpu tf-keras jax jaxlib

You don’t need to restart, because training runs as a separate process. Some packages in that list may not be installed; pip will just say it’s skipping those.
For future sessions, use this Cell 3 instead
python

!pip install -r /content/req_colab.txt
# Colab's TensorFlow/JAX need numpy 2 and crash TensorBoard logging under numpy<2. Not used here.
!pip uninstall -y tensorflow tensorflow-cpu tf-keras jax jaxlib

# Restart so numpy<2.0 takes effect — Colab will report a crash; that's expected. Continue from Cell 4.
import os; os.kill(os.getpid(), 9)

Since you’re on an A100

With 40–80 GB of GPU memory and more CPU cores, you can use settings much larger than the T4 defaults. Once the 2-epoch run works, try:

--data.init_args.batch_size 4 \
--data.init_args.num_workers 8

t_fit.yml has accumulate_grad_batches: 8, so batch size 4 means an effective batch of 32 instead of 8. That changes the training dynamics compared with the paper’s setup. To keep the effective batch at 8, also add --trainer.accumulate_grad_batches 2.

Also, your log shows Colab is now on Python 3.13, not 3.12 like Kaggle. The rest of the pipeline got through model setup fine, so that isn’t causing problems.
Drafting a Colab cell to test the checkpoint.
Drafting a Colab cell to test the checkpoint.

These are two cells to add at the end of the notebook. The first downloads the official Wildtrack checkpoints and shows how each was trained. The second tests one of them on your setup.
Cell A — download the official checkpoints and inspect them
python

import os, glob, zipfile, urllib.request, torch, yaml

ZOO  = '/content/model_zoo'
KEYS = ['model_name', 'use_temporal_cache', 'resolution', 'bounds', 'num_cameras', 'encoder_name', 'conf_threshold']
os.makedirs(ZOO, exist_ok=True)

for name in ['wild_mvdet', 'wild_segnet', 'wild_liftnet']:
    zip_path, out_dir = f'{ZOO}/{name}.zip', f'{ZOO}/{name}'
    if not os.path.exists(out_dir):
        print(f'downloading {name}.zip ...')
        urllib.request.urlretrieve(f'https://github.com/tteepe/TrackTacular/releases/download/v1/{name}.zip', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        os.remove(zip_path)

    files = [f for f in glob.glob(f'{out_dir}/**/*', recursive=True) if os.path.isfile(f)]
    print(f'\n=== {name} ===')
    for f in files:
        print(f'  {os.path.getsize(f)/1e6:7.1f} MB  {os.path.relpath(f, out_dir)}')
    for ck in [f for f in files if f.endswith('.ckpt')]:
        hp = torch.load(ck, map_location='cpu', weights_only=False).get('hyper_parameters', {})
        print('  ckpt hparams :', {k: hp.get(k) for k in KEYS})
    for cfg in [f for f in files if f.endswith(('.yaml', '.yml'))]:
        m = (yaml.safe_load(open(cfg)) or {}).get('model', {})
        print(f'  {os.path.basename(cfg):13s}:', {k: m.get(k) for k in KEYS})

Look at use_temporal_cache in the output. It tells you whether the authors trained Wildtrack with the previous-frame input.
Cell B — test an official checkpoint on your setup
python

%cd /content/TrackTacular/WorldTrack
import glob, os

MODEL = 'segnet'   # 'mvdet', 'segnet' or 'liftnet'

out_dir = f'/content/model_zoo/wild_{MODEL}'
ckpt = sorted(glob.glob(f'{out_dir}/**/*.ckpt', recursive=True))[-1]
cfgs = sorted(glob.glob(f'{out_dir}/**/config.yaml', recursive=True))
print('ckpt  :', ckpt)

if cfgs:
    # Authors' own run config; only data_dir is replaced (theirs points at their machine)
    print('config:', cfgs[-1])
    !python world_track.py test -c {cfgs[-1]} \
        --data.init_args.data_dir /content/data/Wildtrack \
        --trainer.devices 1 \
        --ckpt_path {ckpt}
else:
    # No config in the zip: rebuild it from the repo's Wildtrack + model configs
    print('config: (none in zip) using d_wildtrack_colab.yml + m_' + MODEL + '.yml')
    !python world_track.py test \
        -c configs/t_fit.yml \
        -c configs/d_wildtrack_colab.yml \
        -c configs/m_{MODEL}.yml \
        --trainer.devices 1 \
        --ckpt_path {ckpt}

The test prints the same metric table as your own runs. For Segnet, the paper’s targets are MODA 92.1, IDF1 95.3 and MOTA 91.8.

    Close to those: your environment and data are fine, and any gap in your own runs comes from training.
    Clearly lower: something in the Colab setup differs from the authors’, such as library versions or the dataset mirror.

If the else branch runs, check Cell A’s output first. If the checkpoint says use_temporal_cache: True, add --model.use_temporal_cache true to that command so the model is rebuilt the same way it was trained.

To watch the official model in the live demo, set ckpt in the demo cell to the path Cell B printed.

downloading wild_mvdet.zip ...

=== wild_mvdet ===
      2.1 MB  wild_mvdet/events.out.tfevents.1710775116.tueimmk-iason5.31825.0
      0.0 MB  wild_mvdet/config.yaml
    124.5 MB  wild_mvdet/model-epoch=27-val_loss=8.07-val_center=5.45.ckpt
      0.0 MB  wild_mvdet/hparams.yaml
  ckpt hparams : {'model_name': 'mvdet', 'use_temporal_cache': False, 'resolution': [120, 4, 360], 'bounds': [0, 1440, 0, 480, 0, 200], 'num_cameras': 7, 'encoder_name': 'res18', 'conf_threshold': 0.5}
  config.yaml  : {'model_name': 'mvdet', 'use_temporal_cache': False, 'resolution': [120, 4, 360], 'bounds': [0, 1440, 0, 480, 0, 200], 'num_cameras': 7, 'encoder_name': 'res18', 'conf_threshold': 0.5}
  hparams.yaml : {'model_name': None, 'use_temporal_cache': None, 'resolution': None, 'bounds': None, 'num_cameras': None, 'encoder_name': None, 'conf_threshold': None}
downloading wild_segnet.zip ...

=== wild_segnet ===
      2.1 MB  wild_segnet/events.out.tfevents.1710777427.tueimmk-iason5.17343.0
    151.2 MB  wild_segnet/model-epoch=21-val_loss=7.79-val_center=4.76.ckpt
      0.0 MB  wild_segnet/config.yaml
      0.0 MB  wild_segnet/hparams.yaml
  ckpt hparams : {'model_name': 'segnet', 'use_temporal_cache': False, 'resolution': [120, 4, 360], 'bounds': [0, 1440, 0, 480, 0, 200], 'num_cameras': 7, 'encoder_name': 'res18', 'conf_threshold': 0.5}
  config.yaml  : {'model_name': 'segnet', 'use_temporal_cache': False, 'resolution': [120, 4, 360], 'bounds': [0, 1440, 0, 480, 0, 200], 'num_cameras': 7, 'encoder_name': 'res18', 'conf_threshold': 0.5}
  hparams.yaml : {'model_name': None, 'use_temporal_cache': None, 'resolution': None, 'bounds': None, 'num_cameras': None, 'encoder_name': None, 'conf_threshold': None}
downloading wild_liftnet.zip ...

=== wild_liftnet ===
      2.1 MB  wild_liftnet/events.out.tfevents.1710787660.tueimmk-iason5.22117.0
    130.6 MB  wild_liftnet/model-epoch=23-val_loss=7.79-val_center=4.95.ckpt
      0.0 MB  wild_liftnet/config.yaml
      0.0 MB  wild_liftnet/hparams.yaml
  ckpt hparams : {'model_name': 'liftnet', 'use_temporal_cache': False, 'resolution': [120, 4, 360], 'bounds': [0, 1440, 0, 480, 0, 200], 'num_cameras': 7, 'encoder_name': 'res18', 'conf_threshold': 0.5}
  config.yaml  : {'model_name': 'liftnet', 'use_temporal_cache': False, 'resolution': [120, 4, 360], 'bounds': [0, 1440, 0, 480, 0, 200], 'num_cameras': 7, 'encoder_name': 'res18', 'conf_threshold': 0.5}
  hparams.yaml : {'model_name': None, 'use_temporal_cache': None, 'resolution': None, 'bounds': None, 'num_cameras': None, 'encoder_name': None, 'conf_threshold': None}

Checking whether Wildtrack checkpoints were trained without temporal fusion.
Checking whether Wildtrack checkpoints were trained without temporal fusion.

All three official Wildtrack checkpoints were trained with use_temporal_cache: False. The checkpoint and its config.yaml agree for MVDet, Segnet and Liftnet.

What this means:

    Your Wildtrack setup matches the authors’. Resolution, bounds, 7 cameras, the ResNet-18 encoder and the 0.5 confidence threshold are all identical to your config, and so is the history-off setting.
    The authors’ released Wildtrack models don’t use history fusion. Section 3.2 of the paper describes the previous-frame input as part of the architecture, but these models never received it. Each frame was combined with a copy of itself.
    The paper’s Wildtrack results probably came from history-off models. These checkpoints are very likely the ones behind the Wildtrack rows of Tables 1 and 2, but the release doesn’t say so explicitly. The paper’s only test of history fusion is on MultiviewX (Table 4).

Your baseline is therefore a fair reproduction of the authors’ Wildtrack models, but not of the paper’s full architecture. Idea #4 in your improvement log (history on vs off) now tests something the paper itself never shows for Wildtrack.

The other details in that output:

    hparams.yaml showing None is a bug in my cell, not a problem with the file. Lightning writes that file with the settings at the top level, but my code looked for them under a model: heading. The other two sources already agree, so it doesn’t matter.
    The epoch numbers in the file names show when each saved model scored best: MVDet at epoch 27, Segnet at 21, Liftnet at 23, out of a 30-epoch run. Each was kept for the lowest val_center (the detection-heatmap loss on validation). As your improvement log notes, validation and test are the same 40 frames. So the released models, and the paper’s numbers if they came from them, were chosen partly by looking at test performance, which makes them slightly optimistic.

Next, run Cell B with MODEL = 'segnet' to see whether the official checkpoint reproduces MODA 92.1 / IDF1 95.3 / MOTA 91.8 in your Colab environment. The zips include config.yaml, so Cell B will use the authors’ own config.


Claude is AI and can make mistakes.

