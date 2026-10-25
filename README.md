# Retrack — robust multi-view pedestrian tracking on top of TrackTacular

This repository extends [TrackTacular](https://arxiv.org/abs/2403.12573) with the four stages of the Retrack
methodology (see `final-literature-review-plus-proposed-approach.docx`) and an end-to-end pipeline that compares
the TrackTacular baseline with Retrack on WILDTRACK, with all cameras and with cameras removed.

## Run everything in Colab (one cell)

1. Open `Retrack_Colab.ipynb` in Colab (File → Open notebook → GitHub → `rizwanmustafa/retrack`), or create a new
   notebook and paste the cell below.
2. Runtime → Change runtime type → **A100 GPU**, High-RAM.
3. Set `MODE` (`'quick'` first, then `'full'`) and `BRANCH`, then run the cell.

```python
#@title Retrack — TrackTacular baseline vs Retrack (run this one cell)
# Runtime > Change runtime type > A100 GPU (High-RAM). Everything is saved to Google Drive and the cell resumes
# where it stopped if Colab disconnects: just run it again.
MODE    = 'quick'   # 'quick': 2-epoch run of the whole pipeline (~1-2 h) to check it works | 'full': the study
SEEDS   = 3         # training seeds per fusion variant in 'full' mode (5 variants x SEEDS runs of 30 epochs)
RESULTS = '/content/drive/MyDrive/retrack_results'
REPO    = 'github.com/rizwanmustafa/retrack.git'
BRANCH  = 'main'    # use 'feat/retrack' until that branch is merged into main

import os, glob
from google.colab import drive, userdata
drive.mount('/content/drive')

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

# Optional Colab Secrets (key icon on the left): KAGGLE_USERNAME / KAGGLE_KEY if the WILDTRACK download is refused,
# GITHUB_TOKEN if the repository is private.
for k in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
    if secret(k):
        os.environ[k] = secret(k)
url = f"https://{secret('GITHUB_TOKEN')}@{REPO}" if secret('GITHUB_TOKEN') else f'https://{REPO}'
if not os.path.exists('/content/retrack'):
    !git clone -q -b {BRANCH} {url} /content/retrack
else:
    !git -C /content/retrack fetch -q origin {BRANCH} && git -C /content/retrack checkout -q {BRANCH} && git -C /content/retrack pull -q origin {BRANCH}

extra = f'--seeds {SEEDS}' if MODE == 'full' else ''
!bash /content/retrack/colab/run.sh {MODE} {RESULTS} {extra}

from IPython.display import Markdown, Image, display
report_dir = f'{RESULTS}/{MODE}/report'
if os.path.exists(f'{report_dir}/REPORT.md'):
    display(Markdown(open(f'{report_dir}/REPORT.md').read()))
    for png in sorted(glob.glob(f'{report_dir}/*.png')):
        display(Image(png, width=640))
    print('All results, tables (CSV) and plots:', report_dir)
```

The cell mounts Google Drive, clones this repo, installs the dependencies (no numpy downgrade, so no runtime
restart), downloads WILDTRACK from Kaggle and the official TrackTacular checkpoint, runs every experiment and
displays the report. All outputs go to `MyDrive/retrack_results/<mode>/`. Each step is skipped when its output
already exists and interrupted training resumes from `last.ckpt`, so **if Colab disconnects, run the cell
again**.

| mode | what runs | time on an A100 (rough estimate, not measured) |
|---|---|---|
| `quick` | everything, but 2 epochs, 1 seed, ≤1 removed camera | ~1–2 h |
| `full`  | 5 fusion variants × 3 seeds × 30 epochs (the paper schedule), 0–3 removed cameras (all 64 combinations) | ~1 h per training run, plus evaluation |

Optional Colab Secrets: `KAGGLE_USERNAME`/`KAGGLE_KEY` (if the anonymous Kaggle download is refused) and
`GITHUB_TOKEN` (if the repository is private). Without Colab: `bash colab/run.sh <quick|full> <results_dir>`, or
`cd WorldTrack && python retrack_pipeline.py --help`.

### Outputs (`<results>/<mode>/`)
- `report/REPORT.md`: all tables (also as CSV) and plots:
  0. sanity check: the evaluation engine reproduces TrackTacular's own `test` numbers on the official checkpoint
  1. Stage 1 threshold sweep (recall / precision / MODA / MODP / MOTA / IDF1 per threshold)
  2. Stages 1–3 applied to the trained TrackTacular model, by number of removed cameras
  3. Stage 2 parameter grid (low threshold × track buffer)
  4. Stage 3 parameter grid (λ × appearance threshold)
  5. Stage 4 fusion ablation (5 variants, mean ± std over seeds, spread over camera combinations, slope per removed camera)
  6. final comparison: TrackTacular baseline vs full Retrack, with raw counts of misses, false positives, ID switches and fragmentations
- `eval/<model>/metrics.csv`: every metric for every model × removed-camera combination × configuration
- `runs/`: training logs and checkpoints, `model_zoo/`: the official checkpoint

## What is implemented, and where

| Proposal | Code |
|---|---|
| **Stage 1** threshold sweep, no retraining | `evaluation/retrack_eval.py` (`THRESHOLDS`, `pick_threshold`) |
| **Stage 2** ByteTrack-style association on BEV points: high-confidence pass against all tracks, low-confidence pass (motion only, centre distance instead of IoU) against the unmatched tracks only, new tracks only from high-confidence detections, longer track buffer (5 → 10) | `tracking/multitracker.py` (`ByteBEVTracker`) |
| **Stage 3** appearance from the *working* cameras only: person box (assumed height = 0.9 × z-range, i.e. 180 cm) projected into every working camera, cropped, embedded with pretrained OSNet-x1.0 (MSMT17), averaged over the views that see the person; running-average track descriptor; `D = λ·D_app + (1-λ)·D_motion` (λ = 0.98) in the high-confidence pass only | `tracking/reid.py`, `ByteBEVTracker.first_pass_cost` |
| **Stage 4** camera-dropout training | `WorldTrackModel.get_cam_mask` (`cam_dropout_prob`, `cam_dropout_max`) |
| masked fusion with an explicit camera mask × field-of-view mask (instead of the "feature == 0" test) | `models/fusion.py` (`fuse_views`), FOV masks from `utils/vox.py` (`return_valid=True`) |
| coverage map `c(x,y) = 1/N Σ m_n(x,y)` as an extra BEV channel | `models/fusion.py` (`coverage_map`), used in `models/segnet.py`, `mvdet.py`, `liftnet.py` |
| learned per-camera weighting (masked softmax over a 2-layer scorer; zero-initialised so it starts as the masked mean) | `models/fusion.py` (`CameraScorer`, `masked_softmax_weights`) |
| camera removal at test time (averaged over all combinations of k removed cameras) | `WorldTrackModel.test_drop_cams`, `evaluation/retrack_eval.py` (`camera_conditions`) |
| experiment variants | `configs/v_concat.yml`, `v_concat_drop.yml`, `v_masked_drop.yml`, `v_masked_cov_drop.yml`, `v_weighted_cov_drop.yml`, schedule `configs/t_retrack.yml` |
| orchestration and report | `retrack_pipeline.py`, `evaluation/retrack_report.py` |

With the default settings (`fusion: concat`, no dropout, `tracker: baseline`) the code behaves exactly like the
original TrackTacular, and the released checkpoints load unchanged.

### Design decisions you should know about
- **Camera failure model:** a removed camera's projected BEV features are zeroed and the camera is marked as
  missing (the camera "sends nothing"). The image-view auxiliary loss is unchanged.
- **Lifting method:** Segnet (bilinear sampling), TrackTacular's best WILDTRACK model. `--model mvdet|liftnet`
  also works; robust fusion is implemented for all three. It is not implemented for BEVFormer, which needs mmcv.
- **Which checkpoint is evaluated:** retrained models are evaluated at the **last epoch**. WILDTRACK's validation
  frames are its test frames, so picking the best epoch by validation loss (as the released checkpoints were)
  would be selection on the test set.
- **Tuned threshold:** chosen as the best MODA on the test frames for lack of a validation split. The report says so
  and shows the full sweep. The Stage 2 and 3 parameters are fixed in advance, and the grids are reported separately.
- **Stage 1–3 baseline:** the official released TrackTacular checkpoint. The Stage 4 / final baseline is the
  retrained concatenation model, trained with the same schedule and seeds as the Retrack variants.
- The detection evaluation also counts frames in which nothing is detected; the original code skipped them. This
  only matters with many cameras removed.
- Compatibility fixes for current Colab (torch ≥ 2.2 sampler, mmcv optional, missing datamodule imports, numpy 2
  shim for motmetrics, device-agnostic normalisation buffers).

---

# TrackTacular :octopus:

**Lifting Multi-View Detection and Tracking to the Bird’s Eye View**

[Torben Teepe](https://github.com/tteepe),
[Philipp Wolters](https://github.com/phi-wol),
[Johannes Gilg](https://github.com/Blueblue4),
[Fabian Herzog](https://github.com/fubel),
Gerhard Rigoll

[![arxiv](https://img.shields.io/badge/arXiv-2403.12573-red)](https://arxiv.org/abs/2403.12573)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lifting-multi-view-detection-and-tracking-to/multi-object-tracking-on-wildtrack)](https://paperswithcode.com/sota/multi-object-tracking-on-wildtrack?p=lifting-multi-view-detection-and-tracking-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lifting-multi-view-detection-and-tracking-to/multi-object-tracking-on-multiviewx)](https://paperswithcode.com/sota/multi-object-tracking-on-multiviewx?p=lifting-multi-view-detection-and-tracking-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lifting-multi-view-detection-and-tracking-to/multiview-detection-on-multiviewx)](https://paperswithcode.com/sota/multiview-detection-on-multiviewx?p=lifting-multi-view-detection-and-tracking-to)

> [!TIP]
> This work is an extension of your previous work [EarlyBird  🦅](https://github.com/tteepe/EarlyBird).
> Feel free to check it out and extend our multi-view object detection and tracking pipeline on other datasets!

![Overview](imgs/overview.svg)

## Usage

### Getting Started
1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support
    ```shell
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Install [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip) with CUDA support
   ```shell
   pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
   ```
3. Install remaining dependencies
   ```shell
   pip install -r requirements.txt
   ```

### Training
```shell
python world_track.py fit -c configs/t_fit.yml \
    -c configs/d_{multiviewx,wildtrack,synthehicle}.yml \
    -c configs/m_{mvdet,segnet,liftnet,bevformer}.yml
```
### Testing
```shell
python world_track.py test -c model_weights/config.yaml \
    --ckpt model_weights/model-epoch=35-val_loss=6.50.ckpt
```

## Acknowledgement
- [Simple-BEV](https://simple-bev.github.io): Adam W. Harley
- [MVDeTr](https://github.com/hou-yz/MVDeTr): Yunzhong Hou

## Citation
```bibtex
@article{teepe2023lifting,
      title={Lifting Multi-View Detection and Tracking to the Bird's Eye View}, 
      author={Torben Teepe and Philipp Wolters and Johannes Gilg and Fabian Herzog and Gerhard Rigoll},
      year={2024},
      eprint={2403.12573},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
