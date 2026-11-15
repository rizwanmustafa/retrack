# Tools documentation

Everything in this branch that is not the model itself: training and evaluation on Colab, the report, and the visual
tools for looking at what a checkpoint does. For the two copy-paste Colab cells, see [README.md](README.md).

**Contents**

1. [Overview](#1-overview)
2. [Before you start](#2-before-you-start)
3. [Training and evaluation (`train_eval.py`)](#3-training-and-evaluation-train_evalpy)
4. [The report (`REPORT.md`, `report.py`)](#4-the-report)
5. [The visual tools (`viz.py`)](#5-the-visual-tools-vizpy)
6. [Recipes: answering questions with the tools](#6-recipes)
7. [Command-line use (without the Colab drivers)](#7-command-line-use)
8. [Files and folders](#8-files-and-folders)
9. [Troubleshooting](#9-troubleshooting)
10. [Terms](#10-terms)

---

## 1. Overview

| tool | file | run it when |
|---|---|---|
| Training + evaluation driver | `colab/train_eval.py` | you want new checkpoints and numbers (hours, needs a GPU) |
| Pipeline (what the driver calls) | `colab/pipeline.py` | outside Colab, or to control every option |
| Evaluation of one checkpoint | `WorldTrack/evaluation/r2_eval.py` | you have one checkpoint and want its metrics |
| Report | `colab/report.py` | re-make `REPORT.md` and the figures from existing evaluations |
| Visual tools | `colab/viz.py` | you want to *see* detections, misses, cameras, tracks (minutes, any GPU) |
| Shared helpers | `colab/common.py` | (used by the others: Drive, datasets, packages) |

The two variants that exist on this branch:

| name | what it is | how it is switched on |
|---|---|---|
| **B0** | TrackTacular: cameras → ResNet-18 → features lifted onto a 3D grid → camera fusion → BEV map → dense U-Net decoder → heatmap of people | default |
| **R2** | same up to the BEV map; the dense decoder is replaced by ~100 sparse person queries that also look back into the camera images | `--model.sparse_queries true` (`configs/m_r2.yml`) |

Both are evaluated identically: threshold 0.5, TrackTacular's tracker, last checkpoint.

---

## 2. Before you start

**Datasets on Google Drive.** The tools restore the datasets into the Colab runtime (`/content/data`) from:

| dataset | file on Drive | inside the tar |
|---|---|---|
| WILDTRACK | `MyDrive/datasets/wildtrack.tar` | `Wildtrack/Image_subsets`, `annotations_positions`, `calibrations` |
| MultiviewX | `MyDrive/datasets/multiviewx.tar` | `MultiviewX/Image_subsets`, `annotations_positions`, `calibrations` |

If `wildtrack.tar` is missing, WILDTRACK is downloaded from Kaggle and the tar is written to Drive for next time.
MultiviewX is only distributed on OneDrive (https://github.com/hou-yz/MultiviewX), so its tar must be uploaded once.
A dataset is restored only when it is needed (the viewer restores MultiviewX only when you open a MultiviewX checkpoint).

**The code on Colab comes from GitHub.** Colab clones the branch `feat/r2-sparse-queries`; local changes reach Colab
only after they are committed and pushed.

**GPU.** Training: an A100 or G4 (RTX PRO 6000) runs two trainings at once; an L4/T4 runs one. The visual tools work on
any GPU (a T4 is fine) and, slowly, on a CPU runtime.

---

## 3. Training and evaluation (`train_eval.py`)

### Colab cell

```python
!rm -rf /content/retrack && git clone -q -b feat/r2-sparse-queries https://github.com/rizwanmustafa/retrack.git /content/retrack
SEEDS = '80085 1 2'
try:
    %run -i /content/retrack/colab/train_eval.py
finally:
    from google.colab import drive, runtime
    drive.flush_and_unmount()
    runtime.unassign()   # releases the GPU when the run ends; remove these 3 lines to keep the runtime
```

### Settings

Set any of these in the cell **before** the `%run` line; anything not set keeps its default.

| setting | default | meaning |
|---|---|---|
| `RESULTS` | `'/content/drive/MyDrive/retrack_results_r2'` | output folder (keep it on Drive: the runtime's disk is wiped) |
| `DATASETS` | `'wildtrack multiviewx'` | one or both |
| `VARIANTS` | `'B0 R2'` | one or both |
| `SEEDS` | `'80085 1 2'` | one training per variant × dataset × seed; 3 seeds give mean ± std |
| `REUSE` | `'/content/drive/MyDrive/retrack_results_m5/m5'` | earlier results folder whose finished runs are taken over instead of retrained (ignored if the folder does not exist) |
| `PARALLEL` | automatic: 2 with ≥ 38 GB GPU memory, else 1 | trainings at the same time |
| `EXTRA` | `''` | extra `pipeline.py` options, e.g. `'--max_drop 2'` or `'--skip_preflight'` (see §7) |

Examples:

```python
DATASETS = 'wildtrack'; VARIANTS = 'R2'; SEEDS = '3'        # one extra R2 seed on WILDTRACK only
EXTRA = '--epochs_wildtrack 10 --max_drop 1'                # a quick, shorter experiment (not comparable!)
```

### What happens, step by step

| step | what | time (G4, 2 in parallel) | skipped when |
|---|---|---|---|
| 1. setup | mount Drive, restore datasets, install packages | 2–5 min | already done in this runtime |
| 2. preflight | every variant × dataset: 4 training batches, reload the checkpoint the way the evaluation does, evaluate 3 frames | 1–2 min each | `preflight/<run>/PASSED` exists |
| 3. reuse | runs found in `REUSE` are marked done (the checkpoint is not copied, only referenced) | seconds | — |
| 4. training | 30 epochs (WILDTRACK) / 50 (MultiviewX), batch 2 × 8 accumulation, no temporal history, fp32, last checkpoint | ~40 min / ~60–75 min per run | `runs/<run>/DONE.json` exists |
| 5. evaluation | each finished training: all cameras + every combination of up to 3 removed cameras (64 conditions on WILDTRACK, 42 on MultiviewX) | 3–6 min per run | `eval/<run>/metrics.csv` exists |
| 6. report | `REPORT.md` + `figures/robustness_<dataset>.png`, shown in the notebook | seconds | never (always rebuilt) |
| 7. bundle | `for_claude.zip`: everything except checkpoints and videos | seconds | never |

The default (both datasets, B0 + R2, seeds 80085 1 2, methodology-5 runs reused) trains 8 models: about 4–5 hours.

### Stopping and continuing

Run the same cell again. Finished trainings and evaluations are skipped; a training that was interrupted resumes
from its last checkpoint (`last.ckpt` is written every epoch). The log of everything is appended to
`RESULTS/pipeline.log`; each training also has its own `runs/<run>/train.log`.

### Progress lines

```
[16:19:20] Training wildtrack_R2_s1 (30 epochs); log: .../runs/wildtrack_R2_s1/train.log
[16:20:20] running: multiviewx_B0_s1 (44 min), wildtrack_R2_s1 (1 min)
[16:58:24] Trained wildtrack_R2_s1 in 39 min -> .../last.ckpt
[16:58:24] Evaluating wildtrack_R2_s1
  all cameras: MODA <..>  MOTA <..>  IDF1 <..>  (FP <..>, FN <..>, ID switches <..>)
  64 conditions in <..>s; 12.73 M parameters, <..> ms/frame
```

---

## 4. The report

`RESULTS/REPORT.md` has, per dataset:

1. **All cameras** — one row per variant: MODA, MODP, precision, recall, MOTA, IDF1, ID switches, false detections
   (FP), misses (FN), parameters, milliseconds per frame. With several seeds every cell is *mean ± sample std*.
2. **Cameras removed** — MODA / MOTA / IDF1 for k = 0..3 removed cameras. Each value is the mean over *every*
   combination of k cameras (e.g. 21 pairs on WILDTRACK for k = 2), then over seeds.
3. **R2 − B0, paired by seed** — the difference computed seed by seed, then mean ± std. A difference is convincing
   when it is clearly larger than its std and has the same sign for every seed.
4. **Figure** `figures/robustness_<dataset>.png` — the camera-removal table as curves with a ± 1 std band.

Scale of the numbers: one MODA point is about 9–10 errors (misses + false detections) on WILDTRACK's 40 test frames
and about 15 on MultiviewX's. Differences below ~1 point with fewer than 3 seeds are within run-to-run noise.

Re-make the report after changing anything in `eval/`:

```python
!python /content/retrack/colab/report.py --results /content/drive/MyDrive/retrack_results_r2
```

---

## 5. The visual tools (`viz.py`)

### Starting

```python
import os
if not os.path.exists('/content/retrack'):
    !git clone -q -b feat/r2-sparse-queries https://github.com/rizwanmustafa/retrack.git /content/retrack
%run -i /content/retrack/colab/viz.py
```

This mounts Drive, installs missing packages, prints the device (`cuda` or `cpu`) and lists the checkpoints. Then call
the functions below in new cells. Optional settings, set **before** the `%run` line:

| setting | default | meaning |
|---|---|---|
| `ROOTS` | every `MyDrive/retrack_results*` folder, plus `MyDrive/checkpoints` and `/content/checkpoints` if they exist | where to look for `*.ckpt` (a folder, a list of folders, or a single `.ckpt` file) |
| `DATA_ROOT` | `'/content/data'` | where the datasets are restored |
| `VIDEO_DIR` | `'/content/drive/MyDrive/retrack_videos'` | where `video()` saves files |
| `NUM_WORKERS` | `2` | data-loading processes when the test frames are read |

### Choosing a checkpoint

Every tool takes a checkpoint as its first argument, given in any of three ways:

```python
explore(3)                                   # its number in the list printed by checkpoints()
explore('multiviewx_R2_s1')                  # any part of its path that matches exactly one checkpoint
explore('/content/drive/MyDrive/x/last.ckpt')  # a full path (does not need to be in the list)
```

Loading a checkpoint the first time takes 20–60 s (the 40 test frames of its dataset are read into memory; they are
shared by every checkpoint of the same dataset). After that it is cached for the rest of the session.

**Camera numbers:** in arguments cameras count from 0 (`drop=[0]` switches off the first camera); in figures and
buttons they are named C1, C2, … So `drop=[0, 4]` = C1 and C5 off.

**Frame numbers:** `frame=` is the position in the test split, 0–39. Figures show the dataset's own frame number
(WILDTRACK 360–399, MultiviewX 360–399).

### `checkpoints(roots=None, include_preflight=False, refresh=False)`

Lists every checkpoint found and returns the list.

```
/content/drive/MyDrive/retrack_results_m5
    #  variant  dataset       seed epoch    MB  eval  file
    0  B0       multiviewx   80085    49    51  yes   m5/runs/multiviewx_B0_s80085/lightning_logs/version_0/checkpoints/last.ckpt
    1  R2       multiviewx   80085    49    47  yes   m5/runs/multiviewx_R2_s80085/...
    4  –        multiviewx   80085    49    57        m5/runs/multiviewx_R14_s80085/...   [unsupported: R1 (Depth Anything 3 features), R4 (track queries)]
```

| column | meaning |
|---|---|
| `#` | the number to pass to the tools |
| `variant` | `B0` or `R2`, read from the checkpoint's settings; `–` = a research variant this branch does not contain (it cannot be opened here) |
| `dataset`, `seed`, `epoch` | from the checkpoint and its folder name (`epoch` counts from 0: 49 = the 50th epoch) |
| `eval` | `yes` if the pipeline's evaluation exists for it (`robustness()` then reads it instead of recomputing) |

- `checkpoints(roots='/content/drive/MyDrive/other_folder')` looks somewhere else (also accepts a list).
- `include_preflight=True` also lists the tiny preflight checkpoints (normally hidden: they are untrained).
- `refresh=True` re-reads every checkpoint's settings (use it if a file was overwritten).

### `explore(which, frame=0, drop=(), cameras=True)` — the main viewer

```python
explore(3)                      # start at test frame 0 with all cameras
explore(3, frame=12, drop=[1])  # start at frame 12 with C2 off
```

**Controls**

| control | what it does |
|---|---|
| ▶ play / frame slider | step through the 40 test frames (play shows one frame every 1.5 s) |
| C1 … C7 buttons | click to switch a camera **off** (button turns red) or on again. The model is re-run on all 40 frames with exactly the working cameras: a few seconds on a G4/A100, ~15 s on a T4, about a minute on CPU. The last ~6 camera sets are cached, so switching back is instant. |
| all cameras on | resets the buttons |
| threshold | detection score threshold used for the drawing and the per-frame counts (the evaluation always uses 0.5). Lower it to see the people the model *almost* found. |
| background | **model heatmap**: the model's person score per BEV cell (B0: the centre heatmap; R2: the proposal heatmap the queries start from). **working-camera coverage**: how many working cameras see each cell at mid-body height — dark = few. **none** |
| camera images | show/hide the camera panels under the BEV |
| track IDs | show the tracker's ID next to each tracked person |
| fields of view | outline of what each camera sees (dashed grey for switched-off cameras) |
| R2 queries | R2 only: grey `+` = where each query started (proposal cells), coloured dots = where the queries ended after 3 rounds, colour = score |

**Reading the bird's-eye view (BEV)**

| mark | meaning |
|---|---|
| white dot | a person in the ground truth |
| green dot (+ green line to the white dot) | detected (true positive), the line shows the localisation error |
| **orange circle with a number** | **missed** (false negative); the number = how many *working* cameras see this person according to the annotation boxes. 0–1: the evidence is not there; 2–3: the model saw the person but was not confident enough |
| **red ×** | **false detection** (false positive): above the threshold, no person within 0.5 m |
| cyan number | track ID |
| coloured triangle + label | camera position (C1…), clamped to the edge of the plot if the camera is far outside the area |
| coloured outline | the ground area that camera sees |

The title repeats the frame, the working cameras and the counts (found / missed / false).

**Reading the camera images**

Each person is drawn as a box from the feet to 1.8 m above the ground, projected with the calibration: green = detected,
orange = missed, red = a false detection projected into the image (what the model "thought" was a person there).
The panel title says how many annotated people the camera sees; switched-off cameras are dark with `OFF`.

**The text under the figure** gives the metrics of all 40 frames with the current cameras (MODA, MOTA, IDF1, FP, FN, ID
switches) and then, for the current frame, every missed person (person ID, the working cameras that see them, the
best score of any detection within 0.5 m) and every false detection (its score and the distance to the nearest person).

### `compare(a, b, frame=0, drop=())` — two checkpoints side by side

```python
compare(0, 1)                  # e.g. B0 and R2 on MultiviewX
compare('wildtrack_B0_s1', 'wildtrack_R2_s1', drop=[2, 5])
```

Same frame, cameras, threshold and background for both (they must be of the same dataset). WILDTRACK is drawn one above
the other, MultiviewX side by side. Under the figure: the all-frame metrics of both with the current cameras.

### `camera_evidence(which, frame=0)` — what each camera contributes

```python
camera_evidence(3, 10)
```

One frame. The first panel is the heatmap with all cameras. Then, for every camera C: *heatmap with all cameras − heatmap
without C*. **Red** = score that disappears without that camera (people that depend on it), **blue** = score that only
appears without it (the camera was suppressing something). Black circles = people; the coloured outline = that
camera's field of view. Each title says how many people are missed and falsely detected in this frame without that
camera. Use it to see whether a person was found thanks to one camera only.

### `tracks(which, drop=())` — trajectories and identity errors

```python
tracks(3)               # all cameras
tracks(3, drop=[0, 4])  # with C1 and C5 off
```

All 40 test frames on one BEV: grey lines = ground-truth trajectories, coloured lines = tracks (one colour per track ID,
the ID written at the track's end), **red stars = ID switches** (a person's track changed identity there), small orange
dots = frames where a person was not covered by any track. The title gives MOTA, IDF1 and the number of ID switches.
A person whose grey line is covered by two colours was split into two tracks.

### `robustness(*which, max_drop=3)` — curves against removed cameras

```python
robustness(0, 1)                        # B0 vs R2
robustness('wildtrack_B0', 'wildtrack_R2', max_drop=2)
```

Five panels — MODA, MOTA, IDF1, false detections, misses — against the number of removed cameras k (mean over every
combination of k cameras). If the pipeline already evaluated the checkpoint (`eval` = yes in the list) the numbers are
read from `eval/<run>/metrics.csv`; otherwise they are computed now (WILDTRACK, k ≤ 3: 64 runs over 40 frames, about
4 min on a G4, ~15 min on a T4; `max_drop=2` halves that). Returns the numbers as a dict of tables.

### `video(which, drop=(), path=None, fps=2, thr=0.5, cameras=True, background='heatmap')`

```python
video(3)                          # all cameras
video(3, drop=[2], fps=3)         # C3 off, 3 frames per second
video(3, cameras=False, background='coverage', path='/content/drive/MyDrive/talk/r2_bev.mp4')
```

Renders the viewer for all 40 frames (about 1 min) into an MP4 and plays it in the notebook. Saved by default as
`VIDEO_DIR/<variant>_<dataset>_s<seed>_<all|offN>.mp4`. If no MP4 encoder is available it writes a GIF instead.
`cameras=False` gives a BEV-only video (smaller, easier to put on a slide).

### `session(which)` — the objects behind the tools (for your own analysis)

```python
s = session(3)                 # loaded checkpoint
rec = s.run(drop=[0])          # all 40 frames with C1 off (cached)
print(s.summary(rec))          # MODA / MOTA / IDF1 / FP / FN / ID switches
f = rec['frames'][5]           # one frame: f['gt'] (x, y, person ID), f['xy'], f['score'], f['heat'], f['queries'] (R2)
a = frame_analysis(s, rec, 5)  # matching at 0.5: a['tp_gt'], a['fn'], a['fp'], a['seen'] (working cameras per person)
fig = render(s, rec, 5)        # the viewer figure as a matplotlib Figure (fig.savefig('frame5.png', dpi=150))
```

Coordinates: `f['gt']` and `f['xy']` are in the dataset's ground grid (2.5 cm cells; multiply by 0.025 for metres);
`f['heat']` and the R2 queries are in BEV cells (10 cm). `s.ref2mem()` / `s.mem2ref()` convert between them.

`help()` prints the short version of this section.

---

## 6. Recipes

**"Why does the model miss people when cameras fail?"**
1. `explore(n)`, background = *working-camera coverage*, then switch off two cameras.
2. Watch the orange circles appear: their numbers (working cameras that see the person) are mostly 1–3.
3. Lower the threshold to 0.3: many of them turn green — the model saw them, but with a low score (under-confidence).
   The ones that stay orange with a number of 0–1 had no evidence left.

**"Is R2 better than B0, and where?"**
1. `robustness(b0, r2)` for the overall picture (in the methodology-5 run, seed 80085, R2 was ahead with all cameras
   and behind as cameras were removed; with 3 seeds the report shows whether that holds).
2. `compare(b0, r2)` and play through the frames; then switch cameras off and compare again.
3. `tracks(b0)` vs `tracks(r2)` for identity errors (red stars).

**"Which camera matters for this person?"** `camera_evidence(n, frame)`: the panel where that person's spot is
strongly red is the camera the detection depends on.

**"What do R2's queries do?"** `explore(r2)` with *R2 queries* on: the grey `+` mark where queries started (usually
several around each person). R2 is trained so that after 3 rounds one query per person has a high score (bright dot on
the person) and the duplicates a low one; check whether that is what you see. With cameras off, look for missed people
with no `+` near them: the proposal heatmap did not fire there, so no query could find them.

**Figures and videos for the report or presentation**
- `fig = render(session(n), session(n).run(), 12); fig.savefig('/content/drive/MyDrive/fig_frame12.png', dpi=200)`
- `video(n, drop=[1, 3], cameras=False)` for a short BEV clip; `video(n)` with the camera panels for a demo.
- `RESULTS/figures/robustness_<dataset>.png` from the report is ready for a slide.

---

## 7. Command-line use

All of this also works without the Colab drivers (from a terminal, or `!` lines in Colab).

**Pipeline** (`colab/pipeline.py`, run from the repository root):

```bash
python colab/pipeline.py --results /content/drive/MyDrive/retrack_results_r2 --seeds 80085 1 2 \
    --reuse /content/drive/MyDrive/retrack_results_m5/m5
```

| option | default | meaning |
|---|---|---|
| `--results` | (required) | output folder |
| `--data_root` | `/content/data` | folder holding `Wildtrack/` and `MultiviewX/` |
| `--datasets` | `wildtrack multiviewx` | |
| `--variants` | `B0 R2` | |
| `--seeds` | `80085` | |
| `--reuse` | none | earlier results folder (layout `runs/<dataset>_<variant>_s<seed>/DONE.json`) whose runs are taken over |
| `--parallel` | 2 | trainings at once |
| `--epochs_wildtrack`, `--epochs_multiviewx` | 30, 50 | TrackTacular's schedule; change only for quick experiments |
| `--max_drop` | 3 | evaluate with up to this many cameras removed |
| `--batch_size`, `--accumulate` | 2, 8 | effective batch 16, as TrackTacular |
| `--num_workers` | 5 | data-loading processes per training |
| `--precision` | `32-true` | as the baselines |
| `--train_args` | `''` | extra arguments for `world_track.py fit`, e.g. `'--trainer.max_time 00:02:00:00'` |
| `--skip_preflight` | off | skip step 2 |

**One training by hand** (from `WorldTrack/`):

```bash
python world_track.py fit -c configs/t_colab.yml -c configs/d_wildtrack.yml -c configs/m_segnet.yml -c configs/m_r2.yml \
    --model.use_temporal_cache false --data.init_args.data_dir /content/data/Wildtrack \
    --trainer.default_root_dir /content/drive/MyDrive/my_run --seed_everything 1
```

Leave out `-c configs/m_r2.yml` for B0. R2's own options (`--model.num_queries 100`, `--model.query_layers 3`,
`--model.query_heights 4`, `--model.query_match_cells 10`) can be changed the same way.

**One evaluation by hand** (from `WorldTrack/`):

```bash
python -m evaluation.r2_eval --ckpt <last.ckpt> --data_dir /content/data/Wildtrack --out <folder> [--max_drop 3] [--quick]
```

Writes `metrics.csv` (one row per camera condition: `drop` = removed cameras, `k`, MODA, MODP, precision, recall,
TP/FP/FN, MOTA, IDF1, MT, ML, ID switches), `summary.json` (all-camera row, means per k, parameters, ms/frame) and
`all_cameras_frames.pt` (the detections and tracks of every frame with all cameras). `--quick` = 3 frames, all cameras.

---

## 8. Files and folders

```
RESULTS/                                   (default MyDrive/retrack_results_r2)
  REPORT.md                                the report (§4)
  figures/robustness_<dataset>.png
  for_claude.zip                           everything except checkpoints, for the analysis
  pipeline.log                             log of every pipeline run (appended)
  env_<date>.json                          settings and git commit of each pipeline run
  preflight/<dataset>_<variant>_s0/        smoke tests (PASSED marks success)
  runs/<dataset>_<variant>_s<seed>/
    lightning_logs/version_0/checkpoints/last.ckpt
    train.log
    DONE.json                              {"ckpt": ..., "minutes": ...} or {"reused_from": ...}
  eval/<dataset>_<variant>_s<seed>/
    metrics.csv  summary.json  all_cameras_frames.pt
MyDrive/retrack_videos/                    videos from video()
```

Code: `WorldTrack/models/query_decoder.py` (R2), `WorldTrack/models/segnet.py` (the model; R2 switch and camera mask),
`WorldTrack/world_track.py` (training, losses), `WorldTrack/evaluation/r2_eval.py` (evaluation), `colab/` (this
documentation and the tools).

---

## 9. Troubleshooting

| symptom | cause and fix |
|---|---|
| `No checkpoints found in: ...` | nothing under `MyDrive/retrack_results*` yet, or Drive is not mounted. Train first, or `checkpoints(roots='/content/drive/MyDrive/<your folder>')`. |
| `... matches 2 checkpoints` | the text given to a tool is not unique: use the number from the list or a longer part of the path. |
| `CheckpointNotSupported: ... R1 ... R4` | a methodology-5 R1/R4 checkpoint; it needs the branch it was trained on (`feat/research-methodology-5`). |
| `CheckpointNotSupported: weights do not match this code` | a checkpoint from a different code version; check which branch trained it. |
| `.../multiviewx.tar not found` | upload `multiviewx.tar` (top-level folder `MultiviewX/`) to `MyDrive/datasets/`. |
| the viewer does nothing when a camera button is clicked | it is re-running the model (status line in italics); on a CPU runtime this takes about a minute. Switch to a GPU runtime. |
| all cameras off → red message | at least one camera must stay on. |
| `CUDA out of memory` while training with `PARALLEL = 2` | the GPU has less than ~40 GB: set `PARALLEL = 1` and run the cell again (finished runs are kept). |
| training stopped (runtime disconnected) | run cell 1 again: it resumes from `last.ckpt`. |
| `training ... failed, see .../train.log` | the last 30 lines of that log are printed above the error; send them for a diagnosis. |
| nothing printed for a long time during training | normal: progress lines come once a minute; details are in `runs/<run>/train.log`. |
| the video is a GIF | no MP4 encoder in the runtime; GIFs work in slides too. |
| numbers differ slightly from an earlier run of the same checkpoint | GPU arithmetic is not bit-identical between GPU types; differences of ~1 error are expected. |

---

## 10. Terms

| term | meaning |
|---|---|
| BEV | bird's-eye view: the top-down map of the ground (10 cm cells) the model works on |
| B0 | the baseline: TrackTacular retrained in this code |
| R2 | sparse person queries instead of the dense BEV decoder |
| TP / FP / FN | detected person / false detection / missed person (matching radius 0.5 m) |
| MODA | detection accuracy: 1 − (FN + FP) / number of people, in % |
| MODP | how precisely detected people are localised, in % (100 = exactly on the annotation) |
| precision / recall | TP / (TP + FP) and TP / number of people |
| MOTA | tracking accuracy: like MODA but also counts ID switches (matching radius 1 m) |
| IDF1 | how consistently each person keeps one track ID over time |
| ID switch | a person's track changes identity |
| k | number of removed (switched-off) cameras |
| seed | random seed of a training run; different seeds give slightly different models |
| threshold | minimum score for a detection to count (0.5 in all reported numbers) |
| heatmap | the model's person score for every BEV cell |
| proposal / query (R2) | a candidate person started at a strong heatmap cell / the candidate after refinement |
