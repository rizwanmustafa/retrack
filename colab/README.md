# Colab guide: TrackTacular (B0) vs R2, and the visual tools

Quick start below. Full documentation of every tool, setting, figure and output file: **[TOOLS.md](TOOLS.md)**.

Branch `feat/r2-sparse-queries` = TrackTacular (WorldTrack code) + one change, **R2**: the dense BEV decoder is replaced
by sparse person queries (`WorldTrack/models/query_decoder.py`). Nothing else from the earlier research branches is here.

Datasets are read from Google Drive (`MyDrive/datasets/wildtrack.tar`, `MyDrive/datasets/multiviewx.tar`, each with
one top-level folder `Wildtrack/` or `MultiviewX/`). WILDTRACK is downloaded from Kaggle and saved there if missing.

## Cell 1: train and evaluate

```python
!rm -rf /content/retrack && git clone -q -b feat/r2-sparse-queries https://github.com/rizwanmustafa/retrack.git /content/retrack
SEEDS = '80085 1 2'          # optional settings, see colab/train_eval.py
try:
    %run -i /content/retrack/colab/train_eval.py
finally:
    from google.colab import drive, runtime
    drive.flush_and_unmount()
    runtime.unassign()       # frees the GPU when done; remove these 3 lines to keep the runtime
```

- Trains B0 and R2 on WILDTRACK (30 epochs) and MultiviewX (50 epochs) for every seed, then evaluates each with all
  cameras and with every combination of up to 3 cameras removed.
- The seed-80085 runs of research methodology 5 (`MyDrive/retrack_results_m5/m5`) are reused if that folder exists
  (same architecture), so only seeds 1 and 2 are trained: 8 trainings, about 4–5 hours on a G4/A100 with 2 in parallel.
- Outputs: `MyDrive/retrack_results_r2/` — `REPORT.md`, `figures/robustness_*.png`, `runs/`, `eval/`, and
  `for_claude.zip` (everything except checkpoints) for the analysis.
- Interrupted? Run the same cell again: finished trainings and evaluations are skipped, a half-done training resumes.

## Cell 2: visual tools (any GPU, also a T4)

```python
import os
if not os.path.exists('/content/retrack'):
    !git clone -q -b feat/r2-sparse-queries https://github.com/rizwanmustafa/retrack.git /content/retrack
%run -i /content/retrack/colab/viz.py
```

This mounts Drive and prints every checkpoint it finds in `MyDrive/retrack_results*` (numbered, with variant,
dataset, seed, epoch and whether it has been evaluated). R1/R4 checkpoints of methodology 5 are listed but cannot be
opened here. Then, one call per cell:

| call | what you get |
|---|---|
| `explore(3)` | the main viewer: BEV of one test frame with ground truth, detections, misses and false detections; the 6–7 camera images with the same people as boxes; a frame slider with **Play**; one button per camera (click to switch it off — the model re-runs with that camera missing); threshold slider; background = model heatmap or *how many working cameras see each cell*; track IDs; R2's queries |
| `compare(1, 3)` | two checkpoints (e.g. B0 and R2, same dataset) side by side, same frame and cameras |
| `camera_evidence(3, frame=10)` | for every camera: heatmap with all cameras minus heatmap without it — which people depend on which camera |
| `tracks(3, drop=[0, 4])` | all trajectories of the 40 test frames, ID switches as red stars (here with C1 and C5 off) |
| `robustness(1, 3)` | MODA / MOTA / IDF1 / false detections / misses against the number of removed cameras |
| `video(3, drop=[2])` | MP4 of the viewer over all frames, saved to `MyDrive/retrack_videos/` (for presentations) |
| `checkpoints()` | list again; `checkpoints(roots=['/content/drive/MyDrive/some/folder'])` to look elsewhere |

Checkpoints can be given by number, path, or a unique part of the path: `explore('multiviewx_R2_s1')`.
In arguments cameras are numbered from 0 (`drop=[0]` switches off C1).

Reading the BEV: white dots = ground truth, green = found, **orange circle = missed** (the number next to it = how
many working cameras see that person; the baseline needs about 3), **red x = false detection**, cyan numbers = track
IDs, triangles = cameras with their field of view outlined. The text under the figure lists every miss (with the
cameras that see the person and the best score near them) and every false detection of the frame.
