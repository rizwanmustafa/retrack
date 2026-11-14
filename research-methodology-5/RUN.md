# Running research methodology 5 on Colab

Before the first run, `MyDrive/datasets/multiviewx.tar` must exist, with a top-level `MultiviewX/` folder.
MultiviewX is only on OneDrive, so it can't be downloaded automatically. WILDTRACK is restored from
`MyDrive/datasets/wildtrack.tar`, or downloaded from Kaggle if that file is missing.

Runtime → Change runtime type → **A100 GPU**. Paste this into one cell and run it:

```python
!rm -rf /content/retrack && git clone -q -b feat/research-methodology-5 https://github.com/rizwanmustafa/retrack.git /content/retrack
try:
    %run -i /content/retrack/research-methodology-5/m5_colab.py
finally:
    from google.colab import drive, runtime
    drive.flush_and_unmount()
    runtime.unassign()
```

What happens:

1. Drive is mounted and both datasets are restored to `/content/data/Wildtrack` and `/content/data/MultiviewX`.
2. Dependencies are installed. The Depth Anything 3 code is cloned at a pinned commit, and the DA3-Base weights
   (about 0.5 GB) are downloaded once.
3. Preflight, a few minutes per variant: every variant trains 4 batches, is reloaded and is evaluated on 3 frames.
   B0 and R124 are also checked on MultiviewX.
4. 16 trainings run, two at a time: 8 variants × 2 datasets, in the order B0, R124, R1, R2, R4, R12, R14, R24, each
   on WILDTRACK and then MultiviewX. Each one is evaluated as soon as it finishes, on all cameras and every
   combination of up to 3 removed cameras.
5. `REPORT.md` and `for_claude.zip` are written to `MyDrive/retrack_results_m5/m5/`.

**Duration:** this has not been timed yet. Expect roughly 10–14 hours for all 16 runs. MultiviewX runs take longer
(50 epochs), and so do the DA3 variants. Re-running the same cell continues where it stopped: finished runs are
skipped, an interrupted run resumes from its last checkpoint, and a finished evaluation is skipped.

Settings you can put above the cell (before `%run`):

```python
DATASETS = 'wildtrack multiviewx'      # or just one
VARIANTS = 'B0 R124 R1 R2 R4 R12 R14 R24'
SEEDS = '80085'                        # confirmation run: SEEDS = '80085 1 2' with VARIANTS = 'B0 R124'
PARALLEL = 2                           # trainings at once on the A100
```

Check progress from the Colab terminal:

```bash
nvidia-smi
tail -n 5 /content/drive/MyDrive/retrack_results_m5/m5/pipeline.log
tail -n 3 /content/drive/MyDrive/retrack_results_m5/m5/runs/*/train.log
```

Share `m5/for_claude.zip` for the analysis. It contains the report, logs and metrics, but no checkpoints.
