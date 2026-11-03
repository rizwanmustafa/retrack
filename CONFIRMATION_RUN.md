# Confirmation run: calibration-noise training over three seeds, plus noise-strength sensitivity

Purpose: confirm that calibration-noise training (Approach 1 in [`POTENTIAL_APPROACHES.md`](POTENTIAL_APPROACHES.md))
helps reliably, not just for one training seed. With seed 1, `concat_calib` (TrackTacular trained with calibration
noise) beat `concat` (TrackTacular) by 8–28 MODA under calibration errors and by 1.1 MODA with clean calibration.
The robustness gains are far beyond noise; the clean gain is not yet.

It also runs a **sensitivity analysis** of the noise strength. σ = 0.5° was chosen after looking at the calibration
sweep on the test frames, so on its own it could be challenged as a tuned number. Training the same model with
σ = 0.25° (`concat_calib_weak`) and σ = 1° (`concat_calib_strong`) shows how the result depends on that choice,
and whether weaker noise keeps the calibration robustness while reducing the cost with missing cameras.

## What the run does

1. Trains `concat` and `concat_calib` with seeds 1 and 2 (seed 80085 is already done and is skipped), and evaluates
   each model with cameras removed. The half-trained `concat` seed-1 model from an earlier interrupted run resumes
   from its last saved epoch.
2. Runs the calibration sweep for the new models. The summary covers all three seeds.
3. Runs the other camera faults (lag, frozen camera, single-camera drift) for the new models.
4. Trains and evaluates the two noise-strength variants (one seed each, 80085, the same seed as the existing
   `concat_calib`), runs the calibration sweep and camera faults for them too, and disconnects the runtime.

Time: about 4 h 15 min on an A100 (about 35 min per training run plus evaluation, 6 runs, then about 35 min of
sweeps). The three-seed confirmation runs first, so its results are on Drive after about 2 h 45 min even if the
session stops later. Turn on background execution (Runtime → Change runtime type) and close the tab if you like.

## Colab cell

```python
import os
from google.colab import drive
drive.mount('/content/drive')

BRANCH = 'feat/robustness-pilot'
if not os.path.exists('/content/retrack'):
    !git clone -q -b {BRANCH} https://github.com/rizwanmustafa/retrack.git /content/retrack
else:
    !git -C /content/retrack fetch -q origin {BRANCH} && git -C /content/retrack checkout -q {BRANCH} && git -C /content/retrack pull -q origin {BRANCH}
!pip install -q -r /content/retrack/requirements-colab.txt

RESULTS = '/content/drive/MyDrive/retrack_results'
DATA = '/content/data/Wildtrack'

# 1. Train and evaluate concat and concat_calib for seeds 1 and 2 (~2 h 20 min; seed 80085 is skipped)
!cd /content/retrack/WorldTrack && python retrack_pipeline.py --mode full --results {RESULTS} \
    --variants concat concat_calib --seeds 3

# 2. Calibration sweep for the new models; the summary covers all three seeds (~15 min)
!cd /content/retrack/WorldTrack && python -m evaluation.perturb_eval --mode calib_sweep \
    --results {RESULTS}/full --data_dir {DATA} --seed 80085 1 2

# 3. Other camera faults for the new models (~8 min)
!cd /content/retrack/WorldTrack && python -m evaluation.perturb_eval \
    --results {RESULTS}/full --data_dir {DATA} --seed 80085 1 2

# 4. Noise-strength sensitivity: sigma 0.25 and 1 degree, one seed each (~80 min training + evaluation)
!cd /content/retrack/WorldTrack && python retrack_pipeline.py --mode full --results {RESULTS} \
    --variants concat_calib_weak concat_calib_strong --seeds 1

# 5. Calibration sweep and camera faults for them; summaries are rewritten to include every model (~12 min)
!cd /content/retrack/WorldTrack && python -m evaluation.perturb_eval --mode calib_sweep \
    --results {RESULTS}/full --data_dir {DATA} --seed 80085 1 2
!cd /content/retrack/WorldTrack && python -m evaluation.perturb_eval \
    --results {RESULTS}/full --data_dir {DATA} --seed 80085 1 2

# 6. Disconnect when done (flush first so every file reaches Drive)
from google.colab import drive, runtime
drive.flush_and_unmount()
runtime.unassign()
```

Every step skips work that is already on Drive, so if the session is interrupted, run the cell again and it
continues. The notebook's console output is not copied to Drive: save the notebook (Ctrl+S) before closing it if you
want to keep it. The result files below are on Drive either way.

## Progress markers in the output

- `epoch N/30  train_loss …` — one line per training epoch.
- `===== Training concat_calib (seed 2, 30 epochs)` — a new training run started.
- `===== Evaluating …` — a model finished training and is being evaluated.
- `[concat_calib_s2] k_cameras 7 draw 4 …` — the last condition of that model's calibration sweep.

## Where the results go

All under `MyDrive/retrack_results/full/` on Google Drive:

| What | Location |
|---|---|
| Calibration sweep, all models and seeds (main file to check) | `perturb/calib_sweep/CALIB_SWEEP.md` |
| Calibration sweep raw numbers, one CSV per model and seed | `perturb/calib_sweep/<model>_s<seed>.csv` |
| Other camera faults | `perturb/SUMMARY.md` |
| Camera-removal evaluation per model | `eval/<model>_s<seed>/metrics.csv` |
| Full comparison report | `report/REPORT.md` |
| Checkpoints | `runs/segnet_<model>_s<seed>/` |

In `CALIB_SWEEP.md`, `concat` and `concat_calib` will show "3 seeds", and their "±" is then the standard deviation
over seeds; the other models have one seed and their "±" is over random draws.

## How to judge the result

Calibration-noise training is **confirmed** if, averaged over the three seeds, `concat_calib` still:

- loses clearly less than `concat` under roll/pitch σ = 0.5° (seed 1: −2.4 vs −13.1),
- beats `concat` by at least 5 MODA with every camera rotated 0.75° (seed 1: +8.4),
- does not lose accuracy with clean calibration (no worse than −0.5 MODA).

The clean-calibration **improvement** (+1.1 MODA with seed 1) counts as real only if it is larger than the spread
between seeds. Otherwise report it as "no loss with clean calibration".

Also check the **trade-off** seen with seed 1: `concat_calib` was worse with cameras removed (MODA with 3 cameras
removed 79.5 vs 82.9; Stage 4 table in `REPORT.md`) and with a frozen camera (−10.1 vs −6.3 in `SUMMARY.md`). If
this repeats over three seeds, report calibration-noise training as a trade-off: much more robust to calibration
errors, less robust to missing or frozen cameras.

**Noise-strength sensitivity** (compare `concat_calib_weak`, `concat_calib` and `concat_calib_strong`, seed 80085):

- If the robustness gain holds across σ = 0.25°–1° (smaller at 0.25°, similar or larger at 1°), the method does not
  depend on one tuned value, which is the answer to "why 0.5°?".
- If σ = 0.25° keeps most of the gain under roll/pitch σ = 0.5° and 1° errors **and** loses less with 3 cameras
  removed (`concat_calib` 79.5, `concat` 82.9), it is the better trade-off and should be the recommended setting.
- If σ = 1° costs clean accuracy or missing-camera robustness further, report that stronger noise over-regularises.

After the run, paste `CALIB_SWEEP.md` (and the `concat` / `concat_calib` rows of `REPORT.md`'s Stage 4 table, which
show the effect on missing cameras) for analysis.
