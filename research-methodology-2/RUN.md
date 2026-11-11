# Running research methodology 2

## Colab (one cell, about 45 minutes on an A100)

Runtime → Change runtime type → **A100 GPU**. Then run:

```python
!rm -rf /content/retrack && git clone -q -b feat/research-methodology-2 https://github.com/rizwanmustafa/retrack.git /content/retrack
%run -i /content/retrack/research-methodology-2/M2_Colab.ipynb
```

The notebook does the following:

1. Mounts Drive.
2. Restores WILDTRACK from `MyDrive/datasets/wildtrack.tar`, which attempt 1 saved there. If the file is missing, it
   downloads WILDTRACK from Kaggle.
3. Installs `requirements-colab.txt`.
4. Runs the steps below.

| step | time |
|---|---|
| preflight: 4 training batches plus one full evaluation, to catch crashes early | about 3 min |
| training M2, seed 80085 | about 35–40 min |
| evaluation | about 1 min |
| `REPORT.md` and `for_claude.zip` in `MyDrive/retrack_results_m2/m2/` | — |

If Colab disconnects, run the same cell again. Finished steps are skipped, and training resumes from `last.ckpt`.

To change what runs, define the settings before `%run -i`:

```python
RUNS = 'M2:80085 M2c:80085'   # add GRACE's coverage prior as an ablation
PARALLEL = 2                  # both on the GPU at once (not timed; may exceed an hour)
```

Variants: `M2` (proposed), `M2c` (GRACE coverage prior), `A2f` (scene augmentation only, bug fixed), `VD` (no
scene augmentation). The priority order is in `METHODOLOGY.md` §5.

## Without Colab

```bash
python research-methodology-2/code/m2_pipeline.py --results /path/to/results --data_root /path/to/data --runs M2:80085
python research-methodology-2/code/m2_report.py --results /path/to/results/m2     # rebuild the report only
```

Afterwards, share `m2/for_claude.zip` for analysis.

## Methodology 3 (SA-VM), session 1: about 70 minutes on an A100, about 50 on an H100

```python
RUNS = 'RVM:80085 A2f:2'
!rm -rf /content/retrack && git clone -q -b feat/research-methodology-2 https://github.com/rizwanmustafa/retrack.git /content/retrack
try:
    %run -i /content/retrack/research-methodology-2/M2_Colab.ipynb
finally:
    from google.colab import drive, runtime
    drive.flush_and_unmount()
    runtime.unassign()
```

After training, the pipeline runs the robustness evaluation, `evaluation/robust_eval.py`. It covers clean, mirror TTA,
calibration noise and camera failures. It evaluates every checkpoint in `MyDrive/retrack_results_m2/m2/runs`, plus
attempt 1's E0 and A2 checkpoints in `MyDrive/retrack_results_8nov/arl_full`.

If attempt 1's folder is elsewhere, set the following before `%run`:

```python
EXTRA = '--attempt1 /content/drive/MyDrive/<folder>/arl_full'
```

If an hour is the hard limit, use `RUNS = 'A2f:2'`. That still gives the 3-seed robustness result for E0 vs SA-VM.

## Methodology 3, session 2: about 75 minutes on an A100

Session 1 is done (METHODOLOGY.md §10). Same cell as session 1, with:

```python
RUNS = 'A2x:80085 CD:80085'
```

A2x reproduces SA-VM in the current code. CD is the camera-dropout control. The robustness stage only evaluates the 2
new checkpoints, because existing CSVs are skipped. If an hour is the hard limit, use `RUNS = 'A2x:80085'`.
