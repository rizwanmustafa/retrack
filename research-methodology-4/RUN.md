# Running research methodology 4 on Colab

Runtime → Change runtime type → **A100 GPU**. Paste this into one cell and run it. It takes about 75 minutes.

```python
RUNS = 'A2x:80085 FGx:80085'
!rm -rf /content/retrack && git clone -q -b feat/research-methodology-2 https://github.com/rizwanmustafa/retrack.git /content/retrack
try:
    %run -i /content/retrack/research-methodology-4/M4_Colab.ipynb
finally:
    from google.colab import drive, runtime
    drive.flush_and_unmount()
    runtime.unassign()
```

What happens:

1. Drive is mounted and WILDTRACK is restored from `MyDrive/datasets/wildtrack.tar`. If that file is missing, the
   dataset is downloaded from Kaggle and the tar is saved for next time.
2. `code/fg_background.py` builds the background images (about 3 minutes, first session only). It prints the mask
   check per camera and shows `fg_preview.jpg` (image | background | mask).
3. `m2_pipeline.py --subdir m4` runs the preflight, trains A2x and FGx, evaluates them, runs the robustness benchmark
   and writes the report.

Outputs go to `MyDrive/retrack_results_m2/m4/`. Share `m4/for_claude.zip`, which holds the report, the logs,
`fg_check.csv` and `fg_preview.jpg`.

Check progress from the Colab terminal:

```bash
nvidia-smi
tail -n 5 /content/drive/MyDrive/retrack_results_m2/m4/pipeline.log
```

Session 2: the same cell with `RUNS = 'A2xT:80085 FG0:80085'` (SA-VM + temporal history; the gate alone). It also runs the miss analysis (`code/miss_analysis.py`).

Session 3: `RUNS = 'CD:80085 E0c:80085'` (camera-dropout control; TrackTacular retrained in the current code). No background images are needed, but the cell builds them anyway if they are missing.
