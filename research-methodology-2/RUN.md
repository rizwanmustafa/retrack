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
