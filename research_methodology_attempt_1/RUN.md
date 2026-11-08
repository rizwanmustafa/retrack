# Running ARL attempt 1

## One Colab cell

1. Open `ARL_Colab.ipynb` in Colab (File → Open notebook → GitHub → `rizwanmustafa/retrack`, branch
   `feat/arl-attempt-1`), or paste its single cell into a new notebook.
2. Runtime → Change runtime type → **A100 GPU**, High-RAM.
3. Run the cell with `MODE = 'quick'` first (about 2 variants × 2 trainings × 2 epochs), check that
   `report/REPORT.md` appears, then set `MODE = 'full'` and run it again.

If Colab disconnects, run the same cell again: finished steps are skipped and training resumes from `last.ckpt`.

| setting | what runs | trainings |
|---|---|---|
| `MODE='quick'` | variants `base`, `A1A2A3`, 1 seed, 2 epochs | 4 |
| `MODE='full'`, `VARIANTS='all'`, `SEEDS=3` | 9 variants, 3 seeds, 30 epochs | 36 (9 × (3 + 1 validation model)) |
| `MODE='full'`, `VARIANTS='core'`, `SEEDS=3` | base, A1, A2, A3, A1A2A3 | 20 |

The README measured about 35 min of training per 30-epoch run on an A100, so `all` is roughly 21 h and `core`
roughly 12 h. Evaluation adds a few minutes per model (one camera condition, plus the post-hoc grids).
`PRECISION = 'bf16-mixed'` is faster but differs from the original schedule.

## How each addition is tested

| addition | how it is switched | compared by |
|---|---|---|
| A1 occlusion, A2 scene augmentation, A3 temporal history | one trained variant per combination: `base`, `A1`, `A2`, `A3` (add-one), `A1A2A3` (full), `A2A3`, `A1A3`, `A1A2` (leave-one-out) | Welch t-test over seeds |
| A5 weight averaging | `A1A2A3A5` vs `A1A2A3` | Welch t-test over seeds |
| B1 threshold, C1 online tracker, R appearance, C2 linking, C2b flow, C3 interpolation | every on/off combination on the cached detections of the same model (28 cells) | paired t-test over seeds: add-one, leave-one-out and average marginal effect |

Decision rule (fixed in advance): **KEEP** if a target metric improves by more than one baseline standard deviation
over seeds and no primary metric (MODA, MOTA, IDF1) drops by more than one standard deviation. With one seed the
verdict is "inconclusive", but the deltas are still reported. Parameters are tuned twice. **clean** uses a
validation model (frames 0–323, tuned on 324–359). **oracle** tunes on the test frames and is labelled as such.

## What ends up on Google Drive (`<RESULTS>/arl_<mode>/`)

```
runs/<variant>_<split>_s<seed>/   Lightning logs, config.yaml, last.ckpt (A5: averaged.ckpt), DONE.json
eval/<variant>_<role>_s<seed>/     detections_<split>.pt (cached detections), params*.json, metrics.csv, info.json
report/REPORT.md                   ladder E0-E12 + one verdict per addition; ladder.csv, verdicts.csv,
                                   all_cells.csv, additions.png
best/best_model.ckpt               best checkpoint + _params.json + .json model card; best/per_variant/*.ckpt
for_claude.zip                     everything above except checkpoints (to share for analysis)
MANIFEST.md, env_*.json, pipeline.log
```

To get the results analysed, share either the `arl_<mode>` folder or just `for_claude.zip`. It contains every
metric, the cached detections, the tuned parameters, the training logs and the exact git commit. With these, all
post-hoc analysis can be redone without a GPU.

## Without Colab

```bash
cd WorldTrack
python arl_pipeline.py --results /path/to/results --mode quick
python arl_pipeline.py --results /path/to/results --mode full --seeds 3 --variants core avg
python -m evaluation.arl_report --results /path/to/results/arl_full     # rebuild the report only
```

## Not implemented in attempt 1

A4 (GRACE's volumetric-guided fusion + ray conditioning, or MVFP's large-kernel refiner) is not in the code.
The GRACE paper does not specify its compression and fusion networks (ρ, ψ_vol), so it cannot be reproduced
faithfully from the paper alone. The methodology already runs A4 only after A1–A3 are confirmed.
