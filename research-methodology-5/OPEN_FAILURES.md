# Open failures and to-do list (research methodology 5)

Logged 2026-11-15, after the first 13 of 16 runs of the 2^3 factorial (seed 80085). This is the agreed next direction.
If the work moves elsewhere, check this list first.

## Results so far (all cameras, MODA / MOTA / IDF1)

| | WILDTRACK | MultiviewX |
|---|---|---|
| B0 | 91.49 / 88.97 / 93.07 | 95.78 / 91.10 / 80.27 |
| R1 | 90.13 / 86.97 / 91.75 | 96.18 / 90.90 / 82.18 |
| R2 | 91.81 / 91.70 / 95.19 | 94.31 / 90.76 / 85.48 |
| R4 | 89.81 / 88.34 / 89.86 [87.08 / 92.19] | 95.72 / 92.44 / 78.84 [91.16 / 80.32] |
| R12 | 87.71 / 88.24 / 93.92 | 94.51 / 90.16 / 81.99 |
| R14 | 91.70 / 90.23 / 90.40 [90.55 / 94.83] | pending |
| R124 | 90.34 / 88.87 / 91.17 [89.81 / 94.69] | 95.18 / 91.83 / 73.44 [90.96 / 83.98] |
| R24 | pending | pending |

Brackets: TrackTacular's tracker on the same detections. Only one seed was run; two earlier seeds of the baseline differed by about 1 point.

Novelty status:
- **None of R1, R2 or R4 is novel on its own.** They are close to MV2GF, MVTrackTrans, DETR3D, TransFusion and MVDGC respectively.
- **R2 is roughly a TransFusion-style query decoder on a camera-lifted BEV.**
- **MVDGC is ahead of every variant here on WILDTRACK.** It uses sparse cylinder queries and has no BEV (arXiv 2607.00273). Its reported scores are 94.2 / 92.1 / 95.8 on WILDTRACK and 95.7 / 92.1 / 84.1 on MultiviewX.
- **MVDGC never evaluates with cameras removed.**

## Failure 1 (main direction): ghost detections from smearing features along camera rays

**Observation (hypothesis, not yet measured).** The lifting step (TrackTacular, MVDet, EarlyBird) bilinearly samples each pixel's feature into every BEV cell along that camera's ray. One real person therefore raises the response along a whole line of cells in front of and behind them. MVDeTr calls these "shadows". With many cameras, the rays only agree at the true position. With fewer cameras the shadows no longer cancel and become false positives. If true, this explains the collapse under camera removal.

**Evidence against it (2026-11-15).** The methodology-4 camera-drop results (`for_claude.zip` of 2026-11-11,
`robust/*.csv`; WILDTRACK, seed 80085, 4 TrackTacular-family models) point the other way. Removing cameras raises
misses, not false positives:

| Model | Clean FP / FN | 1 camera off | 2 cameras off | Small calibration noise |
|---|---|---|---|---|
| FG0 (closest to B0) | 36 / 45 | 33.6 / 62.6 | 33.0 / 92.7 | 42.0 / 221.3 |
| A2x (SA-VM) | 26 / 47 | 24.1 / 78.1 | 22.1 / 116.4 | 35.3 / 179.3 |
| FGx | 27 / 65 | 24.7 / 90.7 | 23.3 / 126.6 | 33.3 / 210.3 |
| A2xT | 34 / 50 | 28.0 / 77.7 | 24.7 / 114.9 | 47.3 / 233.7 |

False positives stay flat or fall, while misses double. So explaining-away cannot explain the camera-removal loss. At
most it can remove the clean false positives (26 to 36, worth about 2.7 to 3.8 MODA if every one were a ghost).
Still to confirm on B0 and MultiviewX with the m5 `metrics.csv`.

**Proposed architecture: explaining-away fusion.** Person hypotheses compete for pixels. Iterate 2 to 3 times:
1. Detect people (R2's queries give positions and scores).
2. Render each person as a cylinder (about 0.5 m wide, 1.8 m tall) into every camera with a soft depth test, so each pixel belongs to the nearest confident person on its ray.
3. Lift the features again, weighted by that ownership: a pixel already explained by a nearer person contributes little to the cells behind it.

It is differentiable, needs no new labels, and adds only a few parameters. It reuses `ViewSampler` in `WorldTrack/models/query_decoder.py`.

**Prediction to test.** The gain over B0 should grow as cameras are removed.

**Prior work to separate from:**
- Deep-Occlusion (Baqué et al. 2017): generative model with a CRF, not end to end.
- MVDeTr: shadows handled by learned deformable attention, with no explicit ownership.
- PVH (CVPRW 2025): a visual-hull occupancy volume in one pass, with no competition between people.
- MVDGC: queries do not compete for pixels.
- Visibility-Gated ConvGRU (2026): gates temporal memory, a different mechanism.

## Failure 2: the R4 online tracker loses identities

In every R4 run, the IDF1 of `tracking/query_tracker.py` is below TrackTacular's tracker on the same checkpoint. The worst case is MultiviewX R124: 73.4 against 84.0. The R4 network itself seems to help, since TrackTacular's tracker on R14 and R124 gives the best IDF1.

The suspected cause is association. It matches detections only against the network's predicted track positions, and lost tracks compete equally with active ones.

This can be fixed at evaluation time, without retraining:
- Match the baseline way, using each detection's own previous-position offset.
- Use the network's prediction and presence score only to keep lost tracks alive and to break ties.

## Failure 3: our B0 retrain is below published TrackTacular

As listed in MVDGC's tables, published TrackTacular scores 93.2 / 91.8 / 95.3 on WILDTRACK and 96.5 / 91.8 / 83.4 on MultiviewX. Our B0 scores 91.49 / 88.97 / 93.07 and 95.78 / 91.10 / 80.27.

The likely cause is `--model.use_temporal_cache false`, though this is unconfirmed. Within this study, compare against our own B0, not against published numbers.

## To do (in order)

- [ ] 1. Get `MyDrive/retrack_results_m5/m5/for_claude.zip`. Read the camera-removal results (per k) and the factorial effects for all 16 runs.
- [ ] 2. **Motivating experiment (no training needed).** Measure what fraction of B0's false positives lie on a camera ray through a real person ("ghost ratio"), with all cameras and with k = 1 to 3 cameras removed. Use each run's `eval/*/all_cameras_frames.pt` and the dataset calibrations. Script still to be written.
  First check the false-positive / false-negative split per k (precision and recall in `metrics.csv`). Explaining-away
  mainly removes false positives. Published TrackTacular scores P 97.3 / R 95.8 on WILDTRACK and P 99.4 / R 97.1 on
  MultiviewX (MVDGC Table 2), so with all cameras misses dominate and the room for improvement is small (about 0.6
  MODA on MultiviewX). The idea pays off only if false positives grow as cameras are removed.
- [ ] 2b. **New, from the methodology-4 evidence above.** Find out why people are missed when cameras are removed:
  - Split the misses by how many working cameras see each person (the camera-count split methodology 4 planned but never ran, `research-methodology-4/code/miss_analysis.py`).
  - Check whether missed people still get a score just below 0.5, which would mean under-confidence rather than missing evidence.
- [ ] 3. Decide:
  - If ghosts are a large share of false positives, and growing as cameras are removed, build explaining-away fusion.
  - If not, drop the idea and go back to finding failures in the error breakdown.
- [ ] 4. Run a deeper prior-work search on explaining-away / occlusion-aware fusion for multi-view detection before writing the architecture.
- [ ] 5. Cheap upper bound: re-lift with ownership masks rendered from ground-truth cylinders. If even this oracle does not help, stop.
- [ ] 6. Fix the R4 tracker (Failure 2) and re-evaluate the 4 R4 checkpoints (about 5 minutes each).
- [ ] 7. Run 3 seeds for any variant we want to claim, B0 included.
- [ ] 8. Add MVDGC (and CaMuViD, PVH) to the positioning in `METHODOLOGY.md`.
- [ ] 9. Optional: retrain B0 with the temporal cache to close the gap to published TrackTacular (Failure 3).
