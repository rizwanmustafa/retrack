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
| **E0, plain retrained TrackTacular, 3 seeds** (m2 bundle) | 39.0 / 45.3 | 35.7 / 61.4 | 34.7 / 89.3 | 32.4 / 229.0 |
| FG0 (closest to B0) | 36 / 45 | 33.6 / 62.6 | 33.0 / 92.7 | 42.0 / 221.3 |
| A2x (SA-VM) | 26 / 47 | 24.1 / 78.1 | 22.1 / 116.4 | 35.3 / 179.3 |
| FGx | 27 / 65 | 24.7 / 90.7 | 23.3 / 126.6 | 33.3 / 210.3 |
| A2xT | 34 / 50 | 28.0 / 77.7 | 24.7 / 114.9 | 47.3 / 233.7 |

False positives stay flat or fall, while misses double. So explaining-away cannot explain the camera-removal loss. At
most it can remove the clean false positives (26 to 36, worth about 2.7 to 3.8 MODA if every one were a ghost).
The E0 row confirms this on the plain baseline, on every seed. The same pattern holds for all 9 models in the
methodology-2 bundle (`for_claude(1).zip`). On clean data most misses are near the threshold: for FG0, 26 of 45 score
0.3 to 0.5, but lowering the threshold to 0.3 adds about 50 false positives. Still to confirm on MultiviewX, and for 3
cameras removed, with the m5 `metrics.csv`.

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
  **Result, WILDTRACK B0 s80085 (2026-11-15, `diagnose/wildtrack_B0_s80085/summary.txt`)**, means per condition:

  | k removed | FP | FN | Extra FN vs k=0 | Missed with best score <0.1 / 0.1-0.3 / 0.3-0.5 | Best threshold (MODA) |
  |---|---|---|---|---|---|
  | 0 | 29.0 | 51.0 | | 16.0 / 7.0 / 25.0 | 0.50 (91.60) |
  | 1 | 30.1 | 67.9 | +17 | 16.7 / 10.9 / 36.3 | 0.45 (90.14, against 89.71 at 0.5) |
  | 2 | 29.0 | 92.7 | +42 | 18.6 / 21.5 / 48.8 | 0.40 (88.44, against 87.22) |
  | 3 | 28.4 | 133.3 | +82 | 25.9 / 36.8 / 65.8 | 0.40 (85.07, against 83.02) |

  - **False positives are flat once more, now on B0 itself.**
  - **About 88% of the extra misses at k=3 still get a response (score ≥ 0.1), and half of them score 0.3 to 0.5.** So it is mostly under-confidence, not missing evidence. Lowering the threshold recovers only part of it (+2 MODA at k=3) and adds false positives.
  - **Misses by working cameras at k=3:** 0 cameras 3.9, 1 camera 27.9, 2 cameras 48.1, 3 cameras 44.3, 4 cameras 9.1. About 24% of the misses are people seen by at most one working camera (outcome B). About 69% are people still seen by 2 to 3 working cameras (outcome A).
  - **The miss rate depends mainly on how many working cameras see the person:**
    - with all cameras: 2 cameras 93% (13/14), 3 cameras 6%, 4 cameras 1.8%;
    - at k=3: 1 camera 75%, 2 cameras 34%, 3 cameras 10%.

    B0 effectively needs about 3 views. In training, 816 of 952 test-frame people are seen by 5 or more cameras, so 2-view people are rare.
  **Result, MultiviewX B0 s80085** (6 cameras, 1494 person-frames):

  | k removed | FP | FN | Missed with best score <0.1 / 0.1-0.3 / 0.3-0.5 | Misses with ≤1 / 2 / 3 working cameras | Best threshold (MODA) |
  |---|---|---|---|---|---|
  | 0 | 11.0 | 52.0 | 3 / 9 / 10 | 0 / 3 / 21 | 0.40 (96.05, against 95.78) |
  | 1 | 13.7 | 82.8 | 5 / 18 / 31 | 1 / 34 / 19 | 0.45 (93.76, against 93.54) |
  | 2 | 17.5 | 141.9 | 13 / 42 / 58 | 38 / 55 / 29 | 0.40 (89.83, against 89.34) |
  | 3 | 22.9 | 266.1 | 43 / 78 / 115 | 119 / 104 / 43 | 0.40 (82.27, against 80.66) |

  - **Same pattern as WILDTRACK.** Misses dominate, about 84% of the missed people at k=3 still get a response, and lowering the threshold barely helps.
  - **Miss rate by working cameras at k=3:** 1 camera 88.5%, 2 cameras 21%, 3 cameras 5%. With all cameras, 2-view people are missed 3/3.
  - **New on MultiviewX: false positives do grow (11 to 23).** That is only about 0.8 MODA, against 14 MODA of extra misses.
  - **The two targets at k=3, as MODA points:**

    | Dataset | Seen by 2 to 3 working cameras (dropout target) | Seen by ≤1 working camera (architecture target) |
    |---|---|---|
    | WILDTRACK | 92 misses, about 9.7 MODA | 32, about 3.3 MODA |
    | MultiviewX | 147 misses, about 9.8 MODA | 119, about 8.0 MODA |
  - **Camera dropout already exists in the code:** `--model.cam_dropout_prob`, `--model.cam_dropout_max`, from methodology 2's CD control, which was never run.
  - **Implication:** camera-dropout training, which exposes the model to 2 to 3 view cases, may fix much of outcome A on its own. Run it first. Architecture remains a candidate for people seen by at most one camera (single-view localisation from foot points) and for occlusion.
- [ ] 3. Decide:
  - If ghosts are a large share of false positives, and growing as cameras are removed, build explaining-away fusion.
  - If not, drop the idea and go back to finding failures in the error breakdown.
- [ ] 4. Run a deeper prior-work search on explaining-away / occlusion-aware fusion for multi-view detection before writing the architecture.
- [ ] 5. Cheap upper bound: re-lift with ownership masks rendered from ground-truth cylinders. If even this oracle does not help, stop.
- [ ] 6. Fix the R4 tracker (Failure 2) and re-evaluate the 4 R4 checkpoints (about 5 minutes each).
- [ ] 7. Run 3 seeds for any variant we want to claim, B0 included.
- [ ] 8. Add MVDGC (and CaMuViD, PVH) to the positioning in `METHODOLOGY.md`.
- [ ] 9. Optional: retrain B0 with the temporal cache to close the gap to published TrackTacular (Failure 3).
