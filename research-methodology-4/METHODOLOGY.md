# Research methodology 4: foreground-gated lifting on top of SA-VM

*Literature numbers are read from the PDFs in [`../research-methodology-2/papers/`](../research-methodology-2/papers/)
and cited as **[Pxx] + table/section**. Our own numbers come from the run reports
(`../research-methodology-2/METHODOLOGY.md` §8–10). Design choices that are not from a paper are labelled **our
choice**.*

> **Status, 2026-11-12.** Sessions 1 and 2 done (§8, §9). Neither add-on helps SA-VM. The gate lowers the scores of
> real people (F1 fails). Temporal history brings back the false detections that SA-VM removes (T1 fails). The gate
> alone removes only 5 false detections (F2 fails). SA-VM's gain is larger than the gate's (F3 passes). SA-VM (A2x)
> stays the strongest model. Session 3 (`CD:80085 E0c:80085`) runs the camera-dropout control and the current-code
> baseline that the SA-VM paper needs.

## 1. Idea

WILDTRACK's seven cameras are static [P17]. A per-camera background image therefore gives a foreground mask for every
frame by background subtraction. Before deep learning, this was the standard first step: WILDTRACK notes that "most of
the earlier methods operate per-frame and perform background subtraction preprocessing" [P17, §2.1]. POM, the classic
multi-view occupancy detector, takes background-subtraction images as input. Used as the detector, it scores only
23.2 MODA on WILDTRACK, even with a CNN segmenter [P17, Tab. 2; P18, Tab. 3].

We do not use the mask as the detector. We use it to **gate which image evidence is lifted into the voxel grid**.
TrackTacular lifts every image feature, including features of background pixels, into every voxel along its ray
[P01, P19]. SA-VM's clean-data gain over the identical retrain comes from fewer false detections (15.3 fewer per test
set, p = 0.020; methodology 3, §10.3). A foreground gate attacks the same error type from a different direction.

## 2. Method

1. **Background (once per dataset, our choice).** For each camera, the per-pixel median of the training frames: the
   first 90 % of the sequence, 360 annotated frames. Stored at 1/4 resolution (270 × 480). No test frame is used.
   Script: `code/fg_background.py`.
2. **Mask (inside the model, every image).**
   - The input image is averaged over 4 × 4 pixels.
   - The background is sampled at the same original-image positions. Training-time resize and crop are undone through
     the intrinsics, because they only scale and shift pixel coordinates.
   - d = mean absolute RGB difference. m = clamp((d − 0.05) / 0.10, 0, 1) (**our choice**).
   - m is area-resized to the feature map (H/8 × W/8) and widened by one cell with a 3 × 3 max-pool.
   - Code: `WorldTrack/models/foreground.py`.
3. **Gate.** The features that are lifted become f′ = f · (a + (1 − a) · m), with a = sigmoid(φ). φ is one learned
   scalar, starting at a = 0.5 (**our choice**).
   - The floor a means background evidence is damped, never deleted. If the gate does not help, training can raise a
     towards 1.
   - The image heads keep the ungated features, so per-view supervision is unchanged.
   - Code: `Segnet(fg_background=...)` in `WorldTrack/models/segnet.py`.
4. **Everything else is SA-VM unchanged:** scene flips, fixed cross-sample view mixing (A2x), the paper's schedule,
   and the paper's protocol.

The background is stored in the checkpoint as a buffer. Every evaluation script (arl_eval, robust_eval) therefore
works without changes, and checkpoints stay comparable with the SA-VM runs.

## 3. Variants

| variant | scene flips | view mixing | foreground gate | role |
|---|---|---|---|---|
| A2x | yes | fixed | no | SA-VM rebuilt in the current code (reference) |
| **FGx** | yes | fixed | yes | **proposed** |
| FG0 | no | no | yes | gate alone |
| E0c | no | no | no | baseline: the same network trained without scene flips or view mixing, current code |

## 4. Hypotheses and decision rules (fixed before the runs)

Screening uses seed 80085. A variant goes to seeds 2 and 1 only if it passes screening. The 3-seed rule is the same
as before: predicted sign on all 3 seeds and paired p < 0.05.

| ID | Hypothesis | Comparison | Passes screening if |
|---|---|---|---|
| F0 | The masks mark people, not background | `fg_check.csv`, training frames | mean mask inside person boxes ≥ 2× outside, on every camera |
| F1 | The gate adds to SA-VM | FGx − A2x, same seed | MOTA or IDF1 +1.0 or more, and MODA not below −0.5 |
| F2 | The gate alone removes false positives | FG0 − E0c, same seed | detection FP lower by 10 or more |
| F3 | SA-VM's gain is not just background suppression | (A2x − E0c) vs (FG0 − E0c) | SA-VM's MOTA gain larger than the gate's |
| F4 | The gate costs robustness | robust_eval, FGx vs A2x | measured, no threshold: reported either way |

Prediction for F4 (**ours**): the gate may hurt under calibration noise. A mask is aligned with its image, so a
misplaced view lifts sharp, wrong foreground evidence.

What each outcome means:

| outcome | meaning |
|---|---|
| F1 passes | C5 of the proposal: a cheap component that adds to SA-VM |
| F1 fails, F2 passes and F3 fails | SA-VM's gain is mostly background suppression, which a reviewer would ask about |
| F1 and F2 fail | the network's per-view image heads may already learn where people are; the gate is redundant |

## 5. Schedule

| session | `RUNS` | answers |
|---|---|---|
| 1 | `A2x:80085 FGx:80085` | F0, F1, F4 |
| 2 | `A2xT:80085 FG0:80085` | T1 (§7); F2 and F3 against attempt-1 E0 s80085 until E0c is run |
| 3–4 | `FGx:2 A2x:2`, then `FGx:1 A2x:1`, and the same for A2xT | F1 / T1 at 3 seeds, only if they passed screening (neither did) |
| 3 (new) | `CD:80085 E0c:80085` | H11 of methodology 3 (view mixing vs camera dropout); E0c replaces attempt-1 E0 in F2/F3 |

Each session is about 75 minutes on an A100: about 3 minutes for the background (first session only), about 5 minutes
of preflight, 2 × 31–32 minutes of training, about 2 minutes of evaluation and about 3 minutes of robustness.

## 6. Threats

- Static or slow people can enter the median background and be damped. The floor a > 0 and the learned φ limit the
  damage. `fg_check.csv` measures the mask inside person boxes.
- Thresholds 0.05 / 0.15, the 4 × 4 pooling and the one-cell widening are chosen, not tuned. They may be checked on
  training frames only (`fg_check.csv`), never on test frames.
- Lighting changes over the 200 s sequence are small, but they are not modelled.
- Novelty: none of our 20 downloaded papers gates lifted features with background-subtraction masks. That is a narrow
  check, and a wider literature search is needed before any claim.
- The background is computed from training frames of the same scene as the test frames. This is the classic setting
  of [P17], but it would not transfer to a new camera without its own background.

## 7. Where the misses are (attempt-1 detections, 2026-11-11) and amendment 1

SA-VM trades false detections for misses (methodology 3, §10.3). To see which misses, we re-matched attempt 1's
saved test detections (`for_claude.zip`, `eval/*/detections_test.pt`) at threshold 0.5 and 0.5 m, as the protocol.
Numbers are sums over seeds 1, 2 and 80085; 952 people per seed, 2,856 in total.

| | E0 (retrained) | A3 (+ temporal history) | A2 (SA-VM) |
|---|---|---|---|
| misses | 136 | 108 | 161 |
| false detections | 117 | 122 | 71 |
| miss rate, within 1 m of the area's edge (219 people) | 21.0 % | 24.2 % | 20.1 % |
| miss rate, interior (2,637 people) | 3.4 % | 2.1 % | 4.4 % |
| miss rate, nearest person closer than 0.75 m (765 people) | 6.0 % | 2.5 % | 7.6 % |
| missed with a detection scoring 0.3–0.5 within 0.5 m | 60 | 53 | 94 |
| missed with nothing above 0.1 within 0.5 m | 44 | 38 | 35 |
| single-frame misses (found in both neighbouring frames) | 53 | 46 | 53 |

What this shows:

- **SA-VM's extra misses are people it sees but under-scores.** Against E0 it has 25 more misses. It has 34 more
  near-threshold misses (score 0.3–0.5), 27 more interior misses and 12 more crowded misses. Lowering the threshold is
  not admissible, and on test frames it gained only 0.2 MODA because false detections come back.
- **Temporal history fixes crowded and interior misses.** A3 has 19 crowded misses against 46 for E0, and 55 interior
  misses against 90.
- **About a third of every model's misses are at the edge of the area** (44–53 of them), at a similar rate for all
  three. SA-VM does not cause these, and neither candidate targets them.

The camera-count split needs the annotations, so it runs on Colab (`code/miss_analysis.py`, step 6 of the notebook).

**Amendment 1: A2xT = SA-VM + temporal BEV history** (`--model.use_temporal_cache true`, attempt 1's A3).

| ID | Hypothesis | Comparison | Passes screening if |
|---|---|---|---|
| T1 | Temporal history removes SA-VM's extra misses without bringing back its false detections | A2xT − A2x, seed 80085 | misses lower by 5 or more, false detections up by fewer than that, MODA up |

Risk: attempt 1's all-in-one model (A1 occluders + A2 + A3) did not combine cleanly: 46, 63 and 64 misses over its 3
seeds. A2xT leaves out the occluders, which is the untested combination. Note: for temporal models, the mirror-TTA
condition of robust_eval is not valid, because the mirrored passes overwrite the temporal cache. TTA was dropped in
methodology 3 anyway.

Next candidate if T1 fails (not implemented): switch view mixing off for the last epochs, so that the heatmap is
recalibrated on correct geometry. This targets the near-threshold misses directly.

## 8. Session 1 results (2026-11-11)

Seed 80085, protocol thr0.5|tt. Source: `m4/REPORT.md`, `m4/fg_check.csv`, and the saved test detections re-matched as
in §7.

**A2x reproduces SA-VM in the current code.** Against attempt-1 A2 s80085: MODA +0.32, MOTA +0.00, IDF1 +0.02
(92.33 / 92.65 / 95.44). It is the reference for every comparison below.

**F0 passes.** Mean mask inside person boxes is 3.6–7.5× the mean outside, on every camera (inside 0.69–0.85,
outside 0.11–0.22). 15–29 % of each image is marked foreground. The preview shows that shadows and faint "ghosts" of
people who stood still during the training frames are also marked.

**F1 fails.**

| | A2x | FGx | FGx − A2x |
|---|---|---|---|
| MODA | 92.33 | 90.34 | −2.00 |
| MOTA | 92.65 | 90.55 | −2.10 |
| IDF1 | 95.44 | 94.30 | −1.14 |
| misses | 47 | 65 | +18 |
| false detections | 26 | 27 | +1 |
| misses with a detection scoring 0.3–0.5 within 0.5 m | 24 | 41 | +17 |
| misses within 1 m of the edge (73 people) | 14 | 18 | +4 |
| misses, interior (879 people) | 33 | 47 | +14 |
| single-frame misses | 17 | 24 | +7 |

The gate removes no false detections on top of SA-VM. It lowers the score of real people: almost all of the extra
misses (17 of 18) are found by FGx, but scored between 0.3 and 0.5. This is the same failure as SA-VM's own extra
misses (§7), made worse.

**F4: the gate costs robustness under small calibration noise only.** FGx − A2x: clean −2.00 MODA, calib_low −3.05,
calib_high +0.85, drop1 −1.38, drop2 −1.18. With camera drop, the loss is no larger than the clean loss. Under small
calibration noise it is about 1 point larger, as predicted in §4. All of this is one seed.

**Decision.** By the rule of §4, FGx is not run on seeds 2 and 1. FG0 (session 2) still decides how the negative
result is reported: if the gate alone also removes no false detections (F2 fails), the network's per-view image heads may
already learn where people are, and the gate is redundant in general, not only on top of SA-VM.

Not measured: the learned floor a. It is in the FGx checkpoint (`fg_floor`), which is not in the bundle.

## 9. Session 2 results (2026-11-12)

Seed 80085, protocol thr0.5|tt. Source: `m4/REPORT.md` and the saved test detections re-matched as in §7. E0 is
attempt-1 E0 s80085 (older code), because E0c has not been run.

| | E0 | FG0 | A2x | A2xT |
|---|---|---|---|---|
| MODA | 90.76 | 91.49 | 92.33 | 91.18 |
| MOTA | 89.08 | 90.13 | 92.65 | 90.34 |
| IDF1 | 93.40 | 94.54 | 95.44 | 94.26 |
| misses | 47 | 45 | 47 | 50 |
| false detections | 41 | 36 | 26 | 34 |
| false detections 2 m or more from any person | 19 | 14 | 8 | 18 |
| tracking false positives | 51 | 45 | 23 | 41 |
| misses within 1 m of the edge (73 people) | 14 | 14 | 14 | 22 |
| misses, interior (879 people) | 33 | 31 | 33 | 28 |
| misses with a detection scoring 0.3–0.5 within 0.5 m | 19 | 25 | 24 | 26 |
| single-frame misses | 14 | 14 | 17 | 23 |

E0's metrics are from attempt 1's `report/all_cells.csv` (`base_test_s80085`, thr0.5|tt).

**T1 fails.** A2xT − A2x: MODA −1.16, MOTA −2.31, IDF1 −1.18. Misses go up by 3, not down by 5, and false
detections go up by 8. Temporal history does what §7 predicted in the interior (5 fewer misses), but it adds 8 misses
at the edge of the area. It also brings back distant false detections, 2 m or more from any person: 18, against 8 for
A2x and 19 for E0. That is, the false detections that SA-VM removes come back. The tracker turns them into 41
tracking false positives, against 23. This repeats attempt 1, where the all-in-one model did not combine cleanly.

**F2 fails.** FG0 − E0: false detections −5 (41 → 36), not −10. The 5 are distant false detections (19 → 14). FG0
scores +0.74 MODA, +1.05 MOTA and +1.14 IDF1 over E0. That is inside attempt 1's seed spread of 0.3–1.5 points, and
the baseline is from older code.

**F3 passes.** Against E0, A2x gains +3.57 MOTA and removes 15 false detections. FG0 gains +1.05 MOTA and removes 5.
SA-VM's gain is not explained by background suppression.

So, by the table of §4: F1 and F2 fail. One explanation, not measured here: the network's per-view image heads already learn where people are, and a
background-subtraction gate adds little to them, alone or on top of SA-VM.

**Robustness (one seed, MODA).** A2xT − A2x: calib_low −6.97, calib_high −6.93, drop1 −0.36, drop2 −0.10. Temporal
history makes calibration noise worse, because a misplaced view is carried into the next frame through the cache.
FG0 against A2x: calib_low −5.11, calib_high +1.79, drop1 +0.64, drop2 +1.35. FG0's TTA collapse (31.30) is expected:
mirror TTA is not valid for a model trained without scene flips.

**Not measured: the camera-count split.** The notebook wrote the bundle in step 5, before the miss analysis of step 6,
so `misses.csv` is on Drive (`m4/misses.csv`) but not in the zip. The notebook now adds `misses.csv` and `misses.log`
to the bundle after step 6.

**Decision.** Neither A2xT nor FG0 goes to more seeds. Methodology 4 ends with a negative result: two cheap add-ons,
neither of which helps SA-VM, with the reason for each. Session 3 is `CD:80085 E0c:80085`:

- CD is the camera-dropout control of methodology 3 (H11), which the SA-VM paper's Table 2 lists and which has never
  been run.
- E0c is the baseline (the same network, trained without scene flips or view mixing) in the current code. It
  replaces attempt-1 E0.

