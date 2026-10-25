# Potential approaches: robustness to camera calibration errors

Status: **proposal, nothing here is implemented yet.** The full robustness pilot confirmed the problem (see the
findings below and the decision gate at the end).

## Why this document exists

The Stage 4 experiments (camera-dropout training, masked fusion, coverage map, learned per-camera weighting) did not
improve on TrackTacular's concatenation fusion. With one seed, averaged over every combination of removed cameras:

| Variant | MODA, all cameras | MODA, 3 cameras removed |
|---|---|---|
| concat (baseline) | 90.9 | 82.9 |
| concat + camera dropout | 91.4 | 83.0 |
| masked + dropout | 90.4 | 81.9 |
| masked + coverage + dropout | 90.6 | 81.7 |
| weighted + coverage + dropout | 89.1 | 82.0 |

The baseline already tolerates *missing* cameras well. The robustness pilot (`evaluation/perturb_eval.py`,
evaluation only, all 40 test frames, seed 1) tested other camera faults:

| Fault (TrackTacular baseline, clean MODA 90.9) | ΔMODA | Worse than dropping that camera |
|---|---|---|
| Camera dropped (reference) | −1.4 | — |
| 1 camera rotated 0.5° / 1° / 2° (model keeps the old calibration) | −0.5 / −2.3 / −3.7 | 2/7 / 5/7 / 6/7 cameras |
| 1 camera 1 / 2 frames late (0.5 / 1 s) | −3.2 / −4.1 | 5/7 / 6/7 |
| 1 camera frozen on one image | −6.3 | 7/7 |
| All cameras rotated 0.5° | −5.8 | — |
| **All cameras rotated 1°** | **−35.4** (MODA 55.5) | — |

Findings:

1. **A wrong camera is worse than a missing one.** A frozen camera does more damage than switching it off for every
   camera; 1–2° of drift or 1–2 frames of lag does for most cameras. The model handles missing input well but trusts
   bad input.
2. **All-camera calibration error has a cliff between 0.5° and 1°** (−6 → −35 MODA). Almost all of the loss is
   missed people: recall falls from 95% to 61% while precision stays above 90%, consistent with misaligned views no
   longer reinforcing each other.
3. **None of the Stage 4 fusion variants fixes it.** At all-camera 1°: concat −35.4, concat + dropout −40.1,
   masked + dropout −31.6, masked + coverage + dropout −27.0, weighted + coverage + dropout −33.4. Masked fusion is
   somewhat less sensitive, but this is one seed and one set of random rotation axes.
4. **The ByteTrack-style tracker (Stage 2) helps under drift**: at all-camera 1° it raises the baseline's MOTA from
   57.4 to 64.1, and on masked fusion IDF1 from 69.0 to 78.8, by recovering weakened detections. With clean cameras
   it is slightly worse than the original tracker.

Caveats: one training seed; each rotation condition used a single random axis per camera. The calibration sweep
below repeats the calibration error with several random draws, finer angles and the published roll/pitch protocol.

### Calibration sweep (`--mode calib_sweep`, 5 random draws per setting)

Every model sees exactly the same perturbations, so models can be compared draw by draw. Reprojection error is the
median pixel error the wrong calibration causes at full resolution; WILDTRACK's own calibration already has 2–17 px.

**Where the baseline breaks** (TrackTacular concat, every camera rotated by the same angle):

| Error per camera | Reprojection error | Ground shift | ΔMODA (mean ± std) |
|---|---|---|---|
| 0.25° | 7 px | 17 cm | −0.9 ± 0.4 |
| 0.5° | 14 px | 32 cm | −3.7 ± 2.8 |
| 0.75° | 21 px | 41 cm | −17.9 ± 11.1 |
| 1° | 29 px | 68 cm | −47.4 ± 18.1 |
| 1.25° | 34 px | 73 cm | −61.9 ± 9.5 |

- The cliff lies between about **14 and 21 px** of additional error, just above the error a real installation
  already has. Loss is again mostly recall (people missed), precision stays high.
- **Published roll/pitch protocol**: σ = 0.25° (6 px) −0.9; **σ = 0.5° (9 px median) −13.1 ± 12.3**; σ = 0.75°
  −24.6; σ = 1° −25.5. The large spread comes from draws in which one or two cameras get a large tilt.
- **How many cameras are wrong** (1° each): k = 1 −2.4, k = 2 −3.4, k = 3 −14.5, k = 4 −19.5, k = 5 −23.2,
  k = 6 −20.4, k = 7 −45.9. The model tolerates one or two miscalibrated cameras and degrades from three on.

**Masked fusion is consistently more robust** (ΔMODA, same perturbations):

| Condition | concat | concat + dropout | masked + dropout | masked + coverage + dropout | weighted + coverage + dropout |
|---|---|---|---|---|---|
| All cameras 0.75° | −17.9 | −19.9 | **−12.9** | −13.9 | −14.3 |
| All cameras 1° | −47.4 | −49.5 | **−40.1** | −41.7 | −47.5 |
| Roll/pitch σ = 0.5° | −13.1 | −14.8 | −9.4 | **−9.1** | −12.4 |
| 3 cameras at 1° | −14.5 | −16.0 | −10.1 | **−8.6** | −11.8 |
| 5 cameras at 1° | −23.2 | −26.0 | −15.8 | **−16.6** | −20.7 |
| All 7 cameras at 1° | −45.9 | −50.3 | −38.1 | **−35.8** | −41.6 |

Draw by draw, masked + dropout beat concat + dropout in all 20 draws checked (all-camera 0.75° and 1°, 3 and 7
cameras at 1°). Camera dropout alone does not help (concat + dropout is slightly worse than concat); averaging
over the working cameras is what helps.

**The ByteTrack-style tracker (Stage 2) recovers part of the loss** under large errors: baseline MOTA 49.7 → 54.1 at
1° on all cameras, 50.9 → 57.1 with all 7 cameras at 1°, 68.2 → 73.5 at roll/pitch σ = 1°.

Calibration errors are a real deployment problem: cameras get bumped, mounts move with wind and heat, and
recalibration is never perfect. WILDTRACK's own calibration is not exact either: projected ground positions land
2–17 px from the annotated feet, depending on the camera (`python -m evaluation.diagnose`).

## How a calibration error hurts the model

Every camera's image features are lifted onto the shared ground-plane grid using that camera's calibration. If the
calibration is wrong, the camera's features land in the wrong cells, so the views of one person no longer line up, and
the fused heatmap peak weakens or splits. A wrong extrinsic shifts a camera's footprint by roughly
`distance × angle`: at 20 m, 1° is about 35 cm, close to the 50 cm detection radius used by MODA.

## Related work

- Robustness benchmarks for bird's-eye-view perception include camera faults such as "Camera Crash" and "Frame Lost":
  [RoboBEV](https://arxiv.org/abs/2304.06719), extended in [arXiv:2405.17426](https://arxiv.org/abs/2405.17426).
- Extrinsic perturbation as a test protocol (random roll/pitch noise on the extrinsics), and training with the same
  noise: [BEVHeight](https://arxiv.org/abs/2303.08498), [BEVHeight++](https://arxiv.org/abs/2309.16179),
  [CoBEV](https://arxiv.org/abs/2310.02815), [BEVSpread](https://arxiv.org/abs/2406.08785).
- Learned compensation for extrinsic perturbations: [NCGR](https://arxiv.org/abs/2608.03895),
  [RECO](https://arxiv.org/abs/2607.20947).

These are driving and roadside 3D detection papers. A first search did not find this kind of evaluation for multi-view
pedestrian detection and tracking on WILDTRACK; confirm that with a proper literature review before claiming it as a
gap.

---

## Approach 1: Train with calibration noise

**Idea.** During training, randomly perturb the calibration the model receives while keeping the images unchanged, so
the model learns not to depend on pixel-perfect alignment across views.

**Implementation.**
- In `WorldTrackModel.forward` (`world_track.py`), when `self.training`, multiply each camera's extrinsic by a small
  random rotation before calling the model, exactly as `evaluation/perturb_eval.py` does at test time (reuse its
  `rotation()` helper).
- New hyperparameters, e.g. `calib_noise_deg` (standard deviation of roll/pitch/yaw noise) and `calib_noise_prob`
  (fraction of samples or cameras perturbed), plus a config `configs/v_concat_calib.yml`.
- Use the same noise model as the published protocol (Gaussian roll/pitch noise, as in BEVHeight) so results are
  comparable and the magnitudes are not arbitrary.

**Cost.** About 20 lines of code; one ~30-minute training run per seed.

**Status: implemented** (`WorldTrackModel.perturb_calibration` in `world_track.py`, hyperparameters
`calib_noise_deg` and `calib_noise_prob`). With probability `calib_noise_prob` a training sample gets a random subset
of 1–7 cameras rotated by Gaussian pitch and roll with standard deviation `calib_noise_deg`, the same convention as the
evaluation protocol. Two variants, each differing from an existing model only by the noise:
`configs/v_concat_calib.yml` (vs `concat`) and `configs/v_masked_calib.yml` (vs `masked_drop`), both with
σ = 0.5° and probability 0.5. σ = 0.5° is where the roll/pitch sweep starts to hurt (median 9 px).

**Expected outcome.** Partial tolerance to small errors. It may cost a little accuracy with clean calibration,
similar to how camera-dropout training trades a little clean accuracy for robustness.

**Risks.** It cannot correct large errors, only tolerate them. Too much noise blurs the ground-plane features and
lowers clean accuracy.

**Evaluation.** `perturb_eval` rotated-camera faults (one camera and all cameras) against the baseline, plus the clean
score, over 3 seeds.

**Result (seed 1): works.** Bar agreed before the run: halve the loss under roll/pitch σ = 0.5°, gain at least 5 MODA
with all cameras rotated 0.75°, lose at most 0.5 MODA with clean calibration.

| Criterion | `concat` → `concat_calib` | `masked_drop` → `masked_calib` |
|---|---|---|
| ΔMODA under roll/pitch σ = 0.5° | −13.1 → **−2.4** ✅ | −9.4 → **−2.2** ✅ |
| MODA, all cameras 0.75° | 72.9 → **81.3 (+8.4)** ✅ | 77.5 → 82.1 (+4.6) ❌ just misses |
| MODA, clean calibration | 90.9 → **92.0 (+1.1)** ✅ | 90.4 → 89.7 (−0.7) ❌ just misses |

Under larger errors the gain grows: all cameras 1° 43.5 → 67.0; roll/pitch σ = 1° 65.4 → 85.9; all 7 cameras at 1°
45.0 → 73.4. The spread between random draws also collapses (σ = 0.5°: ±12.3 → ±1.5), so one badly tilted camera no
longer ruins the result. With clean calibration `concat_calib` is the best model so far (MODA 92.0, MOTA 91.2,
IDF1 95.2, close to the paper's 92.1 / 91.8 / 95.3); a plausible reason is that WILDTRACK's own calibration already
has 2–17 px of error, so tolerating small misalignment helps on clean data too. Once trained with calibration noise,
masked fusion is no more robust than concatenation (1°: 67.7 vs 67.0) and is lower on clean data.

**Trade-off: calibration-noise training makes the model worse with missing or frozen cameras.**

| Condition (MODA) | `concat` | `concat_calib` | `masked_drop` | `masked_calib` |
|---|---|---|---|---|
| 1 camera removed (avg over 7) | 89.5 | 89.1 | 88.9 | 87.3 |
| 2 cameras removed (avg over 21) | 87.2 | 85.6 | 86.2 | 83.3 |
| 3 cameras removed (avg over 35) | 82.9 | **79.5** | 81.9 | **76.4** |
| MODA lost per removed camera (slope) | −2.6 | −4.1 | −2.8 | −4.4 |
| 1 camera frozen (ΔMODA) | −6.3 | **−10.1** | −6.5 | −7.1 |
| 1 camera 1 frame late (ΔMODA) | −3.2 | −3.9 | −3.2 | −4.0 |
| 1 camera rotated 1° (ΔMODA; worse than dropping it) | −2.3 (5/7) | **−1.4 (0/7)** | −1.2 (3/7) | −1.2 (1/7) |

With 3 cameras removed `concat_calib` loses 3.4 MODA more than the baseline, and `masked_calib` 5.5 more than
`masked_drop` even though it is also trained with camera dropout, so dropout does not prevent the trade-off. A
possible explanation (not tested): learning to distrust misaligned views makes the model rely on agreement between
many cameras, which hurts when fewer cameras remain. Options to reduce it: weaker noise (σ = 0.25° or a lower
probability), or pairing the noise model with detection of dropped/frozen cameras (Approach 2) so that the
remaining cameras are not misread.

Caveat: one seed. The robustness gains (+8 to +28 MODA) are far beyond noise; the +1.1 clean gain and the
missing-camera cost (−3.4 at k = 3) are not yet. A confirmation run with two more seeds of `concat` and
`concat_calib` is next ([`CONFIRMATION_RUN.md`](CONFIRMATION_RUN.md)).

## Approach 2: Detect and drop the faulty camera

**Idea.** A wrong camera is worse than a missing one (about −20 MODA vs −1 to −2 for a dropped camera). If a camera
disagrees with the others, switch it off and let the remaining views carry the detection.

**Implementation.**
- Per-camera consistency score: the model already predicts a foot heatmap for every image (`img_center` in
  `models/decoder.py`). Project the fused ground-plane detections into each camera with its calibration and measure
  how well they match that camera's own foot detections (match rate and median pixel distance). Alternatively,
  project each camera's foot detections onto the ground plane and match them to the fused detections.
- Accumulate the score over a sliding window of frames, so one crowded frame cannot trigger a false alarm.
- If a camera is a clear outlier compared with the others, set its entry in `cam_mask` to 0. The masking path already
  exists from Stage 4; `concat_drop` handles a missing camera at almost no cost.
- Runs at test time inside the tracking loop; no retraining.

**Cost.** Moderate engineering (scoring, windowing, thresholds), no training.

**Easy first case: frozen cameras.** A frozen camera sends an identical image every frame, which is trivial to
detect from frame differences. Dropping it would recover about 5 MODA (−6.3 → −1.4 for the baseline).

**Expected outcome.** Recovers most of the loss when **one** camera is miscalibrated, lagging or frozen: turns
"wrong camera" into "dropped camera". The same detector handles lag and frozen cameras.

**Risks.** It cannot help when **all** cameras drift, because there is no reliable reference left. Thresholds need
care: a camera that sees few people produces a noisy score.

**Evaluation.** `perturb_eval` single-camera faults (rotated, late, frozen), compared with the baseline and with an
"oracle" that drops the faulty camera. Report how often the faulty camera is correctly detected (and false alarms on
clean data).

## Approach 3: Self-calibration from pedestrians

**Idea.** People walk on the ground plane and are seen by several cameras at once, so they act as moving calibration
targets. While the system runs, estimate a small correction for each camera that makes its detections agree with the
others, so the system notices that a camera has shifted and fixes its own calibration.

**Implementation.**
- Parameters: a small rotation correction δRₙ per camera (3 values; optionally a translation too).
- Correspondences: each camera's foot detections (`img_center` peaks) matched to the fused ground-plane detections
  (or to other cameras' foot points projected to the ground).
- Objective: robust (Huber) reprojection error between each camera's foot points and the matched ground points
  projected with the corrected calibration `δRₙ · Eₙ`, accumulated over many frames. Optimise with a few
  Gauss–Newton or Adam steps in PyTorch (a handful of parameters per camera).
- Alternate: fuse with the current calibration → re-estimate corrections → fuse again; update slowly over time in
  the live system.
- When all cameras drift, the problem has a global ambiguity (the whole rig can move together). Fix it by holding one
  reference camera, or the ground plane, fixed, and regularise corrections towards zero.

**Cost.** The most work and the highest research risk: matching, robust optimisation and convergence checks.

**Expected outcome.** Fixes the cause rather than tolerating it, including when all cameras drift. It makes the live
system self-calibrating, which is a strong demo and a strong project contribution.

**Risks.** Wrong matches in crowded scenes can pull the estimate the wrong way. It needs enough people in shared
views, and it may converge slowly. Check prior work on calibrating cameras from walking people before designing it.

**Evaluation.** Starting from perturbed calibrations, measure how close the recovered rotations get to the true ones
(error in degrees), and MODA/MOTA before and after correction, as a function of the error size and of how many frames
are used.

## Approach 4: Learned alignment

**Idea.** Let the network learn to correct misalignment itself, for example with learned offsets or attention that
can sample features from slightly different image positions, or a gated rectification module as in NCGR.

**Implementation.** Add a small offset-prediction or deformable sampling step where image features are lifted to
the ground plane (`utils/vox.py`, `models/segnet.py`), trained with calibration noise as in Approach 1.

**Cost.** High: new architecture and longer tuning.

**Expected outcome.** Uncertain on WILDTRACK.

**Risks.** Overfitting. TrackTacular's deformable-attention variant reached only 78.4 MODA on WILDTRACK, which its
authors attribute to overfitting the small dataset. Hard to justify for this project.

---

## Comparison

| # | Approach | Retraining | Engineering | One camera drifts | All cameras drift | Risk |
|---|---|---|---|---|---|---|
| 1 | Train with calibration noise | yes | low | partial | partial | low |
| 2 | Detect and drop the faulty camera | no | medium | good | no help | medium |
| 3 | Self-calibration from pedestrians | no | high | good | good (with a reference) | high |
| 4 | Learned alignment | yes | high | uncertain | uncertain | high (overfitting) |

## Recommendation

1. **Approach 1 first**: cheap, has published precedent, and gives a fair comparison with the literature. **Done
   (seed 1): passes for concatenation (`concat_calib`), narrowly misses for masked fusion.** Confirm with more
   seeds.
2. **Then Approach 2 or 3 for what training cannot fix**: frozen and lagging cameras, and errors well beyond the
   training noise. Approach 3 is the strongest story (a system that
   recalibrates itself) but is real research and may not work; Approach 2 is the safer fallback and reuses the
   camera masking built for Stage 4.
3. Skip Approach 4 unless the others clearly fail.

## Decision gate

**Outcome (seed 1): passed.** All-camera calibration error of 1° costs 35 MODA and single-camera faults
are mostly worse than a dropped camera. Remaining check before training: the calibration sweep below.

The gate was: do not start any of these until the full pilot has run:

```
python -m evaluation.perturb_eval --results <results>/full --data_dir <data>/Wildtrack
```

Go ahead only if, on all 40 test frames, calibration errors of about 1° clearly hurt the baseline (well beyond the
roughly ±1 MODA noise level, and worse than dropping cameras). Before running Approach 1, switch the pilot's
rotation model to the published roll/pitch Gaussian protocol so the numbers can be compared with prior work.
