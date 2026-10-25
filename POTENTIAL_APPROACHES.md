# Potential approaches: robustness to camera calibration errors

Status: **proposal, nothing here is implemented yet.** Start only if the full robustness pilot confirms the problem
(see the decision gate at the end).

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

The baseline already tolerates *missing* cameras well. The robustness pilot (`evaluation/perturb_eval.py`) tests
other camera faults. In a first 5-frame check, the baseline was barely affected by one miscalibrated camera, but
**a 1° calibration error on all seven cameras dropped MODA from 85.3 to 65.3**, far more than losing three whole
cameras (about −8). Five frames is about 100 people, so this has to be confirmed on all 40 test frames before acting on
it.

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

**Expected outcome.** Partial tolerance to small errors. It may cost a little accuracy with clean calibration,
similar to how camera-dropout training trades a little clean accuracy for robustness.

**Risks.** It cannot correct large errors, only tolerate them. Too much noise blurs the ground-plane features and
lowers clean accuracy.

**Evaluation.** `perturb_eval` rotated-camera faults (one camera and all cameras) against the baseline, plus the clean
score, over 3 seeds.

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

1. **Approach 1 first**: cheap, has published precedent, and gives a fair comparison with the literature.
2. **Then Approach 2 or 3 as the project's own contribution.** Approach 3 is the strongest story (a system that
   recalibrates itself) but is real research and may not work; Approach 2 is the safer fallback and reuses the
   camera masking built for Stage 4.
3. Skip Approach 4 unless the others clearly fail.

## Decision gate

Do not start any of these until the full pilot has run:

```
python -m evaluation.perturb_eval --results <results>/full --data_dir <data>/Wildtrack
```

Go ahead only if, on all 40 test frames, calibration errors of about 1° clearly hurt the baseline (well beyond the
roughly ±1 MODA noise level, and worse than dropping cameras). Before running Approach 1, switch the pilot's
rotation model to the published roll/pitch Gaussian protocol so the numbers can be compared with prior work.
