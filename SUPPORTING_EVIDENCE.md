# Supporting evidence

Evidence behind the robustness direction in [`POTENTIAL_APPROACHES.md`](POTENTIAL_APPROACHES.md): what the
literature already does, what this project measured itself, and what still has to be checked before it can be
claimed in the write-up.

Status labels:
- **Verified**: the full text (or the exact claim) was read.
- **From abstract or search summary**: plausible, but read the paper before citing the detail.
- **Needs a source**: general engineering knowledge; find a citation before using it.

---

## 1. Evaluating perception models under camera faults is established practice

| Work | What it does | Status |
|---|---|---|
| [RoboBEV](https://arxiv.org/abs/2304.06719) (Xie et al., 2023) | Robustness benchmark for bird's-eye-view perception with camera corruptions including **Camera Crash** and **Frame Lost** (our "dropped" and "frozen" faults), alongside weather and lighting corruptions. | From abstract or search summary |
| [Benchmarking and Improving BEV Perception Robustness](https://arxiv.org/abs/2405.17426) (extended RoboBEV, 2024) | Evaluates 33 BEV models; each camera corruption at three severity levels; also complete sensor failures. | From abstract or search summary |
| Dong et al., *Benchmarking Robustness of 3D Object Detection to Common Corruptions*, CVPR 2023 | Robustness benchmark for 3D detection, cited by NCGR as detector-side robustness work. Check whether it includes calibration or spatial-misalignment corruptions. | Seen in NCGR's reference list only |

## 2. Calibration (extrinsic) perturbation as a test protocol

| Work | Protocol | Status |
|---|---|---|
| [BEVHeight](https://arxiv.org/abs/2303.08498), [BEVHeight++](https://arxiv.org/abs/2309.16179) | Roadside (single camera). Gaussian roll and pitch offsets added to the extrinsics, images re-projected to stay consistent. Reports large drops for existing methods, e.g. BEVFormer from 50.73% to 16.35% vehicle AP at IoU 0.5. | From search summary (check the noise units: degrees or radians) |
| [CoBEV](https://arxiv.org/abs/2310.02815), [BEVSpread](https://arxiv.org/abs/2406.08785) | Same roll/pitch noise, plus focal-length scale noise; the same perturbation is also used during training. | From search summary |
| [RECO](https://arxiv.org/abs/2607.20947) (2026) | Roadside; trains with Gaussian yaw and z-translation jitter, tests under persistent mean shifts. | From abstract or search summary |

## 3. Several or all cameras perturbed at once (multi-camera rigs)

| Work | Evidence | Status |
|---|---|---|
| [NCGR](https://arxiv.org/abs/2608.03895) (Pan et al., 2026), nuScenes, 6 cameras | **Training:** perturbations "applied with a probability of 0.7 to a randomly selected subset of 1–6 cameras", so all six can be perturbed at once. **Evaluation:** fixed subsets of 1 to 5 cameras perturbed simultaneously, per-axis bounds of 15° rotation and 0.1 m translation, resampled per sample ("dynamic"); a static setting in the supplement. BEVFormer drops from 0.518 to 0.280 NDS with 5 cameras perturbed. The main table stops at 5 of 6 cameras, so "all cameras" in evaluation is near-precedent, not exact. | **Verified** (full text) |
| Lift-Splat-Shoot (Philion & Fidler, ECCV 2020) | Applies extrinsic perturbations and camera dropout during training (as stated by NCGR). TrackTacular's depth-splatting lifting builds on this line of work, so it is precedent for training with calibration noise (Approach 1). | Claim taken from NCGR; check the LSS paper |
| Klinghoffer et al., *Towards Viewpoint Robustness in Bird's Eye View Segmentation*, ICCV 2023 | Judging by the title, studies changes of the whole camera rig's viewpoint, i.e. every camera differs from training at once. | Seen in NCGR's reference list only; read before citing |
| *Robust multi-camera BEV perception: an image-perceptive approach* (IROS 2024) | Claims resilience to calibration errors and pose fluctuations on nuScenes. Perturbation protocol unknown. | From search summary |
| *CalibRobustBEV* (JUSTC) | BEV multi-modal detector that generates miscalibration scenarios during training. Camera + LiDAR, not camera-only. | From search summary |

## 4. Fixes in the literature (relevant to the approaches)

| Approach in `POTENTIAL_APPROACHES.md` | Precedent |
|---|---|
| 1. Train with calibration noise | Lift-Splat-Shoot; BEVHeight/CoBEV/BEVSpread train with the same noise they test with; NCGR trains on perturbed subsets of 1–6 cameras. |
| 2. Detect and drop the faulty camera | NCGR learns a per-camera "health" scalar that gates its correction, which is related to scoring each camera's reliability. RoboBEV's "Camera Crash" corresponds to the dropped-camera case we already handle. |
| 3. Self-calibration | Explicit extrinsic recovery (RegNet, CalibNet, LCCNet, SemAlign, CalibFormer, LCCRAFT, BEVCalib, as surveyed in NCGR) estimates corrections, mostly from camera–LiDAR correspondences. Calibrating from walking pedestrians in camera-only setups still needs a literature search. |
| 4. Learned alignment | NCGR (gated 2D offsets inside spatial cross-attention); GARKT (enlarged sampling kernels, cited by NCGR). |

## 5. What this project measured itself

All on WILDTRACK, retrained TrackTacular (Segnet, bilinear sampling), seed 1, all 40 test frames.

- **The released WILDTRACK checkpoints do not work with the public code**: MODA ≈ 0, identical to the numbers in
  [TrackTacular issue #2](https://github.com/tteepe/TrackTacular/issues/2) and
  [issue #7](https://github.com/tteepe/TrackTacular/issues/7). The baseline was therefore retrained: MODA 90.9,
  MOTA 89.3, IDF1 94.1 (paper: 92.1 / 91.8 / 95.3). **Verified** by our own runs.
- **WILDTRACK's own calibration is imperfect**: annotated ground positions projected into the images land 2–17 px
  (median, per camera) from the annotated feet (`python -m evaluation.diagnose`).
- **Missing cameras**: the baseline loses 1.4 MODA per dropped camera on average and 8 MODA with 3 of 7 cameras
  removed; none of the four robust-fusion variants improved on this.
- **Camera faults (robustness pilot)**: a frozen camera is worse than a dropped one for 7 of 7 cameras; 1–2° of drift
  or 1–2 frames of lag is worse for most cameras; **1° of calibration error on all cameras costs 35 MODA**
  (90.9 → 55.5), mostly through missed people (recall 95% → 61%).
- **Sensitivity is much higher than in driving benchmarks**: NCGR stresses models with errors of up to 15°, while
  TrackTacular collapses at 1°. A plausible reason is the metric: pedestrian detection is scored within 0.5 m on the
  ground, which is far stricter than nuScenes' detection score. To be confirmed by the calibration sweep (which also
  reports the ground shift, in cm, that each error causes).

## 6. Real-world causes of calibration drift (needs sources)

| Cause | Affects | Status |
|---|---|---|
| Physical bumps (maintenance, cleaning, vandalism, vehicles), loose or settling mounts | One camera | Needs a source |
| Wind sway and vibration (poles, traffic, construction) | One or several cameras | Needs a source |
| Camera or lens replaced without recalibration | One camera | Needs a source |
| PTZ cameras returning imprecisely to presets | One camera | Cite the camera datasheet's preset accuracy |
| Imperfect initial calibration (manual points, flat-ground assumption) | All cameras | Our own WILDTRACK measurement (Section 5); add a calibration-methods reference |
| Thermal expansion of mounts (day–night) | Cameras on similar mounts together | Needs a source |
| Shared structure moving (several cameras on one building or pole) | Several cameras together | Needs a source |
| Non-flat ground (slopes, kerbs, steps) | All cameras (acts like systematic error) | Needs a source |

## 7. Gaps to close before claiming novelty

- [ ] Literature search: robustness to calibration error, camera lag or frozen cameras in **multi-view pedestrian
      detection/tracking** (WILDTRACK, MultiviewX). A limited search found none.
- [ ] Read Klinghoffer et al. (2023), the Lift-Splat-Shoot paper and BEVHeight's protocol details (noise units).
- [ ] Find sources for the real-world causes in Section 6.
- [ ] Repeat the key results with more seeds before reporting them as findings.
