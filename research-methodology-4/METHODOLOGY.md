# Research methodology 4: foreground-gated lifting on top of SA-VM

*Literature numbers are read from the PDFs in [`../research-methodology-2/papers/`](../research-methodology-2/papers/)
and cited as **[Pxx] + table/section**. Our own numbers come from the run reports
(`../research-methodology-2/METHODOLOGY.md` §8–10). Design choices that are not from a paper are labelled **our
choice**.*

> **Status, 2026-11-11.** Implemented and not yet run. Session 1 is `RUNS='A2x:80085 FGx:80085'` (see `RUN.md`).

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
| E0c | no | no | no | retrained TrackTacular, current code |

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
| F1 and F2 fail | TrackTacular's per-view image heads already learn where people are; the gate is redundant |

## 5. Schedule

| session | `RUNS` | answers |
|---|---|---|
| 1 | `A2x:80085 FGx:80085` | F0, F1, F4 |
| 2 | `FG0:80085 E0c:80085` | F2, F3 |
| 3–4 | `FGx:2 A2x:2`, then `FGx:1 A2x:1` | F1 at 3 seeds, only if F1 passed screening |

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
