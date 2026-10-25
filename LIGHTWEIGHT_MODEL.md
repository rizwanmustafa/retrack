# A lighter fusion model with nearly the same accuracy

Finding from the Stage 4 experiments (seed 1): replacing TrackTacular's camera-concatenation fusion with **masked
fusion** (`configs/v_masked_drop.yml`) gives a model with **22% fewer parameters** that trains **about 18% faster**,
with **the same tracking accuracy** and a **small but consistent loss in detection accuracy**.

## What changes

TrackTacular stacks the seven projected camera feature volumes in fixed slots and compresses them with a 3D
convolution (`cam_compressor` in `models/segnet.py`): 7 × 128 input channels to 128 output channels with a 3×3×3
kernel, about 3.1 M parameters. Masked fusion replaces that convolution with a parameter-free average over the
cameras that are working and can see each cell (`models/fusion.py`, `fuse_views(..., mode='masked')`). The
`masked_drop` model was also trained with camera dropout.

## Cost

| | TrackTacular (`concat`) | Masked fusion (`masked_drop`) | Change |
|---|---|---|---|
| Parameters | 13.8 M | 10.7 M | −3.1 M (−22%) |
| Model size | 55.2 MB | 42.8 MB | −22% |
| Training speed (A100, batch 2) | 3.60 steps/s | 4.27 steps/s | +18% |
| Training time, 30 epochs | ~30 min | ~25 min | −5 min |

Inference speed and FLOPs have **not** been measured yet (only training throughput).

## Accuracy

Averaged over every combination of removed cameras (k = number removed), threshold 0.5, TrackTacular tracker.

| k | MODA: concat → masked | MOTA | IDF1 |
|---|---|---|---|
| 0 | 90.9 → 90.4 (−0.4) | 89.3 → 89.4 (+0.1) | 94.1 → 94.5 (+0.4) |
| 1 | 89.5 → 88.9 (−0.6) | 88.9 → 88.8 (0.0) | 93.8 → 93.9 (0.0) |
| 2 | 87.2 → 86.2 (−1.1) | 87.6 → 87.2 (−0.4) | 92.7 → 92.8 (+0.1) |
| 3 | 82.9 → 81.9 (−1.0) | 84.1 → 83.9 (−0.2) | 89.4 → 89.8 (+0.4) |

Across the 63 camera-removal combinations, masked fusion has the higher MODA in 13 (so a small but consistent
detection loss), the higher MOTA in 29 and the higher IDF1 in 35 (tracking: no meaningful difference).

## Robustness to calibration errors

The calibration sweep (`python -m evaluation.perturb_eval --mode calib_sweep`, 5 random draws per setting, identical
perturbations for every model) shows masked fusion is **consistently more robust** than concatenation:

| Calibration error | concat ΔMODA | masked ΔMODA |
|---|---|---|
| All cameras 0.5° (14 px) | −3.7 | **−2.5** |
| All cameras 0.75° (21 px) | −17.9 | **−12.9** |
| All cameras 1° (29 px) | −47.4 | **−40.1** |
| Roll/pitch Gaussian σ = 0.5° | −13.1 | **−9.4** |
| 3 of 7 cameras at 1° | −14.5 | **−10.1** |
| 5 of 7 cameras at 1° | −23.2 | **−15.8** |

Compared draw by draw with the concatenation model trained with the same camera dropout, masked fusion was better in
all 20 draws checked. A plausible reason: averaging over the cameras that see a cell weakens the effect of one
misaligned view, whereas the concatenation convolution has learned to rely on every view being aligned.

### With calibration-noise training

Calibration-noise training (Approach 1 in `POTENTIAL_APPROACHES.md`) closes the robustness gap: trained with the same
noise, concatenation and masked fusion are about equally robust (all cameras 1°: MODA 67.0 vs 67.7; roll/pitch
σ = 0.5°: 89.6 vs 87.5), and concatenation is better with clean calibration (92.0 vs 89.7). So masked fusion's
robustness advantage only holds for models trained without calibration noise; its remaining advantage is size and
training speed.

## Interpretation

- For **tracking**, which is what the live system shows, the lighter model is as good as TrackTacular.
- For **detection**, it is about 0.5–1 MODA lower, roughly 5–10 people over the 40 test frames.
- It is a reasonable default for the web system when speed and memory matter more than the last MODA point. Without
  calibration-noise training it is also 4–8 MODA more robust to calibration errors; with it, `concat_calib` is as
  robust and more accurate, so choose between them on speed versus accuracy.

## Caveats and next steps

- One training seed. Repeat with two more seeds before reporting it as a result.
- Measure inference latency (frames per second) and FLOPs for both models on the same GPU; the parameter saving
  does not automatically mean a proportional speed-up, because the image encoder and decoder are unchanged.
- Check whether the detection gap closes with each model's own tuned threshold (report table 5b).
