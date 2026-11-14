# Research methodology 5: a geometry-aware sparse query tracker (GeoQ-Track)

## 1. Summary

We change three parts of the network of the TrackTacular baseline (bilinear-sampling lifting, `segnet`) and measure
each change, alone and in every combination, on WILDTRACK and MultiviewX:

| | Component | What changes in the network |
|---|---|---|
| **R1** | Multi-view foundation features | A frozen Depth Anything 3 (DA3) backbone sees all working cameras together (cross-view attention) and its features are added to the ResNet-18 features through a trainable, zero-initialised adapter. |
| **R2** | Sparse person queries | The dense BEV decoder (a ResNet-18 U-Net with per-cell heads) is removed. Candidate people from a light proposal head become queries that read the full-resolution image features of every camera and refine their position, score and motion. |
| **R4** | Track queries | The tracker's tracks become a network input. For each track the network predicts where the person is now and whether they are still there; the tracker matches detections against these predictions. |

R124 (all three) is the proposed architecture; the other six combinations and the baseline B0 form a 2³ factorial
ablation. R3 (test-time self-calibration) from the earlier proposal was dropped: too ambitious for 360 training frames.

What this is **not**: no change to augmentation, schedule, loss weighting of the baseline parts, view fusion
(concatenation, as TrackTacular) or camera dropout (off). Temporal BEV history stays off, as in TrackTacular's
WILDTRACK configuration. Every difference between two variants is a difference in the network.

## 2. Positioning

| Work | What it does | What it leaves open |
|---|---|---|
| TrackTacular (Teepe et al., CVPRW 2024) | Lifts ResNet-18 features to a voxel grid, dense BEV decoder, each detection predicts its previous position | Per-camera features, dense decoding, tracker outside the network |
| MV2GF (arXiv 2608.20639) | Frozen DA3-Giant + ResNet-18, pixels placed by DA3 point maps, max-pooled; 94.7 MODA on WILDTRACK | Detection only (no tracking); 1.4 B parameters, 3.3 FPS |
| CAT-Free (arXiv 2609.34302) | Training-free: VGGT-Ω cameras, YOLO, ray triangulation, OSNet | 89.2 MODA / 90.8 MOTA / 90.4 IDF1 on WILDTRACK — robust, but less accurate |
| DETR3D / Sparse4D (driving) | Object queries sample image features by projecting 3D reference points | Not studied on WILDTRACK/MultiviewX pedestrian tracking |
| Tracktor, MOTR (single camera) | Existing tracks regress their new position / are propagated as queries | Not in multi-view BEV pedestrian tracking |

Our claim, to be confirmed by the factorial study, is the combination: a multi-view tracker whose image features come
from a cross-view geometric foundation model (R1), decoded by sparse ground-anchored queries (R2), with tracks as a
network input (R4), at a small fraction of MV2GF's size (DA3-Base, 0.12 B, frozen).

## 3. Architecture

### 3.1 Shared part (unchanged from the baseline)

Images 720×1280 → ResNet-18 encoder (feature map at 1/4) → bilinear-sampling lifting to the voxel grid
(WILDTRACK 120×4×360, MultiviewX 160×2×250; 10 cm cells) → concatenation over cameras and 3D compression →
BEV compressor and temporal 1×1 block → BEV feature F (256 channels). Image-plane foot-point head (auxiliary loss).

### 3.2 R1 — DA3 multi-view features (`WorldTrack/models/da3_branch.py`)

- Backbone: DA3-Base (`depth-anything/DA3-BASE`, Apache 2.0), a DINOv2 ViT-B/14 in which blocks 4–11 alternate
  per-view and cross-view attention. Only the backbone is built; its weights are frozen and are not stored in our
  checkpoints.
- Input: the same normalised images, resized to 378×672 (27×48 patches). All working cameras of a sample are given
  together, so each camera's features depend on the others. Cameras switched off at test time are not given to DA3.
- Output: the four `out_layers` (5, 7, 9, 11), each 1536 channels (local + global token), concatenated: 6144×27×48.
- Adapter: 1×1 conv 6144→256, GroupNorm, ReLU, 3×3 conv 256→128, GroupNorm, ReLU, bilinear upsampling to the ResNet
  feature size, 1×1 conv 128→128 **zero-initialised**, added to the ResNet-18 features. At initialisation the network
  is exactly the baseline.

### 3.3 R2 — sparse ground-anchored queries (`WorldTrack/models/query_decoder.py`)

- Proposals: conv 3×3 (256→128), InstanceNorm, ReLU, conv 1×1 → heatmap on F, trained with the baseline's centre
  target and focal loss. 3×3 max-pool NMS and the top K = 100 cells give the anchors.
- Query initialisation: linear map of F at the anchor and the proposal score.
- Three decoder layers, each: self-attention between the queries (with a positional embedding of the position) →
  **view sampling**: the query's ground position at 4 heights (voxel centres of the lowest to highest layer) is projected
  into every working camera with the calibration, the image features are sampled bilinearly and averaged over the
  cameras that see the point; the per-height means and the fraction of cameras seeing each height are mapped to the
  query, together with F under the query → FFN → heads: position refinement (zero-initialised), score, motion to the
  previous frame. Positions are refined layer by layer and detached between layers.
- Training: in every layer, one-to-one Hungarian assignment of queries to people (cost = distance / 1 m + 1 − score,
  only within 1 m), sigmoid focal loss on the scores, smooth L1 on the positions, L1 on the motion (when the person is
  in the previous frame within 15 cells, as the baseline's motion target).
- The dense BEV decoder is not built. For evaluation the final queries are written into the dense maps that the
  existing decoding, trackers and metrics expect.

### 3.4 R4 — track queries (`WorldTrack/models/track_queries.py`, `WorldTrack/tracking/query_tracker.py`)

- Input: the tracks (ground positions) the tracker holds before frame t.
- For each track, two rounds: sample features on a 7×7 grid around the current estimate (spacing 4 cells, then 1.5
  cells), attention from the track to its grid → self-attention between tracks → FFN → position update
  (zero-initialised) and a presence logit.
- Features: the decoder's features — the dense BEV decoder output (B0-based variants) or, with R2, F plus the
  multi-view image features under each grid point (the same view sampling as R2).
- Training without sequences: the input tracks are the previous frame's people, jittered by N(0, 1.5 cells); targets
  are the same person's position in frame t, or "gone". 8 ghost tracks per sample, placed more than 1.5 m from
  everyone, must be predicted as gone. Loss: BCE on presence + smooth L1 on position.
- Tracking: as TrackTacular's tracker (thresholds, gate 1.875 m, buffer 5, confirmation on the second hit) except that
  detections are matched to the **network's** predicted track positions; an unmatched track the network still
  believes present moves to the predicted position (not reported) so it can be picked up later.

## 4. Experimental design

- **Variants:** B0, R1, R2, R4, R12, R14, R24, R124 (2³ factorial).
- **Datasets:** WILDTRACK (7 cameras, 400 frames at 2 fps) and MultiviewX (6 cameras, 400 frames, synthetic).
  Standard split: first 90 % training, last 10 % (40 frames) test; the last checkpoint is evaluated, as in all
  earlier methodologies (on these datasets the validation frames are the test frames).
- **Schedule:** TrackTacular's: Adam, one-cycle, max LR 1e-3, batch 2 × 8 accumulation, 30 epochs on WILDTRACK and
  50 on MultiviewX, 32-bit precision; one seed (80085) for screening, more with `SEEDS`.
- **Metrics:** MODA, MODP, precision, recall (0.5 m) at threshold 0.5; MOTA, IDF1, ID switches (1 m). For R4 variants
  MOTA/IDF1 use the query tracker, and TrackTacular's tracker on the same detections is reported too (`base_*`), so
  the tracking gain can be separated from detection changes. Camera removal: all combinations of 1–3 removed cameras
  (WILDTRACK 63 conditions, MultiviewX 42). Cost: trainable and frozen parameters, forward time per frame.
- **Analysis:** per dataset, the main effect of each component (mean of the four variants with it − mean of the four
  without) and the interactions (Yates contrasts), on all cameras and averaged over all camera conditions.

## 5. Hypotheses and decision rules (fixed before the runs)

- **H1 (R1):** main effect on MODA > 0 on both datasets, larger with cameras removed than with all cameras
  (cross-view context should matter most when views are missing).
- **H2 (R2):** main effect on MODA ≥ −0.5 (no worse than the dense decoder) and higher MODP (queries read
  full-resolution image features, while the BEV grid has 10 cm cells).
- **H3 (R4):** main effect on IDF1 > 0 and fewer ID switches, with `base_idf1` showing that the gain is not a detection
  change.
- **H4 (architecture):** R124 beats B0 on MODA, MOTA and IDF1 on both datasets.

With one seed, a difference below about 1 MODA point is not interpreted (earlier seed-to-seed spread). Claims require
the confirmation run with three seeds for B0 and R124 (and the best single-component variant).

## 6. Risks

- R4 and R2 are trained from 360 frames; DETR-style decoders usually need more data. R2's proposals come from the
  dense heatmap, so queries start close to people; R4 is trained per frame from ground truth rather than end-to-end.
- WILDTRACK is close to saturation (≈ 91 MODA for B0); MultiviewX results and camera removal carry more signal.
- DA3 features are at 1/14 resolution and are upsampled; fine localisation still comes from the ResNet-18 features.
- Novelty has been checked against the papers in this repository and a web search (Nov 2026), not exhaustively.

## 7. Implementation map

| File | Content |
|---|---|
| `WorldTrack/models/da3_branch.py` | R1: DA3 backbone loader (pinned commit, backbone only), adapter |
| `WorldTrack/models/query_decoder.py` | R2: view sampling, sparse query decoder, rendering to dense maps |
| `WorldTrack/models/track_queries.py` | R4: track-query head, training inputs from ground truth |
| `WorldTrack/tracking/query_tracker.py` | R4: online tracker that feeds tracks to the network |
| `WorldTrack/models/segnet.py`, `world_track.py` | flags `da3`, `sparse_queries`, `track_queries`; R2 and R4 losses |
| `WorldTrack/datasets/pedestrian_dataset.py` | people of frames t and t−1 as padded lists (for R2/R4 targets) |
| `research-methodology-5/code/m5_pipeline.py` | preflight, training (parallel), evaluation, report, bundle |
| `research-methodology-5/code/m5_eval.py` | evaluation of one checkpoint, every camera condition |
| `research-methodology-5/code/m5_report.py` | tables and factorial effects → `REPORT.md` |
| `research-methodology-5/m5_colab.py` | Colab driver (datasets, DA3 code and weights, pipeline) |

## References

- Teepe et al., *Lifting Multi-View Detection and Tracking to the Bird's Eye View* (TrackTacular), CVPRW 2024, arXiv 2403.12573
- Lin et al., *Depth Anything 3: Recovering the Visual Space from Any Views*, arXiv 2511.10647
- Yamane et al., *MV2GF: Multi-view Pedestrian Detection with a Visual Geometric Foundation Model*, arXiv 2608.20639
- Sakai et al., *CAT-Free: Calibration-, Annotation-, and Target-Training-Free Multi-View BEV Pedestrian Detection and Tracking*, arXiv 2609.34302
- Harley et al., *Simple-BEV*, ICRA 2023, arXiv 2206.07959
- Wang et al., *DETR3D*, CoRL 2021; Lin et al., *Sparse4D*, arXiv 2211.10581
- Bergmann et al., *Tracking without bells and whistles* (Tracktor), ICCV 2019; Zeng et al., *MOTR*, ECCV 2022
- Carion et al., *DETR*, ECCV 2020; Zhu et al., *Deformable DETR*, ICLR 2021
- Hou et al., *Multiview Detection with Feature Perspective Transformation* (MVDet, MultiviewX), ECCV 2020
- Chavdarova et al., *WILDTRACK*, CVPR 2018
