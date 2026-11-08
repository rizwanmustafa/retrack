# ARL: Augment, Remember, Link — a literature-grounded methodology for raising MODA, MOTA and IDF1 on WILDTRACK

*Research methodology, attempt 1. Every quantitative claim below is taken from a paper stored in
[`papers/`](papers/) (cited as **[Pxx]**, see [References](#references) and [`PAPERS.md`](PAPERS.md)) or from
reading this repository's code (marked **[code]**, with the file). Where neither source gives a value, the text
says so and the value is treated as a hyper-parameter to sweep, not as a finding.*

Diagram: [`methodology_diagram.drawio`](methodology_diagram.drawio) (open with diagrams.net / draw.io).

---

## Abstract

We propose ARL, a three-part methodology on top of the retrained TrackTacular baseline already used in this
repository (Segnet bilinear-sampling lifting, ResNet-18, seven cameras). **Augment**: two augmentations that keep
the views geometrically consistent. One is 3D random occlusion, which paints person-sized occluders into all views
at once. The other is scene-level BEV augmentation. Both target MODA, and both have Wildtrack ablations in the
literature. **Remember**: switch on TrackTacular's own temporal BEV history, which the upstream WILDTRACK config
disables. **Link**: add offline association after the existing online ByteTrack-style BEV tracker. This is a
globally optimal tracklet linking step followed by Gaussian-smoothed interpolation. It targets MOTA and IDF1, and
the literature shows large gains from this kind of association with the detector held fixed. Parameters are tuned
under two explicitly labelled protocols. The clean one uses a held-out tail of the training frames. The oracle one
tunes on the test frames, which the brief allows ("don't worry if it overfits a bit"). Every component is
evaluated as a controlled ablation with three seeds against a pre-registered decision rule.

---

## 1. Problem statement and target metrics

**Dataset.** WILDTRACK [P01] has seven synchronised, calibrated cameras (1080×1920) over a 12 m × 36 m area, with
400 annotated frames. Frames were extracted at 10 fps and every fifth was annotated, so consecutive annotated
frames are 0.5 s apart (2 fps) [P01 §3.2]. There are about 20 pedestrians per frame, and each location is seen by
3.74 cameras on average [P08 §4.1]. The standard split uses the first 90 % of frames for training and the last
10 % (40 frames) for testing [P08 §4.1; code: `datasets/pedestrian_dataset.py` lines 48–51].

**Metrics** (as in [P07, P08]; code: `evaluation/mod.py`, `evaluation/mot_bev.py`):

| metric | definition | match radius |
|---|---|---|
| MODA | 1 − (FN + FP) / N_gt, per-frame detections [P40, P08 §4.2] | 0.5 m |
| MOTA | 1 − (FN + FP + IDSW) / N_gt [P40 Eq. 2] | 1 m |
| IDF1 | identity F1 over the whole sequence [P35] | 1 m |

MODA is computed from per-frame detections **before** tracking [code: `world_track.py` `test_step`, and
`moda_pred_list` is filled from thresholded detections]. A tracking-stage change therefore cannot move MODA under
the standard protocol, and we do not change that protocol.

**Objective.** Increase MODA, MOTA and IDF1 over the repository's retrained TrackTacular baseline (B0), with all
seven cameras.

---

## 2. Baseline and diagnostic audit

### 2.1 Reference numbers

| source | MODA | MOTA | IDF1 |
|---|---|---|---|
| TrackTacular paper, bilinear sampling [P08 Tab. 1–2] | 92.1 | 91.8 | 95.3 |
| TrackTacular retrained by an independent group, 7 cams [P22 Tab. 1] | 88.50 | 89.71 | 94.00 |
| B0 in this repo | measured in Exp. 0 (§5) | | |

The README records that the released WILDTRACK checkpoints give MODA ≈ 0 with the public code (TrackTacular
issues #2 and #7), so B0 is retrained. B0 is therefore compared with *our own* retrained numbers, not with the
paper's.

### 2.2 Facts found by reading the code

These come from reading the repository. They are not assumptions, and they shape the method.

1. **Temporal history is off on WILDTRACK.** `configs/d_wildtrack.yml` sets `use_temporal_cache: False`. This is
   the same in the upstream release commit `f17fa5b`, while `d_multiviewx.yml` leaves it on by default. With the
   cache off, `Segnet.forward` concatenates the current BEV with itself (`prev_bev = feat_bev`). The
   history-fusion ablation that TrackTacular reports (Tab. 4) was run on MultiviewX, not WILDTRACK [P08].
2. **Scene-level augmentation is effectively absent.** In `PedestrianDataset.__getitem__` the scene transform has
   an identity rotation (`Rz = torch.eye(3)`) and a translation drawn from ±0.25 world-grid cells. On WILDTRACK one
   cell is 2.5 cm, so the shift is at most about 6 mm.
3. **The previous-frame GT is not transformed by the scene augmentation.** `worldgrid_pts` gets the transform;
   `worldgrid_pts_pre` does not. With today's ≈ 6 mm shift this does not matter. With any real scene augmentation
   it would corrupt the previous-frame offset target (`offset_bev[2:]`), so A2 must fix it.
4. **View augmentation exists**: resize ∈ [0.8, 1.2] and a random crop of up to ±0.8·720 px (`crop_offset`), with
   the intrinsics adjusted. There is no occlusion-type augmentation.
5. **Decoding**: 3×3 max-pool NMS, top-K = 60, one score threshold (`conf_threshold`)
   (`utils/decode.py`, `world_track.py`).
6. **Tracking**: an online JDE-style tracker with a Kalman filter. The cost is the distance between a track's
   predicted position and the detection's *learned previous-frame position*, gated at 75 world-grid units
   (1.875 m) (`tracking/multitracker.py`). Retrack already adds a ByteTrack-style two-pass variant
   (`ByteBEVTracker`) and optional OSNet appearance features (`tracking/reid.py`). Nothing offline exists, and
   there is no interpolation.
7. **The temporal cache is keyed only by frame id** (`load_cache` / `store_cache`), and training batches come from
   `TemporalSampler`, which produces semi-sequential chunks. If scene augmentation changes per sample, the cached
   BEV(t−1) can lie in a different reference frame than BEV(t). This is the main interaction risk between A2 and
   A3 (§4.4).

---

## 3. Evidence base (what the literature shows on WILDTRACK)

### 3.1 Detection (MODA)

| intervention | paper, table | WILDTRACK effect | notes |
|---|---|---|---|
| 3D random occlusion (cylinders projected into all views) | 3DROM [P05] Tab. 3 | MVDet 88.2 → **92.3** MODA | Tab. 6: 3D occlusion 93.5 vs per-view Random Erasing 92.0; Tab. 5: best at p = 100 % |
| multi-layer projection + 3D occlusion | 3DROM [P05] Tab. 3 | 88.2 → **93.5** | multi-layer alone gives 91.2 |
| view + scene affine augmentation | MVAug [P06] Tab. 2–4 | 90.86 → **93.17** | single runs; 50 % augmentation best, 100 % gives 90.44 |
| view-coherent augmentation | MVDeTr [P03] §4.3 | removing it costs −2.0 MODA; adding it to MVDet +0.8 | 5-run p-values reported |
| crop/resize + projection noise (vs none) | EarlyBird [P07] Tab. 3 | 77.8 → 89.5 | the baseline "suffers from strong overfitting" |
| BEV history fusion | TrackTacular [P08] Tab. 4 | **MultiviewX only**: 95.4 → 96.5 MODA, IDF1 81.5 → 83.8, MOTA 90.0 → 87.9, MODP 89.8 → 75.0 | the motion head then lifts MOTA to 92.4 and IDF1 to 85.6 |
| recurrent temporal BEV memory on TrackTacular | Visibility-Gated ConvGRU [P45] Tab. 3 (**WILDTRACK**) | plain ConvGRU: MODA 92.1 → 92.4, IDF1 95.3 → 96.3, MOTA 91.8 → 92.5; visibility-gated: 93.9 / 97.0 / 95.1; MODP drops (76.2 → 71.8) | baseline = TrackTacular paper numbers; single runs |
| hybrid homography + volumetric fusion, ray conditioning | GRACE [P22] Tab. 1 | retrained TrackTacular 88.50 → **92.58** (7 cams) | 2026 arXiv paper |
| 3D feature pulling + large-kernel refiner + foreground selector | MVFP [P21] ablation | 86.7 (Z = 1) → 92.3 (Z = 8) → 93.2 (+LKR) → 94.1 | voxel 10×10×20 cm, Z = 8 |
| sparse projection + density weighting + per-view BEV loss | ATLASFusion [P17] Tab. 4 | 92.1 → 93.6 MODA, 95.3 → 95.9 IDF1 | baseline = TrackTacular paper numbers |
| multi-scale BEV features | MSMVD [P20] Tab. 2 | 94.6 MODA | ablations only on GMVD |

**Evidence against changes we might otherwise make:**

* *Bigger backbone.* EarlyBird [P07] Tab. 4, on WILDTRACK: ResNet-18 91.2 MODA, ResNet-50 89.6, Swin-T 89.5.
  Simple-BEV [P27] Tab. II (nuScenes) and MSMVD [P20] Tab. 4 (GMVD) find the opposite, but on other datasets. The
  only WILDTRACK evidence favours ResNet-18, so we keep it.
* *Deformable-attention lifting.* TrackTacular [P08] Tab. 1 gives 78.4 MODA and reports strong overfitting.
* *Camera dropout during training* (Retrack Stage 4). Simple-BEV [P27] Tab. IIIc: 6/6 cameras 47.4 IoU vs 5/6
  46.4. Stage 4 aims at robustness to missing cameras, not at the all-camera metric this attempt optimises, so ARL
  trains without dropout.
* *Photometric augmentation.* Simple-BEV [P27] (augmentation study, next to Tab. III) reports that blur, colour and contrast
  augmentations "did not help".
* *Per-view random erasing.* Worse than 3D occlusion on WILDTRACK (3DROM [P05] Tab. 6).

### 3.2 Association (MOTA, IDF1)

| intervention | paper, table | effect | detections held fixed? |
|---|---|---|---|
| online DeepSORT → offline muSSP min-cost flow | MVFlow [P09] Tab. 2 | MVDeTr dets: MOTA 24.0 → **89.4**, IDF1 56.3 → **90.7**; MVFlow dets: 52.8/73.3 → 91.3/93.5 | **yes** |
| EarlyBird's online tracker → offline hierarchical GNN (MCBLT†) | MCBLT [P19] Tab. 3 | MOTA 89.5 → **92.6**, IDF1 92.3 → **95.6** | **yes** (EarlyBird detections) |
| AFLink tracklet linking + GSI interpolation | StrongSORT [P12] Tab. II (MOT17 val) | CenterTrack: IDF1 64.6 → 68.4 (+3.8), MOTA +0.1; across 6 trackers IDF1 +0.9 to +3.8, MOTA +0.1 to +2.0 | yes |
| BYTE two-threshold association | ByteTrack [P10] Tab. 3 (MOT17 val) | CenterTrack (Kalman): +1.3 MOTA, +9.8 IDF1 | yes |
| low-score detections may only continue tracks (BTR) | GRACE [P22] §4 | 7 cams: MOTA 90.73 → 91.04 | yes |
| observation-centric re-update / momentum | OC-SORT [P11] ablation (MOT17 val) | IDF1 76.9 → 77.8 | yes |
| multi-timestamp motion and appearance costs | MVTrajecter [P16] Tab. 3, 5–6 | WILDTRACK 94.3 MOTA / 96.5 IDF1; GMVD K = 1 → 7: +3.9 MOTA, +6.2 IDF1 | learned end-to-end |
| learned cross-frame attention affinity | Attention-aware [P14] Tab. 3 | 90.4/94.2 → 92.1/95.9 (MOTA/IDF1) | learned |
| dynamic spatio-temporal GNN | UMPN [P15] Tab. 2 | 93.9 MOTA / 96.3 IDF1 | uses an external pre-trained detector |

The two rows where the detections are explicitly held fixed (MVFlow, MCBLT†) make the strongest case for this
attempt. Changing only the association, from online to offline and global, gave +3.1 MOTA / +3.3 IDF1 on top of
an already strong BEV detector [P19]. The learned trackers [P14–P16] score higher, but each requires training a
new association network. We defer them to attempt 2 (§8).

---

## 4. Method: ARL

The full pipeline is shown in `methodology_diagram.drawio`. Components keep the IDs A1–C3 used in the diagram and
in the ablation ladder.

### 4.1 Stage A: detector training ("Augment", "Remember")

**A1 — 3D random occlusion** (following 3DROM [P05] §3.2, Algorithm 1).
For each training sample, with probability *p*, draw *n* ground locations uniformly inside the BEV bounds. Reject
any location closer than *d* = 1 m to one already chosen [P05]. Model each location as a vertical cylinder of
height 180 cm: [P05] computes the height in each view from the 0 cm and 180 cm planes, and 180 cm is also the
person height already used in `tracking/reid.py`. Project it into **every** camera with that camera's calibration
(K, [R|t]) as a filled rectangle of height H and width αH, painted with a constant value Ω. Labels are unchanged.
An occluded person still has to be detected from the other views, which is the stated purpose of [P05].

* *n* = 25 for WILDTRACK [P05 Tab. 1]; *p* ∈ {0.5, 1.0} (Tab. 5 is best at 1.0 for MVDet).
* α and Ω are **not given numerically** in [P05]. We start from α = 0.4, which is `width_ratio` in
  `tracking/reid.py` (our choice, not a finding), sweep α ∈ {0.3, 0.4, 0.5}, and use Ω = the ImageNet mean pixel
  value (our choice).
* Implementation: in `PedestrianDataset.get_image_data`, draw the occluder set once per *frame* and paint it
  before the resize/crop so that the intrinsics stay consistent. The projection code can be reused from
  `BEVReIDExtractor.project_boxes` (`tracking/reid.py`).

**A2 — scene-level BEV augmentation** (following MVAug [P06] "scene augmentation"; Simple-BEV [P27] reference-frame
randomisation, +0.6 IoU on nuScenes).
With probability 0.5 [P06 Tab. 4], apply one random rigid transform T_s to `ref_T_global`, to the current GT points
**and** to the previous-frame GT points. The last part fixes audit item 3. The BEV grid is a fixed 12 m × 36 m
rectangle, so we restrict T_s to transforms that keep the area valid:

* the two BEV mirror flips (MVAug Tab. 2: scene H-flip alone 91.39 vs 90.86 without augmentation), and
* small translations of up to ±0.5 m together with rotations of up to ±5°. Points that leave the grid are dropped
  from the target, as `get_bev_gt` already does. The ±0.5 m / ±5° range is our choice. MVAug's own affine range
  (45° rotation, 20 % translation) [P06 §4.1] is too large for a fixed, non-square BEV, so we sweep
  {flip only, flip + small affine}.

**A3 — temporal BEV history** (TrackTacular's own mechanism [P08 §3.2]).
Set `use_temporal_cache: True` for WILDTRACK and keep the `TemporalSampler`. This activates the `temporal_bev`
fusion of BEV(t) with the cached BEV(t−1), which the WILDTRACK config disables today. TrackTacular's own ablation is
on MultiviewX [P08 Tab. 4]. On WILDTRACK, adding a recurrent temporal memory to TrackTacular raised MODA, MOTA and
IDF1 in [P45 Tab. 3], with plain ConvGRU giving smaller gains than the visibility-gated version. Both lowered MODP. There, history raised MODA and IDF1 but lowered MODP and MOTA, and the
motion head recovered MOTA. A3 is therefore a hypothesis tested in its own ablation row, not an expected gain. At
test time the batch size must stay 1 so that BEV(t−1) is always cached [P08 §4.3 "Temporal Caching"].

**A4 — optional architectural extension** (run only if A1–A3 are confirmed; **not implemented in the attempt-1 code**, see `RUN.md`).
GRACE's volumetric-guided fusion plus ray conditioning [P22 §2.2–2.3] is the strongest single detection change
reported on *retrained* TrackTacular at seven cameras: 88.50 → 92.58 MODA [P22 Tab. 1]. It gates between a
homography-projected feature (the repo's `MVDet` path) and the bilinear-sampled volume, using the coverage map that
`models/fusion.py` already computes, and appends per-camera ray-direction channels. A lighter alternative is MVFP's
large-kernel BEV refiner (+0.9 MODA in [P21]'s ablation).

**A5 — optional weight averaging.** The README notes that WILDTRACK has no validation split and that the last
epoch is evaluated to avoid selecting on the test set. Stochastic Weight Averaging [P36] over the last *k* epochs
reduces dependence on that one epoch without looking at test data. It is implemented by averaging the
weights of the last *k* saved checkpoints, so it needs no extra library feature. The only evidence is out of domain: +0.6 to
+0.9 % ImageNet top-1 for pretrained models [P36 Tab. 2]. A5 is therefore labelled exploratory.

**Training schedule.** Unchanged from B0 so that only the listed components differ: `configs/t_retrack.yml`
(30 epochs, Adam, one-cycle, max LR 1e-3, effective batch 16 via gradient accumulation, last checkpoint), input
720×1280 [code: `final_dim`]. Simple-BEV [P27] Fig. 4 shows that batch size matters a lot, so the effective batch
size must not change between rows.

### 4.2 Stage B: detection decoding

**B1 — calibrated detection threshold.** Sweep τ ∈ [0.30, 0.70] in steps of 0.05. This is Retrack Stage 1
(`evaluation/retrack_eval.py`). Choose τ to maximise MODA under the **clean** protocol (on the validation model,
§5.2) and, separately and labelled, under the **oracle** protocol. Detections {xy, xy_prev, score} for all
scores ≥ 0.05 are written to disk once per model and reused by every Stage C experiment.

### 4.3 Stage C: association ("Link")

**C1 — online BEV tracker.** Use Retrack's `ByteBEVTracker` (Stage 2). In pass 1, high-confidence detections
(≥ τ) are matched to all tracks using TrackTacular's learned previous-frame position as the motion cue. In pass 2,
detections with τ_low ≤ score < τ can only *continue* unmatched tracks [P10]. This is the same rule as GRACE's BTR
[P22 §2.4]. Optional OSNet appearance is Retrack Stage 3. The output is treated as **tracklets**: they favour
precision and may be fragmented.

**C2 — offline global tracklet linking.** Over the whole 40-frame test sequence, link tracklet ends to tracklet
starts by solving a minimum-cost path cover. Each tracklet may have at most one predecessor and one successor, and
costs are additive along a chain. In that setting the min-cost-flow formulation used by KSP / muSSP [P41, P09]
reduces to a rectangular assignment problem with a "no link" option for every end and every start. It is solved
exactly with `scipy.optimize.linear_sum_assignment`, the same Hungarian solver the repo already uses in
`tracking/matching.py`. A candidate link i → j is allowed only if 1 ≤ Δt ≤ G_link frames and the gap passes a
distance gate. Its cost is

```
c(i→j) = w_m · ‖ x̂_i(t_j) − x_j ‖ / gate  +  w_t · (Δt − 1)  +  w_a · (1 − cos(f_i, f_j))   [w_a = 0 without re-ID]
```

where x̂_i(t_j) is i's last position extrapolated by its Kalman velocity to j's start time. The first term
mirrors the motion term that MVFlow adds to the muSSP edge cost [P09 Eq. 11]. The appearance term reuses
Retrack's OSNet descriptors, by analogy with AFLink [P12], which links on spatio-temporal cues only (so w_a = 0 is
the AFLink-like variant). "No link" costs a constant c_0, which sets the linking threshold.

*C2b (ablation)* — a detection-level min-cost flow over all detections ≥ τ_low, with KSP/muSSP-style source/sink
edges [P41, P09], and the predicted previous-frame offset in the edge cost as in MVFlow's muSSPFlow [P09 Eq. 11].
It tests whether linking at detection level beats linking at tracklet level.

**C3 — Gaussian-smoothed interpolation (GSI)** [P12 §3]. Inside each linked track, fill gaps of ≤ G frames by
Gaussian-process regression of x and y over time, which also smooths observed positions. StrongSORT uses τ = 10 and
a maximum gap of 20 frames at 30 fps [P12 §4]. WILDTRACK runs at 2 fps [P01], so frame counts do not transfer.
We sweep G ∈ {1, 2, 3} frames (0.5–1.5 s) and the kernel length scale. Interpolated points can add false
positives, so they are only placed inside a link chosen by C2, never extrapolated past a track's ends.

### 4.4 Interaction risks handled in the design

* **A2 × A3.** Because of the frame-keyed cache (audit item 7), one scene transform T_s must be shared by all
  frames of a `TemporalSampler` chunk, or the cache must store T_s and warp BEV(t−1) into the current frame. The
  ladder includes A2 and A3 alone and together, so an interaction shows up directly.
* **A2 × image loss.** MVAug [P06 Tab. 3] found that the image-plane loss helped together with view augmentation
  but *hurt* with scene augmentation alone (90.65 vs 91.28). The image-centre loss stays as in B0, and the
  combination is measured, not assumed.
* **A1 × A2 strength.** MVAug [P06 Tab. 4] found 100 % augmentation worse than 50 %. 3DROM [P05 Tab. 5] found
  100 % occlusion best for MVDet. The two papers disagree on intensity, so *p* is swept for A1.

---

## 5. Experimental protocol

### 5.1 Configurations (ablation ladder)

The code (`WorldTrack/arl_pipeline.py`, `evaluation/arl_eval.py`, `evaluation/arl_report.py`; see `RUN.md`) runs
this ladder. It also tests every addition three ways: add-one, leave-one-out, and (for post-hoc stages) the average
marginal effect over all combinations of the other switches. Each test ends in a keep/drop verdict under the rule
of §5.3.

Stages A and B require training. Stage C runs on cached detections of the corresponding model.

| ID | configuration | trained? | primary target |
|---|---|---|---|
| E0 | B0: retrained TrackTacular, `tracker: baseline`, τ = 0.5 | yes | — |
| E1 | B0 + A1 | yes | MODA |
| E2 | B0 + A2 | yes | MODA |
| E3 | B0 + A3 | yes | MODA, IDF1 |
| E4 | B0 + A1 + A2 + A3 (best variants from E1–E3) | yes | MODA |
| E5 | E4 + B1 | no | MODA |
| E6 | E5 + C1 | no | MOTA, IDF1 |
| E7 | E6 + C2 | no | IDF1, MOTA |
| E8 | E7 + C3 = **full ARL** | no | MOTA |
| E9 | E8 with C2b instead of C2 | no | IDF1 |
| E10 | E8 + OSNet appearance in C1 and C2 | no | IDF1 |
| E11 / E12 | E8 + A4 / + A5 (optional) | yes | MODA |

Stage C is also applied to **E0's** detections (E0 + B1 + C1 + C2 + C3). This separates the association gain from
the detector gain, the same held-fixed comparison as [P09, P19].

### 5.2 Tuning protocols

* **Clean.** Train a *validation model* on frames 0–323, then choose τ, τ_low, G_link, gates, weights, c_0 and G on
  frames 324–359 (the last 10 % of the training frames). Retrain on 0–359 and evaluate once on 360–399 with those
  values frozen. This addresses the README's concern that WILDTRACK has no validation split.
* **Oracle** (permitted by the brief). Tune the same parameters on 360–399. Results are reported in a separate,
  explicitly labelled column and are an upper bound, not a generalisation claim.

### 5.3 Seeds and statistics

* Three seeds per trained configuration. Beyond Centers [P38 Tab. 1] reports WILDTRACK standard deviations over
  three runs of 0.6–1.5 MODA and 0.4–1.2 IDF1, which is about the size of most gains in §3. MVDeTr [P03] used five
  runs with p-values for the same reason. MVAug [P06 §5] names single runs as a limitation.
* Report mean ± std of MODA, MODP, precision, recall, MOTA, IDF1, MOTP, MT, ML, and raw FN / FP / IDSW / Frag
  counts. Use a Welch t-test against E0 over seeds.
* **Decision rule, fixed before running.** A component stays in the final method only if it improves its target
  metric by more than one E0 standard deviation and does not lower either of the other two primary metrics by
  more than one standard deviation.

### 5.4 Compute estimate

From the README: about 35 min of training and about 10 min of evaluation per run on an A100. Trained rows E0–E4 at
three seeds, plus one validation model per configuration for the clean protocol, come to roughly
5 × 3 × 45 min + 5 × 45 min ≈ 15 h. Stage C rows reuse cached detections and need minutes each. A4/A5 add about
2–3 h per variant.

---

## 6. Hyper-parameters

| parameter | values | source |
|---|---|---|
| occluders per frame *n* | 25 | [P05 Tab. 1] |
| occlusion probability *p* | {0.5, 1.0} | [P05 Tab. 5] |
| occluder height | 180 cm | [P05 §3.2]; `tracking/reid.py` |
| min occluder spacing *d* | 1 m | [P05 §3.2] |
| occluder width ratio α | {0.3, 0.4, 0.5} | not given in [P05]; 0.4 from `reid.py` (our choice) |
| scene-aug probability | 0.5 | [P06 Tab. 4] |
| scene-aug family | {flips} or {flips + ±0.5 m, ±5°} | flips from [P06 Tab. 2]; affine range our choice |
| detection threshold τ | 0.30–0.70, step 0.05 | Retrack Stage 1 |
| low threshold τ_low | {0.1, 0.2, 0.3} | [P10]; [P22] uses τ − 0.2 |
| track buffer (frames) | {5, 10} | B0 = 5; [P07], [P14] keep tracks 10 frames |
| link window G_link | {10, 20, 40} frames | ours; must exceed C1's track buffer (5–10 frames) |
| link gate | {2.0, 3.0} m + 0.5 m per extra frame | ours; must exceed C1's match gate (1.875 m) — a 1 m gate found no links in the quick run |
| C2b transition gate | {1.875, 2.5} m | 1.875 m = the online trackers' gate (`tracking/multitracker.py`) |
| interpolation gap G | {1, 2, 3} frames | [P12] uses 20 frames at 30 fps; rescaled, ours |
| appearance weight | λ = 0.98 in C1 (as [P07, P14, P16]); w_a ∈ {0, 0.5} in C2 | [P07 §3.5] |

---

## 7. Threats to validity

* **Label noise and saturation.** The TrackTacular authors write that WILDTRACK results "start to saturate" and
  may be limited by labelling accuracy, since the labels were made with a perspective transform [P08 §4.4]. Gains
  of the size reported in §3 can be within seed noise (§5.3).
* **Over-fitting to the test frames.** The oracle protocol is allowed by the brief, but it is always reported
  next to the clean one.
* **Offline vs online.** C2/C3 use future frames. They must be compared with offline methods (MVFlow + muSSP
  [P09], MCBLT [P19]), and the online result (E6) is reported alongside.
* **Out-of-domain evidence.** The ByteTrack, OC-SORT, StrongSORT and SWA effect sizes come from MOT17 or ImageNet,
  not WILDTRACK. They motivate components but do not predict their size here.
* **Reproduction gap.** The independently retrained TrackTacular is 3.6 MODA below the paper [P22 vs P08], so
  absolute numbers are compared with E0, not with published tables.

---

## 8. Considered, not adopted in attempt 1

| idea | why not now | evidence |
|---|---|---|
| larger encoder (R50 / Swin-T) | worse on WILDTRACK | [P07 Tab. 4] |
| deformable-attention lifting | over-fits on small data | [P08 Tab. 1, §4.4] |
| camera dropout (Retrack Stage 4) | aims at robustness, not all-camera accuracy | [P27 Tab. IIIc] |
| photometric augmentation | no gain reported | [P27] |
| learned association (MVTrajecter, attention, UMPN, ReST, SUSHI-style GNN) | highest reported IDF1, but each needs a new trained association model; attempt 2 | [P14, P15, P16, P18, P19, P30] |
| calibration-free detection (CaMuViD, 95.0 MODA) | not a BEV method, so it does not fit this pipeline | [P26] |
| LMGP (98.2 IDF1) | uses ground-truth boxes, not comparable | stated in [P08 §2] |

---

## 9. Closest prior work and what is new here

A targeted search (November 2026) for "TrackTacular / WILDTRACK + occlusion or scene augmentation + temporal
memory + offline linking / interpolation" found no paper that combines these, and none that tests each addition
with add-one and leave-one-out ablations over seeds. That is evidence, not proof: an unpublished thesis or a very
recent preprint could exist. The closest line of work is one group (Hotta lab, Meijo University). It modifies
TrackTacular on WILDTRACK with one module per paper: GRACE [P22] (volumetric/homography fusion, ray conditioning,
low-score track recovery), ATLASFusion [P17] (sparse projection, density weighting, per-view loss) and
Visibility-Gated ConvGRU [P45] (temporal memory). Each compares a single run against the TrackTacular paper's
numbers.

| aspect | prior work | ARL attempt 1 |
|---|---|---|
| geometric augmentation consistent across views | 3DROM [P05], MVAug [P06]; both on homography (MVDet-style) detectors | the same ideas on a bilinear-sampling lifting model (TrackTacular), plus BEV-cache alignment so they can coexist with temporal history |
| temporal history on WILDTRACK | [P45] (new recurrent module); [P08] (ablation on MultiviewX only) | TrackTacular's own mechanism switched on (no new module), tested with and without augmentation |
| offline association | MVFlow + muSSP [P09], MCBLT [P19], StrongSORT AFLink/GSI [P12] (single-view) | exact tracklet path cover + GSI on top of TrackTacular's learned motion, plus a detection-level min-cost flow alternative |
| protocol | mostly single runs vs published baselines; tuned on the test frames | retrained baseline, clean vs oracle tuning, 3 seeds, add-one + leave-one-out + marginal effects, rule fixed in advance |

**Honest framing for a panel:** the components are known, on purpose, because each needs published evidence.
The contribution is (a) the combination and the interactions it needs handled (scene transform × temporal cache ×
tracking-offset target); (b) the evaluation protocol; and (c) the baseline issues found in the code (§2.2).

## 10. Implementation map (this repository)

| component | file(s) | change |
|---|---|---|
| A1 | `datasets/pedestrian_dataset.py` (`get_image_data`) | per-frame occluder set, painted into every view before resize/crop; projection from `tracking/reid.py` |
| A2 | `datasets/pedestrian_dataset.py` (`__getitem__`) | random flip / small rigid T_s on `ref_T_global`, GT(t) **and** GT(t−1); shared within a sampler chunk |
| A3 | new `configs/v_arl.yml` | `use_temporal_cache: True`, `cam_dropout_prob: 0`, `fusion: concat` |
| A4 / A5 | `models/fusion.py`, `models/segnet.py` / average of the last *k* saved checkpoints | optional |
| B1 | `evaluation/retrack_eval.py` | add clean-protocol threshold choice; dump detections ≥ 0.05 |
| C1 | `tracking/multitracker.py` (`ByteBEVTracker`) | unchanged; emit tracklets with Kalman state |
| C2 / C2b / C3 | new `tracking/offline.py` | tracklet path cover (Hungarian), detection-level flow, GSI |
| protocol | `retrack_pipeline.py`, `evaluation/retrack_report.py` | ladder E0–E12, 3 seeds, clean and oracle columns |

---

## References

PDFs are in [`papers/`](papers/). Paper numbers match the file prefixes. P39 is missing on purpose: see the note
under the list.

| ID | file | citation |
|---|---|---|
| P01 | `01_wildtrack_dataset_1707.09299.pdf` | Chavdarova et al., *The WILDTRACK Multi-Camera Person Dataset*, arXiv 1707.09299 (CVPR 2018) |
| P02 | `02_mvdet_2007.07247.pdf` | Hou, Zheng, Gould, *Multiview Detection with Feature Perspective Transformation*, ECCV 2020 |
| P03 | `03_mvdetr_2108.05888.pdf` | Hou, Zheng, *Multiview Detection with Shadow Transformer (and View-Coherent Data Augmentation)*, ACM MM 2021 |
| P04 | `04_shot_iccv2021.pdf` | Song et al., *Stacked Homography Transformations for Multi-View Pedestrian Detection*, ICCV 2021 |
| P05 | `05_3drom_2207.10895.pdf` | Qiu et al., *3D Random Occlusion and Multi-Layer Projection for Deep Multi-Camera Pedestrian Localization*, ECCV 2022 |
| P06 | `06_mvaug_2210.10756.pdf` | Engilberge et al., *Two-level Data Augmentation for Calibrated Multi-view Detection*, WACV 2023 |
| P07 | `07_earlybird_2310.13350.pdf` | Teepe et al., *EarlyBird: Early-Fusion for Multi-View Tracking in the Bird's Eye View*, WACVW 2024 |
| P08 | `08_tracktacular_2403.12573.pdf` | Teepe et al., *Lifting Multi-View Detection and Tracking to the Bird's Eye View* (TrackTacular), arXiv 2403.12573, 2024 |
| P09 | `09_mvflow_tracking_2210.10771.pdf` | Engilberge, Liu, Fua, *Multi-view Tracking Using Weakly Supervised Human Motion Prediction* (MVFlow), WACV 2023 |
| P10 | `10_bytetrack_2110.06864.pdf` | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, arXiv 2110.06864 |
| P11 | `11_ocsort_2203.14360.pdf` | Cao et al., *Observation-Centric SORT*, arXiv 2203.14360 |
| P12 | `12_strongsort_2202.13514.pdf` | Du et al., *StrongSORT: Make DeepSORT Great Again* (AFLink, GSI), IEEE TMM 2023 |
| P13 | `13_centertrack_2004.01177.pdf` | Zhou, Koltun, Krähenbühl, *Tracking Objects as Points*, ECCV 2020 |
| P14 | `14_attention_aware_mvtrack_2504.03047.pdf` | Alturki, Hilton, Guillemaut, *Attention-Aware Multi-View Pedestrian Tracking*, arXiv 2504.03047 |
| P15 | `15_one_graph_dynamic_gnn_2507.08494.pdf` | Engilberge et al., *One Graph to Track Them All* (UMPN), arXiv 2507.08494 |
| P16 | `16_mvtrajecter_2509.01157.pdf` | Yamane et al., *MVTrajecter*, arXiv 2509.01157 |
| P17 | `17_atlasfusion_2509.08421.pdf` | Toida et al., *ATLASFusion* (v1 titled "Sparse BEV Fusion with Self-View Consistency"), arXiv 2509.08421 |
| P18 | `18_reST_2308.13229.pdf` | Cheng et al., *ReST: A Reconfigurable Spatial-Temporal Graph Model for Multi-Camera Multi-Object Tracking*, arXiv 2308.13229 |
| P19 | `19_mcblt_2412.00692.pdf` | Wang et al., *MCBLT: Multi-Camera Multi-Object 3D Tracking in Long Videos*, arXiv 2412.00692 |
| P20 | `20_msmvd_2508.20447.pdf` | Yamane et al., *MSMVD*, arXiv 2508.20447 |
| P21 | `21_feature_pulling_wacv2024.pdf` | Aung et al., *Enhancing Multi-View Pedestrian Detection Through Generalized 3D Feature Pulling* (MVFP), WACV 2024 |
| P22 | `22_grace_2609.16872.pdf` | Sakai et al., *GRACE: Geometry- and Ray-Aware Camera-Efficient Multi-View Pedestrian Tracking*, arXiv 2609.16872 |
| P23 | `23_mv3dt_distributed_2606.13127.pdf` | Hernandez et al., *Fully Distributed Multi-View 3D Tracking in Real-Time*, arXiv 2606.13127 |
| P24 | `24_lmgp_2111.11892.pdf` | Nguyen et al., *LMGP*, CVPR 2022 |
| P25 | `25_booster_shot_2208.09211.pdf` | Hwang, Benz, Kim, *Booster-SHOT*, WACV 2024 |
| P26 | `26_camuvid_cvpr2025.pdf` | Daryani et al., *CaMuViD: Calibration-Free Multi-View Detection*, CVPR 2025 |
| P27 | `27_simplebev_2206.07959.pdf` | Harley et al., *Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?*, ICRA 2023 |
| P28 | `28_botsort_2206.14651.pdf` | Aharon et al., *BoT-SORT*, arXiv 2206.14651 |
| P29 | `29_hybridsort_2308.00783.pdf` | Yang et al., *Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking*, arXiv 2308.00783 |
| P30 | `30_sushi_2212.03038.pdf` | Cetintas et al., *Unifying Short and Long-Term Tracking with Graph Hierarchies* (SUSHI), CVPR 2023 |
| P31 | `31_deep_ocsort_2302.11813.pdf` | Maggiolino et al., *Deep OC-SORT*, arXiv 2302.11813 |
| P32 | `32_centernet_objects_as_points_1904.07850.pdf` | Zhou et al., *Objects as Points* (CenterNet), arXiv 1904.07850 |
| P33 | `33_gmvd_2109.12227.pdf` | Vora et al., *Bringing Generalization to Deep Multi-View Pedestrian Detection* (GMVD), arXiv 2109.12227 |
| P34 | `34_unified_mvmp_tracking_2302.03820.pdf` | Yang et al., *A Unified Multi-view Multi-person Tracking Framework*, CVM |
| P35 | `35_idf1_ristani_1609.01775.pdf` | Ristani et al., *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking* (IDF1), ECCV 2016 (as cited in [P08]) |
| P36 | `36_swa_1803.05407.pdf` | Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization* (SWA), arXiv 1803.05407 |
| P37 | `37_gta_global_tracklet_2411.08216.pdf` | Sun et al., *GTA: Global Tracklet Association for Multi-Object Tracking in Sports*, arXiv 2411.08216 |
| P38 | `38_beyond_centers_bbox_voxel_isprs2026.pdf` | Ali, Mehltretter, Heipke, *Beyond Centers: Bounding-Box Voxel Projection for Multi-View 3D Detection and Tracking*, ISPRS Annals 2026 |
| P40 | `40_clear_mot_bernardin2008.pdf` | Bernardin, Stiefelhagen, *Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics*, EURASIP JIVP 2008 |
| P41 | `41_ksp_berclaz2011.pdf` | Berclaz et al., *Multiple Object Tracking using K-Shortest Paths Optimization*, IEEE TPAMI 2011 |
| P42 | `42_two_stage_mincost_flow_2311.02642.pdf` | Li, Jiang, *An Approach for Multi-Object Tracking with Two-Stage Min-Cost Flow*, arXiv 2311.02642 |
| P43 | `43_arxiv_2602.00450.pdf` | Anderson et al., *Model Optimization for Multi-Camera 3D Detection and Tracking*, arXiv 2602.00450 (checked for overlap; deployment study of Sparse4D, not used) |
| P44 | `44_mv_crowd_tracking_transformer_2604.19318.pdf` | Zhang et al., *Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes*, arXiv 2604.19318 (checked for overlap; criticises WILDTRACK's small test set) |
| P45 | `45_visibility_gated_convgru_visapp2026.pdf` | Sakai, Nakamura, Shimizu, Hotta, *Visibility-Gated ConvGRU for Robust Multi-View Pedestrian Detection and Tracking in BEV Space*, VISAPP 2026, pp. 415–422 |

**Not downloaded.** muSSP (Wang et al., NeurIPS 2019) could not be fetched: the NeurIPS proceedings servers
timed out from this machine. Every muSSP number used here comes from MVFlow [P09], which ran muSSP on WILDTRACK
itself. P42 is a downloaded paper that uses muSSP. The muSSP PDF is at
<https://proceedings.neurips.cc/paper/2019/hash/b1d10e7bafa4421218a51b1e1f1b0ba2-Abstract.html>.
