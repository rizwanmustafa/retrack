# Research methodology 2: View-Diversity-Gated Fusion (VDGF) for multi-view detection and tracking on WILDTRACK

*Every number from the literature below was read from the PDF in [`papers/`](papers/) and is cited as **[Pxx] +
table/section** (index in [`PAPERS.md`](PAPERS.md)). Numbers from our own runs are cited as **[A1]**: the attempt-1
results in `for_claude.zip`, 24 runs, 8 variants × 3 seeds. Numbers from reading this repository are cited as **[code]**
with the file. Anything else is labelled **our choice**.*

---

## 0. Summary

**Goal.** Beat TrackTacular [P01] on WILDTRACK: MODA 92.1, MOTA 91.8, IDF1 95.3 [P01 Tab. 1–2]. Use the paper's own
protocol: last epoch, threshold 0.5, TrackTacular's tracker, nothing tuned on the test frames.

**Method.** The method combines two published lines of work and adds three changes of our own.

| | component | origin |
|---|---|---|
| borrowed | scene-level BEV augmentation (random BEV flips, 50 % of samples) | MVAug [P15 Tab. 2, 4]; our only robust gain in attempt 1 [A1] |
| borrowed | volumetric-guided fusion: a homography-projected BEV branch gated against TrackTacular's volumetric branch | GRACE [P02 §2.2, Eq. 3–5] |
| borrowed | ray conditioning: each camera's ground-plane viewing direction appended to its lifted features | GRACE [P02 §2.3, Eq. 6–8] |
| **new** | **view-diversity gate prior.** The fixed half of the gate uses the *spread of viewing directions* of the cameras that see a cell, not the *number* of cameras | §3.5 |
| **new** | **augmentation-consistent rays.** Rays are computed per sample in the scene-augmented frame, so ray conditioning and scene augmentation can be combined. Ali et al. name this combination as future work [P20 §6] | §3.4 |
| **new (fix)** | **batch-consistent reference frames.** We found and fixed a bug in the TrackTacular code base that projects 3 of 7 views with the wrong frame whenever scene augmentation is used | §3.1 |

**Test in about one hour on one A100.** We train the proposed model M2 with seed 80085 (about 35–40 min) and evaluate
it (about 1 min). We compare it, seed for seed, with the attempt-1 baselines trained with the same seed and schedule.
These baselines already exist, so they do not cost any GPU time. Ablations follow in priority order (§5).

---

## 1. What attempt 1 showed (our data, [A1])

1. **Association is not the bottleneck.** Across runs there are 1–9 identity switches per test sequence, against
   35–55 misses and 25–55 false positives. Offline tracklet linking changed nothing in 71 of 72 evaluations. Any
   remaining gain has to come from detection.
2. **Our retrained TrackTacular is below the paper.** E0 scores MODA 91.14 ± 0.34, MOTA 89.29 ± 0.46 and
   IDF1 93.40 ± 0.88 over 3 seeds, at threshold 0.5. GRACE's authors also retrained it below the paper: 88.50 /
   89.71 / 94.00 [P02 Tab. 1].
3. **Scene augmentation (A2) was the only significant gain.** It reached 91.88 ± 0.24 / 92.19 ± 0.62 / 95.38 ± 0.53,
   with +2.91 MOTA (p = 0.004) and +1.98 IDF1 (p = 0.04) over E0. That matches the paper but does not beat it. It
   traded recall for precision: false negatives went 45 → 54 and false positives 39 → 24 (means over seeds).
4. **A2 together with temporal history (A2A3) collapsed** in 2 of 3 seeds (MODA 87.8, 87.6).
5. **Seed spread is 0.3–1.5 points**, comparable to the 3-run standard deviations in Beyond Centers [P13 Tab. 1]
   (MODA ± 0.6–1.5). One test error is 0.105 points (952 ground-truth detections).

## 2. Literature: what beats TrackTacular on WILDTRACK, and can we use it?

| paper | WILDTRACK result (7 cameras) | how good is the evidence? | usable within an hour in this code base? |
|---|---|---|---|
| **GRACE** [P02] | VGF + Ray + BTR: MODA 92.58, IDF1 94.53, MOTA 91.04 vs its retrained TrackTacular at 88.50 / 94.00 / 89.71 [Tab. 1]. The gain is mostly recall: 91.39 → 96.15 | Built on the TrackTacular code with a *retrained* baseline, so the comparison is controlled. Single runs. The ablation (Tab. 3) is at 2 cameras only | **yes**: the repository already contains both lifting paths (`models/segnet.py`, `models/mvdet.py`) |
| **MVAug** [P15] | Scene + view augmentation and image loss: 93.17 MODA. Scene H-flip alone 91.39 vs 90.86 without augmentation [Tab. 2]. 50 % augmented is best; 100 % gives 90.44 [Tab. 4] | MVDet-style model, single runs. Confirmed on TrackTacular by [A1] over 3 seeds | **yes**: already implemented (A2) |
| ATLASFusion [P03] | 93.6 MODA, 95.9 IDF1, 92.4 MOTA [Tab. 1, 3, 4] | Its baseline row in Tab. 4 is exactly TrackTacular's published numbers (92.1 / 76.2 / 95.3 / 91.8), not a retrained model. Single runs | partly (sparse sampling, per-view loss). Kept as a later option |
| MSMVD [P10] | 94.6 MODA [Tab. 2] (detection threshold 0.4, §4.2) | MVDet-style model with image-FPN and BEV-FPN; no tracking | no: a new architecture, not a module |
| MVDGC [P07] | 94.2 MODA [Tab. 2], 95.8 IDF1 / 92.1 MOTA [Tab. 3] | Query-based detector; lists TrackTacular at 93.2 MODA, which is TrackTacular's *depth-splatting* number in [P01 Tab. 1] | no |
| MV2GF [P08] | builds on a 1.4 B-parameter geometry foundation model (DA3) [Suppl. A] | — | no (compute) |
| PVH [P09] | 93.6 MODA [Tab. 1] | Needs Mask R-CNN segmentations [§4.3]. Its ablations (Tab. 2–3) report MODA ≈ 97, matching its MultiviewX results, so they isolate no WILDTRACK gain | no |
| Attention-aware [P04], MVTrajecter [P05], ModTrack [P06] | 96.1 / 92.7 [P04 Tab. 2]; 96.5 / 94.3 [P05 Tab. 3] (IDF1 / MOTA) | Association methods | no: association is saturated for us (§1.1) |
| Beyond Centers [P13] | 91.0 ± 1.0 MODA [Tab. 1] | 3 seeds; below TrackTacular | used only for the seed spread |

**Choice.** Only GRACE has a *controlled* WILDTRACK gain over a *retrained* TrackTacular in the same code base, and
its gain is in recall. Recall is exactly where scene augmentation lost ground in attempt 1 (§1.3). Combining the two
pairs a precision gain with a recall gain. A plain "A2 + GRACE" combination would still be a mix of other people's
work, so the method adds §3.4 and §3.5, and §3.1 makes the combination possible in the first place.

---

## 3. Method

The base is the repository's TrackTacular: ResNet-18 encoder, bilinear-sampling lifting into a 120 × 4 × 360 voxel
grid, concatenation fusion of the 7 cameras, decoder with centre / offset / previous-frame-offset heads, and the
baseline tracker [code: `models/segnet.py`, `configs/d_wildtrack.yml`]. Temporal history stays **off**, as in the
attempt-1 baselines, because A2 + history collapsed in [A1] (§1.4).

### 3.1 Batch-consistent reference frames (bug fix)

`Segnet.forward` builds every camera's reference transform with `ref_T_global.repeat(S, 1, 1)` [code, all four
models]. That call *tiles* the batch, while `utils.basic.pack_seqdim` orders cameras sample by sample (index
b·S + s). Camera i therefore gets sample (i mod B)'s reference frame. With batch size 2 and 7 cameras, **3 of the 7
views of every training sample are lifted with the other sample's frame.**

- Without scene augmentation the two frames differ only by the random centre offset of ±0.25 grid cells
  (±0.6 cm) [code: `datasets/pedestrian_dataset.py`], so the bug is harmless. The attempt-1 baseline E0 is
  therefore still a valid baseline.
- With scene augmentation the two samples often have different BEV flips, so those three views are lifted
  mirror-imaged.
- At test time the batch size is 1, which is why the bug never showed up in evaluation.

This is consistent with A2's lost recall and the A2A3 collapse in [A1]. The fix is `repeat_interleave(S, dim=0)`.
Its own effect is measured by run **A2f** (§5).

### 3.2 Scene augmentation (borrowed: MVAug [P15])

With probability 0.5, the BEV reference frame, the ground-truth positions and the previous-frame positions are
mirrored by one of the three BEV flips [code: `PedestrianDataset.sample_scene_transform`, mode `flip`]. The 50 %
rate is MVAug's best [P15 Tab. 4]. MVAug found H-flip helpful and V-flip not [P15 Tab. 2]. We keep all three flips,
because that is the variant attempt 1 validated over 3 seeds [A1].

### 3.3 Volumetric-guided fusion (borrowed: GRACE [P02 §2.2])

Let **V** be TrackTacular's volumetric BEV feature (output of `bev_compressor`, 256 channels). We add a homography
branch **H**: each camera's image features are warped onto the ground plane, as MVDet does [P18; code:
`models/mvdet.py`], and the 7 warped maps are fused by a 3×3 conv over their concatenation (*our choice*: the same
form as TrackTacular's own camera compressor; GRACE does not give its layer shapes). The fused feature is

  g = ½ σ(Conv1×1[H, V, p]) + ½ (1 − p), Z = g ⊙ V + (1 − g) ⊙ H  [P02 Eq. 3–5],

where p is the prior map (§3.5). The learned half starts at 0.5 (zero-initialised conv, *our choice*). Z replaces V
as the input to the temporal/decoder stage. Parameter count: the H compressor adds about 2.1 M parameters. GRACE
reports 2.16 M for its VGF [P02 §3], which is consistent.

### 3.4 Augmentation-consistent ray conditioning (borrowed mechanism, new placement)

GRACE appends to each camera's lifted feature a 2-channel map: the unit ground-plane direction r from the camera
centre to every BEV cell. A small network ρ then compresses the result per camera before fusion [P02 Eq. 6–8].
GRACE computes the map "once from calibration" [P02 §2.3]. Under scene augmentation that is wrong: the feature
volume is mirrored, but a fixed ray map is not, so the rays would contradict the geometry half of the time.

We compute r **per sample, from the scene-augmented reference frame** (`Segnet.ground_rays`). A flipped sample
therefore gets flipped rays. Ali et al., who add viewing directions to multi-view features, name joint augmentation
of views and grid as future work for exactly this kind of model [P20 §6].

ρ is a 1×1×1 conv shared by the cameras. It is initialised to pass the image features through unchanged with zero
ray weights, so training starts from the baseline (*our choice*). Its output is masked to each camera's field of
view, so cells a camera cannot see stay zero, as in TrackTacular's lifting.

### 3.5 View-diversity gate prior (new)

GRACE's fixed prior is (1 − m), where m is the fraction of cameras that see a cell [P02 Eq. 4]. The stated reason is
that "in overlapping regions, multiple homography features can reinforce a correctly localized response", while
GRACE also notes that "coverage does not identify the direction along which a projection error spreads" [P02
§2.2–2.3].

**Geometry.** A homography projects a standing person onto the ground as a streak (a "shadow") that starts at the
feet and points away from the camera along r [P02 Fig. 3; P03 Fig. 3]. Two cameras on the same side produce
overlapping streaks: the error is reinforced, not corrected. Cameras looking from different directions produce
streaks that meet only at the feet. Whether the homography branch is reliable therefore depends on the *spread of
directions*, which a camera count cannot capture.

**Definition.** With w_n(x) the fraction of camera n's voxels above cell x that it sees, and r_n(x) its ground ray:

  d(x) = 1 − ‖Σ_n w_n r_n‖ / Σ_n w_n  (d = 0 when no camera sees x)

- d = 0 for one camera, or several cameras all looking the same way.
- d grows as the directions spread out. For two cameras at angle φ, d = 1 − cos(φ/2).

The gate uses p = d in place of GRACE's m, both as the gate input and in the fixed half. A larger d shifts weight to
the homography branch where its errors cancel, and a smaller d shifts weight to the volumetric branch. d has no
parameters and costs one weighted sum per cell (`models/fusion.py: view_diversity`).

**Novelty check.** We searched for direction-diversity or angular-spread gating in multi-view BEV fusion. The
downloaded papers contain none: GRACE [P02] and Ali et al. [P20] *feed* viewing directions to the network, but
neither uses their spread to weight fusion branches. A web search in November 2026 found nothing either. This is the
claim the M2 vs M2c ablation tests.

### 3.6 Unchanged

Training schedule (30 epochs, Adam, one-cycle, max LR 1e-3, effective batch 16), input 720 × 1280, last-epoch
checkpoint, threshold 0.5, TrackTacular's tracker [code: `configs/t_retrack.yml`]. These are the same as the
attempt-1 baselines, so any difference comes from §3.1–3.5.

---

## 4. Hypotheses and decision rule (fixed before running)

All comparisons use the **primary protocol**: threshold 0.5 and TrackTacular's tracker, with nothing tuned on the
test frames. This is the cell `thr0.5|tt`, the same protocol the paper uses.

| | hypothesis | test |
|---|---|---|
| H1 | M2 beats TrackTacular's MODA (92.1) | M2 vs [P01 Tab. 1] |
| H2 | M2 keeps MOTA ≥ 91.8 and IDF1 ≥ 95.3 | M2 vs [P01 Tab. 2] |
| H3 | The view-diversity prior beats GRACE's coverage prior | M2 vs M2c, same seed |
| H4 | VGF + rays add to scene augmentation | M2 vs A2f, same seed |
| H5 | The batch-order fix helps scene augmentation | A2f vs attempt-1 A2, same seed |

**Rule.**

- A difference counts as **"supported"** only if it exceeds the attempt-1 baseline seed spread on that metric (E0
  std: MODA 0.34, MOTA 0.46, IDF1 0.88) *and* has the same sign on every seed run.
- With one seed, a result is reported as **"screening: promising / not promising"**, never as confirmed.
- "Beats TrackTacular" requires a mean over ≥ 3 seeds above the paper's numbers. The paper's numbers are single runs
  without variance.

**Honest expectation.**

- GRACE's controlled gain at 7 cameras was +4.08 MODA, but over a weak baseline (88.50), and it ended at 92.58, just
  above the paper's 92.1 [P02 Tab. 1].
- Our A2 baseline already sits at 91.88 ± 0.24.
- So the realistic target is about +0.5 to +1.5 MODA, with MOTA/IDF1 at or above the paper. We cannot promise it:
  WILDTRACK is close to saturation [P01 §4.4, §5].

## 5. Experiments (about one hour, then in priority order)

| priority | run | what it answers | time on an A100 |
|---|---|---|---|
| 1 | **M2, seed 80085** (default) | H1, H2 (screening). Paired with attempt-1 E0 and A2 at seed 80085 | ~3 min preflight + ~35–40 min training + ~1 min evaluation |
| 2 | M2c, seed 80085 | H3 (our prior vs GRACE's) | +35–40 min |
| 3 | A2f, seed 80085 | H4, H5 (VGF vs the bug fix alone) | +30 min |
| 4 | M2, seeds 1 and 2 | confirms H1/H2 over 3 seeds | +70–80 min |
| 5 | VD, seed 80085 | scene augmentation's share | +35–40 min |

`--parallel 2` trains two runs at once on one GPU. It was *not* timed; expect each run to slow down.

**Outputs.**

- `REPORT.md`:
  - the primary-protocol table against the paper, GRACE and the attempt-1 baselines;
  - an oracle-threshold table, labelled as an upper bound;
  - same-seed paired deltas with false-negative / false-positive counts.
- `for_claude.zip`: all metrics and logs, without checkpoints.

## 6. Threats to validity

- **One seed in the first hour.** Seed spread is 0.3–1.5 points (§1.5). Only priority 4 can confirm "beats
  TrackTacular".
- **Baselines trained before the bug fix.** E0 is unaffected (±0.6 cm frame differences). Attempt-1 A2 *is*
  affected; that is why A2f exists.
- **GRACE's component ablation is at 2 cameras** [P02 Tab. 3]. At 7 cameras only the full method is reported
  [P02 Tab. 1].
- **GRACE's 2-camera TrackTacular MODA is internally inconsistent**: 83.54 in Tab. 1, 82.53 in Tab. 3 and §4.1. We
  use only its 7-camera row and its ablation deltas.
- **Layer shapes are ours.** The shapes of ρ, the H compressor and the gate are not given in [P02]; ours are
  listed in §3.3–3.4.
- **The paper's numbers are single runs.** [P01] reports no variance, so a small lead over it is within seed noise.

## 7. How to run

See [`RUN.md`](RUN.md).

## 8. Screening results, seed 80085 (2026-11-10), and amendment 1

All runs use the primary protocol (`thr0.5|tt`), commit 73e1ed1, an A100, and about 24 min of training each.

| run | MODA | MOTA | IDF1 | det FN / FP | tracker misses / FP |
|---|---|---|---|---|---|
| TrackTacular paper [P01 Tab. 1–2] | 92.10 | 91.80 | 95.30 | – | – |
| attempt-1 E0 (retrained TrackTacular) | 90.76 | 89.08 | 93.40 | 47 / 41 | – |
| attempt-1 A2 (scene aug, old batch order) | 92.02 | 92.65 | 95.42 | 53 / 23 | – |
| A2f (scene aug, fixed order) | 91.49 | 90.76 | 94.51 | 45 / 36 | 44 / 36 |
| M2 (A2f + VGF + rays + diversity prior) | 91.70 | 90.13 | 94.55 | 45 / 34 | 41 / 45 |
| M2c (A2f + VGF + rays + coverage prior) | 92.23 | 89.92 | 94.61 | 34 / 40 | 42 / 49 |

All three are screening results under the §4 rule, from one seed each.

- **H1:** only M2c is above 92.1, by 0.13 (one detection), so this is inconclusive. **H2:** not met by any run.
- **H3, not promising.** M2 − M2c is −0.53 MODA, +0.21 MOTA, −0.06 IDF1. GRACE's coverage prior detects better than our diversity prior.
- **H4, mixed.** VGF with the coverage prior gives +0.74 MODA over A2f, mainly from 11 fewer misses. But it gives −0.84 MOTA. With VGF the TrackTacular tracker makes more false positives than the detector does (49 vs 40), while A2f has 36 vs 36. That tracker matches each detection's predicted previous position to the existing tracks (`tracking/multitracker.py:216–218`), so the offset head is the likely cause. This has not been verified.
- **H5, reversed.** A2f is below attempt-1 A2 on every metric: −0.53 MODA, −1.89 MOTA, −0.90 IDF1. Its MOTA is also below all three A2 seeds (91.49, 92.44, 92.65).
  - The two training configs are identical apart from the seed and paths (a diff of `config.yaml` is empty).
  - With VGF off, the code path differs only by the order fix.
  - The old order therefore behaved like a useful augmentation. With batch 2, cameras 1, 3 and 5 of each sample are lifted in the other sample's scene-augmented frame while the targets stay put.
  - The "same seed" pairing only fixes the data order. GPU nondeterminism remains, so this needs a second seed.

**Amendment 1 (post hoc).** This hypothesis came from the data above, so it must be confirmed on new runs. It has not
yet been checked against the literature for novelty.

- **H6:** view mixing (`--model.view_mix true`, training only) adds to M2c's detection gain.
  - Test: M2cx − M2c at seed 80085.
  - Confirmation: A2f vs attempt-1 A2 at seed 2.

`view_mix` reproduces attempt 1's order exactly and has no effect at test time.

## References

See [`PAPERS.md`](PAPERS.md). Each PDF is in [`papers/`](papers/) (kept locally, not in git).
