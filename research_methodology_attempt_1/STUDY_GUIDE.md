# Study guide: understanding, running and defending ARL attempt 1

This guide covers four things: what you need to know about the current state of the work, whether you can
defend it, the questions a panel is likely to ask (with answers), and a step-by-step way to learn everything
behind it. Paper IDs such as [P08] refer to the PDFs in `papers/` (see `PAPERS.md`).

---

## 1. Things to know about the current state

| # | fact | why it matters |
|---|---|---|
| 1 | **The code has never been run.** Every file compiles, but none of it has been executed. | The first `MODE = 'quick'` run in Colab is the real test. If it crashes, share `pipeline.log` (in `arl_quick/`) and the error. |
| 2 | **The work is not committed yet.** Your commits are GPG-signed and I can't type the passphrase. | Run the commit from `RUN.md` / the chat in your own terminal, then `git push`. Until then the GitHub branch `feat/arl-attempt-1` points at the old `main` commit, and Colab would clone code without ARL. |
| 3 | **A4 (architecture change) is not implemented.** | The GRACE paper [P22] does not describe its fusion networks (ρ, ψ_vol) in enough detail to rebuild them. The methodology only runs A4 after A1–A3 are confirmed anyway. |
| 4 | **The PDFs are not in git.** They are in `.gitignore`. | 307 MB of publishers' PDFs should not be pushed to GitHub. They stay on your disk in `papers/`. |
| 5 | **One change also affects the baseline.** The previous-frame ground truth now gets the same scene transform as the current frame. | This was a bug, harmless before because the transform was a ~6 mm shift. It matters once real scene augmentation (A2) is on. The baseline's effect is negligible, but you should be able to say so if asked. |
| 6 | **The muSSP paper could not be downloaded** (NeurIPS servers timed out). | Its WILDTRACK numbers come from MVFlow [P09], which ran it. Say that if asked for the source. |
| 7 | **`README.md` is corrupted** in the repo (4.6 MB: the timing table was inserted after every character). | Not caused by this work. A recovered copy exists, but the file was not changed. Fix it in a separate commit before showing the repo to anyone. |
| 8 | **Your baseline will be below the TrackTacular paper.** | An independent group retrained TrackTacular and got MODA 88.50 vs the paper's 92.1 [P22 vs P08]. The released checkpoints don't work with the public code (README, issues #2 and #7). Always compare against **your own** retrained baseline E0. |
| 9 | **WILDTRACK is small and noisy.** The test set is 40 frames, and seed-to-seed spread is about ±1 MODA [P38]. | Most published gains are about that size. That is why every result is the mean over 3 seeds and decided by a rule fixed in advance. |
| 10 | **The full study takes about 21 h on an A100** (`VARIANTS = 'all'`, 3 seeds). `core` takes about 12 h. | Colab sessions can stop. Results are on Drive and the cell resumes, so just run it again. |
| 11 | **The "best checkpoint" in `best/` is chosen on test-frame metrics.** | Use it as the model to keep or demo. In a thesis or paper, report the **mean ± std** from `REPORT.md`, never the best seed. |
| 12 | **Offline stages (C2, C2b, C3) use future frames.** | Compare them with offline methods (MVFlow + muSSP, MCBLT). The online-only rows (`tt`, `byte` cells) are in the same report. |

---

## 2. Can you defend this in front of a panel with basic CV and deep learning?

**Yes, with about 3–4 weeks of focused preparation (§4).** Three reasons it's defensible:

1. **No step is exotic.** It uses heatmap detection (CenterNet), projecting camera features to a bird's-eye view
   (BEV), data augmentation, a Kalman filter, the Hungarian algorithm, and a t-test. All of these are taught in
   CV / DL courses.
2. **Every design choice has a citation with a table number.** You never have to say "I thought it would work".
   You can say "3DROM, Table 3: 88.2 → 92.3 MODA on WILDTRACK".
3. **The experiment can fail gracefully.** Every addition gets a KEEP/DROP verdict from a rule fixed before
   running. A DROP is still a finding ("A2 does not help on WILDTRACK under our protocol"). Panels respect that
   more than a list of wins.

**Where you are weakest, and what to do about it:**

| weakness | how to handle it |
|---|---|
| Novelty: no new architecture | Say it plainly. The contribution is (a) a careful, evidence-backed combination; (b) a rigorous protocol (clean vs oracle, seeds, pre-registered rule, add-one *and* leave-one-out); (c) three problems found in the baseline code (temporal history switched off for WILDTRACK, previous-frame GT not transformed, cache misalignment under augmentation). |
| Small dataset | Acknowledge it, cite TrackTacular's own remark that WILDTRACK is saturating [P08 §4.4], and point to the seed statistics. |
| You didn't write the original codebase | Know the files listed in §4 well enough to point at the line where each addition happens. |

---

## 3. Likely panel questions and how to answer them

**Q1. What problem are you solving, in one sentence?**
Detect and track pedestrians on the ground plane from 7 overlapping, calibrated cameras, and improve detection
accuracy (MODA) and tracking accuracy (MOTA, IDF1) over a strong retrained baseline (TrackTacular).

**Q2. Why fuse in the bird's-eye view instead of detecting in each camera?**
A person hidden in one camera is usually visible in another. Projecting all cameras' features onto one ground
grid lets the network combine the evidence before deciding. TrackTacular and EarlyBird beat late-fusion trackers
this way [P07, P08].

**Q3. What do MODA, MOTA and IDF1 measure, and how do they differ?**
MODA = 1 − (misses + false positives) / number of GT people, per frame (match radius 0.5 m). MOTA additionally
subtracts identity switches (radius 1 m). IDF1 is an F1 score over *identities* across the whole sequence: it asks
how much of each person's trajectory carries one consistent ID. MOTA barely punishes a track that switches ID
once. IDF1 punishes it heavily (worked example in §5).

**Q4. Why can't your tracking stages change MODA?**
MODA is computed from per-frame detections *before* tracking (see `world_track.py`, `test_step`). Only the
detector (A1–A3, A5) and the threshold (B1) move MODA.

**Q5. Isn't tuning on the test frames cheating?**
That's why there are two protocols. *Clean*: train a separate model on frames 0–323, tune on 324–359, freeze the
parameters, then evaluate the real model on 360–399. *Oracle*: tune on the test frames; it is always labelled as
an upper bound. Claims are made with the clean numbers.

**Q6. Why 3D random occlusion and not ordinary random erasing?**
Erasing a patch in one image breaks the geometric consistency between views. 3DROM places a person-sized cylinder
on the ground and paints its projection into *every* camera, so the occlusion is physically consistent. On
WILDTRACK it beat per-view random erasing, 93.5 vs 92.0 MODA [P05 Tab. 6]. The labels stay, so the network has
to find the occluded person from the cameras that still see them.

**Q7. Why must scene augmentation also transform the previous frame?**
The model predicts each person's position in the previous frame (the tracking offset). If the current frame is
flipped but the previous-frame target is not, the target points the wrong way. The temporal cache has the same
problem: the cached BEV of frame t−1 was computed in t−1's (possibly flipped) frame. So the code stores each
frame's transform and resamples the cached BEV into the current frame (`align_prev_bev` in `world_track.py`).

**Q8. TrackTacular already has temporal fusion. What's new?**
For WILDTRACK it is switched off in the shipped config (`use_temporal_cache: False`, upstream commit `f17fa5b`).
Its only published ablation is on MultiviewX, where it raised MODA and IDF1 but lowered MOTA [P08 Tab. 4].
A 2026 paper did test temporal memory on TrackTacular on WILDTRACK and found gains (plain ConvGRU: MODA 92.1 → 92.4,
IDF1 95.3 → 96.3, MOTA 91.8 → 92.5 [P45 Tab. 3]), but with a new recurrent module and single runs. ARL switches on
TrackTacular's own mechanism and tests it with and without augmentation, over seeds.

**Q9. Why ResNet-18? Bigger backbones usually help.**
Not on WILDTRACK: EarlyBird measured ResNet-18 91.2 MODA vs ResNet-50 89.6 vs Swin-T 89.5 [P07 Tab. 4]. With 360
training frames a bigger encoder over-fits. Simple-BEV's opposite result was on nuScenes [P27], a much larger
dataset.

**Q10. Why do you expect offline linking to help so much?**
Two papers held the detector fixed and changed only the association. MVFlow: with MVDeTr's detections, DeepSORT
gave 24.0 MOTA / 56.3 IDF1 and muSSP min-cost flow gave 89.4 / 90.7 [P09 Tab. 2]. MCBLT: with EarlyBird's own
detections, 89.5 / 92.3 became 92.6 / 95.6 [P19 Tab. 3]. An online tracker must decide frame by frame. An
offline one can see that a track that "died" at t reappears at t+3 and join the two.

**Q11. Your linking is "exactly optimal". Why?**
Each tracklet can have at most one successor and one predecessor, and link costs add up along a chain. Finding
the cheapest set of chains is a minimum-cost path cover in a directed acyclic graph (links only go forward in
time). That problem reduces to an assignment problem, which the Hungarian algorithm solves exactly. The code
builds the standard 2n × 2n matrix with a "no link" option for every end and every start (`link_tracklets` in
`tracking/offline.py`).

**Q12. And the detection-level min-cost flow (C2b)?**
Each detection is a node with a cost: negative if its score is above the threshold, positive otherwise. Entry and
exit edges cost c_b, so a track must be "worth" more than that to exist. Transition edges cost the distance to
the next detection's predicted previous position. It is solved as a linear program. The constraint matrix of a
flow network is totally unimodular, so the simplex solution is integral: each edge is used 0 or 1 times.

**Q13. What does Gaussian-smoothed interpolation do, and can it hurt?**
It fills short gaps (≤ 1–3 frames) inside a linked track by linear interpolation, then smooths the whole track
with a Gaussian-process posterior mean (StrongSORT, Eq. 12–15 [P12]). It can hurt: an interpolated point that is
more than 1 m from the true person counts as a false positive. That is why it only fills gaps *inside* linked
tracks, never extrapolates past a track's ends, and has its own on/off test (C3).

**Q14. How do you know a gain is real and not seed luck?**
Three seeds per configuration, and two kinds of test. Training-time additions compare independently trained
models, so they use Welch's t-test. Post-hoc additions compare the same model with a switch on and off, so they
use a paired t-test. The decision rule was fixed before running: keep an addition only if a target metric
improves by more than one baseline standard deviation and no primary metric drops by more than one.

**Q15. Why test each addition both "add-one" and "leave-one-out"?**
Additions interact. MVAug found the image loss helped together with view augmentation but hurt with scene
augmentation alone [P06 Tab. 3]. Add-one asks "does it help a plain model?". Leave-one-out asks "does the final
method still need it?". If the two disagree, that is an interaction worth reporting.

**Q16. Why is your baseline lower than the TrackTacular paper?**
The released WILDTRACK checkpoints give MODA ≈ 0 with the public code (README; TrackTacular issues #2 and #7), so
the baseline is retrained. An independent group also got 88.50 instead of 92.1 [P22 Tab. 1]. Comparisons are
therefore against our own retrained E0, trained with the identical schedule.

**Q17. Why didn't you use the newest learned trackers (MVTrajecter, UMPN)?**
They report the best WILDTRACK numbers (MVTrajecter 94.3 MOTA / 96.5 IDF1 [P16]), but each needs a newly trained
association network. Attempt 1 first establishes which cheap, well-understood additions help. Learned
association is the planned attempt 2.

**Q18. Any ethical issues?**
WILDTRACK has been criticised for not having the consent of the people recorded; EarlyBird's authors mention it
[P07, "Ethical Impact"]. Pedestrian tracking can also be misused for surveillance. Mention both briefly and note
that MultiviewX is a synthetic alternative.

**Q19b. Has someone already done exactly this?**
Not that I could find. A targeted search found no paper combining these additions on TrackTacular, or testing each
one with add-one and leave-one-out ablations over seeds (METHODOLOGY.md §9). The closest work is one group's
series (GRACE, ATLASFusion, Visibility-Gated ConvGRU [P22, P17, P45]), which adds one new module per paper. Say
clearly that the ingredients are known on purpose, and that the contribution is the combination, the interaction
fixes and the protocol.

**Q19. What would you do if nothing worked?**
Report it. A DROP verdict on an addition that the literature reports as helpful is a useful finding about
reproducibility on a saturated benchmark. Then use the cached detections to analyse *which* errors remain
(misses vs false positives vs ID switches; the report gives raw counts).

---

## 4. How to learn all this (about 3–4 weeks)

Work through the stages in order. After each one, check yourself with its questions, out loud or on paper. If
you can't answer one, re-read before moving on.

### Stage 1: metrics (2–3 days)
- **Read:** CLEAR MOT metrics [P40] §1–3; IDF1 [P35] §1–4 (definitions only).
- **Do:** compute the §5 examples by hand. Then open `evaluation/retrack_eval.py` and find `detection_metrics`
  (the Hungarian matching at 0.5 m) and `mot_summary`.
- **Check yourself:** Why can MOTA be negative? Why does one ID switch cost little in MOTA but a lot in IDF1?
  What does the 0.5 m vs 1 m radius change?

### Stage 2: detection as heatmaps (2–3 days)
- **Read:** CenterNet "Objects as Points" [P32] §3–4 (heatmap, focal loss, offset head, max-pool NMS).
- **Code:** `utils/decode.py` (3×3 max-pool NMS, top-K) and `models/loss.py`.
- **Check yourself:** Why max-pool instead of normal NMS? What does the offset head correct? What does the
  detection threshold trade off?

### Stage 3: multi-view geometry and BEV (4–5 days)
- **Background:** the pinhole camera model, K[R|t], and homographies. Any CV textbook chapter, or TrackTacular
  §3.1 [P08].
- **Read:** MVDet [P02] (projecting features to the ground), Simple-BEV [P27] (bilinear sampling: voxels "pull"
  features from images), TrackTacular [P08] in full.
- **Code:** `models/segnet.py` `forward` (encoder → `unproject_image_to_mem` → fusion → BEV compressor →
  temporal concat → decoder) and `world_track.py` (`forward`, `loss`, `test_step`).
- **Check yourself:** Why does homography projection create "shadows" for people far from the camera
  [P03, P08 §2]? What is the voxel grid here (120 × 4 × 360, 10 cm × 10 cm × 50 cm)? What does the
  previous-frame offset head predict?

### Stage 4: the three detector additions (3 days)
- **Read:** 3DROM [P05] §3.2 and Tables 3, 5, 6; MVAug [P06] §3 and Tables 2–4; TrackTacular Table 4 [P08].
- **Code:** `datasets/pedestrian_dataset.py` (`sample_occluders`, `paint_occluders`, `sample_scene_transform`,
  `__getitem__`) and `world_track.py` (`load_cache`, `store_cache`, `align_prev_bev`).
- **Check yourself:** Why must the occluders be the same in every view? Why must GT(t−1) be transformed? What
  would go wrong in the temporal cache without `align_prev_bev`?

### Stage 5: tracking (4–5 days)
- **Background:** Kalman filter (predict / update with constant velocity); Hungarian algorithm (minimum-cost
  assignment). Watch one lecture or tutorial on each.
- **Read:** ByteTrack [P10] §3 (two-threshold association); StrongSORT [P12] §III (AFLink, GSI); MVFlow [P09] §3.5
  and Table 2; KSP [P41] §3 (tracking as min-cost flow).
- **Code:** `tracking/multitracker.py` (`JDETracker.update`, `ByteBEVTracker`), then `tracking/offline.py`.
- **Check yourself:** Why can low-score detections continue a track but not start one? Why is tracklet linking a
  path cover, and why does that reduce to assignment? Why are flow LP solutions integral?

### Stage 6: experiment design and statistics (2 days)
- **Learn:** ablation studies; mean ± std over seeds; Welch's t-test (independent groups with unequal variances)
  vs the paired t-test (the same model measured twice); why you fix the decision rule before seeing results.
- **Code:** `evaluation/arl_eval.py` (`tune`, `all_cells`, `run_cell`) and `evaluation/arl_report.py`
  (`training_rows`, `post_hoc_rows`, `verdict`).
- **Check yourself:** Why are training-time comparisons unpaired but post-hoc ones paired? What does "one baseline
  std" mean in the decision rule?

### Stage 7: rehearse (ongoing)
- Open `methodology_diagram.drawio`. For **every box**, say aloud in under 30 seconds: what it does, which paper
  and number justify it, and what result would make you drop it.
- Answer the 19 questions in §3 without looking.
- Once results exist, explain `report/REPORT.md` line by line: the ladder, each verdict, and the raw FN / FP / IDSW
  counts.

**If you only have one week:** do Stage 1, TrackTacular [P08], Stage 4, the ByteTrack and MVFlow parts of
Stage 5, and Stage 7.

---

## 5. Cheat sheet

### Worked metric examples
- **MODA.** 100 GT people over the test frames, 5 missed, 3 false detections:
  MODA = 1 − (5 + 3) / 100 = **92 %**.
- **MOTA.** Same, plus 2 identity switches: MOTA = 1 − (5 + 3 + 2) / 100 = **90 %**.
- **IDF1 vs MOTA.** One person is present for 10 frames and always detected, but the tracker gives frames 1–6
  ID *a* and frames 7–10 ID *b*.
  - MOTA: 1 switch, so 1 − 1/10 = **90 %**.
  - IDF1: the best one-to-one identity match is person ↔ *a*. IDTP = 6; IDFN = 4 (the person's frames under *b*);
    IDFP = 4 (*b*'s frames, unmatched). IDF1 = 2·6 / (2·6 + 4 + 4) = **60 %**.
  - This is why offline linking (joining *a* and *b*) mostly shows up in IDF1.

### Formulas
- MODA = 1 − (FN + FP) / N_gt, with matches within r = 0.5 m.
- MOTA = 1 − (FN + FP + IDSW) / N_gt, with r = 1 m.
- IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN).
- Kalman (constant velocity): state [x, y, vx, vy]; predict x ← x + v; update by blending the prediction with the
  measurement according to their uncertainties.
- GSI: P* = K(F*, F) (K(F, F) + σ² I)⁻¹ P, with an RBF kernel of length scale λ = τ · log(τ³ / l) [P12 Eq. 14–15].
- Link cost (C2): w_m · d / gate + w_t · (Δt − 1) + w_a · (1 − cos(appearance)); "no link" costs c0.

### Glossary
- **BEV**: bird's-eye view, a top-down grid of the ground plane.
- **Lifting**: moving image features into 3D/BEV (homography, bilinear sampling, depth splatting, deformable
  attention).
- **POM / heatmap**: per-cell probability that a person stands there.
- **Tracklet**: a short, reliable piece of a track.
- **Online vs offline**: whether a frame's decision may use future frames.
- **Ablation**: removing or adding one component to measure its effect.
- **Seed**: the random initialisation; different seeds give different results, so report the spread.

### The numbers worth memorising
| what | number | source |
|---|---|---|
| TrackTacular paper (WILDTRACK) | MODA 92.1, MOTA 91.8, IDF1 95.3 | [P08 Tab. 1–2] |
| TrackTacular retrained by others | MODA 88.50, MOTA 89.71, IDF1 94.00 | [P22 Tab. 1] |
| 3D occlusion alone on MVDet | 88.2 → 92.3 MODA | [P05 Tab. 3] |
| view + scene augmentation | 90.86 → 93.17 MODA (single runs) | [P06 Tab. 2–4] |
| online → offline, same detections | MOTA 24.0 → 89.4, IDF1 56.3 → 90.7 | [P09 Tab. 2] |
| online → offline, same detections | MOTA 89.5 → 92.6, IDF1 92.3 → 95.6 | [P19 Tab. 3] |
| seed spread on WILDTRACK | about ±0.6–1.5 MODA, ±0.4–1.2 IDF1 | [P38 Tab. 1] |
| best published tracking (2025) | MOTA 94.3, IDF1 96.5 | MVTrajecter [P16 Tab. 3] |
| temporal memory on TrackTacular (2026) | MOTA 95.1, IDF1 97.0, MODA 93.9 | Visibility-Gated ConvGRU [P45] |

---

## 6. After the run

1. Read `report/REPORT.md`: the ladder first, then the verdict table.
2. Give me `for_claude.zip` (or the whole `arl_<mode>` folder). It has every metric, the cached detections, the
   tuned parameters and the training logs, so I can analyse the errors, re-tune or plot without a GPU.
3. Additions marked DROP stay in the write-up as negative results. Additions marked KEEP become the method you
   present.
