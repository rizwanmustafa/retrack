# Paper index

Every PDF was downloaded on 2026-11-10 into `papers/`, and its title was checked against the first page. The PDFs
are kept locally and not in git (about 200 MB). To fetch them again, use the URL column.

| ID | file | paper | used for | URL |
|---|---|---|---|---|
| P01 | 01_tracktacular_2403.12573.pdf | Teepe et al., *Lifting Multi-View Detection and Tracking to the Bird's Eye View* (TrackTacular), CVPRW 2024 | **target numbers** Tab. 1–2; saturation §4.4, §5 | https://arxiv.org/pdf/2403.12573 |
| P02 | 02_grace_2609.16872.pdf | Sakai et al., *GRACE: Geometry- and Ray-Aware Camera-Efficient Multi-View Pedestrian Tracking*, 2026 | **VGF** Eq. 3–5, **rays** Eq. 6–8; 7-camera results Tab. 1; ablation Tab. 3 | https://arxiv.org/pdf/2609.16872 |
| P03 | 03_atlasfusion_scfusion_2509.08421.pdf | Toida et al., *ATLASFusion*, 2025 | comparison; streak illustration Fig. 3; Tab. 4 baseline = published numbers | https://arxiv.org/pdf/2509.08421 |
| P04 | 04_attention_aware_mvtrack_2504.03047.pdf | Alturki et al., *Attention-Aware Multi-View Pedestrian Tracking*, 2025 | comparison (association) Tab. 2–3 | https://arxiv.org/pdf/2504.03047 |
| P05 | 05_mvtrajecter_2509.01157.pdf | Yamane et al., *MVTrajecter*, ICCV 2025 | comparison (association) Tab. 3 | https://arxiv.org/pdf/2509.01157 |
| P06 | 06_modtrack_2603.15812.pdf | Iyer et al., *ModTrack*, 2026 | comparison (association); led us to P20 | https://arxiv.org/pdf/2603.15812 |
| P07 | 07_mvdgc_2607.00273.pdf | Phan et al., *MVDGC*, TMLR 2026 | comparison Tab. 2–3 | https://arxiv.org/pdf/2607.00273 |
| P08 | 08_mv2gf_2608.20639.pdf | Yamane et al., *MV2GF*, 2026 | comparison (foundation model, Suppl. A) | https://arxiv.org/pdf/2608.20639 |
| P09 | 09_prob_occupancy_volume_2503.10982.pdf | Alturki et al., *Enhanced Multi-View Pedestrian Detection Using Probabilistic Occupancy Volume*, 2025 | comparison Tab. 1 | https://arxiv.org/pdf/2503.10982 |
| P10 | 10_msmvd_2508.20447.pdf | Yamane, *MSMVD*, 2025 | comparison Tab. 2, §4.2 | https://arxiv.org/pdf/2508.20447 |
| P11 | 11_umpn_2507.08494.pdf | Engilberge et al., *One Graph to Track Them All*, 2025 | background (association) | https://arxiv.org/pdf/2507.08494 |
| P12 | 12_mcblt_2412.00692.pdf | Wang et al., *MCBLT*, 2024 | background (association) | https://arxiv.org/pdf/2412.00692 |
| P13 | 13_bbox_voxel_isprs2026.pdf | Ali et al., *Beyond Centers: Bounding-Box Voxel Projection…*, ISPRS Annals 2026 | **seed spread** Tab. 1 (mean ± std over 3 runs) | https://isprs-annals.copernicus.org/articles/XI-2-2026/197/2026/isprs-annals-XI-2-2026-197-2026.pdf |
| P14 | 14_3drom_2207.10895.pdf | Qiu et al., *3D Random Occlusion and Multi-Layer Projection*, 2022 | background (attempt 1's A1) | https://arxiv.org/pdf/2207.10895 |
| P15 | 15_mvaug_2210.10756.pdf | Engilberge et al., *Two-level Data Augmentation for Calibrated Multi-view Detection* (MVAug), WACV 2023 | **scene augmentation** Tab. 2 (flips), Tab. 4 (50 %) | https://arxiv.org/pdf/2210.10756 |
| P16 | 16_earlybird_2310.13350.pdf | Teepe et al., *EarlyBird*, WACVW 2024 | background | https://arxiv.org/pdf/2310.13350 |
| P17 | 17_wildtrack_1707.09299.pdf | Chavdarova et al., *The WILDTRACK Multi-Camera Person Dataset*, 2017 | dataset | https://arxiv.org/pdf/1707.09299 |
| P18 | 18_mvdet_2007.07247.pdf | Hou et al., *Multiview Detection with Feature Perspective Transformation* (MVDet), ECCV 2020 | **homography branch** | https://arxiv.org/pdf/2007.07247 |
| P19 | 19_simplebev_2206.07959.pdf | Harley et al., *Simple-BEV*, ICRA 2023 | background (TrackTacular's bilinear lifting) | https://arxiv.org/pdf/2206.07959 |
| P20 | 20_viewing_direction_ali2025.pdf | Ali et al., *Integrating Viewing Direction and Image Features for Robust Multi-View Multi-Object 3D Pedestrian Tracking*, ISPRS Annals 2025 | **novelty check** for §3.4–3.5: viewing direction as a feature input (§3.1); joint view/grid augmentation named as future work (§6) | https://isprs-annals.copernicus.org/articles/X-G-2025/47/2025/isprs-annals-X-G-2025-47-2025.pdf |

**[A1]** is attempt 1's results, `for_claude.zip` (2026-11-08, commit 06a0685). The rows used as baselines are in
`code/baselines_attempt1.csv`.
