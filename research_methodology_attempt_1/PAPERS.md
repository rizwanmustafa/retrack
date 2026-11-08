# Paper index — what each downloaded paper is used for

All PDFs are in `papers/`. Each one was downloaded during this attempt and its title checked against the first
page. "Used for" names the table or section the methodology takes numbers from.

| ID | file | used for in `METHODOLOGY.md` |
|---|---|---|
| P01 | 01_wildtrack_dataset_1707.09299.pdf | dataset facts: 7 cameras, 400 annotated frames, 2 fps (every 5th of 10 fps) |
| P02 | 02_mvdet_2007.07247.pdf | MVDet baseline that 3DROM, MVDeTr and MVAug build on |
| P03 | 03_mvdetr_2108.05888.pdf | view-coherent augmentation ablation (−2.0 MODA without it); 5-run p-values |
| P04 | 04_shot_iccv2021.pdf | stacked homographies (multi-height projection) background |
| P05 | 05_3drom_2207.10895.pdf | **A1**: Tab. 3 (88.2 → 92.3 / 93.5), Tab. 5 (p), Tab. 6 (vs random erasing), Alg. 1, n = 25, d = 1 m |
| P06 | 06_mvaug_2210.10756.pdf | **A2**: Tab. 2–4 (90.86 → 93.17, 50 % best), interaction with image loss |
| P07 | 07_earlybird_2310.13350.pdf | keep ResNet-18 (Tab. 4); augmentation ablation (Tab. 3); λ = 0.98; 10-frame track buffer |
| P08 | 08_tracktacular_2403.12573.pdf | **baseline** numbers (Tab. 1–2); **A3** evidence (Tab. 4, MultiviewX); temporal caching; saturation note |
| P09 | 09_mvflow_tracking_2210.10771.pdf | **C2**: same-detector DeepSORT vs muSSP (Tab. 2); motion term in edge cost (Eq. 11) |
| P10 | 10_bytetrack_2110.06864.pdf | **C1**: BYTE on CenterTrack (Tab. 3) |
| P11 | 11_ocsort_2203.14360.pdf | ORU/OCM ablation (small online gains) |
| P12 | 12_strongsort_2202.13514.pdf | **C2/C3**: AFLink + GSI (Tab. II), GSI parameters |
| P13 | 13_centertrack_2004.01177.pdf | origin of the previous-frame offset head TrackTacular uses |
| P14 | 14_attention_aware_mvtrack_2504.03047.pdf | learned cross-attention association (Tab. 2–3); deferred to attempt 2 |
| P15 | 15_one_graph_dynamic_gnn_2507.08494.pdf | UMPN 93.9 MOTA / 96.3 IDF1 (Tab. 2); deferred |
| P16 | 16_mvtrajecter_2509.01157.pdf | current best WILDTRACK numbers (Tab. 3); multi-timestamp costs (Tab. 5–6); deferred |
| P17 | 17_atlasfusion_2509.08421.pdf | **A4** alternative: sparse projection / density weighting / per-view loss (Tab. 4) |
| P18 | 18_reST_2308.13229.pdf | graph-based offline association baseline |
| P19 | 19_mcblt_2412.00692.pdf | **C2**: same EarlyBird detections, offline GNN 89.5/92.3 → 92.6/95.6 (Tab. 3) |
| P20 | 20_msmvd_2508.20447.pdf | multi-scale BEV (94.6 MODA); backbone-size study on GMVD |
| P21 | 21_feature_pulling_wacv2024.pdf | **A4** alternative: 3D feature pulling + large-kernel refiner ablation |
| P22 | 22_grace_2609.16872.pdf | **independent TrackTacular reproduction** (88.50 / 89.71 / 94.00); **A4** VGF + Ray; BTR |
| P23 | 23_mv3dt_distributed_2606.13127.pdf | recent WILDTRACK comparison table |
| P24 | 24_lmgp_2111.11892.pdf | excluded from comparison (uses ground-truth boxes) |
| P25 | 25_booster_shot_2208.09211.pdf | detection SOTA context (92.9 MODA) |
| P26 | 26_camuvid_cvpr2025.pdf | detection SOTA context (95.0 MODA, not BEV) |
| P27 | 27_simplebev_2206.07959.pdf | batch size / resolution matter; no camera dropout; photometric augmentation does not help |
| P28 | 28_botsort_2206.14651.pdf | online tracker background |
| P29 | 29_hybridsort_2308.00783.pdf | online tracker background |
| P30 | 30_sushi_2212.03038.pdf | hierarchical offline GNN (basis of MCBLT); deferred |
| P31 | 31_deep_ocsort_2302.11813.pdf | adaptive re-ID weighting background |
| P32 | 32_centernet_objects_as_points_1904.07850.pdf | heatmap + offset heads, max-pool NMS |
| P33 | 33_gmvd_2109.12227.pdf | generalisation benchmark used by MSMVD/MVTrajecter ablations |
| P34 | 34_unified_mvmp_tracking_2302.03820.pdf | appearance features in multi-view tracking |
| P35 | 35_idf1_ristani_1609.01775.pdf | IDF1 definition |
| P36 | 36_swa_1803.05407.pdf | **A5** (out-of-domain evidence only) |
| P37 | 37_gta_global_tracklet_2411.08216.pdf | global tracklet association background |
| P38 | 38_beyond_centers_bbox_voxel_isprs2026.pdf | **seed variance** on WILDTRACK (mean ± std over 3 runs) |
| P40 | 40_clear_mot_bernardin2008.pdf | MOTA / MOTP definitions |
| P41 | 41_ksp_berclaz2011.pdf | k-shortest-paths min-cost-flow formulation (C2b) |
| P42 | 42_two_stage_mincost_flow_2311.02642.pdf | min-cost flow with muSSP in MOT |

Not downloaded: **muSSP** (Wang et al., NeurIPS 2019). The NeurIPS servers timed out from this machine. Its
WILDTRACK numbers are taken from P09, which ran it.
