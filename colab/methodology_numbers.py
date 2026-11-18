"""
Numbers for the methodology deck and document: exact parameter counts per pipeline stage, and one pedestrian of one
WILDTRACK test frame followed through all 8 stages of a trained R2 checkpoint.

Colab (any GPU, a T4 is enough; about 5 minutes):
    !rm -rf /content/retrack && git clone -q -b feat/r2-sparse-queries https://github.com/rizwanmustafa/retrack.git /content/retrack
    %run -i /content/retrack/colab/methodology_numbers.py

Settings: define any of these in the cell before the %run line.
    CKPT   R2 WILDTRACK checkpoint      default: newest runs/wildtrack_R2_s80085/**/last.ckpt in MyDrive/retrack_results*
    FRAME  index of the test frame       default 10 (0..39)
    PID    WILDTRACK person id           default: the person of that frame annotated in the most cameras
Output: MyDrive/retrack_methodology/numbers.json (also printed)
"""
import copy
import glob
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)) if '__file__' in globals() else '/content/retrack/colab')
from common import WORLDTRACK, DRIVE, mount_drive, pip_install, ensure_dataset  # noqa: E402

g = globals()
CKPT = g.get('CKPT', None)
FRAME = int(g.get('FRAME', 10))
PID = g.get('PID', None)
OUT = f'{DRIVE}/retrack_methodology'

mount_drive()
pip_install()
data_dir = ensure_dataset('wildtrack')
os.chdir(WORLDTRACK)
sys.path.insert(0, WORLDTRACK)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from world_track import WorldTrackModel  # noqa: E402
from evaluation import r2_eval  # noqa: E402
from models.query_decoder import ViewSampler, nms_topk, sample_bev  # noqa: E402
from tracking.multitracker import JDETracker, STrack, joint_stracks  # noqa: E402
from utils import basic, geom, decode  # noqa: E402


def r(x, nd=4):
    """JSON-friendly copy with rounded floats."""
    if torch.is_tensor(x):
        x = x.detach().cpu().double().numpy()
    if isinstance(x, np.ndarray):
        return np.round(x, nd).tolist() if x.dtype.kind == 'f' else x.tolist()
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    return x


def n(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ 1. parameter counts (no checkpoint needed)
WILDTRACK = dict(resolution=(120, 4, 360), bounds=(0, 1440, 0, 480, 0, 200), num_cameras=7, depth=(32, 250, 3250))
MULTIVIEWX = dict(resolution=(160, 2, 250), bounds=(0, 1000, 0, 640, 0, 2), num_cameras=6, depth=(32, 0, -20), z_sign=-1)


def stage_counts(seg):
    qd = seg.query_decoder
    scalars = sum(p.numel() for name, p in seg.named_parameters(recurse=False))
    stages = {
        '1 Multi-camera input': 0,
        '2 Feature extraction': n(seg.encoder),
        '3 Geometric feature lifting': 0,
        '4 Camera fusion & BEV': n(seg.cam_compressor) + n(seg.bev_compressor),
        '5 Temporal integration': n(seg.temporal_bev),
        '6 Sparse query refinement': n(qd.proposal) + n(qd.q_in) + n(qd.pos) + n(qd.layers) + n(qd.delta),
        '7 Pedestrian detection': n(qd.score) + n(seg.img_head) + scalars,
        '8 Identity tracking': n(qd.motion),
    }
    assert sum(stages.values()) == n(seg), (sum(stages.values()), n(seg))
    cum, rows = 0, []
    for k, v in stages.items():
        cum += v
        rows.append({'stage': k, 'params': v, 'cumulative': cum, 'share_pct': round(100 * v / n(seg), 2)})
    return rows


def breakdown(seg):
    qd = seg.query_decoder
    L0 = qd.layers[0]
    frozen = [m for m in seg.encoder.modules() if 'FrozenBatchNorm' in type(m).__name__]
    return {
        'encoder': {k: n(getattr(seg.encoder, k)) for k in
                    ('layer0', 'layer1', 'layer2', 'layer3', 'upsampling_layer1', 'upsampling_layer2', 'depth_layer')},
        'encoder_frozen_bn_modules': len(frozen),
        'encoder_frozen_bn_affine_values': sum(m.weight.numel() + m.bias.numel() for m in frozen),
        'encoder_trainable_bn_params': sum(n(m) for m in seg.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)),
        'cam_compressor': [n(m) for m in seg.cam_compressor],
        'bev_compressor': [n(m) for m in seg.bev_compressor],
        'temporal_bev': [n(m) for m in seg.temporal_bev],
        'proposal': n(qd.proposal), 'q_in': n(qd.q_in), 'pos': n(qd.pos),
        'query_layer': {k: n(getattr(L0, k)) for k in
                        ('self_attn', 'norm1', 'img_proj', 'bev_proj', 'norm2', 'ffn', 'norm3')},
        'query_layer_total': n(L0), 'score_each': n(qd.score[0]), 'delta_each': n(qd.delta[0]),
        'motion_each': n(qd.motion[0]), 'img_head': n(seg.img_head),
        'scalars': [name for name, _ in seg.named_parameters(recurse=False)],
    }


print('building models for the parameter counts ...', flush=True)
counts = {}
wt_r2 = WorldTrackModel(model_name='segnet', encoder_name='res18', use_temporal_cache=False, sparse_queries=True,
                        **WILDTRACK)
counts['wildtrack_R2'] = {'total': n(wt_r2), 'stages': stage_counts(wt_r2.model), 'breakdown': breakdown(wt_r2.model)}
assert n(wt_r2) == n(wt_r2.model)
mvx_r2 = WorldTrackModel(model_name='segnet', encoder_name='res18', use_temporal_cache=False, sparse_queries=True,
                         **MULTIVIEWX)
counts['multiviewx_R2'] = {'total': n(mvx_r2), 'stages': stage_counts(mvx_r2.model)}
wt_b0 = WorldTrackModel(model_name='segnet', encoder_name='res18', use_temporal_cache=False, **WILDTRACK)
counts['wildtrack_B0'] = {'total': n(wt_b0), 'dense_decoder': n(wt_b0.model.decoder)}
del wt_r2, mvx_r2, wt_b0
for row in counts['wildtrack_R2']['stages']:
    print(f"{row['stage']:<30} {row['params']:>12,} {row['cumulative']:>12,}")
print('total', f"{counts['wildtrack_R2']['total']:,}", '| B0 total', f"{counts['wildtrack_B0']['total']:,}")


# ------------------------------------------------------------------ 2. checkpoint and test frame
def find_ckpt():
    for pattern in ('wildtrack_R2_s80085', 'wildtrack_R2_s*'):
        found = glob.glob(f'{DRIVE}/retrack_results*/**/runs/{pattern}/**/last.ckpt', recursive=True)
        found = [p for p in found if '/preflight/' not in p]
        if found:
            return max(found, key=osp.getmtime)
    raise FileNotFoundError('no R2 WILDTRACK checkpoint found; set CKPT = "<path to last.ckpt>"')


CKPT = CKPT or find_ckpt()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('checkpoint', CKPT, flush=True)
model, hp = r2_eval.load_model(CKPT, device)
assert r2_eval.variant_of(hp) == 'R2' and r2_eval.dataset_of(hp) == 'wildtrack', (r2_eval.variant_of(hp), hp)
seg, vu = model.model, model.vox_util
Y, Z, X = model.Y, model.Z, model.X
items, dataset = r2_eval.load_test_items(data_dir, model)
item = items[FRAME]
real = int(item['frame'][0]) * dataset.base.frame_step
keys = list(dataset.world_gt.keys())
real_prev = keys[max(keys.index(real) - 1, 0)]
S = dataset.num_cam

# person: annotated in the most cameras, also present in the previous frame
gt_pts, gt_pids = dataset.world_gt[real]
prev_pts, prev_pids = dataset.world_gt[real_prev]
cams_of = {int(p): [c for c in range(S) if int(p) in dataset.imgs_gt[real][c][1].int().tolist()] for p in gt_pids}
if PID is None:
    cand = [int(p) for p in gt_pids if int(p) in prev_pids.int().tolist()]
    PID = max(cand, key=lambda p: (len(cams_of[p]), -p))
PID = int(PID)
gi = gt_pids.int().tolist().index(PID)
gxy = gt_pts[gi].double()  # ground grid (2.5 cm units)
gxy_prev = prev_pts[prev_pids.int().tolist().index(PID)].double() if PID in prev_pids.int().tolist() else None
cams = cams_of[PID]
def box_of(c):
    return dataset.imgs_gt[real][c][0][dataset.imgs_gt[real][c][1].int().tolist().index(PID)].double()


cam = max(cams, key=lambda c: float((box_of(c)[2] - box_of(c)[0]) * (box_of(c)[3] - box_of(c)[1])))  # largest box

# ------------------------------------------------------------------ 3. the forward pass, stage by stage
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
batch['img'] = batch['img'].float() / 255
rgb, K, E, ref_T_global = batch['img'], batch['intrinsic'], batch['extrinsic'], batch['ref_T_global']
B, _, C3, H, W = rgb.shape
ex = {'frame_index': FRAME, 'frame': real, 'prev_frame': real_prev, 'person_id': PID, 'cameras_annotated': cams,
      'camera_shown': cam, 'checkpoint': CKPT, 'epoch': r2_eval.read_hparams(CKPT)[1].get('epoch')}

with torch.no_grad():
    # stage 1: input
    __p = lambda x: basic.pack_seqdim(x, B)  # noqa: E731
    __u = lambda x: basic.unpack_seqdim(x, B)  # noqa: E731
    rgb_, pix_T_cams_, cams_T_global_ = __p(rgb), __p(K), __p(E)
    ref_T_cams = torch.matmul(ref_T_global.repeat_interleave(S, dim=0), torch.inverse(cams_T_global_))
    cams_T_ref_ = torch.inverse(ref_T_cams)
    x_in = (rgb_ - seg.mean) / seg.std
    world_cm = torch.tensor(dataset.base.worldcoord_from_worldgrid_mat, dtype=torch.float64) @ \
        torch.tensor([gxy[0], gxy[1], 1.0], dtype=torch.float64)
    box = box_of(cam)
    sxy = W / 1920.0, H / 1080.0
    box_s = box * torch.tensor([sxy[0], sxy[1], sxy[0], sxy[1]], dtype=torch.float64)
    cu, cv = int((box_s[0] + box_s[2]) / 2), int((box_s[1] + box_s[3]) / 2)
    ex['stage1'] = {
        'num_cameras': S, 'raw_image_hw': [1080, 1920], 'input_image_hw': [H, W], 'resize_factor': sxy,
        'person_grid_xy': gxy, 'person_world_cm': world_cm[:2], 'person_position_id': int(gxy[0] * 480 + gxy[1]),
        'K_raw': dataset.base.intrinsic_matrices[cam], 'K_input': K[0, cam, :3, :3],
        'R': E[0, cam, :3, :3], 't': E[0, cam, :3, 3], 'ref_T_global': ref_T_global[0],
        'bbox_raw_x1y1x2y2': box, 'bbox_input_x1y1x2y2': box_s,
        'pixel_at_box_centre': {'uv': [cu, cv], 'rgb_0_1': rgb[0, cam, :, cv, cu], 'normalised': x_in[cam, :, cv, cu]},
    }

    # stage 2: features
    feat = seg.encoder(x_in)
    _, C, Hf, Wf = feat.shape
    featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, Wf / float(W), Hf / float(H))
    featpix_T_ref_ = basic.matmul2(featpix_T_cams_, cams_T_ref_)

    def project(xy_mem, z_mem):
        """mem (x, y, z) -> per camera: feature-map (u, v), depth, valid (as unproject_image_to_mem)"""
        ref = vu.Mem2Ref(torch.tensor([[[xy_mem[0], xy_mem[1], z_mem]]], device=device, dtype=torch.float32), Y, Z, X)
        ref_ = ref.repeat(S, 1, 1)
        pix = geom.apply_4x4(featpix_T_ref_, ref_)[:, 0]
        camz = geom.apply_4x4(cams_T_ref_, ref_)[:, 0, 2]
        u, v = pix[:, 0] / pix[:, 2], pix[:, 1] / pix[:, 2]
        valid = (u >= 0) & (u <= Wf) & (v >= 0) & (v <= Hf) & (seg.z_sign * camz >= 0)
        return ref[0, 0], u, v, camz, valid

    def bilinear(fm, u, v):
        gy, gx = basic.normalize_grid2d(v.view(1, 1), u.view(1, 1), Hf, Wf)
        return F.grid_sample(fm[None], torch.stack([gx, gy], -1).view(1, 1, 1, 2), align_corners=False)[0, :, 0, 0]

    mem = vu.Ref2Mem(torch.tensor([[[gxy[0], gxy[1], 0.0]]], device=device, dtype=torch.float32), Y, Z, X)[0, 0]
    cx, cy = int(mem[0].floor()), int(mem[1].floor())
    # ground point (z = 0 cm) in the input image of the shown camera: compare with the annotated box
    ground_ref = torch.tensor([[[gxy[0], gxy[1], 0.0]]], device=device, dtype=torch.float32)
    gp = geom.apply_4x4(basic.matmul2(pix_T_cams_, cams_T_ref_)[cam:cam + 1], ground_ref)[0, 0]
    gp_uv = (gp[:2] / gp[2]).double().cpu()
    _, u0, v0, _, _ = project((float(mem[0]), float(mem[1])), 0.0)
    norms = feat[cam].norm(dim=0)
    ex['stage2'] = {
        'feature_shape_per_camera': [C, Hf, Wf], 'stride': H // Hf,
        'ground_point_input_image_uv': gp_uv,
        'box_bottom_centre_uv': [(box_s[0] + box_s[2]) / 2, box_s[3]], 'box_top_centre_uv': [(box_s[0] + box_s[2]) / 2, box_s[1]],
        'image_head_target_uv_feature_map': [(box_s[0] + box_s[2]) / 8, box_s[1] / 4],  # (x centre, ymin) / 4
        'ground_point_feature_uv': [u0[cam], v0[cam]],
        'feature_norm_at_ground_point': bilinear(feat[cam], u0[cam], v0[cam]).norm(),
        'feature_norm_mean_over_map': norms.mean(), 'feature_norm_p95_over_map': norms.flatten().quantile(0.95),
        'feature_first5_at_ground_point': bilinear(feat[cam], u0[cam], v0[cam])[:5],
    }

    # stage 3: lifting
    mems_ = vu.unproject_image_to_mem(feat, featpix_T_ref_, cams_T_ref_, Y, Z, X, xyz_refA=None, z_sign=seg.z_sign)
    mems = __u(mems_)  # B,S,C,Y,Z,X
    voxels = []
    for z in range(Z):
        ref, u, v, camz, valid = project((cx, cy), float(z))
        voxels.append({'z_mem': z, 'voxel_centre_ref': ref, 'u_feat': u, 'v_feat': v, 'cam_depth': camz,
                       'valid': valid.int(), 'num_valid': int(valid.sum()),
                       'lifted_first3_shown_cam': mems[0, cam, :3, cy, z, cx],
                       'bilinear_first3_shown_cam': bilinear(feat[cam], u[cam], v[cam])[:3]})
    cover = (mems[0].abs().sum(1) > 0).float().mean(dim=(1, 2, 3))  # per camera: fraction of voxels it sees
    ex['stage3'] = {
        'person_mem_xy': mem[:2], 'person_cell_xy': [cx, cy], 'voxel_size_ref': [vu.default_vox_size_X,
                                                                                 vu.default_vox_size_Y,
                                                                                 vu.default_vox_size_Z],
        'volume_shape_per_camera': list(mems.shape[2:]), 'voxels_of_person_cell': voxels,
        'fraction_of_voxels_seen_per_camera': cover,
    }

    # stage 4: fusion
    feat_mem = seg.cam_compressor(mems.flatten(1, 2))
    bev_in = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, C * Z, Y, X)
    F_t = seg.bev_compressor(bev_in)
    bev_norm = F_t[0].norm(dim=0)
    ex['stage4'] = {
        'stacked_channels': S * C, 'fused_volume_shape': list(feat_mem.shape[1:]), 'bev_in_shape': list(bev_in.shape[1:]),
        'bev_shape': list(F_t.shape[1:]),
        'camera_slots_nonzero_at_person_voxel_z0': [int(mems[0, v_, :, cy, 0, cx].abs().sum() > 0) for v_ in range(S)],
        'bev_norm_at_person': bev_norm[cy, cx], 'bev_norm_median_all_cells': bev_norm.median(),
        'bev_first5_at_person': F_t[0, :5, cy, cx],
    }

    # stage 5: temporal block (cache off: the "previous" BEV is the current one)
    Ft = seg.temporal_bev(torch.cat([F_t, F_t], dim=1))
    tn = Ft[0].norm(dim=0)
    ex['stage5'] = {
        'use_temporal_cache': bool(model.use_temporal_cache), 'input_channels': 2 * F_t.shape[1],
        'bev_norm_at_person': tn[cy, cx], 'bev_norm_median_all_cells': tn.median(),
        'change_norm_at_person': (Ft - F_t)[0, :, cy, cx].norm(), 'bev_first5_at_person': Ft[0, :5, cy, cx],
    }

    # stage 6: sparse queries (SparseQueryDecoder.forward, step by step)
    qd = seg.query_decoder
    vs = ViewSampler(feat, featpix_T_ref_, cams_T_ref_, None, vu, (Y, Z, X), seg.z_sign, B, S, seg.query_heights)
    prop = qd.proposal(Ft)
    P = prop.sigmoid()
    p_score, cell = nms_topk(P, min(qd.num_queries, Y * X))
    xy = cell + 0.5
    scale = xy.new_tensor([max(X - 1, 1), max(Y - 1, 1)])
    q = qd.q_in(torch.cat([sample_bev(Ft, xy), p_score.unsqueeze(-1)], dim=-1))
    trace = []
    for i, layer in enumerate(qd.layers):
        xy_s = torch.stack([xy[..., 0].clamp(0, X - 1), xy[..., 1].clamp(0, Y - 1)], dim=-1)
        pos, img, bev = qd.pos(xy_s / scale), vs(xy_s), sample_bev(Ft, xy_s)
        k_ = q + pos
        att, att_w = layer.self_attn(k_, k_, q, need_weights=True)
        q1 = layer.norm1(q + att)
        img_c, bev_c = layer.img_proj(img), layer.bev_proj(bev)
        q2 = layer.norm2(q1 + img_c + bev_c)
        ffn_c = layer.ffn(q2)
        q = layer.norm3(q2 + ffn_c)
        delta = qd.delta[i](q)
        trace.append(dict(xy_in=xy_s, img=img, att=att, att_w=att_w, img_c=img_c, bev_c=bev_c, ffn_c=ffn_c,
                          delta=delta, logit=qd.score[i](q).squeeze(-1), motion=qd.motion[i](q), xy_out=xy + delta))
        xy = (xy + delta).detach()
    out_ref = qd(Ft, vs)  # the module itself: the step-by-step pass must give the same result
    assert torch.allclose(out_ref['query_layers'][-1]['xy'], trace[-1]['xy_out'], atol=1e-4)
    gt_mem = mem[:2]
    kq = int((trace[-1]['xy_out'][0] - gt_mem).norm(dim=-1).argmin())  # query closest to the person
    layers = []
    for i, t in enumerate(trace):
        hh = t['img'][0, kq].view(seg.query_heights, C + 1)
        w = t['att_w'][0, kq].clone()
        w[kq] = -1
        top = w.topk(3)
        layers.append({
            'layer': i + 1, 'xy_in_mem': t['xy_in'][0, kq], 'delta_cells': t['delta'][0, kq],
            'xy_out_mem': t['xy_out'][0, kq], 'dist_to_gt_cm': (t['xy_out'][0, kq] - gt_mem).norm() * 10,
            'score': t['logit'][0, kq].sigmoid(), 'logit': t['logit'][0, kq],
            'match_cost': (t['xy_out'][0, kq] - gt_mem).norm() / model.query_match_cells + 1 - t['logit'][0, kq].sigmoid(),
            'cameras_seeing_each_height': hh[:, C] * S, 'img_feature_norm_each_height': hh[:, :C].norm(dim=1),
            'update_norms': {'self_attention': t['att'][0, kq].norm(), 'image': t['img_c'][0, kq].norm(),
                             'bev': t['bev_c'][0, kq].norm(), 'ffn': t['ffn_c'][0, kq].norm()},
            'self_attention_weight_on_itself': t['att_w'][0, kq, kq],
            'top3_attended_other_queries': [{'query': int(j), 'weight': float(top.values[m]),
                                             'xy_mem': t['xy_in'][0, j], 'score': t['logit'][0, j].sigmoid()}
                                            for m, j in enumerate(top.indices)],
        })
    ex['stage6'] = {
        'num_bev_cells': Y * X, 'num_queries': int(cell.shape[1]),
        'proposal_at_person_cell': P[0, 0, cy, cx], 'proposal_max_3x3_around_person': P[0, 0, max(cy - 1, 0):cy + 2,
                                                                                         max(cx - 1, 0):cx + 2].max(),
        'anchors_with_proposal_above_0.5': int((p_score[0] > 0.5).sum()),
        'anchors_with_proposal_above_0.1': int((p_score[0] > 0.1).sum()),
        'query_index': kq, 'anchor_cell': cell[0, kq], 'anchor_xy_mem': cell[0, kq] + 0.5,
        'anchor_proposal_score': p_score[0, kq], 'gt_xy_mem': gt_mem,
        'image_sample_dims': int(trace[0]['img'].shape[-1]), 'layers': layers,
        'final_scores_above_0.5': int((trace[-1]['logit'][0].sigmoid() > 0.5).sum()),
    }

    # stage 7: detection (full model forward, as the evaluation runs it)
    output = model(batch)
    xy_e, xy_prev_e, scores_e, _, _, _ = decode.decoder(output['instance_center'].sigmoid(), output['instance_offset'],
                                                        output['instance_size'], rz_e=output['instance_rot'],
                                                        K=model.max_detections)
    det = r2_eval.to_ref(model, xy_e)[0].cpu().double().numpy()
    det_prev = r2_eval.to_ref(model, xy_prev_e)[0].cpu().double().numpy()
    sc = scores_e[0].cpu().double().numpy()
    jd = int(np.linalg.norm(det - gxy.numpy(), axis=1).argmin())
    gt_all = gt_pts.double().numpy()
    keep = sc > r2_eval.THR
    rr, cc, dd = r2_eval.match_frame(gt_all, det[keep])
    _, target = dataset[keys.index(real)]
    target = {k: v.unsqueeze(0).to(device) for k, v in target.items()}
    _, loss_terms = model.loss(target, output)
    ex['stage7'] = {
        'detection_ref': det[jd], 'detection_score': sc[jd], 'error_cm': np.linalg.norm(det[jd] - gxy.numpy()) * 2.5,
        'frame_num_gt': len(gt_all), 'frame_num_detections_above_0.5': int(keep.sum()),
        'frame_tp': len(rr), 'frame_fp': int(keep.sum()) - len(rr), 'frame_fn': len(gt_all) - len(rr),
        'frame_moda_pct': (1 - (len(gt_all) - len(rr) + int(keep.sum()) - len(rr)) / len(gt_all)) * 100,
        'frame_mean_error_cm': float(dd.mean() * 2.5) if len(dd) else None,
        'frame_loss_terms': {k: float(v) for k, v in loss_terms.items()},
        'center_weight_lambda_c': float(seg.center_weight),
        'image_head_at_its_target': output['img_center'][cam, 0].sigmoid()[
            min(int(box_s[1] / 4), Hf - 1), min(int((box_s[0] + box_s[2]) / 8), Wf - 1)],
        'image_head_at_ground_point': output['img_center'][cam, 0].sigmoid()[
            min(int(v0[cam]), Hf - 1), min(int(u0[cam]), Wf - 1)],
        'image_head_max_shown_cam': output['img_center'][cam, 0].sigmoid().max(),
    }

    # stage 8: motion head and tracker
    frames = r2_eval.run_inference(model, items)
    tracker = JDETracker(conf_thres=r2_eval.THR, track_buffer=r2_eval.TRACK_BUFFER)
    pool_pred, hyps = None, []
    for i, f in enumerate(frames):
        if i == FRAME:
            pool = joint_stracks([t for t in tracker.tracked_stracks if t.is_activated], tracker.lost_stracks)
            pool = copy.deepcopy(pool)
            STrack.multi_predict(pool)
            pool_pred = [(t.track_id, np.asarray(t.xy, dtype=np.float64), np.asarray(t.mean, dtype=np.float64))
                         for t in pool]
        st = tracker.update(torch.from_numpy(f['xy']), torch.from_numpy(f['xy_prev']), torch.from_numpy(f['score']))
        hyps.append([(int(s.track_id), np.asarray(s.xy, dtype=np.float64)) for s in st])
    fd = frames[FRAME]
    jt = int(np.linalg.norm(fd['xy'] - gxy.numpy(), axis=1).argmin())
    d_prev = fd['xy_prev'][jt].astype(np.float64)
    costs = sorted([(tid, float(np.linalg.norm(p - d_prev)), p, m) for tid, p, m in pool_pred], key=lambda c: c[1])[:3]
    after = min(hyps[FRAME], key=lambda h: np.linalg.norm(h[1] - fd['xy'][jt]))
    ids = []
    for i, (f, hy) in enumerate(zip(frames, hyps)):
        g_ = f['gt'][f['gt'][:, 2] == PID]
        if not len(g_) or not hy:
            ids.append([f['frame'] * 5, None])
            continue
        best = min(hy, key=lambda h: np.linalg.norm(h[1] - g_[0, :2]))
        ids.append([f['frame'] * 5, best[0] if np.linalg.norm(best[1] - g_[0, :2]) * 0.025 <= 1 else None])
    ex['stage8'] = {
        'motion_cells': trace[-1]['motion'][0, kq], 'xy_prev_pred_ref': d_prev,
        'gt_prev_ref': gxy_prev, 'xy_prev_error_cm': (np.linalg.norm(d_prev - gxy_prev.numpy()) * 2.5
                                                     if gxy_prev is not None else None),
        'gt_displacement_ref': (gxy - gxy_prev) if gxy_prev is not None else None,
        'closest_tracks_before_update': [{'track_id': tid, 'kalman_predicted_ref': p, 'kalman_state': m,
                                          'distance_to_det_prev': c} for tid, c, p, m in costs],
        'gate_units': 75, 'assigned_track_id': after[0], 'track_xy_after_update': after[1],
        'track_id_of_person_per_frame': ids,
        'distinct_track_ids_for_person': sorted({i for _, i in ids if i is not None}),
    }

# ------------------------------------------------------------------ 4. full-test metrics, if evaluated
ev = None
parts = CKPT.split(os.sep)
if 'runs' in parts:
    i = len(parts) - 1 - parts[::-1].index('runs')
    ev = os.sep.join(parts[:i] + ['eval', parts[i + 1], 'metrics.csv'])
if ev and osp.exists(ev):
    import pandas as pd
    ex['test_metrics_all_cameras'] = pd.read_csv(ev).iloc[0].to_dict()

result = {'param_counts': counts, 'worked_example': ex}
os.makedirs(OUT, exist_ok=True)
path = f'{OUT}/numbers_frame{FRAME}_pid{PID}.json'
with open(path, 'w') as fh:
    json.dump(r(result), fh, indent=1, default=str)
print(json.dumps(r(result), indent=1, default=str))
print(f'\nSaved {path} - share this file (or paste the printed JSON).')
