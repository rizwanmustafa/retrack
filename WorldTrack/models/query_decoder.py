"""
R2: sparse ground-anchored person queries instead of the dense BEV decoder (enabled with --model.sparse_queries true).

TrackTacular decodes people from a dense BEV map with a ResNet-18 U-Net (models/decoder.py) and per-cell heads.
R2 removes that decoder. A light proposal head on the (unchanged) BEV feature gives the top-K candidate cells; each
candidate becomes a query. Every decoder layer
  1. lets the queries attend to each other (crowd reasoning, duplicate suppression),
  2. projects the query's ground position at several heights into every working camera and reads the full-resolution
     image features there (mean over the cameras that see the point), plus the BEV feature under the query,
  3. predicts a position refinement, a score and the motion to the previous frame.
Positions are refined layer by layer (detached between layers, as in Deformable DETR), and every layer is supervised
(one-to-one Hungarian assignment to the ground truth).

Coordinates are BEV "mem" coordinates (x along X, y along Y; BEV pixel i is centred at mem coordinate i), the same as
utils/vox.py. Heights are sampled at mem z in [0, Z-1], i.e. between the centres of the lowest and highest voxel, so
the geometry is exactly that of the lifting step and does not depend on the dataset's units.

For evaluation the queries are written back into dense maps (`instance_center`, `instance_offset`, ...) so that the
existing decoding, trackers and metrics are used unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.basic
import utils.geom

EMPTY_LOGIT = -30.0
MAX_MOTION_CELLS = 15  # as the motion target in datasets/pedestrian_dataset.py (get_bev_gt)


# ------------------------------------------------------------------ sampling helpers
def sample_bev(feat_bev, xy):
    """Bilinear BEV features at mem positions. feat_bev: B,C,Y,X; xy: B,P,2 (mem x, y) -> B,P,C"""
    B, C, Y, X = feat_bev.shape
    gx = xy[..., 0] / max(X - 1, 1) * 2 - 1
    gy = xy[..., 1] / max(Y - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(1).to(feat_bev.dtype)  # B,1,P,2
    out = F.grid_sample(feat_bev, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out[:, :, 0].transpose(1, 2)


class ViewSampler:
    """Projects ground positions at several heights into every camera and averages the image features of the
    working cameras that see the point (the same projection and validity test as vox_util.unproject_image_to_mem)."""

    def __init__(self, feat_cams_, featpix_T_ref_, cams_T_ref_, cam_mask, vox_util, YZX, z_sign, B, S, num_heights):
        self.feat = feat_cams_  # B*S,C,Hf,Wf
        self.featpix_T_ref = featpix_T_ref_  # B*S,4,4
        self.cams_T_ref = cams_T_ref_  # B*S,4,4
        self.cam_mask = cam_mask  # B,S or None
        self.vox_util = vox_util
        self.Y, self.Z, self.X = YZX
        self.z_sign = z_sign
        self.B, self.S = B, S
        self.z_mem = torch.linspace(0, self.Z - 1, num_heights, device=feat_cams_.device)
        self.num_heights = num_heights
        self.C = feat_cams_.shape[1]
        self.out_dim = num_heights * (self.C + 1)

    def __call__(self, xy):
        """xy: B,P,2 mem -> B,P,num_heights*(C+1): per height the mean feature and the fraction of cameras seeing it"""
        B, S, Hh, C = self.B, self.S, self.num_heights, self.C
        P = xy.shape[1]
        if P == 0:
            return xy.new_zeros(B, 0, self.out_dim)
        z = self.z_mem.view(1, 1, Hh, 1).expand(B, P, Hh, 1)
        xyz = torch.cat([xy.unsqueeze(2).expand(B, P, Hh, 2), z], dim=-1).reshape(B, P * Hh, 3)
        ref = self.vox_util.Mem2Ref(xyz, self.Y, self.Z, self.X)
        ref_ = ref.repeat_interleave(S, dim=0)  # b-major, as pack_seqdim
        pix = utils.geom.apply_4x4(self.featpix_T_ref, ref_)
        cam = utils.geom.apply_4x4(self.cams_T_ref, ref_)
        eps = 1e-6
        d = pix[..., 2:3]
        d = torch.where(d >= 0, d.clamp(min=eps), d.clamp(max=-eps))
        u, v = pix[..., 0] / d[..., 0], pix[..., 1] / d[..., 0]
        Hf, Wf = self.feat.shape[-2:]
        valid = (u >= 0) & (u <= Wf) & (v >= 0) & (v <= Hf) & (self.z_sign * cam[..., 2] >= 0)
        gy, gx = utils.basic.normalize_grid2d(v, u, Hf, Wf)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(1).to(self.feat.dtype)  # B*S,1,N,2
        val = F.grid_sample(self.feat, grid, mode='bilinear', align_corners=False)[:, :, 0]  # B*S,C,N
        val = val.reshape(B, S, C, P, Hh)
        w = valid.reshape(B, S, 1, P, Hh).to(val.dtype)
        if self.cam_mask is not None:
            w = w * self.cam_mask.view(B, S, 1, 1, 1).to(val.dtype)
        cnt = w.sum(dim=1)  # B,1,P,Hh
        mean = (val * w).sum(dim=1) / cnt.clamp(min=1)  # B,C,P,Hh
        out = torch.cat([mean, cnt / S], dim=1)  # B,C+1,P,Hh
        return out.permute(0, 2, 3, 1).reshape(B, P, Hh * (C + 1))


def mlp(i, h, o, zero_last=False):
    m = nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))
    if zero_last:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m


def nms_topk(heat, K):
    """3x3 max-pool NMS (utils/decode.py) and the K best cells. heat: B,1,Y,X -> scores B,K, xy (cell index) B,K,2"""
    B, _, Y, X = heat.shape
    keep = (F.max_pool2d(heat, 3, stride=1, padding=1) == heat).to(heat.dtype)
    scores, inds = torch.topk((heat * keep).view(B, -1), K)
    xs = (inds % X).to(heat.dtype)
    ys = torch.div(inds, X, rounding_mode='floor').to(heat.dtype)
    return scores, torch.stack([xs, ys], dim=-1)


def render_dense(xy, logit, motion, Y, X):
    """Writes the queries into the dense maps the existing decoder expects (utils/decode.py):
    instance_center (logits, EMPTY_LOGIT where there is no query), instance_offset = (xy - cell, prev - xy).
    When two queries fall into the same cell the higher-scoring one is kept."""
    B, K = logit.shape
    xy, logit, motion = xy.detach().float(), logit.detach().float(), motion.detach().float()
    cx = xy[..., 0].floor().clamp(0, X - 1)
    cy = xy[..., 1].floor().clamp(0, Y - 1)
    idx = (cy * X + cx).long()  # B,K
    center = torch.full((B, Y * X), EMPTY_LOGIT, device=xy.device)
    center.scatter_reduce_(1, idx, logit, reduce='amax', include_self=True)
    win = logit >= center.gather(1, idx)
    vals = torch.stack([xy[..., 0] - cx, xy[..., 1] - cy, motion[..., 0], motion[..., 1]], dim=1)  # B,4,K
    offset = torch.zeros(B, 4, Y * X, device=xy.device)
    for b in range(B):
        offset[b][:, idx[b][win[b]]] = vals[b][:, win[b]]
    return {
        'instance_center': center.view(B, 1, Y, X),
        'instance_offset': offset.view(B, 4, Y, X),
        'instance_size': torch.zeros(B, 3, Y, X, device=xy.device),
        'instance_rot': torch.zeros(B, 8, Y, X, device=xy.device),
    }


# ------------------------------------------------------------------ decoder
class QueryLayer(nn.Module):
    def __init__(self, d, img_in, bev_in, heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.img_proj = mlp(img_in, d, d)
        self.bev_proj = nn.Linear(bev_in, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = mlp(d, 4 * d, d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, q, pos, img, bev):
        k = q + pos
        q = self.norm1(q + self.self_attn(k, k, q, need_weights=False)[0])
        q = self.norm2(q + self.img_proj(img) + self.bev_proj(bev))
        return self.norm3(q + self.ffn(q))


class SparseQueryDecoder(nn.Module):
    def __init__(self, bev_dim=256, img_dim=128, d=256, num_queries=100, num_layers=3, num_heights=4, heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.num_heights = num_heights
        self.proposal = nn.Sequential(
            nn.Conv2d(bev_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.proposal[-1].bias.data.fill_(-2.19)
        self.q_in = nn.Linear(bev_dim + 1, d)  # BEV feature at the anchor + its proposal score
        self.pos = mlp(2, d, d)
        img_in = num_heights * (img_dim + 1)
        self.layers = nn.ModuleList([QueryLayer(d, img_in, bev_dim, heads) for _ in range(num_layers)])
        self.score = nn.ModuleList([nn.Linear(d, 1) for _ in range(num_layers)])
        self.delta = nn.ModuleList([mlp(d, d, 2, zero_last=True) for _ in range(num_layers)])
        self.motion = nn.ModuleList([mlp(d, d, 2) for _ in range(num_layers)])
        for s in self.score:
            s.bias.data.fill_(-2.19)

    def forward(self, feat_bev, view_sampler):
        B, _, Y, X = feat_bev.shape
        prop = self.proposal(feat_bev)  # B,1,Y,X logits
        with torch.no_grad():
            p_score, cell = nms_topk(prop.sigmoid(), min(self.num_queries, Y * X))
        xy = cell + 0.5  # ground truth inside cell c lies in [c, c+1)
        scale = xy.new_tensor([max(X - 1, 1), max(Y - 1, 1)])
        q = self.q_in(torch.cat([sample_bev(feat_bev, xy), p_score.unsqueeze(-1)], dim=-1))
        layers = []
        for i, layer in enumerate(self.layers):
            xy_s = torch.stack([xy[..., 0].clamp(0, X - 1), xy[..., 1].clamp(0, Y - 1)], dim=-1)
            q = layer(q, self.pos(xy_s / scale), view_sampler(xy_s), sample_bev(feat_bev, xy_s))
            xy_new = xy + self.delta[i](q)
            layers.append({'xy': xy_new, 'logit': self.score[i](q).squeeze(-1), 'motion': self.motion[i](q)})
            xy = xy_new.detach()
        out = {'proposal_center': prop, 'query_layers': layers, 'query_anchor': cell}
        last = layers[-1]
        out.update(render_dense(last['xy'], last['logit'], last['motion'], Y, X))
        return out


class ImageCenterHead(nn.Module):
    """The image-plane foot-point head of models/decoder.py (kept so that R2 keeps the baseline's auxiliary loss)."""

    def __init__(self, feat2d, n_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feat2d, feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(feat2d), nn.ReLU(inplace=True),
            nn.Conv2d(feat2d, n_classes, kernel_size=1),
        )
        self.head[-1].bias.data.fill_(-2.19)

    def forward(self, feat_cams):
        return self.head(feat_cams)
