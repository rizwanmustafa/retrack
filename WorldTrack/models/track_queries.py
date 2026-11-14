"""
R4 (research-methodology-5/METHODOLOGY.md): track queries.

TrackTacular links detections over time by letting every detection predict where it was in the previous frame
(the motion offset) and matching those positions greedily to the tracks. R4 turns this around: the tracker's
existing tracks are an *input* to the network. For each track, the head reads the current frame's features on a
grid around the track's last position, attends from the track to that neighbourhood and to the other tracks, and
predicts (a) where the person is now and (b) whether the person is still in the scene. Two rounds: a coarse grid
(+-12 cells) and a fine grid around the first estimate.

The head reads the features of whichever decoder the model has: the dense BEV decoder output (baseline / R1), or,
with R2, the BEV feature plus the multi-view image features under each sample point (models/query_decoder.py).

Training (no sequences needed): the previous frame's ground-truth people, jittered, are the input tracks; the target
is the same person's position in the current frame (or "gone"). Ghost tracks placed away from everyone teach the
head to say "gone". At test time the tracks come from tracking/query_tracker.py.
"""
import torch
import torch.nn as nn

from models.query_decoder import mlp

MAX_MOTION_CELLS = 15  # as the motion target in datasets/pedestrian_dataset.py (get_bev_gt)


class TrackQueryHead(nn.Module):
    def __init__(self, feat_dim, d=128, grid=7, spacings=(4.0, 1.5), heads=4):
        super().__init__()
        self.d = d
        r = torch.arange(grid, dtype=torch.float32) - (grid - 1) / 2
        gy, gx = torch.meshgrid(r, r, indexing='ij')
        self.register_buffer('unit', torch.stack([gx.flatten(), gy.flatten()], dim=-1), persistent=False)  # G2,2
        self.spacings = tuple(float(s) for s in spacings)
        self.inp = nn.Sequential(nn.Linear(feat_dim, d), nn.LayerNorm(d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.off_emb = mlp(2, d, d)
        self.rounds = nn.ModuleList()
        for _ in self.spacings:
            self.rounds.append(nn.ModuleDict({
                'q': nn.Linear(d, d), 'k': nn.Linear(d, d), 'v': nn.Linear(d, d),
                'norm1': nn.LayerNorm(d),
                'self_attn': nn.MultiheadAttention(d, heads, batch_first=True),
                'norm2': nn.LayerNorm(d),
                'ffn': mlp(d, 4 * d, d),
                'norm3': nn.LayerNorm(d),
                'delta': mlp(d, d, 2, zero_last=True),
                'presence': nn.Linear(d, 1),
            }))

    def forward(self, track_xy, track_valid, sample, Y, X):
        """track_xy: B,T,2 (mem), track_valid: B,T bool, sample: fn(B,P,2) -> B,P,feat_dim"""
        track_xy = track_xy.contiguous()
        B, T, _ = track_xy.shape
        G2 = self.unit.shape[0]
        pad = ~track_valid.bool()
        pad[:, 0] = pad[:, 0] & track_valid.bool().any(dim=1)  # never mask a whole row (MultiheadAttention NaNs)
        lim = track_xy.new_tensor([X - 1, Y - 1])
        center = torch.minimum(torch.maximum(track_xy, torch.zeros_like(lim)), lim)
        e = self.inp(sample(center))
        rounds = []
        for spacing, rd in zip(self.spacings, self.rounds):
            off = self.unit * spacing  # G2,2
            pts = center.unsqueeze(2) + off.view(1, 1, G2, 2)
            pts = torch.minimum(torch.maximum(pts, torch.zeros_like(lim)), lim)
            f = self.inp(sample(pts.reshape(B, T * G2, 2))).reshape(B, T, G2, self.d) + self.off_emb(off / 12.0)
            att = torch.einsum('btd,btgd->btg', rd['q'](e), rd['k'](f)) / self.d ** 0.5
            pooled = torch.einsum('btg,btgd->btd', att.softmax(dim=-1), rd['v'](f))
            h = rd['norm1'](e + pooled)
            h = rd['norm2'](h + rd['self_attn'](h, h, h, key_padding_mask=pad, need_weights=False)[0])
            h = rd['norm3'](h + rd['ffn'](h))
            xy = center + rd['delta'](h)
            rounds.append({'xy': xy, 'logit': rd['presence'](h).squeeze(-1)})
            center = torch.minimum(torch.maximum(xy.detach(), torch.zeros_like(lim)), lim)
            e = h
        return {'track_layers': rounds}


def build_track_inputs(target, Y, X, noise=1.5, ghosts=8, train=True):
    """Input tracks and targets from the ground truth of frames t-1 and t (mem coordinates).
    Returns dict(xy B,T,2, valid B,T, present B,T, tgt_xy B,T,2, pos_valid B,T)."""
    prev_xy, prev_pid, prev_valid = target['prev_xy'], target['prev_pid'], target['prev_valid'].bool()
    cur_xy, cur_pid, cur_valid = target['gt_xy'], target['gt_pid'], target['gt_valid'].bool()
    B, M, _ = prev_xy.shape
    dev = prev_xy.device
    same = (prev_pid.unsqueeze(2) == cur_pid.unsqueeze(1)) & cur_valid.unsqueeze(1) & (prev_pid.unsqueeze(2) >= 0)
    present = same.any(dim=2) & prev_valid  # B,M
    j = same.float().argmax(dim=2)  # B,M index of the same person at t
    tgt = torch.gather(cur_xy, 1, j.unsqueeze(-1).expand(B, M, 2))
    xy = prev_xy.clone()
    if train and noise > 0:
        xy = xy + torch.randn_like(xy) * noise
    pos_valid = present & ((tgt - xy).abs().amax(dim=-1) <= MAX_MOTION_CELLS)
    valid = prev_valid.clone()
    if ghosts > 0:
        # ghost tracks: uniformly over the grid, kept only if no one is within 1.5 m (15 cells) at t or t-1
        g = torch.rand(B, ghosts, 2, device=dev) * xy.new_tensor([X - 1, Y - 1])
        people = torch.cat([cur_xy.masked_fill(~cur_valid.unsqueeze(-1), -1e4),
                            prev_xy.masked_fill(~prev_valid.unsqueeze(-1), -1e4)], dim=1)
        far = torch.cdist(g, people).amin(dim=2) > 15
        xy = torch.cat([xy, g], dim=1)
        valid = torch.cat([valid, far], dim=1)
        present = torch.cat([present, torch.zeros_like(far)], dim=1)
        pos_valid = torch.cat([pos_valid, torch.zeros_like(far)], dim=1)
        tgt = torch.cat([tgt, g], dim=1)
    return {'xy': xy, 'valid': valid, 'present': present & valid, 'tgt_xy': tgt, 'pos_valid': pos_valid & valid}
