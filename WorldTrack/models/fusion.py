"""
Retrack Stage 4: robust view fusion.

The projected per-camera features (B,S,C,*spatial) are fused into one BEV feature volume with one of:
  - 'concat':   TrackTacular baseline. Cameras are concatenated in a fixed order and compressed by a conv
                (`cam_compressor`). A missing camera is a slot of zeros.
  - 'masked':   mean over the cameras that are both working (explicit camera mask) and see the cell
                (field-of-view mask from the projection).
  - 'weighted': learned per-camera weighting. A tiny shared conv scores every camera at every cell and a
                softmax restricted to the valid cameras turns the scores into weights. With equal scores it
                reduces to 'masked'; the last layer is zero-initialised so training starts from 'masked'.

Optionally a coverage map c = 1/N * sum_n m_n (fraction of cameras that see each ground cell) is returned so
the model can append it to the BEV features as an extra channel.
"""
import torch
import torch.nn as nn

import utils.basic

FUSION_MODES = ('concat', 'masked', 'weighted')
EPS = 1e-6


def expand_cam_mask(cam_mask, ref):
    """cam_mask: B,S -> B,S,1,...(broadcastable to ref B,S,C,*spatial)"""
    return cam_mask.to(ref.dtype).view(cam_mask.shape + (1,) * (ref.dim() - 2))


def apply_cam_mask(feat_mems, cam_mask):
    """Switch off cameras by zeroing their projected features. feat_mems: B,S,C,*spatial, cam_mask: B,S"""
    if cam_mask is None:
        return feat_mems
    return feat_mems * expand_cam_mask(cam_mask, feat_mems)


class CameraScorer(nn.Module):
    """Few-parameter conv net shared by all cameras that scores each camera's features at every cell."""

    def __init__(self, in_dim, hidden_dim=16, spatial_dims=3):
        super().__init__()
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.net = nn.Sequential(
            Conv(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv(hidden_dim, 1, kernel_size=1),
        )
        # start from equal scores, i.e. from the masked mean
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat_mems):
        """feat_mems: B,S,C,*spatial -> scores B,S,1,*spatial"""
        B, S = feat_mems.shape[:2]
        scores = self.net(feat_mems.flatten(0, 1))
        return scores.view(B, S, *scores.shape[1:])


def masked_softmax_weights(scores, mask):
    """
    w_n = m_n exp(s_n) / sum_k m_k exp(s_k) over the camera dim (1).
    Cells seen by no valid camera get all-zero weights.
    """
    scores = scores.float()
    s_max = scores.masked_fill(mask == 0, -1e4).amax(dim=1, keepdim=True)
    e = torch.exp((scores - s_max).clamp(max=0.0)) * mask
    return e / (e.sum(dim=1, keepdim=True) + EPS)


def fuse_views(feat_mems, valid_mems, cam_mask, mode, cam_compressor=None, scorer=None):
    """
    feat_mems: B,S,C,*spatial projected camera features
    valid_mems: B,S,1,*spatial field-of-view mask of each camera (1 where the cell projects into the image)
    cam_mask: B,S (1 = camera working) or None
    returns fused features B,C',*spatial and the combined mask m_n(x) B,S,1,*spatial
    """
    if cam_mask is None:
        cam_mask = torch.ones(feat_mems.shape[:2], device=feat_mems.device)
    mask_mems = valid_mems * expand_cam_mask(cam_mask, valid_mems)
    feat_mems = apply_cam_mask(feat_mems, cam_mask)

    if mode == 'concat':
        if cam_compressor is None:
            # original num_cameras=None behaviour: a camera counts as present where its feature is non-zero
            legacy_mask = (torch.abs(feat_mems) > 0).float()
            fused = utils.basic.reduce_masked_mean(feat_mems, legacy_mask, dim=1)
        else:
            fused = cam_compressor(feat_mems.flatten(1, 2))
    elif mode == 'masked':
        fused = utils.basic.reduce_masked_mean(feat_mems, mask_mems.expand_as(feat_mems), dim=1)
    elif mode == 'weighted':
        weights = masked_softmax_weights(scorer(feat_mems), mask_mems)
        fused = (weights.to(feat_mems.dtype) * feat_mems).sum(dim=1)
    else:
        raise ValueError(f'Unknown fusion mode {mode}, expected one of {FUSION_MODES}')
    return fused, mask_mems


def coverage_map(mask_mems, height_dim=None):
    """
    c(x, y) = 1/N sum_n m_n(x, y), N = number of camera slots.
    mask_mems: B,S,1,*spatial; height_dim: index of the height axis in the output to average away (3D grids)
    returns B,1,Y,X
    """
    cov = mask_mems.float().mean(dim=1)  # B,1,*spatial
    if height_dim is not None:
        cov = cov.mean(dim=height_dim)
    return cov
