"""
Research methodology 4 (research-methodology-4/METHODOLOGY.md): foreground masks from background subtraction.

WILDTRACK's cameras are static, so a per-camera background image (median of the training frames, computed once by
research-methodology-4/code/fg_background.py) gives a foreground mask for every image by thresholding |image -
background|. The mask is computed inside the model from the background stored as a buffer, so training-time resize and
crop augmentation is handled through the intrinsics, and every evaluation script works unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F

# soft threshold on the mean absolute RGB difference, images in [0, 1]: 0 below LO, 1 above HI (our choice, checked
# on training frames by fg_background.py)
LO, HI = 0.05, 0.15
POOL = 4  # difference computed on 4x4-averaged pixels, which suppresses sensor noise


def load_background(path):
    """(background S,3,h,w uint8, intrinsic S,3,3 of the original images, image_size (W0, H0))"""
    d = np.load(path)
    return d['background'], d['intrinsic'], d['image_size']


def foreground_mask(rgb, pix_T_cams, background, K0, size0, out_hw, lo=LO, hi=HI, pool=POOL):
    """
    rgb:        B*S,3,H,W in [0, 1], the (possibly resized and cropped) input images, b-major as pack_seqdim
    pix_T_cams: B*S,4,4 their intrinsics
    background: S,3,h,w in [0, 1], background of the original images (any resolution)
    K0:         S,3,3 intrinsics of the original images; size0: (W0, H0) original image size
    returns     B*S,1,out_hw soft foreground mask in [0, 1]
    Resize and crop only scale and shift pixel coordinates, so pixel u of the input image is pixel
    (u - cx) / fx * fx0 + cx0 of the original image, and the background is sampled there.
    """
    BS = rgb.shape[0]
    S = background.shape[0]
    img = F.avg_pool2d(rgb, pool)
    h, w = img.shape[-2:]
    dev = rgb.device
    v, u = torch.meshgrid((torch.arange(h, device=dev) + 0.5) * pool - 0.5,
                          (torch.arange(w, device=dev) + 0.5) * pool - 0.5, indexing='ij')
    K0_ = K0.repeat(BS // S, 1, 1)
    e = lambda t: t[:, None, None]
    u0 = (u[None] - e(pix_T_cams[:, 0, 2])) / e(pix_T_cams[:, 0, 0]) * e(K0_[:, 0, 0]) + e(K0_[:, 0, 2])
    v0 = (v[None] - e(pix_T_cams[:, 1, 2])) / e(pix_T_cams[:, 1, 1]) * e(K0_[:, 1, 1]) + e(K0_[:, 1, 2])
    grid = torch.stack([(u0 + 0.5) / size0[0] * 2 - 1, (v0 + 0.5) / size0[1] * 2 - 1], dim=-1)
    bg = F.grid_sample(background.repeat(BS // S, 1, 1, 1).to(img.dtype), grid.to(img.dtype), mode='bilinear',
                       padding_mode='border', align_corners=False)
    d = (img - bg).abs().mean(dim=1, keepdim=True)
    m = ((d - lo) / (hi - lo)).clamp(0, 1)
    m = F.interpolate(m, size=out_hw, mode='area')
    return F.max_pool2d(m, 3, stride=1, padding=1)  # widen by one feature cell, so person edges are kept
