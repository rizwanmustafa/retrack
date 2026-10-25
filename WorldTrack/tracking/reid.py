"""
Retrack Stage 3: camera-dropout-aware appearance descriptors for BEV detections.

For every BEV detection a person-sized box (assumed height, fixed aspect ratio) is projected into each *working*
camera with the calibration, cropped, and embedded with a pretrained re-ID network (OSNet-x1.0, MSMT17).
The per-view embeddings are averaged over the views that see the person, so a descriptor can still be formed
when cameras drop out.

OSNet architecture adapted from torchreid (Zhou et al., "Omni-Scale Feature Learning for Person
Re-Identification", ICCV 2019, MIT licence: https://github.com/KaiyangZhou/deep-person-reid).
"""
import os
import os.path as osp
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

OSNET_URL = ('https://huggingface.co/kaiyangzhou/osnet/resolve/main/'
             'osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_'
             'flip_jitter.pth')
DEFAULT_CACHE = osp.join(osp.expanduser('~'), '.cache', 'retrack', 'osnet_x1_0_msmt17.pth')


# ---------------------------------------------------------------- OSNet
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(ConvLayer):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__(in_channels, out_channels, 1, stride=stride, padding=0, groups=groups)


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super().__init__()
        mid = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + \
            self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNet(nn.Module):
    """OSNet-x1.0 feature extractor (eval mode returns the 512-d embedding)."""

    def __init__(self, channels=(64, 256, 384, 512), layers=(2, 2, 2), feature_dim=512):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim),
                                nn.ReLU(inplace=True))
        self.feature_dim = feature_dim

    @staticmethod
    def _make_layer(n, in_channels, out_channels, reduce_spatial_size):
        layers = [OSBlock(in_channels, out_channels)]
        layers += [OSBlock(out_channels, out_channels) for _ in range(1, n)]
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        return self.fc(self.global_avgpool(x).flatten(1))


def load_osnet(weights=None, device='cpu'):
    weights = weights or DEFAULT_CACHE
    if not osp.exists(weights):
        os.makedirs(osp.dirname(weights), exist_ok=True)
        print(f'downloading OSNet re-ID weights -> {weights}')
        urllib.request.urlretrieve(OSNET_URL, weights + '.part')
        os.replace(weights + '.part', weights)
    state = torch.load(weights, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    state = {k.replace('module.', '', 1): v for k, v in state.items() if not k.startswith(('classifier', 'module.classifier'))}
    model = OSNet()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, f'OSNet weights mismatch: missing={missing} unexpected={unexpected}'
    return model.to(device).eval()


# ---------------------------------------------------------------- BEV -> multi-view descriptors
class BEVReIDExtractor:
    def __init__(self, person_height, weights=None, device='cuda', z_sign=1, width_ratio=0.4, min_height_px=24,
                 min_visible=0.6, crop_size=(256, 128), max_batch=512):
        """
        person_height: assumed height in reference (world-z) units, e.g. 180 for Wildtrack (cm)
        z_sign: sign of the camera depth for points in front of the camera (as in the model config)
        """
        self.device = torch.device(device)
        self.model = load_osnet(weights, self.device)
        self.person_height = float(person_height)
        self.z_sign = z_sign
        self.width_ratio = width_ratio
        self.min_height_px = min_height_px
        self.min_visible = min_visible
        self.crop_size = crop_size
        self.max_batch = max_batch
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def project_boxes(self, intrinsic, extrinsic, ref_T_global, ref_xy, H, W):
        """returns boxes S,N,4 (x1,y1,x2,y2, clipped) and a visibility mask S,N"""
        N = ref_xy.shape[0]
        global_T_ref = torch.inverse(ref_T_global)
        pix_T_ref = intrinsic @ extrinsic @ global_T_ref  # S,4,4
        foot = torch.cat([ref_xy, torch.zeros(N, 1, device=ref_xy.device), torch.ones(N, 1, device=ref_xy.device)], 1)
        head = foot.clone()
        head[:, 2] = self.person_height
        pts = torch.stack([foot, head], 0)  # 2,N,4
        cam = torch.einsum('sij,knj->skni', pix_T_ref, pts)  # S,2,N,4
        depth = cam[..., 2]
        in_front = (self.z_sign * depth > 0).all(dim=1)  # S,N
        uv = cam[..., :2] / torch.where(depth.abs() < 1e-6, torch.full_like(depth, 1e-6), depth).unsqueeze(-1)

        cx = uv[..., 0].mean(dim=1)
        y1 = uv[..., 1].min(dim=1).values
        y2 = uv[..., 1].max(dim=1).values
        h = y2 - y1
        w = self.width_ratio * h
        x1, x2 = cx - w / 2, cx + w / 2
        area = (w * h).clamp(min=1e-6)
        cx1, cy1, cx2, cy2 = x1.clamp(0, W), y1.clamp(0, H), x2.clamp(0, W), y2.clamp(0, H)
        vis_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
        visible = in_front & (vis_area / area >= self.min_visible) & ((cy2 - cy1) >= self.min_height_px) & \
            ((cx2 - cx1) >= self.min_height_px * self.width_ratio * 0.5)
        return torch.stack([cx1, cy1, cx2, cy2], -1), visible

    @torch.no_grad()
    def __call__(self, imgs, intrinsic, extrinsic, ref_T_global, ref_xy, cam_mask=None):
        """
        imgs: S,3,H,W in [0,1]; intrinsic/extrinsic: S,4,4; ref_T_global: 4,4; ref_xy: N,2 (reference coords)
        cam_mask: S (1 = working) or None
        returns (feats N,D float32 numpy with NaN rows for detections no working camera sees, n_views N)
        """
        S, _, H, W = imgs.shape
        N = ref_xy.shape[0]
        D = self.model.feature_dim
        if N == 0:
            return np.zeros((0, D), np.float32), np.zeros(0, int)
        dev = self.device
        imgs, intrinsic, extrinsic = imgs.to(dev).float(), intrinsic.to(dev).float(), extrinsic.to(dev).float()
        ref_T_global, ref_xy = ref_T_global.to(dev).float(), ref_xy.to(dev).float()

        boxes, visible = self.project_boxes(intrinsic, extrinsic, ref_T_global, ref_xy, H, W)
        if cam_mask is not None:
            visible = visible & (cam_mask.to(dev).view(S, 1) > 0)

        cam_idx, det_idx = visible.nonzero(as_tuple=True)
        feats = torch.zeros(N, D, device=dev)
        n_views = torch.zeros(N, device=dev)
        for start in range(0, len(cam_idx), self.max_batch):
            c, d = cam_idx[start:start + self.max_batch], det_idx[start:start + self.max_batch]
            rois = torch.cat([c.float().unsqueeze(1), boxes[c, d]], 1)
            crops = roi_align(imgs, rois, output_size=self.crop_size, spatial_scale=1.0, sampling_ratio=2,
                              aligned=True)
            emb = F.normalize(self.model((crops - self.mean) / self.std), dim=1)
            feats.index_add_(0, d, emb)
            n_views.index_add_(0, d, torch.ones_like(d, dtype=n_views.dtype))

        feats = F.normalize(feats, dim=1)
        feats[n_views == 0] = float('nan')
        return feats.cpu().numpy().astype(np.float32), n_views.cpu().numpy().astype(int)
