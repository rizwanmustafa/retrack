"""
R1 (research-methodology-5/METHODOLOGY.md): frozen Depth Anything 3 multi-view image features.

DA3's backbone is a DINOv2 ViT whose later blocks alternate per-view and cross-view ("global") attention, so the
features of one camera already depend on the other cameras before anything is lifted to 3D. TrackTacular's
ResNet-18 sees each camera alone. R1 runs the frozen DA3 backbone on all working cameras of a sample together and
adds its features to the ResNet-18 features through a small trainable adapter whose last layer is zero-initialised,
so training starts exactly from the baseline network.

Only the backbone is built (no depth/camera heads), from `depth_anything_3.model.dinov2.dinov2.DinoV2`, so none of
DA3's optional dependencies (open3d, xformers, gsplat, ...) are needed. Its weights are read from the Hugging Face
`model.safetensors` (keys `model.backbone.*`). The backbone is deliberately not an nn.Module attribute: its weights
stay out of checkpoints and of the optimizer, and Lightning's train()/eval() calls do not touch it.

Cameras that are switched off (camera removal at test time) are not given to DA3 at all, so their images cannot
influence the working cameras through cross-view attention.

Known DA3 detail: `export_feat_layers` outputs are not restored to the input camera order after DA3's reference-view
reordering (vision_transformer.py), while the `out_layers` outputs are. We use `out_layers` only.
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

DA3_COMMIT = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'  # code version the loader was written against
EMBED_DIM = {'vits': 384, 'vitb': 768, 'vitl': 1024, 'vitg': 1536}
PATCH = 14


def _import_dinov2():
    try:
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
    except ImportError:
        src = os.environ.get('DA3_SRC')
        if not src:
            raise ImportError('Depth Anything 3 not found: clone https://github.com/ByteDance-Seed/Depth-Anything-3 '
                              f'(commit {DA3_COMMIT}) and set DA3_SRC=<clone>/src')
        sys.path.insert(0, src)
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
    return DinoV2


def da3_net_config(repo):
    """The backbone ('net') part of the model's config.json on the Hugging Face hub."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(repo, 'config.json')) as f:
        cfg = json.load(f)
    net = dict(cfg['config']['net'])
    net.pop('__object__', None)
    return net


class DA3Backbone:
    """Frozen DA3 backbone. Call with ImageNet-normalised images (B,S,3,H,W); returns (B,S,L*C,h,w) float32."""

    def __init__(self, repo='depth-anything/DA3-BASE', input_hw=(378, 672)):
        assert input_hw[0] % PATCH == 0 and input_hw[1] % PATCH == 0, f'DA3 input {input_hw} must be a multiple of 14'
        self.repo = repo
        self.input_hw = tuple(int(v) for v in input_hw)
        self.cfg = da3_net_config(repo)
        width = EMBED_DIM[self.cfg['name']] * (2 if self.cfg.get('cat_token', True) else 1)
        self.num_layers = len(self.cfg['out_layers'])
        self.out_channels = width * self.num_layers
        self.out_hw = (self.input_hw[0] // PATCH, self.input_hw[1] // PATCH)
        self.net = None

    def load(self, device):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        DinoV2 = _import_dinov2()
        net = DinoV2(**self.cfg)
        state = load_file(hf_hub_download(self.repo, 'model.safetensors'))
        prefix = 'model.backbone.'
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = net.load_state_dict(state, strict=False)
        assert not missing and not unexpected, f'DA3 backbone weights: missing={missing}, unexpected={unexpected}'
        self.net = net.eval().requires_grad_(False).to(device)

    def _run(self, x):
        """x: (b,s,3,h,w) on the backbone's device -> (b,s,L*C,h/14,w/14)"""
        b, s = x.shape[:2]
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            feats, _ = self.net(x)  # tuple of (patch tokens (b,s,N,C), camera token) per out_layer
        f = torch.cat([t[0] for t in feats], dim=-1).float()  # b,s,N,L*C
        return f.view(b, s, self.out_hw[0], self.out_hw[1], -1).permute(0, 1, 4, 2, 3).contiguous()

    @torch.no_grad()
    def __call__(self, x, cam_mask=None):
        if self.net is None:
            self.load(x.device)
        elif next(self.net.parameters()).device != x.device:
            self.net.to(x.device)
        B, S = x.shape[:2]
        h, w = self.input_hw
        xs = F.interpolate(x.flatten(0, 1).float(), size=(h, w), mode='bilinear', align_corners=False,
                           antialias=True).view(B, S, 3, h, w)
        if cam_mask is None or bool((cam_mask > 0).all()):
            return self._run(xs)
        out = xs.new_zeros(B, S, self.out_channels, *self.out_hw)
        for b in range(B):
            idx = (cam_mask[b] > 0).nonzero(as_tuple=True)[0]
            if len(idx):
                out[b, idx] = self._run(xs[b:b + 1, idx])[0]
        return out


class DA3Adapter(nn.Module):
    """DA3 features (B*S, L*C, h, w) -> residual for the ResNet-18 features (B*S, out_dim, Hf, Wf).
    The last convolution is zero-initialised, so the network starts as the baseline."""

    def __init__(self, in_channels, out_dim=128, hidden=256):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats, size):
        x = self.reduce(feats)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.out(x)
