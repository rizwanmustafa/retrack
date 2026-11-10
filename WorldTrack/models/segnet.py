import torch
import torch.nn as nn
import numpy as np
import utils.geom
import utils.vox
import utils.basic

from kornia.geometry.transform.imgwarp import warp_perspective
from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18, Encoder_eff, Encoder_swin_t
from models.decoder import Decoder
from models.fusion import FUSION_MODES, CameraScorer, fuse_views, coverage_map, view_diversity

VGF_PRIORS = ('coverage', 'diversity')


class Segnet(nn.Module):
    def __init__(self, Y, Z, X,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=256,
                 feat2d_dim=128,
                 num_classes=None,
                 num_cameras=None,
                 z_sign=1,
                 encoder_type='swin_t',
                 fusion='concat',
                 coverage_map=False,
                 fusion_hidden_dim=16,
                 vgf=False,
                 ray_cond=False,
                 vgf_prior='diversity',
                 view_mix=False):
        """
        Research methodology 2 (research-methodology-2/METHODOLOGY.md), all off by default:
        vgf:       volumetric-guided fusion of this (volumetric) BEV feature with a homography-projected one, GRACE Eq. 3-5
        ray_cond:  each camera's ground-plane viewing direction is appended to its lifted features, GRACE Eq. 6-8.
                   The rays are computed per sample in the (scene-augmented) reference frame.
        vgf_prior: fixed half of the VGF gate. 'coverage' = GRACE Eq. 4 (fraction of cameras seeing the cell),
                   'diversity' = ours (spread of the viewing directions of the cameras seeing the cell)
        view_mix:  training only. Reproduces on purpose the old repeat() order: with batch 2, cameras 1, 3, 5 of each
                   sample are lifted in the other sample's (scene-augmented) reference frame while the targets stay.
                   Attempt 1's A2 trained this way and tracked better than the fixed A2f (research-methodology-2)
        """
        super(Segnet, self).__init__()
        assert (encoder_type in ['res101', 'res50', 'effb0', 'effb4', 'res18', 'vgg11', 'swin_t'])
        assert fusion in FUSION_MODES
        assert vgf_prior in VGF_PRIORS, vgf_prior
        if vgf:
            assert fusion == 'concat' and num_cameras is not None, 'VGF is implemented for the concat fusion only'
            assert not rand_flip, 'VGF does not flip the homography branch'

        self.Y, self.Z, self.X = Y, Z, X
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.z_sign = z_sign
        self.num_cameras = num_cameras
        self.fusion = fusion
        self.use_coverage = coverage_map
        self.vgf = vgf
        self.ray_cond = ray_cond
        self.vgf_prior = vgf_prior
        self.view_mix = view_mix

        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)

        # Encoder
        self.feat2d_dim = feat2d_dim
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(feat2d_dim)
        else:
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEV compressor
        self.cam_compressor = None
        if self.fusion == 'concat' and self.num_cameras is not None:
            self.cam_compressor = nn.Sequential(
                nn.Conv3d(feat2d_dim * self.num_cameras, feat2d_dim, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm3d(feat2d_dim), nn.ReLU(),
                nn.Conv3d(feat2d_dim, feat2d_dim, kernel_size=1),
            )
        if self.fusion == 'weighted':
            self.cam_scorer = CameraScorer(feat2d_dim, fusion_hidden_dim, spatial_dims=3)

        # Ray conditioning: rho of GRACE Eq. 7, shared by all cameras. Initialised to pass the image features through
        # unchanged (ray weights zero), so training starts from the baseline fusion.
        if self.ray_cond:
            self.ray_rho = nn.Conv3d(feat2d_dim + 2, feat2d_dim, kernel_size=1)
            with torch.no_grad():
                self.ray_rho.weight.zero_()
                self.ray_rho.weight[:, :feat2d_dim, 0, 0, 0] = torch.eye(feat2d_dim)
                self.ray_rho.bias.zero_()

        # VGF: homography-projected branch H (MVDet projection, fused by concatenation like models/mvdet.py) and the
        # learned gate of GRACE Eq. 3, zero-initialised so the learned half starts at 0.5
        if self.vgf:
            self.hom_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim * self.num_cameras, latent_dim, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(latent_dim), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )
            self.vgf_gate = nn.Conv2d(latent_dim * 2 + 1, 1, kernel_size=1)
            nn.init.zeros_(self.vgf_gate.weight)
            nn.init.zeros_(self.vgf_gate.bias)

        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.feat2d_dim * self.Z + int(self.use_coverage), latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        self.temporal_bev = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=num_classes,
            feat2d=self.feat2d_dim,
        )

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.tracking_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, prev_bev=None, cam_mask=None):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        cam_mask: (B,S) 1 = camera working, 0 = camera switched off; None = all working
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        """
        B, S, C, H, W = rgb_cams.shape
        assert (C == 3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        # repeat_interleave matches pack_seqdim's b-major order; repeat() gave camera i the frame of sample i % B,
        # which mattered once samples of a batch have different (scene-augmented) reference frames
        ref_T_cams_global = ref_T_global.repeat(S, 1, 1) if (self.view_mix and self.training) \
            else ref_T_global.repeat_interleave(S, dim=0)
        ref_T_cams = torch.matmul(ref_T_cams_global, global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams)  # B*S,4,4

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_cams_.shape
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            rgb_cams_[self.rgb_flip_index] = torch.flip(rgb_cams_[self.rgb_flip_index], [-1])
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,128,H/8,W/8
        if self.rand_flip:
            feat_cams_[self.rgb_flip_index] = torch.flip(feat_cams_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_cams_.shape

        sy = Hf / float(H)
        sx = Wf / float(W)

        Y, Z, X = self.Y, self.Z, self.X
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)

        # unproject image feature to 3d grid
        feat_mems_, valid_mems_ = vox_util.unproject_image_to_mem(
            feat_cams_,  # B*S,128,H/8,W/8
            utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),  # featpix_T_ref  B*S,4,4
            cams_T_ref_, Y, Z, X,
            xyz_refA=None, z_sign=self.z_sign, return_valid=True)
        feat_mems = __u(feat_mems_)  # B, S, C, Y, Z, X
        valid_mems = __u(valid_mems_)  # B, S, 1, Y, Z, X

        rays = None
        if self.ray_cond or (self.vgf and self.vgf_prior == 'diversity'):
            rays = self.ground_rays(ref_T_cams, vox_util, B, S)  # B, S, 2, Y, X
        if self.ray_cond:
            ray_mems = rays.unsqueeze(4).expand(-1, -1, -1, -1, Z, -1)  # B, S, 2, Y, Z, X
            feat_mems_ = self.ray_rho(__p(torch.cat([feat_mems, ray_mems.to(feat_mems.dtype)], dim=2)))
            feat_mems = __u(feat_mems_) * valid_mems  # cells a camera does not see stay zero

        # rgb_cams_raw = rgb_cams_ * self.std.to(device) + self.mean.to(device)
        # rgb_mem_ = vox_util.unproject_image_to_mem(
        #     rgb_cams_raw,  # B*S,128,H/8,W/8
        #     utils.basic.matmul2(pix_T_cams_, cams_T_ref_),  # featpix_T_ref  B*S,4,4
        #     cams_T_ref_, Y, Z, X, z_sign=self.z_sign,
        #     xyz_refA=None)
        # rgb_mems = __u(rgb_mem_)  # B, S, C, Z, Y, X
        # mask_mems = (torch.abs(rgb_mems) > 0).float()
        # rgb_mem = utils.basic.reduce_masked_mean(rgb_mems, mask_mems, dim=1)
        # import matplotlib.pyplot as plt
        # plt.imshow(rgb_mem[0, :, :, 0].permute(1, 2, 0).cpu())
        # plt.show()

        feat_mem, mask_mems = fuse_views(feat_mems, valid_mems, cam_mask, self.fusion,
                                         cam_compressor=self.cam_compressor,
                                         scorer=getattr(self, 'cam_scorer', None))  # B, C, Y, Z, X
        coverage = coverage_map(mask_mems, height_dim=3) if self.use_coverage else None  # B, 1, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])
            if coverage is not None:
                coverage[self.bev_flip1_index] = torch.flip(coverage[self.bev_flip1_index], [-1])
                coverage[self.bev_flip2_index] = torch.flip(coverage[self.bev_flip2_index], [-2])

        # bev compressing
        feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Z, Y, X)
        if coverage is not None:
            feat_bev_ = torch.cat([feat_bev_, coverage.to(feat_bev_.dtype)], dim=1)
        feat_bev = self.bev_compressor(feat_bev_)

        if self.vgf:
            feat_bev = self.volumetric_guided_fusion(feat_bev, feat_cams_, featpix_T_cams_, cams_T_ref_, vox_util,
                                                     mask_mems, rays, cam_mask, B)

        if prev_bev is None:
            prev_bev = feat_bev
        feat_bev = torch.cat([feat_bev, prev_bev], dim=1)
        feat_bev = self.temporal_bev(feat_bev)

        # bev decoder
        out_dict = self.decoder(feat_bev, feat_cams_)
        if coverage is not None:
            out_dict['coverage'] = coverage

        return out_dict

    def ground_rays(self, ref_T_cams, vox_util, B, S):
        """GRACE Eq. 6: unit ground-plane direction from each camera centre to every BEV cell, B,S,2,Y,X.
        Computed from the current reference frame, so a scene-augmented (flipped) frame gets flipped rays."""
        Y, Z, X = self.Y, self.Z, self.X
        xyz_mem = utils.basic.gridcloud3d(1, Y, 1, X, norm=False, device=ref_T_cams.device)  # 1, Y*X, 3 (z = 0)
        xy_ref = vox_util.Mem2Ref(xyz_mem, Y, Z, X)[..., :2]  # 1, Y*X, 2
        d = xy_ref - ref_T_cams[:, None, :2, 3]  # B*S, Y*X, 2
        r = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return r.reshape(B, S, Y, X, 2).permute(0, 1, 4, 2, 3)

    def volumetric_guided_fusion(self, V, feat_cams_, featpix_T_cams_, cams_T_ref_, vox_util, mask_mems, rays,
                                 cam_mask, B):
        """GRACE Eq. 3-5 with our choice of prior: Z = g V + (1 - g) H, g = 1/2 sigma(conv(H, V, p)) + 1/2 (1 - p)."""
        Y, Z, X = self.Y, self.Z, self.X
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        # homography branch: ground-plane projection of every camera, as models/mvdet.py
        featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cams_T_ref_[:, :3, [0, 1, 3]])  # B*S,3,3
        ref_T_mem = vox_util.get_ref_T_mem(1, Y, Z, X, device=V.device)[0, [0, 1, 3]][:, [0, 1, 3]]  # 3,3
        mem_T_featpix = torch.inverse(torch.matmul(featpix_T_ref_, ref_T_mem))  # B*S,3,3
        hom_mems = __u(warp_perspective(feat_cams_, mem_T_featpix, (Y, X), align_corners=False))  # B,S,C,Y,X
        ones_ = torch.ones_like(feat_cams_[:, :1])
        hom_valid = __u((warp_perspective(ones_, mem_T_featpix, (Y, X), align_corners=False) > 0.5).float())
        H, _ = fuse_views(hom_mems, hom_valid, cam_mask, 'concat', cam_compressor=self.hom_compressor)  # B,latent,Y,X

        if self.vgf_prior == 'coverage':
            p = coverage_map(mask_mems, height_dim=3)  # B,1,Y,X, GRACE's m
        else:
            p = view_diversity(rays, mask_mems.float().mean(dim=4))  # B,1,Y,X
        p = p.to(V.dtype)
        g = 0.5 * torch.sigmoid(self.vgf_gate(torch.cat([H, V, p], dim=1))) + 0.5 * (1 - p)
        return g * V + (1 - g) * H
