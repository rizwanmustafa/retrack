import torch
import torch.nn as nn

import utils.geom
import utils.vox
import utils.basic

from kornia.geometry.transform.imgwarp import warp_perspective
from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18, Encoder_eff, Encoder_swin_t, Encoder_res34
from models.decoder import Decoder
from models.fusion import FUSION_MODES, CameraScorer, fuse_views, coverage_map


class MVDet(nn.Module):
    def __init__(self, Y, Z, X,
                 rand_flip=False,
                 num_cameras=None,
                 num_classes=None,
                 latent_dim=256,
                 feat2d_dim=128,
                 encoder_type='res18',
                 fusion='concat',
                 coverage_map=False,
                 fusion_hidden_dim=16):
        super().__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'res34', 'effb0', 'effb4', 'swin_t'])
        assert fusion in FUSION_MODES

        self.Y, self.Z, self.X = Y, Z, X
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.num_cameras = num_cameras
        self.fusion = fusion
        self.use_coverage = coverage_map

        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)

        # Encoder
        self.feat2d_dim = feat2d_dim
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(self.feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(self.feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(self.feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(self.feat2d_dim)
        elif encoder_type == 'res34':
            self.encoder = Encoder_res34(self.feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(self.feat2d_dim)
        else:
            self.encoder = Encoder_eff(self.feat2d_dim, version='b4')

        self.cam_compressor = None
        if self.fusion == 'concat' and self.num_cameras is not None:
            self.cam_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim * self.num_cameras, latent_dim, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(latent_dim), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )
        if self.fusion == 'weighted':
            self.cam_scorer = CameraScorer(feat2d_dim, fusion_hidden_dim, spatial_dims=2)

        # mean-type fusions output feat2d_dim channels (and the coverage map adds one), bring them to latent_dim
        fused_dim = latent_dim if self.cam_compressor is not None else feat2d_dim
        self.fusion_proj = None
        if fused_dim != latent_dim or self.use_coverage:
            self.fusion_proj = nn.Sequential(
                nn.Conv2d(fused_dim + int(self.use_coverage), latent_dim, kernel_size=3, padding=1),
                nn.InstanceNorm2d(latent_dim), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )

        self.temporal_bev = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

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
        """
        B, S, C, H, W = rgb_cams.shape
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams_ = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams_)  # B*S,4,4

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)  # B*S,3,H,W
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,latent_dim,H/8,W/8
        _, C, Hf, Wf = feat_cams_.shape
        sy = Hf / float(H)
        sx = Wf / float(W)
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)  # B*S,4,4

        featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cams_T_ref_[:, :3, [0, 1, 3]])  # B*S,3,3
        ref_T_mem = vox_util.get_ref_T_mem(B, self.Y, self.Z, self.X, device=device)  # B,4,4
        ref_T_mem = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3,3
        featpix_T_mem_ = torch.matmul(featpix_T_ref_, ref_T_mem)  # B*S,3,3
        mem_T_featpix = torch.inverse(featpix_T_mem_)  # B*S,3,3
        proj_mats = mem_T_featpix  # B*S,3,3

        # B*S,latent_dim,Y,X
        feat_mems_ = warp_perspective(feat_cams_, proj_mats, (self.Y, self.X), align_corners=False)
        feat_mems = __u(feat_mems_)  # B,S,latent_dim,Y,X

        # bev_img_ = warp_perspective(rgb_cams_, proj_mats, (self.Y, self.X), align_corners=False)
        # bev_img = __u(bev_img_)
        # import matplotlib.pyplot as plt
        # plt.imshow(bev_img[0, :, :, 0].permute(1, 2, 0).cpu())
        # plt.show()

        # field of view of each camera on the ground plane
        ones_ = torch.ones_like(feat_cams_[:, :1])
        valid_mems = __u((warp_perspective(ones_, proj_mats, (self.Y, self.X), align_corners=False) > 0.5).float())

        feat_mem, mask_mems = fuse_views(feat_mems, valid_mems, cam_mask, self.fusion,
                                         cam_compressor=self.cam_compressor,
                                         scorer=getattr(self, 'cam_scorer', None))  # B, C, Y, X
        coverage = coverage_map(mask_mems) if self.use_coverage else None  # B, 1, Y, X
        if self.fusion_proj is not None:
            if coverage is not None:
                feat_mem = torch.cat([feat_mem, coverage.to(feat_mem.dtype)], dim=1)
            feat_mem = self.fusion_proj(feat_mem)

        if prev_bev is None:
            prev_bev = feat_mem
        bev_features = torch.cat([feat_mem, prev_bev], dim=1)
        bev_features = self.temporal_bev(bev_features)

        out_dict = self.decoder(bev_features, feat_cams_,
                                (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
        if coverage is not None:
            out_dict['coverage'] = coverage

        return out_dict
