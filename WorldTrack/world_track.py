import os
import os.path as osp
from typing import List, Optional

import torch
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from models import Segnet, MVDet, Liftnet, Bevformernet
from models.loss import FocalLoss, compute_rot_loss
from models.track_queries import build_track_inputs, MAX_MOTION_CELLS
from tracking.multitracker import build_tracker
from utils import vox, basic, decode
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics


class WorldTrackModel(pl.LightningModule):
    def __init__(
            self,
            model_name='segnet',
            encoder_name='res18',
            learning_rate=0.001,
            resolution=(200, 4, 200),
            bounds=(-75, 75, -75, 75, -1, 5),
            num_cameras=None,
            depth=(100, 2.0, 25),
            scene_centroid=(0.0, 0.0, 0.0),
            max_detections=60,
            conf_threshold=0.5,
            num_classes=1,
            use_temporal_cache=True,
            z_sign=1,
            feat2d_dim=128,
            # Retrack Stage 4: robust view fusion
            fusion='concat',  # 'concat' (baseline) | 'masked' | 'weighted'
            coverage_map=False,
            fusion_hidden_dim=16,
            # Research methodology 2 (segnet only): volumetric-guided fusion, ray conditioning, gate prior
            vgf=False,
            ray_cond=False,
            vgf_prior='diversity',  # 'coverage' (GRACE) | 'diversity' (ours)
            view_mix=False,  # training only: some cameras lifted in the other sample's frame (the old repeat() order)
            view_mix_random=0.0,  # training only: per-camera probability of the partner's frame (random view mixing)
            cam_dropout_prob=0.0,  # probability that a training sample has cameras switched off
            cam_dropout_max=3,  # at most this many cameras are switched off per sample
            # Research methodology 4 (segnet only): background .npz -> foreground-gated lifting
            fg_background: Optional[str] = None,
            test_drop_cams: Optional[List[int]] = None,  # cameras switched off at test time
            # Research methodology 5 (segnet only): R1 DA3 features, R2 sparse queries, R4 track queries
            da3=False,
            da3_repo='depth-anything/DA3-BASE',
            da3_input_hw=(378, 672),
            sparse_queries=False,
            num_queries=100,
            query_layers=3,
            query_heights=4,
            query_match_cells=10.0,  # R2: a query is assigned to a person only within 10 BEV cells (1 m)
            track_queries=False,
            track_noise=1.5,  # R4 training: jitter (BEV cells) of the previous-frame positions given as tracks
            track_ghosts=8,  # R4 training: ghost tracks per sample that must be predicted as gone
            # Retrack Stages 2-3: tracker used by `test`
            tracker='baseline',  # 'baseline' | 'byte'
            track_buffer=5,
            byte_low_thresh=0.1,
            use_reid=False,
            reid_lambda=0.98,
            reid_thresh=0.6,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.Y, self.Z, self.X = self.resolution
        self.bounds = bounds
        self.max_detections = max_detections
        self.D, self.DMIN, self.DMAX = depth
        self.conf_threshold = conf_threshold

        # Loss
        self.center_loss_fn = FocalLoss()

        # Temporal cache
        self.use_temporal_cache = use_temporal_cache
        self.max_cache = 32
        self.temporal_cache_frames = -2 * torch.ones(self.max_cache, dtype=torch.long)
        self.temporal_cache = None
        # ARL: scene transform (raw reference -> augmented reference, 3x3) each cached BEV was computed in
        self.temporal_cache_T = torch.eye(3).repeat(self.max_cache, 1, 1)

        # Test
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        self.mota_seq_gt_list, self.mota_seq_pred_list = [], []
        self.frame = 0
        self.z_sign = z_sign
        self.tracker_kwargs = dict(tracker=tracker, conf_thres=self.conf_threshold, track_buffer=track_buffer,
                                   low_thres=byte_low_thresh, use_reid=use_reid, reid_lambda=reid_lambda,
                                   reid_thresh=reid_thresh)
        self.test_tracker = build_tracker(**self.tracker_kwargs)
        self.reid_extractor = None

        # Camera dropout
        self.cam_dropout_prob = cam_dropout_prob
        self.cam_dropout_max = cam_dropout_max
        self.test_drop_cams = list(test_drop_cams) if test_drop_cams else []

        # Model
        num_cameras = None if num_cameras == 0 else num_cameras
        fusion_kwargs = dict(fusion=fusion, coverage_map=coverage_map, fusion_hidden_dim=fusion_hidden_dim)
        assert model_name == 'segnet' or not (vgf or ray_cond or view_mix or view_mix_random or fg_background), \
            'vgf / ray_cond / view_mix / fg_background are implemented for segnet only'
        assert model_name == 'segnet' or not (da3 or sparse_queries or track_queries), \
            'da3 / sparse_queries / track_queries are implemented for segnet only'
        self.sparse_queries = sparse_queries
        self.query_match_cells = query_match_cells
        self.track_queries = track_queries
        self.track_noise = track_noise
        self.track_ghosts = track_ghosts
        if model_name == 'segnet':
            self.model = Segnet(self.Y, self.Z, self.X, num_cameras=num_cameras, feat2d_dim=feat2d_dim,
                                encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign,
                                vgf=vgf, ray_cond=ray_cond, vgf_prior=vgf_prior, view_mix=view_mix,
                                view_mix_random=view_mix_random, fg_background=fg_background,
                                da3=da3, da3_repo=da3_repo, da3_input_hw=tuple(da3_input_hw),
                                sparse_queries=sparse_queries, num_queries=num_queries, query_layers=query_layers,
                                query_heights=query_heights, track_queries=track_queries, **fusion_kwargs)
        elif model_name == 'liftnet':
            self.model = Liftnet(self.Y, self.Z, self.X, encoder_type=self.encoder_name, feat2d_dim=feat2d_dim,
                                 DMIN=self.DMIN, DMAX=self.DMAX, D=self.D, num_classes=num_classes, z_sign=z_sign,
                                 num_cameras=num_cameras, **fusion_kwargs)
        elif model_name == 'bevformer':
            assert Bevformernet is not None, 'bevformer needs mmcv'
            assert fusion == 'concat' and not coverage_map, 'robust fusion is not implemented for bevformer'
            self.model = Bevformernet(self.Y, self.Z, self.X, feat2d_dim=feat2d_dim,
                                      encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign)
        elif model_name == 'mvdet':
            self.model = MVDet(self.Y, self.Z, self.X, encoder_type=self.encoder_name,
                               num_cameras=num_cameras, num_classes=num_classes, **fusion_kwargs)
        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        self.scene_centroid = torch.tensor(scene_centroid, device=self.device).reshape([1, 3])
        self.vox_util = vox.VoxelUtil(self.Y, self.Z, self.X, scene_centroid=self.scene_centroid, bounds=self.bounds)
        self.save_hyperparameters()

    def forward(self, item, track_xy=None, track_valid=None):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        ref_T_global: (B,4,4)
        vox_util: vox util object
        """
        B = item['img'].shape[0]
        scene_T = item.get('scene_T')
        scene_T = torch.eye(3).repeat(B, 1, 1) if scene_T is None else scene_T.detach().float().cpu()
        prev_bev, prev_T = self.load_cache(item['frame'].cpu())
        if prev_bev is not None:
            prev_bev = self.align_prev_bev(prev_bev, prev_T, scene_T)
        cam_mask = self.get_cam_mask(item)
        kwargs = {} if cam_mask is None else {'cam_mask': cam_mask}
        if track_xy is not None:
            kwargs.update(track_xy=track_xy, track_valid=track_valid)

        output = self.model(
            rgb_cams=item['img'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=prev_bev,
            **kwargs,
        )

        if self.use_temporal_cache:
            self.store_cache(item['frame'].cpu(), output['bev_raw'].clone().detach(), scene_T)

        return output

    def get_cam_mask(self, item):
        """
        B,S mask of working cameras, or None when all cameras work (keeps the baseline forward unchanged).
        Training: camera-dropout augmentation. Otherwise: item['cam_mask'] if given, else test_drop_cams.
        """
        B, S = item['img'].shape[:2]
        device = item['img'].device
        if self.training:
            if self.cam_dropout_prob <= 0:
                return None
            cam_mask = torch.ones(B, S, device=device)
            for b in range(B):
                if torch.rand(1).item() < self.cam_dropout_prob:
                    k = int(torch.randint(1, min(self.cam_dropout_max, S - 1) + 1, (1,)).item())
                    cam_mask[b, torch.randperm(S)[:k]] = 0
            return cam_mask
        if 'cam_mask' in item:
            return item['cam_mask'].to(device)
        if self.test_drop_cams:
            cam_mask = torch.ones(B, S, device=device)
            cam_mask[:, self.test_drop_cams] = 0
            return cam_mask
        return None

    def load_cache(self, frames):
        """returns (BEV(t-1) for every frame of the batch, their scene transforms) or (None, None)"""
        idx = []
        for frame in frames:
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            if i.nelement() == 1:
                idx.append(i.item())
        if len(idx) != len(frames) or self.temporal_cache is None:
            return None, None
        else:
            return self.temporal_cache[idx], self.temporal_cache_T[idx]

    def store_cache(self, frames, bev_feat, scene_T=None):
        if self.temporal_cache is None:
            shape = list(bev_feat.shape)
            shape[0] = self.max_cache
            self.temporal_cache = torch.zeros(shape, device=bev_feat.device, dtype=bev_feat.dtype)
        if scene_T is None:
            scene_T = torch.eye(3).repeat(len(frames), 1, 1)

        for frame, feat, T in zip(frames, bev_feat, scene_T):
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            # Choose unfilled cache slot
            if i.nelement() == 0:
                i = (self.temporal_cache_frames == -2).nonzero(as_tuple=True)[0]
            # Choose random cache slot
            if i.nelement() == 0:
                i = torch.randint(self.max_cache, (1, 1))

            self.temporal_cache[i[0]] = feat
            self.temporal_cache_frames[i[0]] = frame
            self.temporal_cache_T[i[0]] = T

    def align_prev_bev(self, prev_bev, prev_T, cur_T, tol_cells=0.5):
        """
        ARL A2 x A3: BEV(t-1) was computed in the reference frame prev_T (raw -> augmented reference coordinates,
        B,3,3) and BEV(t) in cur_T. Resamples BEV(t-1) into the frame of BEV(t) so that history fusion sees
        aligned features under scene augmentation. Without scene augmentation (test, or the baseline's sub-cell
        shift) the frames agree to within `tol_cells` reference cells and BEV(t-1) is returned unchanged.
        """
        M = prev_T @ torch.inverse(cur_T)  # augmented(t) -> raw -> augmented(t-1)
        if (M - torch.eye(3)).abs()[:, :2, :2].max() < 1e-6 and (M[:, :2, 2]).abs().max() < tol_cells:
            return prev_bev
        B, _, Yb, Xb = prev_bev.shape
        dev = prev_bev.device
        # BEV feature pixel i is the voxel centred at mem coordinate i (utils/vox.py: mem = (ref - min) / size - 0.5,
        # and unproject_image_to_mem samples the voxels at integer mem coordinates)
        ys, xs = torch.meshgrid(torch.arange(Yb, device=dev, dtype=torch.float32),
                                torch.arange(Xb, device=dev, dtype=torch.float32), indexing='ij')
        mem = torch.stack([xs.flatten(), ys.flatten(), torch.zeros_like(xs.flatten())], -1)
        mem = mem.unsqueeze(0).repeat(B, 1, 1)  # pixel centres of BEV(t), mem coords (x, y, z)
        ref = self.vox_util.Mem2Ref(mem, self.Y, self.Z, self.X)
        xy1 = torch.cat([ref[..., :2], torch.ones_like(ref[..., :1])], -1)
        ref_prev = torch.cat([(xy1 @ M.to(dev).transpose(1, 2))[..., :2], ref[..., 2:3]], -1)
        mem_prev = self.vox_util.Ref2Mem(ref_prev, self.Y, self.Z, self.X)
        # pixel centre i <-> mem coordinate i, hence align_corners=True
        grid = torch.stack([mem_prev[..., 0] / max(Xb - 1, 1) * 2 - 1, mem_prev[..., 1] / max(Yb - 1, 1) * 2 - 1],
                           -1).view(B, Yb, Xb, 2)
        return torch.nn.functional.grid_sample(prev_bev, grid.to(prev_bev.dtype), mode='bilinear',
                                               padding_mode='zeros', align_corners=True)

    def loss(self, target, output, track_inputs=None):
        if 'query_layers' in output:
            total_loss, loss_dict = self.sparse_loss(target, output)
        else:
            total_loss, loss_dict = self.dense_loss(target, output)
        if track_inputs is not None and 'track_layers' in output:
            track_loss, track_dict = self.track_loss(track_inputs, output)
            total_loss = total_loss + track_loss
            loss_dict.update(track_dict)
        return total_loss, loss_dict

    def image_center_loss(self, target, output):
        B, S = target['center_img'].shape[:2]
        center_img_g = basic.pack_seqdim(target['center_img'], B)
        return self.center_loss_fn(basic.sigmoid(output['img_center']), center_img_g) / S

    def sparse_loss(self, target, output):
        """R2: focal loss on the proposal heatmap (same target and uncertainty weighting as the baseline's centre
        heatmap) + per decoder layer, after one-to-one Hungarian assignment of queries to people: sigmoid focal loss
        on the scores, smooth L1 on the positions and L1 on the motion to the previous frame."""
        from scipy.optimize import linear_sum_assignment
        from torchvision.ops import sigmoid_focal_loss
        center_loss = self.center_loss_fn(basic.sigmoid(output['proposal_center']), target['center_bev'])
        center_factor = 1 / torch.exp(self.model.center_weight)
        gt_xy, gt_valid = target['gt_xy'], target['gt_valid'].bool()
        prev_xy, prev_pid, prev_valid = target['prev_xy'], target['prev_pid'], target['prev_valid'].bool()
        same = (target['gt_pid'].unsqueeze(2) == prev_pid.unsqueeze(1)) & prev_valid.unsqueeze(1) & \
            (prev_pid.unsqueeze(1) >= 0)
        has_prev = same.any(dim=2)
        motion_g = torch.gather(prev_xy, 1, same.float().argmax(dim=2).unsqueeze(-1).expand_as(gt_xy)) - gt_xy
        motion_ok = has_prev & gt_valid & (motion_g.abs().amax(dim=-1) <= MAX_MOTION_CELLS)
        B = gt_xy.shape[0]
        n_gt = max(int(gt_valid.sum()), 1)
        score_l = pos_l = motion_l = 0.0
        layers = output['query_layers']
        for layer in layers:
            xy, logit, motion = layer['xy'], layer['logit'], layer['motion']
            tgt = torch.zeros_like(logit)
            pos_terms, motion_terms = [], []
            for b in range(B):
                g = gt_valid[b].nonzero(as_tuple=True)[0]
                if len(g) == 0:
                    continue
                with torch.no_grad():
                    dist = torch.cdist(xy[b].float(), gt_xy[b, g].float())  # K,G
                    cost = dist / self.query_match_cells + (1 - logit[b].sigmoid()).unsqueeze(1)
                    cost = torch.where(dist <= self.query_match_cells, cost, torch.full_like(cost, 1e6))
                    cost_np = cost.cpu().numpy()
                    r, c = linear_sum_assignment(cost_np)
                    ok = cost_np[r, c] < 1e5
                r, c = torch.as_tensor(r[ok], device=xy.device), torch.as_tensor(c[ok], device=xy.device)
                if len(r) == 0:
                    continue
                tgt[b, r] = 1
                gi = g[c]
                pos_terms.append(torch.nn.functional.smooth_l1_loss(xy[b, r], gt_xy[b, gi], reduction='none').sum(-1))
                mo = motion_ok[b, gi]
                if mo.any():
                    motion_terms.append((motion[b, r[mo]] - motion_g[b, gi[mo]]).abs().sum(-1))
            score_l = score_l + sigmoid_focal_loss(logit, tgt, alpha=0.25, gamma=2.0, reduction='sum') / n_gt
            if pos_terms:
                pos_l = pos_l + torch.cat(pos_terms).mean()
            if motion_terms:
                motion_l = motion_l + torch.cat(motion_terms).mean()
        L = len(layers)
        score_l, pos_l, motion_l = score_l / L, pos_l / L, motion_l / L
        center_img_loss = self.image_center_loss(target, output)
        total = 10 * center_factor * center_loss + self.model.center_weight + 2 * score_l + pos_l + motion_l \
            + center_img_loss
        as_t = lambda v: v if torch.is_tensor(v) else torch.tensor(float(v))
        return total, {'center_loss': 10 * center_loss, 'query_score': as_t(score_l), 'query_pos': as_t(pos_l),
                       'query_motion': as_t(motion_l), 'center_img': center_img_loss}

    def track_loss(self, track_inputs, output):
        """R4: per round, BCE on 'still there' and smooth L1 on the new position of the people that are."""
        valid, present, pos_valid = track_inputs['valid'], track_inputs['present'], track_inputs['pos_valid']
        pres_l = pos_l = 0.0
        rounds = output['track_layers']
        for rd in rounds:
            if valid.any():
                pres_l = pres_l + torch.nn.functional.binary_cross_entropy_with_logits(
                    rd['logit'][valid], present[valid].float())
            if pos_valid.any():
                pos_l = pos_l + torch.nn.functional.smooth_l1_loss(
                    rd['xy'][pos_valid], track_inputs['tgt_xy'][pos_valid], reduction='none').sum(-1).mean()
        as_t = lambda v: v if torch.is_tensor(v) else torch.tensor(float(v))
        pres_l, pos_l = as_t(pres_l) / len(rounds), as_t(pos_l) / len(rounds)
        return pres_l + pos_l, {'track_presence': pres_l, 'track_pos': pos_l}

    def track_inputs(self, target, train):
        if not self.track_queries:
            return None
        return build_track_inputs(target, self.Y, self.X, noise=self.track_noise if train else 0.0,
                                  ghosts=self.track_ghosts, train=train)

    def dense_loss(self, target, output):
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        center_img_e = output['img_center']

        valid_g = target['valid_bev']
        center_g = target['center_bev']
        offset_g = target['offset_bev']

        B, S = target['center_img'].shape[:2]
        center_img_g = basic.pack_seqdim(target['center_img'], B)

        center_loss = self.center_loss_fn(basic.sigmoid(center_e), center_g)
        offset_loss = torch.abs(offset_e[:, :2] - offset_g[:, :2]).sum(dim=1, keepdim=True)
        offset_loss = basic.reduce_masked_mean(offset_loss, valid_g)
        tracking_loss = torch.nn.functional.smooth_l1_loss(
            offset_e[:, 2:], offset_g[:, 2:], reduction='none').sum(dim=1, keepdim=True)
        tracking_loss = basic.reduce_masked_mean(tracking_loss, valid_g)

        if 'size_bev' in target:
            size_g = target['size_bev']
            rotbin_g = target['rotbin_bev']
            rotres_g = target['rotres_bev']
            size_loss = torch.abs(size_e - size_g).sum(dim=1, keepdim=True)
            size_loss = basic.reduce_masked_mean(size_loss, valid_g)
            rot_loss = compute_rot_loss(rot_e, rotbin_g, rotres_g, valid_g)
        else:
            size_loss = torch.tensor(0.)
            rot_loss = torch.tensor(0.)

        center_factor = 1 / torch.exp(self.model.center_weight)
        center_loss_weight = center_factor * center_loss
        center_uncertainty_loss = self.model.center_weight

        offset_factor = 1 / torch.exp(self.model.offset_weight)
        offset_loss_weight = offset_factor * offset_loss
        offset_uncertainty_loss = self.model.offset_weight

        size_factor = 1 / torch.exp(self.model.size_weight)
        size_loss_weight = size_factor * size_loss
        size_uncertainty_loss = self.model.size_weight

        rot_factor = 1 / torch.exp(self.model.rot_weight)
        rot_loss_weight = rot_factor * rot_loss
        rot_uncertainty_loss = self.model.rot_weight

        tracking_factor = 1 / torch.exp(self.model.tracking_weight)
        tracking_loss_weight = tracking_factor * tracking_loss
        tracking_uncertainty_loss = self.model.tracking_weight

        # img loss
        center_img_loss = self.center_loss_fn(basic.sigmoid(center_img_e), center_img_g) / S

        loss_dict = {
            'center_loss': 10 * center_loss,
            'offset_loss': 10 * offset_loss,
            'tracking_loss': tracking_loss,
            'size_loss': size_loss,
            'rot_loss': rot_loss,
            'center_img': center_img_loss,
        }
        loss_weight_dict = {
            'center_loss': 10 * center_loss_weight,
            'offset_loss': 10 * offset_loss_weight,
            'tracking_loss': tracking_loss_weight,
            'size_loss': size_loss_weight,
            'rot_loss': rot_loss_weight,
            'center_img': center_img_loss,
        }
        stats_dict = {
            'center_uncertainty_loss': center_uncertainty_loss,
            'offset_uncertainty_loss': offset_uncertainty_loss,
            'tracking_uncertainty_loss': tracking_uncertainty_loss,
            'size_uncertainty_loss': size_uncertainty_loss,
            'rot_uncertainty_loss': rot_uncertainty_loss,
        }
        total_loss = sum(loss_weight_dict.values()) + sum(stats_dict.values())

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        item, target = batch
        tin = self.track_inputs(target, train=True)
        output = self(item) if tin is None else self(item, tin['xy'], tin['valid'])

        total_loss, loss_dict = self.loss(target, output, tin)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        tin = self.track_inputs(target, train=False)
        output = self(item) if tin is None else self(item, tin['xy'], tin['valid'])

        if batch_idx % 100 == 1:
            self.plot_data(target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output, tin)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log('val_center', loss_dict['center_loss'], batch_size=B, sync_dist=True)
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        # ref_T_global = item['ref_T_global']
        # global_T_ref = torch.inverse(ref_T_global)

        # output on bev plane
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
            center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
        )

        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = self.vox_util.Mem2Ref(mem_xyz, self.Y, self.Z, self.X)[..., :2]

        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy_prev = self.vox_util.Mem2Ref(mem_xyz_prev, self.Y, self.Z, self.X)[..., :2]

        # detection
        for frame, grid_gt, xy, score in zip(item['frame'], item['grid_gt'], ref_xy, scores_e):
            frame = int(frame.item())
            valid = score > self.conf_threshold

            self.moda_gt_list.extend([[frame, x.item(), y.item()] for x, y, _ in grid_gt[grid_gt.sum(1) != 0]])
            self.moda_pred_list.extend([[frame, x.item(), y.item()] for x, y in xy[valid]])

        # tracking
        for seq_num, frame, grid_gt, bev_det, bev_prev, score, in (
                zip(item['sequence_num'], item['frame'], item['grid_gt'], ref_xy.cpu(), ref_xy_prev.cpu(),
                    scores_e.cpu())):
            frame = int(frame.item())
            if self.tracker_kwargs['use_reid']:
                feats = self.reid_features(item, ref_xy[0], scores_e[0])
                output_stracks = self.test_tracker.update(bev_det, bev_prev, score, feats=feats)
            else:
                output_stracks = self.test_tracker.update(bev_det, bev_prev, score)

            self.mota_gt_list.extend([[seq_num.item(), frame, i.item(), -1, -1, -1, -1, 1, x.item(),  y.item(), -1]
                                      for x, y, i in grid_gt[grid_gt.sum(1) != 0]])
            self.mota_pred_list.extend([[seq_num.item(), frame, s.track_id, -1, -1, -1, -1, s.score.item()]
                                        + s.xy.tolist() + [-1]
                                        for s in output_stracks])

    def reid_features(self, item, ref_xy, scores):
        """Appearance descriptors of the detections of the first sample in the batch from the working cameras."""
        if self.reid_extractor is None:
            from tracking.reid import BEVReIDExtractor
            self.reid_extractor = BEVReIDExtractor(person_height=0.9 * (self.bounds[5] - self.bounds[4]),
                                                   device=self.device, z_sign=self.z_sign)
        cam_mask = self.get_cam_mask(item)
        keep = scores >= self.tracker_kwargs['low_thres']
        feats = np.full((len(scores), self.reid_extractor.model.feature_dim), np.nan, np.float32)
        if keep.any():
            feats[keep.cpu().numpy()] = self.reid_extractor(
                item['img'][0], item['intrinsic'][0], item['extrinsic'][0], item['ref_T_global'][0],
                ref_xy[keep], None if cam_mask is None else cam_mask[0])[0]
        return feats

    def on_test_epoch_end(self):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'
        os.makedirs(log_dir, exist_ok=True)

        # detection
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        gt_path = osp.join(log_dir, 'moda_gt.txt')
        np.savetxt(pred_path, np.array(self.moda_pred_list), '%f')
        np.savetxt(gt_path, np.array(self.moda_gt_list), '%d')
        recall, precision, moda, modp = modMetricsCalculator(osp.abspath(pred_path), osp.abspath(gt_path))
        if self.test_drop_cams:
            print(f'cameras switched off at test time: {self.test_drop_cams}')
        self.log(f'detect/recall', recall)
        self.log(f'detect/precision', precision)
        self.log(f'detect/moda', moda)
        self.log(f'detect/modp', modp)

        # tracking
        scale = 1 if self.X == 150 else 0.025  # HACK
        pred_path = osp.join(log_dir, 'mota_pred.txt')
        gt_path = osp.join(log_dir, 'mota_gt.txt')
        np.savetxt(pred_path, np.array(self.mota_pred_list), '%f', delimiter=',')
        np.savetxt(gt_path, np.array(self.mota_gt_list), '%f', delimiter=',')
        summary = mot_metrics(osp.abspath(pred_path), osp.abspath(gt_path), scale)
        summary = summary.loc['OVERALL']
        for key, value in summary.to_dict().items():
            if value >= 1 and key[:3] != 'num':
                value /= summary.to_dict()['num_unique_objects']
            value = value * 100 if value < 1 else value
            value = 100 - value if key == 'motp' else value
            self.log(f'track/{key}', value)

    def plot_data(self, target, output, batch_idx=0):
        center_e = output['instance_center']
        center_g = target['center_bev']

        # save plots to tensorboard in eval loop
        writer = getattr(self.logger, 'experiment', None) if self.logger is not None else None
        if not hasattr(writer, 'add_figure'):
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(center_g[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax2.imshow(center_e[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax1.set_title('center_g')
        ax2.set_title('center_e')
        plt.tight_layout()
        writer.add_figure(f'plot/{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }


if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("model.bounds", "data.init_args.bounds")
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")


    cli = MyLightningCLI(WorldTrackModel)
