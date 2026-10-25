import os
import os.path as osp
from typing import List, Optional

import torch
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from models import Segnet, MVDet, Liftnet, Bevformernet
from models.loss import FocalLoss, compute_rot_loss
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
            cam_dropout_prob=0.0,  # probability that a training sample has cameras switched off
            cam_dropout_max=3,  # at most this many cameras are switched off per sample
            test_drop_cams: Optional[List[int]] = None,  # cameras switched off at test time
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
        if model_name == 'segnet':
            self.model = Segnet(self.Y, self.Z, self.X, num_cameras=num_cameras, feat2d_dim=feat2d_dim,
                                encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign,
                                **fusion_kwargs)
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

    def forward(self, item):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        ref_T_global: (B,4,4)
        vox_util: vox util object
        """
        prev_bev = self.load_cache(item['frame'].cpu())
        cam_mask = self.get_cam_mask(item)
        kwargs = {} if cam_mask is None else {'cam_mask': cam_mask}

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
            self.store_cache(item['frame'].cpu(), output['bev_raw'].clone().detach())

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
        idx = []
        for frame in frames:
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            if i.nelement() == 1:
                idx.append(i.item())
        if len(idx) != len(frames):
            return None
        else:
            return self.temporal_cache[idx]

    def store_cache(self, frames, bev_feat):
        if self.temporal_cache is None:
            shape = list(bev_feat.shape)
            shape[0] = self.max_cache
            self.temporal_cache = torch.zeros(shape, device=bev_feat.device, dtype=bev_feat.dtype)

        for frame, feat in zip(frames, bev_feat):
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            # Choose unfilled cache slot
            if i.nelement() == 0:
                i = (self.temporal_cache_frames == -2).nonzero(as_tuple=True)[0]
            # Choose random cache slot
            if i.nelement() == 0:
                i = torch.randint(self.max_cache, (1, 1))

            self.temporal_cache[i[0]] = feat
            self.temporal_cache_frames[i[0]] = frame

    def loss(self, target, output):
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
        output = self(item)

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        if batch_idx % 100 == 1:
            self.plot_data(target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output)

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
