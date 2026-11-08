import os
import json
from operator import itemgetter

import torch
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw

from utils import geom, basic, vox

# Both WILDTRACK and MultiviewX annotate on a 2.5 cm ground grid (reference coordinates are grid cells).
GRID_CELL_M = 0.025
# ARL A1: occluders are painted with the ImageNet mean colour (3DROM's constant value Omega is not given numerically)
OCCLUDER_RGB = (124, 116, 104)
SCENE_AUG_MODES = ('flip', 'flip_affine')


class PedestrianDataset(VisionDataset):
    def __init__(
            self,
            base,
            is_train=True,
            resolution=(160, 4, 250),
            bounds=(-500, 500, -320, 320, 0, 2),
            final_dim: tuple = (720, 1280),
            resize_lim: list = (0.8, 1.2),
            frame_range=None,
            occlusion_prob: float = 0.0,
            occlusion_num: int = 25,
            occlusion_width_ratio: float = 0.4,
            occlusion_min_dist: float = 1.0,
            scene_aug_prob: float = 0.0,
            scene_aug_mode: str = 'flip',
            scene_aug_shift: float = 0.5,
            scene_aug_rot: float = 5.0,
    ):
        """
        frame_range: (start, end) fractions of the sequence; default (0, 0.9) for training, (0.9, 1) otherwise.
                     ARL's clean protocol trains on (0, 0.81) and tunes on (0.81, 0.9) (frames 0-323 / 324-359).
        ARL A1, 3D random occlusion (3DROM, Alg. 1), training only:
            occlusion_prob: probability that a sample gets occluders; occlusion_num: occluders per sample;
            occlusion_width_ratio: occluder width / height in the image; occlusion_min_dist: metres between them.
        ARL A2, scene-level BEV augmentation (MVAug scene augmentation), training only:
            scene_aug_prob: probability per sample; scene_aug_mode: 'flip' (one of the three BEV mirror flips)
            or 'flip_affine' (flip + rotation of up to scene_aug_rot degrees + shift of up to scene_aug_shift m).
        """
        super().__init__(base.root)
        assert scene_aug_mode in SCENE_AUG_MODES, scene_aug_mode
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        # img_shape and worldgrid_shape is the original shape matching the annotations in dataset
        # MultiviewX: [1080, 1920], [640, 1000] Wildtrack: [1080, 1920], [480, 1440]
        self.img_shape = base.img_shape
        self.worldgrid_shape = base.worldgrid_shape
        self.is_train = is_train
        self.bounds = bounds
        self.resolution = resolution
        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim}
        self.kernel_size = 1.5
        self.max_objects = 60
        self.img_downsample = 4

        self.Y, self.Z, self.X = self.resolution
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3])

        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)

        self.occlusion_prob = occlusion_prob
        self.occlusion_num = occlusion_num
        self.occlusion_width_ratio = occlusion_width_ratio
        self.occlusion_min_dist = occlusion_min_dist
        # same person-height convention as tracking/reid.py (180 cm on WILDTRACK)
        self.occluder_height = 0.9 * (self.bounds[5] - self.bounds[4])
        self.scene_aug_prob = scene_aug_prob
        self.scene_aug_mode = scene_aug_mode
        self.scene_aug_shift = scene_aug_shift
        self.scene_aug_rot = scene_aug_rot

        if frame_range is None:
            frame_range = (0.0, 0.9) if self.is_train else (0.9, 1.0)
        self.frame_range_frac = tuple(frame_range)
        frame_range = range(int(self.num_frame * frame_range[0]), int(self.num_frame * frame_range[1]))

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.prepare_gt()

        self.calibration = {}
        self.setup()

    def setup(self):
        intrinsic = torch.tensor(np.stack(self.base.intrinsic_matrices, axis=0), dtype=torch.float32)  # S,3,3
        intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()  # S,4,4
        self.calibration['intrinsic'] = intrinsic
        self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        self.calibration['extrinsic'][:, :3] = torch.tensor(
            np.stack(self.base.extrinsic_matrices, axis=0), dtype=torch.float32)

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]

                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pedestrian['personID'])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                self.world_gt[frame] = (torch.tensor(world_pts, dtype=torch.float32),
                                        torch.tensor(world_pids, dtype=torch.float32))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (torch.tensor(img_bboxs[cam]), torch.tensor(img_pids[cam]))

    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre):
        center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        offset = torch.zeros((4, self.Y, self.X), dtype=torch.float32)
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)

        prev_pts = dict(zip(pids_pre.int().tolist(), mem_pts_prev[0]))

        for pts, pid in zip(mem_pts[0], pids):
            ct = pts[:2]
            ct_int = ct.int()

            if ct_int[0] < 0 or ct_int[0] >= self.X or ct_int[1] < 0 or ct_int[1] >= self.Y:
                continue

            for c in center:
                basic.draw_umich_gaussian(c, ct_int, self.kernel_size)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:2, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid

            if pid in pids_pre:
                t_off = prev_pts[pid.int().item()][:2] - ct_int
                if t_off.abs().max() > 15:
                    continue
                offset[2:, ct_int[1], ct_int[0]] = t_off

        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
        center = torch.zeros((3, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        center_pts = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2), axis=1)
        center_pts = torch.tensor(center_pts, dtype=torch.float32)
        size_pts = np.stack(((-xmin + xmax), (-ymin + ymax)), axis=1)
        size_pts = torch.tensor(size_pts, dtype=torch.float32)
        foot_pts = np.stack(((xmin + xmax) / 2, ymin), axis=1)
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)
        head_pts = np.stack(((xmin + xmax) / 2, ymax), axis=1)
        head_pts = torch.tensor(head_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            for idx, pt in enumerate((foot_pts[pt_idx], )):  # , center_pts[pt_idx], head_pts[pt_idx])):
                if pt[0] < 0 or pt[0] >= W or pt[1] < 0 or pt[1] >= H:
                    continue
                basic.draw_umich_gaussian(center[idx], pt.int(), self.kernel_size)

            ct_int = foot_pts[pt_idx].int()
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = foot_pts[pt_idx] - ct_int
            size[:, ct_int[1], ct_int[0]] = wh
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, person_ids, valid_mask

    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    # ------------------------------------------------------------ ARL A1: 3D random occlusion (3DROM Alg. 1)
    def sample_occluders(self):
        """Ground positions (reference / world-grid coordinates) of the occluders of one frame, or None."""
        if not self.is_train or self.occlusion_prob <= 0 or np.random.rand() >= self.occlusion_prob:
            return None
        xmin, xmax, ymin, ymax = self.bounds[:4]
        min_d = self.occlusion_min_dist / GRID_CELL_M
        pts = []
        for _ in range(self.occlusion_num * 50):
            if len(pts) == self.occlusion_num:
                break
            p = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
            if all(np.hypot(*(p - q)) >= min_d for q in pts):
                pts.append(p)
        return np.array(pts) if pts else None

    def paint_occluders(self, img, cam, occ):
        """Projects every occluder (a vertical cylinder of person height) into camera `cam` of the original-size
        image as a filled rectangle of height h and width occlusion_width_ratio * h. Labels are not changed."""
        K = np.asarray(self.base.intrinsic_matrices[cam], dtype=np.float64)
        Rt = np.asarray(self.base.extrinsic_matrices[cam], dtype=np.float64)
        world_T_grid = np.asarray(self.base.worldcoord_from_worldgrid_mat, dtype=np.float64)
        world_xy = (world_T_grid @ np.c_[occ, np.ones(len(occ))].T)[:2].T
        foot = np.c_[world_xy, np.zeros(len(occ)), np.ones(len(occ))]
        head = foot.copy()
        head[:, 2] = self.occluder_height
        z_sign = -1 if 'multiviewx' in self.base.__class__.__name__.lower() else 1  # as the model configs
        draw = None
        W, H = img.size
        for f_pt, h_pt in zip(foot, head):
            pf, ph = K @ Rt @ f_pt, K @ Rt @ h_pt
            if z_sign * pf[2] <= 0 or z_sign * ph[2] <= 0:
                continue  # behind the camera
            uf, uh = pf[:2] / pf[2], ph[:2] / ph[2]
            y1, y2 = min(uf[1], uh[1]), max(uf[1], uh[1])
            w = self.occlusion_width_ratio * (y2 - y1)
            cx = (uf[0] + uh[0]) / 2
            x1, x2 = cx - w / 2, cx + w / 2
            if not np.all(np.isfinite([x1, y1, x2, y2])):
                continue  # point (almost) on the camera plane
            # clip to the image; skip occluders that are outside it or degenerate after clipping
            x1, y1 = int(round(max(x1, 0))), int(round(max(y1, 0)))
            x2, y2 = int(round(min(x2, W - 1))), int(round(min(y2, H - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            if draw is None:
                img = img.copy()
                draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], fill=OCCLUDER_RGB)
        return img

    def get_image_data(self, frame, cameras):
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []
        occluders = self.sample_occluders()  # one set per frame, shared by all views
        for cam in cameras:
            img = Image.open(self.img_fpaths[cam][frame]).convert('RGB')
            if occluders is not None:
                img = self.paint_occluders(img, cam, occluders)
            W, H = img.size

            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)  # 4,4
            img = basic.img_transform(img, resize_dims, crop)

            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids = self.imgs_gt[frame][cam]
            center_img, offset_img, size_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, sx, sy, crop)

            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return torch.stack(imgs), torch.stack(intrins), torch.stack(extrins), torch.stack(centers), torch.stack(
            offsets), torch.stack(sizes), torch.stack(pids), torch.stack(valids)

    # ------------------------------------------------------------ ARL A2: scene-level BEV augmentation (MVAug)
    def sample_scene_transform(self):
        """4x4 transform of the reference (BEV) frame that keeps the fixed BEV rectangle valid: one of the three
        mirror flips about the grid centre, optionally with a small rotation and shift. Points that leave the grid
        are dropped from the BEV target by get_bev_gt."""
        xmin, xmax, ymin, ymax = self.bounds[:4]
        c = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        flips = [(-1, 1), (1, -1), (-1, -1)]
        A = np.diag(flips[np.random.randint(len(flips))]).astype(np.float64)
        t = np.zeros(2)
        if self.scene_aug_mode == 'flip_affine':
            th = np.deg2rad(np.random.uniform(-self.scene_aug_rot, self.scene_aug_rot))
            A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ A
            t = np.random.uniform(-self.scene_aug_shift, self.scene_aug_shift, 2) / GRID_CELL_M
        T = np.eye(4)
        T[:2, :2] = A
        T[:2, 3] = c + t - A @ c
        return torch.tensor(T, dtype=torch.float32)

    def __len__(self):
        return len(self.world_gt.keys())

    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]
        pre_frame = list(self.world_gt.keys())[max(index - 1, 0)]
        cameras = list(range(self.num_cam))

        # images
        imgs, intrins, extrins, centers_img, offsets_img, sizes_img, pids_img, valids_img \
            = self.get_image_data(frame, cameras)

        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(self.base.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

        worldgrid_pts_org, world_pids = self.world_gt[frame]
        worldgrid_pts_pre, world_pid_pre = self.world_gt[pre_frame]

        worldgrid_pts = torch.cat((worldgrid_pts_org, torch.zeros_like(worldgrid_pts_org[:, 0:1])), dim=1).unsqueeze(0)
        worldgrid_pts_pre = torch.cat((worldgrid_pts_pre, torch.zeros_like(worldgrid_pts_pre[:, 0:1])),
                                      dim=1).unsqueeze(0)

        augment = torch.eye(4)
        if self.is_train:
            Rz = torch.eye(3)
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            off = 0.25
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            if self.scene_aug_prob > 0 and np.random.rand() < self.scene_aug_prob:
                augment = torch.matmul(self.sample_scene_transform(), augment)
            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts)
            # ARL fix: the previous-frame GT (target of the tracking offset) lives in the same reference frame
            worldgrid_pts_pre = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts_pre)
        # 2D part of the transform (raw reference -> augmented reference), used to align the temporal cache
        scene_T = augment[[0, 1, 3]][:, [0, 1, 3]].clone()

        mem_pts = self.vox_util.Ref2Mem(worldgrid_pts, self.Y, self.Z, self.X)
        mem_pts_pre = self.vox_util.Ref2Mem(worldgrid_pts_pre, self.Y, self.Z, self.X)
        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(mem_pts, mem_pts_pre,  world_pids, world_pid_pre)

        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts_org
        grid_gt[:worldgrid_pts.shape[1], 2] = world_pids

        item = {
            'img': imgs,  # S,3,H,W
            'intrinsic': intrins,  # S,4,4
            'extrinsic': extrins,  # S,4,4
            'ref_T_global': worldgrid_T_worldcoord,  # 4,4
            'frame': frame // self.base.frame_step,
            'sequence_num': int(0),
            'grid_gt': grid_gt,
            'scene_T': scene_T,  # 3,3
        }

        target = {
            # bev
            'valid_bev': valid_bev,  # 1,Y,X
            'center_bev': center_bev,  # 1,Y,X
            'offset_bev': offset_bev,  # 2,Y,X
            'pid_bev': pid_bev,  # 1,Y,X
            # img
            'center_img': centers_img,  # S,1,H/8,W/8
            'offset_img': offsets_img,  # S,2,H/8,W/8
            'size_img': sizes_img,  # S,2,H/8,W/8
            'valid_img': valids_img,  # S,1,H/8,W/8
            'pid_img': pids_img  # S,1,H/8,W/8
        }

        return item, target
