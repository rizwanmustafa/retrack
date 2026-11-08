import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.sampler import TemporalSampler


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 2,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
            split: str = 'standard',
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
        split: 'standard' = train on the first 90 % of the frames, validate on the last 10 % (the test frames);
               'val' = ARL clean protocol: train on the first 81 % (WILDTRACK frames 0-323) and validate on
               81-90 % (frames 324-359), so that thresholds and tracker parameters can be chosen without the
               test frames. The test split is the last 10 % in both cases.
        occlusion_* / scene_aug_*: ARL A1 / A2 training augmentations (see PedestrianDataset).
        """
        super().__init__()
        assert split in ('standard', 'val'), split
        self.split = split
        self.train_aug = dict(occlusion_prob=occlusion_prob, occlusion_num=occlusion_num,
                              occlusion_width_ratio=occlusion_width_ratio, occlusion_min_dist=occlusion_min_dist,
                              scene_aug_prob=scene_aug_prob, scene_aug_mode=scene_aug_mode,
                              scene_aug_shift=scene_aug_shift, scene_aug_rot=scene_aug_rot)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        train_range, val_range = ((0.0, 0.9), (0.9, 1.0)) if self.split == 'standard' else ((0.0, 0.81), (0.81, 0.9))
        if stage == 'fit':
            self.data_train = PedestrianDataset(
                base,
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
                frame_range=train_range,
                **self.train_aug,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
                frame_range=val_range,
            )
        if stage == 'test':
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds
            )
        if stage == 'predict':
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_train, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_val, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
