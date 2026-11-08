"""
ARL A5: average the weights of the last k training epochs (Stochastic Weight Averaging, Izmailov et al. 2018) and
save them as `averaged.ckpt` next to last.ckpt, so that no epoch has to be picked on the test frames.

The weights are sampled at the end of each training epoch, just before validation, i.e. before ModelCheckpoint
writes last.ckpt, so the running average is part of last.ckpt and survives a resume. Buffers that are not floating
point are copied from the last epoch; floating-point buffers (BatchNorm running statistics) are averaged like the
weights, which approximates (does not recompute) the BatchNorm statistics of the averaged model.
"""
import os
import os.path as osp

import torch
from lightning.pytorch.callbacks import Callback


class EpochWeightAveraging(Callback):
    def __init__(self, last_k: int = 5, filename: str = 'averaged.ckpt'):
        self.last_k = last_k
        self.filename = filename
        self.avg = None
        self.n = 0
        self.last_epoch = -1

    def _sample(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if trainer.sanity_checking or epoch == self.last_epoch or epoch < trainer.max_epochs - self.last_k:
            return
        state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}
        if self.avg is None:
            self.avg = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
            self.n = 1
        else:
            self.n += 1
            for k, v in state.items():
                if v.is_floating_point():
                    self.avg[k] += (v.float() - self.avg[k]) / self.n
                else:
                    self.avg[k] = v
        self.last_epoch = epoch
        print(f'[weight averaging] epoch {epoch}: {self.n} epochs averaged')

    def on_validation_start(self, trainer, pl_module):
        if trainer.state.fn == 'fit':
            self._sample(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):  # covers runs without validation
        self._sample(trainer, pl_module)

    def state_dict(self):
        return {'avg': self.avg, 'n': self.n, 'last_epoch': self.last_epoch}

    def load_state_dict(self, state_dict):
        self.avg, self.n, self.last_epoch = state_dict['avg'], state_dict['n'], state_dict['last_epoch']

    def on_fit_end(self, trainer, pl_module):
        if self.avg is None or not trainer.is_global_zero:
            return
        cb = trainer.checkpoint_callback
        dirpath = cb.dirpath if cb is not None and cb.dirpath else osp.join(trainer.log_dir or '.', 'checkpoints')
        os.makedirs(dirpath, exist_ok=True)
        path = osp.join(dirpath, self.filename)
        trainer.save_checkpoint(path)  # same structure (hyper-parameters, loops) as last.ckpt
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        ckpt['state_dict'] = {k: v.to(ckpt['state_dict'][k].dtype) for k, v in self.avg.items()}
        ckpt['weight_averaging'] = {'epochs_averaged': self.n, 'last_k': self.last_k}
        torch.save(ckpt, path + '.part')
        os.replace(path + '.part', path)
        print(f'[weight averaging] {self.n} epochs averaged -> {path}')
