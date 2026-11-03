"""
TrackTacular backend for AIModel (SRS 1, FR-18, FR-19).

Wraps the research code in WorldTrack/ (TrackTacular + the Retrack tracker). Only this module imports it, so the
web application runs, and reports which prerequisite is missing, on a host without the inference environment.
Preprocessing matches WorldTrack's test-time path (datasets/pedestrian_dataset.py): each 1920x1080 image is resized
to 1280x720 with nearest-neighbour sampling, without cropping, and the intrinsics are scaled to match. Detection
threshold, tracker and re-ID settings come from the checkpoint's own hyper-parameters (NFR-12).
"""
import inspect
import os
import platform
import subprocess
import sys
import threading

import numpy as np
from PIL import Image

from .. import scene
from ..calibration import rodrigues
from ..textutil import safe_message
from .interface import AIModel, AnalysisResult, BackendUnavailable, CheckpointInfo, ModelError, Observation

FINAL_DIM = (720, 1280)  # H, W
SUPPORTED_MODELS = ('segnet', 'mvdet', 'liftnet', 'bevformer')


class TrackTacularModel(AIModel):
    backend_name = 'tracktacular'

    def __init__(self, worldtrack_dir, device='auto'):
        self.worldtrack_dir = os.path.abspath(worldtrack_dir)
        self.device_preference = device
        self._mods = None
        self._model = None
        self._path = None
        self._mtime = None
        self._tracker = None
        self._frame = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ environment
    def _modules(self):
        if self._mods is not None:
            return self._mods
        if not os.path.isdir(self.worldtrack_dir):
            raise BackendUnavailable('The WorldTrack backend code was not found (RETRACK_WORLDTRACK_DIR).')
        if self.worldtrack_dir not in sys.path:
            sys.path.insert(0, self.worldtrack_dir)
        try:
            import torch
            from world_track import WorldTrackModel
            from tracking.multitracker import BaseTrack, build_tracker
            from utils import decode, geom
        except ImportError as e:
            raise BackendUnavailable(f'The inference environment is not installed (missing Python module '
                                     f'"{e.name}"). Install webapp/requirements-model.txt.')
        self._mods = dict(torch=torch, WorldTrackModel=WorldTrackModel, BaseTrack=BaseTrack,
                          build_tracker=build_tracker, decode=decode, geom=geom)
        return self._mods

    def _device(self):
        torch = self._modules()['torch']
        if self.device_preference == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device_preference == 'cuda' and not torch.cuda.is_available():
            raise BackendUnavailable('CUDA was requested but no CUDA device is available.')
        return self.device_preference

    def describe_environment(self):
        env = {'backend': self.backend_name, 'python': platform.python_version(), 'platform': platform.platform(),
               'preprocessing': f'resize 1920x1080 -> {FINAL_DIM[1]}x{FINAL_DIM[0]} (nearest), no crop'}
        try:
            out = subprocess.run(['git', '-C', self.worldtrack_dir, 'rev-parse', 'HEAD'], capture_output=True,
                                 text=True, timeout=5)
            env['repositoryRevision'] = out.stdout.strip() or 'unknown'
            dirty = subprocess.run(['git', '-C', self.worldtrack_dir, 'status', '--porcelain', '--', '.'],
                                   capture_output=True, text=True, timeout=5)
            env['repositoryModified'] = 'yes' if dirty.stdout.strip() else 'no'
        except (OSError, subprocess.SubprocessError):
            env['repositoryRevision'] = 'unknown'
        try:
            torch = self._modules()['torch']
            env['torch'] = torch.__version__
            env['device'] = self._device()
            if env['device'] == 'cuda':
                env['cudaDevice'] = torch.cuda.get_device_name(0)
                env['cuda'] = str(torch.version.cuda)
        except BackendUnavailable as e:
            env['inferenceEnvironment'] = str(e)
        return env

    # ------------------------------------------------------------------ checkpoints
    def _read_checkpoint(self, path):
        mods = self._modules()
        try:
            ckpt = mods['torch'].load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ModelError(f'The checkpoint file cannot be read ({type(e).__name__}).')
        if not isinstance(ckpt, dict) or 'state_dict' not in ckpt or 'hyper_parameters' not in ckpt:
            raise ModelError('The file is not a WorldTrack Lightning checkpoint (state_dict and hyper_parameters '
                             'are required by the output contract).')
        accepted = inspect.signature(mods['WorldTrackModel'].__init__).parameters
        hparams = {k: v for k, v in ckpt['hyper_parameters'].items() if k in accepted}
        return ckpt, hparams

    def inspect_checkpoint(self, path):
        try:
            ckpt, hp = self._read_checkpoint(path)
        except ModelError as e:
            return CheckpointInfo(False, [str(e)])
        problems = []
        model_name = str(hp.get('model_name', 'segnet'))
        if model_name not in SUPPORTED_MODELS:
            problems.append(f'Backend model "{model_name}" is not supported.')
        if hp.get('num_cameras') != scene.MODEL_NUM_CAMERAS:
            problems.append(f'Camera count is {hp.get("num_cameras")}; the rig has {scene.MODEL_NUM_CAMERAS}.')
        if tuple(hp.get('resolution', ())) != scene.MODEL_RESOLUTION or \
                tuple(hp.get('bounds', ())) != scene.MODEL_BOUNDS:
            problems.append('Scene configuration (BEV resolution and bounds) does not match the configured '
                            'WILDTRACK scene.')
        if not problems:
            try:
                model = self._modules()['WorldTrackModel'](**hp)
                missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
                if missing or unexpected:
                    problems.append(f'Weights do not match the backend ({len(missing)} missing, '
                                    f'{len(unexpected)} unexpected tensors).')
            except Exception as e:
                problems.append(f'The backend cannot build this model ({safe_message(e)}).')
        details = {
            'Model': model_name,
            'Encoder': str(hp.get('encoder_name', '')),
            'Fusion': str(hp.get('fusion', 'concat')),
            'Tracker': str(hp.get('tracker', 'baseline')),
            'Detection threshold': str(hp.get('conf_threshold', '')),
            'Re-identification': str(hp.get('use_reid', False)),
            'Temporal cache': str(hp.get('use_temporal_cache', '')),
            'Lightning': str(ckpt.get('pytorch-lightning_version', 'unknown')),
        }
        version = f'epoch {ckpt.get("epoch", "?")}, step {ckpt.get("global_step", "?")}'
        return CheckpointInfo(not problems, problems, model_name, version, details)

    @property
    def loaded_checkpoint(self):
        return self._path

    def load_checkpoint(self, path):
        with self._lock:
            self._modules()  # report a missing inference environment before anything else
            mtime = os.path.getmtime(path) if os.path.exists(path) else None
            if mtime is None:
                raise ModelError('The selected checkpoint file is no longer available.')
            if self._model is not None and self._path == path and self._mtime == mtime:
                return
            ckpt, hp = self._read_checkpoint(path)
            mods = self._modules()
            try:
                model = mods['WorldTrackModel'](**hp)
                missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
                if missing or unexpected:
                    raise ModelError('The checkpoint weights do not match the backend.')
                model.test_drop_cams = []  # the web system always processes all seven cameras
                self._model = model.to(self._device()).eval()
            except (ModelError, BackendUnavailable):
                raise
            except Exception as e:
                raise ModelError(f'Loading the checkpoint failed ({safe_message(e)}).')
            self._path, self._mtime = path, mtime

    def reset_tracking(self):
        if self._model is None:
            raise ModelError('No checkpoint is loaded.')
        mods = self._modules()
        self._tracker = mods['build_tracker'](**self._model.tracker_kwargs)
        mods['BaseTrack']._count = 0  # track IDs restart at 1 for each session
        self._model.temporal_cache = None
        self._model.temporal_cache_frames = -2 * mods['torch'].ones(self._model.max_cache,
                                                                     dtype=mods['torch'].long)
        self._frame = 0

    # ------------------------------------------------------------------ inference
    def _inputs(self, frame_set, calibration):
        mods = self._modules()
        torch, geom = mods['torch'], mods['geom']
        H, W = FINAL_DIM
        sx, sy = W / scene.IMAGE_WIDTH, H / scene.IMAGE_HEIGHT
        cams = {c['cameraId']: c for c in calibration['cameras']}
        imgs, Ks, exts = [], [], []
        for camera in scene.CAMERA_ORDER:
            img = Image.fromarray(frame_set.images[camera]).resize((W, H), Image.NEAREST)
            imgs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255)
            Ks.append(np.asarray(cams[camera]['intrinsicMatrix'], dtype=np.float32))
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = rodrigues(cams[camera]['rotation']).astype(np.float32)
            ext[:3, 3] = np.asarray(cams[camera]['translationCm'], dtype=np.float32)
            exts.append(ext)
        K = torch.tensor(np.stack(Ks))
        intrinsic = geom.scale_intrinsics(geom.merge_intrinsics(*geom.split_intrinsics(K)), sx, sy)
        m = np.asarray(scene.WORLDCOORD_FROM_WORLDGRID, dtype=np.float32)
        world_from_grid = torch.eye(4)
        world_from_grid[:2, :2] = torch.tensor(m[:2, :2])
        world_from_grid[:2, 3] = torch.tensor(m[:2, 2])
        device = self._model.device
        return {
            'img': torch.stack(imgs)[None].to(device),
            'intrinsic': intrinsic[None].to(device),
            'extrinsic': torch.tensor(np.stack(exts))[None].to(device),
            'ref_T_global': torch.inverse(world_from_grid)[None].to(device),
            'frame': torch.tensor([self._frame]),
        }

    def analyze(self, frame_set, calibration):
        if self._model is None or self._tracker is None:
            raise ModelError('The model is not initialized for this session.')
        mods = self._modules()
        torch, decode = mods['torch'], mods['decode']
        model = self._model
        try:
            with torch.no_grad():
                item = self._inputs(frame_set, calibration)
                output = model(item)
                xy_e, xy_prev_e, scores_e, _, _, _ = decode.decoder(
                    output['instance_center'].sigmoid(), output['instance_offset'], output['instance_size'],
                    rz_e=output['instance_rot'], K=model.max_detections)
                mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
                ref_xy = model.vox_util.Mem2Ref(mem_xyz, model.Y, model.Z, model.X)[..., :2]
                mem_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
                ref_xy_prev = model.vox_util.Mem2Ref(mem_prev, model.Y, model.Z, model.X)[..., :2]
                if model.tracker_kwargs['use_reid']:
                    feats = model.reid_features(item, ref_xy[0], scores_e[0])
                    stracks = self._tracker.update(ref_xy[0].cpu(), ref_xy_prev[0].cpu(), scores_e[0].cpu(),
                                                   feats=feats)
                else:
                    stracks = self._tracker.update(ref_xy[0].cpu(), ref_xy_prev[0].cpu(), scores_e[0].cpu())
        except Exception as e:
            raise ModelError(f'Inference failed ({safe_message(e)}).')
        finally:
            self._frame += 1
        observations = []
        for s in stracks:
            gx, gy = (float(v) for v in np.asarray(s.xy, dtype=np.float64)[:2])
            x_cm, y_cm = scene.worldgrid_to_world_cm(gx, gy)
            observations.append(Observation(int(s.track_id), x_cm, y_cm, float(s.score)))
        return AnalysisResult(observations)
