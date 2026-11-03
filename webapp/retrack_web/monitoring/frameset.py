"""
FrameSet contract shared by LIVE and RECORDED input (FR-14 to FR-17, UC-12, SSD 10.6).

A frame set is accepted only with exactly one decodable 1920x1080 image per configured camera, in the configured
camera order, with one declared frame index and source time; anything else raises InputFault and never reaches
inference.
"""
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from .. import scene
from ..timeutil import parse_iso, utcnow

# Input fault kinds
MISSING_CAMERA = 'MISSING_CAMERA'
DUPLICATE_CAMERA = 'DUPLICATE_CAMERA'
UNKNOWN_CAMERA = 'UNKNOWN_CAMERA'
CAMERA_ORDER = 'CAMERA_ORDER'
UNDECODABLE = 'UNDECODABLE'
WRONG_SIZE = 'WRONG_SIZE'
MISMATCHED_METADATA = 'MISMATCHED_METADATA'
BAD_METADATA = 'BAD_METADATA'
OUT_OF_ORDER = 'OUT_OF_ORDER'
DUPLICATE_FRAME = 'DUPLICATE_FRAME'
BACKLOG_FULL = 'BACKLOG_FULL'
SOURCE_READ_ERROR = 'SOURCE_READ_ERROR'

Image.MAX_IMAGE_PIXELS = 4 * scene.IMAGE_WIDTH * scene.IMAGE_HEIGHT


class InputFault(Exception):
    def __init__(self, kind, reason, frame_index=None, cameras=None):
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.frame_index = frame_index
        self.cameras = list(cameras or [])

    def to_dict(self):
        return {'kind': self.kind, 'reason': self.reason, 'frameIndex': self.frame_index, 'cameras': self.cameras}


class EndOfSource:
    """Explicit end of a RECORDED source (FR-14)."""


@dataclass
class FrameSet:
    frame_index: int
    timestamp: datetime
    images: Dict[str, np.ndarray]  # cameraId -> HxWx3 uint8 RGB, in the configured order
    received_at: datetime = field(default_factory=utcnow)
    resolution: tuple = (scene.IMAGE_WIDTH, scene.IMAGE_HEIGHT)

    def validate(self, camera_order=scene.CAMERA_ORDER):
        return tuple(self.images) == tuple(camera_order) and all(
            img.shape == (scene.IMAGE_HEIGHT, scene.IMAGE_WIDTH, 3) for img in self.images.values())


@dataclass
class CameraPart:
    camera_id: str
    data: bytes
    frame_index: Optional[str] = None  # optional per-camera metadata, must match the set's values
    source_time: Optional[str] = None


def parse_frame_index(value):
    try:
        index = int(str(value).strip())
    except (TypeError, ValueError):
        raise InputFault(BAD_METADATA, 'frameIndex must be a whole number')
    if index < 0:
        raise InputFault(BAD_METADATA, 'frameIndex must not be negative')
    return index


def parse_source_time(value, frame_index=None):
    try:
        return parse_iso(value)
    except (TypeError, ValueError) as e:
        raise InputFault(BAD_METADATA, f'sourceTime must be an ISO 8601 time with a UTC offset ({e})', frame_index)


def decode_image(camera, data, frame_index):
    try:
        img = Image.open(io.BytesIO(data))
        size = img.size
        if size != (scene.IMAGE_WIDTH, scene.IMAGE_HEIGHT):
            raise InputFault(WRONG_SIZE, f'{camera} image is {size[0]}x{size[1]}, expected '
                                         f'{scene.IMAGE_WIDTH}x{scene.IMAGE_HEIGHT}', frame_index, [camera])
        rgb = np.asarray(img.convert('RGB'))
    except InputFault:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise InputFault(UNDECODABLE, f'{camera} image cannot be decoded', frame_index, [camera])
    return rgb


def build_frame_set(frame_index, source_time, parts):
    """Validates one supplied frame set. `parts` is a list of CameraPart in the order they were supplied."""
    index = parse_frame_index(frame_index)
    timestamp = parse_source_time(source_time, index)

    ids = [p.camera_id for p in parts]
    unknown = [c for c in ids if c not in scene.CAMERA_ORDER]
    if unknown:
        raise InputFault(UNKNOWN_CAMERA, f'unknown camera {", ".join(sorted(set(unknown)))}', index, unknown)
    duplicates = sorted({c for c in ids if ids.count(c) > 1})
    if duplicates:
        raise InputFault(DUPLICATE_CAMERA, f'more than one image for {", ".join(duplicates)}', index, duplicates)
    missing = [c for c in scene.CAMERA_ORDER if c not in ids]
    if missing:
        raise InputFault(MISSING_CAMERA, f'missing image from {", ".join(missing)}', index, missing)
    if tuple(ids) != scene.CAMERA_ORDER:
        raise InputFault(CAMERA_ORDER, 'images are not in the configured camera order '
                                       f'({", ".join(scene.CAMERA_ORDER)})', index)
    for p in parts:
        if p.frame_index is not None and parse_frame_index(p.frame_index) != index:
            raise InputFault(MISMATCHED_METADATA, f'{p.camera_id} belongs to frame {p.frame_index}, not {index}',
                             index, [p.camera_id])
        if p.source_time is not None and parse_source_time(p.source_time, index) != timestamp:
            raise InputFault(MISMATCHED_METADATA, f'{p.camera_id} source time differs from the frame set',
                             index, [p.camera_id])
    images = {p.camera_id: decode_image(p.camera_id, p.data, index) for p in parts}
    return FrameSet(index, timestamp, images)


def check_order(frame_set, last_index, last_time):
    """FR-16: frame sets are accepted only in increasing source order."""
    if last_index is None:
        return
    if frame_set.frame_index == last_index:
        raise InputFault(DUPLICATE_FRAME, f'frame {frame_set.frame_index} was already received',
                         frame_set.frame_index)
    if frame_set.frame_index < last_index or frame_set.timestamp <= last_time:
        raise InputFault(OUT_OF_ORDER, f'frame {frame_set.frame_index} arrived after frame {last_index}; frame '
                                       'index and source time must increase', frame_set.frame_index)
