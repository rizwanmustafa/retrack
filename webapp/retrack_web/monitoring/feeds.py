"""
MultiCameraFeed input adapters (FR-14, class diagram: MultiCameraFeed).

RECORDED replays a WILDTRACK-style folder (one sub-folder per camera, one image per frame index) in increasing frame
order and ends with EndOfSource. LIVE receives frame sets pushed by the camera rig over HTTP (SSD 10.6) and queues
the accepted ones for inference. Both produce the same FrameSet contract.
"""
import json
import os
import queue
import re
import threading
import time
from datetime import timedelta

from .. import scene
from ..timeutil import parse_iso
from .frameset import (BACKLOG_FULL, MISSING_CAMERA, SOURCE_READ_ERROR, CameraPart, EndOfSource, InputFault,
                       build_frame_set, check_order)

LIVE, RECORDED = 'LIVE', 'RECORDED'
DEFAULT_CAMERA_FOLDERS = {cam: f'C{i + 1}' for i, cam in enumerate(scene.CAMERA_ORDER)}
_FRAME_FILE = re.compile(r'^(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)


class FeedClosed(Exception):
    pass


class MultiCameraFeed:
    def __init__(self, feed_id, session_id, mode):
        self.feed_id = feed_id
        self.session_id = session_id
        self.mode = mode
        self.status = 'Created'

    def start(self):
        self.status = 'Open'

    def stop(self):
        self.status = 'Closed'

    def get_next_frame_set(self, timeout):
        """FrameSet, InputFault, EndOfSource, or None when nothing arrived within `timeout` seconds."""
        raise NotImplementedError


# ---------------------------------------------------------------------------- RECORDED
def load_recorded_sources(path):
    """Recorded sources configured for this deployment (docs/RECORDED_SOURCES.md)."""
    if not path or not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    sources = []
    for entry in data.get('sources', []):
        sources.append({
            'id': str(entry['id']),
            'name': str(entry.get('name', entry['id'])),
            'path': entry['path'],
            'cameraFolders': entry.get('cameraFolders', DEFAULT_CAMERA_FOLDERS),
            'frameRate': float(entry['frameRate']),
            'startTime': parse_iso(entry['startTime']),
            'frameRange': entry.get('frameRange'),
            'replayRate': float(entry.get('replayRate', 0)),
        })
    return sources


def find_recorded_source(path, source_id):
    for source in load_recorded_sources(path):
        if source['id'] == source_id:
            return source
    return None


class RecordedFeed(MultiCameraFeed):
    def __init__(self, feed_id, session_id, source, stop_event):
        super().__init__(feed_id, session_id, RECORDED)
        self.source = source
        self.stop_event = stop_event
        self._indices = None
        self._files = None
        self._pos = 0
        self._last_index = None
        self._last_time = None
        self._next_due = None

    def start(self):
        files = {}
        for camera in scene.CAMERA_ORDER:
            folder = os.path.join(self.source['path'], self.source['cameraFolders'][camera])
            files[camera] = {}
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    m = _FRAME_FILE.match(name)
                    if m:
                        files[camera].setdefault(int(m.group(1)), []).append(os.path.join(folder, name))
        indices = sorted(set().union(*[set(f) for f in files.values()]))
        frame_range = self.source.get('frameRange')
        if frame_range:
            indices = [i for i in indices if frame_range[0] <= i <= frame_range[1]]
        if not indices:
            raise FileNotFoundError('the recorded source contains no frames')
        self._files, self._indices = files, indices
        super().start()

    def source_time(self, frame_index):
        return self.source['startTime'] + timedelta(seconds=frame_index / self.source['frameRate'])

    def get_next_frame_set(self, timeout):
        if self.status != 'Open':
            return EndOfSource()
        if self._pos >= len(self._indices):
            return EndOfSource()
        rate = self.source.get('replayRate') or 0
        if rate > 0:
            now = time.monotonic()
            if self._next_due is not None and now < self._next_due:
                self.stop_event.wait(min(timeout, self._next_due - now))
                if time.monotonic() < self._next_due:
                    return None
            self._next_due = time.monotonic() + 1.0 / rate
        index = self._indices[self._pos]
        self._pos += 1
        parts = []
        for camera in scene.CAMERA_ORDER:
            for path in self._files[camera].get(index, []):
                try:
                    with open(path, 'rb') as f:
                        parts.append(CameraPart(camera, f.read()))
                except OSError:
                    return InputFault(SOURCE_READ_ERROR, f'{camera} image for frame {index} cannot be read', index,
                                      [camera])
        try:
            frame_set = build_frame_set(index, self.source_time(index).isoformat(), parts)
            check_order(frame_set, self._last_index, self._last_time)
        except InputFault as fault:
            return fault
        self._last_index, self._last_time = frame_set.frame_index, frame_set.timestamp
        return frame_set


# ---------------------------------------------------------------------------- LIVE
class LiveFeed(MultiCameraFeed):
    def __init__(self, feed_id, session_id, queue_size):
        super().__init__(feed_id, session_id, LIVE)
        self._queue = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._last_index = None
        self._last_time = None

    def submit(self, frame_index, source_time, parts):
        """Validates and queues one frame set. Returns the FrameSet or raises InputFault / FeedClosed."""
        with self._lock:
            if self.status != 'Open':
                raise FeedClosed()
            frame_set = build_frame_set(frame_index, source_time, parts)
            check_order(frame_set, self._last_index, self._last_time)
            try:
                self._queue.put_nowait(frame_set)
            except queue.Full:
                raise InputFault(BACKLOG_FULL, 'inference is behind; the frame set was skipped',
                                 frame_set.frame_index)
            self._last_index, self._last_time = frame_set.frame_index, frame_set.timestamp
            return frame_set

    def get_next_frame_set(self, timeout):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stops accepting input. Frame sets still queued are not processed; returns how many were discarded."""
        with self._lock:
            super().stop()
            discarded = 0
            while True:
                try:
                    self._queue.get_nowait()
                    discarded += 1
                except queue.Empty:
                    return discarded


def is_missing_camera(fault):
    return isinstance(fault, InputFault) and fault.kind == MISSING_CAMERA
