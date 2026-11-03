"""
Test fixtures. MongoDB is replaced by mongomock and the TrackTacular backend by FakeAIModel, so the tests cover the
web system (NFR-13: these are web-system tests, not model evaluation).
"""
import io
import json
import os
import re
import zipfile

import pytest
from PIL import Image

from retrack_web import create_app, scene
from retrack_web.ai import AIModel, AnalysisResult, CheckpointInfo, ModelError, Observation
from retrack_web.db import get_db

PASSWORD = 'correct-horse-battery'
CAMERA_FOLDERS = {cam: f'C{i + 1}' for i, cam in enumerate(scene.CAMERA_ORDER)}


class FakeAIModel(AIModel):
    """Deterministic stand-in for the backend: frame index -> observed tracks."""
    backend_name = 'fake'

    def __init__(self):
        self.path = None
        self.fail_load = False
        self.fail_on_frame = None
        self.script = {}
        self.analyzed = []
        self.resets = 0
        self.delay = 0

    def inspect_checkpoint(self, path):
        if 'incompatible' in os.path.basename(path):
            return CheckpointInfo(False, ['Camera count is 4; the rig has 7.'], 'segnet', 'epoch 1, step 1')
        return CheckpointInfo(True, [], 'segnet', 'epoch 30, step 5400', {'Model': 'segnet'})

    def load_checkpoint(self, path):
        if self.fail_load:
            raise ModelError('Loading the checkpoint failed (RuntimeError: CUDA out of memory).')
        self.path = path

    def reset_tracking(self):
        self.resets += 1

    def analyze(self, frame_set, calibration):
        if self.delay:
            import time
            time.sleep(self.delay)
        assert frame_set.validate()
        if self.fail_on_frame is not None and frame_set.frame_index == self.fail_on_frame:
            raise ModelError('Inference failed (RuntimeError: injected fatal failure).')
        self.analyzed.append(frame_set.frame_index)
        tracks = self.script.get(frame_set.frame_index, [(1, 100.0, 500.0, 0.9)])
        return AnalysisResult([Observation(*t) for t in tracks])

    def describe_environment(self):
        return {'backend': 'fake', 'repositoryRevision': 'test'}

    @property
    def loaded_checkpoint(self):
        return self.path


def png_bytes(size=(scene.IMAGE_WIDTH, scene.IMAGE_HEIGHT), color=(90, 120, 150)):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, format='PNG')
    return buf.getvalue()


INTRINSIC = """<?xml version="1.0"?>
<opencv_storage>
<camera_matrix type_id="opencv-matrix">
  <rows>3</rows>
  <cols>3</cols>
  <dt>d</dt>
  <data>{data}</data></camera_matrix>
</opencv_storage>
"""
EXTRINSIC = """<?xml version="1.0"?>
<opencv_storage>
<rvec>{rvec}</rvec>
<tvec>{tvec}</tvec>
</opencv_storage>
"""


def calibration_zip(manifest=None, skip=(), overrides=None, prefix=''):
    """A calibration bundle. `skip` lists member names to leave out; `overrides` replaces member contents."""
    manifest = {'coordinateConvention': scene.COORDINATE_CONVENTION, 'translationUnit': 'cm'} \
        if manifest is None else manifest
    members = {'manifest.json': json.dumps(manifest)}
    for i, cam in enumerate(scene.CAMERA_ORDER):
        members[f'intrinsic_zero/intr_{cam}.xml'] = INTRINSIC.format(
            data='1.7e+03 0. 9.6e+02 0. 1.7e+03 5.4e+02 0. 0. 1.')
        members[f'extrinsic/extr_{cam}.xml'] = EXTRINSIC.format(rvec=f'1.7 {0.1 * i} -0.3',
                                                                 tvec=f'-500 {40 + i} 1000')
    members.update(overrides or {})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, content in members.items():
            if name not in skip:
                z.writestr(prefix + name, content)
    return buf.getvalue()


@pytest.fixture
def recorded_dir(tmp_path):
    """Recorded source with frames 0, 5, 10, 15; frame 10 lacks IDIAP2, frame 15 has a wrongly sized CVLab3."""
    root = tmp_path / 'Image_subsets'
    good, small = png_bytes(), png_bytes((640, 360))
    for cam, folder in CAMERA_FOLDERS.items():
        (root / folder).mkdir(parents=True)
        for frame in (0, 5, 10, 15, 20):
            if frame == 10 and cam == 'IDIAP2':
                continue
            data = small if (frame == 15 and cam == 'CVLab3') else good
            (root / folder / f'{frame:08d}.png').write_bytes(data)
    return root


@pytest.fixture
def app(tmp_path, recorded_dir):
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    (ckpt_dir / 'retrack_segnet.ckpt').write_bytes(b'checkpoint-a')
    (ckpt_dir / 'incompatible_mvdet.ckpt').write_bytes(b'checkpoint-b')
    sources = tmp_path / 'recorded_sources.json'
    sources.write_text(json.dumps({'sources': [
        {'id': 'rec', 'name': 'Test recording', 'path': str(recorded_dir), 'frameRate': 10,
         'startTime': '2026-01-01T00:00:00Z'},
        {'id': 'empty', 'name': 'Empty recording', 'path': str(tmp_path), 'frameRate': 10,
         'startTime': '2026-01-01T00:00:00Z'},
    ]}))
    app = create_app({
        'TESTING': True,
        'SECRET_KEY': 'test',
        'WTF_CSRF_ENABLED': False,
        'MONGO_URI': 'mongomock://localhost',
        'MONGO_DB': f'retrack_test_{os.getpid()}_{id(tmp_path)}',
        'CHECKPOINT_DIR': str(ckpt_dir),
        'RECORDED_SOURCES_FILE': str(sources),
        'AI_MODEL': FakeAIModel(),
    })
    yield app
    app.extensions['retrack_engine'].wait_idle()


@pytest.fixture
def ai(app):
    return app.extensions['retrack_ai']


@pytest.fixture
def engine(app):
    return app.extensions['retrack_engine']


@pytest.fixture
def db(app):
    return get_db(app)


@pytest.fixture
def client(app):
    return app.test_client()


# ------------------------------------------------------------------ helpers
def register(client, username='alice', email=None, name='Alice Operator', password=PASSWORD):
    return client.post('/register', data={'name': name, 'username': username,
                                          'email': email or f'{username}@example.com', 'password': password})


def login(client, email, password=PASSWORD):
    return client.post('/login', data={'email': email, 'password': password})


def logout(client):
    return client.post('/logout')


def make_admin(app, username='root', email='root@example.com'):
    from retrack_web.services.accounts import create_admin
    with app.app_context():
        return create_admin(username, email, PASSWORD)


@pytest.fixture
def admin_client(app):
    make_admin(app)
    c = app.test_client()
    assert login(c, 'root@example.com').status_code == 302
    return c


@pytest.fixture
def user_client(app, admin_client):
    """An approved, signed-in End User 'alice'."""
    c = app.test_client()
    register(c, 'alice')
    user_id = get_db(app).users.find_one({'username': 'alice'})['_id']
    admin_client.post(f'/admin/accounts/{user_id}/decision', data={'decision': 'approve'})
    assert login(c, 'alice@example.com').status_code == 302
    c.user_id = user_id
    return c


@pytest.fixture
def ready_client(app, user_client, admin_client):
    """End User with a selected calibration and an active compatible checkpoint."""
    admin_client.get('/admin/checkpoints')
    ckpt = get_db(app).checkpoints.find_one({'fileName': 'retrack_segnet.ckpt'})
    admin_client.post(f'/admin/checkpoints/{ckpt["_id"]}/select')
    r = user_client.post('/user/setup/calibration/upload',
                         data={'bundle': (io.BytesIO(calibration_zip()), 'calib.zip')},
                         content_type='multipart/form-data')
    assert r.status_code == 302, r.data
    return user_client


def start(client, mode='RECORDED', source='rec'):
    r = client.post('/user/sessions/start', data={'inputMode': mode, 'recordedSource': source})
    assert r.status_code == 302, r.data
    return re.search(r'/sessions/(S-\d+)/live', r.headers['Location']).group(1)


def frameset_files(frame_index, source_time, images=None, order=None):
    order = order or list(scene.CAMERA_ORDER)
    data = {'frameIndex': str(frame_index), 'sourceTime': source_time}
    files = [(cam, (io.BytesIO(images.get(cam, png_bytes()) if images else png_bytes()), f'{cam}.png'))
             for cam in order]
    return data, files
