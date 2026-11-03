"""The real backend adapter on a host without the inference environment (NFR-14: the web layer still works)."""
import importlib.util

import pytest

from retrack_web.ai import BackendUnavailable
from retrack_web.ai.tracktacular import TrackTacularModel
from retrack_web.config import Config

needs_no_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is not None,
                                    reason='checks behaviour without torch installed')


@needs_no_torch
def test_missing_inference_environment_is_reported_not_hidden(tmp_path):
    model = TrackTacularModel(Config.WORLDTRACK_DIR)
    with pytest.raises(BackendUnavailable, match='inference environment is not installed'):
        model.inspect_checkpoint(str(tmp_path / 'x.ckpt'))
    env = model.describe_environment()
    assert env['backend'] == 'tracktacular' and 'inferenceEnvironment' in env


def test_missing_worldtrack_code_is_reported(tmp_path):
    model = TrackTacularModel(str(tmp_path / 'nowhere'))
    with pytest.raises(BackendUnavailable, match='WorldTrack backend code was not found'):
        model.load_checkpoint(str(tmp_path / 'x.ckpt'))


@needs_no_torch
def test_checkpoint_selection_explains_missing_environment(tmp_path):
    from conftest import PASSWORD, login
    from retrack_web import create_app
    from retrack_web.services.accounts import create_admin
    (tmp_path / 'model.ckpt').write_bytes(b'x')
    app = create_app({'TESTING': True, 'SECRET_KEY': 't', 'WTF_CSRF_ENABLED': False,
                      'MONGO_URI': 'mongomock://localhost', 'MONGO_DB': f'tt_{id(tmp_path)}',
                      'CHECKPOINT_DIR': str(tmp_path), 'RECORDED_SOURCES_FILE': str(tmp_path / 'none.json')})
    with app.app_context():
        create_admin('root', 'root@example.com', PASSWORD)
    client = app.test_client()
    login(client, 'root@example.com')
    client.get('/admin/checkpoints')
    r = client.post('/admin/checkpoints/K-000001/select', follow_redirects=True)
    assert b'inference environment is not installed' in r.data
    assert str(tmp_path).encode() not in r.data
