"""NFR-02 to NFR-04: credential handling, server-side enforcement, CSRF, HTTPS and safe errors."""
import os

import pytest
from conftest import FakeAIModel, PASSWORD, login, register

from retrack_web import create_app
from retrack_web.textutil import safe_message


def make(tmp_path, **extra):
    config = {'TESTING': True, 'SECRET_KEY': 'test', 'MONGO_URI': 'mongomock://localhost',
              'MONGO_DB': f'sec_{id(tmp_path)}_{len(extra)}', 'CHECKPOINT_DIR': str(tmp_path),
              'RECORDED_SOURCES_FILE': str(tmp_path / 'none.json'), 'AI_MODEL': FakeAIModel()}
    config.update(extra)
    return create_app(config)


def test_forms_require_a_csrf_token(tmp_path):
    app = make(tmp_path)
    r = app.test_client().post('/register', data={'name': 'A', 'username': 'abc', 'email': 'a@example.com',
                                                  'password': PASSWORD})
    assert r.status_code == 400 and b'expired' in r.data


def test_https_is_required_outside_development(tmp_path):
    app = make(tmp_path, REQUIRE_HTTPS=True, WTF_CSRF_ENABLED=False)
    client = app.test_client()
    r = client.get('/login')
    assert r.status_code == 301 and r.headers['Location'].startswith('https://')
    assert client.post('/login', data={'email': 'a@example.com', 'password': 'x'}).status_code == 403
    assert client.get('/login', base_url='https://localhost').status_code == 200


def test_production_requires_secret_key(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError):
        make(tmp_path, ENVIRONMENT='production', SECRET_KEY=None)


def test_unexpected_errors_do_not_expose_details(tmp_path):
    app = make(tmp_path, WTF_CSRF_ENABLED=False)

    @app.route('/boom')
    def boom():
        raise RuntimeError(f'secret at {os.getcwd()}/config.py')

    app.config['PROPAGATE_EXCEPTIONS'] = False
    r = app.test_client().get('/boom')
    assert r.status_code == 500
    assert b'secret' not in r.data and os.getcwd().encode() not in r.data and b'Traceback' not in r.data


def test_safe_message_strips_paths():
    msg = safe_message(FileNotFoundError('/home/retrack/checkpoints/model.ckpt not found'))
    assert '/home/retrack' not in msg and 'FileNotFoundError' in msg


def test_auth_cookie_is_http_only_and_token_is_hashed(app, client, db):
    from conftest import make_admin
    make_admin(app)
    r = login(client, 'root@example.com')
    cookie = r.headers['Set-Cookie']
    assert 'HttpOnly' in cookie and 'SameSite=Lax' in cookie
    token = client.get_cookie('retrack_auth').value
    stored = db.auth_sessions.find_one()
    assert token not in str(stored) and len(stored['tokenHash']) == 64


def test_hidden_controls_cannot_bypass_approval(app, client, admin_client, db):
    register(client, 'alice')
    # a Pending user cannot reach protected operations by calling them directly
    assert client.post('/user/sessions/start', data={'inputMode': 'LIVE'}).status_code == 302
    assert client.post('/user/profile', data={'name': 'x'}).status_code == 302
    assert db.sessions.count_documents({}) == 0
