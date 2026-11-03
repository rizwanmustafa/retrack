"""FR-01 to FR-06, UC-01 to UC-03, UC-06, NFR-02, NFR-03."""
from datetime import timedelta

from conftest import PASSWORD, login, logout, make_admin, register

from retrack_web.timeutil import utcnow


def test_registration_creates_pending_end_user_with_salted_hash(client, db):
    r = register(client, 'alice')
    assert r.status_code == 201
    assert b'waiting for Admin approval' in r.data
    user = db.users.find_one({'username': 'alice'})
    assert user['accountStatus'] == 'Pending' and user['role'] == 'END_USER'
    assert PASSWORD not in user['passwordHash'] and user['passwordHash'].startswith(('scrypt:', 'pbkdf2:'))
    # same password, different salt
    register(client, 'bob')
    assert db.users.find_one({'username': 'bob'})['passwordHash'] != user['passwordHash']


def test_duplicate_username_or_email_creates_no_account(client, db):
    register(client, 'alice')
    r1 = register(client, 'ALICE', email='other@example.com')
    r2 = register(client, 'alice2', email='Alice@Example.com')
    assert r1.status_code == r2.status_code == 422
    assert b'already in use' in r1.data and b'already in use' in r2.data
    assert db.users.count_documents({}) == 1


def test_registration_identifies_invalid_field_and_creates_nothing(client, db):
    r = client.post('/register', data={'name': '', 'username': 'x', 'email': 'not-an-email', 'password': 'short'})
    assert r.status_code == 422
    for message in (b'Name is required', b'Username must be', b'valid email', b'Password must be'):
        assert message in r.data
    assert db.users.count_documents({}) == 0


def test_public_registration_cannot_create_or_promote_admin(client, db):
    client.post('/register', data={'name': 'Mallory', 'username': 'mallory', 'email': 'm@example.com',
                                   'password': PASSWORD, 'role': 'ADMIN', 'accountStatus': 'Active'})
    user = db.users.find_one({'username': 'mallory'})
    assert user['role'] == 'END_USER' and user['accountStatus'] == 'Pending'
    assert db.admins.count_documents({}) == 0


def test_pending_and_rejected_accounts_cannot_sign_in(app, client, admin_client, db):
    register(client, 'alice')
    r = login(client, 'alice@example.com')
    assert r.status_code == 401 and b'waiting for Admin approval' in r.data
    assert db.auth_sessions.count_documents({}) == 1  # only the admin's
    user_id = db.users.find_one({'username': 'alice'})['_id']
    admin_client.post(f'/admin/accounts/{user_id}/decision', data={'decision': 'reject'})
    r = login(client, 'alice@example.com')
    assert r.status_code == 401 and b'rejected' in r.data
    assert PASSWORD.encode() not in r.data
    assert db.auth_sessions.count_documents({}) == 1


def test_invalid_credentials_do_not_say_which_part_was_wrong(client, user_client):
    unknown = login(client, 'nobody@example.com')
    wrong = login(client, 'alice@example.com', 'wrong-password')
    assert unknown.status_code == wrong.status_code == 401
    assert b'Invalid email or password' in unknown.data and b'Invalid email or password' in wrong.data
    assert 'Set-Cookie' not in wrong.headers or 'retrack_auth' not in wrong.headers.get('Set-Cookie', '')


def test_login_opens_only_the_role_dashboard(app, user_client, admin_client):
    assert user_client.get('/').headers['Location'].endswith('/user/dashboard')
    assert admin_client.get('/').headers['Location'].endswith('/admin/dashboard')
    assert user_client.get('/user/dashboard').status_code == 200
    assert admin_client.get('/admin/dashboard').status_code == 200


def test_end_user_cannot_call_admin_operations_and_admin_cannot_monitor(user_client, admin_client, db):
    for url in ('/admin/dashboard', '/admin/accounts', '/admin/checkpoints', '/admin/audit', '/admin/sessions'):
        assert user_client.get(url).status_code == 403
    pending = db.users.find_one({'username': 'alice'})['_id']
    assert user_client.post(f'/admin/accounts/{pending}/decision', data={'decision': 'reject'}).status_code == 403
    assert user_client.post('/admin/checkpoints/K-000001/select').status_code == 403
    for url in ('/user/dashboard', '/user/setup', '/user/sessions'):
        assert admin_client.get(url).status_code == 403
    assert admin_client.post('/user/sessions/start', data={'inputMode': 'LIVE'}).status_code == 403


def test_protected_requests_need_an_authenticated_session(client):
    r = client.get('/user/dashboard')
    assert r.status_code == 302 and r.headers['Location'].endswith('/login')
    assert client.get('/api/sessions/S-000001/live').status_code == 401


def test_logout_invalidates_the_server_side_session(app, user_client):
    cookie = user_client.get_cookie('retrack_auth').value
    r = logout(user_client)
    assert r.status_code == 302 and r.headers['Location'].endswith('/login')
    replay = app.test_client()
    replay.set_cookie('retrack_auth', cookie)
    r = replay.get('/user/dashboard')
    assert r.status_code == 302 and r.headers['Location'].endswith('/login')


def test_expired_session_returns_to_login(app, user_client, db):
    db.auth_sessions.update_many({}, {'$set': {'lastSeenAt': utcnow() - timedelta(hours=2)}})
    r = user_client.get('/user/dashboard', follow_redirects=True)
    assert b'Your session has expired' in r.data


def test_account_decision_is_made_once_and_recorded(app, client, admin_client, db):
    register(client, 'alice')
    user_id = db.users.find_one({'username': 'alice'})['_id']
    admin_client.post(f'/admin/accounts/{user_id}/decision', data={'decision': 'approve'})
    r = admin_client.post(f'/admin/accounts/{user_id}/decision', data={'decision': 'reject'}, follow_redirects=True)
    assert b'already decided' in r.data
    user = db.users.find_one({'_id': user_id})
    assert user['accountStatus'] == 'Active'
    assert user['decision']['decision'] == 'Approved' and user['decision']['adminUsername'] == 'root'
    approved = db.audit_logs.find_one({'activityType': 'ACCOUNT_APPROVED'})
    assert approved['performedBy']['username'] == 'root' and approved['resource']['id'] == user_id
    assert approved['timestamp'] is not None
    assert db.audit_logs.find_one({'activityType': 'ACCOUNT_DECISION_REFUSED'})['outcome'] == 'REFUSED'


def test_admin_is_provisioned_outside_registration(app):
    import pytest
    from retrack_web.errors import AppError
    make_admin(app)
    with pytest.raises(AppError):
        make_admin(app, 'root2', 'ROOT@example.com')
