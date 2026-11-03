"""FR-07 to FR-09, UC-04, UC-05, UC-07."""
from conftest import register


def submit(client, fields=('name',), reason='Name changed after marriage'):
    return client.post('/user/profile/requests', data={'fields': list(fields), 'reason': reason})


def pending_id(db):
    return db.profile_requests.find_one({'status': 'Pending'})['_id']


def test_request_needs_fields_and_reason(user_client, db):
    assert submit(user_client, fields=()).status_code == 400
    assert submit(user_client, reason=' ').status_code == 400
    assert submit(user_client, fields=('role', 'accountStatus')).status_code == 400  # not requestable
    assert db.profile_requests.count_documents({}) == 0
    assert submit(user_client).status_code == 302
    req = db.profile_requests.find_one()
    assert req['status'] == 'Pending' and req['fieldsToChange'] == ['name']
    assert b'Pending' in user_client.get('/user/profile/requests').data


def test_profile_is_read_only_without_approval(user_client, db):
    page = user_client.get('/user/profile').data
    assert b'read-only' in page
    r = user_client.post('/user/profile', data={'name': 'New Name'})
    assert r.status_code == 403
    assert db.users.find_one({'username': 'alice'})['name'] == 'Alice Operator'


def test_admin_decision_records_permitted_fields_admin_and_time(user_client, admin_client, db):
    submit(user_client, fields=('name', 'email'))
    rid = pending_id(db)
    page = admin_client.get(f'/admin/profile-requests/{rid}').data
    assert b'Alice Operator' in page and b'Name changed after marriage' in page
    admin_client.post(f'/admin/profile-requests/{rid}/decision',
                      data={'decision': 'approve', 'permittedFields': ['name']})
    req = db.profile_requests.find_one({'_id': rid})
    assert req['status'] == 'Approved' and req['permittedFields'] == ['name']
    assert req['reviewedBy']['adminUsername'] == 'root' and req['decidedAt'] is not None
    assert db.audit_logs.find_one({'activityType': 'PROFILE_REQUEST_APPROVED'})['details']['permittedFields'] == ['name']
    r = admin_client.post(f'/admin/profile-requests/{rid}/decision', data={'decision': 'reject'}, follow_redirects=True)
    assert b'already decided' in r.data
    assert db.profile_requests.find_one({'_id': rid})['status'] == 'Approved'


def approve(user_client, admin_client, db, fields=('name',)):
    submit(user_client, fields=fields)
    rid = pending_id(db)
    admin_client.post(f'/admin/profile-requests/{rid}/decision',
                      data={'decision': 'approve', 'permittedFields': list(fields)})
    return rid


def test_approved_fields_update_once_and_request_becomes_used(user_client, admin_client, db):
    rid = approve(user_client, admin_client, db)
    r = user_client.post('/user/profile', data={'name': 'Alice Khan'})
    assert r.status_code == 302
    assert db.users.find_one({'username': 'alice'})['name'] == 'Alice Khan'
    assert db.profile_requests.find_one({'_id': rid})['status'] == 'Used'
    assert user_client.post('/user/profile', data={'name': 'Again'}).status_code == 403
    assert db.users.find_one({'username': 'alice'})['name'] == 'Alice Khan'


def test_unauthorized_field_change_is_refused_and_keeps_approval(user_client, admin_client, db):
    rid = approve(user_client, admin_client, db)
    r = user_client.post('/user/profile', data={'name': 'Alice Khan', 'email': 'evil@example.com'})
    assert r.status_code == 403
    user = db.users.find_one({'username': 'alice'})
    assert user['name'] == 'Alice Operator' and user['email'] == 'alice@example.com'
    assert db.profile_requests.find_one({'_id': rid})['status'] == 'Approved'


def test_role_and_status_cannot_be_edited(user_client, admin_client, db):
    rid = approve(user_client, admin_client, db)
    r = user_client.post('/user/profile', data={'name': 'Alice Khan', 'role': 'ADMIN', 'accountStatus': 'Active'})
    assert r.status_code == 403
    assert db.users.find_one({'username': 'alice'})['role'] == 'END_USER'
    assert db.profile_requests.find_one({'_id': rid})['status'] == 'Approved'


def test_invalid_or_duplicate_values_leave_profile_and_approval(app, user_client, admin_client, db):
    register(app.test_client(), 'bob')
    rid = approve(user_client, admin_client, db, fields=('username',))
    assert user_client.post('/user/profile', data={'username': 'x'}).status_code == 400
    r = user_client.post('/user/profile', data={'username': 'BOB'})
    assert r.status_code == 400 and b'already in use' in r.data
    assert db.users.find_one({'_id': user_client.user_id})['username'] == 'alice'
    assert db.profile_requests.find_one({'_id': rid})['status'] == 'Approved'
    assert user_client.post('/user/profile', data={'username': 'alice_k'}).status_code == 302
    assert db.users.find_one({'_id': user_client.user_id})['usernameLower'] == 'alice_k'


def test_rejected_request_is_visible_to_the_end_user(user_client, admin_client, db):
    submit(user_client)
    rid = pending_id(db)
    admin_client.post(f'/admin/profile-requests/{rid}/decision', data={'decision': 'reject'})
    assert db.profile_requests.find_one({'_id': rid})['status'] == 'Rejected'
    assert b'Rejected' in user_client.get('/user/profile/requests').data
    assert db.audit_logs.find_one({'activityType': 'PROFILE_REQUEST_REJECTED'}) is not None
