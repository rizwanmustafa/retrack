"""FR-13 to FR-24, UC-11 to UC-13, NFR-07, NFR-11."""
import json
import time

import pytest
from conftest import frameset_files, login, logout, png_bytes, register, start

from retrack_web import scene


def wait_for(predicate, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    raise AssertionError('condition not reached')


def running(db, sid):
    return lambda: db.sessions.find_one({'_id': sid})['status'] == 'Running'


def test_recorded_replay_processes_in_order_skips_invalid_and_ends_at_eof(ready_client, ai, engine, db):
    sid = start(ready_client)
    engine.wait_idle()
    session = db.sessions.find_one({'_id': sid})
    assert ai.analyzed == [0, 5, 20]  # 10 lacks IDIAP2, 15 has a wrongly sized CVLab3
    assert session['status'] == 'Completed' and session['endReason'] == 'END_OF_SOURCE'
    assert session['framesProcessed'] == 3 and session['framesRejected'] == 2
    faults = [l['details'] for l in db.audit_logs.find({'activityType': 'INPUT_ERROR'})]
    assert {(f['frameIndex'], f['kind']) for f in faults} == {(10, 'MISSING_CAMERA'), (15, 'WRONG_SIZE')}
    assert [r['frameIndex'] for r in db.frame_results.find({'sessionId': sid}).sort('frameIndex', 1)] == [0, 5, 20]
    report = db.reports.find_one({'sessionId': sid})
    assert report['status'] == 'Available' and report['reportType'] == 'AUTOMATIC'
    assert db.audit_logs.find_one({'activityType': 'SESSION_STARTED'})['performedBy']['username'] == 'alice'
    assert db.audit_logs.find_one({'activityType': 'SESSION_COMPLETED'}) is not None


def test_zero_detection_frame_is_committed_with_occupancy_zero(ready_client, ai, engine, db):
    ai.script = {0: [(1, 0.0, 0.0, 0.9), (2, 50.0, 60.0, 0.8)], 5: [], 20: [(1, 10.0, 0.0, 0.9)]}
    sid = start(ready_client)
    engine.wait_idle()
    zero = db.frame_results.find_one({'sessionId': sid, 'frameIndex': 5})
    assert zero['occupancy'] == 0 and zero['observations'] == []
    summary = db.sessions.find_one({'_id': sid})['summary']
    assert summary == {'framesProcessed': 3, 'uniqueTracks': 2, 'meanOccupancy': 1.0, 'peakOccupancy': 2}
    track = db.tracks.find_one({'sessionId': sid, 'trackId': 1})
    assert [p['frameIndex'] for p in track['trajectory']] == [0, 20] and track['status'] == 'Ended'


def test_initialization_failure_is_recorded_and_never_running(ready_client, ai, engine, db):
    ai.fail_load = True
    sid = start(ready_client)
    engine.wait_idle()
    session = db.sessions.find_one({'_id': sid})
    assert session['status'] == 'Failed' and session['endReason'] == 'INITIALIZATION_FAILED'
    assert session['runningSince'] is None and 'CUDA out of memory' in session['failure']['message']
    log = db.audit_logs.find_one({'activityType': 'SESSION_START_FAILED', 'outcome': 'FAILURE'})
    assert log['errorFlag'] and log['resource']['id'] == sid
    assert db.reports.count_documents({}) == 0


def test_second_start_on_busy_instance_is_refused_and_reports_stay_available(app, ready_client, engine, db):
    sid = start(ready_client, mode='LIVE')
    wait_for(running(db, sid))
    r = ready_client.post('/user/sessions/start', data={'inputMode': 'RECORDED', 'recordedSource': 'rec'})
    assert r.status_code == 409 and b'busy' in r.data
    assert ready_client.get('/reports').status_code == 200
    assert ready_client.get('/user/sessions').status_code == 200
    ready_client.post(f'/user/sessions/{sid}/stop')
    engine.wait_idle()


def test_stop_while_waiting_finalizes_empty_session_without_report(ready_client, engine, db):
    sid = start(ready_client, mode='LIVE')
    wait_for(running(db, sid))
    live = ready_client.get(f'/api/sessions/{sid}/live').get_json()
    assert live['session']['runtimeStatus'] == 'WAITING_FOR_INPUT'
    r = ready_client.post(f'/api/sessions/{sid}/stop')
    assert r.status_code == 202
    engine.wait_idle()
    session = db.sessions.find_one({'_id': sid})
    assert session['status'] == 'Completed' and session['endReason'] == 'STOP'
    assert session['summary']['framesProcessed'] == 0
    assert db.reports.count_documents({'sessionId': sid}) == 0
    live = ready_client.get(f'/api/sessions/{sid}/live').get_json()
    assert live['session']['summary']['framesProcessed'] == 0 and 'no report' in live['session']['statusMessage']


def test_fatal_failure_keeps_committed_results_and_creates_partial_report(ready_client, ai, engine, db):
    ai.fail_on_frame = 5
    sid = start(ready_client)
    engine.wait_idle()
    session = db.sessions.find_one({'_id': sid})
    assert session['status'] == 'Failed' and session['endReason'] == 'FATAL_ERROR'
    assert session['failure']['stage'] == 'inference' and session['failure']['frameIndex'] == 5
    assert db.frame_results.count_documents({'sessionId': sid}) == 1
    report = db.reports.find_one({'sessionId': sid})
    assert report['status'] == 'Available' and report['coverage']['partial'] is True
    assert report['sessionStatus'] == 'Failed'
    diag = db.audit_logs.find_one({'activityType': 'PROCESSING_ERROR'})
    assert diag['errorFlag'] and 'injected fatal failure' in diag['details']['message']


def test_each_session_resets_tracking(ready_client, ai, engine):
    start(ready_client)
    engine.wait_idle()
    start(ready_client)
    engine.wait_idle()
    assert ai.resets == 2


# ---------------------------------------------------------------- LIVE input contract
def ingest(client, sid, token, frame_index, seconds, **kwargs):
    data, files = frameset_files(frame_index, f'2026-03-01T10:00:{seconds:02d}Z', **kwargs)
    payload = dict(data)
    for name, f in files:
        payload.setdefault(name, []).append(f)
    return client.post(f'/api/ingest/sessions/{sid}/framesets', data=payload, content_type='multipart/form-data',
                       headers={'Authorization': f'Bearer {token}'} if token else {})


@pytest.fixture
def live_session(app, ready_client, engine, db):
    sid = start(ready_client, mode='LIVE')
    wait_for(running(db, sid))
    token = engine.ingest_token(sid, type('P', (), {'id': ready_client.user_id})())
    yield sid, token
    if db.sessions.find_one({'_id': sid})['status'] == 'Running':
        ready_client.post(f'/user/sessions/{sid}/stop')
    engine.wait_idle()


def test_live_ingest_accepts_valid_frame_sets_and_rejects_bad_input(app, ready_client, ai, engine, db, live_session):
    sid, token = live_session
    rig = app.test_client()
    assert ingest(rig, sid, 'wrong', 1, 1).status_code == 401
    r = ingest(rig, sid, token, 1, 1)
    assert r.status_code == 202 and r.get_json() == {'frameSetAccepted': {'frameIndex': 1}}
    wait_for(lambda: db.sessions.find_one({'_id': sid})['framesProcessed'] == 1)

    missing = ingest(rig, sid, token, 2, 2, order=[c for c in scene.CAMERA_ORDER if c != 'IDIAP1'])
    assert missing.status_code == 422 and missing.get_json()['frameSetRejected']['kind'] == 'MISSING_CAMERA'
    wait_for(lambda: ready_client.get(f'/api/sessions/{sid}/live').get_json()['session']['runtimeStatus']
             == 'PAUSED_MISSING_CAMERA')

    cases = [
        (dict(images={'CVLab2': png_bytes((1280, 720))}), 'WRONG_SIZE'),
        (dict(images={'CVLab4': b'not an image'}), 'UNDECODABLE'),
        (dict(order=list(reversed(scene.CAMERA_ORDER))), 'CAMERA_ORDER'),
        (dict(order=list(scene.CAMERA_ORDER) + ['CVLab1']), 'DUPLICATE_CAMERA'),
    ]
    for kwargs, kind in cases:
        r = ingest(rig, sid, token, 3, 3, **kwargs)
        assert r.status_code == 422 and r.get_json()['frameSetRejected']['kind'] == kind, kind
    assert ingest(rig, sid, token, 1, 4).get_json()['frameSetRejected']['kind'] == 'DUPLICATE_FRAME'
    assert ingest(rig, sid, token, 0, 5).get_json()['frameSetRejected']['kind'] == 'OUT_OF_ORDER'

    assert ingest(rig, sid, token, 4, 6).status_code == 202  # a complete set resumes processing
    wait_for(lambda: db.sessions.find_one({'_id': sid})['framesProcessed'] == 2)
    assert ai.analyzed == [1, 4]
    assert db.sessions.find_one({'_id': sid})['framesRejected'] == 7
    assert ready_client.get(f'/api/sessions/{sid}/live').get_json()['session']['runtimeStatus'] != \
        'PAUSED_MISSING_CAMERA'

    ready_client.post(f'/user/sessions/{sid}/stop')
    engine.wait_idle()
    assert ingest(rig, sid, token, 5, 7).status_code == 409  # the feed is released
    assert db.reports.find_one({'sessionId': sid})['summary']['framesProcessed'] == 2


def test_mismatched_per_camera_metadata_is_rejected(app, db, live_session):
    sid, token = live_session
    data, files = frameset_files(1, '2026-03-01T10:00:01Z')
    payload = dict(data, **{'frameIndex.CVLab3': '2'})
    for name, f in files:
        payload[name] = f
    r = app.test_client().post(f'/api/ingest/sessions/{sid}/framesets', data=payload,
                               content_type='multipart/form-data', headers={'Authorization': f'Bearer {token}'})
    assert r.get_json()['frameSetRejected']['kind'] == 'MISMATCHED_METADATA'
    assert db.sessions.find_one({'_id': sid})['framesProcessed'] == 0


def test_logout_does_not_stop_the_running_session(app, ready_client, db, live_session):
    sid, token = live_session
    r = logout(ready_client)
    assert r.status_code == 302
    assert db.sessions.find_one({'_id': sid})['status'] == 'Running'
    assert ingest(app.test_client(), sid, token, 1, 1).status_code == 202
    login(ready_client, 'alice@example.com')


def test_live_result_structure_and_access(app, ready_client, ai, engine, db, live_session):
    sid, token = live_session
    ai.script = {7: [(3, 120.0, 800.0, 0.77)]}
    ingest(app.test_client(), sid, token, 7, 7)
    wait_for(lambda: db.sessions.find_one({'_id': sid})['framesProcessed'] == 1)
    data = ready_client.get(f'/api/sessions/{sid}/live?images=1').get_json()
    assert set(data) >= {'session', 'model', 'frameSet', 'results'}
    assert data['session']['sessionId'] == sid and data['session']['inputMode'] == 'LIVE'
    assert data['model']['checkpointId'] and data['frameSet']['cameraOrder'] == list(scene.CAMERA_ORDER)
    assert data['frameSet']['frameIndex'] == 7 and data['frameSet']['sourceTime'] == '2026-03-01T10:00:07.000Z'
    assert data['results']['units']['groundPosition'] == 'cm'
    assert data['results']['occupancy'] == 1
    track = data['results']['tracks'][0]
    assert track['trackId'] == 3 and track['groundPosition'] == {'x': 120.0, 'y': 800.0}
    text = json.dumps(data)
    for secret in ('password', 'Hash', token, app.config['CHECKPOINT_DIR'], '/Users/', '/tmp'):
        assert secret not in text

    other = app.test_client()
    register(other, 'bob')
    uid = db.users.find_one({'username': 'bob'})['_id']
    db.users.update_one({'_id': uid}, {'$set': {'accountStatus': 'Active'}})
    login(other, 'bob@example.com')
    assert other.get(f'/api/sessions/{sid}/live').status_code == 404
    assert other.get(f'/user/sessions/{sid}/live').status_code == 404
    assert other.post(f'/user/sessions/{sid}/stop').status_code == 404
    assert db.sessions.find_one({'_id': sid})['status'] == 'Running'


def test_interrupted_sessions_are_marked_failed_on_restart(app, ready_client, ai, engine, db):
    from retrack_web.monitoring.engine import MonitoringEngine
    sid = start(ready_client)
    engine.wait_idle()
    db.sessions.update_one({'_id': sid}, {'$set': {'status': 'Running'}})
    fresh = MonitoringEngine(app)
    with app.app_context():
        fresh.recover_interrupted()
    session = db.sessions.find_one({'_id': sid})
    assert session['status'] == 'Failed' and session['endReason'] == 'SERVER_RESTART'
    assert db.frame_results.count_documents({'sessionId': sid}) == 3
