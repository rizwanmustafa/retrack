"""FR-10 to FR-12, UC-08 to UC-10."""
import io
import os

from conftest import calibration_zip, start

from retrack_web import scene
from retrack_web.calibration import validate_bundle


def upload(client, data, name='bundle.zip'):
    return client.post('/user/setup/calibration/upload', data={'bundle': (io.BytesIO(data), name)},
                       content_type='multipart/form-data')


def test_valid_bundle_is_stored_and_selected(user_client, db):
    r = upload(user_client, calibration_zip(prefix='calibrations/'))
    assert r.status_code == 302
    calib = db.calibrations.find_one()
    assert calib['validationStatus'] == 'Valid' and calib['cameraOrder'] == list(scene.CAMERA_ORDER)
    assert [c['cameraId'] for c in calib['cameras']] == list(scene.CAMERA_ORDER)
    assert db.session_setups.find_one({'_id': user_client.user_id})['calibrationId'] == calib['_id']


def test_missing_file_identifies_camera_and_file_and_prevents_start(app, user_client, admin_client, db):
    admin_client.get('/admin/checkpoints')
    ckpt = db.checkpoints.find_one({'fileName': 'retrack_segnet.ckpt'})
    admin_client.post(f'/admin/checkpoints/{ckpt["_id"]}/select')
    r = upload(user_client, calibration_zip(skip=('extrinsic/extr_IDIAP2.xml',)))
    assert r.status_code == 422
    assert b'IDIAP2' in r.data and b'extrinsic/extr_IDIAP2.xml' in r.data and b'missing' in r.data
    assert db.calibrations.count_documents({}) == 0
    page = user_client.get('/user/setup').data
    assert b'id="startButton"' in page and b'disabled' in page.split(b'id="startButton"')[1][:200]
    r = user_client.post('/user/sessions/start', data={'inputMode': 'LIVE'})
    assert r.status_code == 422 and db.sessions.count_documents({}) == 0


def problems_for(**kwargs):
    _, _, problems = validate_bundle(calibration_zip(**kwargs))
    return problems


def test_malformed_matrix_and_non_finite_values_are_reported():
    bad_dims = problems_for(overrides={'intrinsic_zero/intr_CVLab3.xml':
                                       '<opencv_storage><camera_matrix><rows>2</rows><cols>3</cols>'
                                       '<data>1 2 3 4 5 6</data></camera_matrix></opencv_storage>'})
    assert bad_dims[0]['camera'] == 'CVLab3' and bad_dims[0]['field'] == 'camera_matrix'
    nan = problems_for(overrides={'extrinsic/extr_IDIAP1.xml':
                                  '<opencv_storage><rvec>1 nan 0</rvec><tvec>1 2 3</tvec></opencv_storage>'})
    assert nan == [{'camera': 'IDIAP1', 'file': 'extrinsic/extr_IDIAP1.xml', 'field': 'rvec',
                    'reason': 'contains a non-finite value'}]
    short = problems_for(overrides={'extrinsic/extr_CVLab1.xml':
                                    '<opencv_storage><rvec>1 2 3</rvec><tvec>1 2</tvec></opencv_storage>'})
    assert short[0]['field'] == 'tvec' and 'must have 3 values' in short[0]['reason']


def test_convention_and_unit_must_be_declared_and_consistent():
    wrong_unit = problems_for(manifest={'coordinateConvention': scene.COORDINATE_CONVENTION, 'translationUnit': 'mm'})
    assert wrong_unit[0]['field'] == 'translationUnit'
    mixed = problems_for(manifest={'coordinateConvention': scene.COORDINATE_CONVENTION, 'translationUnit': 'cm',
                                   'cameras': {'IDIAP3': {'translationUnit': 'm'}}})
    assert mixed[0]['camera'] == 'IDIAP3'
    assert problems_for(skip=('manifest.json',))[0]['reason'] == 'missing manifest.json'
    assert validate_bundle(b'not a zip')[2][0]['reason'] == 'the bundle is not a valid .zip file'


def test_session_retains_calibration_reference_and_camera_order(ready_client, engine, db):
    sid = start(ready_client)
    engine.wait_idle()
    session = db.sessions.find_one({'_id': sid})
    assert session['calibrationId'] == db.calibrations.find_one()['_id']
    assert session['cameraOrder'] == list(scene.CAMERA_ORDER)


def test_stored_calibration_can_be_selected_by_another_user(app, ready_client, admin_client, db):
    from conftest import login, register
    other = app.test_client()
    register(other, 'bob')
    uid = db.users.find_one({'username': 'bob'})['_id']
    admin_client.post(f'/admin/accounts/{uid}/decision', data={'decision': 'approve'})
    login(other, 'bob@example.com')
    cid = db.calibrations.find_one()['_id']
    assert other.post('/user/setup/calibration/select', data={'calibrationId': cid}).status_code == 302
    assert db.session_setups.find_one({'_id': uid})['calibrationId'] == cid


# ---------------------------------------------------------------- checkpoints
def test_incompatible_checkpoint_is_refused_and_previous_stays_active(app, admin_client, db):
    admin_client.get('/admin/checkpoints')
    good = db.checkpoints.find_one({'fileName': 'retrack_segnet.ckpt'})
    bad = db.checkpoints.find_one({'fileName': 'incompatible_mvdet.ckpt'})
    admin_client.post(f'/admin/checkpoints/{good["_id"]}/select')
    r = admin_client.post(f'/admin/checkpoints/{bad["_id"]}/select', follow_redirects=True)
    assert b'not compatible' in r.data and b'Camera count is 4' in r.data
    assert db.checkpoints.find_one({'isActive': True})['_id'] == good['_id']
    changed = db.audit_logs.find_one({'activityType': 'CHECKPOINT_CHANGED'})
    assert changed['performedBy']['username'] == 'root' and changed['details']['sha256']
    assert db.audit_logs.find_one({'activityType': 'CHECKPOINT_CHANGE_REFUSED'})['outcome'] == 'REFUSED'


def test_unavailable_checkpoint_is_refused(app, admin_client, db):
    admin_client.get('/admin/checkpoints')
    good = db.checkpoints.find_one({'fileName': 'retrack_segnet.ckpt'})
    os.remove(good['filePath'])
    r = admin_client.post(f'/admin/checkpoints/{good["_id"]}/select', follow_redirects=True)
    assert b'unavailable' in r.data
    assert db.checkpoints.find_one({'isActive': True}) is None


def test_checkpoint_change_refused_while_a_session_is_active(app, ready_client, admin_client, engine, db):
    sid = start(ready_client, mode='LIVE')
    bad = db.checkpoints.find_one({'fileName': 'incompatible_mvdet.ckpt'})
    good = db.checkpoints.find_one({'fileName': 'retrack_segnet.ckpt'})
    db.checkpoints.update_one({'_id': good['_id']}, {'$set': {'isActive': False}})
    r = admin_client.post(f'/admin/checkpoints/{bad["_id"]}/select', follow_redirects=True)
    assert b'processing an active session' in r.data
    ready_client.post(f'/user/sessions/{sid}/stop')
    engine.wait_idle()


def test_checkpoint_paths_are_not_shown(app, admin_client, db):
    page = admin_client.get('/admin/checkpoints').data.decode()
    assert app.config['CHECKPOINT_DIR'] not in page
    assert 'retrack_segnet.ckpt' in page
