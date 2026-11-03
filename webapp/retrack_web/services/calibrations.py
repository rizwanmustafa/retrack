"""Calibration upload, selection and the End User's session setup (FR-10, FR-11, UC-08, UC-09)."""
import math

from .. import scene
from ..calibration import bundle_digest, validate_bundle
from ..db import get_db, next_id
from ..errors import AppError
from ..security import Principal
from ..timeutil import utcnow

VALID = 'Valid'


def ensure_rig(app=None):
    """The configured camera rig document (JSON tree: CameraRig)."""
    db = get_db(app)
    db.camera_rigs.update_one(
        {'_id': scene.RIG_ID},
        {'$setOnInsert': {'name': scene.RIG_NAME, 'cameraCount': scene.CAMERA_COUNT,
                          'cameraOrder': list(scene.CAMERA_ORDER), 'status': 'Configured', 'calibrationID': None}},
        upsert=True)


def upload_bundle(user: Principal, filename, file_bytes):
    """Validates an uploaded bundle; a valid bundle is stored and becomes the user's selected calibration."""
    if not file_bytes:
        raise AppError('Choose a calibration bundle (.zip) to upload.', field='bundle')
    cameras, manifest, problems = validate_bundle(file_bytes)
    if problems:
        raise AppError('The calibration bundle is not valid. Start is unavailable until a valid calibration is '
                       'selected.', code='invalid_calibration', problems=problems)
    doc = {
        '_id': next_id('calibrations'),
        'rigID': scene.RIG_ID,
        'calibrationFile': (filename or 'bundle.zip')[:200],
        'sha256': bundle_digest(file_bytes),
        'validationStatus': VALID,
        'coordinateConvention': manifest['coordinateConvention'],
        'translationUnit': manifest['translationUnit'],
        'cameraOrder': list(scene.CAMERA_ORDER),
        'cameras': cameras,
        'uploadedBy': {'userId': user.id, 'username': user.username},
        'uploadedAt': utcnow(),
    }
    get_db().calibrations.insert_one(doc)
    select_calibration(user, doc['_id'])
    return doc


def stored_calibrations():
    return list(get_db().calibrations.find({'validationStatus': VALID, 'rigID': scene.RIG_ID},
                                           {'cameras': 0}).sort('uploadedAt', -1))


def get_calibration(calibration_id):
    return get_db().calibrations.find_one({'_id': calibration_id})


def revalidate(calibration):
    """UC-09 on a stored bundle: the stored values must still pass the structural checks."""
    problems = []
    if calibration.get('coordinateConvention') != scene.COORDINATE_CONVENTION:
        problems.append({'camera': None, 'field': 'coordinateConvention', 'reason': 'inconsistent convention'})
    if calibration.get('translationUnit') != scene.TRANSLATION_UNIT:
        problems.append({'camera': None, 'field': 'translationUnit', 'reason': 'translation unit is not cm'})
    cams = calibration.get('cameras') or []
    if [c.get('cameraId') for c in cams] != list(scene.CAMERA_ORDER):
        problems.append({'camera': None, 'field': 'cameras', 'reason': 'camera set or order does not match the rig'})
    for c in cams:
        K, r, t = c.get('intrinsicMatrix'), c.get('rotation'), c.get('translationCm')
        ok = (isinstance(K, list) and len(K) == 3 and all(isinstance(row, list) and len(row) == 3 for row in K)
              and isinstance(r, list) and len(r) == 3 and isinstance(t, list) and len(t) == 3)
        if not ok:
            problems.append({'camera': c.get('cameraId'), 'field': 'matrices', 'reason': 'malformed dimensions'})
            continue
        values = [v for row in K for v in row] + r + t
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            problems.append({'camera': c.get('cameraId'), 'field': 'matrices', 'reason': 'non-finite value'})
    return problems


def select_calibration(user: Principal, calibration_id):
    """UC-08 step 3: attaches a validated calibration reference and the camera order to the user's setup."""
    calibration = get_calibration(calibration_id)
    if calibration is None or calibration.get('rigID') != scene.RIG_ID:
        raise AppError('This calibration does not exist for the configured rig.', status=404, code='not_found',
                       field='calibrationId')
    problems = revalidate(calibration)
    if problems:
        raise AppError('The selected calibration is not valid.', code='invalid_calibration', problems=problems)
    get_db().session_setups.update_one(
        {'_id': user.id},
        {'$set': {'calibrationId': calibration['_id'], 'cameraOrder': calibration['cameraOrder'],
                  'updatedAt': utcnow()}},
        upsert=True)
    return calibration


def current_setup(user_id):
    setup = get_db().session_setups.find_one({'_id': user_id})
    if not setup:
        return None, None
    calibration = get_calibration(setup['calibrationId'])
    if calibration is None or revalidate(calibration):
        return setup, None
    return setup, calibration
