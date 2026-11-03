"""
Model checkpoint administration (FR-12, UC-10).

Trained checkpoints are prepared offline (SRS 2.4) and placed in RETRACK_CHECKPOINT_DIR. The Admin lists them and
selects one; the system accepts it only if the backend confirms compatibility, and refuses any change while the model
instance is processing a session. The file path is stored for the backend but never shown or returned.
"""
import hashlib
import os

from flask import current_app

from .. import audit
from ..ai import BackendUnavailable
from ..db import get_db, next_id
from ..errors import AppError
from ..security import ADMIN, Principal
from ..timeutil import utcnow


def _checkpoint_dir(app=None):
    return (app or current_app).config['CHECKPOINT_DIR']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scan():
    """Registers new *.ckpt files and marks files that disappeared as unavailable."""
    db = get_db()
    root = _checkpoint_dir()
    found = {}
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.endswith('.ckpt'):
                    path = os.path.join(dirpath, name)
                    found[os.path.relpath(path, root)] = path
    known = {doc['fileName']: doc for doc in db.checkpoints.find()}
    now = utcnow()
    for rel, path in found.items():
        stat = os.stat(path)
        if rel in known:
            doc = known[rel]
            changed = doc.get('sizeBytes') != stat.st_size or doc.get('modifiedTime') != stat.st_mtime
            update = {'available': True, 'filePath': path}
            if changed:
                # A replaced file is a different checkpoint: it must be inspected again before use
                update.update({'sizeBytes': stat.st_size, 'modifiedTime': stat.st_mtime, 'sha256': None,
                               'lastInspection': None})
            db.checkpoints.update_one({'_id': doc['_id']}, {'$set': update})
        else:
            db.checkpoints.insert_one({
                '_id': next_id('checkpoints'), 'fileName': rel, 'filePath': path, 'modelName': None,
                'version': None, 'sha256': None, 'sizeBytes': stat.st_size, 'modifiedTime': stat.st_mtime,
                'available': True, 'isActive': False, 'lastInspection': None, 'registeredAt': now,
            })
    for rel, doc in known.items():
        if rel not in found and doc.get('available'):
            db.checkpoints.update_one({'_id': doc['_id']}, {'$set': {'available': False}})


def list_checkpoints():
    return list(get_db().checkpoints.find().sort('fileName', 1))


def active_checkpoint(app=None):
    return get_db(app).checkpoints.find_one({'isActive': True})


def select(checkpoint_id, admin: Principal):
    """UC-10: activates a compatible checkpoint for new sessions, or refuses with the reason."""
    assert admin.role == ADMIN
    db = get_db()
    doc = db.checkpoints.find_one({'_id': checkpoint_id})
    if doc is None:
        raise AppError('This checkpoint does not exist.', status=404, code='not_found')

    def refuse(reason, code, status=409, problems=None):
        audit.record(audit.CHECKPOINT_CHANGE_REFUSED, audit.REFUSED, 'ModelCheckpoint', checkpoint_id,
                     {'reason': reason, 'problems': problems or []})
        raise AppError(reason, status=status, code=code, problems=[{'reason': p} for p in problems or []])

    engine = current_app.extensions['retrack_engine']
    if engine.is_busy():
        refuse('The model instance is processing an active session. The checkpoint can be changed after that '
               'session ends.', 'model_busy')
    path = doc['filePath']
    if not doc.get('available') or not os.path.isfile(path):
        db.checkpoints.update_one({'_id': checkpoint_id}, {'$set': {'available': False}})
        refuse('This checkpoint file is unavailable. The previous checkpoint stays active.', 'unavailable')
    try:
        info = current_app.extensions['retrack_ai'].inspect_checkpoint(path)
    except BackendUnavailable as e:
        refuse(f'The checkpoint cannot be checked: {e} The previous checkpoint stays active.', 'backend_unavailable',
               status=503)
    inspection = {'compatible': info.compatible, 'problems': info.problems, 'details': info.details,
                  'inspectedAt': utcnow()}
    db.checkpoints.update_one({'_id': checkpoint_id}, {'$set': {'lastInspection': inspection,
                                                                'modelName': info.model_name or None,
                                                                'version': info.version or None}})
    if not info.compatible:
        refuse('This checkpoint is not compatible with the configured scene and backend. The previous checkpoint '
               'stays active.', 'incompatible', status=422, problems=info.problems)
    digest = sha256_file(path)
    previous = active_checkpoint()
    with engine.exclusive():  # no session can start while the active checkpoint changes
        if engine.is_busy():  # a session may have started while the file was being checked
            refuse('The model instance is processing an active session. The checkpoint can be changed after that '
                   'session ends.', 'model_busy')
        db.checkpoints.update_many({'isActive': True}, {'$set': {'isActive': False}})
        db.checkpoints.update_one({'_id': checkpoint_id}, {'$set': {'isActive': True, 'sha256': digest,
                                                                    'activatedAt': utcnow()}})
    audit.record(audit.CHECKPOINT_CHANGED, audit.SUCCESS, 'ModelCheckpoint', checkpoint_id,
                 {'previous': previous['_id'] if previous else None, 'fileName': doc['fileName'],
                  'modelName': info.model_name, 'version': info.version, 'sha256': digest})
    return db.checkpoints.find_one({'_id': checkpoint_id})
