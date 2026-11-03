"""
JSON endpoints: live tracking results (FR-22), session commands and the LIVE ingest contract (SSD 10.6).

Ingest: POST /api/ingest/sessions/<sessionId>/framesets with `Authorization: Bearer <ingest token>` and a
multipart body: fields `frameIndex` and `sourceTime` (ISO 8601 with offset) and one file per camera named by its
camera ID, in the configured order. See docs/LIVE_INGEST.md.
"""
from flask import Blueprint, current_app, g, jsonify, request

from ..db import get_db
from ..errors import AppError
from ..monitoring.frameset import CameraPart
from ..security import END_USER, login_required
from ..timeutil import iso

bp = Blueprint('api', __name__, url_prefix='/api')


def _engine():
    return current_app.extensions['retrack_engine']


@bp.route('/sessions/<session_id>/live')
@login_required(END_USER)
def live_result(session_id):
    return jsonify(_engine().snapshot(session_id, g.principal, include_images=request.args.get('images') == '1'))


@bp.route('/sessions/<session_id>/stop', methods=['POST'])
@login_required(END_USER)
def stop(session_id):
    session = _engine().stop(session_id, g.principal)
    return jsonify({'sessionId': session_id, 'status': session['status'], 'runtimeStatus': session['runtimeStatus'],
                    'stopRequested': True}), 202


@bp.route('/sessions/<session_id>/status')
@login_required(END_USER)
def status(session_id):
    session = get_db().sessions.find_one({'_id': session_id, 'userId': g.principal.id})
    if session is None:
        raise AppError('This session does not exist or you may not access it.', status=404, code='not_found')
    return jsonify({'sessionId': session_id, 'status': session['status'], 'runtimeStatus': session['runtimeStatus'],
                    'framesProcessed': session['framesProcessed'], 'endTime': iso(session['endTime'])})


@bp.route('/ingest/sessions/<session_id>/framesets', methods=['POST'])
def ingest_frame_set(session_id):
    auth = request.headers.get('Authorization', '')
    token = auth[7:].strip() if auth.lower().startswith('bearer ') else None
    form = request.form
    parts = [CameraPart(name, f.read(), form.get(f'frameIndex.{name}'), form.get(f'sourceTime.{name}'))
             for name, f in request.files.items(multi=True)]
    outcome, value = _engine().ingest(session_id, token, form.get('frameIndex'), form.get('sourceTime'), parts)
    if outcome == 'accepted':
        return jsonify({'frameSetAccepted': {'frameIndex': value.frame_index}}), 202
    return jsonify({'frameSetRejected': {'frameIndex': value.frame_index, 'kind': value.kind,
                                         'reason': value.reason, 'cameras': value.cameras}}), 422
