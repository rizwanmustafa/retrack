"""End User pages: dashboard, session setup, monitoring, sessions and profile (UC-04, UC-05, UC-08, UC-11, UC-13)."""
from flask import Blueprint, current_app, flash, g, redirect, render_template, request, url_for

from ..db import get_db
from ..errors import AppError
from ..monitoring.engine import ACTIVE_STATUSES
from ..monitoring.feeds import LIVE, RECORDED, load_recorded_sources
from ..reports.service import eligibility_problem
from ..security import END_USER, login_required
from ..services import calibrations, profiles
from ..services.accounts import PROFILE_FIELDS
from ..services.checkpoints import active_checkpoint

bp = Blueprint('enduser', __name__, url_prefix='/user')


def _engine():
    return current_app.extensions['retrack_engine']


def _own_session(session_id):
    session = get_db().sessions.find_one({'_id': session_id, 'userId': g.principal.id})
    if session is None:
        raise AppError('This session does not exist or you may not access it.', status=404, code='not_found')
    return session


@bp.route('/dashboard')
@login_required(END_USER)
def dashboard():
    db = get_db()
    sessions = list(db.sessions.find({'userId': g.principal.id}).sort('startTime', -1).limit(5))
    active = db.sessions.find_one({'userId': g.principal.id, 'status': {'$in': list(ACTIVE_STATUSES)}})
    _, calibration = calibrations.current_setup(g.principal.id)
    requests = profiles.requests_for_user(g.principal.id)
    reports = list(db.reports.find({'ownerUserId': g.principal.id}).sort('generatedAt', -1).limit(5))
    return render_template('enduser/dashboard.html', sessions=sessions, active=active, calibration=calibration,
                           latest_request=requests[0] if requests else None, reports=reports,
                           checkpoint=active_checkpoint(), busy=_engine().is_busy())


# ---------------------------------------------------------------- setup (UC-08, UC-09, UC-11)
def _render_setup(problems=None, status=200, upload_error=None):
    setup, calibration = calibrations.current_setup(g.principal.id)
    sources = load_recorded_sources(current_app.config['RECORDED_SOURCES_FILE'])
    checkpoint = active_checkpoint()
    busy = _engine().is_busy()
    return render_template('enduser/setup.html', calibration=calibration, stored=calibrations.stored_calibrations(),
                           sources=sources, checkpoint=checkpoint, busy=busy, problems=problems or [],
                           upload_error=upload_error,
                           can_start=calibration is not None and checkpoint is not None and not busy), status


@bp.route('/setup')
@login_required(END_USER)
def setup():
    return _render_setup()


@bp.route('/setup/calibration/upload', methods=['POST'])
@login_required(END_USER)
def upload_calibration():
    file = request.files.get('bundle')
    try:
        doc = calibrations.upload_bundle(g.principal, file.filename if file else None, file.read() if file else b'')
    except AppError as e:
        return _render_setup(problems=e.problems, status=422, upload_error=e.message)
    flash(f'Calibration {doc["_id"]} is valid and is selected for your next session.', 'success')
    return redirect(url_for('enduser.setup'))


@bp.route('/setup/calibration/select', methods=['POST'])
@login_required(END_USER)
def select_calibration():
    try:
        doc = calibrations.select_calibration(g.principal, request.form.get('calibrationId', ''))
    except AppError as e:
        return _render_setup(problems=e.problems, status=e.status, upload_error=e.message)
    flash(f'Calibration {doc["_id"]} is valid and is selected for your next session.', 'success')
    return redirect(url_for('enduser.setup'))


@bp.route('/sessions/start', methods=['POST'])
@login_required(END_USER)
def start_session():
    mode = request.form.get('inputMode')
    try:
        session = _engine().start(g.principal, mode, request.form.get('recordedSource') if mode == RECORDED else None)
    except AppError as e:
        flash(e.message, 'danger')
        return _render_setup(problems=e.problems, status=e.status)
    return redirect(url_for('enduser.live', session_id=session['_id']))


# ---------------------------------------------------------------- sessions
@bp.route('/sessions')
@login_required(END_USER)
def sessions():
    rows = list(get_db().sessions.find({'userId': g.principal.id}).sort('startTime', -1).limit(200))
    return render_template('enduser/sessions.html', sessions=rows)


@bp.route('/sessions/<session_id>')
@login_required(END_USER)
def session_detail(session_id):
    session = _own_session(session_id)
    reports = list(get_db().reports.find({'sessionId': session_id}).sort('generatedAt', -1))
    return render_template('shared/session_detail.html', session=session, reports=reports,
                           report_problem=eligibility_problem(session))


@bp.route('/sessions/<session_id>/live')
@login_required(END_USER)
def live(session_id):
    session = _own_session(session_id)
    token = _engine().ingest_token(session_id, g.principal)
    return render_template('enduser/live.html', session=session, ingest_token=token, LIVE=LIVE)


@bp.route('/sessions/<session_id>/stop', methods=['POST'])
@login_required(END_USER)
def stop_session(session_id):
    _own_session(session_id)
    try:
        _engine().stop(session_id, g.principal)
        flash('Stop requested. Input is closed and the session is being finalized.', 'info')
    except AppError as e:
        flash(e.message, 'warning')
    return redirect(url_for('enduser.live', session_id=session_id))


# ---------------------------------------------------------------- profile (UC-04, UC-05)
@bp.route('/profile', methods=['GET', 'POST'])
@login_required(END_USER)
def profile():
    errors, message, status = {}, None, 200
    if request.method == 'POST':
        try:
            request_id, changed = profiles.apply_update(g.principal, request.form)
            flash(f'Your profile was updated ({", ".join(PROFILE_FIELDS[f] for f in changed)}). '
                  f'Request {request_id} is now Used.', 'success')
            return redirect(url_for('enduser.profile'))
        except AppError as e:
            errors = {p['field']: p['message'] for p in e.problems}
            if e.field and e.field not in errors:
                errors[e.field] = e.message
            message, status = e.message, e.status
    account = get_db().users.find_one({'_id': g.principal.id})
    approval = profiles.approved_request(g.principal.id)
    return render_template('enduser/profile.html', account=account, approval=approval, fields=PROFILE_FIELDS,
                           errors=errors, message=message, form=request.form if errors else None), status


@bp.route('/profile/requests', methods=['GET', 'POST'])
@login_required(END_USER)
def profile_requests():
    errors, message, status = {}, None, 200
    if request.method == 'POST':
        try:
            doc = profiles.submit_request(g.principal, request.form.getlist('fields'), request.form.get('reason'))
            flash(f'Profile-update request {doc["_id"]} was submitted and is Pending Admin review.', 'success')
            return redirect(url_for('enduser.profile_requests'))
        except AppError as e:
            errors = {p['field']: p['message'] for p in e.problems}
            message, status = e.message, e.status
    return render_template('enduser/profile_requests.html', requests=profiles.requests_for_user(g.principal.id),
                           open_request=profiles.open_request(g.principal.id), fields=PROFILE_FIELDS,
                           errors=errors, message=message, form=request.form), status
