"""Admin pages: account and profile decisions, checkpoints, audit log, sessions (UC-06, UC-07, UC-10, UC-17)."""
import math

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, url_for

from .. import audit
from ..db import get_db
from ..errors import AppError
from ..monitoring.engine import ACTIVE_STATUSES
from ..reports.service import eligibility_problem
from ..security import ADMIN, login_required
from ..services import accounts, checkpoints, profiles
from ..services.accounts import PROFILE_FIELDS
from ..timeutil import day_range, parse_date

bp = Blueprint('admin', __name__, url_prefix='/admin')
PAGE_SIZE = 50


@bp.route('/dashboard')
@login_required(ADMIN)
def dashboard():
    db = get_db()
    return render_template(
        'admin/dashboard.html',
        pending_accounts=db.users.count_documents({'accountStatus': 'Pending'}),
        pending_requests=db.profile_requests.count_documents({'status': 'Pending'}),
        checkpoint=checkpoints.active_checkpoint(),
        active_session=db.sessions.find_one({'status': {'$in': list(ACTIVE_STATUSES)}}),
        recent_errors=list(db.audit_logs.find({'errorFlag': True}).sort('timestamp', -1).limit(8)),
    )


# ---------------------------------------------------------------- accounts (UC-06)
@bp.route('/accounts')
@login_required(ADMIN)
def accounts_page():
    return render_template('admin/accounts.html', pending=accounts.pending_accounts(),
                           decided=accounts.recent_account_decisions())


@bp.route('/accounts/<user_id>/decision', methods=['POST'])
@login_required(ADMIN)
def decide_account(user_id):
    decision = request.form.get('decision')
    if decision not in ('approve', 'reject'):
        raise AppError('Choose approve or reject.', field='decision')
    try:
        status = accounts.decide_account(user_id, decision == 'approve', g.principal)
        flash(f'Account {user_id} is now {status}.', 'success')
    except AppError as e:
        flash(e.message, 'warning')
    return redirect(url_for('admin.accounts_page'))


# ---------------------------------------------------------------- profile requests (UC-07)
@bp.route('/profile-requests')
@login_required(ADMIN)
def profile_requests():
    db = get_db()
    pending = profiles.pending_requests()
    decided = list(db.profile_requests.find({'status': {'$ne': 'Pending'}}).sort('decidedAt', -1).limit(20))
    users = {u['_id']: u for u in db.users.find({'_id': {'$in': [r['userId'] for r in pending + decided]}})}
    return render_template('admin/profile_requests.html', pending=pending, decided=decided, users=users,
                           fields=PROFILE_FIELDS)


@bp.route('/profile-requests/<request_id>')
@login_required(ADMIN)
def profile_request_detail(request_id):
    db = get_db()
    req = db.profile_requests.find_one({'_id': request_id})
    if req is None:
        raise AppError('This profile request does not exist.', status=404, code='not_found')
    return render_template('admin/profile_request_detail.html', req=req, user=db.users.find_one({'_id': req['userId']}),
                           fields=PROFILE_FIELDS)


@bp.route('/profile-requests/<request_id>/decision', methods=['POST'])
@login_required(ADMIN)
def decide_profile_request(request_id):
    decision = request.form.get('decision')
    if decision not in ('approve', 'reject'):
        raise AppError('Choose approve or reject.', field='decision')
    try:
        status = profiles.decide_request(request_id, decision == 'approve', request.form.getlist('permittedFields'),
                                         g.principal)
        flash(f'Request {request_id} is now {status}.', 'success')
        return redirect(url_for('admin.profile_requests'))
    except AppError as e:
        flash(e.message, 'warning')
        return redirect(url_for('admin.profile_request_detail', request_id=request_id))


# ---------------------------------------------------------------- checkpoints (UC-10)
@bp.route('/checkpoints')
@login_required(ADMIN)
def checkpoints_page():
    checkpoints.scan()
    engine = current_app.extensions['retrack_engine']
    return render_template('admin/checkpoints.html', checkpoints=checkpoints.list_checkpoints(),
                           active=checkpoints.active_checkpoint(), busy=engine.is_busy())


@bp.route('/checkpoints/<checkpoint_id>/select', methods=['POST'])
@login_required(ADMIN)
def select_checkpoint(checkpoint_id):
    try:
        doc = checkpoints.select(checkpoint_id, g.principal)
        flash(f'Checkpoint {doc["_id"]} ({doc["fileName"]}) is now active for new sessions.', 'success')
    except AppError as e:
        details = '; '.join(p['reason'] for p in e.problems)
        flash(e.message + (f' Problems: {details}' if details else ''), 'danger')
    return redirect(url_for('admin.checkpoints_page'))


# ---------------------------------------------------------------- audit (UC-17)
@bp.route('/audit')
@login_required(ADMIN)
def audit_log():
    args = request.args
    query, error = {}, None
    activity = args.get('type', '')
    if activity in audit.ACTIVITY_GROUPS:
        query['activityType'] = {'$in': audit.ACTIVITY_GROUPS[activity]}
    elif activity in audit.ACTIVITY_TYPES:
        query['activityType'] = activity
    if args.get('errors') == '1':
        query['errorFlag'] = True
    if args.get('resource'):
        query['resource.id'] = args['resource'].strip()
    try:
        start, end = parse_date(args.get('from')), parse_date(args.get('to'))
        if start and end and start > end:
            raise ValueError('the start date is after the end date')
        lo, hi = day_range(start, end)
        if lo or hi:
            query['timestamp'] = {k: v for k, v in (('$gte', lo), ('$lte', hi)) if v}
    except ValueError as e:
        error = f'Invalid date range: {e}.'
    records, total, page, pages = [], 0, 1, 1
    if not error:
        db = get_db()
        total = db.audit_logs.count_documents(query)
        pages = max(1, math.ceil(total / PAGE_SIZE))
        page = min(max(1, args.get('page', 1, type=int)), pages)
        records = list(db.audit_logs.find(query).sort('timestamp', -1).skip((page - 1) * PAGE_SIZE).limit(PAGE_SIZE))
    return render_template('admin/audit.html', records=records, total=total, page=page, pages=pages, error=error,
                           groups=audit.ACTIVITY_GROUPS, args=args), (400 if error else 200)


@bp.route('/audit/<log_id>')
@login_required(ADMIN)
def audit_detail(log_id):
    db = get_db()
    record = db.audit_logs.find_one({'_id': log_id})
    if record is None:
        raise AppError('This audit record does not exist.', status=404, code='not_found')
    session = None
    if record['resource']['type'] == 'Session' and record['resource']['id']:
        session = db.sessions.find_one({'_id': record['resource']['id']}, {'ingestTokenHash': 0})
    return render_template('admin/audit_detail.html', record=record, session=session)


# ---------------------------------------------------------------- sessions (FR-25 manual reports)
@bp.route('/sessions')
@login_required(ADMIN)
def sessions():
    rows = list(get_db().sessions.find().sort('startTime', -1).limit(200))
    return render_template('admin/sessions.html', sessions=rows)


@bp.route('/sessions/<session_id>')
@login_required(ADMIN)
def session_detail(session_id):
    db = get_db()
    session = db.sessions.find_one({'_id': session_id})
    if session is None:
        raise AppError('This session does not exist.', status=404, code='not_found')
    reports = list(db.reports.find({'sessionId': session_id}).sort('generatedAt', -1))
    return render_template('shared/session_detail.html', session=session, reports=reports,
                           report_problem=eligibility_problem(session))
