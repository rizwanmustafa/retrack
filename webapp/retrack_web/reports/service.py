"""
Report generation, search and download (FR-25 to FR-29, UC-14 to UC-16).

A report is published by one final update that sets status Available after both files are stored, so a report is
never visible as available while partially written (NFR-10). A failed attempt is recorded, its files are removed and
the session's committed results stay in place for a later retry.
"""
from flask import current_app

from .. import audit
from ..db import get_db, get_fs, next_id
from ..errors import AppError
from ..security import ADMIN, END_USER, Principal
from ..textutil import safe_message
from ..timeutil import utcnow
from .render import render_csv, render_pdf
from .summary import compute_summary, coverage, frame_rows

AUTOMATIC, MANUAL = 'AUTOMATIC', 'MANUAL'
GENERATING, AVAILABLE, FAILED = 'Generating', 'Available', 'Failed'
FORMATS = {'PDF': ('application/pdf', 'pdf'), 'CSV': ('text/csv', 'csv')}
FINAL_STATUSES = ('Completed', 'Failed')


def can_access_session(principal: Principal, session):
    return principal.role == ADMIN or (principal.role == END_USER and session['userId'] == principal.id)


def can_access_report(principal: Principal, report):
    return principal.role == ADMIN or (principal.role == END_USER and report['ownerUserId'] == principal.id)


def eligibility_problem(session):
    """None when the session can have a report, otherwise the reason (UC-14 1a)."""
    if session['status'] not in FINAL_STATUSES:
        return 'The session is still running. A report can be generated after it is finalized.'
    if not session.get('framesProcessed'):
        return 'The session has no processed frame sets, so there is nothing to report.'
    return None


def generate(session_id, report_type, actor, app=None):
    """Generates PDF and CSV for an eligible session. Returns the report document (Available or Failed)."""
    app = app or current_app._get_current_object()
    db = get_db(app)
    session = db.sessions.find_one({'_id': session_id})
    problem = eligibility_problem(session)
    if problem:
        raise AppError(problem, status=409, code='not_eligible')

    report = {
        '_id': next_id('reports', app),
        'sessionId': session_id,
        'ownerUserId': session['userId'],
        'ownerUsername': session.get('username'),
        'reportType': report_type,
        'generatedBy': actor,
        'status': GENERATING,
        'generatedAt': utcnow(),
        'sessionStatus': session['status'],
        'inputMode': session['inputMode'],
        'model': session['checkpoint'],
        'calibrationId': session['calibrationId'],
        'cameraOrder': session['cameraOrder'],
        'files': {},
    }
    db.reports.insert_one(report)
    fs = get_fs(app)
    stored = []
    try:
        rows = frame_rows(session_id, app)
        report['summary'] = compute_summary(rows)
        report['coverage'] = coverage(rows, session)
        contents = {'PDF': render_pdf(report, rows), 'CSV': render_csv(report, rows)}
        for fmt, data in contents.items():
            mime, ext = FORMATS[fmt]
            file_id = fs.put(data, filename=f'retrack-report-{report["_id"]}.{ext}', contentType=mime,
                             metadata={'reportId': report['_id'], 'format': fmt})
            stored.append(file_id)
            report['files'][fmt] = {'fileId': file_id, 'sizeBytes': len(data)}
        db.reports.update_one({'_id': report['_id']}, {'$set': {
            'summary': report['summary'], 'coverage': report['coverage'], 'files': report['files'],
            'status': AVAILABLE}})
        report['status'] = AVAILABLE
        audit.record(audit.REPORT_GENERATED, audit.SUCCESS, 'Report', report['_id'],
                     {'sessionId': session_id, 'reportType': report_type, **report['summary']}, actor=actor, app=app)
    except Exception as e:
        app.logger.exception('Report generation failed for %s', session_id)
        for file_id in stored:
            fs.delete(file_id)
        message = safe_message(e)
        db.reports.update_one({'_id': report['_id']}, {'$set': {'status': FAILED, 'files': {}, 'error': message}})
        report.update(status=FAILED, files={}, error=message)
        audit.record(audit.REPORT_GENERATION_FAILED, audit.FAILURE, 'Report', report['_id'],
                     {'sessionId': session_id, 'reportType': report_type, 'error': message}, actor=actor,
                     error=True, app=app)
    return report


def generate_manual(session_id, principal: Principal):
    """FR-25 manual request from the owning End User or an Admin."""
    session = get_db().sessions.find_one({'_id': session_id})
    if session is None or not can_access_session(principal, session):
        # the same answer for missing and inaccessible sessions, so IDs of other users are not revealed
        raise AppError('This session does not exist or you may not access it.', status=404, code='not_found')
    return generate(session_id, MANUAL, audit.actor_from_principal(principal))


def search(principal: Principal, session_id=None, start=None, end=None, limit=200):
    """FR-28: only reports the user may see, filtered by session identifier and generation time range."""
    query = {}
    if principal.role == END_USER:
        query['ownerUserId'] = principal.id
    if session_id:
        query['sessionId'] = session_id
    if start or end:
        query['generatedAt'] = {}
        if start:
            query['generatedAt']['$gte'] = start
        if end:
            query['generatedAt']['$lte'] = end
    return list(get_db().reports.find(query).sort('generatedAt', -1).limit(limit))


def get_report(report_id, principal: Principal):
    report = get_db().reports.find_one({'_id': report_id})
    if report is None or not can_access_report(principal, report):
        raise AppError('This report does not exist or you may not access it.', status=404, code='not_found')
    return report


def open_file(report_id, fmt, principal: Principal):
    """FR-29: checks access, loads the file and records the download. Returns (report, grid_out)."""
    fmt = (fmt or '').upper()
    if fmt not in FORMATS:
        raise AppError('Choose PDF or CSV.', field='format')
    report = get_report(report_id, principal)
    if report['status'] != AVAILABLE or fmt not in report.get('files', {}):
        raise AppError('This report file is not available. Generate the report again from its session.',
                       status=404, code='file_unavailable')
    try:
        grid_out = get_fs().get(report['files'][fmt]['fileId'])
    except Exception:
        raise AppError('This report file is not available in storage.', status=404, code='file_unavailable')
    audit.record(audit.REPORT_DOWNLOADED, audit.SUCCESS, 'Report', report_id,
                 {'format': fmt, 'sessionId': report['sessionId']})
    return report, grid_out
