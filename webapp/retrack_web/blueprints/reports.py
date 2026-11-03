"""Report search, view, manual generation and download (UC-14, UC-15, UC-16)."""
from flask import Blueprint, flash, g, redirect, render_template, request, send_file, url_for

from ..errors import AppError
from ..reports import service
from ..security import login_required
from ..timeutil import day_range, parse_date

bp = Blueprint('reports', __name__, url_prefix='/reports')


@bp.route('')
@login_required()
def search():
    args = request.args
    session_id = (args.get('sessionId') or '').strip()
    error, results, searched = None, [], any(args.get(k) for k in ('sessionId', 'from', 'to'))
    try:
        start, end = parse_date(args.get('from')), parse_date(args.get('to'))
        if start and end and start > end:
            raise ValueError('the start date is after the end date')
    except ValueError as e:
        error = f'The date range is invalid: {e}. Use YYYY-MM-DD with the start on or before the end.'
    if not error:
        lo, hi = day_range(start, end)
        results = service.search(g.principal, session_id or None, lo, hi)
    return render_template('reports/search.html', results=results, error=error, args=args,
                           searched=searched), (400 if error else 200)


@bp.route('/<report_id>')
@login_required()
def detail(report_id):
    return render_template('reports/detail.html', report=service.get_report(report_id, g.principal))


@bp.route('/generate', methods=['POST'])
@login_required()
def generate():
    session_id = (request.form.get('sessionId') or '').strip()
    back = request.form.get('next') or url_for('reports.search')
    if not back.startswith('/'):
        back = url_for('reports.search')
    try:
        report = service.generate_manual(session_id, g.principal)
    except AppError as e:
        flash(e.message, 'warning')
        return redirect(back)
    if report['status'] == service.AVAILABLE:
        flash(f'Report {report["_id"]} was generated.', 'success')
        return redirect(url_for('reports.detail', report_id=report['_id']))
    flash(f'Report generation failed and was recorded ({report.get("error")}). The session results are kept, so '
          'you can try again.', 'danger')
    return redirect(back)


@bp.route('/<report_id>/download/<fmt>')
@login_required()
def download(report_id, fmt):
    report, grid_out = service.open_file(report_id, fmt, g.principal)
    mime, ext = service.FORMATS[fmt.upper()]
    return send_file(grid_out, mimetype=mime, as_attachment=True,
                     download_name=f'retrack-report-{report["_id"]}-{report["sessionId"]}.{ext}')
