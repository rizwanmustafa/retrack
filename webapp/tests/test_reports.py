"""FR-25 to FR-30, UC-14 to UC-17."""
import base64
import csv
import io
import re
import zlib

import pytest
from conftest import PASSWORD, login, register, start

from retrack_web.reports import service
from retrack_web.timeutil import utcnow


@pytest.fixture
def finished(ready_client, ai, engine, db):
    ai.script = {0: [(1, 0.0, 0.0, 0.9), (2, 50.0, 60.0, 0.8)], 5: [], 20: [(1, 10.0, 0.0, 0.9)]}
    sid = start(ready_client)
    engine.wait_idle()
    return sid, db.reports.find_one({'sessionId': sid})


def pdf_text(data):
    """Concatenated content of the PDF's streams (reportlab encodes them as ASCII85 + Flate)."""
    out = []
    for m in re.finditer(rb'stream\r?\n(.*?)endstream', data, re.S):
        chunk = m.group(1).strip()
        try:
            out.append(zlib.decompress(base64.a85decode(chunk, adobe=True)))
        except (ValueError, zlib.error):
            out.append(chunk)
    return b'\n'.join(out).decode('latin-1')


def test_pdf_and_csv_carry_the_same_summary(ready_client, finished):
    sid, report = finished
    csv_text = ready_client.get(f'/reports/{report["_id"]}/download/csv').data.decode()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    summary = {r['name']: r['value'] for r in rows if r['recordType'] == 'summary'}
    frames = [r for r in rows if r['recordType'] == 'frame']
    assert summary['framesProcessed'] == '3' and summary['uniqueTracks'] == '2'
    assert summary['meanOccupancy'] == '1.000' and summary['peakOccupancy'] == '2'
    assert summary['finalStatus'] == 'Completed' and summary['inputMode'] == 'RECORDED'
    assert 'not a guaranteed count of unique real people' in summary['uniqueTracksNote']
    # frame-level rows reproduce the summary
    occ = [int(r['occupancy']) for r in frames]
    ids = {i for r in frames for i in r['trackIds'].split(';') if i}
    assert len(frames) == 3 and sum(occ) / len(occ) == 1.0 and max(occ) == 2 and len(ids) == 2
    assert [r['frameIndex'] for r in frames] == ['0', '5', '20'] and frames[1]['occupancy'] == '0'

    pdf = ready_client.get(f'/reports/{report["_id"]}/download/pdf').data
    text = pdf_text(pdf)
    # each summary label is followed by its value in the PDF's summary table
    for label, value in (('framesProcessed', '3'), ('meanOccupancy', '1.000'), ('peakOccupancy', '2'),
                         ('uniqueTracks \\(session-local track IDs\\)', '2')):
        literal = f'({label}) Tj'
        after = text[text.index(literal) + len(literal):]
        assert re.search(r'\(([^()]*)\) Tj', after).group(1) == value, label
    assert sid in text and 'RECORDED' in text


def test_partial_report_is_clearly_identified(ready_client, ai, engine, db):
    ai.fail_on_frame = 20
    sid = start(ready_client)
    engine.wait_idle()
    report = db.reports.find_one({'sessionId': sid})
    csv_text = ready_client.get(f'/reports/{report["_id"]}/download/csv').data.decode()
    assert 'summary,finalStatus,Failed' in csv_text and 'summary,partialCoverage,yes' in csv_text
    assert 'PARTIAL REPORT' in pdf_text(ready_client.get(f'/reports/{report["_id"]}/download/pdf').data)
    assert b'Partial report' in ready_client.get(f'/reports/{report["_id"]}').data


def test_manual_generation_rules(app, ready_client, admin_client, engine, db, finished):
    sid, _ = finished
    # running session is refused
    live = start(ready_client, mode='LIVE')
    r = ready_client.post('/reports/generate', data={'sessionId': live}, follow_redirects=True)
    assert b'still running' in r.data
    ready_client.post(f'/user/sessions/{live}/stop')
    engine.wait_idle()
    # empty session is refused
    r = ready_client.post('/reports/generate', data={'sessionId': live}, follow_redirects=True)
    assert b'no processed frame sets' in r.data
    assert db.reports.count_documents({'sessionId': live}) == 0
    # owner and Admin may generate
    r = ready_client.post('/reports/generate', data={'sessionId': sid})
    assert r.status_code == 302 and '/reports/R-' in r.headers['Location']
    r = admin_client.post('/reports/generate', data={'sessionId': sid})
    assert r.status_code == 302
    manual = list(db.reports.find({'sessionId': sid, 'reportType': 'MANUAL'}))
    assert {m['generatedBy']['username'] for m in manual} == {'alice', 'root'}
    # another End User may not
    other = app.test_client()
    register(other, 'bob')
    db.users.update_one({'username': 'bob'}, {'$set': {'accountStatus': 'Active'}})
    login(other, 'bob@example.com')
    r = other.post('/reports/generate', data={'sessionId': sid}, follow_redirects=True)
    assert b'may not access' in r.data
    assert db.reports.count_documents({'sessionId': sid}) == 3


def test_render_failure_is_recorded_and_can_be_retried(ready_client, db, finished, monkeypatch):
    sid, _ = finished
    calls = {'n': 0}
    original = service.render_pdf

    def flaky(report, rows):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('font cache unavailable')
        return original(report, rows)

    monkeypatch.setattr(service, 'render_pdf', flaky)
    r = ready_client.post('/reports/generate', data={'sessionId': sid}, follow_redirects=True)
    assert b'Report generation failed' in r.data
    failed = db.reports.find_one({'sessionId': sid, 'status': 'Failed'})
    assert failed['files'] == {} and db.report_files.files.count_documents({'metadata.reportId': failed['_id']}) == 0
    assert ready_client.get(f'/reports/{failed["_id"]}/download/pdf').status_code == 404
    assert db.audit_logs.find_one({'activityType': 'REPORT_GENERATION_FAILED'})['errorFlag']
    assert db.frame_results.count_documents({'sessionId': sid}) == 3
    assert ready_client.post('/reports/generate', data={'sessionId': sid}).status_code == 302
    assert db.reports.count_documents({'sessionId': sid, 'status': 'Available'}) == 2


def test_search_respects_access_and_date_boundaries(app, ready_client, admin_client, db, finished):
    sid, report = finished
    today = utcnow().date().isoformat()
    r = ready_client.get(f'/reports?sessionId={sid}&from={today}&to={today}')
    assert r.status_code == 200 and report['_id'].encode() in r.data
    r = ready_client.get('/reports?from=2026-12-31&to=2026-01-01')
    assert r.status_code == 400 and b'date range is invalid' in r.data
    assert ready_client.get('/reports?from=yesterday').status_code == 400
    r = ready_client.get('/reports?from=2000-01-01&to=2000-01-02')
    assert r.status_code == 200 and b'No matching reports' in r.data
    r = ready_client.get('/reports?sessionId=S-999999')
    assert b'No matching reports' in r.data

    other = app.test_client()
    register(other, 'bob')
    db.users.update_one({'username': 'bob'}, {'$set': {'accountStatus': 'Active'}})
    login(other, 'bob@example.com')
    assert report['_id'].encode() not in other.get(f'/reports?sessionId={sid}').data
    assert other.get(f'/reports/{report["_id"]}').status_code == 404
    assert other.get(f'/reports/{report["_id"]}/download/pdf').status_code == 404
    assert report['_id'].encode() in admin_client.get('/reports').data


def test_downloads_are_checked_and_audited(app, ready_client, admin_client, db, finished):
    sid, report = finished
    r = admin_client.get(f'/reports/{report["_id"]}/download/csv')
    assert r.status_code == 200 and r.mimetype == 'text/csv'
    assert f'retrack-report-{report["_id"]}' in r.headers['Content-Disposition']
    log = db.audit_logs.find_one({'activityType': 'REPORT_DOWNLOADED'})
    assert log['performedBy']['username'] == 'root' and log['details']['format'] == 'CSV'
    assert ready_client.get(f'/reports/{report["_id"]}/download/docx').status_code == 400
    # missing stored file -> clear error, no download record
    db.report_files.files.delete_many({'metadata.reportId': report['_id'], 'metadata.format': 'PDF'})
    r = ready_client.get(f'/reports/{report["_id"]}/download/pdf')
    assert r.status_code == 404 and b'not available' in r.data
    assert db.audit_logs.count_documents({'activityType': 'REPORT_DOWNLOADED'}) == 1


def test_audit_log_covers_activity_and_contains_no_credentials(app, ready_client, admin_client, engine, db, finished):
    sid, report = finished
    ready_client.get(f'/reports/{report["_id"]}/download/pdf')
    types = set(db.audit_logs.distinct('activityType'))
    assert {'ACCOUNT_APPROVED', 'CHECKPOINT_CHANGED', 'SESSION_STARTED', 'SESSION_COMPLETED', 'INPUT_ERROR',
            'REPORT_GENERATED', 'REPORT_DOWNLOADED'} <= types
    hashes = [u['passwordHash'] for u in db.users.find()] + [a['passwordHash'] for a in db.admins.find()]
    dump = str(list(db.audit_logs.find()))
    assert PASSWORD not in dump and not any(h in dump for h in hashes)
    assert 'tokenHash' not in dump and 'passwordHash' not in dump

    page = admin_client.get('/admin/audit?type=Errors').data
    assert b'<span class="activity">INPUT_ERROR</span>' in page
    assert b'<span class="activity">ACCOUNT_APPROVED</span>' not in page
    assert admin_client.get('/admin/audit?from=2026-12-31&to=2026-01-01').status_code == 400
    assert b'No matching records' in admin_client.get('/admin/audit?from=2000-01-01&to=2000-01-01').data
    error = db.audit_logs.find_one({'activityType': 'INPUT_ERROR'})
    detail = admin_client.get(f'/admin/audit/{error["_id"]}').data
    assert b'MISSING_CAMERA' in detail or b'WRONG_SIZE' in detail
    assert ready_client.get('/admin/audit').status_code == 403
