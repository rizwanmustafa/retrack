"""
PDF and CSV report files (FR-27). Both are rendered from the same `report` dictionary, so their summary figures are
identical. The CSV schema is documented in docs/REPORT_CSV_SCHEMA.md.
"""
import csv
import io

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..timeutil import iso

CSV_COLUMNS = ['recordType', 'name', 'value', 'frameIndex', 'sourceTime', 'occupancy', 'trackIds']
UNIQUE_TRACKS_NOTE = 'Session-local track IDs; not a guaranteed count of unique real people.'


def summary_items(report):
    """Ordered (name, value) pairs shared by the PDF and the CSV."""
    s, c = report['summary'], report['coverage']
    return [
        ('reportId', report['_id']),
        ('sessionId', report['sessionId']),
        ('finalStatus', report['sessionStatus']),
        ('partialCoverage', 'yes' if c['partial'] else 'no'),
        ('inputMode', report['inputMode']),
        ('generatedAt', iso(report['generatedAt'])),
        ('reportType', report['reportType']),
        ('firstFrameIndex', c['firstFrameIndex']),
        ('lastFrameIndex', c['lastFrameIndex']),
        ('firstSourceTime', iso(c['firstSourceTime'])),
        ('lastSourceTime', iso(c['lastSourceTime'])),
        ('framesRejected', c['framesRejected']),
        ('framesDiscardedAtStop', c['framesDiscarded']),
        ('framesProcessed', s['framesProcessed']),
        ('uniqueTracks', s['uniqueTracks']),
        ('uniqueTracksNote', UNIQUE_TRACKS_NOTE),
        ('meanOccupancy', f'{s["meanOccupancy"]:.3f}'),
        ('peakOccupancy', s['peakOccupancy']),
        ('occupancyUnit', 'observed tracks per processed frame set'),
        ('checkpointId', report['model'].get('checkpointId')),
        ('modelName', report['model'].get('modelName')),
        ('modelVersion', report['model'].get('version')),
        ('checkpointSha256', report['model'].get('sha256')),
        ('calibrationId', report['calibrationId']),
        ('cameraOrder', ' '.join(report['cameraOrder'])),
        ('timeConvention', 'ISO 8601, UTC'),
    ]


def render_csv(report, rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(CSV_COLUMNS)
    for name, value in summary_items(report):
        writer.writerow(['summary', name, '' if value is None else value, '', '', '', ''])
    for row in rows:
        ids = ';'.join(str(o['trackId']) for o in row['observations'])
        writer.writerow(['frame', '', '', row['frameIndex'], iso(row['timestamp']), row['occupancy'], ids])
    return out.getvalue().encode('utf-8')


def _occupancy_chart(rows, width):
    drawing = Drawing(width, 60 * mm)
    plot = LinePlot()
    plot.x, plot.y = 12 * mm, 10 * mm
    plot.width, plot.height = width - 20 * mm, 42 * mm
    plot.data = [[(r['frameIndex'], r['occupancy']) for r in rows]]
    plot.lines[0].strokeColor = colors.HexColor('#0d6efd')
    plot.yValueAxis.valueMin = 0
    plot.xValueAxis.labels.fontSize = 7
    plot.yValueAxis.labels.fontSize = 7
    drawing.add(plot)
    drawing.add(String(12 * mm, 56 * mm, 'Occupancy per processed frame set (x: frame index)', fontSize=8))
    return drawing


def render_pdf(report, rows):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f'Retrack report {report["_id"]}',
                            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    small = ParagraphStyle('small', parent=styles['BodyText'], fontSize=8.5, leading=11)
    banner = ParagraphStyle('banner', parent=styles['BodyText'], textColor=colors.white, fontSize=10, leading=13)
    story = [Paragraph('Retrack session report', styles['Title'])]
    if report['coverage']['partial']:
        t = Table([[Paragraph('<b>PARTIAL REPORT.</b> The session ended with status Failed. The figures cover only '
                              'the frame sets committed before the failure.', banner)]],
                  colWidths=[doc.width])
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#dc3545')),
                               ('BOX', (0, 0), (-1, -1), 0, colors.white),
                               ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
        story += [t, Spacer(1, 4 * mm)]

    items = dict(summary_items(report))
    figures = [['Figure', 'Value'],
               ['framesProcessed', items['framesProcessed']],
               ['uniqueTracks (session-local track IDs)', items['uniqueTracks']],
               ['meanOccupancy', items['meanOccupancy']],
               ['peakOccupancy', items['peakOccupancy']]]
    ident_keys = ['reportId', 'sessionId', 'finalStatus', 'partialCoverage', 'inputMode', 'generatedAt',
                  'reportType', 'firstFrameIndex', 'lastFrameIndex', 'firstSourceTime', 'lastSourceTime',
                  'framesRejected', 'framesDiscardedAtStop', 'checkpointId', 'modelName', 'modelVersion',
                  'checkpointSha256', 'calibrationId', 'cameraOrder', 'timeConvention']
    ident = [[k, Paragraph('' if items[k] is None else str(items[k]), small)] for k in ident_keys]
    table_style = TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#ced4da')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ])
    head_style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9ecef')),
                             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')])

    story.append(Paragraph('Summary', styles['Heading2']))
    t = Table(figures, colWidths=[doc.width * 0.6, doc.width * 0.4])
    t.setStyle(table_style)
    t.setStyle(head_style)
    story += [t, Spacer(1, 2 * mm),
              Paragraph(f'uniqueTracks: {UNIQUE_TRACKS_NOTE} Occupancy counts tracks observed in a processed frame '
                        'set; valid frame sets with no pedestrians count as zero and rejected input is excluded. '
                        'Frame-level data is in the CSV report.', small)]
    if len(rows) > 1:
        story += [Spacer(1, 4 * mm), _occupancy_chart(rows, doc.width)]
    story.append(Paragraph('Session and source', styles['Heading2']))
    t = Table(ident, colWidths=[doc.width * 0.3, doc.width * 0.7])
    t.setStyle(table_style)
    story.append(t)
    story += [Spacer(1, 3 * mm),
              Paragraph('Times are ISO 8601 in UTC. Track IDs are unique within this session only and do not '
                        'identify real people.', small)]
    doc.build(story)
    return buf.getvalue()
