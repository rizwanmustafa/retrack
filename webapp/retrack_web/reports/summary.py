"""
Report figures (FR-26), computed only from committed frame records (`frame_results`).

framesProcessed  number of committed processed frame sets
uniqueTracks     number of distinct session-local track IDs observed in them (not a count of unique real people)
meanOccupancy    mean number of observed tracks per processed frame set; frames with nobody contribute zero
peakOccupancy    largest number of observed tracks in one processed frame set
Rejected input never creates a frame record, and only tracks observed in a frame count towards its occupancy.
"""
from ..db import get_db


def frame_rows(session_id, app=None):
    cursor = get_db(app).frame_results.find(
        {'sessionId': session_id},
        {'_id': 0, 'frameIndex': 1, 'timestamp': 1, 'occupancy': 1, 'observations': 1}).sort('frameIndex', 1)
    return list(cursor)


def compute_summary(rows):
    frames = len(rows)
    track_ids = {o['trackId'] for row in rows for o in row['observations']}
    occupancies = [row['occupancy'] for row in rows]
    return {
        'framesProcessed': frames,
        'uniqueTracks': len(track_ids),
        'meanOccupancy': round(sum(occupancies) / frames, 3) if frames else 0.0,
        'peakOccupancy': max(occupancies) if frames else 0,
    }


def coverage(rows, session):
    first, last = (rows[0], rows[-1]) if rows else (None, None)
    return {
        'firstFrameIndex': first['frameIndex'] if first else None,
        'lastFrameIndex': last['frameIndex'] if last else None,
        'firstSourceTime': first['timestamp'] if first else None,
        'lastSourceTime': last['timestamp'] if last else None,
        'framesRejected': session.get('framesRejected', 0),
        'framesDiscarded': session.get('framesDiscarded', 0),
        'partial': session['status'] == 'Failed',
    }
