"""
Measured processing figures for one session (NFR-06 evidence): throughput and 95th-percentile result delay
(frame set received -> frame record committed), computed from the stored frame records.

    python scripts/measure_session.py --session S-000004 [--warmup 5]

Reads RETRACK_MONGO_URI / RETRACK_MONGO_DB. Record the printed figures together with the hardware, checkpoint,
browser versions, source rate and load, as the SRS requires; real-time capability must not be claimed from a target.
"""
import argparse
import json
import os

from pymongo import MongoClient


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--session', required=True)
    p.add_argument('--warmup', type=int, default=5, help='frame sets excluded as warm-up')
    args = p.parse_args()
    db = MongoClient(os.environ.get('RETRACK_MONGO_URI', 'mongodb://localhost:27017'), tz_aware=True)[
        os.environ.get('RETRACK_MONGO_DB', 'retrack')]
    session = db.sessions.find_one({'_id': args.session})
    if session is None:
        raise SystemExit('unknown session')
    rows = list(db.frame_results.find({'sessionId': args.session}).sort('committedAt', 1))[args.warmup:]
    if len(rows) < 2:
        raise SystemExit('not enough processed frame sets after warm-up')
    delays = [(r['committedAt'] - r['receivedAt']).total_seconds() for r in rows]
    span = (rows[-1]['committedAt'] - rows[0]['committedAt']).total_seconds()
    print(json.dumps({
        'sessionId': args.session,
        'inputMode': session['inputMode'],
        'checkpoint': session['checkpoint'],
        'environment': session.get('environment'),
        'frameSetsMeasured': len(rows),
        'warmupExcluded': args.warmup,
        'throughputFrameSetsPerSecond': round((len(rows) - 1) / span, 3) if span > 0 else None,
        'resultDelaySeconds': {'p50': round(percentile(delays, 0.5), 3), 'p95': round(percentile(delays, 0.95), 3),
                               'max': round(max(delays), 3)},
        'inferenceMilliseconds': {'p95': percentile([r['processingMs'] for r in rows], 0.95)},
    }, indent=1, default=str))


if __name__ == '__main__':
    main()
