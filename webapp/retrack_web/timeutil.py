"""
Time convention (NFR-15): every stored and displayed time is UTC, written as ISO 8601 with millisecond precision
and a trailing 'Z', for example 2026-11-03T14:05:09.120Z. Report date filters are whole UTC days, inclusive.
"""
from datetime import datetime, date, time, timedelta, timezone


def utcnow():
    # MongoDB stores milliseconds; truncating here keeps stored and in-memory values identical
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.') + f'{dt.microsecond // 1000:03d}Z'


def parse_iso(value):
    """Parses an ISO 8601 timestamp that carries a UTC offset or 'Z'. Raises ValueError otherwise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('timestamp is missing')
    text = value.strip()
    if text.endswith('Z') or text.endswith('z'):
        text = text[:-1] + '+00:00'
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError('timestamp must include a UTC offset or Z')
    return dt.astimezone(timezone.utc)


def parse_date(value):
    """Parses YYYY-MM-DD; returns None for an empty value. Raises ValueError for a malformed one."""
    if value is None or not value.strip():
        return None
    return date.fromisoformat(value.strip())


def day_range(start, end):
    """[start 00:00:00.000Z, end 23:59:59.999Z] for inclusive UTC day boundaries; either side may be None."""
    lo = datetime.combine(start, time.min, tzinfo=timezone.utc) if start else None
    hi = datetime.combine(end, time.min, tzinfo=timezone.utc) + timedelta(days=1) - timedelta(milliseconds=1) \
        if end else None
    return lo, hi
