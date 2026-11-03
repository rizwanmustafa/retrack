"""
Audit and diagnostic records (FR-30, UC-17).

Each record has the time, the actor or system identity, the affected resource and the outcome. Records never contain
passwords, password hashes, authentication secrets or session tokens (NFR-02): details are filtered through
`_clean` before they are stored.
"""
from flask import g, has_request_context

from .db import get_db, next_id
from .timeutil import utcnow

# Activity types, grouped as FR-30 lists them
ACCOUNT_APPROVED = 'ACCOUNT_APPROVED'
ACCOUNT_REJECTED = 'ACCOUNT_REJECTED'
ACCOUNT_DECISION_REFUSED = 'ACCOUNT_DECISION_REFUSED'
PROFILE_REQUEST_APPROVED = 'PROFILE_REQUEST_APPROVED'
PROFILE_REQUEST_REJECTED = 'PROFILE_REQUEST_REJECTED'
PROFILE_DECISION_REFUSED = 'PROFILE_DECISION_REFUSED'
CHECKPOINT_CHANGED = 'CHECKPOINT_CHANGED'
CHECKPOINT_CHANGE_REFUSED = 'CHECKPOINT_CHANGE_REFUSED'
SESSION_STARTED = 'SESSION_STARTED'
SESSION_START_FAILED = 'SESSION_START_FAILED'
SESSION_COMPLETED = 'SESSION_COMPLETED'
SESSION_FAILED = 'SESSION_FAILED'
REPORT_GENERATED = 'REPORT_GENERATED'
REPORT_GENERATION_FAILED = 'REPORT_GENERATION_FAILED'
REPORT_DOWNLOADED = 'REPORT_DOWNLOADED'
INPUT_ERROR = 'INPUT_ERROR'
PROCESSING_ERROR = 'PROCESSING_ERROR'

ACTIVITY_GROUPS = {
    'Account decisions': [ACCOUNT_APPROVED, ACCOUNT_REJECTED, ACCOUNT_DECISION_REFUSED],
    'Profile decisions': [PROFILE_REQUEST_APPROVED, PROFILE_REQUEST_REJECTED, PROFILE_DECISION_REFUSED],
    'Checkpoint changes': [CHECKPOINT_CHANGED, CHECKPOINT_CHANGE_REFUSED],
    'Sessions': [SESSION_STARTED, SESSION_START_FAILED, SESSION_COMPLETED, SESSION_FAILED],
    'Reports': [REPORT_GENERATED, REPORT_GENERATION_FAILED, REPORT_DOWNLOADED],
    'Errors': [INPUT_ERROR, PROCESSING_ERROR],
}
ACTIVITY_TYPES = [t for group in ACTIVITY_GROUPS.values() for t in group]

SUCCESS, FAILURE, REFUSED = 'SUCCESS', 'FAILURE', 'REFUSED'
SYSTEM_ACTOR = {'type': 'SYSTEM', 'id': 'SYSTEM', 'username': 'system'}

_SECRET_KEYS = ('password', 'passwordhash', 'token', 'tokenhash', 'secret', 'cookie', 'authorization')


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not any(s in k.lower() for s in _SECRET_KEYS)}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def actor_from_principal(principal):
    return {'type': principal.role, 'id': principal.id, 'username': principal.username}


def current_actor():
    if has_request_context() and getattr(g, 'principal', None) is not None:
        return actor_from_principal(g.principal)
    return SYSTEM_ACTOR


def record(activity_type, outcome, resource_type, resource_id, details=None, actor=None, error=False, app=None):
    assert activity_type in ACTIVITY_TYPES, activity_type
    doc = {
        '_id': next_id('audit_logs', app),
        'timestamp': utcnow(),
        'activityType': activity_type,
        'performedBy': actor or current_actor(),
        'resource': {'type': resource_type, 'id': resource_id},
        'outcome': outcome,
        'details': _clean(details or {}),
        'errorFlag': bool(error),
    }
    get_db(app).audit_logs.insert_one(doc)
    return doc
