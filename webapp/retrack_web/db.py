"""MongoDB access: one client per application, collection indexes and readable identifiers."""
import gridfs
from flask import current_app
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument

# Readable identifiers, e.g. S-000012 for a session
ID_PREFIXES = {
    'admins': 'A', 'users': 'U', 'profile_requests': 'P', 'calibrations': 'C', 'checkpoints': 'K',
    'sessions': 'S', 'reports': 'R', 'audit_logs': 'L', 'feeds': 'F',
}


def init_db(app):
    uri = app.config['MONGO_URI']
    if uri.startswith('mongomock://'):
        # In-memory database for tests and local trials without a MongoDB server; nothing is persisted.
        import mongomock
        import mongomock.gridfs
        mongomock.gridfs.enable_gridfs_integration()
        client = mongomock.MongoClient(tz_aware=True)
    else:
        client = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=5000)
    app.extensions['retrack_mongo'] = client
    app.extensions['retrack_db'] = client[app.config['MONGO_DB']]
    ensure_indexes(app.extensions['retrack_db'])


def get_db(app=None):
    return (app or current_app).extensions['retrack_db']


def get_fs(app=None):
    return gridfs.GridFS(get_db(app), collection='report_files')


def next_id(collection, app=None):
    counter = get_db(app).counters.find_one_and_update(
        {'_id': collection}, {'$inc': {'seq': 1}}, upsert=True, return_document=ReturnDocument.AFTER)
    return f'{ID_PREFIXES[collection]}-{counter["seq"]:06d}'


def ensure_indexes(db):
    for name in ('admins', 'users'):
        db[name].create_index('emailLower', unique=True)
        db[name].create_index('usernameLower', unique=True)
    db.users.create_index('accountStatus')
    db.auth_sessions.create_index('tokenHash', unique=True)
    db.profile_requests.create_index([('userId', ASCENDING), ('createdAt', DESCENDING)])
    db.profile_requests.create_index('status')
    db.calibrations.create_index('uploadedAt')
    db.sessions.create_index([('userId', ASCENDING), ('startTime', DESCENDING)])
    db.sessions.create_index('status')
    db.frame_results.create_index([('sessionId', ASCENDING), ('frameIndex', ASCENDING)], unique=True)
    db.tracks.create_index([('sessionId', ASCENDING), ('trackId', ASCENDING)], unique=True)
    db.reports.create_index([('ownerUserId', ASCENDING), ('generatedAt', DESCENDING)])
    db.reports.create_index('sessionId')
    db.audit_logs.create_index([('timestamp', DESCENDING)])
    db.audit_logs.create_index('activityType')
