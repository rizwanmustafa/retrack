"""Registration (FR-01, UC-01), Admin provisioning and account decisions (FR-06, UC-06)."""
import re

from pymongo.errors import DuplicateKeyError

from .. import audit
from ..db import get_db, next_id
from ..errors import AppError
from ..security import ACTIVE, ADMIN, END_USER, PENDING, REJECTED, Principal, hash_password
from ..timeutil import utcnow

USERNAME_RE = re.compile(r'^[A-Za-z0-9._-]{3,32}$')
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

# Profile fields an End User may request to change (UC-04). Role and account status are never editable.
PROFILE_FIELDS = {'name': 'Name', 'username': 'Username', 'email': 'Email'}


def validate_field(field, value):
    """Returns (clean value, error message or None) for one account field."""
    value = (value or '').strip() if field != 'password' else (value or '')
    if field == 'name':
        if not value:
            return value, 'Name is required.'
        if len(value) > 100:
            return value, 'Name must be at most 100 characters.'
    elif field == 'username':
        if not USERNAME_RE.match(value):
            return value, 'Username must be 3 to 32 characters: letters, digits, dot, underscore or hyphen.'
    elif field == 'email':
        if len(value) > 254 or not EMAIL_RE.match(value):
            return value, 'Enter a valid email address.'
    elif field == 'password':
        if len(value) < 8 or len(value) > 128:
            return value, 'Password must be 8 to 128 characters.'
    return value, None


def identifier_in_use(field, value, exclude_user_id=None):
    """Usernames and emails are unique across End User and Admin accounts, ignoring case."""
    key = 'usernameLower' if field == 'username' else 'emailLower'
    db = get_db()
    query = {key: value.lower()}
    if db.admins.find_one(query):
        return True
    if exclude_user_id:
        query['_id'] = {'$ne': exclude_user_id}
    return db.users.find_one(query) is not None


def register_end_user(form):
    """FR-01: validates and creates an End User account as Pending. Raises AppError with field problems."""
    values, problems = {}, []
    for field in ('name', 'username', 'email', 'password'):
        values[field], error = validate_field(field, form.get(field))
        if error:
            problems.append({'field': field, 'message': error})
    if not problems:
        for field in ('username', 'email'):
            if identifier_in_use(field, values[field]):
                problems.append({'field': field, 'message': f'This {field} is already in use.'})
    if problems:
        raise AppError('The registration was not accepted.', problems=problems)

    doc = {
        '_id': next_id('users'),
        'name': values['name'],
        'username': values['username'],
        'usernameLower': values['username'].lower(),
        'email': values['email'],
        'emailLower': values['email'].lower(),
        'passwordHash': hash_password(values['password']),
        'role': END_USER,  # public registration creates End User accounts only
        'accountStatus': PENDING,
        'createdAt': utcnow(),
        'decision': None,
    }
    try:
        get_db().users.insert_one(doc)
    except DuplicateKeyError:
        raise AppError('The registration was not accepted.',
                       problems=[{'field': 'username', 'message': 'This username or email is already in use.'}])
    return doc


def create_admin(username, email, password):
    """Admin accounts are provisioned from the command line, never through registration (UC-01 constraints)."""
    for field, value in (('username', username), ('email', email), ('password', password)):
        _, error = validate_field(field, value)
        if error:
            raise AppError(error, field=field)
    for field, value in (('username', username), ('email', email)):
        if identifier_in_use(field, value.strip()):
            raise AppError(f'This {field} is already in use.', field=field)
    doc = {
        '_id': next_id('admins'),
        'username': username.strip(),
        'usernameLower': username.strip().lower(),
        'email': email.strip(),
        'emailLower': email.strip().lower(),
        'passwordHash': hash_password(password),
        'createdAt': utcnow(),
    }
    get_db().admins.insert_one(doc)
    return doc


def pending_accounts():
    return list(get_db().users.find({'accountStatus': PENDING}).sort('createdAt', 1))


def recent_account_decisions(limit=20):
    return list(get_db().users.find({'accountStatus': {'$in': [ACTIVE, REJECTED]}, 'decision': {'$ne': None}})
                .sort('decision.decidedAt', -1).limit(limit))


def decide_account(user_id, approve, admin: Principal):
    """FR-06: Pending -> Active or Rejected exactly once; a second decision is refused."""
    assert admin.role == ADMIN
    new_status = ACTIVE if approve else REJECTED
    decision = {'decision': 'Approved' if approve else 'Rejected', 'adminId': admin.id,
                'adminUsername': admin.username, 'decidedAt': utcnow()}
    db = get_db()
    result = db.users.update_one({'_id': user_id, 'accountStatus': PENDING},
                                 {'$set': {'accountStatus': new_status, 'decision': decision}})
    if result.modified_count == 0:
        user = db.users.find_one({'_id': user_id})
        if user is None:
            raise AppError('This account does not exist.', status=404, code='not_found')
        audit.record(audit.ACCOUNT_DECISION_REFUSED, audit.REFUSED, 'User', user_id,
                     {'attempted': decision['decision'], 'currentStatus': user['accountStatus']})
        raise AppError(f'This registration was already decided ({user["accountStatus"]}). '
                       'The first decision remains in effect.', status=409, code='already_decided')
    audit.record(audit.ACCOUNT_APPROVED if approve else audit.ACCOUNT_REJECTED, audit.SUCCESS, 'User', user_id,
                 {'newStatus': new_status})
    return new_status
