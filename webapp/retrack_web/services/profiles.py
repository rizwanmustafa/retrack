"""Profile-update requests, decisions and approved edits (FR-07 to FR-09, UC-04, UC-05, UC-07)."""
from pymongo.errors import DuplicateKeyError

from .. import audit
from ..db import get_db, next_id
from ..errors import AppError
from ..security import ADMIN, END_USER, Principal
from ..timeutil import utcnow
from .accounts import PROFILE_FIELDS, identifier_in_use, validate_field

PENDING, APPROVED, REJECTED, USED = 'Pending', 'Approved', 'Rejected', 'Used'


def requests_for_user(user_id):
    return list(get_db().profile_requests.find({'userId': user_id}).sort('createdAt', -1))


def open_request(user_id):
    """The user's Pending request or unused Approved request, if any."""
    return get_db().profile_requests.find_one({'userId': user_id, 'status': {'$in': [PENDING, APPROVED]}})


def approved_request(user_id):
    return get_db().profile_requests.find_one({'userId': user_id, 'status': APPROVED})


def submit_request(user: Principal, fields, reason):
    """FR-07: stores a Pending request naming the fields to change and a reason."""
    assert user.role == END_USER
    fields = [f for f in PROFILE_FIELDS if f in (fields or [])]
    reason = (reason or '').strip()
    problems = []
    if not fields:
        problems.append({'field': 'fields', 'message': 'Select at least one field to change.'})
    if not reason:
        problems.append({'field': 'reason', 'message': 'Enter the reason for the change.'})
    elif len(reason) > 1000:
        problems.append({'field': 'reason', 'message': 'The reason must be at most 1000 characters.'})
    if problems:
        raise AppError('Complete the request before submitting it.', problems=problems)
    existing = open_request(user.id)
    if existing:
        raise AppError(f'You already have an open request ({existing["_id"]}, {existing["status"]}). '
                       'It must be decided or used before you submit another.', status=409, code='request_open')
    doc = {
        '_id': next_id('profile_requests'),
        'userId': user.id,
        'fieldsToChange': fields,
        'reason': reason,
        'status': PENDING,
        'createdAt': utcnow(),
        'reviewedBy': None,
        'decidedAt': None,
        'permittedFields': [],
        'usedAt': None,
    }
    get_db().profile_requests.insert_one(doc)
    return doc


def pending_requests():
    return list(get_db().profile_requests.find({'status': PENDING}).sort('createdAt', 1))


def decide_request(request_id, approve, permitted_fields, admin: Principal):
    """FR-08: records the decision, the permitted fields, the deciding Admin and the time."""
    assert admin.role == ADMIN
    db = get_db()
    req = db.profile_requests.find_one({'_id': request_id})
    if req is None:
        raise AppError('This profile request does not exist.', status=404, code='not_found')
    permitted = [f for f in req['fieldsToChange'] if f in (permitted_fields or [])] if approve else []
    if approve and not permitted:
        raise AppError('Select at least one requested field to permit.', field='permittedFields')
    status = APPROVED if approve else REJECTED
    result = db.profile_requests.update_one(
        {'_id': request_id, 'status': PENDING},
        {'$set': {'status': status, 'permittedFields': permitted, 'decidedAt': utcnow(),
                  'reviewedBy': {'adminId': admin.id, 'adminUsername': admin.username}}})
    if result.modified_count == 0:
        audit.record(audit.PROFILE_DECISION_REFUSED, audit.REFUSED, 'ProfileRequest', request_id,
                     {'attempted': status, 'currentStatus': req['status']})
        raise AppError(f'This request was already decided ({req["status"]}).', status=409, code='already_decided')
    audit.record(audit.PROFILE_REQUEST_APPROVED if approve else audit.PROFILE_REQUEST_REJECTED, audit.SUCCESS,
                 'ProfileRequest', request_id, {'userId': req['userId'], 'permittedFields': permitted})
    return status


def apply_update(user: Principal, form):
    """
    FR-09 / UC-05: updates only the fields an unused Approved request permits and marks the request Used.
    Any invalid value, duplicate identifier or change to a field outside the approval leaves the profile and the
    approval unchanged.
    """
    db = get_db()
    req = approved_request(user.id)
    if req is None:
        raise AppError('You have no approved, unused profile-update request. Request a profile update first.',
                       status=403, code='no_approval')
    account = db.users.find_one({'_id': user.id})
    if any(k in form for k in ('role', 'accountStatus', 'status', 'password')):
        raise AppError('Role, account status and password cannot be changed through a profile update.',
                       status=403, code='field_not_permitted')

    permitted = req['permittedFields']
    for field in PROFILE_FIELDS:
        if field in form and field not in permitted and (form.get(field) or '').strip() != account[field]:
            raise AppError(f'{PROFILE_FIELDS[field]} is not covered by your approval, so the change was refused.',
                           status=403, code='field_not_permitted', field=field)

    updates, problems = {}, []
    for field in permitted:
        value, error = validate_field(field, form.get(field))
        if error:
            problems.append({'field': field, 'message': error})
        elif field in ('username', 'email') and value.lower() != account[field].lower() \
                and identifier_in_use(field, value, exclude_user_id=user.id):
            problems.append({'field': field, 'message': f'This {field} is already in use.'})
        else:
            updates[field] = value
    if problems:
        raise AppError('The profile was not changed. Your approval is still available.', problems=problems)

    for field in ('username', 'email'):
        if field in updates:
            updates[field + 'Lower'] = updates[field].lower()
    # Consume the approval first so it can be used once even with concurrent saves.
    claimed = db.profile_requests.update_one({'_id': req['_id'], 'status': APPROVED},
                                             {'$set': {'status': USED, 'usedAt': utcnow()}})
    if claimed.modified_count == 0:
        raise AppError('This approval has already been used.', status=409, code='approval_used')
    try:
        db.users.update_one({'_id': user.id}, {'$set': updates})
    except DuplicateKeyError:
        db.profile_requests.update_one({'_id': req['_id']}, {'$set': {'status': APPROVED, 'usedAt': None}})
        raise AppError('The profile was not changed: the username or email is already in use. '
                       'Your approval is still available.', status=409, code='duplicate')
    return req['_id'], [f for f in permitted if f in updates]
