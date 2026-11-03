"""
Authentication and authorization (FR-02 to FR-05, UC-02, UC-03, NFR-02, NFR-03).

The browser holds a random token in an HttpOnly cookie; the server stores only its SHA-256 hash in
`auth_sessions`, so logout invalidates the session on the server and a copied cookie stops working. Every protected
request re-reads the account, checks that it is still Active and checks the role on the server.
"""
import functools
import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta

from flask import abort, current_app, flash, g, jsonify, redirect, request, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .db import get_db
from .timeutil import utcnow

END_USER = 'END_USER'
ADMIN = 'ADMIN'
ROLE_LABELS = {END_USER: 'End User', ADMIN: 'Admin'}

PENDING, ACTIVE, REJECTED = 'Pending', 'Active', 'Rejected'

# Verifying against this hash when the account is unknown keeps the response time similar for unknown accounts.
_DUMMY_HASH = generate_password_hash('retrack-dummy-password')


@dataclass
class Principal:
    id: str
    role: str
    username: str
    email: str
    name: str = ''

    @property
    def is_admin(self):
        return self.role == ADMIN

    @property
    def role_label(self):
        return ROLE_LABELS[self.role]


def hash_password(password):
    # Werkzeug's default is salted scrypt (NFR-02)
    return generate_password_hash(password)


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def find_account_by_email(email):
    db = get_db()
    email_lower = (email or '').strip().lower()
    admin = db.admins.find_one({'emailLower': email_lower})
    if admin:
        return ADMIN, admin
    user = db.users.find_one({'emailLower': email_lower})
    if user:
        return END_USER, user
    return None, None


class LoginRefused(Exception):
    def __init__(self, message, reason):
        super().__init__(message)
        self.message = message
        self.reason = reason


def authenticate(email, password):
    """UC-03 at login: credentials first, then account status. Returns (role, account)."""
    role, account = find_account_by_email(email)
    if account is None:
        check_password_hash(_DUMMY_HASH, password or '')
        raise LoginRefused('Invalid email or password.', 'invalid_credentials')
    if not check_password_hash(account['passwordHash'], password or ''):
        raise LoginRefused('Invalid email or password.', 'invalid_credentials')
    if role == END_USER and account['accountStatus'] != ACTIVE:
        if account['accountStatus'] == PENDING:
            raise LoginRefused('Your account is waiting for Admin approval. You can sign in once it is approved.',
                               'account_pending')
        raise LoginRefused('Your registration was rejected by the Admin, so this account cannot sign in.',
                           'account_rejected')
    return role, account


def start_auth_session(role, account, response):
    token = secrets.token_urlsafe(32)
    now = utcnow()
    get_db().auth_sessions.insert_one({
        'tokenHash': _token_hash(token), 'principalType': role, 'principalId': account['_id'],
        'createdAt': now, 'lastSeenAt': now, 'invalidatedAt': None,
    })
    response.set_cookie(current_app.config['AUTH_COOKIE_NAME'], token, httponly=True, samesite='Lax',
                        secure=current_app.config['REQUIRE_HTTPS'])
    return response


def end_auth_session(response):
    """FR-05: invalidates the current authentication session only (a running analysis session keeps running)."""
    token = request.cookies.get(current_app.config['AUTH_COOKIE_NAME'])
    if token:
        get_db().auth_sessions.update_one({'tokenHash': _token_hash(token), 'invalidatedAt': None},
                                          {'$set': {'invalidatedAt': utcnow()}})
    response.delete_cookie(current_app.config['AUTH_COOKIE_NAME'])
    return response


def load_principal():
    """Runs before every request. Sets g.principal and g.auth_expired."""
    g.principal = None
    g.auth_expired = False
    token = request.cookies.get(current_app.config['AUTH_COOKIE_NAME'])
    if not token:
        return
    db = get_db()
    auth = db.auth_sessions.find_one({'tokenHash': _token_hash(token)})
    if auth is None or auth['invalidatedAt'] is not None:
        g.auth_expired = True
        return
    now = utcnow()
    if now - auth['lastSeenAt'] > timedelta(minutes=current_app.config['AUTH_IDLE_TIMEOUT_MINUTES']):
        db.auth_sessions.update_one({'_id': auth['_id']}, {'$set': {'invalidatedAt': now}})
        g.auth_expired = True
        return
    collection = db.admins if auth['principalType'] == ADMIN else db.users
    account = collection.find_one({'_id': auth['principalId']})
    if account is None or (auth['principalType'] == END_USER and account['accountStatus'] != ACTIVE):
        db.auth_sessions.update_one({'_id': auth['_id']}, {'$set': {'invalidatedAt': now}})
        g.auth_expired = True
        return
    if now - auth['lastSeenAt'] > timedelta(seconds=30):
        db.auth_sessions.update_one({'_id': auth['_id']}, {'$set': {'lastSeenAt': now}})
    g.principal = Principal(id=account['_id'], role=auth['principalType'], username=account['username'],
                            email=account['email'], name=account.get('name', ''))


def _refuse_unauthenticated():
    if request.path.startswith('/api/'):
        return jsonify({'error': 'unauthenticated', 'message': 'Sign in to continue.'}), 401
    if g.get('auth_expired'):
        flash('Your session has expired. Sign in again.', 'warning')
    return redirect(url_for('auth.login'))


def login_required(role=None):
    """Protects a view: an authenticated session (UC-03 1b) and, when given, the role (UC-03 3a)."""
    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            if g.get('principal') is None:
                return _refuse_unauthenticated()
            if role is not None and g.principal.role != role:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


def dashboard_endpoint(principal):
    return 'admin.dashboard' if principal.role == ADMIN else 'enduser.dashboard'


def enforce_https():
    """NFR-04: outside local development, plain HTTP is redirected to HTTPS."""
    if not current_app.config['REQUIRE_HTTPS']:
        return None
    forwarded = request.headers.get('X-Forwarded-Proto', '')
    if request.is_secure or forwarded.lower() == 'https':
        return None
    if request.method in ('GET', 'HEAD'):
        return redirect(request.url.replace('http://', 'https://', 1), code=301)
    abort(403)
