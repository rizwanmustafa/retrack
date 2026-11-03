"""Register, log in and log out (UC-01, UC-02, UC-03)."""
from flask import Blueprint, current_app, flash, g, make_response, redirect, render_template, request, url_for

from ..errors import AppError
from ..db import get_db
from ..security import (LoginRefused, Principal, authenticate, dashboard_endpoint, end_auth_session,
                        start_auth_session)
from ..services.accounts import register_end_user

bp = Blueprint('auth', __name__)


@bp.route('/')
def index():
    if g.principal is not None:
        return redirect(url_for(dashboard_endpoint(g.principal)))
    return redirect(url_for('auth.login'))


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if g.principal is not None:
        return redirect(url_for(dashboard_endpoint(g.principal)))
    error, status_message, email = None, None, ''
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        try:
            role, account = authenticate(email, request.form.get('password'))
        except LoginRefused as refused:
            if refused.reason == 'invalid_credentials':
                error = refused.message
            else:
                status_message = refused.message
            return render_template('auth/login.html', error=error, status_message=status_message,
                                   email=email), 401
        principal = Principal(account['_id'], role, account['username'], account['email'], account.get('name', ''))
        response = make_response(redirect(url_for(dashboard_endpoint(principal))))
        return start_auth_session(role, account, response)
    return render_template('auth/login.html', error=error, status_message=status_message, email=email)


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if g.principal is not None:
        return redirect(url_for(dashboard_endpoint(g.principal)))
    if request.method == 'POST':
        try:
            user = register_end_user(request.form)
        except AppError as e:
            errors = {p['field']: p['message'] for p in e.problems}
            return render_template('auth/register.html', errors=errors, form=request.form), 422
        return render_template('auth/register_done.html', user=user), 201
    return render_template('auth/register.html', errors={}, form={})


@bp.route('/logout', methods=['POST'])
def logout():
    running = None
    if g.principal is not None and not g.principal.is_admin:
        engine = current_app.extensions['retrack_engine']
        session_id = engine.active_session_id()
        if session_id and get_db().sessions.find_one({'_id': session_id, 'userId': g.principal.id}):
            running = session_id
    response = make_response(redirect(url_for('auth.login')))
    end_auth_session(response)
    if running:
        flash(f'You have signed out. Monitoring session {running} keeps running; sign in again to stop it.', 'info')
    else:
        flash('You have signed out.', 'success')
    return response
