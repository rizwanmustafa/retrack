"""
Error responses (FR intro, NFR-01, NFR-04): an error names the failed condition and, where there is one, the
affected camera, field or resource. Server paths, stack traces and secrets are never returned to the client.
"""
from flask import jsonify, render_template, request
from werkzeug.exceptions import HTTPException


class AppError(Exception):
    """A refused or failed operation with a client-safe message."""

    def __init__(self, message, status=400, code='invalid_request', field=None, problems=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.field = field
        self.problems = problems or []

    def to_dict(self):
        body = {'error': self.code, 'message': self.message}
        if self.field:
            body['field'] = self.field
        if self.problems:
            body['problems'] = self.problems
        return body


def wants_json():
    return request.path.startswith('/api/') or request.accept_mimetypes.best == 'application/json'


def register_error_handlers(app):
    @app.errorhandler(AppError)
    def handle_app_error(err):
        if wants_json():
            return jsonify(err.to_dict()), err.status
        return render_template('errors/error.html', status=err.status, title='Request refused',
                               message=err.message), err.status

    @app.errorhandler(HTTPException)
    def handle_http_error(err):
        messages = {
            400: 'The request was not valid.',
            403: 'You do not have permission for this operation.',
            404: 'The requested page or resource does not exist.',
            405: 'This operation is not allowed here.',
            413: 'The uploaded data is larger than the configured limit.',
        }
        message = messages.get(err.code, err.description if err.code and err.code < 500 else 'Request failed.')
        if err.code == 400 and 'CSRF' in (err.description or ''):
            message = 'The form has expired. Reload the page and try again.'
        if wants_json():
            return jsonify({'error': err.name.lower().replace(' ', '_'), 'message': message}), err.code
        return render_template('errors/error.html', status=err.code, title=err.name, message=message), err.code

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        # Full details go to the server log only (NFR-04)
        app.logger.exception('Unhandled error on %s %s', request.method, request.path)
        message = 'An internal error occurred. The administrator can find the details in the server log.'
        if wants_json():
            return jsonify({'error': 'internal_error', 'message': message}), 500
        return render_template('errors/error.html', status=500, title='Internal error', message=message), 500
