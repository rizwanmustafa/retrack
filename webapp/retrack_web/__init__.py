"""Retrack web application (Flask + MongoDB + Bootstrap)."""
from flask import Flask, g

from . import scene
from .config import Config
from .db import init_db
from .errors import register_error_handlers
from .security import enforce_https, load_principal
from .timeutil import iso


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)
    Config.finalize(app)

    init_db(app)

    from flask_wtf.csrf import CSRFProtect
    csrf = CSRFProtect(app)

    from .ai import create_ai_model
    app.extensions['retrack_ai'] = app.config.get('AI_MODEL') or create_ai_model(app.config)

    from .monitoring.engine import MonitoringEngine
    engine = MonitoringEngine(app)
    app.extensions['retrack_engine'] = engine

    from .services.calibrations import ensure_rig
    with app.app_context():
        ensure_rig(app)
        engine.recover_interrupted()

    @app.before_request
    def before():
        redirect_response = enforce_https()
        if redirect_response is not None:
            return redirect_response
        load_principal()

    @app.after_request
    def security_headers(response):
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Referrer-Policy', 'same-origin')
        if g.get('principal') is not None:
            response.headers.setdefault('Cache-Control', 'no-store')
        if app.config['REQUIRE_HTTPS']:
            response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000')
        return response

    @app.context_processor
    def template_globals():
        return {'principal': g.get('principal'), 'scene': scene}

    app.jinja_env.filters['iso'] = iso

    from .blueprints import admin, api, auth, enduser, reports
    app.register_blueprint(auth.bp)
    app.register_blueprint(enduser.bp)
    app.register_blueprint(admin.bp)
    app.register_blueprint(reports.bp)
    app.register_blueprint(api.bp)
    csrf.exempt(api.ingest_frame_set)  # the camera rig authenticates with the session's ingest token

    register_error_handlers(app)

    from .cli import register_cli
    register_cli(app)
    return app
