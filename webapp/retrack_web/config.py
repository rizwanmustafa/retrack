"""Application configuration, read from environment variables (see .env.example)."""
import os
import secrets

WEBAPP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(WEBAPP_DIR)


def _bool(name, default):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ('1', 'true', 'yes', 'on')


class Config:
    # 'development' or 'production'. Production requires HTTPS and an explicit secret key (NFR-04).
    ENVIRONMENT = os.environ.get('RETRACK_ENV', 'development')

    SECRET_KEY = os.environ.get('RETRACK_SECRET_KEY') or None

    MONGO_URI = os.environ.get('RETRACK_MONGO_URI', 'mongodb://localhost:27017')
    MONGO_DB = os.environ.get('RETRACK_MONGO_DB', 'retrack')

    # Browser authentication session (UC-02, UC-03)
    AUTH_COOKIE_NAME = 'retrack_auth'
    AUTH_IDLE_TIMEOUT_MINUTES = int(os.environ.get('RETRACK_AUTH_IDLE_TIMEOUT_MINUTES', '30'))
    REQUIRE_HTTPS = _bool('RETRACK_REQUIRE_HTTPS', ENVIRONMENT == 'production')

    # Model integration (NFR-14): the web layer only talks to the AIModel interface.
    AI_BACKEND = os.environ.get('RETRACK_AI_BACKEND', 'tracktacular')
    WORLDTRACK_DIR = os.environ.get('RETRACK_WORLDTRACK_DIR', os.path.join(REPO_DIR, 'WorldTrack'))
    CHECKPOINT_DIR = os.environ.get('RETRACK_CHECKPOINT_DIR', os.path.join(WEBAPP_DIR, 'instance', 'checkpoints'))
    INFERENCE_DEVICE = os.environ.get('RETRACK_INFERENCE_DEVICE', 'auto')  # auto | cuda | cpu

    # RECORDED sources available for replay (see docs/RECORDED_SOURCES.md)
    RECORDED_SOURCES_FILE = os.environ.get('RETRACK_RECORDED_SOURCES_FILE',
                                           os.path.join(WEBAPP_DIR, 'instance', 'recorded_sources.json'))

    # LIVE input: accepted frame sets waiting for inference
    LIVE_QUEUE_SIZE = int(os.environ.get('RETRACK_LIVE_QUEUE_SIZE', '8'))

    # A LIVE frame set is seven 1920x1080 images; calibration bundles are small.
    MAX_CONTENT_LENGTH = int(os.environ.get('RETRACK_MAX_UPLOAD_MB', '80')) * 1024 * 1024

    WTF_CSRF_TIME_LIMIT = None

    @classmethod
    def finalize(cls, app):
        if not app.config.get('SECRET_KEY'):
            if app.config['ENVIRONMENT'] == 'production':
                raise RuntimeError('RETRACK_SECRET_KEY must be set in production')
            app.config['SECRET_KEY'] = secrets.token_hex(32)
            app.logger.warning('RETRACK_SECRET_KEY is not set; using a random development key')
        app.config['SESSION_COOKIE_HTTPONLY'] = True
        app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
        app.config['SESSION_COOKIE_SECURE'] = app.config['REQUIRE_HTTPS']
