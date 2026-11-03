"""Entry point: `flask --app wsgi run` (development) or a WSGI server with ONE worker process (see README)."""
from retrack_web import create_app

app = create_app()
