import re

_PATH_RE = re.compile(r'(?:[A-Za-z]:)?(?:[\\/][^\s\\/:\'"]+){2,}[\\/]?')


def safe_message(exc, limit=300):
    """Exception text without file-system paths, for diagnostics shown in the browser (NFR-04)."""
    text = _PATH_RE.sub('<path>', str(exc)).strip()
    name = type(exc).__name__
    return f'{name}: {text[:limit]}' if text else name
