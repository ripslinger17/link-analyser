"""Load environment variables from the project .env file."""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_loaded = False


def load_env() -> None:
    """Read .env once without overriding variables already set in the shell."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def get_env(name: str, default: str = "") -> str:
    load_env()
    return os.getenv(name, default)
