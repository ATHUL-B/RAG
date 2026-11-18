"""Load ``KEY=VALUE`` lines from a local ``.env`` file into the environment.

Secrets (e.g. ``ANTHROPIC_API_KEY``) live in ``.env`` (gitignored) or the shell, never in
configs. Existing environment variables always win, and values are never logged.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path = Path(".env")) -> list[str]:
    """Set variables from ``path`` that are not already set; return the names loaded."""
    if not path.exists():
        return []
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
