from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> bool:
    """Load environment variables from a .env file if it exists.

    Values already set in the process environment are left untouched.
    """
    env_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

    return True
