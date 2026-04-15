"""FRED API key resolution with clear error messages."""

import os
from pathlib import Path


def get_fred_api_key() -> str | None:
    """Return FRED API key from env var or key file.

    Resolution order:
        1. FRED_API_KEY environment variable
        2. File path in FRED_API_KEY_FILE environment variable
    """
    key = os.getenv("FRED_API_KEY")
    if key:
        return key.strip()

    key_file = os.getenv("FRED_API_KEY_FILE")
    if key_file:
        path = Path(key_file)
        if path.is_file():
            return path.read_text().strip()

    return None
