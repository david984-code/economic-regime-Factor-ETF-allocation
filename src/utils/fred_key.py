"""Resolve FRED API key from environment or file (never hardcoded)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PLACEHOLDER_KEYS = frozenset(
    {
        "your_fred_api_key_here",
        "your_api_key_here",
        "<fred_api_key_here>",
        "changeme",
        "replace_me",
        "placeholder",
        "xxx",
        "xxx...",
    }
)
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")


def validate_fred_key_format(key: str | None) -> None:
    """Validate a resolved FRED API key before use."""
    if key is None or not str(key).strip():
        raise ValueError("FRED_API_KEY is missing or empty")

    normalized = str(key).strip().strip('"').strip("'")
    lowered = normalized.lower()

    if lowered in _PLACEHOLDER_KEYS or lowered.startswith("your_") or "<" in lowered:
        raise ValueError("FRED_API_KEY is a placeholder, not a real key")

    if len(normalized) != 32:
        raise ValueError(f"FRED_API_KEY must be exactly 32 characters (got {len(normalized)})")

    if not _HEX32_RE.match(normalized):
        logging.warning("FRED_API_KEY does not match expected lowercase hex format (32 chars)")


def get_fred_api_key() -> str | None:
    """Return FRED API key if configured.

    Resolution order:
    1. Environment variable ``FRED_API_KEY`` (after ``load_dotenv()``).
    2. File path in ``FRED_API_KEY_FILE`` — file should contain the key only (one line).

    In production you can inject the key via CI, Vault, or any secrets manager by
    setting ``FRED_API_KEY`` in the process environment.
    """
    load_dotenv()
    direct = (os.getenv("FRED_API_KEY") or "").strip()
    if direct:
        validate_fred_key_format(direct)
        return direct
    key_file = (os.getenv("FRED_API_KEY_FILE") or "").strip()
    if not key_file:
        return None
    path = Path(key_file).expanduser()
    if not path.is_file():
        return None
    file_key = path.read_text(encoding="utf-8").strip() or None
    if file_key is not None:
        validate_fred_key_format(file_key)
    return file_key
