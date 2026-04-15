"""Ticker universe resolution: CLI args > env var > config default."""

from __future__ import annotations

import os

from src.config import TICKERS


def parse_tickers_arg(raw: str) -> list[str] | None:
    """Parse comma-separated ticker string from CLI.

    Returns None if empty (caller should fall through to defaults).
    """
    if not raw or not raw.strip():
        return None
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def resolve_tickers(cli_tickers: list[str] | None = None) -> list[str]:
    """Resolve final ticker list: CLI > PIPELINE_TICKERS env > config."""
    if cli_tickers:
        return cli_tickers

    env_tickers = os.getenv("PIPELINE_TICKERS", "")
    if env_tickers.strip():
        return [t.strip().upper() for t in env_tickers.split(",") if t.strip()]

    return list(TICKERS)
