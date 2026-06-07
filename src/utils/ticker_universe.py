"""Ticker universe: CLI / env overrides over defaults in ``src.config``."""

from __future__ import annotations

import os
import re

from src.config import TICKERS


def parse_tickers_arg(value: str | None) -> list[str] | None:
    """Parse comma-separated tickers; return None if empty."""
    if not value or not str(value).strip():
        return None
    parts = [p.strip().upper() for p in str(value).split(",") if p.strip()]
    return parts or None


def tickers_from_env() -> list[str] | None:
    """Optional override: ``PIPELINE_TICKERS=SPY,GLD,...``."""
    raw = (os.getenv("PIPELINE_TICKERS") or "").strip()
    return parse_tickers_arg(raw)


def resolve_tickers(cli_tickers: list[str] | None) -> list[str]:
    """Prefer CLI, then env, then ``config.TICKERS``."""
    if cli_tickers:
        return _validate_tickers(cli_tickers)
    env_tickers = tickers_from_env()
    if env_tickers:
        return _validate_tickers(env_tickers)
    return list(TICKERS)


def _validate_tickers(tickers: list[str]) -> list[str]:
    out: list[str] = []
    for t in tickers:
        t = t.strip().upper()
        if not t or not re.match(r"^[A-Z][A-Z0-9.\-]{0,20}$", t):
            raise ValueError(f"Invalid ticker symbol: {t!r}")
        out.append(t)
    if not out:
        raise ValueError("Ticker list is empty")
    return out
