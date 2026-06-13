"""Vintage (point-in-time) FRED data ingestion.

FRED's standard `get_series` returns the LATEST REVISED values, which is
look-ahead-tainted for backtesting: at training date T, the model sees
values that have since been retroactively revised to reflect future
information (e.g. Q1 2020 GDP was reported at +1.7% YoY at the time and is
now -1.4% YoY in the latest revision, so a model fit on revised data "knew"
the COVID contraction was coming).

This module fetches vintage release histories via ALFRED's
`realtime_start` / `realtime_end` semantics and exposes a point-in-time
view: for each (series, as_of_date) pair, the values are exactly what
FRED would have shown on that date.

Caches release histories to a single parquet per series so the full
walk-forward backtest only needs one API call per series rather than
one per (series, month) pair.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)

VINTAGE_CACHE_DIR = OUTPUTS_DIR / "macro_cache" / "vintage"


def _cache_path(series_id: str) -> Path:
    VINTAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return VINTAGE_CACHE_DIR / f"{series_id}.parquet"


def fetch_all_releases(api_key: str, series_id: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch the full release history for a FRED series.

    Returns:
        DataFrame with columns: 'date' (observation period),
        'realtime_start' (when this value was first published or revised),
        'value' (the value as of that realtime).

        Multi-row per observation date when the value was revised; the
        row with the latest realtime_start <= as_of is the value that
        would have been known on as_of.
    """
    cache_p = _cache_path(series_id)
    if use_cache and cache_p.exists():
        df = pd.read_parquet(cache_p)
        logger.debug("Vintage cache HIT: %s (%d rows)", series_id, len(df))
        return df

    from fredapi import Fred
    fred = Fred(api_key=api_key)
    logger.info("Fetching ALFRED vintage history for %s ...", series_id)
    df = fred.get_series_all_releases(series_id)
    df = df.rename(columns={"date": "date", "realtime_start": "realtime_start", "value": "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df = df.sort_values(["date", "realtime_start"]).reset_index(drop=True)
    df.to_parquet(cache_p)
    logger.info("  cached %d release rows to %s", len(df), cache_p)
    return df


def vintage_series_as_of(releases: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Construct the series as it would have appeared on `as_of`.

    For each observation date D, keeps the latest released value
    with realtime_start <= as_of. Returns a Series indexed by D.
    """
    as_of = pd.Timestamp(as_of)
    eligible = releases[releases["realtime_start"] <= as_of]
    if eligible.empty:
        return pd.Series(dtype=float)
    latest_per_date = eligible.sort_values("realtime_start").drop_duplicates(
        subset="date", keep="last"
    )
    out = latest_per_date.set_index("date")["value"].astype(float).sort_index()
    return out


def fetch_vintage_core(
    api_key: str,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch release histories for the core macro series.

    Returns dict series_name -> release-history DataFrame.
    """
    series_map = {
        "gdp": "GDPC1",          # real GDP (chained 2017 $); ALFRED-supported with vintage
        "cpi": "CPIAUCSL",
        "yield_10y": "DGS10",
        "yield_3m": "DGS3MO",
        "m2": "M2SL",
        "velocity": "M2V",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, sid in series_map.items():
        try:
            out[name] = fetch_all_releases(api_key, sid, use_cache=use_cache)
        except Exception as e:
            logger.warning("Failed to fetch vintage %s (%s): %s. Falling back.", name, sid, e)
    return out


def fetch_vintage_optional(
    api_key: str,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch release histories for optional / high-frequency series."""
    series_map = {
        "pmi": "NAPM",
        "claims": "ICSA",
        "hy_spread": "BAMLH0A0HYM2",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, sid in series_map.items():
        try:
            out[name] = fetch_all_releases(api_key, sid, use_cache=use_cache)
        except Exception as e:
            logger.warning("Failed to fetch vintage %s (%s): %s.", name, sid, e)
    return out
