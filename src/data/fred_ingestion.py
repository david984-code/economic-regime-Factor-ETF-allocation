"""FRED API data ingestion for macroeconomic series.

Includes:
- In-memory cache by (series_id, end_date) for same-run reuse
- Local file cache (macro_cache/) for cross-run persistence and incremental fetch
"""

import logging
from typing import Any

import pandas as pd

from src.config import FRED_SERIES_OPTIONAL, get_end_date
from src.data.fred_cache import fetch_series_with_cache

logger = logging.getLogger(__name__)

# In-memory cache: (series_id, end_date) -> pd.Series (same-run only)
_FRED_CACHE: dict[tuple[str, str], pd.Series] = {}
_FRED_CACHE_HITS = 0
_FRED_CACHE_MISSES = 0


def _get_fred_series_cached(
    fred: Any,
    series_id: str,
    end_date: str,
) -> pd.Series:
    """Fetch FRED series with cache. Logs hit/miss."""
    global _FRED_CACHE_HITS, _FRED_CACHE_MISSES
    key = (series_id, end_date)
    if key in _FRED_CACHE:
        _FRED_CACHE_HITS += 1
        s = _FRED_CACHE[key].copy()
        logger.debug("[FRED] Cache HIT: %s (end=%s)", series_id, end_date)
        return s
    _FRED_CACHE_MISSES += 1
    s = fred.get_series(series_id, observation_end=end_date)
    s.index = pd.to_datetime(s.index)
    _FRED_CACHE[key] = s.copy()
    logger.debug(
        "[FRED] Cache MISS: %s (end=%s) - fetched from API", series_id, end_date
    )
    return s


def get_fred_cache_stats() -> tuple[int, int]:
    """Return (hits, misses) for FRED cache."""
    return _FRED_CACHE_HITS, _FRED_CACHE_MISSES


def clear_fred_cache() -> None:
    """Clear FRED cache (e.g. between runs for testing)."""
    global _FRED_CACHE, _FRED_CACHE_HITS, _FRED_CACHE_MISSES
    _FRED_CACHE = {}
    _FRED_CACHE_HITS = 0
    _FRED_CACHE_MISSES = 0
    logger.debug("[FRED] Cache cleared")


def fetch_fred_core(api_key: str, end_date: str | None = None) -> tuple[pd.Series, ...]:
    """Fetch core FRED series: GDP, CPI, 10Y, 3M, M2, M2V. Full API fetch (no local cache)."""
    return _fetch_fred_core_impl(api_key, end_date, use_local_cache=False)


def fetch_fred_core_cached(
    api_key: str, end_date: str | None = None
) -> tuple[pd.Series, ...]:
    """Fetch core FRED series with local cache and incremental API when possible."""
    return _fetch_fred_core_impl(api_key, end_date, use_local_cache=True)


def _fetch_fred_core_impl(
    api_key: str,
    end_date: str | None,
    use_local_cache: bool,
) -> tuple[pd.Series, ...]:
    """Internal: fetch core series, optionally using local cache."""
    from fredapi import Fred

    end = end_date or get_end_date()
    fred = Fred(api_key=api_key)

    core_series = [
        ("GDP", "gdp"),
        ("CPIAUCSL", "cpi"),
        ("DGS10", "yield_10y"),
        ("DGS3MO", "yield_3m"),
        ("M2SL", "m2"),
        ("M2V", "velocity"),
    ]

    try:
        result = []
        for series_id, _ in core_series:
            if use_local_cache:
                s, _ = fetch_series_with_cache(fred, series_id, end, use_cache=True)
            else:
                s = _get_fred_series_cached(fred, series_id, end)
            result.append(s)
        gdp, cpi, yield_10y, yield_3m, m2, velocity = result
    except Exception as e:
        logger.exception("FRED fetch failed")
        raise ValueError(f"FRED API error: {e}") from e

    logger.info(
        "FRED core: GDP %s, CPI %s, 10Y %s, 3M %s, M2 %s, V %s",
        gdp.index.max(),
        cpi.index.max(),
        yield_10y.index.max(),
        yield_3m.index.max(),
        m2.index.max(),
        velocity.index.max(),
    )
    return gdp, cpi, yield_10y, yield_3m, m2, velocity


def fetch_fred_optional(
    api_key: str, end_date: str | None = None
) -> dict[str, pd.Series]:
    """Fetch optional FRED series. Full API fetch (no local cache)."""
    return _fetch_fred_optional_impl(api_key, end_date, use_local_cache=False)


def fetch_fred_optional_cached(
    api_key: str, end_date: str | None = None
) -> dict[str, pd.Series]:
    """Fetch optional FRED series with local cache and incremental API."""
    return _fetch_fred_optional_impl(api_key, end_date, use_local_cache=True)


def _fetch_fred_optional_impl(
    api_key: str,
    end_date: str | None,
    use_local_cache: bool,
) -> dict[str, pd.Series]:
    """Internal: fetch optional series, optionally using local cache."""
    from fredapi import Fred

    end = end_date or get_end_date()
    fred = Fred(api_key=api_key)
    out: dict[str, pd.Series] = {}

    for name, series_id in FRED_SERIES_OPTIONAL.items():
        try:
            if use_local_cache:
                s, src = fetch_series_with_cache(fred, series_id, end, use_cache=True)
            else:
                s = _get_fred_series_cached(fred, series_id, end)
                src = "api"
            if len(s) > 12:
                out[name] = s
                logger.info(
                    "FRED optional: %s (%s) latest %s [%s]",
                    name,
                    series_id,
                    s.index.max(),
                    src,
                )
        except Exception as e:
            logger.debug("FRED %s (%s) failed: %s", name, series_id, e)

    if "pmi" not in out and "indpro" in out:
        out["pmi"] = out.pop("indpro")
        logger.info("Using INDPRO as PMI proxy (NAPM unavailable)")
    elif "indpro" in out:
        del out["indpro"]

    return out
