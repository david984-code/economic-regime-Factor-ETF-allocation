"""Local macro cache for FRED series. Incremental fetch to minimize API calls."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MACRO_CACHE_DIR

logger = logging.getLogger(__name__)


def _cache_path(series_id: str) -> Path:
    """Path to cached CSV for a series."""
    MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MACRO_CACHE_DIR / f"{series_id}.csv"


def load_cached_series(series_id: str) -> pd.Series | None:
    """Load series from local cache. Returns None if missing or empty."""
    path = _cache_path(series_id)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if df.empty or "value" not in df.columns:
            return None
        s = df["value"].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        logger.warning("[FRED cache] Failed to load %s: %s", series_id, e)
        return None


def _save_series(series_id: str, s: pd.Series) -> None:
    """Save series to local cache."""
    path = _cache_path(series_id)
    df = pd.DataFrame({"date": s.index, "value": s.values})
    df.to_csv(path, index=False)
    logger.debug("[FRED cache] Saved %s (%d rows)", series_id, len(df))


def fetch_series_with_cache(
    fred: Any,
    series_id: str,
    end_date: str,
    *,
    use_cache: bool = True,
) -> tuple[pd.Series, str]:
    """Fetch FRED series, using local cache and incremental API when possible.

    Returns:
        (series, source) where source is "cache", "api_full", or "api_incremental"
    """
    cached = load_cached_series(series_id) if use_cache else None

    if cached is not None and len(cached) > 0:
        cached_max = cached.index.max()
        if isinstance(cached_max, pd.Timestamp):
            cached_max_str = cached_max.strftime("%Y-%m-%d")
        else:
            cached_max_str = str(cached_max)[:10]
        if cached_max_str >= end_date:
            logger.info("[FRED cache] Loaded %s from cache (latest %s)", series_id, cached_max_str)
            return cached, "cache"

        fetch_start = (pd.Timestamp(cached_max_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            new_data = fred.get_series(
                series_id,
                observation_start=fetch_start,
                observation_end=end_date,
            )
            new_data.index = pd.to_datetime(new_data.index)
            if len(new_data) == 0:
                logger.info("[FRED cache] %s: cache hit (no new data)", series_id)
                return cached, "cache"
            combined = pd.concat([cached, new_data])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            _save_series(series_id, combined)
            logger.info(
                "[FRED cache] %s: api_incremental (%d new obs, total %d)",
                series_id,
                len(new_data),
                len(combined),
            )
            return combined, "api_incremental"
        except Exception as e:
            logger.warning("[FRED cache] Incremental fetch failed for %s: %s. Falling back to full fetch.", series_id, e)

    s = fred.get_series(series_id, observation_end=end_date)
    s.index = pd.to_datetime(s.index)
    if use_cache:
        _save_series(series_id, s)
    logger.info("[FRED cache] %s: api_full (%d rows)", series_id, len(s))
    return s, "api_full"
