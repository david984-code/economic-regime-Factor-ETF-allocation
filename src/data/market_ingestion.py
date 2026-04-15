"""Market data ingestion via yfinance with local parquet caching."""

import hashlib
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import PROJECT_ROOT, START_DATE, TICKERS, get_end_date
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

PRICE_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
PRICE_STALENESS_HOURS = 24


def _cache_path(tickers: list[str], start: str) -> Path:
    """Deterministic cache path based on ticker set and start date."""
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{sorted(tickers)}_{start}".encode()).hexdigest()[:10]
    return PRICE_CACHE_DIR / f"prices_{key}.parquet"


def _is_stale(path: Path, max_age_hours: int = PRICE_STALENESS_HOURS) -> bool:
    """True if file doesn't exist or is older than max_age_hours."""
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > timedelta(hours=max_age_hours)


def _extract_close_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Extract Adj Close (or Close) from yfinance result. Handles MultiIndex."""
    if isinstance(px.columns, pd.MultiIndex):
        if "Adj Close" in px.columns.levels[0]:
            out = px["Adj Close"].copy()
        else:
            out = px["Close"].copy()
    else:
        out = px["Adj Close"] if "Adj Close" in px.columns else px["Close"]
        if isinstance(out, pd.Series):
            out = out.to_frame()
    return out


def fetch_prices(
    tickers: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily price data, using local parquet cache when fresh.

    Cache staleness: re-downloads if cache file is older than 24 hours.
    Network failures retry 3 times with exponential backoff (1s, 2s, 4s).

    Args:
        tickers: List of ticker symbols. Defaults to config.TICKERS.
        start: Start date (YYYY-MM-DD). Defaults to config.START_DATE.
        end: End date. Defaults to today.
        use_cache: If False, bypass cache and fetch fresh data.

    Returns:
        DataFrame with date index and one column per ticker (Adj Close).
    """
    syms = list(tickers) if tickers else TICKERS
    s = start or START_DATE
    e = end or get_end_date()

    cache_file = _cache_path(syms, s)

    if use_cache and not _is_stale(cache_file):
        try:
            prices = pd.read_parquet(cache_file)
            logger.info(
                "[CACHE HIT] Loaded prices from %s (%d tickers, %d rows)",
                cache_file.name,
                len(prices.columns),
                len(prices),
            )
            return prices
        except Exception as exc:
            logger.warning(
                "[CACHE] Failed to read %s: %s — refetching", cache_file, exc
            )

    def _download() -> pd.DataFrame:
        raw = yf.download(syms, start=s, end=e, progress=False, auto_adjust=False)
        result = _extract_close_prices(raw)
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)
        if isinstance(result, pd.Series):
            result = result.to_frame()
        return result.dropna()

    prices = retry_with_backoff(
        _download,
        max_attempts=3,
        base_wait=1.0,
        exceptions=(Exception,),
    )

    if use_cache:
        try:
            prices.to_parquet(cache_file)
            logger.info(
                "[CACHE WRITE] Saved prices to %s (%d tickers, %d rows)",
                cache_file.name,
                len(prices.columns),
                len(prices),
            )
        except Exception as exc:
            logger.warning("[CACHE] Failed to write %s: %s", cache_file, exc)

    logger.info(
        "Fetched prices for %d tickers, %d rows (source: yfinance)",
        len(prices.columns),
        len(prices),
    )
    return prices


def fetch_monthly_returns(
    tickers: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch monthly returns (month-end).

    Args:
        tickers: Ticker symbols. Defaults to config.TICKERS.
        start: Start date.
        end: End date.
        use_cache: If False, bypass price cache.

    Returns:
        DataFrame with month-end index and return columns.
    """
    prices = fetch_prices(tickers=tickers, start=start, end=end, use_cache=use_cache)
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna()
    returns.index.name = "Date"
    return returns
