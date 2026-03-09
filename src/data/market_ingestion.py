"""Market data ingestion via yfinance."""

import logging
from typing import Sequence

import pandas as pd
import yfinance as yf

from src.config import START_DATE, TICKERS, get_end_date

logger = logging.getLogger(__name__)


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
) -> pd.DataFrame:
    """Fetch daily price data from yfinance.

    Args:
        tickers: List of ticker symbols. Defaults to config.TICKERS.
        start: Start date (YYYY-MM-DD). Defaults to config.START_DATE.
        end: End date. Defaults to today.

    Returns:
        DataFrame with date index and one column per ticker (Adj Close).
    """
    syms = list(tickers) if tickers else TICKERS
    s = start or START_DATE
    e = end or get_end_date()

    raw = yf.download(syms, start=s, end=e, progress=False, auto_adjust=False)
    prices = _extract_close_prices(raw)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.dropna()
    logger.info("Fetched prices for %d tickers, %d rows", len(prices.columns), len(prices))
    return prices


def fetch_monthly_returns(
    tickers: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch monthly returns (month-end).

    Args:
        tickers: Ticker symbols. Defaults to config.TICKERS.
        start: Start date.
        end: End date.

    Returns:
        DataFrame with month-end index and return columns.
    """
    prices = fetch_prices(tickers=tickers, start=start, end=end)
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna()
    returns.index.name = "Date"
    return returns
