"""Shared market data layer for pipeline steps.

Fetches prices once per run and provides derived views (monthly returns,
momentum features) to avoid repeated yfinance calls.
"""

import logging
import time
from typing import Sequence

import pandas as pd

from src.config import START_DATE, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices

logger = logging.getLogger(__name__)


class PipelineData:
    """Centralized market data cache for a single pipeline run.

    Fetches daily prices once; exposes getters for prices, monthly returns,
    and momentum features. All downstream steps reuse the same data.
    """

    def __init__(
        self,
        tickers: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        self._tickers = list(tickers) if tickers else list(TICKERS)
        self._start = start or START_DATE
        self._end = end or get_end_date()
        self._prices: pd.DataFrame | None = None
        self._fetch_time_sec: float | None = None
        self._source = "cache"  # "fetch" or "cache" for logging

    def get_prices(self) -> pd.DataFrame:
        """Return daily prices. Fetches once on first call, then returns cached."""
        if self._prices is None:
            t0 = time.perf_counter()
            self._prices = fetch_prices(
                tickers=self._tickers,
                start=self._start,
                end=self._end,
            )
            self._fetch_time_sec = time.perf_counter() - t0
            self._source = "fetch"
            logger.info(
                "[DATA] Fetched prices: %d tickers, %d rows in %.2fs (source: yfinance)",
                len(self._prices.columns),
                len(self._prices),
                self._fetch_time_sec,
            )
        else:
            logger.debug(
                "[DATA] Reusing cached prices: %d tickers, %d rows (cache hit)",
                len(self._prices.columns),
                len(self._prices),
            )
        return self._prices

    def set_prices(self, prices: pd.DataFrame) -> None:
        """Inject pre-fetched prices (e.g. for testing). Marks as cache hit."""
        self._prices = prices.copy()
        self._fetch_time_sec = 0.0
        self._source = "injected"
        logger.debug("[DATA] Injected prices: %d tickers, %d rows", len(prices.columns), len(prices))

    def get_monthly_returns(self) -> pd.DataFrame:
        """Return monthly returns (month-end). Same logic as fetch_monthly_returns."""
        prices = self.get_prices()
        monthly = prices.resample("ME").last()
        returns = monthly.pct_change().dropna()
        returns.index.name = "Date"
        logger.debug(
            "[DATA] get_monthly_returns: derived from cached prices (%d rows)",
            len(returns),
        )
        return returns

    def get_momentum_features(
        self,
        ticker: str = "SPY",
    ) -> pd.DataFrame:
        """Return 1/3/6 month momentum for a ticker. Same logic as build_momentum_features."""
        prices = self.get_prices()
        if ticker not in prices.columns:
            raise ValueError(f"Ticker {ticker} not in cached prices: {list(prices.columns)}")
        px = prices[ticker]
        monthly = px.resample("ME").last()
        ret_1m = monthly.pct_change(1)
        ret_3m = monthly.pct_change(3)
        ret_6m = monthly.pct_change(6)
        df = pd.DataFrame(
            {"spy_1m": ret_1m, "spy_3m": ret_3m, "spy_6m": ret_6m},
            index=monthly.index,
        )
        df.index = df.index.to_period("M").to_timestamp("M")
        logger.debug(
            "[DATA] get_momentum_features(%s): derived from cached prices (%d rows)",
            ticker,
            len(df),
        )
        return df

    @property
    def fetch_time_sec(self) -> float | None:
        """Time spent on initial fetch (None if not yet fetched)."""
        return self._fetch_time_sec

    @property
    def source(self) -> str:
        """'fetch', 'cache', or 'injected'."""
        return self._source
