"""Market feature engineering: momentum, seasonality."""

import pandas as pd

from src.config import START_DATE, get_end_date
from src.data.market_ingestion import fetch_prices


def build_momentum_features(
    ticker: str = "SPY",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Build 1/3/6 month momentum features for a ticker."""
    s = start or START_DATE
    e = end or get_end_date()
    prices = fetch_prices(tickers=[ticker], start=s, end=e)
    if len(prices.columns) == 1:
        px = prices.iloc[:, 0]
    else:
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
    return df


def build_seasonality_features(regime_df: pd.DataFrame) -> pd.Series:
    """Month-of-year seasonality: average historical risk_on by month."""
    regime_df = regime_df.copy()
    regime_df["month"] = pd.to_datetime(regime_df["date"]).dt.month
    return regime_df.groupby("month")["risk_on"].mean()
