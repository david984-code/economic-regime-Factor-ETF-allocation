"""Benchmark return series for strategy comparison."""

import numpy as np
import pandas as pd

from src.config import RISK_ON_REGIMES, TICKERS

CASH_DAILY_YIELD = (1.045) ** (1 / 252) - 1


def compute_benchmark_returns(
    returns: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Compute benchmark return series from daily returns.

    Args:
        returns: Daily returns with ticker columns + cash.
        regime_df: Regime labels with date index, regime and optional risk_on.

    Returns:
        Dict of benchmark_name -> daily return series.
    """
    regime_df = regime_df.sort_index().reindex(returns.index).ffill()
    regime_df["regime"] = regime_df["regime"].astype(str).str.strip()

    benchmarks: dict[str, pd.Series] = {}

    # SPY buy-and-hold
    if "SPY" in returns.columns:
        benchmarks["SPY"] = returns["SPY"].copy()

    # 60/40 (SPY 60%, IEF 40%)
    if "SPY" in returns.columns and "IEF" in returns.columns:
        benchmarks["60/40"] = 0.6 * returns["SPY"] + 0.4 * returns["IEF"]

    # Equal-weight across current asset universe
    ticker_cols = [c for c in TICKERS if c in returns.columns]
    if ticker_cols:
        benchmarks["Equal_Weight"] = returns[ticker_cols].mean(axis=1)

    # Risk-on/risk-off baseline: blend equal-weight risk-on vs risk-off by regime
    # Risk-on regime: 100% equal-weight of risk-on assets (e.g. SPY, MTUM, VLUE, USMV, QUAL, IJR, VIG)
    # Risk-off regime: 100% equal-weight of risk-off assets (bonds: IEF, TLT; gold: GLD)
    risk_on_tickers = ["SPY", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
    risk_off_tickers = ["IEF", "TLT", "GLD"]
    risk_on_cols = [c for c in risk_on_tickers if c in returns.columns]
    risk_off_cols = [c for c in risk_off_tickers if c in returns.columns]

    if risk_on_cols and risk_off_cols:
        ew_on = returns[risk_on_cols].mean(axis=1)
        ew_off = returns[risk_off_cols].mean(axis=1)
        alpha = regime_df["risk_on"] if "risk_on" in regime_df.columns else np.nan
        if alpha.notna().any():
            alpha = alpha.fillna(0.5)
            alpha = alpha.clip(0.0, 1.0)
            benchmarks["Risk_On_Off"] = (1 - alpha) * ew_off + alpha * ew_on
        else:
            # Fallback: use regime label
            is_risk_on = regime_df["regime"].isin(RISK_ON_REGIMES)
            benchmarks["Risk_On_Off"] = pd.Series(
                np.where(is_risk_on.values, ew_on.values, ew_off.values),
                index=returns.index,
            )

    return benchmarks
