"""Performance metrics: CAGR, volatility, Sharpe, max drawdown, turnover, hit rate."""

import numpy as np
import pandas as pd


def compute_metrics(
    rets: pd.Series,
    rf_daily: float = 0.0,
    bench_rets: pd.Series | None = None,
) -> dict[str, float]:
    """Compute performance metrics from daily returns.

    Args:
        rets: Daily return series.
        rf_daily: Daily risk-free rate (for excess return).
        bench_rets: Optional benchmark returns for hit rate (strategy vs benchmark).

    Returns:
        Dict with CAGR, Volatility, Sharpe, Max Drawdown, Hit Rate (if bench provided).
    """
    rets = rets.dropna()
    if len(rets) < 5:
        return {
            "CAGR": float("nan"),
            "Volatility": float("nan"),
            "Sharpe": float("nan"),
            "Max Drawdown": float("nan"),
        }
    excess = rets - rf_daily
    mean_daily = float(excess.mean())
    std_daily = float(excess.std())

    equity_curve = (1 + rets).cumprod()
    n_days = len(rets)
    years = n_days / 252

    cagr = float(equity_curve.iloc[-1] ** (1 / years) - 1)
    volatility = float(rets.std() * np.sqrt(252))
    sharpe = (
        (mean_daily / std_daily) * np.sqrt(252)
        if std_daily != 0
        else float("nan")
    )

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = float(drawdown.min())

    out: dict[str, float] = {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }

    if bench_rets is not None:
        common = rets.index.intersection(bench_rets.index).drop_duplicates()
        if len(common) >= 2:
            s = rets.reindex(common).ffill().dropna()
            b = bench_rets.reindex(common).ffill().dropna()
            common_idx = s.index.intersection(b.index)
            if len(common_idx) >= 2:
                s = s.loc[common_idx]
                b = b.loc[common_idx]
                hit = (s > b).sum() / len(common_idx)
                out["Hit Rate"] = float(hit)
    return out


def compute_turnover(
    weights: pd.DataFrame,
    freq: str = "ME",
) -> float:
    """Compute annualized turnover from weight changes.

    Args:
        weights: DataFrame with date index and asset columns.
        freq: Resample freq for rebalance (ME = month-end).

    Returns:
        Annualized turnover (sum of abs weight changes per period, annualized).
    """
    if weights.empty or len(weights) < 2:
        return 0.0
    w = weights.resample(freq).last().dropna(how="all")
    if len(w) < 2:
        return 0.0
    diff = w.diff().abs()
    period_turnover = diff.sum(axis=1).sum()
    n_periods = len(w) - 1
    if n_periods <= 0:
        return 0.0
    periods_per_year = 12 if freq == "ME" else 252
    annualized = period_turnover / n_periods * periods_per_year
    return float(annualized)
