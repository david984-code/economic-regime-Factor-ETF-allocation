"""Performance metrics: CAGR, volatility, Sharpe, max drawdown, turnover, hit rate.

Extended metrics (compute_full_metrics):
  Sortino, Calmar, Ulcer Index, CVaR 5%,
  Beta, Alpha, Correlation, Information Ratio,
  Downside/Upside Capture vs benchmark,
  Bear Market Alpha (alpha when benchmark is down).
"""

import numpy as np
import pandas as pd

from src.config import RF_DAILY


def compute_metrics(
    rets: pd.Series,
    rf_daily: float | None = None,
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
    rf_eff = RF_DAILY if rf_daily is None else rf_daily
    excess = rets - rf_eff
    mean_daily = float(excess.mean())
    std_daily = float(excess.std())

    equity_curve = (1 + rets).cumprod()
    n_days = len(rets)
    years = n_days / 252

    cagr = float(equity_curve.iloc[-1] ** (1 / years) - 1)
    volatility = float(rets.std() * np.sqrt(252))
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else float("nan")

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


def compute_full_metrics(
    rets: pd.Series,
    rf_daily: float | None = None,
    bench_rets: pd.Series | None = None,
) -> dict[str, float]:
    """Extended metrics for risk-adjusted return and bear market analysis.

    Adds on top of compute_metrics:
      Sortino        -- excess return / downside deviation (penalises only losses)
      Calmar         -- CAGR / abs(max drawdown); higher = better recovery per unit risk
      Ulcer_Index    -- RMS of drawdowns; lower = smoother equity curve
      CVaR_5pct      -- avg daily loss in worst 5% of days (tail risk)
      Beta           -- systematic exposure to benchmark
      Alpha_Ann      -- annualised Jensen's alpha vs benchmark (risk-adjusted excess return)
      Correlation    -- daily return correlation to benchmark
      Info_Ratio     -- active return / tracking error (skill after controlling for risk)
      Downside_Cap   -- % of benchmark's down moves captured; <100% = protection
      Upside_Cap     -- % of benchmark's up moves captured
      Bear_Alpha_Ann -- annualised alpha specifically when benchmark has negative returns
                        (core metric for "protecting in down markets")
      Bear_Hit_Rate  -- % of benchmark-down days where strategy > benchmark

    Args:
        rets:       Daily return series (strategy).
        rf_daily:   Daily risk-free rate.
        bench_rets: Benchmark daily returns (e.g. SPY). Required for capture ratios.

    Returns:
        dict of all metrics.
    """
    out = compute_metrics(rets, rf_daily=rf_daily, bench_rets=bench_rets)
    rets = rets.dropna()
    if len(rets) < 5:
        return out

    rf_eff = RF_DAILY if rf_daily is None else rf_daily
    cagr = out["CAGR"]
    max_dd = out["Max Drawdown"]
    equity_curve = (1 + rets).cumprod()

    # -- Sortino: excess return / annualised downside deviation ---------------
    excess = rets - rf_eff
    downside_rets = excess[excess < 0]
    if len(downside_rets) > 1:
        downside_vol = float(downside_rets.std() * np.sqrt(252))
        ann_excess = float(excess.mean() * 252)
        out["Sortino"] = ann_excess / downside_vol if downside_vol > 0 else float("nan")
    else:
        out["Sortino"] = float("nan")

    # -- Calmar: CAGR / |max drawdown| ----------------------------------------
    out["Calmar"] = abs(cagr / max_dd) if (max_dd < 0 and not np.isnan(cagr)) else float("nan")

    # -- Ulcer Index: RMS of all drawdown depths (%) ---------------------------
    dd_pct = (equity_curve / equity_curve.cummax() - 1) * 100
    out["Ulcer_Index"] = float(np.sqrt((dd_pct**2).mean()))

    # -- CVaR 5%: mean loss on worst 5% of days --------------------------------
    q05 = rets.quantile(0.05)
    tail = rets[rets <= q05]
    out["CVaR_5pct"] = float(tail.mean()) if len(tail) > 0 else float("nan")

    # -- Max consecutive loss days ---------------------------------------------
    loss_streak = 0
    max_streak = 0
    for r in rets.values:
        if r < 0:
            loss_streak += 1
            max_streak = max(max_streak, loss_streak)
        else:
            loss_streak = 0
    out["Max_Loss_Streak"] = float(max_streak)

    # -- Benchmark-relative metrics -------------------------------------------
    if bench_rets is not None:
        common = rets.index.intersection(bench_rets.index).drop_duplicates()
        if len(common) >= 30:
            s = rets.reindex(common).dropna()
            b = bench_rets.reindex(common).dropna()
            common2 = s.index.intersection(b.index)
            s = s.loc[common2]
            b = b.loc[common2]

            if len(s) >= 30 and len(b) >= 30:
                # Beta and Jensen's Alpha
                cov_matrix = np.cov(s.values, b.values)
                bench_var = cov_matrix[1, 1]
                if bench_var > 0:
                    beta = cov_matrix[0, 1] / bench_var
                    alpha_ann = (s.mean() - beta * b.mean()) * 252
                    out["Beta"] = float(beta)
                    out["Alpha_Ann"] = float(alpha_ann)
                else:
                    out["Beta"] = float("nan")
                    out["Alpha_Ann"] = float("nan")

                # Correlation
                out["Correlation"] = float(np.corrcoef(s.values, b.values)[0, 1])

                # Information Ratio: active return / tracking error
                active = s - b
                if active.std() > 0:
                    out["Info_Ratio"] = float(active.mean() / active.std() * np.sqrt(252))
                else:
                    out["Info_Ratio"] = float("nan")

                # Downside capture (benchmark-down days only)
                down_mask = b < 0
                up_mask = b > 0
                n_down = down_mask.sum()
                n_up = up_mask.sum()

                if n_down >= 10:
                    s_dn = s[down_mask]
                    b_dn = b[down_mask]
                    # Annualised compounded return during down periods
                    s_dn_ret = float((1 + s_dn).prod() ** (252 / n_down) - 1)
                    b_dn_ret = float((1 + b_dn).prod() ** (252 / n_down) - 1)
                    out["Downside_Cap"] = (s_dn_ret / b_dn_ret) if b_dn_ret != 0 else float("nan")

                    # Bear alpha: mean daily alpha on benchmark-down days, annualised
                    out["Bear_Alpha_Ann"] = float((s_dn - b_dn).mean() * 252)
                    # Bear hit rate: % of down-market days where strategy > benchmark
                    out["Bear_Hit_Rate"] = float((s_dn > b_dn).sum() / n_down)
                    # Avg strategy return on benchmark down days
                    out["Bear_Avg_Daily_Return"] = float(s_dn.mean())
                    # Avg benchmark return on those days (shows severity)
                    out["Bear_Bench_Avg"] = float(b_dn.mean())

                if n_up >= 10:
                    s_up = s[up_mask]
                    b_up = b[up_mask]
                    s_up_ret = float((1 + s_up).prod() ** (252 / n_up) - 1)
                    b_up_ret = float((1 + b_up).prod() ** (252 / n_up) - 1)
                    out["Upside_Cap"] = (s_up_ret / b_up_ret) if b_up_ret != 0 else float("nan")

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
