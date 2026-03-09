"""Backtest engine: regime-based allocation with vol scaling and transaction costs."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from src.allocation.vol_scaling import vol_scaled_weights, vol_scaled_weights_from_std
from src.config import (
    ASSETS,
    COST_BPS,
    REGIME_ALIASES,
    RISK_OFF_REGIMES,
    RISK_ON_REGIMES,
    TICKERS,
    VOL_LOOKBACK,
    get_end_date,
)
from src.data.market_ingestion import fetch_prices
from src.utils.database import Database
from src.backtest.metrics import compute_metrics

if TYPE_CHECKING:
    from src.data.pipeline_data import PipelineData

logger = logging.getLogger(__name__)

CASH_DAILY_YIELD = (1.045) ** (1 / 252) - 1


def _avg_alloc(
    allocations: dict[str, dict[str, float]],
    regimes: set[str],
    assets: list[str],
) -> dict[str, float]:
    """Average allocation across regimes."""
    regs = [r for r in regimes if r in allocations]
    if not regs:
        raise ValueError(f"None of {regimes} found in allocations")
    out: dict[str, float] = {a: 0.0 for a in assets}
    for r in regs:
        for a in assets:
            out[a] += float(allocations[r].get(a, 0.0))
    for a in assets:
        out[a] /= len(regs)
    return out


def _blend_alloc(
    w_off: dict[str, float],
    w_on: dict[str, float],
    alpha: float,
    assets: list[str],
) -> dict[str, float]:
    """Blend risk-off and risk-on allocations by alpha (0=off, 1=on)."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    w = {
        a: (1.0 - alpha) * float(w_off.get(a, 0.0))
        + alpha * float(w_on.get(a, 0.0))
        for a in assets
    }
    s = sum(w.values())
    if s <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    return {a: v / s for a, v in w.items()}


def _compute_returns_and_setup(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float], pd.Series]:
    """Shared setup: returns, regime alignment, w_risk_on/off, equal_weight_returns.
    Matches original polars-based returns computation."""
    returns = prices[TICKERS].pct_change().iloc[1:]
    returns["cash"] = CASH_DAILY_YIELD
    regime_df = regime_df.sort_index()
    if regime_df.index.duplicated().any():
        regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
    regime_df = regime_df.reindex(returns.index).ffill()
    regime_df = regime_df.reindex(returns.index)
    regime_df["regime"] = regime_df["regime"].astype(str).str.strip()

    w_risk_on = _avg_alloc(allocations, RISK_ON_REGIMES, ASSETS)
    w_risk_off = _avg_alloc(allocations, RISK_OFF_REGIMES, ASSETS)
    equal_weight_returns = returns[TICKERS].mean(axis=1)
    return returns, regime_df, w_risk_on, w_risk_off, equal_weight_returns


def _run_backtest_loop(
    returns: pd.DataFrame,
    regime_df: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
    w_risk_on: dict[str, float],
    w_risk_off: dict[str, float],
    equal_weight_returns: pd.Series,
    use_stagflation_override: bool = True,
    use_stagflation_risk_on_cap: bool = False,
    stagflation_risk_on_cap: float = 0.2,
) -> pd.Series:
    """Original per-date loop. Kept for parity validation."""
    portfolio_returns_list: list[float] = []
    prev_month: pd.Period | None = None
    current_weights: dict[str, float] = {
        a: 1.0 / (len(TICKERS) + 1) for a in TICKERS
    }
    current_weights["cash"] = 1.0 / (len(TICKERS) + 1)
    prev_weights_for_cost = dict(current_weights)

    for date in returns.index:
        regime = regime_df.loc[date, "regime"]
        if pd.isna(regime):
            portfolio_returns_list.append(np.nan)
            continue

        rebalanced = False
        month = date.to_period("M")
        if prev_month is None or month != prev_month:
            regime_stripped = str(regime).strip()
            if use_stagflation_override and regime_stripped == "Stagflation" and "Stagflation" in allocations:
                current_weights = {str(k): float(v) for k, v in allocations["Stagflation"].items()}
            elif "risk_on" in regime_df.columns and not pd.isna(regime_df.loc[date, "risk_on"]):
                alpha = float(regime_df.loc[date, "risk_on"])
                if use_stagflation_risk_on_cap and regime_stripped == "Stagflation":
                    alpha = min(alpha, stagflation_risk_on_cap)
                current_weights = _blend_alloc(w_risk_off, w_risk_on, alpha, ASSETS)
            else:
                rk = REGIME_ALIASES.get(regime_stripped, regime_stripped)
                if rk in allocations:
                    current_weights = {
                        str(k): float(v) for k, v in allocations[rk].items()
                    }
                else:
                    logger.warning("Unknown regime '%s' on %s", regime, date.date())

            risky = [a for a in TICKERS if a in current_weights]
            trailing_pd = returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)
            trailing_pl = pl.from_pandas(trailing_pd)
            current_weights = vol_scaled_weights(current_weights, trailing_pl, risky)
            prev_month = month
            rebalanced = True

        daily_ret = sum(
            returns.loc[date, a] * float(current_weights.get(a, 0.0))
            for a in ASSETS
        )
        if rebalanced:
            turnover = sum(
                abs(float(current_weights.get(a, 0.0)) - float(prev_weights_for_cost.get(a, 0.0)))
                for a in ASSETS
            )
            daily_ret -= turnover * COST_BPS
            prev_weights_for_cost = dict(current_weights)
        portfolio_returns_list.append(daily_ret)

    return pd.Series(portfolio_returns_list, index=returns.index)


def _run_backtest_vectorized(
    returns: pd.DataFrame,
    regime_df: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
    w_risk_on: dict[str, float],
    w_risk_off: dict[str, float],
    equal_weight_returns: pd.Series,
    return_weights: bool = False,
    use_stagflation_override: bool = True,
    use_stagflation_risk_on_cap: bool = False,
    stagflation_risk_on_cap: float = 0.2,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Vectorized backtest: precompute vols, build weight matrix, vectorized returns."""
    rolling_std = returns[TICKERS].rolling(VOL_LOOKBACK, min_periods=1).std()

    dates = returns.index
    months = pd.Series(dates).dt.to_period("M").values
    month_changed = np.concatenate([[True], months[1:] != months[:-1]])

    weight_cols = [a for a in ASSETS if a in returns.columns]
    weights = np.zeros((len(dates), len(weight_cols)))
    asset_idx = {a: i for i, a in enumerate(weight_cols)}

    eq_w = 1.0 / (len(TICKERS) + 1)
    current = {a: eq_w for a in TICKERS}
    current["cash"] = eq_w
    prev_weights = np.array([current.get(a, 0.0) for a in weight_cols])

    for i, date in enumerate(dates):
        regime = regime_df.loc[date, "regime"]
        if pd.isna(regime):
            weights[i] = prev_weights
            continue

        if month_changed[i]:
            regime_stripped = str(regime).strip()
            if use_stagflation_override and regime_stripped == "Stagflation" and "Stagflation" in allocations:
                current = {str(k): float(v) for k, v in allocations["Stagflation"].items()}
            elif "risk_on" in regime_df.columns and not pd.isna(regime_df.loc[date, "risk_on"]):
                alpha = float(regime_df.loc[date, "risk_on"])
                if use_stagflation_risk_on_cap and regime_stripped == "Stagflation":
                    alpha = min(alpha, stagflation_risk_on_cap)
                current = _blend_alloc(w_risk_off, w_risk_on, alpha, ASSETS)
            else:
                rk = REGIME_ALIASES.get(regime_stripped, regime_stripped)
                if rk in allocations:
                    current = {str(k): float(v) for k, v in allocations[rk].items()}
                else:
                    current = {a: 1.0 / len(ASSETS) for a in ASSETS}

            std_row = rolling_std.loc[date]
            std_dict = {a: float(std_row[a]) if a in std_row.index and pd.notna(std_row[a]) else None for a in TICKERS}
            current = vol_scaled_weights_from_std(current, std_dict, list(TICKERS))
            prev_weights = np.array([current.get(a, 0.0) for a in weight_cols])

        weights[i] = prev_weights

    ret_arr = returns[weight_cols].values
    portfolio_ret = (ret_arr * weights).sum(axis=1)

    turnover_cost = np.zeros(len(dates))
    prev_w = np.full(len(weight_cols), eq_w)
    for i in range(len(dates)):
        if month_changed[i]:
            turnover_cost[i] = np.abs(weights[i] - prev_w).sum() * COST_BPS
            prev_w = weights[i].copy()
    portfolio_ret -= turnover_cost

    nan_regime = regime_df["regime"].isna()
    portfolio_ret[nan_regime.values] = np.nan

    if return_weights:
        weights_df = pd.DataFrame(weights, index=dates, columns=weight_cols)
        return pd.Series(portfolio_ret, index=dates), weights_df
    return pd.Series(portfolio_ret, index=dates)


def run_backtest_with_allocations(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
    return_weights: bool = False,
    use_stagflation_override: bool = True,
    use_stagflation_risk_on_cap: bool = False,
    stagflation_risk_on_cap: float = 0.2,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Run backtest with given data. No DB. Returns portfolio returns (and optionally weights).

    Used by walk-forward evaluation to test on out-of-sample periods.

    Args:
        prices: Daily prices (date index, ticker columns).
        regime_df: Regime labels with date index.
        allocations: Regime -> {asset: weight}.
        return_weights: If True, return (returns, weights_df).
        use_stagflation_override: If True, use optimizer Stagflation allocation when regime==Stagflation.
        use_stagflation_risk_on_cap: If True, cap risk_on at stagflation_risk_on_cap when regime==Stagflation.
        stagflation_risk_on_cap: Max risk_on when regime==Stagflation (e.g. 0.2).

    Returns:
        Daily portfolio return series, or (returns, weights_df) if return_weights.
    """
    for alloc in allocations.values():
        if "cash" not in alloc:
            alloc["cash"] = 0.0
    returns, regime_df, w_risk_on, w_risk_off, _ = _compute_returns_and_setup(
        prices, regime_df, allocations
    )
    result = _run_backtest_vectorized(
        returns, regime_df, allocations, w_risk_on, w_risk_off,
        returns[TICKERS].mean(axis=1),
        return_weights=return_weights,
        use_stagflation_override=use_stagflation_override,
        use_stagflation_risk_on_cap=use_stagflation_risk_on_cap,
        stagflation_risk_on_cap=stagflation_risk_on_cap,
    )
    return result


def run_backtest(pipeline_data: "PipelineData | None" = None) -> dict[str, Any]:
    """Run full backtest. Returns dict with portfolio_returns, metrics, etc.

    Args:
        pipeline_data: If provided, use cached prices. Otherwise fetch via fetch_prices.
    """
    if pipeline_data is not None:
        prices = pipeline_data.get_prices()
        logger.debug("[DATA] Backtest using shared pipeline_data (cache hit)")
    else:
        prices = fetch_prices(end=get_end_date())

    db = Database()
    regime_df = db.load_regime_labels()
    allocations = db.load_optimal_allocations()
    allocations = {str(k).strip(): v for k, v in allocations.items()}
    for alloc in allocations.values():
        if "cash" not in alloc:
            alloc["cash"] = 0.0

    returns, regime_df, w_risk_on, w_risk_off, equal_weight_returns = _compute_returns_and_setup(
        prices, regime_df, allocations
    )

    portfolio_returns = _run_backtest_vectorized(
        returns, regime_df, allocations, w_risk_on, w_risk_off, equal_weight_returns,
        use_stagflation_override=True,
    )

    metrics = compute_metrics(portfolio_returns, rf_daily=CASH_DAILY_YIELD)
    bench_metrics = compute_metrics(equal_weight_returns, rf_daily=CASH_DAILY_YIELD)

    db.save_backtest_results(metrics, bench_metrics)

    asof = returns.index[-1]
    asof_regime = regime_df.loc[asof, "regime"]
    asof_alpha = None
    if "risk_on" in regime_df.columns and not pd.isna(regime_df.loc[asof, "risk_on"]):
        asof_alpha = float(regime_df.loc[asof, "risk_on"])

    next_month = (asof.to_period("M") + 1).strftime("%Y-%m")
    forecast = db.load_latest_regime_forecast(next_month)
    use_forecast = forecast is not None

    if use_forecast:
        blend_alpha = 0.5 * (asof_alpha or 0.5) + 0.5 * forecast["risk_on_forecast"]
    else:
        blend_alpha = asof_alpha

    asof_regime_stripped = str(asof_regime).strip()
    if asof_regime_stripped == "Stagflation" and "Stagflation" in allocations:
        base_weights = {str(k): float(v) for k, v in allocations["Stagflation"].items()}
    else:
        alpha_for_weights = blend_alpha if blend_alpha is not None else asof_alpha
        if alpha_for_weights is not None:
            base_weights = _blend_alloc(w_risk_off, w_risk_on, alpha_for_weights, ASSETS)
        else:
            rk = REGIME_ALIASES.get(asof_regime_stripped, asof_regime_stripped)
            base_weights = allocations.get(
                rk,
                {a: 1.0 / len(ASSETS) for a in ASSETS},
            )
            base_weights = {str(k): float(v) for k, v in base_weights.items()}

    trailing_pd = returns[TICKERS].loc[:asof].tail(VOL_LOOKBACK)
    trailing_pl = pl.from_pandas(trailing_pd)
    scaled_weights = vol_scaled_weights(base_weights, trailing_pl, list(TICKERS))
    db.save_current_weights(str(asof.date()), pd.Series(scaled_weights))
    db.close()

    return {
        "portfolio_returns": portfolio_returns,
        "metrics": metrics,
        "bench_metrics": bench_metrics,
        "current_weights": scaled_weights,
        "asof_date": asof,
        "asof_regime": asof_regime,
        "asof_alpha": asof_alpha,
        "forecast": forecast,
    }


def main() -> None:
    """Entry point: run backtest and print results."""
    result = run_backtest()
    m = result["metrics"]
    b = result["bench_metrics"]

    print("\n[PERFORMANCE] Portfolio:")
    for k, v in m.items():
        print(f"  {k}: {v:.2%}" if k != "Sharpe" else f"  {k}: {v:.2f}")

    print("\n[BENCHMARK] Equal-Weight:")
    for k, v in b.items():
        print(f"  {k}: {v:.2%}" if k != "Sharpe" else f"  {k}: {v:.2f}")

    print("\n[CURRENT] TARGET WEIGHTS")
    print(f"As-of: {result['asof_date'].date()} | Regime: {result['asof_regime']}")
    if result["forecast"]:
        print(f"Next month forecast: {result['forecast']['risk_on_forecast']:.3f}")
    for a, w in sorted(result["current_weights"].items(), key=lambda x: -x[1]):
        print(f"  {a:>6}: {w:6.2%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
