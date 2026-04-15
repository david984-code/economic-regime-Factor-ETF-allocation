"""Sortino optimization per regime with regime-specific constraints."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import (
    DEFAULT_MAX_CASH,
    DEFAULT_MIN_CASH,
    OUTPUTS_DIR,
    REGIME_CASH,
    REGIME_MAX_ASSETS,
    REGIME_MAX_RISK_ON,
    REGIME_MIN_ASSETS,
    REGIME_MIN_RISK_ON,
    RISK_ON_ASSETS_BASE,
    get_end_date,
)
from src.data.market_ingestion import fetch_monthly_returns
from src.utils.database import Database

if TYPE_CHECKING:
    from src.data.pipeline_data import PipelineData

logger = logging.getLogger(__name__)


def load_regimes() -> pd.DataFrame:
    """Load regime labels from database, fallback to CSV."""
    db = Database()
    try:
        df = db.load_regime_labels()
        db.close()
        return df
    except Exception as exc:
        db.close()
        path = OUTPUTS_DIR / "regime_labels_expanded.csv"
        if not path.exists():
            raise FileNotFoundError(
                "No regime data found. Run regime classification first."
            ) from exc
        return pd.read_csv(path, parse_dates=["date"], index_col="date")


def _negative_sortino(
    weights: np.ndarray,
    returns_risky: np.ndarray,
    risk_free: float = 0.0,
) -> float:
    """Negative Sortino (return / downside vol) for minimization."""
    risky_weights = np.asarray(weights[:-1], dtype=float)
    cash_weight = float(weights[-1])
    risky_sum = risky_weights.sum()
    if risky_sum <= 1e-10:
        return 1e9
    w = risky_weights / risky_sum
    port_rets = returns_risky @ w
    mean_ret = float(np.mean(port_rets))
    downside_rets = np.minimum(port_rets - risk_free, 0.0)
    downside_var = float(np.mean(downside_rets**2))
    downside_vol = np.sqrt(downside_var) if downside_var > 1e-12 else 1e-8
    sortino = (mean_ret - risk_free) / downside_vol
    if not np.isfinite(sortino):
        return 1e9
    return float(-sortino + 0.05 * cash_weight)


def _get_constraints(
    num_assets: int,
    regime: str,
    asset_list: list[str],
) -> list[dict[str, Any]]:
    """Build scipy constraints for optimizer.

    Constraints:
      - Weights sum to 1
      - Cash within regime-specific bounds
      - Per-asset min/max floors and caps
      - Min aggregate risk-on sleeve weight (Recovery/Overheating)
    """
    min_cash, max_cash = REGIME_CASH.get(regime, (DEFAULT_MIN_CASH, DEFAULT_MAX_CASH))
    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w, mc=min_cash: w[-1] - mc},
        {"type": "ineq", "fun": lambda w, mx=max_cash: mx - w[-1]},
    ]

    # Per-asset minimum floors
    min_assets = REGIME_MIN_ASSETS.get(regime, {})
    for asset, min_w in min_assets.items():
        if asset in asset_list:
            idx = asset_list.index(asset)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, i=idx, m=min_w: w[i] - m,
                }
            )

    # Per-asset maximum caps (prevent gold/bond concentration in risk-on regimes)
    max_assets = REGIME_MAX_ASSETS.get(regime, {})
    for asset, max_w in max_assets.items():
        if asset in asset_list:
            idx = asset_list.index(asset)
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, i=idx, m=max_w: m - w[i],
                }
            )

    # Min aggregate risk-on sleeve weight
    risk_on_indices = [
        asset_list.index(a) for a in RISK_ON_ASSETS_BASE if a in asset_list
    ]
    min_risk_on = REGIME_MIN_RISK_ON.get(regime)
    if min_risk_on is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, idxs=risk_on_indices, m=min_risk_on: (
                    sum(w[i] for i in idxs) - m
                ),
            }
        )

    # Max aggregate risk-on sleeve weight (defensive regimes)
    max_risk_on = REGIME_MAX_RISK_ON.get(regime)
    if max_risk_on is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, idxs=risk_on_indices, m=max_risk_on: (
                    m - sum(w[i] for i in idxs)
                ),
            }
        )

    return constraints


def _ensure_cash_column(returns: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic cash column (~5% annualized) if missing."""
    if "cash" not in returns.columns:
        returns = returns.copy()
        returns["cash"] = (1.05) ** (1 / 12) - 1
    return returns


def _merge_returns_and_regimes(
    returns: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge monthly returns with regime labels on Period.

    Returns (merged_df, risky_assets, full_asset_list).
    """
    ret = returns.copy()
    if "Period" not in ret.columns and ret.index is not None:
        ret["Period"] = ret.index.to_period("M")

    reg = regime_df.copy()
    if "Period" not in reg.columns:
        reg["Period"] = pd.to_datetime(reg.index).to_period("M")

    ret = _ensure_cash_column(ret)
    all_assets = [c for c in ret.columns if c != "Period"]
    risky_assets = [a for a in all_assets if a != "cash"]
    full_asset_list = risky_assets + ["cash"]

    regime_cols = [c for c in ["Period", "regime"] if c in reg.columns]
    merged = pd.merge(ret, reg[regime_cols], on="Period", how="inner")
    merged.set_index("Period", inplace=True)
    return merged, risky_assets, full_asset_list


def _optimize_single_regime(
    regime: str,
    subset_risky: pd.DataFrame,
    risky_assets: list[str],
    full_asset_list: list[str],
) -> dict[str, float] | None:
    """Run Sortino optimization for a single regime. Returns weights or None."""
    if len(subset_risky) < 2:
        return None
    returns_risky = subset_risky.values.astype(float)
    n = len(full_asset_list)
    min_cash, max_cash = REGIME_CASH.get(regime, (DEFAULT_MIN_CASH, DEFAULT_MAX_CASH))
    init_guess = np.full(n, 0.0)
    start_cash = (min_cash + max_cash) / 2
    init_guess[:-1] = (1 - start_cash) / (n - 1)
    init_guess[-1] = start_cash

    result = minimize(
        _negative_sortino,
        init_guess,
        args=(returns_risky,),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=_get_constraints(n, regime, risky_assets),
    )
    if result.success:
        return dict(zip(full_asset_list, result.x, strict=False))
    logger.error("Optimization failed for %s: %s", regime, result.message)
    return None


def _save_allocations(allocations: dict[str, dict[str, float]]) -> None:
    """Persist optimized allocations to DB, CSV, and formatted Excel."""
    db = Database()
    db.save_optimal_allocations(allocations)
    db.close()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    df_opt = pd.DataFrame(allocations).T
    df_opt.index.name = "regime"
    df_opt.to_csv(OUTPUTS_DIR / "optimal_allocations.csv")
    logger.info("Saved optimal allocations")

    from src.allocation.format_allocations import format_allocations_to_excel

    format_allocations_to_excel()


def optimize_allocations_from_data(
    returns: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Optimize Sortino allocations per regime from given data. No DB.

    Used by walk-forward evaluation to train on train-period data only.

    Args:
        returns: Monthly returns with Period index (or date index, will add Period).
        regime_df: Regime labels with date index. Must have 'regime' column.

    Returns:
        Dict regime -> {asset: weight}.
    """
    merged, risky_assets, full_asset_list = _merge_returns_and_regimes(
        returns, regime_df
    )
    optimal_allocations: dict[str, dict[str, float]] = {}
    for regime in merged["regime"].unique():
        subset_risky = merged[merged["regime"] == regime][risky_assets].fillna(0)
        weights = _optimize_single_regime(
            regime, subset_risky, risky_assets, full_asset_list
        )
        if weights is not None:
            optimal_allocations[regime] = weights
    return optimal_allocations


def run_optimizer(pipeline_data: "PipelineData | None" = None) -> None:
    """Run Sortino optimization per regime and save.

    Args:
        pipeline_data: If provided, use cached monthly returns.
            Otherwise fetch via fetch_monthly_returns.
    """
    if pipeline_data is not None:
        returns = pipeline_data.get_monthly_returns()
        logger.debug("[DATA] Optimizer using shared pipeline_data (cache hit)")
    else:
        returns = fetch_monthly_returns(end=get_end_date())
    returns = _ensure_cash_column(returns)

    regimes = load_regimes()
    merged, risky_assets, full_asset_list = _merge_returns_and_regimes(returns, regimes)

    optimal_allocations: dict[str, dict[str, float]] = {}
    for regime in merged["regime"].unique():
        logger.info("Optimizing for regime: %s (Sortino)", regime)
        subset_risky = merged[merged["regime"] == regime][risky_assets].fillna(0)
        weights = _optimize_single_regime(
            regime, subset_risky, risky_assets, full_asset_list
        )
        if weights is not None:
            optimal_allocations[regime] = weights
            logger.info("Optimized: %s", regime)

    if optimal_allocations:
        _save_allocations(optimal_allocations)
    else:
        logger.warning("No successful optimizations")


def main() -> None:
    """Entry point."""
    run_optimizer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
