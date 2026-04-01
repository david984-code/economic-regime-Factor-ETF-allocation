"""Backtest engine: regime-based allocation with vol scaling and transaction costs."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from src.allocation.vol_scaling import vol_scaled_weights, vol_scaled_weights_from_std
from src.backtest.metrics import compute_metrics
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
from src.features.transforms import sigmoid
from src.utils.database import Database

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
    out: dict[str, float] = dict.fromkeys(assets, 0.0)
    for r in regs:
        for a in assets:
            out[a] += float(allocations[r].get(a, 0.0))
    for a in assets:
        out[a] /= len(regs)
    return out


def _compute_asset_momentum_timeseries(
    prices: pd.DataFrame,
    assets: list[str],
    lookback_months: int,
) -> pd.DataFrame:
    """Compute momentum for each asset as a time series (no lookahead).
    
    Args:
        prices: Daily price DataFrame.
        assets: List of assets to compute momentum for.
        lookback_months: Lookback window in months.
    
    Returns:
        DataFrame with monthly momentum for each asset (expanding window, no lookahead).
    """
    # Get monthly prices
    monthly_prices = prices.resample("ME").last()

    momentum_df = pd.DataFrame(index=monthly_prices.index)

    for asset in assets:
        if asset not in prices.columns or asset == "cash":
            momentum_df[asset] = 0.0
            continue

        asset_monthly = monthly_prices[asset]

        # Compute momentum at each month using only past data
        momentum_series = []
        for i in range(len(asset_monthly)):
            if i < lookback_months:
                momentum_series.append(0.0)
            else:
                price_now = asset_monthly.iloc[i]
                price_lookback = asset_monthly.iloc[i - lookback_months]
                momentum = (price_now / price_lookback) - 1
                momentum_series.append(momentum)

        momentum_df[asset] = momentum_series

    return momentum_df


def _equal_weight_alloc(
    assets: list[str],
    is_risk_on: bool = True,
    risk_on_sleeve: list[str] | None = None,
    risk_off_sleeve: list[str] | None = None,
) -> dict[str, float]:
    """Equal-weight allocation across assets with separate risk-on/risk-off sleeves.
    
    Args:
        assets: Full list of assets (for structure).
        is_risk_on: If True, equal weight risk-on sleeve. If False, equal weight risk-off sleeve.
        risk_on_sleeve: Custom list of risk-on assets (defaults to base universe).
        risk_off_sleeve: Custom list of risk-off assets (defaults to base universe).
    
    Returns:
        Dict with equal weights for selected sleeve, zero for others.
    """
    from src.config import RISK_OFF_ASSETS_BASE, RISK_ON_ASSETS_BASE

    # Use provided sleeves or defaults
    risk_on_assets = risk_on_sleeve if risk_on_sleeve is not None else RISK_ON_ASSETS_BASE
    risk_off_assets = risk_off_sleeve if risk_off_sleeve is not None else RISK_OFF_ASSETS_BASE

    if is_risk_on:
        # Equal weight across risk-on assets
        weight = 1.0 / len(risk_on_assets)
        return {a: weight if a in risk_on_assets else 0.0 for a in assets}
    else:
        # Equal weight across risk-off assets
        weight = 1.0 / len(risk_off_assets)
        return {a: weight if a in risk_off_assets else 0.0 for a in assets}


def _compute_asset_momentum_allocations_timeseries(
    momentum_df: pd.DataFrame,
    assets: list[str],
    filter_method: str = "positive",
    top_n: int = 5,
    risk_on_sleeve: list[str] | None = None,
) -> pd.DataFrame:
    """Compute dynamic equal-weight allocations based on asset momentum (monthly).
    
    Args:
        momentum_df: Monthly momentum DataFrame (from _compute_asset_momentum_timeseries).
        assets: Full list of assets.
        filter_method: 'positive' (only positive momentum), 'top_n' (top N by momentum).
        top_n: Number of top assets to select (if filter_method='top_n').
        risk_on_sleeve: Custom list of risk-on assets (defaults to base universe).
    
    Returns:
        DataFrame with monthly risk-on allocation weights for each asset.
    """
    from src.config import RISK_ON_ASSETS_BASE

    # Use provided sleeve or default
    risk_on_assets = risk_on_sleeve if risk_on_sleeve is not None else RISK_ON_ASSETS_BASE

    alloc_df = pd.DataFrame(index=momentum_df.index, columns=assets, data=0.0)

    for date in momentum_df.index:
        momentum_row = momentum_df.loc[date, risk_on_assets]

        # Filter based on method
        if filter_method == "positive":
            # Only include assets with positive momentum
            selected = [a for a in risk_on_assets if momentum_row.get(a, 0) > 0]

            # Fallback: if no assets qualify, use top 3
            if len(selected) == 0:
                sorted_assets = sorted(risk_on_assets, key=lambda a: momentum_row.get(a, -999), reverse=True)
                selected = sorted_assets[:3]

        elif filter_method == "top_n":
            # Select top N by momentum
            sorted_assets = sorted(risk_on_assets, key=lambda a: momentum_row.get(a, -999), reverse=True)
            selected = sorted_assets[:top_n]

        else:
            raise ValueError(f"Unknown filter_method: {filter_method}")

        # Equal weight among selected
        if len(selected) > 0:
            weight = 1.0 / len(selected)
            for a in selected:
                alloc_df.loc[date, a] = weight

    return alloc_df


def _risk_parity_alloc(
    prices: pd.DataFrame,
    assets: list[str],
    lookback_days: int = 63,
) -> dict[str, float]:
    """Risk parity allocation using trailing volatility (inverse vol weighted)."""
    # Only use tickers that have prices (exclude cash)
    price_tickers = [a for a in assets if a in prices.columns]

    # Compute trailing volatility for each asset
    returns = prices[price_tickers].pct_change().dropna()
    trailing_vol = returns.rolling(window=lookback_days, min_periods=lookback_days).std().iloc[-1]

    # Inverse volatility weights
    inv_vol = 1.0 / trailing_vol
    total_inv_vol = inv_vol.sum()
    weights = inv_vol / total_inv_vol

    # Convert to dict and add cash with zero weight
    weights_dict = weights.to_dict()
    if "cash" in assets:
        weights_dict["cash"] = 0.0

    return weights_dict


def _heuristic_alloc(is_risk_on: bool) -> dict[str, float]:
    """Fixed heuristic allocation using simple rules.
    
    Risk-on: 60% equity, 30% quality/value, 10% bonds
    Risk-off: 20% equity, 10% quality, 70% bonds
    """
    if is_risk_on:
        # Risk-on: favor equities and factors
        return {
            "SPY": 0.30,
            "MTUM": 0.10,
            "VLUE": 0.10,
            "QUAL": 0.10,
            "USMV": 0.0,
            "IJR": 0.10,
            "VIG": 0.0,
            "GLD": 0.0,
            "IEF": 0.10,
            "TLT": 0.0,
            "cash": 0.20,
        }
    else:
        # Risk-off: favor bonds and defensive
        return {
            "SPY": 0.10,
            "MTUM": 0.0,
            "VLUE": 0.0,
            "QUAL": 0.05,
            "USMV": 0.05,
            "IJR": 0.0,
            "VIG": 0.0,
            "GLD": 0.10,
            "IEF": 0.30,
            "TLT": 0.30,
            "cash": 0.10,
        }



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


def _compute_vol_scaling(
    prices: pd.DataFrame,
    vol_scaling_method: str,
    vol_lookback_days: int = 20,
) -> pd.Series:
    """Compute volatility scaling factor (no lookahead).
    
    Args:
        prices: Daily price DataFrame with SPY column.
        vol_scaling_method: One of 'none', 'realized_20d', 'realized_63d', 'percentile'.
        vol_lookback_days: Lookback for realized vol (20 or 63).
    
    Returns:
        Daily scaling factor series (0.5 to 1.5, where 1.0 = no scaling).
    """
    spy = prices["SPY"].copy()
    returns = spy.pct_change().dropna()

    if vol_scaling_method == "none":
        # No scaling - always 1.0
        return pd.Series(1.0, index=spy.index)

    elif vol_scaling_method in ["realized_20d", "realized_63d"]:
        # Scale by realized vol vs long-run average
        if vol_scaling_method == "realized_20d":
            lookback = 20
        else:
            lookback = 63

        # Realized vol (annualized)
        realized_vol = returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)

        # Long-run average vol (expanding window, no lookahead)
        scaling_factors = []
        for i in range(len(realized_vol)):
            if pd.isna(realized_vol.iloc[i]):
                scaling_factors.append(1.0)
            else:
                # Use expanding window for long-run average
                trailing_vols = realized_vol.iloc[:i+1].dropna()
                if len(trailing_vols) >= lookback * 2:  # Need sufficient history
                    long_run_avg = trailing_vols.mean()
                    current_vol = realized_vol.iloc[i]
                    # Scale inversely: high vol -> reduce exposure
                    vol_ratio = long_run_avg / current_vol
                    # Cap scaling between 0.5 and 1.5
                    scaling_factors.append(np.clip(vol_ratio, 0.5, 1.5))
                else:
                    scaling_factors.append(1.0)

        scaling = pd.Series(scaling_factors, index=realized_vol.index)
        # Reindex to match SPY
        return scaling.reindex(spy.index).fillna(1.0)

    elif vol_scaling_method == "percentile":
        # Scale by vol percentile (reduce exposure in top quintile)
        lookback = 63  # Use 63-day vol for percentile

        # Realized vol (annualized)
        realized_vol = returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)

        # Compute percentile rank using expanding window (no lookahead)
        scaling_factors = []
        for i in range(len(realized_vol)):
            if pd.isna(realized_vol.iloc[i]):
                scaling_factors.append(1.0)
            else:
                # Use expanding window for percentile
                trailing_vols = realized_vol.iloc[:i+1].dropna()
                if len(trailing_vols) >= lookback * 2:
                    current_vol = realized_vol.iloc[i]
                    percentile = (trailing_vols < current_vol).sum() / len(trailing_vols)

                    # Reduce exposure in top quintile (80th percentile+)
                    if percentile >= 0.80:
                        # Linear scale from 1.0 at 80th to 0.5 at 100th
                        scaling_factors.append(1.0 - 0.5 * ((percentile - 0.80) / 0.20))
                    else:
                        scaling_factors.append(1.0)
                else:
                    scaling_factors.append(1.0)

        scaling = pd.Series(scaling_factors, index=realized_vol.index)
        return scaling.reindex(spy.index).fillna(1.0)

    else:
        raise ValueError(f"Unknown vol_scaling_method: {vol_scaling_method}")


def _compute_volatility_regime_score(
    prices: pd.DataFrame,
    lookback_days: int = 63,
    percentile_window_years: int = 3,
) -> pd.Series:
    """Compute volatility regime score (no lookahead).
    
    High volatility → negative score (reduce risk_on)
    Low volatility → positive score (increase risk_on)
    
    Args:
        prices: Daily price DataFrame with SPY column.
        lookback_days: Window for realized volatility (default 63 days = ~3 months).
        percentile_window_years: Years of history for percentile calculation (default 3).
    
    Returns:
        Monthly series with volatility regime score (z-scored, inverted).
        Higher score = lower vol = favorable for risk-on.
    """
    spy = prices["SPY"].copy()
    returns = spy.pct_change()

    # Compute realized volatility (annualized)
    realized_vol = returns.rolling(window=lookback_days, min_periods=lookback_days).std() * np.sqrt(252)

    # Convert to monthly (end-of-month volatility)
    vol_monthly = realized_vol.resample("ME").last()

    # Compute percentile at each month using expanding window (no lookahead)
    min_history = percentile_window_years * 12  # Months
    percentile_list = []

    for i in range(len(vol_monthly)):
        if pd.isna(vol_monthly.iloc[i]):
            percentile_list.append(np.nan)
        else:
            # Use expanding window (up to percentile_window_years * 12 months)
            lookback_months = min(i + 1, min_history * 12)
            trailing_vols = vol_monthly.iloc[max(0, i + 1 - lookback_months):i + 1].dropna()

            if len(trailing_vols) >= min_history:
                current_vol = vol_monthly.iloc[i]
                percentile = (trailing_vols < current_vol).sum() / len(trailing_vols)
                percentile_list.append(percentile)
            else:
                percentile_list.append(np.nan)

    percentile_series = pd.Series(percentile_list, index=vol_monthly.index)

    # Convert percentile to score:
    # percentile 0 (low vol) → high score
    # percentile 1 (high vol) → low score
    # Use z-score transformation (invert so high vol = negative)
    vol_regime_score = percentile_series.copy()

    # Expanding window z-score (invert: high percentile → negative z-score)
    for i in range(len(percentile_series)):
        if pd.isna(percentile_series.iloc[i]):
            vol_regime_score.iloc[i] = 0.0
        else:
            trailing = percentile_series.iloc[:i + 1].dropna()
            if len(trailing) >= min_history:
                mean_pctl = trailing.mean()
                std_pctl = trailing.std()
                if std_pctl > 0:
                    # Invert: high percentile (high vol) → negative score
                    vol_regime_score.iloc[i] = -(percentile_series.iloc[i] - mean_pctl) / std_pctl
                else:
                    vol_regime_score.iloc[i] = 0.0
            else:
                vol_regime_score.iloc[i] = 0.0

    return vol_regime_score


def _compute_trend_filter(
    prices: pd.DataFrame,
    filter_type: str,
) -> pd.Series:
    """Compute absolute trend filter on SPY (no lookahead).
    
    Args:
        prices: Daily price DataFrame with SPY column.
        filter_type: One of '200dma', '12m_return', '10mma', or 'none'.
    
    Returns:
        Daily boolean series (True = trend is positive, False = trend is negative).
    """
    spy = prices["SPY"].copy()

    if filter_type == "none":
        # No filter - always on
        return pd.Series(True, index=spy.index)

    elif filter_type == "200dma":
        # SPY price > 200-day moving average
        ma_200 = spy.rolling(window=200, min_periods=200).mean()
        return spy > ma_200

    elif filter_type == "12m_return":
        # SPY 12-month return > 0
        # Shift by 252 trading days (approximately 12 months)
        spy_12m_ago = spy.shift(252)
        return_12m = (spy / spy_12m_ago) - 1
        return return_12m > 0

    elif filter_type == "10mma":
        # SPY price > 10-month moving average
        # 10 months ≈ 210 trading days
        ma_10m = spy.rolling(window=210, min_periods=210).mean()
        return spy > ma_10m

    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")


def _compute_sector_breadth(
    breadth_prices: pd.DataFrame,
    sector_etfs: list[str],
) -> pd.Series:
    """Compute monthly market breadth: % of sector ETFs above their 200-day MA.

    Timing: uses only data available up to each date (no lookahead).
    At each month-end, compute the fraction of ETFs with price > 200-day rolling mean.

    Args:
        breadth_prices: Daily price DataFrame containing the sector ETFs.
        sector_etfs: List of sector ETF tickers to use for breadth.

    Returns:
        Monthly pd.Series (indexed at month-end) of breadth values in [0, 1].
    """
    etfs = [e for e in sector_etfs if e in breadth_prices.columns]
    if not etfs:
        raise ValueError("No sector ETFs found in breadth_prices.")

    above_df = pd.DataFrame(index=breadth_prices.index)
    for etf in etfs:
        px = breadth_prices[etf].ffill()
        ma200 = px.rolling(200, min_periods=200).mean()
        above_df[etf] = (px > ma200).astype(float)
        above_df.loc[ma200.isna(), etf] = np.nan  # exclude warmup rows

    valid = above_df.notna().sum(axis=1).clip(lower=1)
    daily_breadth = above_df.sum(axis=1) / valid

    return daily_breadth.resample("ME").last()


def _compute_market_primary_risk_on(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    market_lookback_months: int = 24,
    sigmoid_scale: float = 0.25,
    macro_stress_cap: float = 0.40,
) -> pd.DataFrame:
    """Market-primary signal: raw trailing return through sigmoid.

    Unlike the hybrid signal, this does NOT apply expanding z-score
    normalization. Raw trailing return is mapped directly through sigmoid,
    so strong bull markets produce high risk_on and bear markets produce low.

    Macro regime acts as a stress override only: when the macro regime is
    Contraction or Stagflation, risk_on is capped at macro_stress_cap.

    Args:
        prices: Daily prices with SPY column.
        regime_df: Regime labels with regime and risk_on columns.
        market_lookback_months: Trailing return lookback (default 24).
        sigmoid_scale: Sigmoid mapping steepness (default 0.25).
        macro_stress_cap: Max risk_on during Contraction/Stagflation.

    Returns:
        Updated regime_df with market-primary risk_on.
    """
    regime_df = regime_df.copy()
    spy_monthly = prices["SPY"].resample("ME").last()

    # Compute raw trailing return at each month-end (no lookahead)
    trailing_ret = []
    for i in range(len(spy_monthly)):
        if i < market_lookback_months:
            trailing_ret.append(0.0)
        else:
            ret = (spy_monthly.iloc[i] / spy_monthly.iloc[i - market_lookback_months]) - 1
            trailing_ret.append(ret)

    raw_signal = pd.Series(trailing_ret, index=spy_monthly.index)

    # Map directly through sigmoid — no z-score normalization
    # Scale so that typical annual returns (~10%) map to risk_on ~0.6-0.7
    # and negative returns map to risk_on ~0.3-0.4
    risk_on_monthly = 1.0 / (1.0 + np.exp(-raw_signal * sigmoid_scale * 10))

    # Forward-fill to daily
    risk_on_daily = risk_on_monthly.reindex(regime_df.index, method="ffill")
    regime_df["risk_on"] = risk_on_daily.fillna(regime_df["risk_on"])

    # Macro stress override: cap risk_on during Contraction/Stagflation
    stress_mask = regime_df["regime"].isin(["Contraction", "Stagflation"])
    regime_df.loc[stress_mask, "risk_on"] = regime_df.loc[stress_mask, "risk_on"].clip(
        upper=macro_stress_cap
    )

    return regime_df


def _compute_hybrid_risk_on(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    macro_weight: float = 0.5,
    market_lookback_months: int = 12,
    use_momentum: bool = False,
    trend_filter_type: str = "none",
    trend_filter_risk_on_cap: float = 0.3,
    vol_scaling_method: str = "none",
    momentum_12m_weight: float = 0.0,
    use_vol_regime: bool = False,
    vol_regime_weight: float = 0.0,
    sigmoid_scale: float = 0.25,
    momentum_6m_weight: float = 0.0,
    breadth_weight: float = 0.0,
    breadth_prices: "pd.DataFrame | None" = None,
    vol_zscore_weight: float = 0.0,
    yield_curve_weight: float = 0.0,
    yield_curve_data: "pd.Series | None" = None,
    inversion_flag_offset: float = 0.0,
    breadth_flag_offset: float = 0.0,
) -> pd.DataFrame:
    """Compute hybrid risk_on combining macro_score with market signal and volatility regime.
    
    TIMING: Uses expanding window to avoid lookahead bias. At each month-end,
    computes momentum using only data available up to that date.
    
    Args:
        prices: Daily price DataFrame with SPY column.
        regime_df: Regime DataFrame with macro_score and risk_on columns.
        macro_weight: Weight for macro_score in hybrid (0-1).
        market_lookback_months: Primary lookback window for momentum (e.g., 24).
        use_momentum: If True, signal = +momentum. If False, signal = -momentum (mean reversion).
        trend_filter_type: Trend filter to apply ('none', '200dma', '12m_return', '10mma').
        trend_filter_risk_on_cap: Max risk_on when trend filter is OFF (default 0.3).
        vol_scaling_method: Volatility scaling method ('none', 'realized_20d', 'realized_63d', 'percentile').
        momentum_12m_weight: Weight for 12M momentum in ensemble (0-1). Remaining weight goes to primary lookback.
        use_vol_regime: If True, include volatility regime in signal combination.
        vol_regime_weight: Weight for volatility regime (0-1). Weights are normalized.
        breadth_weight: Weight for sector-breadth signal (0-1). Combined score =
            (1 - breadth_weight) * market_signal_z + breadth_weight * breadth_z.
            Requires breadth_prices to be provided.
        breadth_prices: Daily price DataFrame for breadth sector ETFs. Required when
            breadth_weight > 0.
        vol_zscore_weight: Weight for direct expanding z-score of 63-day realized SPY vol.
            combined_score = (1 - vol_zscore_weight) * combined_score
                           - vol_zscore_weight * zscore_63d_realized_vol
            Higher realized vol reduces risk_on. Applied after all other blends.
        yield_curve_weight: Weight for 10Y-3M Treasury term-spread z-score (0-1).
            combined_score = (1 - yield_curve_weight) * combined_score
                           + yield_curve_weight * zscore_term_spread
            Steeper curve (positive spread) raises risk_on; inverted curve lowers it.
            Requires yield_curve_data (monthly pd.Series of spread values) to be provided.
        yield_curve_data: Monthly pd.Series of 10Y-3M term spread (pct points).
            Required when yield_curve_weight > 0 or inversion_flag_offset > 0.
        inversion_flag_offset: Fixed negative offset applied to combined_score when the
            10Y-3M term spread has been negative for 3+ consecutive months.
            combined_score = combined_score - inversion_flag_offset * inversion_flag
            where inversion_flag ∈ {0, 1}.
            Requires yield_curve_data to be provided.
        breadth_flag_offset: Fixed negative offset when sector breadth < 30% for 2+ consecutive months.
            breadth_flag ∈ {0, 1}; combined_score = combined_score - breadth_flag_offset * breadth_flag.
            Requires breadth_prices to be provided.
        momentum_6m_weight: Weight for 6-month momentum z-score blended with primary (24M) z-score.
            combined_signal_z = (1 - momentum_6m_weight) * market_signal_z + momentum_6m_weight * z_6M.
            Used for dual-speed momentum (e.g. 0.3 = 70% 24M, 30% 6M).
    
    Returns:
        Updated regime_df with hybrid risk_on.
    """
    regime_df = regime_df.copy()

    # Get monthly regime and macro_score
    regime_monthly = regime_df.resample("ME").last()
    macro_score_monthly = regime_monthly["macro_score"].copy()

    # Compute momentum incrementally at each month-end (expanding window)
    spy_monthly = prices["SPY"].resample("ME").last()

    # Compute primary lookback momentum
    primary_signal_list = []
    for i, date in enumerate(spy_monthly.index):
        if i < market_lookback_months:
            primary_signal_list.append(np.nan)
        else:
            price_now = spy_monthly.iloc[i]
            price_lookback = spy_monthly.iloc[i - market_lookback_months]
            momentum = (price_now / price_lookback) - 1

            if use_momentum:
                signal = momentum
            else:
                signal = -momentum

            primary_signal_list.append(signal)

    primary_signal = pd.Series(primary_signal_list, index=spy_monthly.index)

    # If ensemble, compute 12M momentum as well
    if momentum_12m_weight > 0:
        momentum_12m_list = []
        for i, date in enumerate(spy_monthly.index):
            if i < 12:
                momentum_12m_list.append(np.nan)
            else:
                price_now = spy_monthly.iloc[i]
                price_12m_ago = spy_monthly.iloc[i - 12]
                momentum_12m = (price_now / price_12m_ago) - 1

                if use_momentum:
                    signal_12m = momentum_12m
                else:
                    signal_12m = -momentum_12m

                momentum_12m_list.append(signal_12m)

        momentum_12m_signal = pd.Series(momentum_12m_list, index=spy_monthly.index)

        # Blend signals (before z-score normalization)
        market_signal = (1 - momentum_12m_weight) * primary_signal + momentum_12m_weight * momentum_12m_signal
    else:
        market_signal = primary_signal

    # Align to regime monthly index
    market_signal_aligned = market_signal.reindex(regime_monthly.index)

    # Expanding window z-score normalization (no lookahead)
    market_signal_z = market_signal_aligned.copy()
    min_history = max(market_lookback_months, 12)
    for i in range(len(market_signal_aligned)):
        trailing = market_signal_aligned.iloc[:i + 1].dropna()
        if len(trailing) >= min_history:
            market_signal_z.iloc[i] = (market_signal_aligned.iloc[i] - trailing.mean()) / trailing.std()
        else:
            market_signal_z.iloc[i] = 0.0

    # Optionally blend in 6M momentum z-score (dual-speed: 24M + 6M)
    if momentum_6m_weight > 0.0:
        mom6_list = []
        for i in range(len(spy_monthly.index)):
            if i < 6:
                mom6_list.append(np.nan)
            else:
                price_now = spy_monthly.iloc[i]
                price_6m_ago = spy_monthly.iloc[i - 6]
                mom6 = (price_now / price_6m_ago) - 1
                mom6_list.append(mom6 if use_momentum else -mom6)
        mom6_signal = pd.Series(mom6_list, index=spy_monthly.index)
        mom6_aligned = mom6_signal.reindex(regime_monthly.index)
        mom6_z = mom6_aligned.copy()
        min_6m = 6
        for i in range(len(mom6_aligned)):
            trailing = mom6_aligned.iloc[: i + 1].dropna()
            if len(trailing) >= min_6m:
                std = float(trailing.std())
                mom6_z.iloc[i] = (float(mom6_aligned.iloc[i]) - float(trailing.mean())) / std if std > 1e-10 else 0.0
            else:
                mom6_z.iloc[i] = 0.0
        mom6_z = mom6_z.fillna(0.0)
        market_signal_z = (1.0 - momentum_6m_weight) * market_signal_z + momentum_6m_weight * mom6_z

    # Optionally compute volatility regime score
    if use_vol_regime and vol_regime_weight > 0:
        vol_regime_score = _compute_volatility_regime_score(prices, lookback_days=63, percentile_window_years=3)
        vol_regime_aligned = vol_regime_score.reindex(regime_monthly.index).fillna(0.0)

        # Normalize weights so they sum to 1.0
        total_weight = macro_weight + (1 - macro_weight) + vol_regime_weight
        norm_macro = macro_weight / total_weight
        norm_market = (1 - macro_weight) / total_weight
        norm_vol = vol_regime_weight / total_weight

        # Combine three signals
        combined_score = (
            norm_macro * macro_score_monthly +
            norm_market * market_signal_z +
            norm_vol * vol_regime_aligned
        )
    else:
        # Original two-signal combination
        combined_score = macro_weight * macro_score_monthly + (1 - macro_weight) * market_signal_z

    # Optionally blend in sector-breadth signal
    # combined_score = (1 - breadth_weight) * market_signal_z + breadth_weight * breadth_z
    # (macro_weight=0 in the accepted baseline, so combined_score == market_signal_z above)
    if breadth_weight > 0.0 and breadth_prices is not None:
        sector_etfs = [
            "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
        ]
        monthly_breadth = _compute_sector_breadth(breadth_prices, sector_etfs)
        breadth_aligned = monthly_breadth.reindex(regime_monthly.index)

        # Expanding window z-score of breadth (same approach as momentum z-score)
        breadth_z_list: list[float] = []
        min_breadth_history = 12
        for i in range(len(breadth_aligned)):
            trailing = breadth_aligned.iloc[: i + 1].dropna()
            if len(trailing) >= min_breadth_history:
                val = breadth_aligned.iloc[i]
                std = float(trailing.std())
                breadth_z_list.append(
                    float((val - trailing.mean()) / std) if std > 1e-10 else 0.0
                )
            else:
                breadth_z_list.append(0.0)
        breadth_z = pd.Series(breadth_z_list, index=regime_monthly.index)

        # Replace NaN breadth with 0 (treated as neutral) during warmup
        breadth_z = breadth_z.fillna(0.0)

        # Blend: market momentum gets (1-breadth_weight), breadth gets breadth_weight
        combined_score = (1.0 - breadth_weight) * combined_score + breadth_weight * breadth_z

    # Optionally blend in direct expanding z-score of 63-day realized SPY vol.
    # combined_score = (1 - w) * combined_score - w * zscore_63d_vol
    # (high realized vol -> negative contribution -> lower risk_on)
    if vol_zscore_weight > 0.0:
        spy_ret = prices["SPY"].pct_change()
        realized_vol = spy_ret.rolling(63, min_periods=63).std() * np.sqrt(252)
        vol_monthly_raw = realized_vol.resample("ME").last()
        vol_aligned = vol_monthly_raw.reindex(regime_monthly.index)

        # Expanding z-score of realized vol (same method as momentum z-score)
        vol_z_list: list[float] = []
        min_vol_history = 12
        for i in range(len(vol_aligned)):
            trailing = vol_aligned.iloc[: i + 1].dropna()
            if len(trailing) >= min_vol_history:
                val = float(vol_aligned.iloc[i])
                std = float(trailing.std())
                vol_z_list.append(
                    float((val - float(trailing.mean())) / std) if std > 1e-10 else 0.0
                )
            else:
                vol_z_list.append(0.0)
        vol_z = pd.Series(vol_z_list, index=regime_monthly.index).fillna(0.0)

        # Subtract: high vol raises vol_z → reduces risk_on
        combined_score = (1.0 - vol_zscore_weight) * combined_score - vol_zscore_weight * vol_z

    # Optionally blend in expanding z-score of 10Y-3M Treasury term spread.
    # combined_score = (1 - w) * combined_score + w * zscore_term_spread
    # (inverted / flat curve → negative contribution → lower risk_on)
    if yield_curve_weight > 0.0 and yield_curve_data is not None:
        yc_aligned = yield_curve_data.resample("ME").last().reindex(regime_monthly.index)

        # Expanding z-score of term spread (same method as momentum z-score)
        yc_z_list: list[float] = []
        min_yc_history = 12
        for i in range(len(yc_aligned)):
            trailing = yc_aligned.iloc[: i + 1].dropna()
            if len(trailing) >= min_yc_history:
                val = float(yc_aligned.iloc[i])
                std = float(trailing.std())
                yc_z_list.append(
                    float((val - float(trailing.mean())) / std) if std > 1e-10 else 0.0
                )
            else:
                yc_z_list.append(0.0)
        yc_z = pd.Series(yc_z_list, index=regime_monthly.index).fillna(0.0)

        # Add: steeper curve raises yc_z → increases risk_on
        combined_score = (1.0 - yield_curve_weight) * combined_score + yield_curve_weight * yc_z

    # Optionally apply a binary inversion-event offset.
    # inversion_flag = 1 when 10Y-3M spread has been < 0 for 3+ consecutive months.
    # combined_score = combined_score - inversion_flag_offset * inversion_flag
    if inversion_flag_offset > 0.0 and yield_curve_data is not None:
        yc_m = yield_curve_data.resample("ME").last().reindex(regime_monthly.index)
        inv_flag_list: list[float] = []
        for i in range(len(yc_m)):
            if i < 2:
                inv_flag_list.append(0.0)
            else:
                # Check 3 consecutive months (current + two prior) all negative
                v0 = yc_m.iloc[i]
                v1 = yc_m.iloc[i - 1]
                v2 = yc_m.iloc[i - 2]
                flag = 1.0 if (
                    not (pd.isna(v0) or pd.isna(v1) or pd.isna(v2))
                    and float(v0) < 0.0
                    and float(v1) < 0.0
                    and float(v2) < 0.0
                ) else 0.0
                inv_flag_list.append(flag)
        inv_flag = pd.Series(inv_flag_list, index=regime_monthly.index)
        # Subtract: active inversion flag lowers combined_score → lowers risk_on
        combined_score = combined_score - inversion_flag_offset * inv_flag

    # Optionally apply binary breadth breakdown flag.
    # breadth_flag = 1 when sector breadth < 30% for 2+ consecutive months.
    if breadth_flag_offset > 0.0 and breadth_prices is not None:
        sector_etfs_bf = [
            "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
        ]
        monthly_breadth_bf = _compute_sector_breadth(breadth_prices, sector_etfs_bf)
        breadth_aligned_bf = monthly_breadth_bf.reindex(regime_monthly.index)
        bf_list: list[float] = []
        for i in range(len(breadth_aligned_bf)):
            if i < 1:
                bf_list.append(0.0)
            else:
                b0 = breadth_aligned_bf.iloc[i]
                b1 = breadth_aligned_bf.iloc[i - 1]
                flag = 1.0 if (
                    not (pd.isna(b0) or pd.isna(b1))
                    and float(b0) < 0.30
                    and float(b1) < 0.30
                ) else 0.0
                bf_list.append(flag)
        breadth_flag = pd.Series(bf_list, index=regime_monthly.index)
        combined_score = combined_score - breadth_flag_offset * breadth_flag

    # Apply sigmoid transformation (same as macro_score)
    hybrid_risk_on = sigmoid(combined_score * sigmoid_scale)

    # Forward-fill to daily
    regime_monthly["hybrid_risk_on"] = hybrid_risk_on
    regime_daily_hybrid = regime_monthly[["hybrid_risk_on"]].reindex(regime_df.index, method="ffill")

    # Replace risk_on with hybrid
    regime_df["risk_on"] = regime_daily_hybrid["hybrid_risk_on"].fillna(regime_df["risk_on"])

    # Apply trend filter if specified
    if trend_filter_type != "none":
        trend_filter = _compute_trend_filter(prices, trend_filter_type)
        trend_filter_aligned = trend_filter.reindex(regime_df.index).fillna(False)

        # When trend filter is OFF, cap risk_on
        regime_df.loc[~trend_filter_aligned, "risk_on"] = regime_df.loc[~trend_filter_aligned, "risk_on"].clip(upper=trend_filter_risk_on_cap)

    # Apply volatility scaling if specified
    if vol_scaling_method != "none":
        vol_scaling = _compute_vol_scaling(prices, vol_scaling_method)
        vol_scaling_aligned = vol_scaling.reindex(regime_df.index).fillna(1.0)

        # Scale risk_on by vol factor
        regime_df["risk_on"] = regime_df["risk_on"] * vol_scaling_aligned
        # Ensure risk_on stays in [0, 1] range
        regime_df["risk_on"] = regime_df["risk_on"].clip(0, 1)

    return regime_df


def _smooth_regime_labels(
    regime_df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Apply rolling mode smoothing to regime labels.
    
    Args:
        regime_df: Regime DataFrame with date index and 'regime' column.
        window: Rolling window size in months (e.g., 3 = last 3 months).
    
    Returns:
        Smoothed regime DataFrame (same structure).
    """
    regime_df = regime_df.copy()
    regime_df = regime_df.sort_index()

    # Resample to monthly, take last value per month
    regime_monthly = regime_df.resample("ME").last()
    regimes = regime_monthly["regime"].copy()

    # Manual rolling mode for string data
    smoothed = []
    for i in range(len(regimes)):
        start_idx = max(0, i - window + 1)
        window_values = regimes.iloc[start_idx:i + 1]
        mode_val = window_values.mode()
        if len(mode_val) > 0:
            smoothed.append(mode_val.iloc[0])
        else:
            smoothed.append(regimes.iloc[i])

    regime_monthly["regime_smoothed"] = smoothed

    # Forward-fill to daily
    regime_daily = regime_monthly[["regime_smoothed"]].reindex(regime_df.index, method="ffill")
    regime_df["regime"] = regime_daily["regime_smoothed"].fillna(regime_df["regime"])

    return regime_df


def _compute_returns_and_setup(
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
    use_regime_smoothing: bool = False,
    regime_smoothing_window: int = 3,
    use_hybrid_signal: bool = False,
    hybrid_macro_weight: float = 0.5,
    market_lookback_months: int = 12,
    use_momentum: bool = False,
    trend_filter_type: str = "none",
    trend_filter_risk_on_cap: float = 0.3,
    vol_scaling_method: str = "none",
    portfolio_construction_method: str = "optimizer",
    momentum_12m_weight: float = 0.0,
    tickers: list[str] | None = None,
    assets: list[str] | None = None,
    risk_on_sleeve: list[str] | None = None,
    risk_off_sleeve: list[str] | None = None,
    use_vol_regime: bool = False,
    vol_regime_weight: float = 0.0,
    sigmoid_scale: float = 0.25,
    momentum_6m_weight: float = 0.0,
    breadth_weight: float = 0.0,
    breadth_prices: "pd.DataFrame | None" = None,
    vol_zscore_weight: float = 0.0,
    yield_curve_weight: float = 0.0,
    yield_curve_data: "pd.Series | None" = None,
    inversion_flag_offset: float = 0.0,
    breadth_flag_offset: float = 0.0,
    use_market_primary: bool = False,
    market_primary_stress_cap: float = 0.40,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float], pd.Series]:
    """Shared setup: returns, regime alignment, w_risk_on/off, equal_weight_returns.
    Matches original polars-based returns computation."""
    # Use provided universe or defaults
    tickers_to_use = tickers if tickers is not None else TICKERS
    assets_to_use = assets if assets is not None else ASSETS

    returns = prices[tickers_to_use].pct_change().iloc[1:]
    returns["cash"] = CASH_DAILY_YIELD
    regime_df = regime_df.sort_index()
    if regime_df.index.duplicated().any():
        regime_df = regime_df[~regime_df.index.duplicated(keep="last")]

    if use_regime_smoothing:
        regime_df = _smooth_regime_labels(regime_df, window=regime_smoothing_window)

    if use_market_primary:
        regime_df = _compute_market_primary_risk_on(
            prices, regime_df,
            market_lookback_months=market_lookback_months,
            sigmoid_scale=sigmoid_scale,
            macro_stress_cap=market_primary_stress_cap,
        )
    elif use_hybrid_signal:
        regime_df = _compute_hybrid_risk_on(
            prices, regime_df,
            macro_weight=hybrid_macro_weight,
            market_lookback_months=market_lookback_months,
            use_momentum=use_momentum,
            trend_filter_type=trend_filter_type,
            trend_filter_risk_on_cap=trend_filter_risk_on_cap,
            vol_scaling_method=vol_scaling_method,
            momentum_12m_weight=momentum_12m_weight,
            use_vol_regime=use_vol_regime,
            vol_regime_weight=vol_regime_weight,
            sigmoid_scale=sigmoid_scale,
            momentum_6m_weight=momentum_6m_weight,
            breadth_weight=breadth_weight,
            breadth_prices=breadth_prices,
            vol_zscore_weight=vol_zscore_weight,
            yield_curve_weight=yield_curve_weight,
            yield_curve_data=yield_curve_data,
            inversion_flag_offset=inversion_flag_offset,
            breadth_flag_offset=breadth_flag_offset,
        )
    elif trend_filter_type != "none":
        # Apply trend filter in non-hybrid mode (was previously unreachable)
        trend_filter = _compute_trend_filter(prices, trend_filter_type)
        trend_aligned = trend_filter.reindex(regime_df.index).fillna(False)
        regime_df.loc[~trend_aligned, "risk_on"] = (
            regime_df.loc[~trend_aligned, "risk_on"].clip(upper=trend_filter_risk_on_cap)
        )

    regime_df = regime_df.reindex(returns.index).ffill()
    regime_df = regime_df.reindex(returns.index)
    regime_df["regime"] = regime_df["regime"].astype(str).str.strip()

    # Portfolio construction: optimizer vs alternatives
    # For asset momentum methods, we return DataFrames with monthly weights
    # For other methods, we return static dicts
    w_risk_on_dynamic = None
    w_risk_off_dynamic = None

    if portfolio_construction_method == "optimizer":
        w_risk_on = _avg_alloc(allocations, RISK_ON_REGIMES, assets_to_use)
        w_risk_off = _avg_alloc(allocations, RISK_OFF_REGIMES, assets_to_use)
    elif portfolio_construction_method == "equal_weight":
        w_risk_on = _equal_weight_alloc(assets_to_use, is_risk_on=True,
                                         risk_on_sleeve=risk_on_sleeve, risk_off_sleeve=risk_off_sleeve)
        w_risk_off = _equal_weight_alloc(assets_to_use, is_risk_on=False,
                                          risk_on_sleeve=risk_on_sleeve, risk_off_sleeve=risk_off_sleeve)
    elif portfolio_construction_method == "asset_momentum_positive":
        # Asset-specific momentum filter (positive only) - dynamic
        momentum_ts = _compute_asset_momentum_timeseries(prices, assets_to_use, market_lookback_months)
        w_risk_on_dynamic = _compute_asset_momentum_allocations_timeseries(
            momentum_ts, assets_to_use, filter_method="positive", risk_on_sleeve=risk_on_sleeve
        )
        w_risk_on = None
        w_risk_off = _equal_weight_alloc(assets_to_use, is_risk_on=False,
                                          risk_on_sleeve=risk_on_sleeve, risk_off_sleeve=risk_off_sleeve)
    elif portfolio_construction_method == "asset_momentum_top3":
        # Asset-specific momentum (top 3) - dynamic
        momentum_ts = _compute_asset_momentum_timeseries(prices, assets_to_use, market_lookback_months)
        w_risk_on_dynamic = _compute_asset_momentum_allocations_timeseries(
            momentum_ts, assets_to_use, filter_method="top_n", top_n=3, risk_on_sleeve=risk_on_sleeve
        )
        w_risk_on = None
        w_risk_off = _equal_weight_alloc(assets_to_use, is_risk_on=False,
                                          risk_on_sleeve=risk_on_sleeve, risk_off_sleeve=risk_off_sleeve)
    elif portfolio_construction_method == "asset_momentum_top5":
        # Asset-specific momentum (top 5) - dynamic
        momentum_ts = _compute_asset_momentum_timeseries(prices, assets_to_use, market_lookback_months)
        w_risk_on_dynamic = _compute_asset_momentum_allocations_timeseries(
            momentum_ts, assets_to_use, filter_method="top_n", top_n=5, risk_on_sleeve=risk_on_sleeve
        )
        w_risk_on = None
        w_risk_off = _equal_weight_alloc(assets_to_use, is_risk_on=False,
                                          risk_on_sleeve=risk_on_sleeve, risk_off_sleeve=risk_off_sleeve)
    elif portfolio_construction_method == "xs_momentum_top4":
        # Cross-sectional momentum: rank risk-on sleeve by 12M momentum, select top 4.
        # The 12M lookback is fixed and independent of market_lookback_months (24M market signal).
        xs_mom_ts = _compute_asset_momentum_timeseries(prices, assets_to_use, 12)
        w_risk_on_dynamic = _compute_asset_momentum_allocations_timeseries(
            xs_mom_ts, assets_to_use, filter_method="top_n", top_n=4, risk_on_sleeve=risk_on_sleeve
        )
        w_risk_on = None
        w_risk_off = _equal_weight_alloc(assets_to_use, is_risk_on=False,
                                          risk_on_sleeve=risk_on_sleeve, risk_off_sleeve=risk_off_sleeve)
    elif portfolio_construction_method == "risk_parity":
        w_risk_on = _risk_parity_alloc(prices, assets_to_use, lookback_days=63)
        w_risk_off = _risk_parity_alloc(prices, assets_to_use, lookback_days=63)
    elif portfolio_construction_method == "heuristic":
        w_risk_on = _heuristic_alloc(is_risk_on=True)
        w_risk_off = _heuristic_alloc(is_risk_on=False)
    else:
        raise ValueError(f"Unknown portfolio_construction_method: {portfolio_construction_method}")

    # Store dynamic allocations in regime_df if present
    if w_risk_on_dynamic is not None:
        # Align to daily frequency
        w_risk_on_daily = w_risk_on_dynamic.reindex(returns.index, method="ffill")
        for asset in assets_to_use:
            if asset in w_risk_on_daily.columns:
                regime_df[f"w_risk_on_{asset}"] = w_risk_on_daily[asset]
            else:
                regime_df[f"w_risk_on_{asset}"] = 0.0

    equal_weight_returns = returns[tickers_to_use].mean(axis=1)
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
    quarterly_rebalance: bool = False,
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
        is_new_month = prev_month is None or month != prev_month
        is_quarter_start = month.month in (1, 4, 7, 10)
        if is_new_month and (not quarterly_rebalance or is_quarter_start):
            regime_stripped = str(regime).strip()

            # Check for dynamic risk-on allocations
            has_dynamic_risk_on = any(f"w_risk_on_{a}" in regime_df.columns for a in ASSETS)

            if use_stagflation_override and regime_stripped == "Stagflation" and "Stagflation" in allocations:
                current_weights = {str(k): float(v) for k, v in allocations["Stagflation"].items()}
            elif "risk_on" in regime_df.columns and not pd.isna(regime_df.loc[date, "risk_on"]):
                alpha = float(regime_df.loc[date, "risk_on"])
                if use_stagflation_risk_on_cap and regime_stripped == "Stagflation":
                    alpha = min(alpha, stagflation_risk_on_cap)

                # Use dynamic risk-on weights if available
                if has_dynamic_risk_on:
                    w_risk_on_monthly = {a: float(regime_df.loc[date, f"w_risk_on_{a}"]) for a in ASSETS}
                    current_weights = _blend_alloc(w_risk_off, w_risk_on_monthly, alpha, ASSETS)
                else:
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


def _apply_sleeve_cap(
    current: dict[str, float],
    sleeve_assets: frozenset,
    cap: float,
) -> dict[str, float]:
    """Cap each asset within a sleeve at `cap` fraction of the sleeve total, then renormalize.

    Uses iterative redistribution to handle cases where multiple assets hit the cap.
    The total sleeve weight in the full portfolio is preserved; only intra-sleeve
    fractions are adjusted.
    """
    rf = [a for a in sleeve_assets if a in current]
    rf_total = sum(current.get(a, 0.0) for a in rf)
    if rf_total < 1e-10 or not rf:
        return current

    fracs = {a: current[a] / rf_total for a in rf}
    for _ in range(30):
        over = [a for a in rf if fracs[a] > cap + 1e-9]
        if not over:
            break
        excess = sum(fracs[a] - cap for a in over)
        for a in over:
            fracs[a] = cap
        under = [a for a in rf if fracs[a] < cap - 1e-9]
        if not under:
            break
        per = excess / len(under)
        for a in under:
            fracs[a] += per

    total = sum(fracs.values())
    if total > 1e-10:
        fracs = {a: v / total for a, v in fracs.items()}

    result = dict(current)
    for a in rf:
        result[a] = fracs[a] * rf_total
    return result


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
    quarterly_rebalance: bool = False,
    tolerance: float = 0.0,
    rf_sleeve_cap: float = 0.0,
    rf_sleeve_assets: frozenset | None = None,
    vol_target_annual: float = 0.0,
    return_turnover_attribution: bool = False,
    use_post_blend_inv_vol: bool = True,
) -> pd.Series | tuple[pd.Series, pd.DataFrame] | tuple[pd.Series, pd.DataFrame, pd.Series] | tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Vectorized backtest: precompute vols, build weight matrix, vectorized returns."""
    rolling_std = returns[TICKERS].rolling(VOL_LOOKBACK, min_periods=1).std()

    dates = returns.index
    months = pd.Series(dates).dt.to_period("M").values
    month_changed = np.concatenate([[True], months[1:] != months[:-1]])
    if quarterly_rebalance:
        quarter_months = np.isin(pd.Series(dates).dt.month.values, [1, 4, 7, 10])
        month_changed = month_changed & quarter_months

    weight_cols = [a for a in ASSETS if a in returns.columns]
    weights = np.zeros((len(dates), len(weight_cols)))
    asset_idx = {a: i for i, a in enumerate(weight_cols)}
    cash_idx = weight_cols.index("cash") if "cash" in weight_cols else None
    scale_at_rebalance: list[tuple[pd.Timestamp, float]] = []  # (date, scale) when vol_target_annual > 0

    eq_w = 1.0 / (len(TICKERS) + 1)
    current = dict.fromkeys(TICKERS, eq_w)
    current["cash"] = eq_w
    prev_weights = np.array([current.get(a, 0.0) for a in weight_cols])

    ret_arr = returns[weight_cols].values  # needed for vol targeting and later for portfolio_ret

    # Check for dynamic risk-on allocations
    has_dynamic_risk_on = any(f"w_risk_on_{a}" in regime_df.columns for a in ASSETS)

    # Turnover attribution: store prev risk_on and std for incremental decomposition
    prev_ro: float | None = None
    prev_std_dict: dict[str, float | None] | None = None
    attribution_records: list[dict] = []

    for i, date in enumerate(dates):
        regime = regime_df.loc[date, "regime"]
        if pd.isna(regime):
            weights[i] = prev_weights
            continue

        if month_changed[i]:
            prev_start = prev_weights.copy() if return_turnover_attribution else None

            regime_stripped = str(regime).strip()
            if use_stagflation_override and regime_stripped == "Stagflation" and "Stagflation" in allocations:
                current = {str(k): float(v) for k, v in allocations["Stagflation"].items()}
            elif "risk_on" in regime_df.columns and not pd.isna(regime_df.loc[date, "risk_on"]):
                alpha = float(regime_df.loc[date, "risk_on"])
                if use_stagflation_risk_on_cap and regime_stripped == "Stagflation":
                    alpha = min(alpha, stagflation_risk_on_cap)

                # Use dynamic risk-on weights if available
                if has_dynamic_risk_on:
                    w_risk_on_monthly = {a: float(regime_df.loc[date, f"w_risk_on_{a}"]) for a in ASSETS}
                    current = _blend_alloc(w_risk_off, w_risk_on_monthly, alpha, ASSETS)
                else:
                    current = _blend_alloc(w_risk_off, w_risk_on, alpha, ASSETS)
            else:
                rk = REGIME_ALIASES.get(regime_stripped, regime_stripped)
                if rk in allocations:
                    current = {str(k): float(v) for k, v in allocations[rk].items()}
                else:
                    current = {a: 1.0 / len(ASSETS) for a in ASSETS}

            std_row = rolling_std.loc[date]
            std_dict = {a: float(std_row[a]) if a in std_row.index and pd.notna(std_row[a]) else None for a in TICKERS}
            if use_post_blend_inv_vol:
                current = vol_scaled_weights_from_std(current, std_dict, list(TICKERS))
            if rf_sleeve_cap > 0.0 and rf_sleeve_assets:
                current = _apply_sleeve_cap(current, rf_sleeve_assets, rf_sleeve_cap)
            new_w = np.array([current.get(a, 0.0) for a in weight_cols])
            if tolerance > 0.0:
                trade = new_w - prev_weights
                exec_trade = np.where(np.abs(trade) > tolerance, trade, 0.0)
                w_exec = prev_weights + exec_trade
                total = w_exec.sum()
                prev_weights = w_exec / total if total > 0 else new_w
            else:
                prev_weights = new_w

            # Turnover attribution (baseline path: risk_on blend + post-blend inv-vol + tau)
            if return_turnover_attribution and prev_start is not None and not has_dynamic_risk_on and "risk_on" in regime_df.columns and not pd.isna(regime_df.loc[date, "risk_on"]):
                alpha = float(regime_df.loc[date, "risk_on"])
                to_target_pre_tau = float(np.abs(new_w - prev_start).sum())
                to_removed_by_tau = float(np.abs(new_w - prev_weights).sum())
                to_executed = float(np.abs(prev_weights - prev_start).sum())
                if use_post_blend_inv_vol and prev_std_dict is not None:
                    w_hyp1 = vol_scaled_weights_from_std(
                        _blend_alloc(w_risk_off, w_risk_on, alpha, ASSETS),
                        prev_std_dict,
                        list(TICKERS),
                    )
                    w_hyp1_arr = np.array([w_hyp1.get(a, 0.0) for a in weight_cols])
                    to_signal = float(np.abs(w_hyp1_arr - prev_start).sum())
                    to_invvol = float(np.abs(new_w - w_hyp1_arr).sum())
                else:
                    # No inv-vol: target pre-tau is blend; signal = |blend - prev_start|, invvol = 0
                    w_hyp1_arr = new_w.copy()
                    to_signal = float(np.abs(w_hyp1_arr - prev_start).sum())
                    to_invvol = 0.0
                to_sleeve_internal = 0.0  # equal-weight sleeves: no sleeve-internal change
                valid_vols = [v for v in std_dict.values() if v is not None and not np.isnan(v)]
                realized_vol_avg = float(np.mean(valid_vols)) if valid_vols else np.nan
                attribution_records.append({
                    "date": date,
                    "risk_on": alpha,
                    "to_signal": to_signal,
                    "to_sleeve_internal": to_sleeve_internal,
                    "to_invvol": to_invvol,
                    "to_removed_by_tau": to_removed_by_tau,
                    "to_executed": to_executed,
                    "to_target_pre_tau": to_target_pre_tau,
                    "realized_vol_avg": realized_vol_avg,
                })
                prev_ro = alpha
                prev_std_dict = dict(std_dict)

            # Portfolio-level volatility targeting (post-construction, no leverage)
            if vol_target_annual > 0 and cash_idx is not None:
                if i >= 63:
                    port_ret_63 = (ret_arr[i - 63 : i] * weights[i - 63 : i]).sum(axis=1)
                    realized_vol = float(np.std(port_ret_63) * np.sqrt(252))
                    scale = min(1.0, vol_target_annual / max(realized_vol, 1e-10))
                    scale = max(0.0, scale)
                else:
                    scale = 1.0
                scale_at_rebalance.append((date, scale))
                prev_weights = prev_weights * scale
                prev_weights[cash_idx] += 1.0 - scale

        weights[i] = prev_weights

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
        scale_series = None
        if vol_target_annual > 0 and scale_at_rebalance:
            scale_dates, scale_vals = zip(*scale_at_rebalance)
            scale_series = pd.Series(scale_vals, index=pd.DatetimeIndex(scale_dates), name="vol_scale")
        if return_turnover_attribution and attribution_records:
            att_df = pd.DataFrame(attribution_records).set_index("date")
            return pd.Series(portfolio_ret, index=dates), weights_df, scale_series, att_df
        if scale_series is not None:
            return pd.Series(portfolio_ret, index=dates), weights_df, scale_series
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
    use_regime_smoothing: bool = False,
    regime_smoothing_window: int = 3,
    use_hybrid_signal: bool = False,
    hybrid_macro_weight: float = 0.5,
    market_lookback_months: int = 12,
    use_momentum: bool = False,
    trend_filter_type: str = "none",
    trend_filter_risk_on_cap: float = 0.3,
    vol_scaling_method: str = "none",
    portfolio_construction_method: str = "optimizer",
    momentum_12m_weight: float = 0.0,
    tickers: list[str] | None = None,
    assets: list[str] | None = None,
    risk_on_sleeve: list[str] | None = None,
    risk_off_sleeve: list[str] | None = None,
    use_vol_regime: bool = False,
    vol_regime_weight: float = 0.0,
    quarterly_rebalance: bool = False,
    tolerance: float = 0.0,
    sigmoid_scale: float = 0.25,
    rf_sleeve_cap: float = 0.0,
    momentum_6m_weight: float = 0.0,
    breadth_weight: float = 0.0,
    breadth_prices: "pd.DataFrame | None" = None,
    vol_zscore_weight: float = 0.0,
    yield_curve_weight: float = 0.0,
    yield_curve_data: "pd.Series | None" = None,
    inversion_flag_offset: float = 0.0,
    breadth_flag_offset: float = 0.0,
    vol_target_annual: float = 0.0,
    return_turnover_attribution: bool = False,
    use_post_blend_inv_vol: bool = True,
    use_market_primary: bool = False,
    market_primary_stress_cap: float = 0.40,
) -> pd.Series | tuple[pd.Series, pd.DataFrame] | tuple[pd.Series, pd.DataFrame, pd.Series] | tuple[pd.Series, pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Run backtest with given data. No DB. Returns portfolio returns (and optionally weights, scale series, attribution).

    Used by walk-forward evaluation to test on out-of-sample periods.

    Args:
        prices: Daily prices (date index, ticker columns).
        regime_df: Regime labels with date index.
        allocations: Regime -> {asset: weight}.
        return_weights: If True, return (returns, weights_df).
        use_market_primary: If True, use raw market momentum as primary signal
            (no expanding z-score) with macro stress override.
        market_primary_stress_cap: Max risk_on during Contraction/Stagflation
            when use_market_primary=True.

    Returns:
        Daily portfolio return series, or (returns, weights_df) if return_weights.
    """
    for alloc in allocations.values():
        if "cash" not in alloc:
            alloc["cash"] = 0.0
    returns, regime_df, w_risk_on, w_risk_off, _ = _compute_returns_and_setup(
        prices, regime_df, allocations, use_regime_smoothing, regime_smoothing_window,
        use_hybrid_signal, hybrid_macro_weight, market_lookback_months, use_momentum,
        trend_filter_type, trend_filter_risk_on_cap, vol_scaling_method,
        portfolio_construction_method, momentum_12m_weight,
        tickers, assets, risk_on_sleeve, risk_off_sleeve,
        use_vol_regime, vol_regime_weight, sigmoid_scale,
        momentum_6m_weight=momentum_6m_weight,
        breadth_weight=breadth_weight,
        breadth_prices=breadth_prices,
        vol_zscore_weight=vol_zscore_weight,
        yield_curve_weight=yield_curve_weight,
        yield_curve_data=yield_curve_data,
        inversion_flag_offset=inversion_flag_offset,
        breadth_flag_offset=breadth_flag_offset,
        use_market_primary=use_market_primary,
        market_primary_stress_cap=market_primary_stress_cap,
    )
    rf_assets_frozen = frozenset(risk_off_sleeve) if risk_off_sleeve else None
    result = _run_backtest_vectorized(
        returns, regime_df, allocations, w_risk_on, w_risk_off,
        returns[TICKERS].mean(axis=1),
        return_weights=return_weights,
        use_stagflation_override=use_stagflation_override,
        use_stagflation_risk_on_cap=use_stagflation_risk_on_cap,
        stagflation_risk_on_cap=stagflation_risk_on_cap,
        quarterly_rebalance=quarterly_rebalance,
        tolerance=tolerance,
        rf_sleeve_cap=rf_sleeve_cap,
        rf_sleeve_assets=rf_assets_frozen,
        vol_target_annual=vol_target_annual,
        return_turnover_attribution=return_turnover_attribution,
        use_post_blend_inv_vol=use_post_blend_inv_vol,
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
