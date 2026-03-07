"""Backtest module using Polars for high-performance computations.

This version uses:
- Polars for fast DataFrame operations (returns, rolling volatility)
- SQLite database instead of CSV files
- Optimized for large time series (3,700+ daily observations)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf

from .database import Database

#  Settings
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


#  Tickers
TICKERS = [
    "SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG",
    "IEF", "TLT",
]
ASSETS = TICKERS + ["cash"]

# Load Price Data
prices = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)


VOL_LOOKBACK = 63  # ~3 months of trading days
VOL_EPS = 1e-8
MIN_VOL = 0.05  # floor vol to avoid crazy leverage in low-vol assets
MAX_VOL = 2.00  # cap vol to avoid div by huge numbers

# Transaction costs: 8 bps per dollar traded (one-way; round-trip = 16 bps)
COST_BPS = 0.0008


def vol_scaled_weights_polars(
    raw_w: dict[str, float], trailing_rets_pl: pl.DataFrame, risky_assets: list[str]
) -> dict[str, float]:
    """
    Scale weights by inverse volatility using Polars for fast computation.

    Args:
        raw_w: Dict mapping asset -> weight (includes cash)
        trailing_rets_pl: Polars DataFrame with daily returns
        risky_assets: List of risky assets (excludes cash)

    Returns:
        Dict of volatility-scaled weights
    """
    w = raw_w.copy()

    if len(risky_assets) == 0:
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    # Fast standard deviation with Polars
    vol_pl = trailing_rets_pl.select([pl.col(col).std() for col in risky_assets])
    vol_values = vol_pl.row(0)  # Get first (and only) row as tuple

    vol_dict = dict(zip(risky_assets, vol_values))
    # Filter out None/NaN values before calculating median
    valid_vols = [v for v in vol_dict.values() if v is not None and not np.isnan(v)]
    if not valid_vols:
        # If no valid volatilities, return equal weights
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}
    vol_median = float(np.median(valid_vols))

    # Apply clipping and replacement
    for asset in risky_assets:
        v = vol_dict[asset]
        if v is None or not np.isfinite(v):
            v = vol_median if vol_median > 0 else VOL_EPS
        else:
            v = max(MIN_VOL * vol_median, min(MAX_VOL * vol_median, v))
            if v == 0.0:
                v = VOL_EPS
        vol_dict[asset] = v

    # Inverse-vol scaling
    for asset in risky_assets:
        w[asset] = w[asset] / vol_dict[asset]

    # Normalize
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


# converters
def dict_to_weights(d: dict[str, float]) -> dict[str, float]:
    """Ensure all assets are present in weights dict."""
    return {asset: float(d.get(asset, 0.0)) for asset in ASSETS}


# yfinance sometimes returns a MultiIndex columns object
if isinstance(prices.columns, pd.MultiIndex):
    prices = (
        prices["Adj Close"]
        if "Adj Close" in prices.columns.levels[0]
        else prices["Close"]
    )
else:
    prices = prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]

prices = prices.dropna()


# Load data from database
ROOT_DIR = Path(__file__).resolve().parent.parent
db = Database()

regime_df = db.load_regime_labels()
regime_df = regime_df.reindex(prices.index, method="ffill")
# Clean regime strings (prevents hidden whitespace / weird characters)
regime_df["regime"] = regime_df["regime"].astype(str).str.strip()

#  Load Optimized Allocations from database
allocations = db.load_optimal_allocations()

# Also normalize the dict keys (extra safety)
allocations = {str(k).strip(): v for k, v in allocations.items()}


#  Ensure All Regimes Include Cash Allocation
for alloc in allocations.values():
    if "cash" not in alloc:
        alloc["cash"] = 0.0

# --- Regime label normalization + risk_on blending ---
REGIME_ALIASES = {
    "Expansion": "Overheating",
    "Slowdown": "Contraction",
}

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}


def _avg_alloc(regimes: set[str]) -> dict[str, float]:
    regs = [r for r in regimes if r in allocations]
    if not regs:
        raise ValueError(f"None of these regimes found in allocations: {regimes}")
    out: dict[str, float] = dict.fromkeys(ASSETS, 0.0)
    for r in regs:
        for a in ASSETS:
            out[a] += float(allocations[r].get(a, 0.0))
    for a in ASSETS:
        out[a] /= len(regs)
    return out


def _blend_alloc(
    w_off: dict[str, float], w_on: dict[str, float], alpha: float
) -> dict[str, float]:
    # alpha in [0,1]; 0=risk_off, 1=risk_on
    alpha = float(np.clip(alpha, 0.0, 1.0))
    w = {
        a: (1.0 - alpha) * float(w_off.get(a, 0.0))
        + alpha * float(w_on.get(a, 0.0))
        for a in ASSETS
    }
    s = sum(w.values())
    if s <= 0:
        return {a: 1.0 / len(ASSETS) for a in ASSETS}
    return {a: v / s for a, v in w.items()}


# endpoints for blending
W_RISK_ON = _avg_alloc(RISK_ON_REGIMES)
W_RISK_OFF = _avg_alloc(RISK_OFF_REGIMES)


# Calculate Daily Returns using Polars for speed
# Convert prices to Polars DataFrame
prices_pd = prices.reset_index()
prices_pl = pl.from_pandas(prices_pd)
# Rename the date column (it might be 'Date', 'index', or the actual index name)
date_col = prices_pl.columns[0]
if date_col != "date":
    prices_pl = prices_pl.rename({date_col: "date"})

# Calculate pct_change using Polars (faster than pandas)
returns_pl = prices_pl.select(
    [pl.col("date")]
    + [
        ((pl.col(ticker) / pl.col(ticker).shift(1)) - 1).alias(ticker)
        for ticker in TICKERS
    ]
)

# Add constant yield for synthetic cash (DAILY rate)
CASH_DAILY_YIELD = (1.045) ** (1 / 252) - 1
returns_pl = returns_pl.with_columns(pl.lit(CASH_DAILY_YIELD).alias("cash"))

# Drop first row (NaN from pct_change)
returns_pl = returns_pl.slice(1)

# Convert back to pandas for date indexing compatibility
returns = returns_pl.to_pandas().set_index("date")

#  Equal-weight benchmark (no regime timing)
benchmark_assets = TICKERS  # only the ETFs, NOT cash
equal_weight_returns = returns[benchmark_assets].mean(axis=1)


#  Compute Portfolio Returns Based on Regime

portfolio_returns_list: list[float] = []
prev_regime = None
prev_month = None  # track monthly rebalances

# Start with equal weights as default
current_weights = {asset: 1 / (len(TICKERS) + 1) for asset in TICKERS}
current_weights["cash"] = 1 / (len(TICKERS) + 1)
prev_weights_for_cost = dict(current_weights)


for date in returns.index:
    # Get the current regime
    regime = regime_df.loc[date, "regime"]

    # If we don't know the regime for this date, skip it
    if pd.isna(regime):
        portfolio_returns_list.append(np.nan)
        continue

    # Rebalance at the first trading day of each month
    rebalanced = False
    month = date.to_period("M")
    if (prev_month is None) or (month != prev_month):
        # Prefer continuous risk_on (0..1) if present
        if "risk_on" in regime_df.columns and (
            not pd.isna(regime_df.loc[date, "risk_on"])
        ):
            alpha = float(regime_df.loc[date, "risk_on"])
            current_weights = _blend_alloc(W_RISK_OFF, W_RISK_ON, alpha)
        else:
            regime_key = REGIME_ALIASES.get(str(regime).strip(), str(regime).strip())
            if regime_key in allocations:
                current_weights = {
                    str(k): float(v) for k, v in allocations[regime_key].items()
                }
            else:
                print(
                    f"[WARNING] Unknown regime label '{regime}' on {date.date()} (keeping previous weights)"
                )

        # Volatility scaling using Polars for speed
        risky_assets = [a for a in TICKERS if a in current_weights]
        trailing_pd = returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)
        trailing_pl = pl.from_pandas(trailing_pd)

        current_weights = vol_scaled_weights_polars(
            current_weights, trailing_pl, risky_assets
        )

        prev_month = month
        prev_regime = regime
        rebalanced = True

    # Compute daily return using current weights
    daily_return = sum(
        returns.loc[date, a] * float(current_weights.get(a, 0.0)) for a in ASSETS
    )

    # Transaction costs on rebalance days
    if rebalanced:
        turnover = sum(
            abs(float(current_weights.get(a, 0.0)) - float(prev_weights_for_cost.get(a, 0.0)))
            for a in ASSETS
        )
        daily_return -= turnover * COST_BPS
        prev_weights_for_cost = dict(current_weights)

    portfolio_returns_list.append(daily_return)

# Convert to pandas Series for further analysis
portfolio_returns = pd.Series(portfolio_returns_list, index=returns.index)


#  Performance Metrics
def compute_metrics(rets: pd.Series, rf_daily: float = 0.0) -> dict[str, float]:
    rets = rets.dropna()
    excess = rets - rf_daily

    mean_daily = float(excess.mean())
    std_daily = float(excess.std())

    print("[RETURN] Mean daily EXCESS return:", mean_daily)
    print("[RISK] Std dev daily:", std_daily)

    equity_curve = (1 + rets).cumprod()
    n_days = len(rets)
    years = n_days / 252

    cagr = float(equity_curve.iloc[-1] ** (1 / years) - 1)
    volatility = float(rets.std() * np.sqrt(252))
    sharpe = (
        (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else float(np.nan)
    )

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = float(drawdown.min())

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }


# Results
metrics = compute_metrics(portfolio_returns, rf_daily=CASH_DAILY_YIELD)
bench_metrics = compute_metrics(equal_weight_returns, rf_daily=CASH_DAILY_YIELD)

# Save results to database
db.save_backtest_results(metrics, bench_metrics)

print("\n[PERFORMANCE] Portfolio Performance Based on Dynamic Regime Allocations:")
for k, v in metrics.items():
    if k == "Sharpe":
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v:.2%}")

print("\n[BENCHMARK] Equal-Weight Benchmark (no regime timing):")
for k, v in bench_metrics.items():
    if k == "Sharpe":
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v:.2%}")

# Current weights (as-of latest trading day)
asof = returns.index[-1]
asof_regime = regime_df.loc[asof, "regime"]
asof_alpha = None
if "risk_on" in regime_df.columns and (not pd.isna(regime_df.loc[asof, "risk_on"])):
    asof_alpha = float(regime_df.loc[asof, "risk_on"])

# Prefer ML forecast for next month when available (active trading)
next_month = (asof.to_period("M") + 1).strftime("%Y-%m")
forecast = db.load_latest_regime_forecast(next_month)
use_forecast = forecast is not None

print("\n============================")
print("[CURRENT] CURRENT TARGET WEIGHTS")
print(f"As-of date: {asof.date()}")
print(f"Regime label (ffill): {asof_regime}")
if use_forecast:
    real_str = f"{asof_alpha:.3f}" if asof_alpha is not None else "NA"
    print(f"risk_on: {real_str} (realized) | Next month forecast: {forecast['risk_on_forecast']:.3f}")
    blend_alpha = 0.5 * (asof_alpha or 0.5) + 0.5 * forecast["risk_on_forecast"]
else:
    if asof_alpha is not None:
        print(f"risk_on (0..1): {asof_alpha:.3f}")
        blend_alpha = asof_alpha
    else:
        print("risk_on (0..1): NA (falling back to discrete regime)")
        blend_alpha = None
print("============================")

# Use forecast blend when available for more forward-looking allocation
alpha_for_weights = blend_alpha if blend_alpha is not None else asof_alpha

if alpha_for_weights is not None:
    base_weights = _blend_alloc(W_RISK_OFF, W_RISK_ON, alpha_for_weights)
else:
    rk: str = str(REGIME_ALIASES.get(str(asof_regime).strip(), str(asof_regime).strip()))
    base_weights_raw = allocations.get(rk, {a: 1.0 / len(ASSETS) for a in ASSETS})
    base_weights = {str(k): float(v) for k, v in base_weights_raw.items()}

# Apply the same vol scaling as the backtest rebalance
risky_assets = list(TICKERS)
trailing_pd = returns[TICKERS].loc[:asof].tail(VOL_LOOKBACK)
trailing_pl = pl.from_pandas(trailing_pd)

scaled_weights = vol_scaled_weights_polars(base_weights, trailing_pl, risky_assets)

# Print nicely
sorted_weights = sorted(scaled_weights.items(), key=lambda x: x[1], reverse=True)
for asset_name, weight in sorted_weights:
    print(f"{asset_name:>6}: {weight:6.2%}")

# Save to database
db.save_current_weights(str(asof.date()), pd.Series(scaled_weights))

db.close()

print(f"\n[SUCCESS] Saved results to database: {db.db_path}")
