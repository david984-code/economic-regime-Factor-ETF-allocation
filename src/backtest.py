from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

#  Settings
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


#  Tickers
TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
ASSETS = TICKERS + ["cash"]

# Load Price Data
prices = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)


VOL_LOOKBACK = 63  # ~3 months of trading days
VOL_EPS = 1e-8
MIN_VOL = 0.05  # floor vol to avoid crazy leverage in low-vol assets
MAX_VOL = 2.00  # cap vol to avoid div by huge numbers


def vol_scaled_weights(raw_w: pd.Series, trailing_rets: pd.DataFrame) -> pd.Series:
    """
    Scale weights by inverse volatility (approx risk parity on a per-asset basis).
    raw_w: Series indexed by assets incl 'cash'
    trailing_rets: daily returns for risky assets only (no cash column)
    """
    w = raw_w.copy()

    # cash not vol-scaled (treat as stable)
    risky_assets = [c for c in w.index if c != "cash" and c in trailing_rets.columns]

    if len(risky_assets) == 0:
        result: pd.Series = w / w.sum()
        return result

    vol = trailing_rets[risky_assets].std()  # daily vol
    vol = vol.clip(lower=MIN_VOL * vol.median(), upper=MAX_VOL * vol.median())
    vol = vol.replace(0.0, VOL_EPS).fillna(vol.median())

    # inverse-vol scaling
    w_risky = w[risky_assets] / vol
    w[risky_assets] = w_risky

    # normalize
    normalized: pd.Series = w / w.sum()
    return normalized


# converters
def dict_to_series(d: dict[str, float], cols: list[str]) -> pd.Series:
    return pd.Series({c: float(d.get(c, 0.0)) for c in cols}, index=cols)


def series_to_dict(s: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in s.to_dict().items()}


# yfinance sometimes returns a MultiIndex columns object
if isinstance(prices.columns, pd.MultiIndex):
    prices = prices["Adj Close"] if "Adj Close" in prices.columns.levels[0] else prices["Close"]
else:
    prices = prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]

prices = prices.dropna()


# Load Regime Labels
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
regime_df = pd.read_csv(OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"])
regime_df.set_index("date", inplace=True)
regime_df = regime_df.reindex(prices.index, method="ffill")
# Clean regime strings (prevents hidden whitespace / weird characters)
regime_df["regime"] = regime_df["regime"].astype(str).str.strip()


#  Load Optimized Allocations
opt_alloc_df = pd.read_csv(OUTPUTS_DIR / "optimal_allocations.csv")
opt_alloc_df["regime"] = opt_alloc_df["regime"].astype(str).str.strip()
opt_alloc_df.set_index("regime", inplace=True)

allocations = opt_alloc_df.to_dict(orient="index")

# Also normalize the dict keys (extra safety)
allocations = {str(k).strip(): v for k, v in allocations.items()}


#  Ensure All Regimes Include Stablecoin Allocation
for alloc in allocations.values():
    # ensure cash exists (optional sleeve)
    if "cash" not in alloc:
        alloc["cash"] = 0.0
# --- Regime label normalization + risk_on blending ---
REGIME_ALIASES = {
    # map any alternate labels produced upstream to your allocations keys
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


def _blend_alloc(w_off: dict[str, float], w_on: dict[str, float], alpha: float) -> dict[str, float]:
    # alpha in [0,1]; 0=risk_off, 1=risk_on
    alpha = float(np.clip(alpha, 0.0, 1.0))
    w = {
        a: (1.0 - alpha) * float(w_off.get(a, 0.0)) + alpha * float(w_on.get(a, 0.0))
        for a in ASSETS
    }
    s = sum(w.values())
    if s <= 0:
        return {a: 1.0 / len(ASSETS) for a in ASSETS}
    return {a: v / s for a, v in w.items()}


# endpoints for blending
W_RISK_ON = _avg_alloc(RISK_ON_REGIMES)
W_RISK_OFF = _avg_alloc(RISK_OFF_REGIMES)


# Build risk-on / risk-off anchors
w_recovery = dict_to_series({str(k): float(v) for k, v in allocations["Recovery"].items()}, ASSETS)
w_overheat = dict_to_series(
    {str(k): float(v) for k, v in allocations["Overheating"].items()}, ASSETS
)
w_contract = dict_to_series(
    {str(k): float(v) for k, v in allocations["Contraction"].items()}, ASSETS
)
w_stag = dict_to_series({str(k): float(v) for k, v in allocations["Stagflation"].items()}, ASSETS)

# Risk-on = favorable growth / inflation
w_on = 0.5 * (w_recovery + w_overheat)

# Risk-off = defensive macro
w_off = 0.5 * (w_contract + w_stag)

# Normalize anchors
w_on = w_on / w_on.sum()
w_off = w_off / w_off.sum()


# Calculate Daily Returns
returns = prices.pct_change().dropna()

# Add constant yield for synthetic cash (DAILY rate)
CASH_DAILY_YIELD = (1.045) ** (1 / 252) - 1  # ~4.5% annual, compounded daily
returns["cash"] = CASH_DAILY_YIELD


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


for date in returns.index:
    # Get the current regime
    regime = regime_df.loc[date, "regime"]

    # If we don't know the regime for this date, skip it
    if pd.isna(regime):
        portfolio_returns_list.append(np.nan)
        continue

    # Rebalance at the first trading day of each month (macro signal is monthly / forward-filled)
    month = date.to_period("M")
    if (prev_month is None) or (month != prev_month):
        # Prefer continuous risk_on (0..1) if present; fall back to discrete regime labels
        if "risk_on" in regime_df.columns and (not pd.isna(regime_df.loc[date, "risk_on"])):
            alpha = float(regime_df.loc[date, "risk_on"])
            current_weights = _blend_alloc(W_RISK_OFF, W_RISK_ON, alpha)
        else:
            regime_key = REGIME_ALIASES.get(str(regime).strip(), str(regime).strip())
            if regime_key in allocations:
                current_weights = {str(k): float(v) for k, v in allocations[regime_key].items()}
            else:
                print(
                    f"[WARNING] Unknown regime label '{regime}' on {date.date()} (keeping previous weights)"
                )

        # Volatility scaling (inverse-vol) so high-vol ETFs like MTUM don't over-dominate
        raw_w = dict_to_series(current_weights, ASSETS)
        trailing = returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)
        scaled_w = vol_scaled_weights(raw_w, trailing)
        current_weights = series_to_dict(scaled_w)

        prev_month = month
        prev_regime = regime

    # Compute daily return using current weights
    daily_return = sum(returns.loc[date, a] * float(current_weights.get(a, 0.0)) for a in ASSETS)
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
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily != 0 else float(np.nan)

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = float(drawdown.min())

    return {"CAGR": cagr, "Volatility": volatility, "Sharpe": sharpe, "Max Drawdown": max_dd}


# Results
metrics = compute_metrics(portfolio_returns, rf_daily=CASH_DAILY_YIELD)
bench_metrics = compute_metrics(equal_weight_returns, rf_daily=CASH_DAILY_YIELD)

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

print("\n============================")
print("[CURRENT] CURRENT TARGET WEIGHTS")
print(f"As-of date: {asof.date()}")
print(f"Regime label (ffill): {asof_regime}")
if asof_alpha is not None:
    print(f"risk_on (0..1): {asof_alpha:.3f}")
else:
    print("risk_on (0..1): NA (falling back to discrete regime)")
print("============================")

# Recompute the same way you do at rebalance
if asof_alpha is not None:
    base_weights = _blend_alloc(W_RISK_OFF, W_RISK_ON, asof_alpha)
else:
    rk: str = str(REGIME_ALIASES.get(str(asof_regime).strip(), str(asof_regime).strip()))
    base_weights_raw = allocations.get(rk, {a: 1.0 / len(ASSETS) for a in ASSETS})
    base_weights = {str(k): float(v) for k, v in base_weights_raw.items()}

# Apply the same vol scaling as the backtest rebalance
raw_w = dict_to_series(base_weights, ASSETS)
trailing = returns[TICKERS].loc[:asof].tail(VOL_LOOKBACK)
scaled_w = vol_scaled_weights(raw_w, trailing)

# Print nicely
out = scaled_w.sort_values(ascending=False)
for asset_key, v in out.items():
    asset_name: str = str(asset_key)
    print(f"{asset_name:>6}: {v:6.2%}")

# Save for reuse / handoff
out_df = out.rename("weight").reset_index().rename(columns={"index": "asset"})
out_df.to_csv(OUTPUTS_DIR / "current_factor_weights.csv", index=False)
print(f"\n[SUCCESS] Saved: {OUTPUTS_DIR / 'current_factor_weights.csv'}")
