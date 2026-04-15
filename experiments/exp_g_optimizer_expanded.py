"""
EXPERIMENT G -- Optimizer Re-run with Expanded Universe
========================================================

Re-runs the Sortino optimizer from scratch with EFA, EEM, and TIP added
to the asset universe. Unlike Experiment E (which carved from existing
allocations), this lets the optimizer find its own regime-specific weights
for the new assets.

Compares the expanded-universe optimized allocation against the current
8-ticker baseline through the full backtest pipeline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

from overrides import (
    apply_ma_filter,
    apply_momentum_tilt,
    apply_vix_override,
    check_intramonth_trigger,
    fetch_market_filters,
)

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

BASE_TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
EXPANDED_TICKERS = BASE_TICKERS + ["EFA", "EEM", "TIP"]

VOL_LOOKBACK = 63
VOL_EPS = 1e-8
MIN_VOL = 0.05
MAX_VOL = 2.00
CASH_DAILY_YIELD = (1.045) ** (1 / 252) - 1

USE_VIX_OVERRIDE = True
VIX_THRESHOLD = 25.0
VIX_RISK_ON_CAP = 0.25
USE_MA_FILTER = True
MA_LOOKBACK = 200
MA_EQUITY_CAP = 0.30
VIX_INTRAMONTH_THRESHOLD = 27.0
SPY_WEEKLY_DROP_THRESHOLD = 0.99
COOLDOWN_DAYS = 5
MOMENTUM_LOOKBACK = 10
MOMENTUM_STRENGTH = 0.20
MOMENTUM_MAX_TILT = 2.0

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}


def load_data(tickers):
    print(f"Downloading prices for {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = (
            raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
        )
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    prices = prices.dropna()

    print(
        f"  {len(prices)} days: {prices.index.min().date()} to {prices.index.max().date()}"
    )

    mkt_filters = fetch_market_filters(START_DATE, END_DATE)

    outputs = ROOT / "outputs"
    regime_df = pd.read_csv(
        outputs / "regime_labels_expanded.csv", parse_dates=["date"]
    )
    regime_df.set_index("date", inplace=True)
    regime_df = regime_df.reindex(prices.index, method="ffill")
    regime_df["regime"] = regime_df["regime"].astype(str).str.strip()

    return prices, mkt_filters, regime_df


def optimize_allocations(prices, regime_df, tickers):
    """Run Sortino-optimized allocation per regime for given tickers."""
    tickers + ["cash"]
    monthly_px = prices[tickers].resample("ME").last()
    returns = monthly_px.pct_change().dropna()
    returns["cash"] = (1.045) ** (1 / 12) - 1

    regime_monthly = regime_df[["regime"]].resample("ME").last().dropna()
    merged = returns.join(regime_monthly, how="inner")

    risky = tickers
    full_list = risky + ["cash"]
    n = len(full_list)

    min_cash, max_cash = 0.05, 0.15

    def neg_sortino(weights, mean_rets, cov_mat):
        w_risky = np.asarray(weights[:-1], dtype=float)
        cash_w = float(weights[-1])
        risky_sum = w_risky.sum()
        if risky_sum <= 1e-10:
            return 1e9
        w = w_risky / risky_sum
        mu = np.asarray(mean_rets, dtype=float)
        port_ret = float(np.dot(w, mu))
        port_var = float(w.T @ cov_mat @ w)
        if not np.isfinite(port_var) or port_var <= 1e-12:
            return 1e9
        port_vol = np.sqrt(port_var)
        sharpe = port_ret / port_vol if port_vol > 0 else 0
        if not np.isfinite(sharpe):
            return 1e9
        return -sharpe + 0.05 * cash_w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: w[-1] - min_cash},
        {"type": "ineq", "fun": lambda w: max_cash - w[-1]},
    ]
    bounds = [(0.0, 1.0)] * n

    allocations = {}
    for regime in merged["regime"].unique():
        if pd.isna(regime):
            continue
        regime = str(regime).strip()
        subset = merged[merged["regime"] == regime][risky].fillna(0)
        if len(subset) < 3:
            print(f"  Skipping {regime}: only {len(subset)} months")
            continue

        mean_r = subset.mean().values.astype(float)
        cov_r = subset.cov().values.astype(float) + 1e-6 * np.eye(len(risky))

        start_cash = (min_cash + max_cash) / 2
        x0 = np.full(n, (1 - start_cash) / (n - 1))
        x0[-1] = start_cash

        result = minimize(
            neg_sortino,
            x0,
            args=(mean_r, cov_r),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            alloc = dict(zip(full_list, result.x, strict=False))
            allocations[regime] = alloc
            top3 = sorted(alloc.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}={v:.0%}" for k, v in top3)
            print(f"  {regime}: {top_str}")
        else:
            print(f"  {regime}: optimizer FAILED ({result.message})")

    return allocations


# Helpers
def dict_to_series(d, cols):
    return pd.Series({c: float(d.get(c, 0.0)) for c in cols}, index=cols)


def series_to_dict(s):
    return {k: float(v) for k, v in s.to_dict().items()}


def vol_scaled_weights(raw_w, trailing_rets):
    w = raw_w.copy()
    risky = [c for c in w.index if c != "cash" and c in trailing_rets.columns]
    if not risky:
        return w / w.sum()
    vol = trailing_rets[risky].std()
    vol = vol.clip(lower=MIN_VOL * vol.median(), upper=MAX_VOL * vol.median())
    vol = vol.replace(0.0, VOL_EPS).fillna(vol.median())
    w[risky] = w[risky] / vol
    return w / w.sum()


def _avg_alloc(regimes, allocations, assets):
    regs = [r for r in regimes if r in allocations]
    if not regs:
        return {a: 1.0 / len(assets) for a in assets}
    out = dict.fromkeys(assets, 0.0)
    for r in regs:
        for a in assets:
            out[a] += float(allocations[r].get(a, 0.0))
    for a in assets:
        out[a] /= len(regs)
    return out


def _blend_alloc(w_off, w_on, alpha, assets):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    w = {
        a: (1.0 - alpha) * float(w_off.get(a, 0.0)) + alpha * float(w_on.get(a, 0.0))
        for a in assets
    }
    s = sum(w.values())
    if s <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    return {a: v / s for a, v in w.items()}


def run_backtest(prices, mkt_filters, regime_df, allocations, tickers, label=""):
    assets = tickers + ["cash"]
    asset_prices = prices[[t for t in tickers if t in prices.columns]].copy()
    returns = asset_prices.pct_change().dropna()
    returns["cash"] = CASH_DAILY_YIELD

    equity_tickers = [t for t in tickers if t not in ("GLD", "TIP", "cash")]
    W_ON = _avg_alloc(RISK_ON_REGIMES, allocations, assets)
    W_OFF = _avg_alloc(RISK_OFF_REGIMES, allocations, assets)

    def calc_weights(date):
        row = regime_df.loc[date] if date in regime_df.index else None
        alpha = None
        if (
            row is not None
            and "risk_on" in regime_df.columns
            and not pd.isna(row["risk_on"])
        ):
            alpha = float(row["risk_on"])
        if USE_VIX_OVERRIDE and alpha is not None:
            alpha = apply_vix_override(
                alpha, mkt_filters, date, VIX_THRESHOLD, VIX_RISK_ON_CAP
            )
        if alpha is not None:
            w = _blend_alloc(W_OFF, W_ON, alpha, assets)
        else:
            regime_val = (
                str(row["regime"]).strip() if row is not None else "Contraction"
            )
            regime_key = REGIME_ALIASES.get(regime_val, regime_val)
            w = allocations.get(regime_key, {a: 1.0 / len(assets) for a in assets})

        raw_w = dict_to_series(w, assets)
        trail = returns[tickers].loc[:date].tail(VOL_LOOKBACK)
        scaled = vol_scaled_weights(raw_w, trail)
        w = series_to_dict(scaled)

        if USE_MA_FILTER and "SPY" in asset_prices.columns:
            w = apply_ma_filter(
                w,
                asset_prices["SPY"].loc[:date],
                equity_tickers,
                MA_LOOKBACK,
                MA_EQUITY_CAP,
            )
        w = apply_momentum_tilt(
            w,
            asset_prices,
            date,
            lookback=MOMENTUM_LOOKBACK,
            strength=MOMENTUM_STRENGTH,
            max_tilt=MOMENTUM_MAX_TILT,
        )
        return w

    port_rets = []
    prev_month = None
    current_w = {a: 1.0 / len(assets) for a in assets}
    last_intra_idx = -COOLDOWN_DAYS - 1
    intramonth_count = 0

    date_list = list(returns.index)
    for i, date in enumerate(date_list):
        regime = regime_df.loc[date, "regime"] if date in regime_df.index else np.nan
        if pd.isna(regime):
            port_rets.append(np.nan)
            continue

        month = date.to_period("M")
        if prev_month is None or month != prev_month:
            current_w = calc_weights(date)
            prev_month = month
            last_intra_idx = i - COOLDOWN_DAYS - 1
        elif (i - last_intra_idx) >= COOLDOWN_DAYS and date.weekday() == 4:
            try:
                triggered, _ = check_intramonth_trigger(
                    mkt_filters,
                    date,
                    VIX_INTRAMONTH_THRESHOLD,
                    SPY_WEEKLY_DROP_THRESHOLD,
                )
            except (ValueError, KeyError):
                triggered = False
            if triggered:
                current_w = calc_weights(date)
                intramonth_count += 1
                last_intra_idx = i

        daily_ret = sum(
            returns.loc[date, a] * float(current_w.get(a, 0.0))
            for a in assets
            if a in returns.columns
        )
        port_rets.append(daily_ret)

    port_rets = pd.Series(port_rets, index=returns.index).dropna()
    eq = (1 + port_rets).cumprod()
    n_years = len(port_rets) / 252
    cagr = eq.iloc[-1] ** (1 / n_years) - 1
    vol = port_rets.std() * np.sqrt(252)
    sharpe = (
        (port_rets.mean() / port_rets.std()) * np.sqrt(252)
        if port_rets.std() > 0
        else 0
    )
    dd = (eq / eq.cummax() - 1).min()
    downside = port_rets[port_rets < 0].std() * np.sqrt(252)
    sortino = (port_rets.mean() * 252) / downside if downside > 0 else 0

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": dd,
        "vol": vol,
        "intramonth": intramonth_count,
    }


def main():
    prices_expanded, mkt_filters, regime_df = load_data(EXPANDED_TICKERS)
    prices_base, _, _ = load_data(BASE_TICKERS)

    print("\n" + "=" * 70)
    print("EXPERIMENT G: Optimizer Re-run with Expanded Universe")
    print("=" * 70)

    # Baseline: optimize on 8 tickers
    print("\nOptimizing BASELINE (8 tickers) ...")
    base_alloc = optimize_allocations(prices_base, regime_df, BASE_TICKERS)

    # Expanded: optimize on 11 tickers
    print("\nOptimizing EXPANDED (11 tickers: +EFA, EEM, TIP) ...")
    exp_alloc = optimize_allocations(prices_expanded, regime_df, EXPANDED_TICKERS)

    # Run backtests
    print("\n> Backtest BASELINE ...")
    base_result = run_backtest(
        prices_base, mkt_filters, regime_df, base_alloc, BASE_TICKERS, "BASE"
    )

    print("> Backtest EXPANDED ...")
    exp_result = run_backtest(
        prices_expanded, mkt_filters, regime_df, exp_alloc, EXPANDED_TICKERS, "EXPANDED"
    )

    # Print allocations comparison
    print("\n" + "=" * 70)
    print("OPTIMIZED ALLOCATIONS COMPARISON")
    print("=" * 70)
    for regime in sorted(set(list(base_alloc.keys()) + list(exp_alloc.keys()))):
        print(f"\n  {regime}:")
        ba = base_alloc.get(regime, {})
        ea = exp_alloc.get(regime, {})
        all_assets = sorted(set(list(ba.keys()) + list(ea.keys())))
        print(f"    {'Asset':<8} {'Base':>8} {'Expanded':>10} {'Delta':>8}")
        for a in all_assets:
            bw = float(ba.get(a, 0))
            ew = float(ea.get(a, 0))
            if bw > 0.005 or ew > 0.005:
                print(f"    {a:<8} {bw:>7.1%} {ew:>9.1%} {ew - bw:>+7.1%}")

    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    hdr = f"{'Config':<14} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Vol':>7}"
    print(hdr)
    print("-" * len(hdr))

    for label, m in [("BASELINE", base_result), ("EXPANDED", exp_result)]:
        print(
            f"{label:<14} {m['cagr']:>6.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
            f" {m['max_dd']:>7.2%} {m['vol']:>6.2%}"
        )

    ds = exp_result["sharpe"] - base_result["sharpe"]
    dc = exp_result["cagr"] - base_result["cagr"]
    dd = exp_result["max_dd"] - base_result["max_dd"]

    print(f"\n  Delta:  Sharpe {ds:+.3f}  |  CAGR {dc:+.2%}  |  MaxDD {dd:+.2%}")

    if ds >= 0.02 or dc >= 0.0025:
        print("\n  >> PASS -- escalate to full walk-forward validation")
    else:
        print("\n  >> REJECT -- below escalation thresholds")


if __name__ == "__main__":
    main()
