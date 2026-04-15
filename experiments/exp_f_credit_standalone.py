"""
EXPERIMENT F -- Credit Spread as Primary Intramonth Trigger
============================================================

Hypothesis
----------
A credit-spread-based signal (HYG/LQD ratio deterioration) used as the
*primary* intramonth trigger — replacing VIX entirely — will produce
better risk-adjusted returns because credit deterioration leads equity
drawdowns by days/weeks, fires at different times than VIX, and captures
risk dimensions that equity vol misses.

Variable changed
----------------
Intramonth trigger signal source. VIX removed; replaced by:
  1. HYG/LQD ratio z-score (rolling mean reversion)
  2. HYG trailing return (momentum-based, from Exp D)
  3. Combined z-score + momentum

Control: VIX-only trigger (production baseline)
All other parameters (allocation, vol-scaling, MA filter, momentum tilt) unchanged.

Data note: FRED API unavailable; using yfinance HYG and LQD ETF prices as proxies.
HYG/LQD ratio approximates credit spread: when HY underperforms IG, spreads widen.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from overrides import (
    apply_ma_filter,
    apply_momentum_tilt,
    apply_vix_override,
    fetch_market_filters,
)

# ── Static settings (match production) ----------------------------------------
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
ASSETS = TICKERS + ["cash"]

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
EQUITY_TICKERS_LIST = ["SPY", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]

MOMENTUM_LOOKBACK = 10
MOMENTUM_STRENGTH = 0.20
MOMENTUM_MAX_TILT = 2.0

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}


# ── Data loading ---------------------------------------------------------------


def load_data():
    all_tickers = TICKERS + ["HYG", "LQD"]
    print("Downloading price data (incl HYG + LQD) ...")
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = (
            raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
        )
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    prices = prices.dropna()

    hyg = prices["HYG"].copy()
    lqd = prices["LQD"].copy()
    asset_prices = prices[TICKERS].copy()

    # HYG/LQD ratio — falling = credit stress (HY underperforms IG)
    ratio = hyg / lqd
    print(f"  HYG/LQD ratio: {ratio.iloc[-1]:.4f} (mean: {ratio.mean():.4f})")

    returns = asset_prices.pct_change().dropna()
    returns["cash"] = CASH_DAILY_YIELD

    print("Fetching VIX + SPY + HYG for market filters ...")
    mkt_filters = fetch_market_filters(START_DATE, END_DATE)

    outputs = ROOT / "outputs"
    regime_df = pd.read_csv(
        outputs / "regime_labels_expanded.csv", parse_dates=["date"]
    )
    regime_df.set_index("date", inplace=True)
    regime_df = regime_df.reindex(asset_prices.index, method="ffill")
    regime_df["regime"] = regime_df["regime"].astype(str).str.strip()

    opt_alloc_df = pd.read_csv(outputs / "optimal_allocations.csv")
    opt_alloc_df["regime"] = opt_alloc_df["regime"].astype(str).str.strip()
    opt_alloc_df.set_index("regime", inplace=True)
    allocations = {
        str(k).strip(): v for k, v in opt_alloc_df.to_dict(orient="index").items()
    }
    for alloc in allocations.values():
        if "cash" not in alloc:
            alloc["cash"] = 0.0

    return asset_prices, returns, mkt_filters, regime_df, allocations, hyg, lqd, ratio


# ── Helpers (match production) -------------------------------------------------


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


def _avg_alloc(regimes, allocations):
    regs = [r for r in regimes if r in allocations]
    out = dict.fromkeys(ASSETS, 0.0)
    for r in regs:
        for a in ASSETS:
            out[a] += float(allocations[r].get(a, 0.0))
    for a in ASSETS:
        out[a] /= len(regs)
    return out


def _blend_alloc(w_off, w_on, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    w = {
        a: (1.0 - alpha) * float(w_off.get(a, 0.0)) + alpha * float(w_on.get(a, 0.0))
        for a in ASSETS
    }
    s = sum(w.values())
    if s <= 0:
        return {a: 1.0 / len(ASSETS) for a in ASSETS}
    return {a: v / s for a, v in w.items()}


# ── Credit spread signal construction -----------------------------------------


def build_credit_signals(
    hyg: pd.Series, lqd: pd.Series, ratio: pd.Series
) -> pd.DataFrame:
    """Build credit spread signals from HYG/LQD data.

    Returns DataFrame with columns:
      ratio_zscore: rolling z-score of HYG/LQD ratio (negative = stress)
      hyg_10d_ret: HYG trailing 10-day return
      hyg_20d_ret: HYG trailing 20-day return
      ratio_10d_chg: 10-day change in ratio (negative = widening)
    """
    sig = pd.DataFrame(index=ratio.index)

    # Rolling z-score of ratio (60-day window)
    ratio_ma = ratio.rolling(60, min_periods=30).mean()
    ratio_std = ratio.rolling(60, min_periods=30).std()
    sig["ratio_zscore"] = (ratio - ratio_ma) / ratio_std.replace(0, np.nan)

    sig["hyg_10d_ret"] = hyg.pct_change(10)
    sig["hyg_20d_ret"] = hyg.pct_change(20)
    sig["ratio_10d_chg"] = ratio.pct_change(10)

    return sig


# ── Backtest engine -----------------------------------------------------------


def run_backtest(
    prices,
    returns,
    mkt_filters,
    regime_df,
    allocations,
    credit_signals: pd.DataFrame,
    *,
    trigger_mode: str = "vix_only",
    # VIX params
    vix_hi: float = 27.0,
    # Credit z-score params
    zscore_stress: float = -1.5,
    zscore_recovery: float = -0.5,
    # HYG momentum params
    hyg_stress: float = -0.02,
    hyg_recovery: float = 0.00,
    hyg_memory: int = 20,
    # General
    cooldown_days: int = 5,
) -> dict:
    """Run backtest with configurable intramonth trigger.

    trigger_mode:
      "vix_only"       : production VIX > vix_hi (baseline)
      "zscore_only"    : HYG/LQD ratio z-score
      "hyg_mom_only"   : HYG momentum (same as Exp D credit_only)
      "zscore_hyg"     : z-score OR HYG momentum
      "zscore_replaces": z-score replaces VIX entirely (no VIX at all)
    """
    W_ON = _avg_alloc(RISK_ON_REGIMES, allocations)
    W_OFF = _avg_alloc(RISK_OFF_REGIMES, allocations)

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
            w = _blend_alloc(W_OFF, W_ON, alpha)
        else:
            regime_val = (
                str(row["regime"]).strip() if row is not None else "Contraction"
            )
            regime_key = REGIME_ALIASES.get(regime_val, regime_val)
            w = allocations.get(regime_key, {a: 1.0 / len(ASSETS) for a in ASSETS})

        raw_w = dict_to_series(w, ASSETS)
        trail = returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)
        scaled = vol_scaled_weights(raw_w, trail)
        w = series_to_dict(scaled)

        if USE_MA_FILTER and "SPY" in prices.columns:
            w = apply_ma_filter(
                w,
                prices["SPY"].loc[:date],
                EQUITY_TICKERS_LIST,
                MA_LOOKBACK,
                MA_EQUITY_CAP,
            )

        w = apply_momentum_tilt(
            w,
            prices,
            date,
            lookback=MOMENTUM_LOOKBACK,
            strength=MOMENTUM_STRENGTH,
            max_tilt=MOMENTUM_MAX_TILT,
        )
        return w

    def check_vix(date):
        if date not in mkt_filters.index:
            return False, ""
        vix = mkt_filters.loc[date, "VIX_close"]
        if pd.isna(vix):
            return False, ""
        if float(vix) > vix_hi:
            return True, f"VIX={float(vix):.1f}>{vix_hi}"
        return False, ""

    def check_zscore(date):
        if date not in credit_signals.index:
            return False, ""
        z = credit_signals.loc[date, "ratio_zscore"]
        if pd.isna(z):
            return False, ""
        z = float(z)

        # Defensive: ratio z-score very negative (HY underperforming IG badly)
        if z < zscore_stress:
            return True, f"HYG/LQD z={z:.2f}<{zscore_stress} (credit stress)"

        # Recovery: z-score rising back through recovery threshold after stress
        if z > zscore_recovery:
            trailing_z = credit_signals["ratio_zscore"].loc[:date].tail(hyg_memory)
            if (trailing_z < zscore_stress).any():
                return True, f"HYG/LQD z={z:.2f} recovery after stress"

        return False, ""

    def check_hyg_mom(date):
        if date not in credit_signals.index:
            return False, ""
        ret = credit_signals.loc[date, "hyg_10d_ret"]
        if pd.isna(ret):
            return False, ""
        ret = float(ret)

        if ret < hyg_stress:
            return True, f"HYG 10d={ret:.2%} (stress)"

        if ret > hyg_recovery:
            trailing = credit_signals["hyg_10d_ret"].loc[:date].tail(hyg_memory)
            if (trailing < hyg_stress).any():
                return True, f"HYG 10d={ret:.2%} recovery"

        return False, ""

    def check_trigger(date):
        if trigger_mode == "vix_only":
            return check_vix(date)
        if trigger_mode == "zscore_only":
            return check_zscore(date)
        if trigger_mode == "hyg_mom_only":
            return check_hyg_mom(date)
        if trigger_mode == "zscore_hyg":
            t1, r1 = check_zscore(date)
            if t1:
                return t1, r1
            return check_hyg_mom(date)
        if trigger_mode == "zscore_replaces":
            return check_zscore(date)
        return False, ""

    # Rebalance loop
    port_rets = []
    prev_month = None
    current_w = {a: 1.0 / len(ASSETS) for a in ASSETS}
    prev_w = None
    total_turnover = 0.0
    rebalance_count = 0
    intramonth_count = 0
    intramonth_log = []
    days_since_intra = cooldown_days + 1

    for date in returns.index:
        regime = regime_df.loc[date, "regime"] if date in regime_df.index else np.nan
        if pd.isna(regime):
            port_rets.append(np.nan)
            days_since_intra += 1
            continue

        month = date.to_period("M")
        did_rebalance = False

        if prev_month is None or month != prev_month:
            current_w = calc_weights(date)
            prev_month = month
            did_rebalance = True
            rebalance_count += 1
            days_since_intra = cooldown_days + 1

        elif days_since_intra >= cooldown_days and date.weekday() == 4:
            triggered, reason = check_trigger(date)
            if triggered:
                current_w = calc_weights(date)
                did_rebalance = True
                intramonth_count += 1
                intramonth_log.append((date, reason))
                days_since_intra = 0

        if did_rebalance and prev_w is not None:
            to = (
                sum(
                    abs(float(current_w.get(a, 0)) - float(prev_w.get(a, 0)))
                    for a in ASSETS
                )
                / 2.0
            )
            total_turnover += to
        if did_rebalance:
            prev_w = current_w.copy()

        daily_ret = sum(
            returns.loc[date, a] * float(current_w.get(a, 0.0)) for a in ASSETS
        )
        port_rets.append(daily_ret)
        days_since_intra += 1

    port_rets = pd.Series(port_rets, index=returns.index).dropna()

    # Metrics
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
    turnover_annual = total_turnover / n_years if n_years > 0 else 0
    downside = port_rets[port_rets < 0].std() * np.sqrt(252)
    sortino = (port_rets.mean() * 252) / downside if downside > 0 else 0

    subperiods = {
        "2013-2017": ("2013-01-01", "2017-12-31"),
        "2018-2020": ("2018-01-01", "2020-12-31"),
        "2021-2022": ("2021-01-01", "2022-12-31"),
        "2023-2025": ("2023-01-01", "2025-12-31"),
        "2026 YTD": ("2026-01-01", END_DATE),
    }
    sub_sharpes = {}
    for name, (s, e) in subperiods.items():
        mask = (port_rets.index >= s) & (port_rets.index <= e)
        sr = port_rets[mask]
        if len(sr) > 20:
            sub_sharpes[name] = (
                (sr.mean() / sr.std()) * np.sqrt(252) if sr.std() > 0 else 0
            )
        else:
            sub_sharpes[name] = None

    bear_periods = [
        ("COVID crash", "2020-02-01", "2020-04-01"),
        ("2022 bear", "2022-01-01", "2022-10-31"),
        ("2018 Q4", "2018-10-01", "2018-12-31"),
        ("2016 oil scare", "2016-01-01", "2016-02-28"),
    ]
    bear_rets = {}
    for pname, s, e in bear_periods:
        mask = (port_rets.index >= s) & (port_rets.index <= e)
        br = port_rets[mask]
        if len(br) > 0:
            beq = (1 + br).cumprod()
            bear_rets[pname] = {
                "return": beq.iloc[-1] - 1,
                "max_dd": (beq / beq.cummax() - 1).min(),
            }

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": dd,
        "vol": vol,
        "turnover": turnover_annual,
        "rebalances": rebalance_count,
        "intramonth": intramonth_count,
        "intramonth_log": intramonth_log,
        "sub_sharpes": sub_sharpes,
        "bear_periods": bear_rets,
    }


# ── Main -----------------------------------------------------------------------


def main():
    prices, returns, mkt_filters, regime_df, allocations, hyg, lqd, ratio = load_data()
    credit_signals = build_credit_signals(hyg, lqd, ratio)

    print("\nCredit signal stats:")
    print(
        f"  ratio_zscore: mean={credit_signals['ratio_zscore'].mean():.3f}, "
        f"std={credit_signals['ratio_zscore'].std():.3f}"
    )
    print(f"  Days z < -1.5: {(credit_signals['ratio_zscore'] < -1.5).sum()}")
    print(f"  Days z < -2.0: {(credit_signals['ratio_zscore'] < -2.0).sum()}")

    print("\n" + "=" * 80)
    print("EXPERIMENT F: Credit Spread as Primary Intramonth Trigger")
    print("=" * 80)

    results = {}

    # Baseline: VIX only (current production)
    print("\n> Running BASELINE (VIX > 27 only) ...")
    results["BASELINE (VIX>27)"] = run_backtest(
        prices,
        returns,
        mkt_filters,
        regime_df,
        allocations,
        credit_signals,
        trigger_mode="vix_only",
        vix_hi=27.0,
    )

    # Z-score only: replace VIX entirely with credit ratio z-score
    print("\n--- Z-Score Only (replacing VIX) ---")
    for z_stress, z_recov in [
        (-1.5, -0.5),
        (-2.0, -0.5),
        (-1.5, 0.0),
        (-2.0, -1.0),
        (-1.0, 0.0),
    ]:
        label = f"Z_s{z_stress}_r{z_recov}"
        print(f"> Running {label} ...")
        results[label] = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            credit_signals,
            trigger_mode="zscore_replaces",
            zscore_stress=z_stress,
            zscore_recovery=z_recov,
        )

    # Combined z-score + HYG momentum (no VIX)
    print("\n--- Z-Score + HYG Momentum (no VIX) ---")
    for z_stress, hyg_stress in [(-1.5, -0.02), (-2.0, -0.02), (-1.5, -0.03)]:
        label = f"Z{z_stress}+HYG{int(hyg_stress * 100)}pct"
        print(f"> Running {label} ...")
        results[label] = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            credit_signals,
            trigger_mode="zscore_hyg",
            zscore_stress=z_stress,
            hyg_stress=hyg_stress,
        )

    # HYG momentum only (no VIX, same as Exp D credit_only)
    print("\n--- HYG Momentum Only (no VIX) ---")
    results["HYG_mom_10d_2pct"] = run_backtest(
        prices,
        returns,
        mkt_filters,
        regime_df,
        allocations,
        credit_signals,
        trigger_mode="hyg_mom_only",
        hyg_stress=-0.02,
    )

    baseline = results["BASELINE (VIX>27)"]
    experiments = {k: v for k, v in results.items() if k != "BASELINE (VIX>27)"}

    # Results table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    hdr = f"{'Config':<24} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Vol':>7} {'Intra':>6}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, m):
        print(
            f"{label:<24} {m['cagr']:>6.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
            f" {m['max_dd']:>7.2%} {m['vol']:>6.2%} {m['intramonth']:>5d}"
        )

    row("BASELINE (VIX>27)", baseline)
    for label, m in experiments.items():
        row(label, m)

    # Deltas
    print("\n" + "=" * 80)
    print("DELTAS vs BASELINE")
    print("=" * 80)
    hdr2 = f"{'Config':<24} {'dCAGR':>7} {'dSharpe':>8} {'dSortino':>9} {'dMaxDD':>8}"
    print(hdr2)
    print("-" * len(hdr2))

    for label, m in experiments.items():
        dc = m["cagr"] - baseline["cagr"]
        ds = m["sharpe"] - baseline["sharpe"]
        dso = m["sortino"] - baseline["sortino"]
        dd_d = m["max_dd"] - baseline["max_dd"]
        flag = " <--" if ds >= 0.02 or dc >= 0.0025 else ""
        print(f"{label:<24} {dc:>+6.2%} {ds:>+8.3f} {dso:>+9.3f} {dd_d:>+7.2%}{flag}")

    # Subperiod Sharpe
    print("\n" + "=" * 80)
    print("SUBPERIOD SHARPE RATIOS")
    print("=" * 80)
    sub_names = list(baseline["sub_sharpes"].keys())
    sub_hdr = f"{'Config':<24} " + " ".join(f"{n:>10}" for n in sub_names)
    print(sub_hdr)
    print("-" * len(sub_hdr))

    def sub_row(label, m):
        vals = [
            f"{m['sub_sharpes'].get(n, 0):>10.2f}"
            if m["sub_sharpes"].get(n) is not None
            else f"{'N/A':>10}"
            for n in sub_names
        ]
        print(f"{label:<24} " + " ".join(vals))

    sub_row("BASELINE (VIX>27)", baseline)
    for label, m in experiments.items():
        sub_row(label, m)

    # Bear periods
    print("\n" + "=" * 80)
    print("BEAR / STRESS PERIOD PERFORMANCE")
    print("=" * 80)
    for pname in ["COVID crash", "2022 bear", "2018 Q4", "2016 oil scare"]:
        bp = baseline["bear_periods"].get(pname)
        if not bp:
            continue
        print(f"\n  {pname}:")
        print(
            f"    {'BASELINE (VIX>27)':<24}: return={bp['return']:>+7.2%}  max_dd={bp['max_dd']:>+7.2%}"
        )
        for label, m in experiments.items():
            ep = m["bear_periods"].get(pname)
            if ep:
                d_dd = ep["max_dd"] - bp["max_dd"]
                flag = " ***" if d_dd > 0.005 else (" !!!" if d_dd < -0.005 else "")
                print(
                    f"    {label:<24}: return={ep['return']:>+7.2%}  max_dd={ep['max_dd']:>+7.2%}"
                    f"  (dd: {d_dd:>+.2%}){flag}"
                )

    # Trigger logs for best experiments
    for label, m in experiments.items():
        if m["intramonth_log"] and m["sharpe"] >= baseline["sharpe"]:
            print(f"\n{'=' * 80}")
            print(f"TRIGGER LOG: {label} ({m['intramonth']} triggers)")
            print(f"{'=' * 80}")
            for d, r in m["intramonth_log"][-20:]:
                print(f"  {d.date()} -- {r}")

    # Escalation decision
    best_label = max(experiments, key=lambda k: experiments[k]["sharpe"])
    best = experiments[best_label]
    ds = best["sharpe"] - baseline["sharpe"]
    dc = best["cagr"] - baseline["cagr"]

    has_bear = False
    for pn in ["COVID crash", "2022 bear", "2018 Q4"]:
        bpp = baseline["bear_periods"].get(pn, {})
        epp = best["bear_periods"].get(pn, {})
        if epp and bpp and epp.get("max_dd", -1) > bpp.get("max_dd", -1) + 0.005:
            has_bear = True

    print(f"\n{'=' * 80}")
    print(f"ESCALATION DECISION (best: {best_label})")
    print(f"{'=' * 80}")
    print(f"  Sharpe delta      : {ds:+.3f}  (threshold: +0.020)")
    print(f"  CAGR delta        : {dc:+.3%}  (threshold: +0.25%)")
    print(f"  Bear improvement  : {'YES' if has_bear else 'NO'}")

    if ds >= 0.02 or dc >= 0.0025 or has_bear:
        print("\n  >> PASS -- escalate to full walk-forward validation")
    else:
        if ds < 0.02 and dc < 0.0025 and not has_bear:
            print(
                "\n  >> REJECT -- below all escalation thresholds (noise kill switch)"
            )
        else:
            print("\n  >> MARGINAL -- consider re-testing with different parameters")


if __name__ == "__main__":
    main()
