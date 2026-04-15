"""
EXPERIMENT B -- Cross-Sectional Momentum Tilt at Monthly Rebalance
===================================================================

Hypothesis
----------
Tilting portfolio weights toward assets with stronger trailing momentum
(and away from laggards) at each monthly rebalance will improve risk-adjusted
returns by exploiting short-term factor/asset momentum within the existing
allocation framework.

Variable changed
----------------
Portfolio construction pipeline gains one step after inverse-vol scaling:
  - Compute trailing N-day return for each asset
  - Cross-sectional z-score of those returns
  - Tilt: w_new = w_base * (1 + strength * z_momentum)
  - Renormalise, cap to prevent concentration

All other parameters unchanged (regime signal, allocations, VIX overrides, etc).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from overrides import apply_ma_filter, apply_vix_override, fetch_market_filters

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

ENABLE_INTRAMONTH_TRIGGER = True
VIX_INTRAMONTH_THRESHOLD = 27.0
SPY_WEEKLY_DROP_THRESHOLD = 0.99

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}


# ── Data loading ---------------------------------------------------------------


def load_data():
    print("Downloading price data ...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = (
            raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
        )
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    prices = prices.dropna()

    returns = prices.pct_change().dropna()
    returns["cash"] = CASH_DAILY_YIELD

    print("Downloading VIX + SPY for market filters ...")
    mkt_filters = fetch_market_filters(START_DATE, END_DATE)

    outputs = ROOT / "outputs"
    regime_df = pd.read_csv(
        outputs / "regime_labels_expanded.csv", parse_dates=["date"]
    )
    regime_df.set_index("date", inplace=True)
    regime_df = regime_df.reindex(prices.index, method="ffill")
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

    return prices, returns, mkt_filters, regime_df, allocations


# ── Helpers --------------------------------------------------------------------


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


def apply_momentum_tilt(
    weights: dict,
    prices_df: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = 20,
    strength: float = 0.10,
    max_tilt: float = 2.0,
) -> dict:
    """Cross-sectional momentum tilt on portfolio weights.

    For each non-cash asset with weight > 0, compute trailing `lookback`-day
    return, z-score across assets, then tilt:
        w_new = w_base * (1 + strength * z_momentum)

    Capped so no asset exceeds `max_tilt` * its base weight.
    Cash weight is untouched; everything renormalises to 1.
    """
    w = {k: float(v) for k, v in weights.items()}
    cash_w = w.get("cash", 0.0)

    risky_assets = [
        a for a in w if a != "cash" and w[a] > 1e-6 and a in prices_df.columns
    ]
    if len(risky_assets) < 2:
        return w

    trailing = prices_df[risky_assets].loc[:date].tail(lookback + 1)
    if len(trailing) < lookback + 1:
        return w

    mom = (trailing.iloc[-1] / trailing.iloc[0] - 1).to_dict()

    vals = np.array([mom[a] for a in risky_assets])
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-10:
        return w

    for a in risky_assets:
        z = (mom[a] - mu) / sigma
        tilt_factor = 1.0 + strength * z
        tilt_factor = max(1.0 / max_tilt, min(max_tilt, tilt_factor))
        w[a] *= tilt_factor

    # Renormalise risky weights to preserve original risky total
    risky_total_new = sum(w[a] for a in risky_assets)
    risky_total_orig = 1.0 - cash_w
    if risky_total_new > 0 and risky_total_orig > 0:
        scale = risky_total_orig / risky_total_new
        for a in risky_assets:
            w[a] *= scale

    w["cash"] = cash_w
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


# ── Backtest engine ------------------------------------------------------------


def check_intramonth(mkt_filters, date, vix_thresh, spy_drop):
    if date not in mkt_filters.index:
        return False
    vix = mkt_filters.loc[date, "VIX_close"]
    if pd.isna(vix):
        return False
    return float(vix) > vix_thresh


def run_backtest(
    prices,
    returns,
    mkt_filters,
    regime_df,
    allocations,
    *,
    momentum_lookback: int | None = None,
    momentum_strength: float = 0.0,
    momentum_max_tilt: float = 2.0,
) -> dict:
    """Run full backtest. If momentum_lookback is None, no tilt is applied (baseline)."""
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

        # Momentum tilt (the experiment variable)
        if momentum_lookback is not None and momentum_strength > 0:
            w = apply_momentum_tilt(
                w,
                prices,
                date,
                lookback=momentum_lookback,
                strength=momentum_strength,
                max_tilt=momentum_max_tilt,
            )

        return w

    # ── Rebalance loop ---------------------------------------------------------
    port_rets = []
    prev_month = None
    current_w = {a: 1.0 / len(ASSETS) for a in ASSETS}
    prev_w = None
    total_turnover = 0.0
    rebalance_count = 0
    intramonth_count = 0

    for date in returns.index:
        regime = regime_df.loc[date, "regime"] if date in regime_df.index else np.nan
        if pd.isna(regime):
            port_rets.append(np.nan)
            continue

        month = date.to_period("M")
        did_rebalance = False

        if prev_month is None or month != prev_month:
            current_w = calc_weights(date)
            prev_month = month
            did_rebalance = True
            rebalance_count += 1

        elif ENABLE_INTRAMONTH_TRIGGER and date.weekday() == 4:
            if check_intramonth(
                mkt_filters, date, VIX_INTRAMONTH_THRESHOLD, SPY_WEEKLY_DROP_THRESHOLD
            ):
                current_w = calc_weights(date)
                did_rebalance = True
                intramonth_count += 1

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

    port_rets = pd.Series(port_rets, index=returns.index).dropna()

    # ── Metrics ----------------------------------------------------------------
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

    # Subperiod Sharpes
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
        "sub_sharpes": sub_sharpes,
        "bear_periods": bear_rets,
    }


# ── Main -----------------------------------------------------------------------


def main():
    prices, returns, mkt_filters, regime_df, allocations = load_data()

    print("\n" + "=" * 72)
    print("EXPERIMENT B: Cross-Sectional Momentum Tilt")
    print("=" * 72)

    print("\n> Running BASELINE (no tilt) ...")
    baseline = run_backtest(
        prices,
        returns,
        mkt_filters,
        regime_df,
        allocations,
        momentum_lookback=None,
        momentum_strength=0.0,
    )

    configs = [
        {"lb": 5, "str": 0.10},
        {"lb": 5, "str": 0.20},
        {"lb": 10, "str": 0.10},
        {"lb": 10, "str": 0.20},
        {"lb": 20, "str": 0.10},
        {"lb": 20, "str": 0.15},
        {"lb": 20, "str": 0.20},
        {"lb": 40, "str": 0.10},
        {"lb": 40, "str": 0.20},
        {"lb": 63, "str": 0.10},
        {"lb": 63, "str": 0.20},
    ]
    experiments = {}

    for cfg in configs:
        label = f"lb={cfg['lb']:>2d}_str={cfg['str']:.2f}"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            momentum_lookback=cfg["lb"],
            momentum_strength=cfg["str"],
        )
        experiments[label] = result

    # ── Results table -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    hdr = f"{'Config':<18} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Vol':>7} {'Turn':>6}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, m):
        print(
            f"{label:<18} {m['cagr']:>6.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
            f" {m['max_dd']:>7.2%} {m['vol']:>6.2%} {m['turnover']:>5.2f}x"
        )

    row("BASELINE", baseline)
    for label, m in experiments.items():
        row(label, m)

    # ── Deltas -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DELTAS vs BASELINE")
    print("=" * 72)

    hdr2 = f"{'Config':<18} {'dCAGR':>7} {'dSharpe':>8} {'dSortino':>9} {'dMaxDD':>8} {'dVol':>7} {'dTurn':>7}"
    print(hdr2)
    print("-" * len(hdr2))

    for label, m in experiments.items():
        dc = m["cagr"] - baseline["cagr"]
        ds = m["sharpe"] - baseline["sharpe"]
        dso = m["sortino"] - baseline["sortino"]
        dd = m["max_dd"] - baseline["max_dd"]
        dv = m["vol"] - baseline["vol"]
        dt = m["turnover"] - baseline["turnover"]
        print(
            f"{label:<18} {dc:>+6.2%} {ds:>+8.3f} {dso:>+9.3f} {dd:>+7.2%} {dv:>+6.2%} {dt:>+6.2f}x"
        )

    # ── Subperiod Sharpe --------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUBPERIOD SHARPE RATIOS")
    print("=" * 72)

    sub_names = list(baseline["sub_sharpes"].keys())
    sub_hdr = f"{'Config':<18} " + " ".join(f"{n:>10}" for n in sub_names)
    print(sub_hdr)
    print("-" * len(sub_hdr))

    def sub_row(label, m):
        vals = []
        for n in sub_names:
            v = m["sub_sharpes"].get(n)
            vals.append(f"{v:>10.2f}" if v is not None else f"{'N/A':>10}")
        print(f"{label:<18} " + " ".join(vals))

    sub_row("BASELINE", baseline)
    for label, m in experiments.items():
        sub_row(label, m)

    # ── Bear periods ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BEAR PERIOD PERFORMANCE")
    print("=" * 72)

    for pname in ["COVID crash", "2022 bear"]:
        print(f"\n  {pname}:")
        bp = baseline["bear_periods"].get(pname)
        if bp:
            print(
                f"    {'BASELINE':<18}: return={bp['return']:>+7.2%}  max_dd={bp['max_dd']:>+7.2%}"
            )
        for label, m in experiments.items():
            ep = m["bear_periods"].get(pname)
            if ep:
                ep["return"] - bp["return"] if bp else 0
                d_dd = ep["max_dd"] - bp["max_dd"] if bp else 0
                flag = " ***" if d_dd > 0.005 else ""
                print(
                    f"    {label:<18}: return={ep['return']:>+7.2%}  max_dd={ep['max_dd']:>+7.2%}"
                    f"  (dd delta: {d_dd:>+.2%}){flag}"
                )

    # ── Escalation decision -----------------------------------------------------
    best_label = max(experiments, key=lambda k: experiments[k]["sharpe"])
    best = experiments[best_label]
    ds = best["sharpe"] - baseline["sharpe"]
    dc = best["cagr"] - baseline["cagr"]
    has_bear = False
    for pn in ["COVID crash", "2022 bear"]:
        bp = baseline["bear_periods"].get(pn, {})
        ep = best["bear_periods"].get(pn, {})
        if ep and bp and ep.get("max_dd", -1) > bp.get("max_dd", -1) + 0.005:
            has_bear = True

    # Check subperiod consistency
    consistent = True
    for name in sub_names:
        bv = baseline["sub_sharpes"].get(name)
        ev = best["sub_sharpes"].get(name)
        if bv is not None and ev is not None and ev < bv - 0.05:
            consistent = False

    print(f"\n{'=' * 72}")
    print(f"BEST EXPERIMENT: {best_label}")
    print(f"{'=' * 72}")
    print(f"  Sharpe delta     : {ds:+.3f}  (threshold: +0.020)")
    print(f"  CAGR delta       : {dc:+.3%}  (threshold: +0.25%)")
    print(f"  Bear improvement : {'YES' if has_bear else 'NO'}")
    print(
        f"  Subperiod consist: {'YES' if consistent else 'NO (degradation in some periods)'}"
    )

    if ds >= 0.02 or dc >= 0.0025 or has_bear:
        if consistent:
            print("\n  >> PASS -- escalate to full walk-forward validation")
        else:
            print("\n  >> MARGINAL PASS -- escalate but flag subperiod inconsistency")
    else:
        print("\n  >> REJECT -- below escalation thresholds (statistical noise)")


if __name__ == "__main__":
    main()
