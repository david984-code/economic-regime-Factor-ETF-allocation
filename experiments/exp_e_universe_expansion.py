"""
EXPERIMENT E -- Asset Universe Expansion (EFA, EEM, TIPS)
==========================================================

Hypothesis
----------
Adding international equity (EFA, EEM) and inflation-protected bonds (TIPS)
to the portfolio improves risk-adjusted returns through diversification.
EFA/EEM provide non-US equity exposure in risk-on regimes; TIPS provides
real-rate-hedged bond exposure in risk-off regimes (currently absent —
only GLD serves as defensive).

Variable changed
----------------
Asset universe expanded from 8 tickers to up to 11.
  Risk-on additions:  EFA (developed intl), EEM (emerging markets)
  Risk-off addition:  TIPS (inflation-protected bonds)

Weight is carved proportionally from existing allocations — the optimizer
is NOT re-run (single-variable principle). Vol-scaling, overrides, momentum
tilt, and credit trigger all operate identically.

Configurations tested
---------------------
1. Baseline: current 8-ticker universe
2. +TIPS only: carve from GLD in all regimes (5%, 10%, 15%)
3. +EFA/EEM only: carve from risk-on equity sleeve (5%, 10%, 15% each)
4. +EFA+EEM+TIPS combined at best individual carve levels
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
    check_intramonth_trigger,
    fetch_market_filters,
)

# ── Static settings (match production) ----------------------------------------
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

BASE_TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
NEW_TICKERS = ["EFA", "EEM", "TIP"]

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

ENABLE_CREDIT_TRIGGER = True
ENABLE_MOMENTUM_TILT = True
MOMENTUM_LOOKBACK = 10
MOMENTUM_STRENGTH = 0.20
MOMENTUM_MAX_TILT = 2.0

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}

RISK_ON_ASSETS = {"SPY", "MTUM", "VLUE", "QUAL", "USMV", "IJR", "VIG", "EFA", "EEM"}
RISK_OFF_ASSETS = {"GLD", "TIP"}


# ── Data loading ---------------------------------------------------------------


def load_data():
    all_tickers = BASE_TICKERS + NEW_TICKERS + ["HYG"]
    print("Downloading price data (base + EFA/EEM/TIPS + HYG) ...")
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = (
            raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
        )
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    prices = prices.dropna()

    print(
        f"  Price data: {len(prices)} days, {prices.index.min().date()} to {prices.index.max().date()}"
    )
    for t in NEW_TICKERS:
        if t in prices.columns:
            print(f"  {t}: available from {prices[t].first_valid_index().date()}")
        else:
            print(f"  {t}: NOT AVAILABLE")

    print("\nFetching market filter data (VIX + SPY + HYG) ...")
    mkt_filters = fetch_market_filters(START_DATE, END_DATE)
    print(f"  Market filters: {len(mkt_filters)} days")

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
    base_allocations = {
        str(k).strip(): v for k, v in opt_alloc_df.to_dict(orient="index").items()
    }
    for alloc in base_allocations.values():
        if "cash" not in alloc:
            alloc["cash"] = 0.0

    return prices, mkt_filters, regime_df, base_allocations


# ── Allocation modification helpers -------------------------------------------


def expand_allocations(
    base_alloc: dict,
    tickers: list[str],
    efa_carve: float = 0.0,
    eem_carve: float = 0.0,
    tip_carve: float = 0.0,
) -> dict:
    """Create modified allocations with new assets carved from existing ones.

    efa_carve / eem_carve: fraction carved proportionally from risk-on assets.
    tip_carve: fraction carved proportionally from GLD weight.
    """
    expanded = {}
    for regime, alloc in base_alloc.items():
        w = {a: float(alloc.get(a, 0.0)) for a in tickers}
        w.setdefault("cash", float(alloc.get("cash", 0.0)))

        # Carve EFA/EEM from existing risk-on equity
        risk_on_in_regime = [
            a for a in BASE_TICKERS if a in RISK_ON_ASSETS and w.get(a, 0) > 1e-6
        ]
        total_risk_on = sum(w.get(a, 0) for a in risk_on_in_regime)

        for new_asset, carve in [("EFA", efa_carve), ("EEM", eem_carve)]:
            if carve > 0 and total_risk_on > 1e-6 and new_asset in tickers:
                amount = total_risk_on * carve
                for a in risk_on_in_regime:
                    w[a] *= 1 - carve
                w[new_asset] = amount

        # Carve TIP (TIPS bonds) from GLD
        if tip_carve > 0 and "TIP" in tickers:
            gld_w = w.get("GLD", 0.0)
            if gld_w > 1e-6:
                tip_amount = gld_w * tip_carve
                w["GLD"] = gld_w - tip_amount
                w["TIP"] = tip_amount
            else:
                w["TIP"] = 0.0

        for t in tickers:
            w.setdefault(t, 0.0)

        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        expanded[regime] = w
    return expanded


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


def _avg_alloc(regimes, allocations, assets):
    regs = [r for r in regimes if r in allocations]
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


# ── Backtest engine -----------------------------------------------------------


def run_backtest(
    all_prices,
    mkt_filters,
    regime_df,
    allocations,
    tickers: list[str],
    assets: list[str],
    label: str = "",
) -> dict:
    """Run full backtest with given ticker universe and allocations."""
    prices = all_prices[[t for t in tickers if t in all_prices.columns]].copy()
    returns = prices.pct_change().dropna()
    returns["cash"] = CASH_DAILY_YIELD

    equity_tickers = [t for t in tickers if t in RISK_ON_ASSETS]

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

        if USE_MA_FILTER and "SPY" in prices.columns:
            w = apply_ma_filter(
                w, prices["SPY"].loc[:date], equity_tickers, MA_LOOKBACK, MA_EQUITY_CAP
            )

        if ENABLE_MOMENTUM_TILT:
            w = apply_momentum_tilt(
                w,
                prices,
                date,
                lookback=MOMENTUM_LOOKBACK,
                strength=MOMENTUM_STRENGTH,
                max_tilt=MOMENTUM_MAX_TILT,
            )
        return w

    # Rebalance loop
    port_rets = []
    prev_month = None
    current_w = {a: 1.0 / len(assets) for a in assets}
    prev_w = None
    total_turnover = 0.0
    rebalance_count = 0
    intramonth_count = 0
    last_intra_idx = -COOLDOWN_DAYS - 1

    date_list = list(returns.index)
    for i, date in enumerate(date_list):
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
            last_intra_idx = i - COOLDOWN_DAYS - 1

        elif (i - last_intra_idx) >= COOLDOWN_DAYS and date.weekday() == 4:
            try:
                triggered, reason = check_intramonth_trigger(
                    mkt_filters,
                    date,
                    VIX_INTRAMONTH_THRESHOLD,
                    SPY_WEEKLY_DROP_THRESHOLD,
                )
            except (ValueError, KeyError):
                triggered = False
            if triggered:
                current_w = calc_weights(date)
                did_rebalance = True
                intramonth_count += 1
                last_intra_idx = i

        if did_rebalance and prev_w is not None:
            to = (
                sum(
                    abs(float(current_w.get(a, 0)) - float(prev_w.get(a, 0)))
                    for a in assets
                )
                / 2.0
            )
            total_turnover += to
        if did_rebalance:
            prev_w = current_w.copy()

        daily_ret = sum(
            returns.loc[date, a] * float(current_w.get(a, 0.0))
            for a in assets
            if a in returns.columns
        )
        port_rets.append(daily_ret)

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
        "sub_sharpes": sub_sharpes,
        "bear_periods": bear_rets,
    }


# ── Main -----------------------------------------------------------------------


def main():
    all_prices, mkt_filters, regime_df, base_allocations = load_data()

    print("\n" + "=" * 80)
    print("EXPERIMENT E: Asset Universe Expansion (EFA, EEM, TIPS)")
    print("=" * 80)

    results = {}

    # ── 1. Baseline: current universe (with all production features) -----------
    print("\n> Running BASELINE (8-ticker, all features) ...")
    base_assets = BASE_TICKERS + ["cash"]
    base_alloc_exp = expand_allocations(base_allocations, base_assets)
    results["BASELINE"] = run_backtest(
        all_prices,
        mkt_filters,
        regime_df,
        base_alloc_exp,
        tickers=BASE_TICKERS,
        assets=base_assets,
        label="BASELINE",
    )

    # ── 2. TIP only (carve from GLD) -------------------------------------------
    print("\n--- TIP (TIPS bonds) Addition (carved from GLD) ---")
    for tip_pct in [0.15, 0.25, 0.35, 0.50]:
        tickers = BASE_TICKERS + ["TIP"]
        assets = tickers + ["cash"]
        alloc = expand_allocations(base_allocations, assets, tip_carve=tip_pct)
        label = f"+TIP_{tip_pct:.0%}"
        print(f"> Running {label} ...")
        results[label] = run_backtest(
            all_prices,
            mkt_filters,
            regime_df,
            alloc,
            tickers=tickers,
            assets=assets,
            label=label,
        )

    # ── 3. EFA/EEM only (carve from risk-on equity) ----------------------------
    print("\n--- EFA/EEM Addition (carved from risk-on equity) ---")
    for eq_pct in [0.05, 0.10, 0.15]:
        tickers = BASE_TICKERS + ["EFA", "EEM"]
        assets = tickers + ["cash"]
        alloc = expand_allocations(
            base_allocations, assets, efa_carve=eq_pct, eem_carve=eq_pct
        )
        label = f"+EFA/EEM_{eq_pct:.0%}ea"
        print(f"> Running {label} ...")
        results[label] = run_backtest(
            all_prices,
            mkt_filters,
            regime_df,
            alloc,
            tickers=tickers,
            assets=assets,
            label=label,
        )

    # ── 4. EFA-only (no EEM) --------------------------------------------------
    print("\n--- EFA Only (developed intl, carved from risk-on) ---")
    for eq_pct in [0.05, 0.10, 0.15]:
        tickers = BASE_TICKERS + ["EFA"]
        assets = tickers + ["cash"]
        alloc = expand_allocations(base_allocations, assets, efa_carve=eq_pct)
        label = f"+EFA_{eq_pct:.0%}"
        print(f"> Running {label} ...")
        results[label] = run_backtest(
            all_prices,
            mkt_filters,
            regime_df,
            alloc,
            tickers=tickers,
            assets=assets,
            label=label,
        )

    # ── 5. Full expansion: EFA + EEM + TIP -------------------------------------
    print("\n--- Full Expansion (EFA + EEM + TIP) ---")
    for eq_pct, tip_pct in [(0.05, 0.25), (0.10, 0.25), (0.10, 0.35), (0.05, 0.35)]:
        tickers = BASE_TICKERS + NEW_TICKERS
        assets = tickers + ["cash"]
        alloc = expand_allocations(
            base_allocations,
            assets,
            efa_carve=eq_pct,
            eem_carve=eq_pct,
            tip_carve=tip_pct,
        )
        label = f"+ALL_eq{eq_pct:.0%}_tip{tip_pct:.0%}"
        print(f"> Running {label} ...")
        results[label] = run_backtest(
            all_prices,
            mkt_filters,
            regime_df,
            alloc,
            tickers=tickers,
            assets=assets,
            label=label,
        )

    baseline = results["BASELINE"]
    experiments = {k: v for k, v in results.items() if k != "BASELINE"}

    # ── Results table -----------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    hdr = f"{'Config':<24} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Vol':>7} {'TO/yr':>7} {'Intra':>6}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, m):
        print(
            f"{label:<24} {m['cagr']:>6.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
            f" {m['max_dd']:>7.2%} {m['vol']:>6.2%} {m['turnover']:>6.1f} {m['intramonth']:>5d}"
        )

    row("BASELINE", baseline)
    for label, m in experiments.items():
        row(label, m)

    # ── Deltas -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DELTAS vs BASELINE")
    print("=" * 80)

    hdr2 = f"{'Config':<24} {'dCAGR':>7} {'dSharpe':>8} {'dSortino':>9} {'dMaxDD':>8} {'dVol':>7}"
    print(hdr2)
    print("-" * len(hdr2))

    for label, m in experiments.items():
        dc = m["cagr"] - baseline["cagr"]
        ds = m["sharpe"] - baseline["sharpe"]
        dso = m["sortino"] - baseline["sortino"]
        dd = m["max_dd"] - baseline["max_dd"]
        dv = m["vol"] - baseline["vol"]
        flag = " <--" if ds >= 0.02 or dc >= 0.0025 else ""
        print(
            f"{label:<24} {dc:>+6.2%} {ds:>+8.3f} {dso:>+9.3f} {dd:>+7.2%} {dv:>+6.2%}{flag}"
        )

    # ── Subperiod Sharpe --------------------------------------------------------
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

    sub_row("BASELINE", baseline)
    for label, m in experiments.items():
        sub_row(label, m)

    # ── Bear periods ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEAR / STRESS PERIOD PERFORMANCE")
    print("=" * 80)

    for pname in ["COVID crash", "2022 bear", "2018 Q4", "2016 oil scare"]:
        bp = baseline["bear_periods"].get(pname)
        if not bp:
            continue
        print(f"\n  {pname}:")
        print(
            f"    {'BASELINE':<24}: return={bp['return']:>+7.2%}  max_dd={bp['max_dd']:>+7.2%}"
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

    # ── Escalation decision -----------------------------------------------------
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

    sub_consistent = True
    for name in sub_names:
        bs = baseline["sub_sharpes"].get(name)
        es = best["sub_sharpes"].get(name)
        if bs is not None and es is not None and es < bs - 0.10:
            sub_consistent = False

    print(f"\n{'=' * 80}")
    print(f"ESCALATION DECISION (best: {best_label})")
    print(f"{'=' * 80}")
    print(f"  Sharpe delta      : {ds:+.3f}  (threshold: +0.020)")
    print(f"  CAGR delta        : {dc:+.3%}  (threshold: +0.25%)")
    print(f"  Bear improvement  : {'YES' if has_bear else 'NO'}")
    print(
        f"  Subperiod consist.: {'YES' if sub_consistent else 'NO (>0.10 degradation in a subperiod)'}"
    )

    if ds >= 0.02 or dc >= 0.0025 or has_bear:
        print("\n  >> PASS -- escalate to full walk-forward validation")
    else:
        if ds < 0.02 and dc < 0.0025 and not has_bear:
            print(
                "\n  >> REJECT -- below all escalation thresholds (noise kill switch)"
            )
        else:
            print("\n  >> REJECT -- insufficient evidence for escalation")


if __name__ == "__main__":
    main()
