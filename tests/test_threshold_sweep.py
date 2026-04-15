"""
VIX intramonth-trigger threshold sensitivity sweep
====================================================
Sweeps VIX_INTRAMONTH_THRESHOLD over [25, 27, 28, 29, 30] with
SPY_WEEKLY_DROP_THRESHOLD fixed at 0.02, across three periods:
  • 2023 full year  (calm — false-positive test)
  • 2024 full year  (moderate vol — false-positive test)
  • Q1 2026         (stress test)

No existing logic is modified.  All thresholds are passed as arguments
to check_intramonth_trigger(), which already accepts them.

False-positive definition
--------------------------
A trigger that fires on date T is a false positive if the BASELINE portfolio
(no overrides) compounded return over the 10 trading days T+1 … T+10 is
positive — i.e. we went defensive into a recovery, not a crash.

Usage
-----
  python tests/test_threshold_sweep.py
  .venv\\Scripts\\python tests/test_threshold_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    EQUITY_TICKERS,
    MA_EQUITY_CAP,
    MA_LOOKBACK,
    VIX_RISK_ON_CAP,
    VIX_THRESHOLD,
)
from src.overrides import (
    apply_ma_filter,
    apply_vix_override,
    check_intramonth_trigger,
    fetch_market_filters,
)

# ── Sweep parameters (do not change logic — only values swept) ───────────────
VIX_THRESHOLDS_TO_TEST = [25, 27, 28, 29, 30]
SPY_WEEKLY_DROP_FIXED = 0.02  # lower than current config (0.03)

PERIODS = {
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "Q1-2026": ("2026-01-01", "2026-04-05"),
}

# Max acceptable false positives per year in calm periods to consider "good"
FP_ANNUAL_LIMIT = 2

# Warmup must reach at least 200 trading days before 2023-01-01 (~Jun 2022)
WARMUP_START = "2022-01-01"
DATA_END = "2026-04-05"
OUTPUTS = ROOT / "outputs"

# ── Backtest constants — mirror src/backtest.py exactly ──────────────────────
TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
ASSETS = TICKERS + ["cash"]
VOL_LOOKBACK = 63
VOL_EPS = 1e-8
MIN_VOL = 0.05
MAX_VOL = 2.00
CASH_DAILY = (1.045) ** (1 / 252) - 1
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}


# ── Pure weight/metric helpers ────────────────────────────────────────────────


def _d2s(d: dict) -> pd.Series:
    return pd.Series({c: float(d.get(c, 0.0)) for c in ASSETS}, index=ASSETS)


def _s2d(s: pd.Series) -> dict:
    return {k: float(v) for k, v in s.to_dict().items()}


def _vol_scale(raw_w: pd.Series, trail: pd.DataFrame) -> pd.Series:
    w = raw_w.copy()
    risky = [c for c in w.index if c != "cash" and c in trail.columns]
    if not risky:
        return w / w.sum()
    vol = trail[risky].std()
    vol = vol.clip(lower=MIN_VOL * vol.median(), upper=MAX_VOL * vol.median())
    vol = vol.replace(0.0, VOL_EPS).fillna(vol.median())
    w[risky] = w[risky] / vol
    return w / w.sum()


def _blend(w_off: dict, w_on: dict, alpha: float) -> dict:
    a = float(np.clip(alpha, 0.0, 1.0))
    out = {
        k: (1 - a) * float(w_off.get(k, 0)) + a * float(w_on.get(k, 0)) for k in ASSETS
    }
    s = sum(out.values())
    return (
        {k: v / s for k, v in out.items()}
        if s > 0
        else {k: 1 / len(ASSETS) for k in ASSETS}
    )


def _max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _total_ret(rets: pd.Series) -> float:
    return float((1 + rets).prod() - 1)


# ── Data loader — called once, shared across all sweep iterations ─────────────

_DATA_CACHE: dict = {}


def _load_data() -> tuple:
    if _DATA_CACHE:
        return (
            _DATA_CACHE["prices"],
            _DATA_CACHE["returns"],
            _DATA_CACHE["regime_daily"],
            _DATA_CACHE["allocations"],
            _DATA_CACHE["W_ON"],
            _DATA_CACHE["W_OFF"],
        )

    print(f"  Downloading prices {WARMUP_START} -> {DATA_END} ...")
    raw = yf.download(
        TICKERS, start=WARMUP_START, end=DATA_END, progress=False, auto_adjust=True
    )
    prices = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(
        how="all"
    )
    returns = prices.pct_change().fillna(0.0)
    returns["cash"] = CASH_DAILY

    rdf = pd.read_csv(OUTPUTS / "regime_labels_expanded.csv", parse_dates=["date"])
    rdf.set_index("date", inplace=True)
    rdf.index = rdf.index.to_period("M").to_timestamp("M")
    rdf["regime"] = rdf["regime"].astype(str).str.strip()
    regime_daily = rdf.reindex(prices.index, method="ffill")

    adf = pd.read_csv(OUTPUTS / "optimal_allocations.csv")
    adf["regime"] = adf["regime"].astype(str).str.strip()
    adf.set_index("regime", inplace=True)
    allocations = {
        str(k).strip(): {a: float(v) for a, v in d.items()}
        for k, d in adf.to_dict(orient="index").items()
    }
    for d in allocations.values():
        d.setdefault("cash", 0.0)

    def _avg(regs):
        regs = [r for r in regs if r in allocations]
        out = dict.fromkeys(ASSETS, 0.0)
        for r in regs:
            for a in ASSETS:
                out[a] += float(allocations[r].get(a, 0.0))
        n = max(len(regs), 1)
        return {a: v / n for a, v in out.items()}

    W_ON = _avg({"Recovery", "Overheating"})
    W_OFF = _avg({"Contraction", "Stagflation"})

    _DATA_CACHE.update(
        {
            "prices": prices,
            "returns": returns,
            "regime_daily": regime_daily,
            "allocations": allocations,
            "W_ON": W_ON,
            "W_OFF": W_OFF,
        }
    )
    print(
        f"  Prices: {len(prices)} days  "
        f"{prices.index.min().date()} -> {prices.index.max().date()}"
    )
    return prices, returns, regime_daily, allocations, W_ON, W_OFF


# ── Per-date weight calculator ─────────────────────────────────────────────────


def _calc_w(
    date,
    prices,
    returns,
    regime_daily,
    allocations,
    W_ON,
    W_OFF,
    mkt_filters,
    use_vix: bool,
    use_ma: bool,
) -> dict:
    """Compute rebalance weights for `date` (same pipeline as src/backtest.py)."""
    row = regime_daily.loc[date] if date in regime_daily.index else None
    alpha: float | None = None
    if (
        row is not None
        and "risk_on" in regime_daily.columns
        and not pd.isna(row["risk_on"])
    ):
        alpha = float(row["risk_on"])

    if use_vix and mkt_filters is not None and alpha is not None:
        alpha = apply_vix_override(
            alpha, mkt_filters, date, VIX_THRESHOLD, VIX_RISK_ON_CAP
        )

    if alpha is not None:
        w = _blend(W_OFF, W_ON, alpha)
    else:
        rv = str(row["regime"]).strip() if row is not None else "Contraction"
        key = REGIME_ALIASES.get(rv, rv)
        w = allocations.get(key, {a: 1 / len(ASSETS) for a in ASSETS})

    raw_w = _d2s(w)
    trail = returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)
    scaled = _vol_scale(raw_w, trail)
    w = _s2d(scaled)

    if use_ma and mkt_filters is not None and "SPY" in prices.columns:
        w = apply_ma_filter(
            w, prices["SPY"].loc[:date], EQUITY_TICKERS, MA_LOOKBACK, MA_EQUITY_CAP
        )
    return w


# ── Core replay ────────────────────────────────────────────────────────────────


def run_replay(
    *,
    use_vix: bool,
    use_ma: bool,
    use_intra: bool,
    vix_intra: float,
    spy_drop: float,
    mkt_filters: pd.DataFrame,
) -> tuple[pd.Series, list[tuple]]:
    """Full replay from WARMUP_START to DATA_END.

    Returns
    -------
    rets : pd.Series  — daily portfolio returns (full date range, not sliced)
    triggers : list of (date, reason)  — every intramonth trigger fired
    """
    prices, returns, regime_daily, allocations, W_ON, W_OFF = _load_data()

    prev_month = None
    cur_w: dict = {a: 1 / len(ASSETS) for a in ASSETS}
    port: list = []
    triggers: list[tuple] = []

    for date in returns.index:
        row = regime_daily.loc[date] if date in regime_daily.index else None
        if row is None or pd.isna(row.get("regime", None)):
            port.append(np.nan)
            continue

        month = date.to_period("M")

        if prev_month is None or month != prev_month:
            # Monthly rebalance — always runs
            cur_w = _calc_w(
                date,
                prices,
                returns,
                regime_daily,
                allocations,
                W_ON,
                W_OFF,
                mkt_filters,
                use_vix,
                use_ma,
            )
            prev_month = month

        elif use_intra and mkt_filters is not None and date.weekday() == 4:
            # Friday intramonth check — only between monthly rebalances
            fired, reason = check_intramonth_trigger(
                mkt_filters, date, vix_intra, spy_drop
            )
            if fired:
                cur_w = _calc_w(
                    date,
                    prices,
                    returns,
                    regime_daily,
                    allocations,
                    W_ON,
                    W_OFF,
                    mkt_filters,
                    use_vix,
                    use_ma,
                )
                triggers.append((date, reason))

        ret = sum(
            float(returns.loc[date, a]) * float(cur_w.get(a, 0.0)) for a in ASSETS
        )
        port.append(ret)

    full_rets = pd.Series(port, index=returns.index, name="port").dropna()
    return full_rets, triggers


# ── False-positive counter ─────────────────────────────────────────────────────


def count_false_positives(
    triggers: list[tuple],
    baseline: pd.Series,
    period_start: str,
    period_end: str,
    fwd_days: int = 10,
) -> tuple[int, int]:
    """Count triggers in period and how many are false positives.

    False positive: baseline compounded return over the next `fwd_days`
    trading days after the trigger date is > 0 (went defensive into recovery).

    Returns (n_triggers_in_period, n_false_positives)
    """
    p0 = pd.Timestamp(period_start)
    p1 = pd.Timestamp(period_end)
    period_triggers = [(d, r) for d, r in triggers if p0 <= d <= p1]

    n_fp = 0
    for trig_date, _ in period_triggers:
        fwd = baseline[baseline.index > trig_date].head(fwd_days)
        if len(fwd) == 0:
            continue
        fwd_ret = float((1 + fwd).prod() - 1)
        if fwd_ret > 0:
            n_fp += 1

    return len(period_triggers), n_fp


# ── Main sweep ────────────────────────────────────────────────────────────────


def main() -> None:
    sep = "=" * 78
    sep2 = "-" * 78

    print(sep)
    print("  VIX INTRAMONTH THRESHOLD SENSITIVITY SWEEP")
    print(f"  Thresholds : {VIX_THRESHOLDS_TO_TEST}")
    print(f"  SPY drop   : {SPY_WEEKLY_DROP_FIXED:.0%}  (fixed, lowered from 3%)")
    print("  FP window  : 10 trading days (~2 weeks)")
    print(f"  FP limit   : < {FP_ANNUAL_LIMIT} per year (calm periods)")
    print(sep)

    # ── Step 1: load data once ──────────────────────────────────────────────
    print("\n[1/3] Loading shared data ...")
    _load_data()

    print("\n[2/3] Fetching VIX + SPY filter data ...")
    mkt_filters = fetch_market_filters(WARMUP_START, DATA_END)
    print(
        f"  {len(mkt_filters)} trading days  "
        f"{mkt_filters.index.min().date()} -> {mkt_filters.index.max().date()}"
    )

    # ── Step 2: run baseline once ───────────────────────────────────────────
    print("\n[3/3] Running sweep ...")
    print("  Computing baseline (monthly only, no overrides) ...")
    rets_base, _ = run_replay(
        use_vix=False,
        use_ma=False,
        use_intra=False,
        vix_intra=30.0,
        spy_drop=SPY_WEEKLY_DROP_FIXED,
        mkt_filters=mkt_filters,
    )

    # ── Step 3: sweep ───────────────────────────────────────────────────────
    # results[thresh][period] = dict
    results: dict[float, dict[str, dict]] = {}
    # baseline metrics per period (for delta columns)
    base_metrics: dict[str, dict] = {}
    for pname, (pstart, pend) in PERIODS.items():
        r_base = rets_base[(rets_base.index >= pstart) & (rets_base.index <= pend)]
        base_metrics[pname] = {
            "total_ret": _total_ret(r_base),
            "max_dd": _max_dd(r_base),
        }

    for thresh in VIX_THRESHOLDS_TO_TEST:
        print(f"  Threshold {thresh:5.1f} ...", end=" ", flush=True)
        rets_enh, trig_list = run_replay(
            use_vix=True,
            use_ma=True,
            use_intra=True,
            vix_intra=float(thresh),
            spy_drop=SPY_WEEKLY_DROP_FIXED,
            mkt_filters=mkt_filters,
        )
        results[thresh] = {}
        for pname, (pstart, pend) in PERIODS.items():
            r = rets_enh[(rets_enh.index >= pstart) & (rets_enh.index <= pend)]
            n_trig, n_fp = count_false_positives(trig_list, rets_base, pstart, pend)
            results[thresh][pname] = {
                "total_ret": _total_ret(r),
                "max_dd": _max_dd(r),
                "n_trig": n_trig,
                "n_fp": n_fp,
                "triggers": [
                    (d, rsn)
                    for d, rsn in trig_list
                    if pd.Timestamp(pstart) <= d <= pd.Timestamp(pend)
                ],
            }
        total_trigs = sum(results[thresh][p]["n_trig"] for p in PERIODS)
        print(f"{total_trigs} total triggers")

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESULTS TABLE")
    print(
        f"  {'Thresh':>7}  {'Period':<10}  {'TotalRet':>9}  {'DD vs Base':>10}  "
        f"{'MaxDD':>8}  {'#Trig':>6}  {'#FP':>5}  FP OK?"
    )
    print(f"  {sep2[:74]}")

    for thresh in VIX_THRESHOLDS_TO_TEST:
        for pname in PERIODS:
            r = results[thresh][pname]
            bm = base_metrics[pname]
            dd_imp = (r["max_dd"] - bm["max_dd"]) * 10_000  # bp, positive=better
            fp_ok = "YES" if r["n_fp"] < FP_ANNUAL_LIMIT else f"NO({r['n_fp']})"
            print(
                f"  {thresh:>7.0f}  {pname:<10}  {r['total_ret']:>9.2%}  "
                f"{dd_imp:>+10.1f}bp  {r['max_dd']:>8.2%}  "
                f"{r['n_trig']:>6}  {r['n_fp']:>5}  {fp_ok}"
            )
        print(f"  {'-' * 74}")

    # ── Trigger detail ────────────────────────────────────────────────────────
    print("\n  TRIGGER DETAIL (date, reason, FP?)")
    print(f"  {sep2[:74]}")
    for thresh in VIX_THRESHOLDS_TO_TEST:
        print(f"  Threshold {thresh:.0f}:")
        any_trig = False
        for pname, (_pstart, _pend) in PERIODS.items():
            r = results[thresh][pname]
            for trig_date, reason in r["triggers"]:
                fwd = rets_base[rets_base.index > trig_date].head(10)
                fwd_ret = float((1 + fwd).prod() - 1) if len(fwd) > 0 else float("nan")
                is_fp = fwd_ret > 0
                fp_tag = "FP" if is_fp else "TP"
                print(
                    f"    {trig_date.date()}  [{pname}]  {reason:<45s}  "
                    f"10d-fwd: {fwd_ret:+.2%}  [{fp_tag}]"
                )
                any_trig = True
        if not any_trig:
            print("    (no triggers)")

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  OPTIMAL THRESHOLD ANALYSIS")
    print("  Criteria: maximise Q1-2026 MaxDD improvement")
    print(f"            subject to FP < {FP_ANNUAL_LIMIT} in BOTH 2023 and 2024")
    print(f"  {sep2[:74]}")

    candidates: list[tuple] = []  # (dd_improvement_bp, thresh)
    for thresh in VIX_THRESHOLDS_TO_TEST:
        fp_2023 = results[thresh]["2023"]["n_fp"]
        fp_2024 = results[thresh]["2024"]["n_fp"]
        fp_ok = fp_2023 < FP_ANNUAL_LIMIT and fp_2024 < FP_ANNUAL_LIMIT
        dd_imp = (
            results[thresh]["Q1-2026"]["max_dd"] - base_metrics["Q1-2026"]["max_dd"]
        ) * 10_000
        trig_q1 = results[thresh]["Q1-2026"]["n_trig"]
        label = "ELIGIBLE" if fp_ok else "DISQUALIFIED (FP)"
        print(
            f"  Thresh {thresh:2.0f}:  Q1-DD +{dd_imp:+.0f}bp  "
            f"FP-2023={fp_2023}  FP-2024={fp_2024}  "
            f"Triggers-Q1={trig_q1}  [{label}]"
        )
        if fp_ok:
            candidates.append((dd_imp, thresh))

    print()
    if candidates:
        best_dd_bp, best_thresh = max(candidates)
        print(f"  RECOMMENDED: VIX_INTRAMONTH_THRESHOLD = {best_thresh:.0f}")
        print(
            f"               (+{best_dd_bp:.0f} bp Q1-2026 max DD improvement, "
            f"FP < {FP_ANNUAL_LIMIT}/yr in calm periods)"
        )
        print("\n  To adopt: edit src/config.py →")
        print(f"    VIX_INTRAMONTH_THRESHOLD   = {best_thresh:.0f}")
        print(f"    SPY_WEEKLY_DROP_THRESHOLD  = {SPY_WEEKLY_DROP_FIXED:.2f}")
    else:
        print("  No threshold satisfies both criteria simultaneously.")
        print(f"  Consider relaxing FP_ANNUAL_LIMIT (currently {FP_ANNUAL_LIMIT}).")

    print(sep)


if __name__ == "__main__":
    main()
