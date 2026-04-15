"""
Q1 2026 three-scenario override stress test
============================================

Scenarios
---------
  (a) Baseline      — monthly rebalance only, no VIX/MA overrides
  (b) Monthly+OV    — monthly rebalance with VIX override + MA filter (current state)
  (c) Weekly+OV     — monthly + VIX/MA overrides + intramonth Friday trigger (new)

Pass condition
--------------
  Scenario (c) max drawdown must be at least MATERIALLY_BETTER_DD_BP basis points
  less negative than scenario (a).

Usage
-----
  python tests/test_regime_overrides.py          # from project root
  .venv\\Scripts\\python tests/test_regime_overrides.py
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
    SPY_WEEKLY_DROP_THRESHOLD,
    VIX_INTRAMONTH_THRESHOLD,
    VIX_RISK_ON_CAP,
    VIX_THRESHOLD,
)
from src.overrides import (
    apply_ma_filter,
    apply_vix_override,
    check_intramonth_trigger,
    fetch_market_filters,
)

# ── Test configuration ───────────────────────────────────────────────────────
Q1_START = "2026-01-01"
Q1_END = "2026-04-05"
WARMUP = "2024-06-01"  # need 200-day MA + 63-day vol window pre-loaded

MATERIALLY_BETTER_DD_BP = 25  # basis points; (c) must beat (a) by this much

# ── Regression test configuration ────────────────────────────────────────────
# Extended warmup covers full 2023, 2024, and Q1-2026 with proper vol/MA history.
REGRESSION_WARMUP = "2022-01-01"
REGRESSION_END = Q1_END
# Q1-2026 max DD floor: locks in the +93 bp improvement found at VIX threshold 27.
# At threshold=27: max_dd ≈ -7.12%  vs  baseline ≈ -8.05%  → floor set at -7.50%.
REGRESSION_DD_FLOOR = -0.075

# ── Backtest constants (mirror src/backtest.py) ───────────────────────────────
TICKERS = ["SPY", "GLD", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]
ASSETS = TICKERS + ["cash"]
VOL_LOOKBACK = 63
VOL_EPS = 1e-8
MIN_VOL = 0.05
MAX_VOL = 2.00
CASH_DAILY = (1.045) ** (1 / 252) - 1
OUTPUTS = ROOT / "outputs"
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}


# ── Pure helpers ─────────────────────────────────────────────────────────────


def _d2s(d: dict, cols: list) -> pd.Series:
    return pd.Series({c: float(d.get(c, 0.0)) for c in cols}, index=cols)


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


# ── Metrics ──────────────────────────────────────────────────────────────────


def _max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _sortino(rets: pd.Series, rf: float = CASH_DAILY) -> float:
    exc = rets - rf
    down = exc[exc < 0]
    if len(down) == 0 or down.std() == 0:
        return float("nan")
    return float(exc.mean() * 252 / (down.std() * np.sqrt(252)))


def _total_ret(rets: pd.Series) -> float:
    return float((1 + rets).prod() - 1)


# ── Data loader (called once, shared across scenarios) ───────────────────────

_CACHE: dict = {}


def _load_shared_data() -> tuple:
    if _CACHE:
        return tuple(
            _CACHE[k]
            for k in (
                "prices",
                "returns",
                "regime_daily",
                "allocations",
                "W_ON",
                "W_OFF",
            )
        )

    raw = yf.download(
        TICKERS, start=WARMUP, end=Q1_END, progress=False, auto_adjust=True
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

    RISK_ON = {"Recovery", "Overheating"}
    RISK_OFF = {"Contraction", "Stagflation"}

    def _avg(regs):
        regs = [r for r in regs if r in allocations]
        out = dict.fromkeys(ASSETS, 0.0)
        for r in regs:
            for a in ASSETS:
                out[a] += float(allocations[r].get(a, 0.0))
        n = len(regs) or 1
        return {a: v / n for a, v in out.items()}

    W_ON = _avg(RISK_ON)
    W_OFF = _avg(RISK_OFF)

    _CACHE.update(
        {
            "prices": prices,
            "returns": returns,
            "regime_daily": regime_daily,
            "allocations": allocations,
            "W_ON": W_ON,
            "W_OFF": W_OFF,
        }
    )
    return prices, returns, regime_daily, allocations, W_ON, W_OFF


# ── Per-date weight calculator ────────────────────────────────────────────────


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
    """Compute weights for `date` with optional override filters."""
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

    raw_w = _d2s(w, ASSETS)
    trail = returns[TICKERS].loc[:date].tail(VOL_LOOKBACK)
    scaled = _vol_scale(raw_w, trail)
    w = _s2d(scaled)

    if use_ma and mkt_filters is not None and "SPY" in prices.columns:
        w = apply_ma_filter(
            w, prices["SPY"].loc[:date], EQUITY_TICKERS, MA_LOOKBACK, MA_EQUITY_CAP
        )
    return w


# ── Core replay engine ────────────────────────────────────────────────────────


def replay(
    *,
    use_vix: bool,
    use_ma: bool,
    use_intramonth: bool,
    mkt_filters: pd.DataFrame,
) -> tuple[pd.Series, int]:
    """Replay from WARMUP through Q1_END; return (returns_q1, n_intramonth).

    Returns daily returns sliced to [Q1_START, Q1_END] only.
    """
    prices, returns, regime_daily, allocations, W_ON, W_OFF = _load_shared_data()

    prev_month = None
    intra_count = 0
    cur_w: dict = {a: 1 / len(ASSETS) for a in ASSETS}
    port: list = []

    for date in returns.index:
        row = regime_daily.loc[date] if date in regime_daily.index else None
        if row is None or pd.isna(row.get("regime", None)):
            port.append(np.nan)
            continue

        month = date.to_period("M")

        # ── Monthly rebalance ────────────────────────────────────────────────
        if prev_month is None or month != prev_month:
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

        # ── Intramonth Friday trigger ────────────────────────────────────────
        # Only fires when NOT a monthly rebalance day (elif).
        # Never updates prev_month — monthly cadence is preserved.
        elif use_intramonth and mkt_filters is not None and date.weekday() == 4:
            triggered, _reason = check_intramonth_trigger(
                mkt_filters,
                date,
                VIX_INTRAMONTH_THRESHOLD,
                SPY_WEEKLY_DROP_THRESHOLD,
            )
            if triggered:
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
                intra_count += 1

        ret = sum(
            float(returns.loc[date, a]) * float(cur_w.get(a, 0.0)) for a in ASSETS
        )
        port.append(ret)

    full = pd.Series(port, index=returns.index, name="port").dropna()
    q1 = full[(full.index >= Q1_START) & (full.index <= Q1_END)]
    return q1, intra_count


# ── Trigger diagnostics ───────────────────────────────────────────────────────


def _show_trigger_diary(mkt_filters: pd.DataFrame) -> None:
    q1 = mkt_filters[
        (mkt_filters.index >= Q1_START) & (mkt_filters.index <= Q1_END)
    ].copy()

    spy_full = mkt_filters["SPY_close"].loc[:Q1_END]
    q1["SPY_MA200"] = spy_full.rolling(MA_LOOKBACK, min_periods=MA_LOOKBACK).mean()

    fridays = q1[q1.index.dayofweek == 4].copy()

    print(f"\n  Fridays in Q1 2026: {len(fridays)}")
    print(
        f"  {'Date':<12} {'VIX':>7} {'VIX>30?':>8}  "
        f"{'SPYwkRet':>10} {'SPY<MA?':>8}  Trigger?"
    )
    print(f"  {'-' * 12} {'-' * 7} {'-' * 8}  {'-' * 10} {'-' * 8}  {'-' * 8}")

    spy_weekly_series = mkt_filters["SPY_close"].resample("W-FRI").last().dropna()

    for dt, row in fridays.iterrows():
        vix = float(row["VIX_close"])
        vix_fire = vix > VIX_INTRAMONTH_THRESHOLD

        # Weekly return ending this Friday
        wk = spy_weekly_series.loc[:dt]
        if len(wk) >= 2:
            wk_ret = float(wk.iloc[-1]) / float(wk.iloc[-2]) - 1.0
        else:
            wk_ret = float("nan")

        spy_today = float(row["SPY_close"])
        ma = row["SPY_MA200"]
        ma_fire = (not pd.isna(ma)) and (spy_today < float(ma))
        spy_fire = (not np.isnan(wk_ret)) and (wk_ret < -SPY_WEEKLY_DROP_THRESHOLD)

        trigger = vix_fire or spy_fire
        flag = "*** FIRE" if trigger else ""

        wk_str = f"{wk_ret:+.2%}" if not np.isnan(wk_ret) else "   n/a"
        print(
            f"  {str(dt.date()):<12} {vix:>7.1f} {'YES' if vix_fire else 'no':>8}  "
            f"{wk_str:>10} {'YES' if ma_fire else 'no':>8}  {flag}"
        )


# ── Regression data (2022-2026) ───────────────────────────────────────────────

_REGRESSION_CACHE: dict = {}


def _load_regression_data() -> tuple:
    """Load price/regime data from REGRESSION_WARMUP for multi-year regression tests."""
    if _REGRESSION_CACHE:
        return tuple(
            _REGRESSION_CACHE[k]
            for k in (
                "prices",
                "returns",
                "regime_daily",
                "allocations",
                "W_ON",
                "W_OFF",
            )
        )

    raw = yf.download(
        TICKERS,
        start=REGRESSION_WARMUP,
        end=REGRESSION_END,
        progress=False,
        auto_adjust=True,
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
        n = len(regs) or 1
        return {a: v / n for a, v in out.items()}

    W_ON = _avg({"Recovery", "Overheating"})
    W_OFF = _avg({"Contraction", "Stagflation"})

    _REGRESSION_CACHE.update(
        {
            "prices": prices,
            "returns": returns,
            "regime_daily": regime_daily,
            "allocations": allocations,
            "W_ON": W_ON,
            "W_OFF": W_OFF,
        }
    )
    return prices, returns, regime_daily, allocations, W_ON, W_OFF


def replay_regression(mkt_filters: pd.DataFrame) -> tuple[pd.Series, list]:
    """Full replay from REGRESSION_WARMUP to REGRESSION_END.

    Uses current config values (VIX_INTRAMONTH_THRESHOLD=27,
    SPY_WEEKLY_DROP_THRESHOLD=0.99, VIX/MA overrides enabled).

    Returns
    -------
    full_rets : pd.Series   daily portfolio returns across the full range
    triggers  : list of (pd.Timestamp, str)   every intramonth trigger fired
    """
    prices, returns, regime_daily, allocations, W_ON, W_OFF = _load_regression_data()

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
            cur_w = _calc_w(
                date,
                prices,
                returns,
                regime_daily,
                allocations,
                W_ON,
                W_OFF,
                mkt_filters,
                use_vix=True,
                use_ma=True,
            )
            prev_month = month

        elif mkt_filters is not None and date.weekday() == 4:
            # Guard: skip dates absent from mkt_filters (rare holiday gaps)
            if date not in mkt_filters.index:
                pass
            else:
                fired, reason = check_intramonth_trigger(
                    mkt_filters,
                    date,
                    VIX_INTRAMONTH_THRESHOLD,
                    SPY_WEEKLY_DROP_THRESHOLD,
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
                        use_vix=True,
                        use_ma=True,
                    )
                    triggers.append((date, reason))

        ret = sum(
            float(returns.loc[date, a]) * float(cur_w.get(a, 0.0)) for a in ASSETS
        )
        port.append(ret)

    full_rets = pd.Series(port, index=returns.index, name="port").dropna()
    return full_rets, triggers


# ── Regression assertions ─────────────────────────────────────────────────────


def regression_tests(mkt_filters_extended: pd.DataFrame) -> None:
    """Three locked-in assertions that must hold at the final config:

    1. Zero intramonth triggers in 2023 — VIX stayed below 27 in a calm year.
    2. Zero intramonth triggers in 2024 — (moderate vol; Aug-2024 spike assessed).
    3. Q1-2026 max drawdown with overrides > REGRESSION_DD_FLOOR (-7.50%),
       locking in the +93 bp improvement measured in the threshold sweep.
    """
    sep = "=" * 68
    print(f"\n{sep}")
    print("  REGRESSION TESTS")
    print(f"  Config: VIX_INTRAMONTH_THRESHOLD = {VIX_INTRAMONTH_THRESHOLD}")
    print(
        f"          SPY_WEEKLY_DROP_THRESHOLD = {SPY_WEEKLY_DROP_THRESHOLD}  (disabled)"
    )
    print(f"          Q1-2026 DD floor          = {REGRESSION_DD_FLOOR:.1%}")
    print(sep)

    print("\n  [R1] Loading 2022-2026 price/regime data ...")
    _load_regression_data()

    print("  [R2] Replaying 2022 -> 2026-04-05 with overrides + VIX=27 trigger ...")
    full_rets, trigger_list = replay_regression(mkt_filters_extended)
    print(
        f"       {len(full_rets)} trading days  "
        f"{full_rets.index.min().date()} -> {full_rets.index.max().date()}"
    )
    print(f"       Total triggers fired: {len(trigger_list)}")

    # ── Slice triggers by year ───────────────────────────────────────────────
    def _trigs_in(y_start: str, y_end: str) -> list:
        p0, p1 = pd.Timestamp(y_start), pd.Timestamp(y_end)
        return [(d, r) for d, r in trigger_list if p0 <= d <= p1]

    trig_2023 = _trigs_in("2023-01-01", "2023-12-31")
    trig_2024 = _trigs_in("2024-01-01", "2024-12-31")

    # ── Q1-2026 max DD ───────────────────────────────────────────────────────
    q1_rets = full_rets[(full_rets.index >= Q1_START) & (full_rets.index <= Q1_END)]
    max_dd_q1 = _max_dd(q1_rets)

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n  Triggers in 2023: {len(trig_2023)}")
    for d, r in trig_2023:
        print(f"    {d.date()}  {r}")

    print(f"  Triggers in 2024: {len(trig_2024)}")
    for d, r in trig_2024:
        print(f"    {d.date()}  {r}")

    print(f"  Q1-2026 max DD  : {max_dd_q1:.2%}  (floor: > {REGRESSION_DD_FLOOR:.1%})")

    # ── Assertions ───────────────────────────────────────────────────────────
    failures: list[str] = []

    if len(trig_2023) != 0:
        failures.append(
            f"2023 trigger count = {len(trig_2023)}, expected 0 "
            f"(VIX crossed {VIX_INTRAMONTH_THRESHOLD} on: "
            f"{[str(d.date()) for d, _ in trig_2023]})"
        )

    if len(trig_2024) != 0:
        failures.append(
            f"2024 trigger count = {len(trig_2024)}, expected 0 "
            f"(VIX crossed {VIX_INTRAMONTH_THRESHOLD} on: "
            f"{[str(d.date()) for d, _ in trig_2024]})"
        )

    if max_dd_q1 <= REGRESSION_DD_FLOOR:
        failures.append(
            f"Q1-2026 max DD = {max_dd_q1:.2%}, must be > {REGRESSION_DD_FLOOR:.1%}"
        )

    print()
    if failures:
        for msg in failures:
            print(f"  REGRESSION FAIL: {msg}")
        print(f"\n  REGRESSION RESULT: FAIL ({len(failures)} assertion(s) failed)")
        print(sep)
        sys.exit(1)

    print("  PASS: 2023 trigger count = 0")
    print("  PASS: 2024 trigger count = 0")
    print(f"  PASS: Q1-2026 max DD = {max_dd_q1:.2%} > {REGRESSION_DD_FLOOR:.1%}")
    print("\n  REGRESSION RESULT: PASS")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    sep = "=" * 68
    sep2 = "-" * 68

    print(sep)
    print("  Q1 2026  THREE-SCENARIO OVERRIDE STRESS TEST")
    print(f"  Period  : {Q1_START}  ->  {Q1_END}")
    print(
        f"  Monthly VIX override threshold : {VIX_THRESHOLD}  (cap -> {VIX_RISK_ON_CAP})"
    )
    print(
        f"  MA filter                      : 200d  (equity cap -> {MA_EQUITY_CAP:.0%})"
    )
    print(f"  Intramonth VIX threshold       : {VIX_INTRAMONTH_THRESHOLD}")
    print(f"  Intramonth SPY weekly drop     : -{SPY_WEEKLY_DROP_THRESHOLD:.0%}")
    print(sep)

    # ── Fetch market data ────────────────────────────────────────────────────
    print("\n[1/4] Fetching VIX + SPY filter data ...")
    mkt_filters = fetch_market_filters(WARMUP, Q1_END)  # raises ValueError if stale
    print(
        f"  {len(mkt_filters)} trading days  "
        f"{mkt_filters.index.min().date()} -> {mkt_filters.index.max().date()}"
    )

    # ── Friday trigger diary ─────────────────────────────────────────────────
    print("\n[2/4] Friday trigger diary ...")
    _show_trigger_diary(mkt_filters)

    # ── Run all three scenarios ───────────────────────────────────────────────
    print("\n[3/4] Replaying Q1 2026 (3 scenarios) ...")
    ra, na = replay(
        use_vix=False, use_ma=False, use_intramonth=False, mkt_filters=mkt_filters
    )
    rb, nb = replay(
        use_vix=True, use_ma=True, use_intramonth=False, mkt_filters=mkt_filters
    )
    rc, nc = replay(
        use_vix=True, use_ma=True, use_intramonth=True, mkt_filters=mkt_filters
    )

    scenarios = {
        "(a) Baseline": (ra, na),
        "(b) Monthly+OV": (rb, nb),
        "(c) Weekly+OV": (rc, nc),
    }

    # ── Results table ────────────────────────────────────────────────────────
    print("\n[4/4] Results\n")
    print(
        f"  {'Scenario':<18}  {'TotalRet':>10}  {'MaxDD':>9}  {'Sortino':>9}  {'#Intra':>7}"
    )
    print(f"  {sep2[:66]}")

    rows: dict = {}
    for label, (rets, n_intra) in scenarios.items():
        rows[label] = {
            "Total Return": _total_ret(rets),
            "Max Drawdown": _max_dd(rets),
            "Sortino": _sortino(rets),
            "n_intra": n_intra,
        }
        dd = rows[label]["Max Drawdown"]
        tr = rows[label]["Total Return"]
        so = rows[label]["Sortino"]
        print(f"  {label:<18}  {tr:>10.2%}  {dd:>9.2%}  {so:>9.2f}  {n_intra:>7}")

    print(f"  {sep2[:66]}")

    # ── Monthly breakdown ────────────────────────────────────────────────────
    print("\n  Monthly return breakdown\n")
    months = sorted(
        set(ra.index.to_period("M"))
        | set(rb.index.to_period("M"))
        | set(rc.index.to_period("M"))
    )
    print(f"  {'Month':<10}  {'(a)':>9}  {'(b)':>9}  {'(c)':>9}  {'(c)-(a)':>9}")
    print(f"  {'-' * 10}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 9}")
    for ym in months:

        def _mr(s, _ym=ym):
            sub = s[s.index.to_period("M") == _ym]
            return _total_ret(sub) if not sub.empty else float("nan")

        a_r, b_r, c_r = _mr(ra), _mr(rb), _mr(rc)
        diff = c_r - a_r if not (np.isnan(a_r) or np.isnan(c_r)) else float("nan")
        sgn = "+" if diff >= 0 else ""
        print(
            f"  {str(ym):<10}  {a_r:>9.2%}  {b_r:>9.2%}  {c_r:>9.2%}    {sgn}{diff:.2%}"
        )

    # ── Pass / fail ──────────────────────────────────────────────────────────
    dd_a = rows["(a) Baseline"]["Max Drawdown"]
    dd_c = rows["(c) Weekly+OV"]["Max Drawdown"]
    improvement_bp = (dd_c - dd_a) * 10_000  # positive = less negative = better

    print(f"\n{sep}")
    print(f"  Max DD improvement (c) vs (a): {improvement_bp:+.1f} bp")
    print(f"  Required for PASS           : >= {MATERIALLY_BETTER_DD_BP} bp")
    print()

    if improvement_bp >= MATERIALLY_BETTER_DD_BP:
        print("  RESULT: PASS  — weekly trigger materially reduces drawdown")
    else:
        print(
            f"  RESULT: FAIL  — improvement {improvement_bp:.1f} bp < "
            f"{MATERIALLY_BETTER_DD_BP} bp threshold"
        )
        print("  NOTE : Trigger may need lower thresholds or the stress event")
        print("         occurred before enough weekly data accumulated.")
        sys.exit(1)

    print(sep)

    # ── Regression tests (locked-in assertions at final config) ──────────────
    print("\nFetching extended VIX+SPY data for regression tests (2022-2026) ...")
    mkt_filters_extended = fetch_market_filters(REGRESSION_WARMUP, REGRESSION_END)
    print(
        f"  {len(mkt_filters_extended)} trading days  "
        f"{mkt_filters_extended.index.min().date()} -> "
        f"{mkt_filters_extended.index.max().date()}"
    )
    regression_tests(mkt_filters_extended)


if __name__ == "__main__":
    main()
