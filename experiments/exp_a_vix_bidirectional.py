"""
EXPERIMENT A -- Bidirectional VIX Intramonth Rebalance Trigger
=============================================================

Hypothesis
----------
Adding a VIX *recovery* trigger (VIX drops below a low threshold AFTER having
been in a stressed state) will improve risk-adjusted returns by capturing
mid-month stress-to-recovery regime transitions, without materially increasing
turnover.

Variable changed
----------------
Intramonth trigger logic:
  - Baseline : fires only when VIX > VIX_HI (defensive)
  - Experiment: fires when VIX > VIX_HI  **OR**  when VIX drops below VIX_LO
                after having been above VIX_STRESS within the trailing window
                (state-transition trigger, not a level trigger)

All other parameters (allocation, vol-scaling, MA filter, VIX cap) unchanged.
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

# ── Static settings (match production) ────────────────────────────────────────
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
EQUITY_TICKERS = ["SPY", "MTUM", "VLUE", "USMV", "QUAL", "IJR", "VIG"]

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}


# ── Data loading (shared across runs) ────────────────────────────────────────


def load_data():
    """Download prices, regime labels, allocations — called once."""
    print("Downloading price data …")
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

    print("Downloading VIX + SPY for market filters …")
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


# ── Helpers (copied from backtest.py to keep experiment self-contained) ───────


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


# ── Rebalance engine ─────────────────────────────────────────────────────────


def run_backtest(
    prices,
    returns,
    mkt_filters,
    regime_df,
    allocations,
    *,
    vix_hi: float = 27.0,
    vix_lo: float | None = None,
    vix_stress_memory: float = 25.0,
    stress_lookback_days: int = 20,
    trigger_on_friday_only: bool = True,
    cooldown_days: int = 5,
) -> dict:
    """Run the full backtest with parameterised intramonth trigger.

    Parameters
    ----------
    vix_hi : VIX level above which defensive trigger fires (production=27).
    vix_lo : VIX level below which risk-on trigger fires.  None = disabled.
    vix_stress_memory : VIX must have been above this level within the lookback
                        for the vix_lo recovery trigger to be eligible.
    stress_lookback_days : how many trading days to look back for prior stress.
    trigger_on_friday_only : only check triggers on Friday closes.
    cooldown_days : minimum trading days between intramonth rebalances.
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
                w, prices["SPY"].loc[:date], EQUITY_TICKERS, MA_LOOKBACK, MA_EQUITY_CAP
            )

        return w

    def was_recently_stressed(date):
        """Check if VIX was above vix_stress_memory within the trailing window."""
        vix_series = mkt_filters["VIX_close"]
        trailing = vix_series.loc[:date].tail(stress_lookback_days)
        return (trailing > vix_stress_memory).any()

    def check_trigger(date):
        """Return (triggered: bool, reason: str)."""
        if date not in mkt_filters.index:
            return False, "no market data"
        vix = mkt_filters.loc[date, "VIX_close"]
        if pd.isna(vix):
            return False, "VIX NaN"
        vix = float(vix)

        # Defensive trigger: VIX spikes above high threshold
        if vix > vix_hi:
            return True, f"VIX={vix:.1f} > {vix_hi} (defensive)"

        # Recovery trigger: VIX drops below low threshold
        # ONLY fires if VIX was recently stressed (state transition)
        if vix_lo is not None and vix < vix_lo:
            if was_recently_stressed(date):
                return True, f"VIX={vix:.1f} < {vix_lo} after stress (risk-on recovery)"

        return False, "no trigger"

    # ── Rebalance loop ────────────────────────────────────────────────────────
    port_rets = []
    prev_month = None
    current_w = {a: 1.0 / len(ASSETS) for a in ASSETS}
    prev_w = None

    total_turnover = 0.0
    rebalance_count = 0
    intramonth_count = 0
    intramonth_reasons = []
    days_since_last_intramonth = cooldown_days + 1

    for date in returns.index:
        regime = regime_df.loc[date, "regime"] if date in regime_df.index else np.nan
        if pd.isna(regime):
            port_rets.append(np.nan)
            days_since_last_intramonth += 1
            continue

        month = date.to_period("M")
        did_rebalance = False

        # Monthly rebalance
        if prev_month is None or month != prev_month:
            current_w = calc_weights(date)
            prev_month = month
            did_rebalance = True
            rebalance_count += 1
            days_since_last_intramonth = cooldown_days + 1

        # Intramonth trigger
        elif days_since_last_intramonth >= cooldown_days:
            if not trigger_on_friday_only or date.weekday() == 4:
                triggered, reason = check_trigger(date)
                if triggered:
                    current_w = calc_weights(date)
                    did_rebalance = True
                    intramonth_count += 1
                    intramonth_reasons.append((date, reason))
                    days_since_last_intramonth = 0

        if did_rebalance and prev_w is not None:
            turnover = (
                sum(
                    abs(float(current_w.get(a, 0)) - float(prev_w.get(a, 0)))
                    for a in ASSETS
                )
                / 2.0
            )
            total_turnover += turnover
        prev_w = current_w.copy()

        daily_ret = sum(
            returns.loc[date, a] * float(current_w.get(a, 0.0)) for a in ASSETS
        )
        port_rets.append(daily_ret)
        days_since_last_intramonth += 1

    port_rets = pd.Series(port_rets, index=returns.index).dropna()

    # ── Metrics ───────────────────────────────────────────────────────────────
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

    # Annual turnover
    turnover_annual = total_turnover / n_years if n_years > 0 else 0

    # Sortino
    downside = port_rets[port_rets < 0].std() * np.sqrt(252)
    sortino = (port_rets.mean() * 252) / downside if downside > 0 else 0

    # Bear-period analysis (2020-02 to 2020-03, 2022-01 to 2022-10)
    bear_periods = [
        ("COVID crash", "2020-02-01", "2020-04-01"),
        ("2022 bear", "2022-01-01", "2022-10-31"),
    ]
    bear_rets = {}
    for name, s, e in bear_periods:
        mask = (port_rets.index >= s) & (port_rets.index <= e)
        br = port_rets[mask]
        if len(br) > 0:
            bear_eq = (1 + br).cumprod()
            bear_dd = (bear_eq / bear_eq.cummax() - 1).min()
            bear_ret = bear_eq.iloc[-1] - 1
            bear_rets[name] = {"return": bear_ret, "max_dd": bear_dd}

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": dd,
        "vol": vol,
        "turnover": turnover_annual,
        "rebalances": rebalance_count,
        "intramonth": intramonth_count,
        "intramonth_reasons": intramonth_reasons,
        "bear_periods": bear_rets,
        "equity_curve": eq,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    prices, returns, mkt_filters, regime_df, allocations = load_data()

    print("\n" + "=" * 70)
    print("EXPERIMENT A: Bidirectional VIX Intramonth Trigger")
    print("=" * 70)

    # ── Baseline: production settings (VIX high trigger only at 27) ───────
    print("\n> Running BASELINE (VIX > 27 only) ...")
    baseline = run_backtest(
        prices,
        returns,
        mkt_filters,
        regime_df,
        allocations,
        vix_hi=27.0,
        vix_lo=None,
        cooldown_days=5,
    )

    # ── Experiment grid: state-transition VIX recovery triggers ──────────
    # vix_lo: recovery threshold (VIX must drop below this)
    # stress_mem: VIX must have been above this recently for low trigger to fire
    # lookback: how many trading days to look back for prior stress
    configs = [
        {"vix_lo": 16, "stress_mem": 25, "lookback": 20, "cd": 5},
        {"vix_lo": 16, "stress_mem": 25, "lookback": 30, "cd": 5},
        {"vix_lo": 18, "stress_mem": 25, "lookback": 20, "cd": 5},
        {"vix_lo": 18, "stress_mem": 25, "lookback": 30, "cd": 5},
        {"vix_lo": 18, "stress_mem": 27, "lookback": 20, "cd": 5},
        {"vix_lo": 20, "stress_mem": 27, "lookback": 20, "cd": 5},
        {"vix_lo": 20, "stress_mem": 27, "lookback": 30, "cd": 5},
        {"vix_lo": 20, "stress_mem": 30, "lookback": 20, "cd": 5},
    ]
    experiments = {}

    for cfg in configs:
        label = f"lo={cfg['vix_lo']}_mem={cfg['stress_mem']}_lb={cfg['lookback']}"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            vix_hi=27.0,
            vix_lo=cfg["vix_lo"],
            vix_stress_memory=cfg["stress_mem"],
            stress_lookback_days=cfg["lookback"],
            cooldown_days=cfg["cd"],
        )
        experiments[label] = result

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    header = f"{'Config':<22} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Vol':>7} {'Turn':>6} {'Intra':>6}"
    print(header)
    print("-" * len(header))

    def row(label, m):
        print(
            f"{label:<22} {m['cagr']:>6.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
            f" {m['max_dd']:>7.2%} {m['vol']:>6.2%} {m['turnover']:>5.1f}x {m['intramonth']:>5d}"
        )

    row("BASELINE (hi=27)", baseline)
    for label, m in experiments.items():
        row(label, m)

    # ── Delta table (vs baseline) ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DELTAS vs BASELINE")
    print("=" * 70)

    header2 = f"{'Config':<22} {'dCAGR':>7} {'dSharpe':>8} {'dMaxDD':>8} {'dVol':>7} {'dTurn':>7}"
    print(header2)
    print("-" * len(header2))

    for label, m in experiments.items():
        dc = m["cagr"] - baseline["cagr"]
        ds = m["sharpe"] - baseline["sharpe"]
        dd = m["max_dd"] - baseline["max_dd"]
        dv = m["vol"] - baseline["vol"]
        dt = m["turnover"] - baseline["turnover"]
        print(
            f"{label:<22} {dc:>+6.2%} {ds:>+8.3f} {dd:>+7.2%} {dv:>+6.2%} {dt:>+6.1f}x"
        )

    # ── Bear-period comparison ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BEAR PERIOD PERFORMANCE")
    print("=" * 70)

    for period_name in ["COVID crash", "2022 bear"]:
        print(f"\n  {period_name}:")
        bp = baseline["bear_periods"].get(period_name)
        if bp:
            print(
                f"    BASELINE         : return={bp['return']:>+7.2%}  max_dd={bp['max_dd']:>+7.2%}"
            )
        for label, m in experiments.items():
            ep = m["bear_periods"].get(period_name)
            if ep:
                print(
                    f"    {label:<18}: return={ep['return']:>+7.2%}  max_dd={ep['max_dd']:>+7.2%}"
                )

    # ── Intramonth trigger details (best experiment) ──────────────────────
    best_label = max(experiments, key=lambda k: experiments[k]["sharpe"])
    best = experiments[best_label]
    print(f"\n{'=' * 70}")
    print(f"BEST EXPERIMENT: {best_label}")
    print(f"{'=' * 70}")
    print(f"  Intramonth rebalances: {best['intramonth']}")
    if best["intramonth_reasons"]:
        print("  Trigger log (last 20):")
        for date, reason in best["intramonth_reasons"][-20:]:
            print(f"    {date.date()} — {reason}")

    # ── Kill switch ───────────────────────────────────────────────────────
    ds = best["sharpe"] - baseline["sharpe"]
    dc = best["cagr"] - baseline["cagr"]
    has_bear_improvement = False
    for pn in ["COVID crash", "2022 bear"]:
        bp = baseline["bear_periods"].get(pn, {})
        ep = best["bear_periods"].get(pn, {})
        if ep and bp and ep.get("max_dd", -1) > bp.get("max_dd", -1):
            has_bear_improvement = True

    print(f"\n{'=' * 70}")
    print("ESCALATION DECISION")
    print(f"{'=' * 70}")
    print(f"  Sharpe delta : {ds:+.3f}  (threshold: +0.020)")
    print(f"  CAGR delta   : {dc:+.3%}  (threshold: +0.25%)")
    print(f"  Bear improve : {'YES' if has_bear_improvement else 'NO'}")

    if ds >= 0.02 or dc >= 0.0025 or has_bear_improvement:
        print("\n  ✓ PASS — escalate to full walk-forward validation")
    else:
        print("\n  ✗ REJECT — below escalation thresholds (statistical noise)")


if __name__ == "__main__":
    main()
