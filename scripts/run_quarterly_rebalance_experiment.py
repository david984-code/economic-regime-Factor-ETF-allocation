"""Experiment: Quarterly Rebalancing vs Monthly Baseline.

One variable changed: rebalance trigger -- monthly -> quarterly (Jan/Apr/Jul/Oct only).
Signal (24M SPY momentum), sleeves, blend logic, and costs are unchanged.
Fast-mode window: 2018-01-01 to 2024-12-31.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR
from src.data.market_ingestion import fetch_prices
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FAST_START = "2018-01-01"
FAST_END = "2024-12-31"

# Shared kwargs -- identical for both runs; only quarterly_rebalance differs
BASE_KWARGS = {
    "start": FAST_START,
    "end": FAST_END,
    "min_train_months": 60,
    "test_months": 12,
    "expanding": True,
    "use_stagflation_override": False,
    "use_stagflation_risk_on_cap": False,
    "use_regime_smoothing": False,
    "use_hybrid_signal": True,
    "hybrid_macro_weight": 0.0,  # pure market signal
    "market_lookback_months": 24,  # 24M momentum
    "use_momentum": True,  # +momentum (risk_on when momentum > 0)
    "trend_filter_type": "none",
    "vol_scaling_method": "none",
    "portfolio_construction_method": "equal_weight",
    "momentum_12m_weight": 0.0,
    "fast_mode": True,
    "skip_persist": True,
    "use_vol_regime": False,
}


def _overall(df: pd.DataFrame) -> pd.Series:
    return df[df["segment"] == "OVERALL"].iloc[0]


def _build_diagnostics(prices: pd.DataFrame, regime_df: pd.DataFrame) -> dict:
    """Compute signal flip dates, rebalance execution dates, and allocation divergence."""
    spy_monthly = prices["SPY"].resample("ME").last()

    # --- Signal: 24M momentum at each month-end ---
    signal_records = []
    for i in range(len(spy_monthly)):
        if i < 24:
            signal_records.append({"date": spy_monthly.index[i], "risk_on_raw": np.nan})
            continue
        mom = (spy_monthly.iloc[i] / spy_monthly.iloc[i - 24]) - 1
        signal_records.append(
            {
                "date": spy_monthly.index[i],
                "momentum_24m": mom,
                "risk_on_raw": 1 if mom > 0 else 0,
            }
        )

    signal_df = pd.DataFrame(signal_records).set_index("date")

    # Log A: signal flip dates (where risk_on_raw changes)
    signal_clean = signal_df["risk_on_raw"].dropna()
    flips = signal_clean[signal_clean != signal_clean.shift(1)].dropna()
    log_a = []
    for date, val in flips.items():
        signal_clean.shift(1).loc[date]
        direction = "risk_off -> risk_on" if val == 1 else "risk_on -> risk_off"
        mom_val = (
            signal_df.loc[date, "momentum_24m"]
            if "momentum_24m" in signal_df.columns
            else np.nan
        )
        log_a.append(
            {
                "month_end_date": date.strftime("%Y-%m-%d"),
                "direction": direction,
                "momentum_24m": f"{mom_val:+.4f}" if not np.isnan(mom_val) else "n/a",
            }
        )

    # Log B: rebalance execution dates
    # Baseline: first trading day of every new month
    # Experiment: first trading day of Jan/Apr/Jul/Oct only
    daily_idx = prices.index
    months = pd.Series(daily_idx).dt.to_period("M").values
    month_changed_mask = np.concatenate([[True], months[1:] != months[:-1]])
    month_changed_dates = daily_idx[month_changed_mask]

    quarter_months = {1, 4, 7, 10}
    log_b = []
    for d in month_changed_dates:
        if d < pd.Timestamp(FAST_START) or d > pd.Timestamp(FAST_END):
            continue
        is_q = d.month in quarter_months
        # Signal value: forward-filled month-end signal as of this date
        me_date = spy_monthly.index[spy_monthly.index <= d]
        if len(me_date) < 24:
            continue
        last_me = me_date[-1]
        sig = (
            signal_df.loc[last_me, "risk_on_raw"]
            if last_me in signal_df.index
            else np.nan
        )
        log_b.append(
            {
                "execution_date": d.strftime("%Y-%m-%d"),
                "baseline_executes": "YES",
                "experiment_executes": "YES" if is_q else "NO (held)",
                "risk_on_signal": f"{sig:.0f}"
                if not (isinstance(sig, float) and np.isnan(sig))
                else "n/a",
            }
        )

    # Log C: allocation divergence periods
    # Divergence opens when baseline rebalances on a non-quarter month and signal differs from prior quarter
    log_c = []
    last_q_signal = None
    divergence_open = None
    divergence_baseline_signal = None

    for rec in log_b:
        d = pd.Timestamp(rec["execution_date"])
        sig = (
            int(rec["risk_on_signal"])
            if rec["risk_on_signal"] not in ("n/a", "")
            else None
        )
        if sig is None:
            continue

        if rec["experiment_executes"] == "YES":
            # Quarter-start: both models sync up
            if divergence_open is not None:
                # Close divergence
                log_c.append(
                    {
                        "period_start": divergence_open.strftime("%Y-%m-%d"),
                        "period_end": d.strftime("%Y-%m-%d"),
                        "baseline_risk_on": divergence_baseline_signal,
                        "experiment_risk_on": f"{last_q_signal:.0f}"
                        if last_q_signal is not None
                        else "n/a",
                        "note": "experiment held prior quarter signal",
                    }
                )
                divergence_open = None
            last_q_signal = sig
        else:
            # Non-quarter month: baseline rebalanced, experiment held
            if last_q_signal is not None and sig != last_q_signal:
                # Signal changed between quarters -> real divergence
                if divergence_open is None:
                    divergence_open = d
                    divergence_baseline_signal = sig

    return {
        "signal_df": signal_df,
        "log_a": log_a,
        "log_b": log_b,
        "log_c": log_c,
    }


def _march_2020_timeline(log_b: list, signal_df: pd.DataFrame) -> dict:
    """Extract March 2020 crash allocation timeline."""
    # When did signal flip to risk-off?
    signal_clean = signal_df["risk_on_raw"].dropna()
    flip_to_off = signal_clean[(signal_clean == 0) & (signal_clean.shift(1) == 1)]
    flip_2020 = flip_to_off[flip_to_off.index.year.isin([2019, 2020, 2021])]

    signal_flip_date = (
        flip_2020.index[0].strftime("%Y-%m-%d")
        if len(flip_2020) > 0
        else "no flip in window"
    )

    # First execution date after signal flip for each model
    baseline_exec = "not found"
    experiment_exec = "not found"
    for rec in log_b:
        d = pd.Timestamp(rec["execution_date"])
        if d < pd.Timestamp("2020-01-01") or d > pd.Timestamp("2020-12-31"):
            continue
        sig = rec["risk_on_signal"]
        if sig == "0":
            if baseline_exec == "not found":
                baseline_exec = rec["execution_date"]
            if rec["experiment_executes"] == "YES" and experiment_exec == "not found":
                experiment_exec = rec["execution_date"]

    # Allocations on trough date (2020-03-23)
    pd.Timestamp("2020-03-23")
    timeline_rows = []
    [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-02-03"),
        pd.Timestamp("2020-03-02"),
        pd.Timestamp("2020-03-23"),  # trough
        pd.Timestamp("2020-04-01"),
    ]
    for rec in log_b:
        d = pd.Timestamp(rec["execution_date"])
        if pd.Timestamp("2019-12-01") <= d <= pd.Timestamp("2020-06-30"):
            timeline_rows.append(rec)

    return {
        "signal_flip_date": signal_flip_date,
        "baseline_first_risk_off_exec": baseline_exec,
        "experiment_first_risk_off_exec": experiment_exec,
        "timeline_rows": timeline_rows,
    }


def _2022_risk_off(log_b: list, signal_df: pd.DataFrame) -> dict:
    """Find first risk-off execution in 2022."""
    signal_clean = signal_df["risk_on_raw"].dropna()
    # Find signal flip to risk-off in 2021-2022
    flip_to_off = signal_clean[(signal_clean == 0) & (signal_clean.shift(1) == 1)]
    flip_2022 = flip_to_off[flip_to_off.index.year.isin([2021, 2022])]
    flip_date = (
        flip_2022.index[0].strftime("%Y-%m-%d") if len(flip_2022) > 0 else "no flip"
    )

    baseline_exec = "not found"
    experiment_exec = "not found"
    for rec in log_b:
        d = pd.Timestamp(rec["execution_date"])
        if d.year not in (2021, 2022, 2023):
            continue
        if rec["risk_on_signal"] == "0":
            if baseline_exec == "not found":
                baseline_exec = rec["execution_date"]
            if rec["experiment_executes"] == "YES" and experiment_exec == "not found":
                experiment_exec = rec["execution_date"]

    return {
        "signal_flip_date": flip_date,
        "baseline_first_exec": baseline_exec,
        "experiment_first_exec": experiment_exec,
    }


def main():
    logger.info("=" * 70)
    logger.info("EXPERIMENT: Quarterly Rebalancing (fast-mode)")
    logger.info("Window: %s to %s", FAST_START, FAST_END)
    logger.info("=" * 70)

    logger.info("\n[1/2] Running BASELINE (monthly rebalancing)...")
    df_base = run_walk_forward_evaluation(**BASE_KWARGS, quarterly_rebalance=False)

    logger.info("\n[2/2] Running EXPERIMENT (quarterly rebalancing)...")
    df_exp = run_walk_forward_evaluation(**BASE_KWARGS, quarterly_rebalance=True)

    if df_base.empty or df_exp.empty:
        logger.error("One or both runs returned empty results. Aborting.")
        sys.exit(1)

    b = _overall(df_base)
    e = _overall(df_exp)

    # ── 5-metric table ──────────────────────────────────────────────────────
    cagr_delta = e["Strategy_CAGR"] - b["Strategy_CAGR"]
    sharpe_delta = e["Strategy_Sharpe"] - b["Strategy_Sharpe"]
    maxdd_delta = e["Strategy_MaxDD"] - b["Strategy_MaxDD"]
    vol_delta = e["Strategy_Vol"] - b["Strategy_Vol"]
    to_base = b.get("Strategy_Turnover", float("nan"))
    to_exp = e.get("Strategy_Turnover", float("nan"))
    to_delta = (
        to_exp - to_base
        if not (np.isnan(to_base) or np.isnan(to_exp))
        else float("nan")
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT: Quarterly Rebalancing")
    print("Variable changed: rebalance trigger -- monthly -> quarterly")
    print("=" * 70)

    print(f"\n{'METRICS vs BASELINE':}")
    print(f"{'':30} {'Baseline':>12} {'Experiment':>12} {'Delta':>10}")
    print("-" * 66)
    print(
        f"{'CAGR':30} {b['Strategy_CAGR']:>12.2%} {e['Strategy_CAGR']:>12.2%} {cagr_delta:>+10.2%}"
    )
    print(
        f"{'Sharpe':30} {b['Strategy_Sharpe']:>12.3f} {e['Strategy_Sharpe']:>12.3f} {sharpe_delta:>+10.3f}"
    )
    print(
        f"{'MaxDD':30} {b['Strategy_MaxDD']:>12.2%} {e['Strategy_MaxDD']:>12.2%} {maxdd_delta:>+10.2%}"
    )
    print(
        f"{'Vol':30} {b['Strategy_Vol']:>12.2%} {e['Strategy_Vol']:>12.2%} {vol_delta:>+10.2%}"
    )
    if not np.isnan(to_base):
        print(f"{'Turnover':30} {to_base:>12.2%} {to_exp:>12.2%} {to_delta:>+10.2%}")
    else:
        print(f"{'Turnover':30} {'n/a':>12} {'n/a':>12} {'n/a':>10}")

    # ── Kill switch ──────────────────────────────────────────────────────────
    print("\nKILL SWITCH EVALUATION")
    print("-" * 40)
    ks_sharpe = sharpe_delta < 0.02
    ks_cagr = cagr_delta < 0.0025
    # difficult-period: check segments containing 2020 and 2022
    seg_base = df_base[df_base["segment"] != "OVERALL"].copy()
    seg_exp = df_exp[df_exp["segment"] != "OVERALL"].copy()
    difficult_years = {2020, 2022}

    def _in_difficult(row):
        try:
            y_start = pd.Period(row["test_start"], freq="M").year
            y_end = pd.Period(row["test_end"], freq="M").year
            return any(y in difficult_years for y in range(y_start, y_end + 1))
        except Exception:
            return False

    diff_mask_b = seg_base.apply(_in_difficult, axis=1)
    diff_mask_e = seg_exp.apply(_in_difficult, axis=1)
    diff_sharpe_b = (
        seg_base.loc[diff_mask_b, "Strategy_Sharpe"].mean()
        if diff_mask_b.any()
        else float("nan")
    )
    diff_sharpe_e = (
        seg_exp.loc[diff_mask_e, "Strategy_Sharpe"].mean()
        if diff_mask_e.any()
        else float("nan")
    )
    difficult_improved = (
        not np.isnan(diff_sharpe_e)
        and not np.isnan(diff_sharpe_b)
        and diff_sharpe_e > diff_sharpe_b
    )
    ks_no_difficult = not difficult_improved

    kill = ks_sharpe and ks_cagr and ks_no_difficult
    print(
        f"  Sharpe delta < +0.02?         {'YES' if ks_sharpe else 'NO':4}  (delta = {sharpe_delta:+.3f})"
    )
    print(
        f"  CAGR delta < +0.25%?          {'YES' if ks_cagr else 'NO':4}  (delta = {cagr_delta:+.2%})"
    )
    print(
        f"  No difficult-period impr.?    {'YES' if ks_no_difficult else 'NO':4}  "
        f"(baseline = {diff_sharpe_b:.3f}, experiment = {diff_sharpe_e:.3f})"
    )
    print(
        f"  Kill switch fires?            {'YES -> REJECT' if kill else 'NO -> continue'}"
    )

    # ── Diagnostics (signal, rebalance, divergence logs) ────────────────────
    logger.info("\nBuilding signal and execution diagnostics...")
    prices = fetch_prices(start=FAST_START, end=FAST_END)

    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes

        regime_df = load_regimes()
    regime_df = regime_df.sort_index()

    diag = _build_diagnostics(prices, regime_df)
    m2020 = _march_2020_timeline(diag["log_b"], diag["signal_df"])
    r2022 = _2022_risk_off(diag["log_b"], diag["signal_df"])

    # Log A
    print("\nLOG A: SIGNAL FLIP DATES (24M SPY momentum crosses zero)")
    print("-" * 66)
    print(f"  {'Month-end date':15} {'Direction':25} {'Momentum 24M':>14}")
    for row in diag["log_a"]:
        print(
            f"  {row['month_end_date']:15} {row['direction']:25} {row['momentum_24m']:>14}"
        )

    # Log B
    print("\nLOG B: REBALANCE EXECUTION DATES")
    print("-" * 66)
    print(
        f"  {'Exec date':12} {'Baseline':12} {'Experiment':20} {'Signal risk_on':>14}"
    )
    for row in diag["log_b"]:
        print(
            f"  {row['execution_date']:12} {row['baseline_executes']:12} "
            f"{row['experiment_executes']:20} {row['risk_on_signal']:>14}"
        )

    # Log C
    print("\nLOG C: ALLOCATION DIVERGENCE PERIODS")
    print("-" * 66)
    if diag["log_c"]:
        for row in diag["log_c"]:
            print(
                f"  {row['period_start']} -> {row['period_end']}: "
                f"baseline risk_on={row['baseline_risk_on']}, "
                f"experiment held={row['experiment_risk_on']}  [{row['note']}]"
            )
    else:
        print(
            "  No allocation divergences detected (signal did not change between quarters)."
        )

    # March 2020
    print("\nMARCH 2020 TIMELINE")
    print("-" * 66)
    print(f"  Signal flip to risk-off (month-end): {m2020['signal_flip_date']}")
    print(
        f"  Baseline first risk-off execution:   {m2020['baseline_first_risk_off_exec']}"
    )
    print(
        f"  Experiment first risk-off execution: {m2020['experiment_first_risk_off_exec']}"
    )
    print("\n  Execution log (2019-12 -> 2020-06):")
    print(f"  {'Exec date':12} {'Baseline':12} {'Experiment':20} {'Signal':>8}")
    for row in m2020["timeline_rows"]:
        print(
            f"  {row['execution_date']:12} {row['baseline_executes']:12} "
            f"{row['experiment_executes']:20} {row['risk_on_signal']:>8}"
        )

    # 2022
    print("\n2022 FIRST RISK-OFF EXECUTION")
    print("-" * 66)
    print(f"  Signal flip to risk-off (month-end): {r2022['signal_flip_date']}")
    print(f"  Baseline first executed on:          {r2022['baseline_first_exec']}")
    print(f"  Experiment first executed on:        {r2022['experiment_first_exec']}")

    # Escalation
    print("\nESCALATION DECISION")
    print("-" * 66)
    esc_sharpe = sharpe_delta >= 0.02
    esc_turnover = (
        not np.isnan(to_delta) and to_delta <= -0.40 * to_base and sharpe_delta >= -0.02
    )
    esc_difficult = difficult_improved
    escalate = esc_sharpe or esc_turnover or esc_difficult
    print(
        f"  Sharpe delta ≥ +0.02?                      {'YES' if esc_sharpe else 'NO'}"
    )
    print(
        f"  Turnover reduction ≥ 40%, Sharpe ≥ -0.02? {'YES' if esc_turnover else 'NO'}"
    )
    print(
        f"  Difficult-period improvement?              {'YES' if esc_difficult else 'NO'}"
    )
    print(f"\n  -> {'ESCALATE TO FULL WALK-FORWARD' if escalate else 'REJECT'}")

    if kill:
        print("\n  FAST-MODE VERDICT: REJECTED (kill switch fired)")
    elif escalate:
        print("\n  FAST-MODE VERDICT: PASS -- escalate to full walk-forward validation")
    else:
        print(
            "\n  FAST-MODE VERDICT: INSUFFICIENT -- does not meet escalation thresholds"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
