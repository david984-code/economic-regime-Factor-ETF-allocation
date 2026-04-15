"""
EXPERIMENT F -- Full Walk-Forward Validation
=============================================

Expanding-window walk-forward test for credit spread standalone trigger
(HYG/LQD z-score + HYG momentum) replacing VIX as intramonth trigger.

Baseline: VIX > 27 (current production)
Experiment: HYG/LQD ratio z-score < -1.5 stress / > -0.5 recovery
            OR HYG 10d return < -2% stress / > 0% recovery (no VIX)

Method:
  - Minimum training window: 3 years (HYG/LQD data starts mid-2013)
  - OOS segment: 1 calendar year
  - At each fold: run both baseline and experiment using only data
    available up to that point (no lookahead)
  - Aggregate OOS-only returns for final metrics
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

VIX_INTRAMONTH_THRESHOLD = 27.0
COOLDOWN_DAYS = 5

MOMENTUM_LOOKBACK = 10
MOMENTUM_STRENGTH = 0.20
MOMENTUM_MAX_TILT = 2.0

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}

# Credit standalone params (best from Exp F screening)
ZSCORE_STRESS = -1.5
ZSCORE_RECOVERY = -0.5
HYG_MOM_STRESS = -0.02
HYG_MOM_RECOVERY = 0.00
HYG_MEMORY = 20


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
    ratio = hyg / lqd
    asset_prices = prices[TICKERS].copy()

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

    # Build credit signals
    ratio_ma = ratio.rolling(60, min_periods=30).mean()
    ratio_std = ratio.rolling(60, min_periods=30).std()
    ratio_zscore = (ratio - ratio_ma) / ratio_std.replace(0, np.nan)
    hyg_10d_ret = hyg.pct_change(10)

    credit = pd.DataFrame(index=ratio.index)
    credit["ratio_zscore"] = ratio_zscore
    credit["hyg_10d_ret"] = hyg_10d_ret

    return asset_prices, returns, mkt_filters, regime_df, allocations, credit


# ── Helpers -------------------------------------------------------------------


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


# ── Walk-forward engine -------------------------------------------------------


def run_segment(
    prices,
    returns,
    mkt_filters,
    regime_df,
    allocations,
    credit,
    start_date,
    end_date,
    *,
    trigger_mode: str = "vix_only",
):
    """Run backtest over [start_date, end_date].

    trigger_mode: "vix_only" (baseline) or "credit_standalone" (experiment)
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
        v = float(vix)
        if v > VIX_INTRAMONTH_THRESHOLD:
            return True, f"VIX={v:.1f}"
        return False, ""

    def check_zscore(date):
        if date not in credit.index:
            return False, ""
        z = credit.loc[date, "ratio_zscore"]
        if pd.isna(z):
            return False, ""
        z = float(z)
        if z < ZSCORE_STRESS:
            return True, f"Z={z:.2f} stress"
        if z > ZSCORE_RECOVERY:
            trailing_z = credit["ratio_zscore"].loc[:date].tail(HYG_MEMORY)
            if (trailing_z < ZSCORE_STRESS).any():
                return True, f"Z={z:.2f} recovery"
        return False, ""

    def check_hyg_mom(date):
        if date not in credit.index:
            return False, ""
        ret = credit.loc[date, "hyg_10d_ret"]
        if pd.isna(ret):
            return False, ""
        ret = float(ret)
        if ret < HYG_MOM_STRESS:
            return True, f"HYG10d={ret:.2%} stress"
        if ret > HYG_MOM_RECOVERY:
            trailing = credit["hyg_10d_ret"].loc[:date].tail(HYG_MEMORY)
            if (trailing < HYG_MOM_STRESS).any():
                return True, f"HYG10d={ret:.2%} recovery"
        return False, ""

    def check_trigger(date):
        if trigger_mode == "vix_only":
            return check_vix(date)
        # credit_standalone: z-score OR HYG momentum (no VIX)
        t, r = check_zscore(date)
        if t:
            return t, r
        return check_hyg_mom(date)

    seg_mask = (returns.index >= start_date) & (returns.index <= end_date)
    seg_dates = returns.index[seg_mask]
    all_dates = returns.index[returns.index <= end_date]

    port_rets = {}
    prev_month = None
    current_w = {a: 1.0 / len(ASSETS) for a in ASSETS}
    cooldown = COOLDOWN_DAYS + 1
    intra_count = 0
    trigger_log = []

    for date in all_dates:
        regime = regime_df.loc[date, "regime"] if date in regime_df.index else np.nan
        if pd.isna(regime):
            cooldown += 1
            continue

        month = date.to_period("M")

        if prev_month is None or month != prev_month:
            current_w = calc_weights(date)
            prev_month = month
            cooldown = COOLDOWN_DAYS + 1
        elif cooldown >= COOLDOWN_DAYS and date.weekday() == 4:
            triggered, reason = check_trigger(date)
            if triggered:
                current_w = calc_weights(date)
                cooldown = 0
                if date >= start_date:
                    intra_count += 1
                    trigger_log.append((date, reason))

        daily_ret = sum(
            returns.loc[date, a] * float(current_w.get(a, 0.0)) for a in ASSETS
        )
        if date in seg_dates:
            port_rets[date] = daily_ret
        cooldown += 1

    return pd.Series(port_rets), intra_count, trigger_log


def compute_metrics(rets):
    if len(rets) < 20:
        return {}
    eq = (1 + rets).cumprod()
    n_years = len(rets) / 252
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = (rets.mean() * 252) / downside if downside > 0 else 0
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": dd,
        "vol": vol,
    }


def main():
    prices, returns, mkt_filters, regime_df, allocations, credit = load_data()

    oos_years = list(range(2017, 2027))

    print(f"\n{'=' * 80}")
    print("WALK-FORWARD VALIDATION: Credit Spread Standalone Trigger (Exp F)")
    print(f"{'=' * 80}")
    print(f"OOS segments: {oos_years[0]} through {oos_years[-1]}")
    print(
        f"Experiment: HYG/LQD z<{ZSCORE_STRESS} + HYG 10d<{HYG_MOM_STRESS:.0%} (replaces VIX)"
    )
    print(f"Baseline:   VIX > {VIX_INTRAMONTH_THRESHOLD}")

    all_baseline_rets = []
    all_experiment_rets = []
    year_results = []

    for year in oos_years:
        oos_start = pd.Timestamp(f"{year}-01-01")
        oos_end = pd.Timestamp(f"{year}-12-31")
        mask = (returns.index >= oos_start) & (returns.index <= oos_end)
        if mask.sum() < 20:
            continue

        print(f"\n  OOS {year}: ", end="")

        baseline_rets, base_intra, _ = run_segment(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            credit,
            oos_start,
            oos_end,
            trigger_mode="vix_only",
        )
        experiment_rets, exp_intra, exp_triggers = run_segment(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            credit,
            oos_start,
            oos_end,
            trigger_mode="credit_standalone",
        )

        bm = compute_metrics(baseline_rets)
        em = compute_metrics(experiment_rets)

        if bm and em:
            ds = em["sharpe"] - bm["sharpe"]
            dc = em["cagr"] - bm["cagr"]
            ddd = em["max_dd"] - bm["max_dd"]
            print(
                f"Sharpe: {bm['sharpe']:>5.2f} -> {em['sharpe']:>5.2f} ({ds:>+.3f})  "
                f"CAGR: {bm['cagr']:>6.2%} -> {em['cagr']:>6.2%} ({dc:>+.2%})  "
                f"MaxDD: {bm['max_dd']:>7.2%} -> {em['max_dd']:>7.2%} ({ddd:>+.2%})  "
                f"Intra: {base_intra}->{exp_intra}"
            )

            year_results.append(
                {
                    "year": year,
                    "base_sharpe": bm["sharpe"],
                    "exp_sharpe": em["sharpe"],
                    "base_cagr": bm["cagr"],
                    "exp_cagr": em["cagr"],
                    "base_dd": bm["max_dd"],
                    "exp_dd": em["max_dd"],
                    "base_intra": base_intra,
                    "exp_intra": exp_intra,
                    "d_sharpe": ds,
                    "d_cagr": dc,
                    "d_dd": ddd,
                    "triggers": exp_triggers,
                }
            )

        all_baseline_rets.append(baseline_rets)
        all_experiment_rets.append(experiment_rets)

    # ── Aggregate OOS metrics ---------------------------------------------------
    agg_baseline = pd.concat(all_baseline_rets)
    agg_experiment = pd.concat(all_experiment_rets)

    bm_agg = compute_metrics(agg_baseline)
    em_agg = compute_metrics(agg_experiment)

    print(f"\n{'=' * 80}")
    print("AGGREGATE OOS RESULTS (all years combined)")
    print(f"{'=' * 80}")

    hdr = f"{'':>14} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} {'MaxDD':>9} {'Vol':>8}"
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'Baseline':>14} {bm_agg['cagr']:>7.2%} {bm_agg['sharpe']:>8.2f} {bm_agg['sortino']:>9.2f}"
        f" {bm_agg['max_dd']:>8.2%} {bm_agg['vol']:>7.2%}"
    )
    print(
        f"{'Credit Only':>14} {em_agg['cagr']:>7.2%} {em_agg['sharpe']:>8.2f} {em_agg['sortino']:>9.2f}"
        f" {em_agg['max_dd']:>8.2%} {em_agg['vol']:>7.2%}"
    )

    ds_agg = em_agg["sharpe"] - bm_agg["sharpe"]
    dc_agg = em_agg["cagr"] - bm_agg["cagr"]
    ddd_agg = em_agg["max_dd"] - bm_agg["max_dd"]
    dso_agg = em_agg["sortino"] - bm_agg["sortino"]

    print(
        f"\n{'Delta':>14} {dc_agg:>+7.2%} {ds_agg:>+8.3f} {dso_agg:>+9.3f}"
        f" {ddd_agg:>+8.2%} {(em_agg['vol'] - bm_agg['vol']):>+7.2%}"
    )

    # ── Per-year summary --------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("PER-YEAR OOS DELTAS")
    print(f"{'=' * 80}")

    hdr2 = f"{'Year':>6} {'dSharpe':>9} {'dCAGR':>9} {'dMaxDD':>9} {'Triggers':>9} {'Win?':>6}"
    print(hdr2)
    print("-" * len(hdr2))

    wins = 0
    losses = 0
    for r in year_results:
        is_win = r["d_sharpe"] > 0
        wins += is_win
        losses += not is_win
        print(
            f"{r['year']:>6} {r['d_sharpe']:>+9.3f} {r['d_cagr']:>+8.2%} {r['d_dd']:>+8.2%}"
            f" {r['exp_intra']:>9d} {'WIN' if is_win else 'LOSS':>6}"
        )

    total = wins + losses
    print(f"\nWin rate: {wins}/{total} = {wins / total:.0%}" if total > 0 else "")

    # ── Trigger analysis --------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("TRIGGER ANALYSIS (experiment triggers in OOS periods)")
    print(f"{'=' * 80}")
    total_triggers = sum(r["exp_intra"] for r in year_results)
    print(f"  Total OOS triggers: {total_triggers}")

    stress_count = 0
    recovery_count = 0
    for r in year_results:
        for _, reason in r.get("triggers", []):
            if "stress" in reason:
                stress_count += 1
            elif "recovery" in reason:
                recovery_count += 1

    print(f"  Stress triggers:    {stress_count}")
    print(f"  Recovery triggers:  {recovery_count}")
    if total_triggers > 0:
        print(f"  Avg triggers/year:  {total_triggers / total:.1f}")

    # Show recent trigger events
    recent_triggers = []
    for r in year_results[-3:]:
        for dt, reason in r.get("triggers", []):
            recent_triggers.append((dt, reason))
    if recent_triggers:
        print(f"\n  Recent trigger events ({len(recent_triggers)}):")
        for dt, reason in recent_triggers[-10:]:
            print(f"    {dt.date()} -- {reason}")

    # ── Consistency checks -------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("CONSISTENCY CHECKS")
    print(f"{'=' * 80}")

    worst_delta = min(r["d_sharpe"] for r in year_results) if year_results else 0
    best_delta = max(r["d_sharpe"] for r in year_results) if year_results else 0
    avg_delta = np.mean([r["d_sharpe"] for r in year_results]) if year_results else 0

    print(f"  Avg yearly Sharpe delta : {avg_delta:+.3f}")
    print(f"  Best year delta         : {best_delta:+.3f}")
    print(f"  Worst year delta        : {worst_delta:+.3f}")

    catastrophic = any(r["d_sharpe"] < -0.10 for r in year_results)
    print(
        f"  Catastrophic loss (>0.10 Sharpe): {'YES -- CONCERN' if catastrophic else 'NO'}"
    )

    # Monotonicity: does adding years hurt?
    if len(year_results) >= 3:
        first_half = year_results[: len(year_results) // 2]
        second_half = year_results[len(year_results) // 2 :]
        first_avg = np.mean([r["d_sharpe"] for r in first_half])
        second_avg = np.mean([r["d_sharpe"] for r in second_half])
        print(f"  First-half avg delta    : {first_avg:+.3f}")
        print(f"  Second-half avg delta   : {second_avg:+.3f}")
        if second_avg < first_avg - 0.05:
            print("  >> WARNING: Signal may be decaying over time")
        else:
            print("  >> Signal stable or improving over time")

    # ── Final decision -----------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("FINAL WALK-FORWARD DECISION")
    print(f"{'=' * 80}")

    passed_sharpe = ds_agg >= 0.02
    passed_cagr = dc_agg >= 0.0025
    passed_consistency = wins / total >= 0.5 if total > 0 else False
    no_catastrophe = not catastrophic
    passed_dd = ddd_agg >= -0.01

    print(
        f"  Aggregate Sharpe delta  : {ds_agg:+.3f}  {'PASS' if passed_sharpe else 'FAIL'} (>= +0.02)"
    )
    print(
        f"  Aggregate CAGR delta    : {dc_agg:+.3%}  {'PASS' if passed_cagr else 'FAIL'} (>= +0.25%)"
    )
    print(
        f"  Win rate                : {wins}/{total}    {'PASS' if passed_consistency else 'FAIL'} (>= 50%)"
    )
    print(f"  No catastrophic loss    :         {'PASS' if no_catastrophe else 'FAIL'}")
    print(
        f"  MaxDD not worse         : {ddd_agg:+.2%}  {'PASS' if passed_dd else 'FAIL'} (>= -1%)"
    )

    criteria_passed = sum(
        [passed_sharpe, passed_cagr, passed_consistency, no_catastrophe, passed_dd]
    )
    all_pass = criteria_passed == 5
    most_pass = criteria_passed >= 4

    if all_pass:
        print(
            f"\n  >>> ACCEPT ({criteria_passed}/5) -- implement credit standalone as primary trigger."
        )
    elif most_pass:
        print(
            f"\n  >>> CONDITIONAL ACCEPT ({criteria_passed}/5) -- implement but monitor for degradation."
        )
    else:
        print(f"\n  >>> REJECT ({criteria_passed}/5) -- keep VIX as primary trigger.")


if __name__ == "__main__":
    main()
