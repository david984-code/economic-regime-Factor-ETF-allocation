"""
EXPERIMENT D -- Credit Stress Intramonth Trigger (HYG-based)
=============================================================

Hypothesis
----------
Using high-yield credit stress (HYG ETF drawdown / recovery) as an
intramonth rebalance trigger will be more effective than VIX because
credit deterioration leads equity selloffs and captures a different
dimension of risk. Bidirectional: defensive when HYG falls fast,
risk-on recovery when HYG stabilises after stress.

Variable changed
----------------
Intramonth trigger: replace VIX-only with HYG-based credit stress.
  Defensive: HYG trailing N-day return < -threshold
  Recovery:  HYG trailing N-day return > 0 after recent stress episode

All other parameters (allocation, vol-scaling, MA filter) unchanged.
Also tested: combined VIX + credit trigger.
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

VIX_INTRAMONTH_THRESHOLD = 27.0

RISK_ON_REGIMES = {"Recovery", "Overheating"}
RISK_OFF_REGIMES = {"Contraction", "Stagflation"}
REGIME_ALIASES = {"Expansion": "Overheating", "Slowdown": "Contraction"}


# ── Data loading ---------------------------------------------------------------


def load_data():
    all_tickers = TICKERS + ["HYG"]
    print("Downloading price data (incl HYG) ...")
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = (
            raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
        )
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    prices = prices.dropna()

    hyg_prices = prices["HYG"].copy()
    asset_prices = prices[TICKERS].copy()

    returns = asset_prices.pct_change().dropna()
    returns["cash"] = CASH_DAILY_YIELD

    print("Downloading VIX + SPY for market filters ...")
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

    # HYG stats
    hyg_prices.pct_change().dropna()
    print(
        f"\n  HYG data: {len(hyg_prices)} days, {hyg_prices.index.min().date()} to {hyg_prices.index.max().date()}"
    )
    hyg_10d = hyg_prices.pct_change(10)
    print(f"  HYG 10d return: mean={hyg_10d.mean():.3%}, std={hyg_10d.std():.3%}")
    print(f"  HYG 10d < -2%: {(hyg_10d < -0.02).sum()} days")
    print(f"  HYG 10d < -3%: {(hyg_10d < -0.03).sum()} days")
    print(f"  HYG 10d < -5%: {(hyg_10d < -0.05).sum()} days")

    return asset_prices, returns, mkt_filters, regime_df, allocations, hyg_prices


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


# ── Backtest engine ------------------------------------------------------------


def run_backtest(
    prices,
    returns,
    mkt_filters,
    regime_df,
    allocations,
    hyg_prices,
    *,
    trigger_mode: str = "vix_only",
    hyg_lookback: int = 10,
    hyg_stress_threshold: float = -0.02,
    hyg_recovery_threshold: float = 0.0,
    hyg_stress_memory_days: int = 20,
    vix_hi: float = 27.0,
    cooldown_days: int = 5,
    friday_only: bool = True,
) -> dict:
    """Run backtest with configurable intramonth trigger.

    trigger_mode:
      "none"       : no intramonth triggers
      "vix_only"   : production VIX > vix_hi (baseline)
      "credit_only": HYG-based credit stress/recovery
      "vix_credit" : either VIX or credit triggers fire
    """
    W_ON = _avg_alloc(RISK_ON_REGIMES, allocations)
    W_OFF = _avg_alloc(RISK_OFF_REGIMES, allocations)

    hyg_trailing_ret = hyg_prices.pct_change(hyg_lookback)

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
        return w

    def check_vix_trigger(date):
        if date not in mkt_filters.index:
            return False, ""
        vix = mkt_filters.loc[date, "VIX_close"]
        if pd.isna(vix):
            return False, ""
        if float(vix) > vix_hi:
            return True, f"VIX={float(vix):.1f}>{vix_hi}"
        return False, ""

    def was_hyg_recently_stressed(date):
        """Check if HYG had a stress episode within trailing window."""
        if date not in hyg_trailing_ret.index:
            return False
        trailing = hyg_trailing_ret.loc[:date].tail(hyg_stress_memory_days)
        return (trailing < hyg_stress_threshold).any()

    def check_credit_trigger(date):
        if date not in hyg_trailing_ret.index:
            return False, ""
        hyg_ret = hyg_trailing_ret.loc[date]
        if pd.isna(hyg_ret):
            return False, ""
        hyg_ret = float(hyg_ret)

        # Defensive: HYG falling fast
        if hyg_ret < hyg_stress_threshold:
            return (
                True,
                f"HYG_{hyg_lookback}d={hyg_ret:.2%}<{hyg_stress_threshold:.0%} (stress)",
            )

        # Recovery: HYG positive after recent stress (state transition)
        if hyg_ret > hyg_recovery_threshold and was_hyg_recently_stressed(date):
            return True, f"HYG_{hyg_lookback}d={hyg_ret:.2%} recovery after stress"

        return False, ""

    def check_trigger(date):
        if trigger_mode == "none":
            return False, ""
        if trigger_mode == "vix_only":
            return check_vix_trigger(date)
        if trigger_mode == "credit_only":
            return check_credit_trigger(date)
        if trigger_mode == "vix_credit":
            v_trig, v_reason = check_vix_trigger(date)
            c_trig, c_reason = check_credit_trigger(date)
            if v_trig and c_trig:
                return True, f"BOTH: {v_reason} + {c_reason}"
            if v_trig:
                return True, v_reason
            if c_trig:
                return True, c_reason
            return False, ""
        return False, ""

    # ── Rebalance loop ---------------------------------------------------------
    port_rets = []
    prev_month = None
    current_w = {a: 1.0 / len(ASSETS) for a in ASSETS}
    prev_w = None
    total_turnover = 0.0
    rebalance_count = 0
    intramonth_count = 0
    intramonth_log = []
    days_since_last_intra = cooldown_days + 1

    for date in returns.index:
        regime = regime_df.loc[date, "regime"] if date in regime_df.index else np.nan
        if pd.isna(regime):
            port_rets.append(np.nan)
            days_since_last_intra += 1
            continue

        month = date.to_period("M")
        did_rebalance = False

        if prev_month is None or month != prev_month:
            current_w = calc_weights(date)
            prev_month = month
            did_rebalance = True
            rebalance_count += 1
            days_since_last_intra = cooldown_days + 1

        elif days_since_last_intra >= cooldown_days:
            if not friday_only or date.weekday() == 4:
                triggered, reason = check_trigger(date)
                if triggered:
                    current_w = calc_weights(date)
                    did_rebalance = True
                    intramonth_count += 1
                    intramonth_log.append((date, reason))
                    days_since_last_intra = 0

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
        days_since_last_intra += 1

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
    prices, returns, mkt_filters, regime_df, allocations, hyg_prices = load_data()

    print("\n" + "=" * 76)
    print("EXPERIMENT D: Credit Stress Intramonth Trigger (HYG)")
    print("=" * 76)

    # ── Baseline: VIX only (production) ----------------------------------------
    print("\n> Running BASELINE (VIX > 27 only) ...")
    baseline = run_backtest(
        prices,
        returns,
        mkt_filters,
        regime_df,
        allocations,
        hyg_prices,
        trigger_mode="vix_only",
        vix_hi=27.0,
        cooldown_days=5,
    )

    # ── No trigger at all (pure monthly) ---------------------------------------
    print("> Running NO_TRIGGER (monthly only) ...")
    no_trigger = run_backtest(
        prices,
        returns,
        mkt_filters,
        regime_df,
        allocations,
        hyg_prices,
        trigger_mode="none",
    )

    # ── Credit-only triggers ---------------------------------------------------
    print("\n--- Credit-Only Triggers ---")
    credit_configs = [
        {"lb": 10, "stress": -0.02, "recov": 0.00, "mem": 20, "cd": 5},
        {"lb": 10, "stress": -0.03, "recov": 0.00, "mem": 20, "cd": 5},
        {"lb": 10, "stress": -0.02, "recov": 0.005, "mem": 30, "cd": 5},
        {"lb": 5, "stress": -0.015, "recov": 0.00, "mem": 15, "cd": 5},
        {"lb": 5, "stress": -0.02, "recov": 0.00, "mem": 20, "cd": 5},
        {"lb": 20, "stress": -0.03, "recov": 0.00, "mem": 30, "cd": 5},
        {"lb": 20, "stress": -0.05, "recov": 0.00, "mem": 30, "cd": 5},
    ]
    credit_experiments = {}
    for cfg in credit_configs:
        label = f"HYG_{cfg['lb']}d_{abs(cfg['stress']) * 100:.0f}pct"
        if cfg["recov"] > 0:
            label += f"_r{cfg['recov'] * 100:.0f}"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            hyg_prices,
            trigger_mode="credit_only",
            hyg_lookback=cfg["lb"],
            hyg_stress_threshold=cfg["stress"],
            hyg_recovery_threshold=cfg["recov"],
            hyg_stress_memory_days=cfg["mem"],
            cooldown_days=cfg["cd"],
        )
        credit_experiments[label] = result

    # ── Combined VIX + Credit --------------------------------------------------
    print("\n--- Combined VIX + Credit ---")
    combo_configs = [
        {"lb": 10, "stress": -0.02, "recov": 0.00, "mem": 20},
        {"lb": 10, "stress": -0.03, "recov": 0.00, "mem": 20},
        {"lb": 5, "stress": -0.02, "recov": 0.00, "mem": 20},
    ]
    combo_experiments = {}
    for cfg in combo_configs:
        label = f"VIX+HYG_{cfg['lb']}d_{abs(cfg['stress']) * 100:.0f}pct"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            hyg_prices,
            trigger_mode="vix_credit",
            hyg_lookback=cfg["lb"],
            hyg_stress_threshold=cfg["stress"],
            hyg_recovery_threshold=cfg["recov"],
            hyg_stress_memory_days=cfg["mem"],
            vix_hi=27.0,
            cooldown_days=5,
        )
        combo_experiments[label] = result

    all_experiments = {
        "NO_TRIGGER": no_trigger,
        **credit_experiments,
        **combo_experiments,
    }

    # ── Results table -----------------------------------------------------------
    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)

    hdr = f"{'Config':<26} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Vol':>7} {'Intra':>6}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, m):
        print(
            f"{label:<26} {m['cagr']:>6.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
            f" {m['max_dd']:>7.2%} {m['vol']:>6.2%} {m['intramonth']:>5d}"
        )

    row("BASELINE (VIX>27)", baseline)
    for label, m in all_experiments.items():
        row(label, m)

    # ── Deltas -------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("DELTAS vs BASELINE (VIX>27)")
    print("=" * 76)

    hdr2 = f"{'Config':<26} {'dCAGR':>7} {'dSharpe':>8} {'dSortino':>9} {'dMaxDD':>8}"
    print(hdr2)
    print("-" * len(hdr2))

    for label, m in all_experiments.items():
        dc = m["cagr"] - baseline["cagr"]
        ds = m["sharpe"] - baseline["sharpe"]
        dso = m["sortino"] - baseline["sortino"]
        dd = m["max_dd"] - baseline["max_dd"]
        print(f"{label:<26} {dc:>+6.2%} {ds:>+8.3f} {dso:>+9.3f} {dd:>+7.2%}")

    # ── Subperiod Sharpe --------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUBPERIOD SHARPE RATIOS")
    print("=" * 76)

    sub_names = list(baseline["sub_sharpes"].keys())
    sub_hdr = f"{'Config':<26} " + " ".join(f"{n:>10}" for n in sub_names)
    print(sub_hdr)
    print("-" * len(sub_hdr))

    def sub_row(label, m):
        vals = [
            f"{m['sub_sharpes'].get(n, 0):>10.2f}"
            if m["sub_sharpes"].get(n) is not None
            else f"{'N/A':>10}"
            for n in sub_names
        ]
        print(f"{label:<26} " + " ".join(vals))

    sub_row("BASELINE (VIX>27)", baseline)
    for label, m in all_experiments.items():
        sub_row(label, m)

    # ── Bear periods ------------------------------------------------------------
    print("\n" + "=" * 76)
    print("BEAR / STRESS PERIOD PERFORMANCE")
    print("=" * 76)

    for pname in ["COVID crash", "2022 bear", "2018 Q4", "2016 oil scare"]:
        bp = baseline["bear_periods"].get(pname)
        if not bp:
            continue
        print(f"\n  {pname}:")
        print(
            f"    {'BASELINE (VIX>27)':<26}: return={bp['return']:>+7.2%}  max_dd={bp['max_dd']:>+7.2%}"
        )
        for label, m in all_experiments.items():
            ep = m["bear_periods"].get(pname)
            if ep:
                d_dd = ep["max_dd"] - bp["max_dd"]
                flag = " ***" if d_dd > 0.005 else (" !!!" if d_dd < -0.005 else "")
                print(
                    f"    {label:<26}: return={ep['return']:>+7.2%}  max_dd={ep['max_dd']:>+7.2%}"
                    f"  (dd: {d_dd:>+.2%}){flag}"
                )

    # ── Trigger log for best credit experiment ---------------------------------
    best_credit_label = max(
        credit_experiments, key=lambda k: credit_experiments[k]["sharpe"]
    )
    best_credit = credit_experiments[best_credit_label]

    best_combo_label = max(
        combo_experiments, key=lambda k: combo_experiments[k]["sharpe"]
    )
    best_combo = combo_experiments[best_combo_label]

    for tag, label, exp in [
        ("CREDIT", best_credit_label, best_credit),
        ("COMBO", best_combo_label, best_combo),
    ]:
        print(f"\n{'=' * 76}")
        print(f"BEST {tag}: {label}")
        print(f"{'=' * 76}")
        print(f"  Intramonth triggers: {exp['intramonth']}")
        if exp["intramonth_log"]:
            print("  Trigger log (last 25):")
            for date, reason in exp["intramonth_log"][-25:]:
                print(f"    {date.date()} -- {reason}")

    # ── Escalation decision -----------------------------------------------------
    best_label = max(
        all_experiments,
        key=lambda k: all_experiments[k]["sharpe"] if k != "NO_TRIGGER" else -999,
    )
    best = all_experiments[best_label]
    ds = best["sharpe"] - baseline["sharpe"]
    dc = best["cagr"] - baseline["cagr"]

    has_bear = False
    for pn in ["COVID crash", "2022 bear", "2018 Q4"]:
        bpp = baseline["bear_periods"].get(pn, {})
        epp = best["bear_periods"].get(pn, {})
        if epp and bpp and epp.get("max_dd", -1) > bpp.get("max_dd", -1) + 0.005:
            has_bear = True

    print(f"\n{'=' * 76}")
    print(f"ESCALATION DECISION (best: {best_label})")
    print(f"{'=' * 76}")
    print(f"  Sharpe delta     : {ds:+.3f}  (threshold: +0.020)")
    print(f"  CAGR delta       : {dc:+.3%}  (threshold: +0.25%)")
    print(f"  Bear improvement : {'YES' if has_bear else 'NO'}")

    if ds >= 0.02 or dc >= 0.0025 or has_bear:
        print("\n  >> PASS -- escalate to full walk-forward validation")
    else:
        print("\n  >> REJECT -- below escalation thresholds")


if __name__ == "__main__":
    main()
