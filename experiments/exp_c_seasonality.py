"""
EXPERIMENT C -- Seasonality & Election Cycle Overlays
======================================================

C1: Monthly Seasonal Tilt (Halloween Indicator)
------------------------------------------------
Hypothesis: Tilting risk_on up in Nov-Apr and down in May-Oct (the well-
documented "Sell in May" effect) will improve risk-adjusted returns.

Variable: seasonal tilt factor applied to risk_on alpha before blending.
  Nov-Apr: risk_on * (1 + tilt)
  May-Oct: risk_on * (1 - tilt)

C2: Presidential Election Cycle
--------------------------------
Hypothesis: Tilting risk_on based on the presidential cycle year will
capture the well-documented year-3 (pre-election) rally and year-2
(midterm) weakness.

Variable: cycle tilt applied to risk_on alpha.
  Year 3 (pre-election): risk_on * (1 + tilt)
  Year 2 (midterm):      risk_on * (1 - tilt)
  Years 1 and 4:         unchanged

C3: Per-Month Historical Tilt (walk-forward)
----------------------------------------------
Hypothesis: Using trailing 10-year average monthly returns to tilt risk_on
by calendar month captures persistent seasonal patterns.

Variable: trailing month z-score scaled tilt on risk_on.
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

# Presidential election years (used to compute cycle year)
ELECTION_YEARS = [2008, 2012, 2016, 2020, 2024, 2028]


def election_cycle_year(year: int) -> int:
    """Return 1-4 where 1=post-election, 2=midterm, 3=pre-election, 4=election."""
    for ey in ELECTION_YEARS:
        diff = year - ey
        if 0 <= diff <= 3:
            return diff + 1
        if diff < 0:
            return (year - (ey - 4)) + 1
    return ((year - 2008) % 4) + 1


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


def check_intramonth(mkt_filters, date, vix_thresh, spy_drop):
    if date not in mkt_filters.index:
        return False
    vix = mkt_filters.loc[date, "VIX_close"]
    if pd.isna(vix):
        return False
    return float(vix) > vix_thresh


# ── Seasonal alpha adjusters ---------------------------------------------------


def halloween_adjuster(date: pd.Timestamp, tilt: float) -> float:
    """Return a multiplier for risk_on based on Halloween indicator.
    Nov-Apr: bullish -> *(1 + tilt), May-Oct: bearish -> *(1 - tilt).
    """
    month = date.month
    if month >= 11 or month <= 4:
        return 1.0 + tilt
    return 1.0 - tilt


def election_cycle_adjuster(date: pd.Timestamp, tilt: float) -> float:
    """Return a multiplier for risk_on based on the presidential cycle.
    Year 3 (pre-election): bullish, Year 2 (midterm): bearish.
    """
    cy = election_cycle_year(date.year)
    if cy == 3:
        return 1.0 + tilt
    if cy == 2:
        return 1.0 - tilt
    return 1.0


def walk_forward_month_adjuster(
    date: pd.Timestamp,
    port_rets: pd.Series,
    tilt: float,
    trailing_years: int = 10,
) -> float:
    """Return a multiplier based on trailing average return for this calendar month.
    Uses only data available at `date` (walk-forward safe).
    """
    cutoff = date - pd.DateOffset(years=trailing_years)
    trailing = port_rets.loc[cutoff:date]
    if len(trailing) < 252:
        return 1.0

    month_num = date.month
    month_mask = trailing.index.month == month_num
    month_rets = trailing[month_mask]
    if len(month_rets) < 20:
        return 1.0

    all_mean = trailing.mean()
    month_mean = month_rets.mean()
    all_std = trailing.std()

    if all_std < 1e-10:
        return 1.0

    z = (month_mean - all_mean) / all_std
    z = np.clip(z, -2.0, 2.0)
    return 1.0 + tilt * z


# ── Backtest engine ------------------------------------------------------------


def run_backtest(
    prices,
    returns,
    mkt_filters,
    regime_df,
    allocations,
    *,
    seasonal_mode: str = "none",
    seasonal_tilt: float = 0.0,
    wf_trailing_years: int = 10,
    portfolio_returns_for_wf: pd.Series | None = None,
) -> dict:
    """Run the backtest with an optional seasonal alpha adjuster.

    seasonal_mode: "none", "halloween", "election", "walk_forward", "combined"
    """
    W_ON = _avg_alloc(RISK_ON_REGIMES, allocations)
    W_OFF = _avg_alloc(RISK_OFF_REGIMES, allocations)

    def calc_weights(date, alpha_adj=1.0):
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
            alpha = float(np.clip(alpha * alpha_adj, 0.0, 1.0))
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

    # ── Rebalance loop ---------------------------------------------------------
    port_rets_list = []
    prev_month = None
    current_w = {a: 1.0 / len(ASSETS) for a in ASSETS}
    prev_w = None
    total_turnover = 0.0
    rebalance_count = 0
    intramonth_count = 0

    # For walk-forward seasonal: build returns incrementally
    wf_rets = (
        portfolio_returns_for_wf
        if portfolio_returns_for_wf is not None
        else pd.Series(dtype=float)
    )

    for date in returns.index:
        regime = regime_df.loc[date, "regime"] if date in regime_df.index else np.nan
        if pd.isna(regime):
            port_rets_list.append(np.nan)
            continue

        month = date.to_period("M")
        did_rebalance = False

        if prev_month is None or month != prev_month:
            # Compute seasonal adjustment
            adj = 1.0
            if seasonal_mode == "halloween":
                adj = halloween_adjuster(date, seasonal_tilt)
            elif seasonal_mode == "election":
                adj = election_cycle_adjuster(date, seasonal_tilt)
            elif seasonal_mode == "walk_forward" and len(wf_rets) > 252:
                adj = walk_forward_month_adjuster(
                    date, wf_rets, seasonal_tilt, wf_trailing_years
                )
            elif seasonal_mode == "combined":
                h = halloween_adjuster(date, seasonal_tilt * 0.5)
                e = election_cycle_adjuster(date, seasonal_tilt * 0.5)
                adj = h * e

            current_w = calc_weights(date, alpha_adj=adj)
            prev_month = month
            did_rebalance = True
            rebalance_count += 1

        elif ENABLE_INTRAMONTH_TRIGGER and date.weekday() == 4:
            if check_intramonth(
                mkt_filters, date, VIX_INTRAMONTH_THRESHOLD, SPY_WEEKLY_DROP_THRESHOLD
            ):
                current_w = calc_weights(date, alpha_adj=1.0)
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
        port_rets_list.append(daily_ret)

        # Update walk-forward series
        if seasonal_mode == "walk_forward":
            wf_rets = pd.concat([wf_rets, pd.Series([daily_ret], index=[date])])

    port_rets = pd.Series(port_rets_list, index=returns.index).dropna()

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

    # Monthly return analysis
    monthly_rets = port_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    month_avg = monthly_rets.groupby(monthly_rets.index.month).mean()

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
        "month_avg_returns": month_avg,
        "equity_curve": eq,
    }


# ── Main -----------------------------------------------------------------------


def main():
    prices, returns, mkt_filters, regime_df, allocations = load_data()

    print("\n" + "=" * 72)
    print("EXPERIMENT C: Seasonality & Election Cycle Overlays")
    print("=" * 72)

    # ── Baseline ---------------------------------------------------------------
    print("\n> Running BASELINE ...")
    baseline = run_backtest(
        prices,
        returns,
        mkt_filters,
        regime_df,
        allocations,
        seasonal_mode="none",
    )

    # Print baseline monthly pattern
    print("\n  Baseline average monthly returns:")
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for m, name in enumerate(month_names, 1):
        avg = baseline["month_avg_returns"].get(m, 0)
        print(f"    {name}: {avg:>+.3%}")

    # ── C1: Halloween indicator ------------------------------------------------
    print("\n--- C1: Halloween Indicator (Nov-Apr bullish / May-Oct bearish) ---")
    c1_configs = [0.05, 0.10, 0.15, 0.20, 0.25]
    c1_experiments = {}
    for tilt in c1_configs:
        label = f"halloween_{tilt:.2f}"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            seasonal_mode="halloween",
            seasonal_tilt=tilt,
        )
        c1_experiments[label] = result

    # ── C2: Election cycle -----------------------------------------------------
    print("\n--- C2: Presidential Election Cycle ---")
    c2_configs = [0.05, 0.10, 0.15, 0.20]
    c2_experiments = {}
    for tilt in c2_configs:
        label = f"election_{tilt:.2f}"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            seasonal_mode="election",
            seasonal_tilt=tilt,
        )
        c2_experiments[label] = result

    # ── C3: Walk-forward month tilt --------------------------------------------
    print("\n--- C3: Walk-Forward Monthly Tilt ---")
    c3_configs = [0.05, 0.10, 0.15, 0.20]
    c3_experiments = {}
    for tilt in c3_configs:
        label = f"wf_month_{tilt:.2f}"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            seasonal_mode="walk_forward",
            seasonal_tilt=tilt,
        )
        c3_experiments[label] = result

    # ── C4: Combined (Halloween + Election) ------------------------------------
    print("\n--- C4: Combined Halloween + Election ---")
    c4_configs = [0.10, 0.15, 0.20]
    c4_experiments = {}
    for tilt in c4_configs:
        label = f"combined_{tilt:.2f}"
        print(f"> Running {label} ...")
        result = run_backtest(
            prices,
            returns,
            mkt_filters,
            regime_df,
            allocations,
            seasonal_mode="combined",
            seasonal_tilt=tilt,
        )
        c4_experiments[label] = result

    all_experiments = {
        **c1_experiments,
        **c2_experiments,
        **c3_experiments,
        **c4_experiments,
    }

    # ── Results table -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    hdr = f"{'Config':<20} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Vol':>7}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, m):
        print(
            f"{label:<20} {m['cagr']:>6.2%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f}"
            f" {m['max_dd']:>7.2%} {m['vol']:>6.2%}"
        )

    row("BASELINE", baseline)
    print()
    for label, m in c1_experiments.items():
        row(label, m)
    print()
    for label, m in c2_experiments.items():
        row(label, m)
    print()
    for label, m in c3_experiments.items():
        row(label, m)
    print()
    for label, m in c4_experiments.items():
        row(label, m)

    # ── Deltas -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DELTAS vs BASELINE")
    print("=" * 72)

    hdr2 = f"{'Config':<20} {'dCAGR':>7} {'dSharpe':>8} {'dMaxDD':>8} {'dVol':>7}"
    print(hdr2)
    print("-" * len(hdr2))

    for label, m in all_experiments.items():
        dc = m["cagr"] - baseline["cagr"]
        ds = m["sharpe"] - baseline["sharpe"]
        dd = m["max_dd"] - baseline["max_dd"]
        dv = m["vol"] - baseline["vol"]
        print(f"{label:<20} {dc:>+6.2%} {ds:>+8.3f} {dd:>+7.2%} {dv:>+6.2%}")

    # ── Subperiod Sharpe --------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUBPERIOD SHARPE RATIOS")
    print("=" * 72)

    sub_names = list(baseline["sub_sharpes"].keys())
    sub_hdr = f"{'Config':<20} " + " ".join(f"{n:>10}" for n in sub_names)
    print(sub_hdr)
    print("-" * len(sub_hdr))

    def sub_row(label, m):
        vals = []
        for n in sub_names:
            v = m["sub_sharpes"].get(n)
            vals.append(f"{v:>10.2f}" if v is not None else f"{'N/A':>10}")
        print(f"{label:<20} " + " ".join(vals))

    sub_row("BASELINE", baseline)
    for label, m in all_experiments.items():
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
                f"    {'BASELINE':<20}: return={bp['return']:>+7.2%}  max_dd={bp['max_dd']:>+7.2%}"
            )
        for label, m in all_experiments.items():
            ep = m["bear_periods"].get(pname)
            if ep and bp:
                d_dd = ep["max_dd"] - bp["max_dd"]
                flag = " ***" if abs(d_dd) > 0.005 else ""
                print(
                    f"    {label:<20}: return={ep['return']:>+7.2%}  max_dd={ep['max_dd']:>+7.2%}"
                    f"  (dd: {d_dd:>+.2%}){flag}"
                )

    # ── Best per category -------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("BEST PER CATEGORY")
    print(f"{'=' * 72}")

    for cat_name, cat_exps in [
        ("Halloween", c1_experiments),
        ("Election", c2_experiments),
        ("Walk-Forward", c3_experiments),
        ("Combined", c4_experiments),
    ]:
        if not cat_exps:
            continue
        best_l = max(cat_exps, key=lambda k: cat_exps[k]["sharpe"])
        best_m = cat_exps[best_l]
        ds = best_m["sharpe"] - baseline["sharpe"]
        dc = best_m["cagr"] - baseline["cagr"]
        print(f"\n  {cat_name}: {best_l}")
        print(f"    Sharpe delta: {ds:+.3f}    CAGR delta: {dc:+.3%}")

        has_bear = False
        for pn in ["COVID crash", "2022 bear"]:
            bpp = baseline["bear_periods"].get(pn, {})
            epp = best_m["bear_periods"].get(pn, {})
            if epp and bpp and epp.get("max_dd", -1) > bpp.get("max_dd", -1) + 0.005:
                has_bear = True

        if ds >= 0.02 or dc >= 0.0025 or has_bear:
            print("    -> PASS (escalate)")
        elif ds < 0.02 and dc < 0.0025 and not has_bear:
            print("    -> REJECT (noise)")
        else:
            print("    -> MARGINAL")

    # ── Election year detail ----------------------------------------------------
    print(f"\n{'=' * 72}")
    print("ELECTION CYCLE DETAIL (baseline returns by cycle year)")
    print(f"{'=' * 72}")

    (
        1
        + pd.Series(
            [(1 + r) for r in baseline["equity_curve"].pct_change().dropna()],
            index=baseline["equity_curve"].pct_change().dropna().index,
        )
    ).cumprod()

    baseline_daily = baseline["equity_curve"].pct_change().dropna()
    annual_by_cycle = {}
    for year, grp in baseline_daily.groupby(baseline_daily.index.year):
        cy = election_cycle_year(year)
        ann_ret = (1 + grp).prod() ** (252 / len(grp)) - 1 if len(grp) > 50 else None
        if ann_ret is not None:
            annual_by_cycle.setdefault(cy, []).append((year, ann_ret))

    for cy in sorted(annual_by_cycle.keys()):
        cy_names = {1: "Post-election", 2: "Midterm", 3: "Pre-election", 4: "Election"}
        entries = annual_by_cycle[cy]
        avg = np.mean([r for _, r in entries])
        print(f"\n  Year {cy} ({cy_names.get(cy, '?')}) -- avg annualized: {avg:+.2%}")
        for year, ret in entries:
            print(f"    {year}: {ret:+.2%}")


if __name__ == "__main__":
    main()
