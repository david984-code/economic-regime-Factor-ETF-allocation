"""Strategy performance analysis: risk-adjusted returns and bear market protection.

FOCUS: How well does the strategy protect capital when markets are down?
       -- Downside capture ratio (what % of market losses does the strategy absorb?)
       -- Bear market alpha (excess return vs benchmark during market declines)
       -- Sortino ratio (penalises losses more than Sharpe does)
       -- Calmar ratio (return per unit of max drawdown)
       -- Regime-level performance (how does the strategy behave in each macro regime?)

WHAT WE ARE NOT TRYING TO DO:
  - Consistently beat SPY every year (that would require high beta / concentration)
  - Maximise raw CAGR at the expense of drawdowns

WHAT GOOD LOOKS LIKE FOR THIS STRATEGY:
  - Downside capture < 60% (absorb less than 60 cents of every dollar SPY loses)
  - Upside capture > 60% (still participate in market rallies)
  - Sortino > Sharpe (reward / downside risk ratio better than generic risk)
  - Bear market alpha > 0% annualised (positive excess return in down markets)
  - Max drawdown meaningfully lower than SPY / QQQ in bear markets
  - Calmar > 0.4 (CAGR at least 40% of its max drawdown)

Usage:
  uv run python scripts/performance_analysis.py
  uv run python scripts/performance_analysis.py --since 2020-01-01
  uv run python scripts/performance_analysis.py --bench QQQ
  uv run python scripts/performance_analysis.py --save-csv outputs/perf_report.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import logging
import math
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.allocation.optimizer import optimize_allocations_from_data
from src.backtest.engine import run_backtest_with_allocations
from src.backtest.metrics import compute_full_metrics
from src.config import OUTPUTS_DIR, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices
from src.evaluation.benchmarks import CASH_DAILY_YIELD

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RF_DAILY = CASH_DAILY_YIELD  # ~4.5% annual / 252

# Frozen production baseline knobs
BASELINE_KW = {
    "use_stagflation_override": False,
    "use_hybrid_signal": True,
    "hybrid_macro_weight": 0.0,
    "use_momentum": True,
    "market_lookback_months": 24,
    "trend_filter_type": "none",
    "vol_scaling_method": "none",
    "portfolio_construction_method": "equal_weight",
    "momentum_12m_weight": 0.0,
    "quarterly_rebalance": False,
    "tolerance": 0.015,
    "sigmoid_scale": 0.25,
    "use_post_blend_inv_vol": False,
}

BENCH_MAP = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "IWM": "Russell 2000",
    "AGG": "US Agg Bond",
    "GLD": "Gold",
}

# ---- Data helpers --------------------------------------------------------


def _load_regime_df(prices: pd.DataFrame) -> pd.DataFrame:
    csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if csv_path.exists():
        regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        from src.allocation.optimizer import load_regimes

        regime_df = load_regimes()
    regime_df = regime_df.dropna(subset=["regime"]).sort_index()
    if regime_df.index.duplicated().any():
        regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
    return regime_df.reindex(prices.index).ffill()


def _run_strategy(start: str = "2010-01-01") -> pd.Series:
    end = get_end_date()
    logger.info("Fetching strategy prices %s to %s...", start, end)
    prices = fetch_prices(tickers=TICKERS, start=start, end=end)
    regime_df = _load_regime_df(prices)

    train_returns = prices.resample("ME").last().pct_change().dropna()
    if "cash" not in train_returns.columns:
        train_returns["cash"] = (1.05) ** (1 / 12) - 1
    train_regimes = regime_df.resample("ME").last().dropna(how="all")
    train_regimes = train_regimes.loc[train_regimes.index <= train_returns.index.max()]

    allocations = optimize_allocations_from_data(train_returns, train_regimes)
    result = run_backtest_with_allocations(
        prices, regime_df, allocations, return_weights=False, **BASELINE_KW
    )
    strat_rets = result[0] if isinstance(result, tuple) else result
    return strat_rets.dropna()


def _fetch_benchmarks(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    logger.info("Fetching benchmark prices: %s", tickers)
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, threads=True)
    if raw.empty:
        raise RuntimeError("yfinance returned no benchmark data")
    close = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.pct_change().iloc[1:].dropna(how="all")


def _slice(s: pd.Series, since: str | None) -> pd.Series:
    if since:
        return s.loc[s.index >= since]
    return s


# ---- Bear market period detection ----------------------------------------


def _bear_periods(
    bench: pd.Series, threshold: float = -0.20
) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Identify peak-to-trough periods where bench fell >= threshold."""
    equity = (1 + bench).cumprod()
    peaks: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    peak_val = equity.iloc[0]
    peak_date = equity.index[0]
    in_bear = False
    trough_date = peak_date
    trough_val = peak_val

    for date, val in equity.items():
        if val >= peak_val:
            if in_bear:
                dd = trough_val / peak_val - 1
                if dd <= threshold:
                    peaks.append((peak_date, trough_date, dd))
            peak_val = val
            peak_date = date
            in_bear = False
            trough_date = date
            trough_val = val
        else:
            in_bear = True
            if val < trough_val:
                trough_val = val
                trough_date = date

    if in_bear:
        dd = trough_val / peak_val - 1
        if dd <= threshold:
            peaks.append((peak_date, trough_date, dd))

    return peaks


# ---- Printing helpers ----------------------------------------------------


def _pct(v: float, decimals: int = 2) -> str:
    if math.isnan(v):
        return "  n/a"
    return f"{v * 100:+.{decimals}f}%"


def _num(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "  n/a"
    return f"{v:+.{decimals}f}"


def _pos_pct(v: float, decimals: int = 2) -> str:
    if math.isnan(v):
        return "  n/a"
    return f"{v * 100:.{decimals}f}%"


def _score(v: float, good_above: float | None = None, bad_above: float | None = None) -> str:
    """Append a simple qualitative flag."""
    if math.isnan(v):
        return ""
    if good_above is not None and v >= good_above:
        return " [GOOD]"
    if bad_above is not None and v >= bad_above:
        return " [WARN]"
    return " [POOR]"


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _subsection(title: str) -> None:
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


# ---- Main report ---------------------------------------------------------


def run_analysis(
    since: str | None = None, primary_bench: str = "SPY", save_csv: str | None = None
) -> None:
    _section("STRATEGY PERFORMANCE ANALYSIS")
    print("  Focus: risk-adjusted returns + bear market protection")
    print(f"  Primary benchmark: {primary_bench} ({BENCH_MAP.get(primary_bench, primary_bench)})")
    if since:
        print(f"  Period filter: from {since}")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # ---- Load data -------------------------------------------------------
    strat_raw = _run_strategy(start="2010-01-01")
    all_bench_tickers = list(BENCH_MAP.keys())
    bench_raw = _fetch_benchmarks(all_bench_tickers, start="2010-01-01", end=get_end_date())

    strat = _slice(strat_raw, since)
    primary_b = (
        _slice(bench_raw[primary_bench], since) if primary_bench in bench_raw.columns else None
    )

    common_end = min(strat.index.max(), bench_raw.index.max())
    logger.info(
        "Strategy: %s to %s  (%d days)", strat.index.min().date(), common_end.date(), len(strat)
    )

    # ---- Section 1: Core metrics -----------------------------------------
    _subsection("1. CORE METRICS  (full history)")
    m_strat = compute_full_metrics(strat, rf_daily=RF_DAILY, bench_rets=primary_b)

    rows: list[dict] = [{"Name": f"Strategy (vs {primary_bench})", **m_strat}]
    for ticker, label in BENCH_MAP.items():
        if ticker in bench_raw.columns:
            b_slice = _slice(bench_raw[ticker], since)
            mb = compute_full_metrics(b_slice, rf_daily=RF_DAILY)
            rows.append({"Name": label, **mb})

    print(
        f"\n  {'Name':<30}  {'CAGR':>8}  {'Vol':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'Calmar':>7}  {'MaxDD':>8}  {'Ulcer':>7}"
    )
    print("  " + "-" * 70)
    for row in rows:
        print(
            f"  {row['Name']:<30}"
            f"  {_pct(row.get('CAGR', float('nan'))):>8}"
            f"  {_pos_pct(row.get('Volatility', float('nan'))):>8}"
            f"  {_num(row.get('Sharpe', float('nan'))):>7}"
            f"  {_num(row.get('Sortino', float('nan'))):>8}"
            f"  {_num(row.get('Calmar', float('nan'))):>7}"
            f"  {_pct(row.get('Max Drawdown', float('nan'))):>8}"
            f"  {_num(row.get('Ulcer_Index', float('nan'))):>7}"
        )

    # ---- Section 2: Bear market protection (THE KEY SECTION) -------------
    _subsection(f"2. BEAR MARKET PROTECTION vs {primary_bench}  (core objective)")
    strat_m = m_strat
    dc = strat_m.get("Downside_Cap", float("nan"))
    uc = strat_m.get("Upside_Cap", float("nan"))
    ba = strat_m.get("Bear_Alpha_Ann", float("nan"))
    bhr = strat_m.get("Bear_Hit_Rate", float("nan"))
    bad = strat_m.get("Bear_Avg_Daily_Return", float("nan"))
    bbd = strat_m.get("Bear_Bench_Avg", float("nan"))
    beta = strat_m.get("Beta", float("nan"))
    alpha = strat_m.get("Alpha_Ann", float("nan"))
    corr = strat_m.get("Correlation", float("nan"))
    ir = strat_m.get("Info_Ratio", float("nan"))

    print("\n  WHAT THE STRATEGY CAPTURES OF MARKET MOVES:")
    print(
        f"    Downside capture ratio  : {_pos_pct(dc)}"
        + _score(dc if not math.isnan(dc) else 1, good_above=None, bad_above=None)
        + "   (target < 70%)"
    )
    if not math.isnan(dc):
        quality = (
            "EXCELLENT"
            if dc < 0.50
            else ("GOOD" if dc < 0.70 else ("FAIR" if dc < 0.90 else "POOR"))
        )
        note = (
            f"absorbs less than half of {primary_bench}'s losses"
            if dc < 0.50
            else "absorbs less than 70% of losses"
            if dc < 0.70
            else "absorbs less than 90% of losses"
            if dc < 0.90
            else "nearly fully exposed to market drawdowns"
        )
        print(f"    -> {quality}: strategy {note}")

    print(
        f"\n    Upside capture ratio    : {_pos_pct(uc)}"
        + "   (target > 60% — participate in rallies)"
    )
    if not math.isnan(uc) and not math.isnan(dc):
        ratio = uc / dc if dc != 0 else float("nan")
        print(
            f"    -> Capture ratio (up/down): {_num(ratio, 2)}x  (>1.0 means strategy is asymmetric in your favor)"
        )

    print("\n  ALPHA ON BENCHMARK-DOWN DAYS:")
    print(f"    Bear market alpha (ann.) : {_pct(ba)}   (positive = outperforms when market falls)")
    print(
        f"    Bear market hit rate     : {_pos_pct(bhr)}   (% of down-market days strategy > {primary_bench})"
    )
    if not math.isnan(bad) and not math.isnan(bbd):
        print(f"    Avg strategy return  (down days): {_pct(bad)} per day")
        print(f"    Avg {primary_bench} return   (down days): {_pct(bbd)} per day")

    print("\n  SYSTEMATIC RISK:")
    print(
        f"    Beta vs {primary_bench:<6}: {_num(beta, 3)}   (<1.0 = less market-sensitive; target ~0.5-0.7)"
    )
    print(f"    Alpha (annualised): {_pct(alpha)}   (risk-adjusted excess return vs CAPM)")
    print(f"    Correlation       : {_num(corr, 3)}   (lower = more diversification benefit)")
    print(
        f"    Info ratio        : {_num(ir, 3)}   (skill after controlling for active risk; >0.5 = strong)"
    )

    # ---- Section 3: Historical bear market episodes ----------------------
    if primary_b is not None:
        _subsection(f"3. HISTORICAL BEAR MARKETS  (>= -15% drawdown in {primary_bench})")
        bear_eps = _bear_periods(primary_b, threshold=-0.15)
        if not bear_eps:
            print("  No bear markets >= -15% found in the selected period.")
        else:
            print(
                f"  {'Period':<28}  {primary_bench + ' DD':>10}  {'Strat':>8}  {primary_bench:>8}  {'Alpha':>8}  {'Verdict':>10}"
            )
            print("  " + "-" * 68)
            total_strat_ret = 0.0
            total_bench_ret = 0.0
            for peak_d, trough_d, dd in bear_eps:
                mask = (strat.index >= peak_d) & (strat.index <= trough_d)
                b_mask = (primary_b.index >= peak_d) & (primary_b.index <= trough_d)
                s_ep = strat[mask]
                b_ep = primary_b[b_mask]
                if len(s_ep) < 2:
                    continue
                s_ret = float((1 + s_ep).prod() - 1)
                b_ret = float((1 + b_ep).prod() - 1)
                alpha_ep = s_ret - b_ret
                verdict = (
                    "PROTECT" if alpha_ep > 0.05 else "NEUTRAL" if alpha_ep > -0.02 else "LAGGED"
                )
                total_strat_ret += s_ret
                total_bench_ret += b_ret
                period = f"{peak_d.date()} -> {trough_d.date()}"
                print(
                    f"  {period:<28}  {_pct(dd):>10}  {_pct(s_ret):>8}"
                    f"  {_pct(b_ret):>8}  {_pct(alpha_ep):>8}  {verdict:>10}"
                )
            if bear_eps:
                (total_strat_ret - total_bench_ret) / len(bear_eps)
                sum(
                    1
                    for *_, dd2 in bear_eps
                    if _bear_alpha_for_episode(strat, primary_b, *(_[0] for _ in [bear_eps[0]])) > 0
                )
                print(f"\n  Bear episodes found: {len(bear_eps)}")

    # ---- Section 4: Regime-level breakdown ---------------------------------
    regime_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
    if regime_path.exists():
        _subsection("4. PERFORMANCE BY MACRO REGIME")
        reg_df = pd.read_csv(regime_path, parse_dates=["date"], index_col="date")
        reg_df = reg_df.dropna(subset=["regime"]).sort_index()
        reg_aligned = reg_df.reindex(strat.index).ffill()

        if primary_b is not None:
            b_aligned = primary_b.reindex(strat.index).ffill()

        regimes_found = reg_aligned["regime"].dropna().unique()
        print(
            f"\n  {'Regime':<22}  {'Days':>5}  {'Strat CAGR':>11}  {'Strat Vol':>10}  "
            f"{'Strat Sharpe':>13}  {primary_bench + ' CAGR':>11}  {'Alpha':>8}"
        )
        print("  " + "-" * 80)

        for regime in sorted(regimes_found):
            mask = reg_aligned["regime"] == regime
            s_reg = strat[mask].dropna()
            if len(s_reg) < 20:
                continue
            m_reg = compute_full_metrics(s_reg, rf_daily=RF_DAILY)
            b_reg_cagr: float = float("nan")
            if primary_b is not None:
                b_reg = b_aligned[mask].dropna()
                if len(b_reg) >= 20:
                    mb_reg = compute_full_metrics(b_reg, rf_daily=RF_DAILY)
                    b_reg_cagr = mb_reg.get("CAGR", float("nan"))
            alpha_reg = m_reg["CAGR"] - b_reg_cagr if not math.isnan(b_reg_cagr) else float("nan")
            print(
                f"  {regime:<22}  {len(s_reg):>5}"
                f"  {_pct(m_reg['CAGR']):>11}"
                f"  {_pos_pct(m_reg['Volatility']):>10}"
                f"  {_num(m_reg['Sharpe'], 2):>13}"
                f"  {_pct(b_reg_cagr):>11}"
                f"  {_pct(alpha_reg):>8}"
            )
        print(f"\n  Key: positive Alpha = strategy outperformed {primary_bench} in that regime")
        print("  Stagflation/Recession performance is the most important row for this strategy")
    else:
        logger.warning(
            "regime_labels_expanded.csv not found; skipping regime breakdown. Run pipeline first."
        )

    # ---- Section 5: Tail risk and drawdown profile -----------------------
    _subsection("5. TAIL RISK AND DRAWDOWN PROFILE")
    cvar = strat_m.get("CVaR_5pct", float("nan"))
    streak = strat_m.get("Max_Loss_Streak", float("nan"))
    ulcer = strat_m.get("Ulcer_Index", float("nan"))

    print(
        f"  CVaR 5% (avg worst-5% daily loss) : {_pct(cvar)}  per day"
        f"  (benchmark CVaR: "
        + (
            _pct(
                compute_full_metrics(_slice(bench_raw[primary_bench], since)).get(
                    "CVaR_5pct", float("nan")
                )
            )
            if primary_bench in bench_raw.columns
            else "n/a"
        )
        + ")"
    )
    print(
        f"  Max consecutive loss days          : {int(streak) if not math.isnan(streak) else 'n/a'}"
    )
    print(
        f"  Ulcer index (smoothness score)     : {_num(ulcer, 2)}  (lower = smoother; benchmark: "
        + (
            _num(
                compute_full_metrics(_slice(bench_raw[primary_bench], since)).get(
                    "Ulcer_Index", float("nan")
                ),
                2,
            )
            if primary_bench in bench_raw.columns
            else "n/a"
        )
        + ")"
    )

    # ---- Section 6: Interpretation summary --------------------------------
    _subsection("6. INTERPRETATION: WHAT IS WORKING vs NOT WORKING")

    strat_m.get("CAGR", float("nan"))
    sharpe_s = strat_m.get("Sharpe", float("nan"))
    sortino_s = strat_m.get("Sortino", float("nan"))
    calmar_s = strat_m.get("Calmar", float("nan"))
    strat_m.get("Max Drawdown", float("nan"))

    working: list[str] = []
    not_working: list[str] = []
    investigate: list[str] = []

    # Downside capture
    if not math.isnan(dc):
        if dc < 0.65:
            working.append(
                f"Downside capture {dc * 100:.0f}% vs {primary_bench}: strong bear market protection"
            )
        elif dc < 0.85:
            investigate.append(
                f"Downside capture {dc * 100:.0f}%: partial protection but above 65% target"
            )
        else:
            not_working.append(
                f"Downside capture {dc * 100:.0f}%: strategy absorbs most of market losses (poor protection)"
            )

    # Bear alpha
    if not math.isnan(ba):
        if ba > 0.03:
            working.append(
                f"Bear market alpha +{ba * 100:.1f}% ann: strategy adds positive return when market is down"
            )
        elif ba > 0:
            working.append(
                f"Bear market alpha +{ba * 100:.1f}% ann: slight positive alpha in down markets"
            )
        else:
            not_working.append(
                f"Bear market alpha {ba * 100:.1f}% ann: strategy underperforms {primary_bench} in down markets"
            )

    # Sortino vs Sharpe
    if not math.isnan(sortino_s) and not math.isnan(sharpe_s):
        if sortino_s > sharpe_s:
            working.append(
                f"Sortino ({sortino_s:.2f}) > Sharpe ({sharpe_s:.2f}): downside risk lower than upside, asymmetric returns"
            )
        else:
            investigate.append(
                f"Sortino ({sortino_s:.2f}) < Sharpe ({sharpe_s:.2f}): losses more frequent/severe than gains suggest"
            )

    # Calmar
    if not math.isnan(calmar_s):
        if calmar_s > 0.5:
            working.append(f"Calmar {calmar_s:.2f}: good return per unit of max drawdown")
        elif calmar_s > 0.3:
            investigate.append(
                f"Calmar {calmar_s:.2f}: acceptable but drawdown recovery could be faster"
            )
        else:
            not_working.append(f"Calmar {calmar_s:.2f}: poor; drawdowns large relative to returns")

    # Beta
    if not math.isnan(beta):
        if beta < 0.6:
            working.append(
                f"Beta {beta:.2f}: low systematic market exposure, provides diversification"
            )
        elif beta < 0.8:
            investigate.append(
                f"Beta {beta:.2f}: moderate market exposure; higher than ideal for protection strategy"
            )
        else:
            not_working.append(
                f"Beta {beta:.2f}: high market exposure; strategy moves almost with the market"
            )

    # Upside capture -- don't want too low
    if not math.isnan(uc):
        if uc > 0.65:
            working.append(f"Upside capture {uc * 100:.0f}%: participates well in market rallies")
        elif uc > 0.45:
            investigate.append(
                f"Upside capture {uc * 100:.0f}%: limited rally participation; verify this is acceptable tradeoff"
            )
        else:
            not_working.append(
                f"Upside capture {uc * 100:.0f}%: very low rally participation; consider if regime signal is too defensive"
            )

    print(f"\n  WORKING  ({len(working)} items):")
    for w in working:
        print(f"    [OK]  {w}")

    print(f"\n  INVESTIGATE  ({len(investigate)} items):")
    for w in investigate:
        print(f"    [?]   {w}")

    print(f"\n  NOT WORKING  ({len(not_working)} items):")
    for w in not_working:
        print(f"    [X]   {w}")

    # ---- Section 7: How to improve ----------------------------------------
    _subsection("7. HOW TO IMPROVE (HYPOTHESIS QUEUE)")
    print("""
  Based on the above, here are the highest-leverage experiments to run:

  [A] If downside capture is too high (> 70%):
      -> Hypothesis: the risk-off sleeve (IEF/TLT/GLD) weight is too small
         or the tau band is too wide, leaving equity exposure too long.
      -> Test: run_paper_trading.py --dry-run shows current weights
      -> Experiment: run_capped_rf_experiment.py (raises risk-off cap to 30%)
      -> Experiment: run_tau_020_experiment.py (faster rebalance response)

  [B] If bear market alpha is negative:
      -> Hypothesis: regime signal is too slow (using 24M lookback)
      -> Experiment: run_12m_lookback_experiment.py (faster regime detection)
      -> Experiment: run_200dma_experiment.py (trend filter as early warning)
      -> Experiment: run_inversion_flag_experiment.py (yield curve early warning)

  [C] If Sortino < Sharpe (losses worse than gains):
      -> Hypothesis: position sizing is too uniform; some assets are higher-risk
      -> Experiment: run_invvol_ablation_experiment.py (inverse vol weighting)
      -> Experiment: run_vol_target_experiment.py (explicit vol targeting at 8%)

  [D] If Calmar < 0.4 (slow recovery from drawdowns):
      -> Hypothesis: max drawdown too large; need faster de-risking
      -> Experiment: run_regime_smoothing_experiment.py (smoother regime labels)
      -> Experiment: run_breadth_flag_experiment.py (breadth as de-risking trigger)

  [E] If upside capture < 50% (missing rallies):
      -> Hypothesis: strategy is too defensively positioned in risk-on regimes
      -> Experiment: run_sigmoid_scale_experiment.py (increase risk-on allocation)
      -> Experiment: run_ro_expansion_experiment.py (expand risk-on sleeve assets)

  To run any experiment:
    uv run python scripts/run_<experiment_name>_experiment.py
  Then compare results vs baseline with:
    uv run python scripts/analyze_walk_forward.py
""")

    # ---- Save CSV ---------------------------------------------------------
    if save_csv:
        all_rows = []
        for row in rows:
            name = row.pop("Name")
            all_rows.append({"Name": name, **row})
        try:
            pd.DataFrame(all_rows).to_csv(save_csv, index=False)
            logger.info("Saved metrics CSV: %s", save_csv)
        except Exception as e:
            logger.warning("Could not save CSV: %s", e)

    print(f"\n{'=' * 70}")
    print("  Run with --help for options (--since, --bench, --save-csv)")
    print(f"{'=' * 70}\n")


def _bear_alpha_for_episode(
    strat: pd.Series, bench: pd.Series, peak: pd.Timestamp, trough: pd.Timestamp
) -> float:
    mask_s = (strat.index >= peak) & (strat.index <= trough)
    mask_b = (bench.index >= peak) & (bench.index <= trough)
    s_ret = float((1 + strat[mask_s]).prod() - 1) if mask_s.any() else float("nan")
    b_ret = float((1 + bench[mask_b]).prod() - 1) if mask_b.any() else float("nan")
    return s_ret - b_ret if not (math.isnan(s_ret) or math.isnan(b_ret)) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strategy performance and bear market analysis")
    parser.add_argument(
        "--since", default=None, metavar="YYYY-MM-DD", help="Only analyse from this date forward"
    )
    parser.add_argument(
        "--bench",
        default="SPY",
        choices=list(BENCH_MAP.keys()),
        help="Primary benchmark ticker (default: SPY)",
    )
    parser.add_argument(
        "--save-csv", default=None, metavar="PATH", help="Save full metrics table to CSV"
    )
    args = parser.parse_args()

    try:
        run_analysis(since=args.since, primary_bench=args.bench, save_csv=args.save_csv)
    except Exception as exc:
        logger.exception("Performance analysis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
