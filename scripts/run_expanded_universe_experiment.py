"""Test expanded asset universe vs current baseline."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import (
    OUTPUTS_DIR,
    TICKERS,
    ASSETS,
    TICKERS_EXPANDED,
    ASSETS_EXPANDED,
    RISK_ON_ASSETS_BASE,
    RISK_OFF_ASSETS_BASE,
    RISK_ON_ASSETS_EXPANDED,
    RISK_OFF_ASSETS_EXPANDED,
)
from src.evaluation.walk_forward import run_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _run_experiment(
    universe_name: str,
    tickers: list[str],
    assets: list[str],
    risk_on_sleeve: list[str],
    risk_off_sleeve: list[str],
    fast_mode: bool = True,
) -> pd.DataFrame:
    """Run 24M momentum experiment with specified universe."""
    logger.info("=" * 80)
    logger.info("RUNNING: %s", universe_name)
    logger.info("=" * 80)
    logger.info(f"Total assets: {len(tickers)}")
    logger.info(f"Risk-on sleeve: {len(risk_on_sleeve)} assets")
    logger.info(f"Risk-off sleeve: {len(risk_off_sleeve)} assets")
    
    df = run_walk_forward_evaluation(
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=False,
        use_stagflation_risk_on_cap=False,
        use_regime_smoothing=False,
        use_hybrid_signal=True,
        hybrid_macro_weight=0.0,  # Pure market
        market_lookback_months=24,  # 24M momentum
        use_momentum=True,  # Momentum
        trend_filter_type="none",
        vol_scaling_method="none",
        portfolio_construction_method="equal_weight",
        tickers=tickers,
        assets=assets,
        risk_on_sleeve=risk_on_sleeve,
        risk_off_sleeve=risk_off_sleeve,
        fast_mode=fast_mode,
        max_segments=20 if fast_mode else None,
        skip_persist=fast_mode,
        use_cache=fast_mode,
        show_timing=False,
    )
    return df


def _difficult_period_metrics(df: pd.DataFrame) -> dict:
    """Extract 2021-2022 period metrics."""
    test_segments = df[df["segment"] != "OVERALL"].copy()
    
    if test_segments.empty:
        return {}
    
    difficult_segments = []
    for _, row in test_segments.iterrows():
        test_start = pd.Period(row["test_start"], freq="M").to_timestamp()
        test_end = pd.Period(row["test_end"], freq="M").to_timestamp()
        
        if (test_start.year >= 2021 and test_start.year <= 2022) or \
           (test_end.year >= 2021 and test_end.year <= 2022):
            difficult_segments.append(row)
    
    if not difficult_segments:
        return {
            "n_difficult": 0,
            "difficult_cagr": np.nan,
            "difficult_sharpe": np.nan,
            "difficult_maxdd": np.nan,
        }
    
    difficult_df = pd.DataFrame(difficult_segments)
    
    return {
        "n_difficult": len(difficult_df),
        "difficult_cagr": difficult_df["Strategy_CAGR"].mean(),
        "difficult_sharpe": difficult_df["Strategy_Sharpe"].mean(),
        "difficult_maxdd": difficult_df["Strategy_MaxDD"].mean(),
    }


def main():
    """Run expanded universe experiment."""
    logger.info("Starting expanded universe experiment")
    logger.info("Fixed: 24M momentum + equal-weight sleeves")
    logger.info("")
    
    # Universe definitions
    universes = [
        {
            "name": "Baseline (10 assets)",
            "tickers": TICKERS,
            "assets": ASSETS,
            "risk_on": RISK_ON_ASSETS_BASE,
            "risk_off": RISK_OFF_ASSETS_BASE,
        },
        {
            "name": "Expanded (21 assets + sectors)",
            "tickers": TICKERS_EXPANDED,
            "assets": ASSETS_EXPANDED,
            "risk_on": RISK_ON_ASSETS_EXPANDED,
            "risk_off": RISK_OFF_ASSETS_EXPANDED,
        },
    ]
    
    # === PHASE 1: Fast mode screening ===
    logger.info("=" * 80)
    logger.info("PHASE 1: FAST MODE SCREENING")
    logger.info("=" * 80)
    logger.info("Recent 8 years, max 20 segments, no persistence")
    logger.info("")
    
    fast_start = time.perf_counter()
    fast_results = []
    
    for config in universes:
        df = _run_experiment(
            universe_name=config["name"],
            tickers=config["tickers"],
            assets=config["assets"],
            risk_on_sleeve=config["risk_on"],
            risk_off_sleeve=config["risk_off"],
            fast_mode=True,
        )
        
        if df.empty:
            logger.error(f"Experiment failed for {config['name']}")
            continue
        
        overall = df[df["segment"] == "OVERALL"].iloc[0]
        difficult = _difficult_period_metrics(df)
        
        result = {
            "universe": config["name"],
            "n_assets": len(config["tickers"]),
            "n_risk_on": len(config["risk_on"]),
            "n_risk_off": len(config["risk_off"]),
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
            "difficult_cagr": difficult.get("difficult_cagr", np.nan),
            "difficult_sharpe": difficult.get("difficult_sharpe", np.nan),
            "difficult_maxdd": difficult.get("difficult_maxdd", np.nan),
        }
        fast_results.append(result)
    
    fast_elapsed = (time.perf_counter() - fast_start) / 60
    
    if not fast_results:
        logger.error("All fast mode experiments failed.")
        sys.exit(1)
    
    fast_df = pd.DataFrame(fast_results)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("FAST MODE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Runtime: {fast_elapsed:.1f} minutes")
    logger.info("")
    logger.info("| Universe | CAGR | Sharpe | MaxDD | Turnover |")
    logger.info("|----------|------|--------|-------|----------|")
    
    for _, row in fast_df.iterrows():
        logger.info(
            f"| {row['universe']:>30} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['turnover']:>8.1%} |"
        )
    
    logger.info("")
    
    # Check if expanded universe is promising
    baseline_fast = fast_df[fast_df["universe"] == "Baseline (10 assets)"].iloc[0]
    expanded_fast = fast_df[fast_df["universe"] == "Expanded (21 assets + sectors)"].iloc[0]
    
    sharpe_improvement = expanded_fast["sharpe"] - baseline_fast["sharpe"]
    cagr_improvement = expanded_fast["cagr"] - baseline_fast["cagr"]
    
    logger.info("FAST MODE COMPARISON:")
    logger.info(f"  Sharpe improvement: {sharpe_improvement:+.3f}")
    logger.info(f"  CAGR improvement: {cagr_improvement:+.2%}")
    logger.info("")
    
    # Decision: run full validation if promising
    run_full_validation = sharpe_improvement > 0.02 or cagr_improvement > 0.01
    
    if not run_full_validation:
        logger.info("=" * 80)
        logger.info("DECISION: SKIP FULL VALIDATION")
        logger.info("=" * 80)
        logger.info("Fast mode shows marginal or negative improvement.")
        logger.info("Expanded universe does not appear promising.")
        logger.info("")
        
        # Generate fast-mode-only report
        _generate_report(fast_df, None, fast_elapsed, 0, run_full_validation=False)
        return
    
    # === PHASE 2: Full validation ===
    logger.info("=" * 80)
    logger.info("PHASE 2: FULL VALIDATION")
    logger.info("=" * 80)
    logger.info("Expanded universe shows promise in fast mode.")
    logger.info("Running full backtest (all segments, full history)...")
    logger.info("")
    
    full_start = time.perf_counter()
    full_results = []
    
    for config in universes:
        df = _run_experiment(
            universe_name=config["name"],
            tickers=config["tickers"],
            assets=config["assets"],
            risk_on_sleeve=config["risk_on"],
            risk_off_sleeve=config["risk_off"],
            fast_mode=False,
        )
        
        if df.empty:
            logger.error(f"Full validation failed for {config['name']}")
            continue
        
        overall = df[df["segment"] == "OVERALL"].iloc[0]
        difficult = _difficult_period_metrics(df)
        
        result = {
            "universe": config["name"],
            "n_assets": len(config["tickers"]),
            "n_risk_on": len(config["risk_on"]),
            "n_risk_off": len(config["risk_off"]),
            "cagr": overall["Strategy_CAGR"],
            "sharpe": overall["Strategy_Sharpe"],
            "maxdd": overall["Strategy_MaxDD"],
            "vol": overall["Strategy_Vol"],
            "turnover": overall.get("Strategy_Turnover", 0.0),
            "difficult_cagr": difficult.get("difficult_cagr", np.nan),
            "difficult_sharpe": difficult.get("difficult_sharpe", np.nan),
            "difficult_maxdd": difficult.get("difficult_maxdd", np.nan),
        }
        full_results.append(result)
    
    full_elapsed = (time.perf_counter() - full_start) / 60
    
    if not full_results:
        logger.error("All full validation experiments failed.")
        sys.exit(1)
    
    full_df = pd.DataFrame(full_results)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("FULL VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Runtime: {full_elapsed:.1f} minutes")
    logger.info("")
    logger.info("| Universe | CAGR | Sharpe | MaxDD | Turnover |")
    logger.info("|----------|------|--------|-------|----------|")
    
    for _, row in full_df.iterrows():
        logger.info(
            f"| {row['universe']:>30} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['turnover']:>8.1%} |"
        )
    
    logger.info("")
    
    # Generate report
    _generate_report(fast_df, full_df, fast_elapsed, full_elapsed, run_full_validation=True)


def _generate_report(
    fast_df: pd.DataFrame,
    full_df: pd.DataFrame | None,
    fast_elapsed: float,
    full_elapsed: float,
    run_full_validation: bool,
):
    """Generate experiment report."""
    report_lines = [
        "# Expanded Universe Experiment",
        "",
        "## Experiment Setup",
        "",
        "**Fixed Setup:**",
        "- Pure market model (0.0 macro / 1.0 market)",
        "- 24-month momentum signal",
        "- Equal-weight sleeve construction",
        "- Corrected no-lookahead implementation",
        "",
        "**Universes Tested:**",
        "",
    ]
    
    baseline = fast_df[fast_df["universe"] == "Baseline (10 assets)"].iloc[0]
    expanded = fast_df[fast_df["universe"] == "Expanded (21 assets + sectors)"].iloc[0]
    
    report_lines.extend([
        "### 1. Baseline Universe (10 assets)",
        "",
        "**Risk-on sleeve (7 assets):**",
        "- SPY (S&P 500)",
        "- MTUM (Momentum factor)",
        "- VLUE (Value factor)",
        "- QUAL (Quality factor)",
        "- USMV (Minimum volatility factor)",
        "- IJR (Small cap)",
        "- VIG (Dividend growth)",
        "",
        "**Risk-off sleeve (3 assets):**",
        "- IEF (7-10Y Treasuries)",
        "- TLT (20+ Y Treasuries)",
        "- GLD (Gold)",
        "",
        "### 2. Expanded Universe (21 assets)",
        "",
        "**Risk-on sleeve (18 assets):**",
        "- All 7 baseline risk-on assets",
        "- **Plus 11 sector ETFs:**",
        "  - XLK (Technology)",
        "  - XLF (Financials)",
        "  - XLE (Energy)",
        "  - XLV (Healthcare)",
        "  - XLI (Industrials)",
        "  - XLP (Consumer Staples)",
        "  - XLY (Consumer Discretionary)",
        "  - XLU (Utilities)",
        "  - XLB (Materials)",
        "  - XLRE (Real Estate)",
        "  - XLC (Communication Services)",
        "",
        "**Risk-off sleeve (3 assets):**",
        "- Same as baseline (IEF, TLT, GLD)",
        "",
        "## Phase 1: Fast Mode Screening",
        "",
        f"**Runtime:** {fast_elapsed:.1f} minutes",
        "",
        "| Universe | N Assets | Risk-On | Risk-Off | CAGR | Sharpe | MaxDD | Turnover |",
        "|----------|----------|---------|----------|------|--------|-------|----------|",
    ])
    
    for _, row in fast_df.iterrows():
        report_lines.append(
            f"| {row['universe']:>30} | {row['n_assets']:>8} | {row['n_risk_on']:>7} | "
            f"{row['n_risk_off']:>8} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
            f"{row['maxdd']:>5.2%} | {row['turnover']:>8.1%} |"
        )
    
    report_lines.extend([
        "",
        "### Fast Mode Comparison",
        "",
        f"**Baseline:** {baseline['sharpe']:.3f} Sharpe, {baseline['cagr']:.2%} CAGR",
        f"**Expanded:** {expanded['sharpe']:.3f} Sharpe, {expanded['cagr']:.2%} CAGR",
        "",
        f"**Improvement:** {expanded['sharpe'] - baseline['sharpe']:+.3f} Sharpe, {expanded['cagr'] - baseline['cagr']:+.2%} CAGR",
        "",
    ])
    
    # Full validation results
    if run_full_validation and full_df is not None:
        baseline_full = full_df[full_df["universe"] == "Baseline (10 assets)"].iloc[0]
        expanded_full = full_df[full_df["universe"] == "Expanded (21 assets + sectors)"].iloc[0]
        
        report_lines.extend([
            "## Phase 2: Full Validation",
            "",
            f"**Runtime:** {full_elapsed:.1f} minutes",
            "",
            "| Universe | CAGR | Sharpe | MaxDD | Vol | Turnover |",
            "|----------|------|--------|-------|-----|----------|",
        ])
        
        for _, row in full_df.iterrows():
            report_lines.append(
                f"| {row['universe']:>30} | {row['cagr']:>4.2%} | {row['sharpe']:>6.3f} | "
                f"{row['maxdd']:>5.2%} | {row['vol']:>3.2%} | {row['turnover']:>8.1%} |"
            )
        
        report_lines.extend([
            "",
            "### Full Validation Comparison",
            "",
            f"**Baseline:** {baseline_full['sharpe']:.3f} Sharpe, {baseline_full['cagr']:.2%} CAGR",
            f"**Expanded:** {expanded_full['sharpe']:.3f} Sharpe, {expanded_full['cagr']:.2%} CAGR",
            "",
            f"**Improvement:** {expanded_full['sharpe'] - baseline_full['sharpe']:+.3f} Sharpe, "
            f"{expanded_full['cagr'] - baseline_full['cagr']:+.2%} CAGR",
            "",
            "### Difficult Period Performance (2021-2022)",
            "",
            "| Universe | CAGR | Sharpe | MaxDD |",
            "|----------|------|--------|-------|",
        ])
        
        for _, row in full_df.iterrows():
            if not pd.isna(row["difficult_cagr"]):
                report_lines.append(
                    f"| {row['universe']:>30} | {row['difficult_cagr']:>4.2%} | "
                    f"{row['difficult_sharpe']:>6.3f} | {row['difficult_maxdd']:>5.2%} |"
                )
        
        report_lines.extend([
            "",
            "### Analysis",
            "",
        ])
        
        sharpe_imp_full = expanded_full['sharpe'] - baseline_full['sharpe']
        turnover_diff_full = expanded_full['turnover'] - baseline_full['turnover']
        
        # Recommendation based on full validation
        if sharpe_imp_full > 0.03:
            verdict = "**ADOPT EXPANDED UNIVERSE**"
            explanation = (
                f"Expanded universe materially outperforms baseline:\n"
                f"- Sharpe: {sharpe_imp_full:+.3f}\n"
                f"- CAGR: {expanded_full['cagr'] - baseline_full['cagr']:+.2%}\n"
                f"- Turnover: {turnover_diff_full:+.1%}\n"
                f"- {expanded_full['n_risk_on']} risk-on assets vs {baseline_full['n_risk_on']}\n"
                f"- Sector rotation provides meaningful diversification"
            )
        elif sharpe_imp_full > 0.0:
            verdict = "**MARGINAL IMPROVEMENT: Consider expanded universe**"
            explanation = (
                f"Expanded universe improves Sharpe by {sharpe_imp_full:+.3f}.\n"
                "Benefit is modest but may justify added complexity if:\n"
                f"- Turnover impact is acceptable ({turnover_diff_full:+.1%})\n"
                "- Sector diversification aligns with strategy goals\n"
                "- Implementation cost is low"
            )
        else:
            verdict = "**KEEP BASELINE: Expanded universe does not improve performance**"
            explanation = (
                f"Expanded universe underperforms baseline by {sharpe_imp_full:+.3f} Sharpe.\n"
                "Current 10-asset universe is optimal.\n"
                "Sector ETFs add complexity without improving risk-adjusted returns."
            )
        
        report_lines.append(verdict)
        report_lines.append("")
        report_lines.append(explanation)
        report_lines.append("")
    
    else:
        # Fast mode only (no full validation)
        report_lines.extend([
            "## Fast Mode Decision",
            "",
        ])
        
        sharpe_imp_fast = expanded['sharpe'] - baseline['sharpe']
        
        verdict = "**SKIP FULL VALIDATION: No improvement in fast mode**"
        explanation = (
            f"Fast mode screening shows {sharpe_imp_fast:+.3f} Sharpe improvement.\n"
            "This is below the threshold (+0.02) for running full validation.\n"
            "Expanded universe does not appear to improve performance."
        )
        
        report_lines.append(verdict)
        report_lines.append("")
        report_lines.append(explanation)
        report_lines.append("")
    
    # Next experiment recommendations
    report_lines.extend([
        "## Next Experiment Recommendations",
        "",
    ])
    
    if run_full_validation and full_df is not None:
        baseline_full = full_df[full_df["universe"] == "Baseline (10 assets)"].iloc[0]
        expanded_full = full_df[full_df["universe"] == "Expanded (21 assets + sectors)"].iloc[0]
        
        if expanded_full["sharpe"] > baseline_full["sharpe"] + 0.03:
            report_lines.append("1. **Test alternative sector selection** - filter sectors by momentum or regime")
            report_lines.append("2. **Test sector-specific sleeves** - create sector-rotational strategies")
            report_lines.append("3. **Test dynamic universe** - vary sector inclusion by macro regime")
        else:
            report_lines.append("1. **Test alternative asset classes** - commodities, international, REITs")
            report_lines.append("2. **Test rebalance frequency** - quarterly vs monthly (reduce turnover)")
            report_lines.append("3. **Test regime-conditional universes** - vary assets by macro regime")
    else:
        report_lines.append("1. **Test alternative diversifiers** - international, commodities, alternatives")
        report_lines.append("2. **Test rebalance frequency** - quarterly vs monthly")
        report_lines.append("3. **Test regime-conditional universes** - vary assets by macro regime")
    
    report_lines.append("4. **Test cross-sectional momentum** - rank assets by relative strength")
    report_lines.append("5. **Test alternative signal horizons** - blend multiple momentum lookbacks")
    
    report_lines.extend([
        "",
        "## Summary",
        "",
        "### Fast Mode Performance",
        "",
        "| Universe | Assets | CAGR | Sharpe | MaxDD | Vol | Turnover | Difficult Sharpe |",
        "|----------|--------|------|--------|-------|-----|----------|------------------|",
    ])
    
    for _, row in fast_df.iterrows():
        report_lines.append(
            f"| {row['universe']:>30} | {row['n_assets']:>6} | {row['cagr']:>4.2%} | "
            f"{row['sharpe']:>6.3f} | {row['maxdd']:>5.2%} | {row['vol']:>3.2%} | "
            f"{row['turnover']:>8.1%} | {row['difficult_sharpe']:>16.3f} |"
        )
    
    if run_full_validation and full_df is not None:
        report_lines.extend([
            "",
            "### Full Validation Performance",
            "",
            "| Universe | Assets | CAGR | Sharpe | MaxDD | Vol | Turnover | Difficult Sharpe |",
            "|----------|--------|------|--------|-------|-----|----------|------------------|",
        ])
        
        for _, row in full_df.iterrows():
            report_lines.append(
                f"| {row['universe']:>30} | {row['n_assets']:>6} | {row['cagr']:>4.2%} | "
                f"{row['sharpe']:>6.3f} | {row['maxdd']:>5.2%} | {row['vol']:>3.2%} | "
                f"{row['turnover']:>8.1%} | {row['difficult_sharpe']:>16.3f} |"
            )
    
    report = "\n".join(report_lines)
    
    output_path = OUTPUTS_DIR / "EXPANDED_UNIVERSE_EXPERIMENT.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {output_path}")
    logger.info("")
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
