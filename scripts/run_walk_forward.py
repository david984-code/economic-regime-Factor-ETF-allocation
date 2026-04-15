"""Run walk-forward evaluation. Requires regime classification and data first."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.walk_forward import run_walk_forward_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-stagflation-override",
        action="store_true",
        help="Use risk_on blending in Stagflation (baseline behavior)",
    )
    parser.add_argument(
        "--stagflation-risk-on-cap",
        action="store_true",
        help="Cap risk_on at 0.2 when regime==Stagflation (experiment)",
    )
    parser.add_argument(
        "--stagflation-cap-value",
        type=float,
        default=0.2,
        help="Max risk_on when Stagflation (default 0.2)",
    )
    parser.add_argument(
        "--regime-smoothing",
        action="store_true",
        help="Apply rolling mode smoothing to regime labels (experiment)",
    )
    parser.add_argument(
        "--regime-smoothing-window",
        type=int,
        default=3,
        help="Rolling window size in months for regime smoothing (default 3)",
    )
    parser.add_argument(
        "--hybrid-signal",
        action="store_true",
        help="Combine macro_score with market signal (experiment)",
    )
    parser.add_argument(
        "--hybrid-macro-weight",
        type=float,
        default=0.5,
        help="Weight for macro_score in hybrid (default 0.5)",
    )
    parser.add_argument(
        "--market-lookback-months",
        type=int,
        default=12,
        help="Lookback window for market signal in months (default 12)",
    )
    parser.add_argument(
        "--use-momentum",
        action="store_true",
        help="Use momentum signal (+momentum) instead of mean-reversion (-momentum)",
    )
    parser.add_argument(
        "--trend-filter",
        type=str,
        default="none",
        choices=["none", "200dma", "12m_return", "10mma"],
        help="Trend filter to apply (default: none)",
    )
    parser.add_argument(
        "--trend-filter-cap",
        type=float,
        default=0.3,
        help="Max risk_on when trend filter is OFF (default 0.3)",
    )
    parser.add_argument(
        "--vol-scaling",
        type=str,
        default="none",
        choices=["none", "realized_20d", "realized_63d", "percentile"],
        help="Volatility scaling method (default: none)",
    )
    parser.add_argument(
        "--portfolio-construction",
        type=str,
        default="optimizer",
        choices=[
            "optimizer",
            "equal_weight",
            "risk_parity",
            "heuristic",
            "asset_momentum_positive",
            "asset_momentum_top3",
            "asset_momentum_top5",
        ],
        help="Portfolio construction method (default: optimizer)",
    )
    parser.add_argument(
        "--momentum-12m-weight",
        type=float,
        default=0.0,
        help="Weight for 12M momentum in ensemble (0-1, default 0.0)",
    )

    # Fast mode flags
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast experiment mode (recent data, fewer segments, skip persistence)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Fast mode defaults to recent 8 years.",
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Maximum number of walk-forward segments (most recent)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip CSV and SQLite persistence (faster for quick tests)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached intermediate results where available",
    )
    parser.add_argument(
        "--show-timing", action="store_true", help="Show detailed timing breakdown"
    )

    args = parser.parse_args()
    use_override = (
        not args.no_stagflation_override
        and not args.stagflation_risk_on_cap
        and not args.regime_smoothing
        and not args.hybrid_signal
    )
    use_cap = args.stagflation_risk_on_cap
    use_smoothing = args.regime_smoothing
    use_hybrid = args.hybrid_signal
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = run_walk_forward_evaluation(
        start=args.start_date,
        end=args.end_date,
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=use_override,
        use_stagflation_risk_on_cap=use_cap,
        stagflation_risk_on_cap=args.stagflation_cap_value,
        use_regime_smoothing=use_smoothing,
        regime_smoothing_window=args.regime_smoothing_window,
        use_hybrid_signal=use_hybrid,
        hybrid_macro_weight=args.hybrid_macro_weight,
        market_lookback_months=args.market_lookback_months,
        use_momentum=args.use_momentum,
        trend_filter_type=args.trend_filter,
        trend_filter_risk_on_cap=args.trend_filter_cap,
        vol_scaling_method=args.vol_scaling,
        portfolio_construction_method=args.portfolio_construction,
        momentum_12m_weight=args.momentum_12m_weight,
        fast_mode=args.fast_mode,
        max_segments=args.max_segments,
        skip_persist=args.no_persist,
        use_cache=args.use_cache,
        show_timing=args.show_timing,
    )
    if df.empty:
        sys.exit(1)
    print("\n[WALK-FORWARD] Sample results (first 3 segments):")
    print(df.head(3).to_string())
    print("\n[WALK-FORWARD] OVERALL (averaged across segments):")
    overall = df[df["segment"] == "OVERALL"]
    if not overall.empty:
        print(overall.T.to_string())
