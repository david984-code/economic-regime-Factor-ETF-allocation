"""Walk-forward evaluation framework for regime model robustness."""

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.allocation.optimizer import optimize_allocations_from_data
from src.backtest.engine import run_backtest_with_allocations
from src.backtest.metrics import compute_metrics, compute_turnover
from src.config import OUTPUTS_DIR, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices
from src.evaluation.benchmarks import CASH_DAILY_YIELD, compute_benchmark_returns
from src.utils.cache import get_cached, set_cached
from src.utils.timing import TimingReport

logger = logging.getLogger(__name__)


def _make_segments(
    start: str,
    end: str,
    min_train_months: int = 60,
    test_months: int = 12,
    expanding: bool = True,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate walk-forward train/test segments.

    Args:
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        min_train_months: Minimum months for training.
        test_months: Months per test period.
        expanding: If True, expanding window; else rolling.

    Returns:
        List of (train_start, train_end, test_start, test_end).
    """
    dates = pd.date_range(start=start, end=end, freq="ME")
    if len(dates) < min_train_months + test_months:
        return []

    segments: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for i in range(min_train_months, len(dates) - test_months + 1):
        train_end = dates[i - 1]
        test_start = dates[i]
        test_end = dates[i + test_months - 1]
        if expanding:
            train_start = dates[0]
        else:
            train_start = dates[i - min_train_months]
        segments.append((train_start, train_end, test_start, test_end))
    return segments


def _metrics_row(
    name: str,
    rets: pd.Series,
    bench_rets: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Build metrics dict for one series."""
    rets = rets.dropna()
    if len(rets) < 5:
        return {f"{name}_CAGR": np.nan, f"{name}_Sharpe": np.nan, f"{name}_MaxDD": np.nan, f"{name}_Vol": np.nan}
    m = compute_metrics(rets, rf_daily=CASH_DAILY_YIELD, bench_rets=bench_rets)
    row = {
        f"{name}_CAGR": m["CAGR"],
        f"{name}_Sharpe": m["Sharpe"],
        f"{name}_MaxDD": m["Max Drawdown"],
        f"{name}_Vol": m["Volatility"],
    }
    if "Hit Rate" in m:
        row[f"{name}_HitRate"] = m["Hit Rate"]
    if weights is not None and not weights.empty:
        row[f"{name}_Turnover"] = compute_turnover(weights)
    return row


def run_walk_forward_evaluation(
    start: str | None = None,
    end: str | None = None,
    min_train_months: int = 60,
    test_months: int = 12,
    expanding: bool = True,
    output_path: Path | None = None,  # Unused: CSVs always go to outputs/
    use_stagflation_override: bool = True,
    use_stagflation_risk_on_cap: bool = False,
    stagflation_risk_on_cap: float = 0.2,
    use_regime_smoothing: bool = False,
    regime_smoothing_window: int = 3,
    use_hybrid_signal: bool = False,
    hybrid_macro_weight: float = 0.5,
    market_lookback_months: int = 12,
    use_momentum: bool = False,
    trend_filter_type: str = "none",
    trend_filter_risk_on_cap: float = 0.3,
    vol_scaling_method: str = "none",
    portfolio_construction_method: str = "optimizer",
    momentum_12m_weight: float = 0.0,
    fast_mode: bool = False,
    max_segments: int | None = None,
    skip_persist: bool = False,
    use_cache: bool = False,
    show_timing: bool = False,
    tickers: list[str] | None = None,
    assets: list[str] | None = None,
    risk_on_sleeve: list[str] | None = None,
    risk_off_sleeve: list[str] | None = None,
    use_vol_regime: bool = False,
    vol_regime_weight: float = 0.0,
    quarterly_rebalance: bool = False,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Run walk-forward evaluation and save results.

    Writes to outputs/walk_forward_results.csv (latest) and
    outputs/walk_forward_{run_id}.csv (run-specific). SQLite persistence
    is automatic.

    Args:
        start: Start date. Defaults to config START_DATE.
        end: End date. Defaults to today.
        min_train_months: Minimum training months.
        test_months: Test period length in months.
        expanding: Use expanding (True) or rolling (False) train window.
        output_path: Unused; retained for API compatibility.
        use_stagflation_override: If True, use optimizer Stagflation allocation directly
            when regime==Stagflation (experiment). If False, use risk_on blending.
        use_regime_smoothing: If True, smooth regime labels using rolling mode.
        regime_smoothing_window: Rolling window size in months (e.g., 3).
        use_hybrid_signal: If True, combine macro_score with market signal.
        hybrid_macro_weight: Weight for macro_score in hybrid (default 0.5).
        market_lookback_months: Primary lookback for momentum signal (default 12).
        use_momentum: If True, signal = +momentum. If False, signal = -momentum (mean reversion).
        trend_filter_type: Trend filter to apply ('none', '200dma', '12m_return', '10mma').
        trend_filter_risk_on_cap: Max risk_on when trend filter is OFF (default 0.3).
        vol_scaling_method: Volatility scaling method ('none', 'realized_20d', 'realized_63d', 'percentile').
        portfolio_construction_method: Portfolio construction method ('optimizer', 'equal_weight', 'risk_parity', 'heuristic').
        momentum_12m_weight: Weight for 12M momentum in ensemble (0-1).

    Returns:
        DataFrame with metrics per segment and overall.
    """
    from src.config import START_DATE

    # Initialize timing report
    timing = TimingReport() if show_timing else None
    
    start = start or START_DATE
    end = end or get_end_date()
    
    # Fast mode: use recent data only
    if fast_mode and start == START_DATE:
        start = (pd.Timestamp(end) - pd.DateOffset(years=8)).strftime("%Y-%m-%d")
        logger.info("[FAST MODE] Using recent data only: %s to %s", start, end)

    logger.info("Walk-forward: %s to %s, train>=%dmo, test=%dmo, %s",
                start, end, min_train_months, test_months, "expanding" if expanding else "rolling")

    # Use provided universe or defaults
    from src.config import TICKERS, ASSETS
    tickers_to_use = tickers if tickers is not None else TICKERS
    assets_to_use = assets if assets is not None else ASSETS
    
    # Fetch prices (with caching)
    if timing:
        with timing.time("fetch_prices"):
            prices = fetch_prices(tickers=tickers_to_use, start=start, end=end)
    else:
        prices = fetch_prices(tickers=tickers_to_use, start=start, end=end)
    
    # Load regime labels (with caching)
    if timing:
        with timing.time("load_regime_labels"):
            csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
            if csv_path.exists():
                regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            else:
                from src.allocation.optimizer import load_regimes
                regime_df = load_regimes()
            regime_df = regime_df.dropna(subset=["regime"])
            regime_df = regime_df.sort_index()
            if regime_df.index.duplicated().any():
                regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
            regime_df = regime_df.reindex(prices.index).ffill()
    else:
        csv_path = OUTPUTS_DIR / "regime_labels_expanded.csv"
        if csv_path.exists():
            regime_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        else:
            from src.allocation.optimizer import load_regimes
            regime_df = load_regimes()
        regime_df = regime_df.dropna(subset=["regime"])
        regime_df = regime_df.sort_index()
        if regime_df.index.duplicated().any():
            regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
        regime_df = regime_df.reindex(prices.index).ffill()
    
    returns_daily = prices[tickers_to_use].pct_change().iloc[1:]
    returns_daily["cash"] = CASH_DAILY_YIELD

    segments = _make_segments(start, end, min_train_months, test_months, expanding)
    if not segments:
        logger.warning("No segments generated. Need more history.")
        return pd.DataFrame()
    
    # Fast mode: limit segments
    if max_segments and len(segments) > max_segments:
        logger.info("[FAST MODE] Limiting segments from %d to %d", len(segments), max_segments)
        segments = segments[-max_segments:]
    
    logger.info("Generated %d segments", len(segments))

    rows: list[dict] = []
    
    # Pre-compute benchmarks once (outside segment loop)
    if timing:
        with timing.time("compute_benchmarks"):
            benchmarks = compute_benchmark_returns(returns_daily, regime_df)
    else:
        benchmarks = compute_benchmark_returns(returns_daily, regime_df)
    
    for seg_idx, (train_start, train_end, test_start, test_end) in enumerate(segments):
        # Train: filter returns and regimes to train period
        if timing:
            with timing.time("segment_train_data_prep"):
                train_returns = prices.resample("ME").last().pct_change().dropna()
                train_returns = train_returns.loc[train_start:train_end]
                if "cash" not in train_returns.columns:
                    train_returns["cash"] = (1.05) ** (1 / 12) - 1
                train_regimes = regime_df.loc[:train_end].resample("ME").last().dropna(how="all")
                train_regimes = train_regimes.loc[train_regimes.index <= train_end]
        else:
            train_returns = prices.resample("ME").last().pct_change().dropna()
            train_returns = train_returns.loc[train_start:train_end]
            if "cash" not in train_returns.columns:
                train_returns["cash"] = (1.05) ** (1 / 12) - 1
            train_regimes = regime_df.loc[:train_end].resample("ME").last().dropna(how="all")
            train_regimes = train_regimes.loc[train_regimes.index <= train_end]

        if len(train_returns) < 24 or len(train_regimes) < 12:
            continue

        # Optimization (with caching)
        if use_cache:
            opt_params = {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "portfolio_method": portfolio_construction_method,
            }
            allocations = get_cached("allocations", opt_params)
            if allocations is None:
                if timing:
                    with timing.time("segment_optimization"):
                        allocations = optimize_allocations_from_data(train_returns, train_regimes)
                        set_cached("allocations", opt_params, allocations)
                else:
                    allocations = optimize_allocations_from_data(train_returns, train_regimes)
                    set_cached("allocations", opt_params, allocations)
            else:
                if timing:
                    timing.add("segment_optimization_cached", 0)
        else:
            if timing:
                with timing.time("segment_optimization"):
                    allocations = optimize_allocations_from_data(train_returns, train_regimes)
            else:
                allocations = optimize_allocations_from_data(train_returns, train_regimes)
        
        if not allocations:
            continue

        # Test: run backtest on full history, slice to test period
        if timing:
            with timing.time("segment_backtest"):
                result = run_backtest_with_allocations(
                    prices, regime_df, allocations,
                    return_weights=True,
                    use_stagflation_override=use_stagflation_override,
                    use_stagflation_risk_on_cap=use_stagflation_risk_on_cap,
                    stagflation_risk_on_cap=stagflation_risk_on_cap,
                    use_regime_smoothing=use_regime_smoothing,
                    regime_smoothing_window=regime_smoothing_window,
                    use_hybrid_signal=use_hybrid_signal,
                    hybrid_macro_weight=hybrid_macro_weight,
                    market_lookback_months=market_lookback_months,
                    use_momentum=use_momentum,
                    trend_filter_type=trend_filter_type,
                    trend_filter_risk_on_cap=trend_filter_risk_on_cap,
                    vol_scaling_method=vol_scaling_method,
                    portfolio_construction_method=portfolio_construction_method,
                    momentum_12m_weight=momentum_12m_weight,
                    tickers=tickers_to_use,
                    assets=assets_to_use,
                    risk_on_sleeve=risk_on_sleeve,
                    risk_off_sleeve=risk_off_sleeve,
                    use_vol_regime=use_vol_regime,
                    vol_regime_weight=vol_regime_weight,
                    quarterly_rebalance=quarterly_rebalance,
                    tolerance=tolerance,
                )
        else:
            result = run_backtest_with_allocations(
                prices, regime_df, allocations,
                return_weights=True,
                use_stagflation_override=use_stagflation_override,
                use_stagflation_risk_on_cap=use_stagflation_risk_on_cap,
                stagflation_risk_on_cap=stagflation_risk_on_cap,
                use_regime_smoothing=use_regime_smoothing,
                regime_smoothing_window=regime_smoothing_window,
                use_hybrid_signal=use_hybrid_signal,
                hybrid_macro_weight=hybrid_macro_weight,
                market_lookback_months=market_lookback_months,
                use_momentum=use_momentum,
                trend_filter_type=trend_filter_type,
                trend_filter_risk_on_cap=trend_filter_risk_on_cap,
                vol_scaling_method=vol_scaling_method,
                portfolio_construction_method=portfolio_construction_method,
                momentum_12m_weight=momentum_12m_weight,
                tickers=tickers_to_use,
                assets=assets_to_use,
                risk_on_sleeve=risk_on_sleeve,
                risk_off_sleeve=risk_off_sleeve,
                use_vol_regime=use_vol_regime,
                vol_regime_weight=vol_regime_weight,
                quarterly_rebalance=quarterly_rebalance,
                tolerance=tolerance,
            )
        
        if isinstance(result, tuple):
            strat_rets, strat_weights = result
        else:
            strat_rets = result
            strat_weights = None

        test_rets = strat_rets.loc[test_start:test_end].dropna()
        if len(test_rets) < 5:
            continue

        test_bench = {k: v.loc[test_start:test_end].dropna() for k, v in benchmarks.items()}
        bench_spy = bench_rets = test_bench.get("SPY") if "SPY" in test_bench else None

        w = strat_weights.loc[test_start:test_end] if strat_weights is not None else None
        row: dict = {
            "segment": seg_idx,
            "train_start": train_start.strftime("%Y-%m"),
            "train_end": train_end.strftime("%Y-%m"),
            "test_start": test_start.strftime("%Y-%m"),
            "test_end": test_end.strftime("%Y-%m"),
        }
        
        if timing:
            with timing.time("segment_metrics"):
                row.update(_metrics_row("Strategy", test_rets, bench_rets=bench_spy, weights=w))
                for bname, bret in test_bench.items():
                    row.update(_metrics_row(bname, bret))
        else:
            row.update(_metrics_row("Strategy", test_rets, bench_rets=bench_spy, weights=w))
            for bname, bret in test_bench.items():
                row.update(_metrics_row(bname, bret))

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No valid segments.")
        return df

    # Overall: average of segment-level metrics (standard for walk-forward)
    if timing:
        with timing.time("compute_overall_metrics"):
            overall_row: dict = {
                "segment": "OVERALL",
                "train_start": "",
                "train_end": "",
                "test_start": start,
                "test_end": end,
            }
            for col in df.columns:
                if col not in overall_row and df[col].dtype in (np.float64, np.float32):
                    overall_row[col] = df[col].mean()
            df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)
    else:
        overall_row: dict = {
            "segment": "OVERALL",
            "train_start": "",
            "train_end": "",
            "test_start": start,
            "test_end": end,
        }
        for col in df.columns:
            if col not in overall_row and df[col].dtype in (np.float64, np.float32):
                overall_row[col] = df[col].mean()
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

    run_id = str(uuid.uuid4())
    df.insert(0, "run_id", run_id)

    # Persistence (optional in fast mode)
    if not skip_persist:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        
        if timing:
            with timing.time("csv_write"):
                latest_path = OUTPUTS_DIR / "walk_forward_results.csv"
                df.to_csv(latest_path, index=False)
                run_csv_path = OUTPUTS_DIR / f"walk_forward_{run_id}.csv"
                df.to_csv(run_csv_path, index=False)
        else:
            latest_path = OUTPUTS_DIR / "walk_forward_results.csv"
            df.to_csv(latest_path, index=False)
            run_csv_path = OUTPUTS_DIR / f"walk_forward_{run_id}.csv"
            df.to_csv(run_csv_path, index=False)
        
        if timing:
            with timing.time("sqlite_persist"):
                from src.evaluation.model_results_db import persist_walk_forward_run
                persist_walk_forward_run(
                    run_id,
                    df,
                    {
                        "start_date": start,
                        "end_date": end,
                        "min_train_months": min_train_months,
                        "test_months": test_months,
                        "expanding": expanding,
                        "use_stagflation_override": use_stagflation_override,
                        "use_stagflation_risk_on_cap": use_stagflation_risk_on_cap,
                        "stagflation_risk_on_cap": stagflation_risk_on_cap,
                        "use_regime_smoothing": use_regime_smoothing,
                        "regime_smoothing_window": regime_smoothing_window,
                        "use_hybrid_signal": use_hybrid_signal,
                        "hybrid_macro_weight": hybrid_macro_weight,
                        "market_lookback_months": market_lookback_months,
                        "use_momentum": use_momentum,
                        "trend_filter_type": trend_filter_type,
                        "trend_filter_risk_on_cap": trend_filter_risk_on_cap,
                        "vol_scaling_method": vol_scaling_method,
                        "portfolio_construction_method": portfolio_construction_method,
                        "momentum_12m_weight": momentum_12m_weight,
                        "quarterly_rebalance": quarterly_rebalance,
                    },
                    str(run_csv_path),
                )
        else:
            from src.evaluation.model_results_db import persist_walk_forward_run
            persist_walk_forward_run(
                run_id,
                df,
                {
                    "start_date": start,
                    "end_date": end,
                    "min_train_months": min_train_months,
                    "test_months": test_months,
                    "expanding": expanding,
                    "use_stagflation_override": use_stagflation_override,
                    "use_stagflation_risk_on_cap": use_stagflation_risk_on_cap,
                    "stagflation_risk_on_cap": stagflation_risk_on_cap,
                    "use_regime_smoothing": use_regime_smoothing,
                    "regime_smoothing_window": regime_smoothing_window,
                    "use_hybrid_signal": use_hybrid_signal,
                    "hybrid_macro_weight": hybrid_macro_weight,
                    "market_lookback_months": market_lookback_months,
                    "use_momentum": use_momentum,
                    "trend_filter_type": trend_filter_type,
                    "trend_filter_risk_on_cap": trend_filter_risk_on_cap,
                    "vol_scaling_method": vol_scaling_method,
                    "portfolio_construction_method": portfolio_construction_method,
                    "momentum_12m_weight": momentum_12m_weight,
                    "quarterly_rebalance": quarterly_rebalance,
                    "tolerance": tolerance,
                },
                str(run_csv_path),
            )

        logger.info("Saved walk-forward results to %s and %s", latest_path, run_csv_path)
    else:
        logger.info("[FAST MODE] Skipping CSV and SQLite persistence")
    
    # Log timing report
    if timing:
        timing.log_summary()
    
    return df
