"""Diagnose Stagflation underperformance in walk-forward results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.config import OUTPUTS_DIR, START_DATE, TICKERS, get_end_date
from src.data.market_ingestion import fetch_prices
from src.allocation.optimizer import optimize_allocations_from_data
from src.backtest.engine import run_backtest_with_allocations
from src.backtest.metrics import compute_turnover
from src.evaluation.benchmarks import compute_benchmark_returns

ASSETS = TICKERS + ["cash"]


def _get_stagflation_segments() -> pd.DataFrame:
    """Load walk-forward results and tag Stagflation-dominant segments."""
    r = None
    try:
        from src.evaluation.model_results_db import get_latest_run
        r = get_latest_run()
    except Exception:
        pass
    csv_path = OUTPUTS_DIR / f"walk_forward_{r['run_id']}.csv" if r else OUTPUTS_DIR / "walk_forward_results.csv"
    if not csv_path.exists():
        csv_path = OUTPUTS_DIR / "walk_forward_results.csv"
    df = pd.read_csv(csv_path)
    df = df[df["segment"] != "OVERALL"].copy()

    regimes = pd.read_csv(OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"])
    regimes["month"] = pd.to_datetime(regimes["date"]).dt.to_period("M")

    seg_regimes = []
    for _, row in df.iterrows():
        ts = pd.Period(row["test_start"], freq="M")
        te = pd.Period(row["test_end"], freq="M")
        sub = regimes[(regimes["month"] >= ts) & (regimes["month"] <= te)]
        if len(sub) > 0:
            dom = sub["regime"].mode().iloc[0] if len(sub["regime"].mode()) > 0 else "Unknown"
            pct_stag = (sub["regime"] == "Stagflation").mean()
            pct_cont = (sub["regime"] == "Contraction").mean()
        else:
            dom, pct_stag, pct_cont = "Unknown", 0, 0
        seg_regimes.append({"dominant_regime": dom, "pct_stagflation": pct_stag, "pct_contraction": pct_cont})
    df = pd.concat([df, pd.DataFrame(seg_regimes)], axis=1)
    return df[df["dominant_regime"] == "Stagflation"]


def _run_segment_diagnostic(
    seg_idx: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    prices: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> dict:
    """Run one segment and return metrics + allocations."""
    train_returns = prices.resample("ME").last().pct_change().dropna()
    train_returns = train_returns.loc[train_start:train_end]
    if "cash" not in train_returns.columns:
        train_returns["cash"] = (1.05) ** (1 / 12) - 1
    train_regimes = regime_df.loc[:train_end].resample("ME").last().dropna(how="all")
    train_regimes = train_regimes.loc[train_regimes.index <= train_end]
    if len(train_returns) < 24 or len(train_regimes) < 12:
        return None
    allocations = optimize_allocations_from_data(train_returns, train_regimes)
    if not allocations:
        return None
    result = run_backtest_with_allocations(prices, regime_df, allocations, return_weights=True)
    if isinstance(result, tuple):
        strat_rets, strat_weights = result
    else:
        return None
    test_rets = strat_rets.loc[test_start:test_end].dropna()
    test_weights = strat_weights.loc[test_start:test_end]
    if len(test_rets) < 5:
        return None

    returns_daily = prices[TICKERS].pct_change().iloc[1:]
    returns_daily["cash"] = (1.045) ** (1 / 252) - 1
    benchmarks = compute_benchmark_returns(returns_daily, regime_df)
    bench_rets = {k: v.loc[test_start:test_end].dropna() for k, v in benchmarks.items()}

    avg_weights = test_weights.mean()
    end_weights = test_weights.iloc[-1]
    turnover = compute_turnover(test_weights, freq="ME")

    test_regimes = regime_df.loc[test_start:test_end].resample("ME").last()
    regime_dist = test_regimes["regime"].value_counts(normalize=True)
    risk_on_avg = test_regimes["risk_on"].mean() if "risk_on" in test_regimes.columns else np.nan

    from src.backtest.metrics import compute_metrics
    rf = (1.045) ** (1 / 252) - 1
    strat_m = compute_metrics(test_rets, rf_daily=rf)
    bench_metrics = {}
    for b, br in bench_rets.items():
        if len(br) >= 5:
            bench_metrics[b] = compute_metrics(br, rf_daily=rf)

    return {
        "seg_idx": seg_idx,
        "test_start": str(test_start)[:7],
        "test_end": str(test_end)[:7],
        "strategy_cagr": strat_m["CAGR"],
        "strategy_sharpe": strat_m["Sharpe"],
        "strategy_maxdd": strat_m["Max Drawdown"],
        "strategy_vol": strat_m["Volatility"],
        "strategy_turnover": turnover,
        "spy_cagr": bench_metrics.get("SPY", {}).get("CAGR"),
        "spy_sharpe": bench_metrics.get("SPY", {}).get("Sharpe"),
        "b60_40_cagr": bench_metrics.get("60/40", {}).get("CAGR"),
        "eqw_cagr": bench_metrics.get("Equal_Weight", {}).get("CAGR"),
        "risk_on_off_cagr": bench_metrics.get("Risk_On_Off", {}).get("CAGR"),
        "risk_on_avg": risk_on_avg,
        "avg_weights": avg_weights.to_dict(),
        "end_weights": end_weights.to_dict(),
        "regime_dist": regime_dist.to_dict(),
        "stagflation_alloc": allocations.get("Stagflation", {}),
        "contraction_alloc": allocations.get("Contraction", {}),
    }


def main() -> None:
    print("Loading data...")
    stag_df = _get_stagflation_segments()
    if stag_df.empty:
        print("No Stagflation segments found.")
        return
    print(f"Found {len(stag_df)} Stagflation-dominant segments")

    prices = fetch_prices(start=START_DATE, end=get_end_date())
    regime_df = pd.read_csv(OUTPUTS_DIR / "regime_labels_expanded.csv", parse_dates=["date"])
    regime_df = regime_df.dropna(subset=["regime"]).set_index("date").sort_index()
    if regime_df.index.duplicated().any():
        regime_df = regime_df[~regime_df.index.duplicated(keep="last")]
    regime_df = regime_df.reindex(prices.index).ffill()

    seg_list = []
    for _, row in stag_df.iterrows():
        p_start = pd.Period(row["test_start"], freq="M")
        p_end = pd.Period(row["test_end"], freq="M")
        train_end = (p_start - 1).to_timestamp("M")
        test_start = p_start.to_timestamp("M")
        test_end = p_end.to_timestamp("M")
        train_start = pd.Timestamp(START_DATE)
        seg_idx = int(row["segment"])
        diag = _run_segment_diagnostic(
            seg_idx, train_start, train_end, test_start, test_end,
            prices, regime_df,
        )
        if diag:
            seg_list.append(diag)

    if not seg_list:
        print("No diagnostic data collected.")
        return

    # Table of Stagflation segments
    print("\n" + "=" * 80)
    print("STAGFLATION SEGMENTS: METRICS")
    print("=" * 80)
    rows = []
    for d in seg_list[:15]:  # first 15
        rows.append({
            "test_start": d["test_start"],
            "test_end": d["test_end"],
            "Strategy_CAGR": d["strategy_cagr"],
            "Strategy_Sharpe": d["strategy_sharpe"],
            "Strategy_MaxDD": d["strategy_maxdd"],
            "Strategy_Vol": d["strategy_vol"],
            "Strategy_Turnover": d["strategy_turnover"],
            "SPY_CAGR": d["spy_cagr"],
            "60/40_CAGR": d["b60_40_cagr"],
            "EqW_CAGR": d["eqw_cagr"],
            "RiskOnOff_CAGR": d["risk_on_off_cagr"],
            "risk_on_avg": d["risk_on_avg"],
        })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"))
    if len(seg_list) > 15:
        print(f"... and {len(seg_list) - 15} more segments")

    # Average allocations across Stagflation segments
    print("\n" + "=" * 80)
    print("AVERAGE ALLOCATIONS IN STAGFLATION (across segments)")
    print("=" * 80)
    all_avg = {}
    for a in ASSETS:
        vals = [d["avg_weights"].get(a, 0) for d in seg_list if isinstance(d["avg_weights"].get(a), (int, float))]
        all_avg[a] = np.mean(vals) if vals else 0
    for a, w in sorted(all_avg.items(), key=lambda x: -x[1]):
        print(f"  {a:>6}: {w:.2%}")
    print("\nStagflation regime allocation (from optimizer):")
    stag_alloc = seg_list[0]["stagflation_alloc"]
    for a, w in sorted(stag_alloc.items(), key=lambda x: -x[1]):
        print(f"  {a:>6}: {w:.2%}")

    # risk_on analysis
    risk_on_vals = [d["risk_on_avg"] for d in seg_list if pd.notna(d["risk_on_avg"])]
    print(f"\nAverage risk_on during Stagflation test periods: {np.mean(risk_on_vals):.3f}" if risk_on_vals else "\nrisk_on: N/A")

    # Simple alternative
    print("\n" + "=" * 80)
    print("SIMPLE STAGFLATION ALTERNATIVE: 30% GLD, 40% IEF, 20% TLT, 10% cash")
    print("=" * 80)
    rets = prices[[c for c in TICKERS + ["cash"] if c in prices.columns]].copy()
    rets = rets.pct_change().iloc[1:]
    if "cash" not in rets.columns:
        rets["cash"] = (1.045) ** (1 / 252) - 1
    alt_ret = 0.30 * rets["GLD"] + 0.40 * rets["IEF"] + 0.20 * rets["TLT"] + 0.10 * rets["cash"]
    for d in seg_list[:5]:
        ts, te = d["test_start"], d["test_end"]
        ts_d = pd.Period(ts, freq="M").to_timestamp("M")
        te_d = pd.Period(te, freq="M").to_timestamp("M")
        sub = alt_ret.loc[ts_d:te_d].dropna()
        if len(sub) >= 5:
            from src.backtest.metrics import compute_metrics
            m = compute_metrics(sub, rf_daily=(1.045)**(1/252)-1)
            print(f"  {ts} to {te}: Alt CAGR={m['CAGR']:.2%}, Strategy={d['strategy_cagr']:.2%}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    avg_strat_cagr = np.mean([d["strategy_cagr"] for d in seg_list])
    avg_spy_cagr = np.mean([d["spy_cagr"] for d in seg_list])
    avg_risk_on = np.mean(risk_on_vals) if risk_on_vals else 0.5
    avg_gld = np.mean([d["avg_weights"].get("GLD", 0) for d in seg_list])
    avg_equity = np.mean([sum(d["avg_weights"].get(t, 0) for t in ["SPY","MTUM","VLUE","USMV","QUAL","IJR","VIG"]) for d in seg_list])
    print(f"  Strategy avg CAGR in Stagflation: {avg_strat_cagr:.2%}")
    print(f"  SPY avg CAGR in Stagflation: {avg_spy_cagr:.2%}")
    print(f"  Avg risk_on during Stagflation: {avg_risk_on:.3f}")
    print(f"  Avg GLD weight: {avg_gld:.2%}")
    print(f"  Avg equity (SPY+factors) weight: {avg_equity:.2%}")


if __name__ == "__main__":
    main()
