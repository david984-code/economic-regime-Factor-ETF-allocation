# Walk-Forward Evaluation Framework

## Overview

The walk-forward evaluation layer assesses whether the macro regime model is robust enough for paper trading and iteration. It avoids look-ahead bias by training the optimizer only on past data and testing on out-of-sample periods.

## Methodology

### Walk-Forward Design

- **Expanding window (default):** Train on all data from start through `train_end`; test on the next `test_months` months. Each segment adds one month to the training set.
- **Rolling window:** Train on the last `min_train_months` months; test on the next `test_months` months. Fixed training window slides forward.

**Parameters:**
- `min_train_months`: Minimum training history (default 60)
- `test_months`: Length of each out-of-sample test period (default 12)
- `expanding`: `True` for expanding, `False` for rolling

### No Look-Ahead Bias

1. **Optimizer:** Trained only on `(returns, regimes)` for `[train_start, train_end]`. No test-period data is used.
2. **Regime labels:** Rule-based from FRED macro z-scores; computed point-in-time (no future data).
3. **Backtest:** Uses allocations from the train period; runs on full history but metrics are computed only on the test period.

### Per-Segment Flow

1. Filter monthly returns and regime labels to train period.
2. Run `optimize_allocations_from_data(train_returns, train_regimes)` → allocations.
3. Run backtest with those allocations on full prices/regimes.
4. Slice portfolio returns to test period.
5. Compute metrics (CAGR, Sharpe, MaxDD, Vol, Hit Rate, Turnover) for strategy and benchmarks.

## Benchmark Definitions

| Benchmark | Definition |
|-----------|------------|
| **SPY** | SPY buy-and-hold |
| **60/40** | 60% SPY, 40% IEF (bonds) |
| **Equal_Weight** | Equal weight across current asset universe (SPY, GLD, MTUM, VLUE, USMV, QUAL, IJR, VIG, IEF, TLT) |
| **Risk_On_Off** | Blend of equal-weight risk-on assets (SPY, MTUM, VLUE, USMV, QUAL, IJR, VIG) vs risk-off (IEF, TLT, GLD) by `risk_on` value or regime label |

## Metrics

| Metric | Description |
|--------|-------------|
| CAGR | Compound annual growth rate |
| Sharpe | Annualized Sharpe ratio (excess return / vol) |
| MaxDD | Maximum drawdown |
| Vol | Annualized volatility |
| Hit Rate | % of days strategy return > SPY return |
| Turnover | Annualized turnover (Strategy only) |

## Example Output Table

```
segment,train_start,train_end,test_start,test_end,Strategy_CAGR,Strategy_Sharpe,Strategy_MaxDD,Strategy_Vol,Strategy_HitRate,Strategy_Turnover,SPY_CAGR,SPY_Sharpe,...
0,2010-01,2014-12,2015-01,2015-12,0.08,0.95,-0.12,0.15,0.52,2.1,0.12,1.1,...
1,2010-01,2015-12,2016-01,2016-12,0.06,0.72,-0.08,0.14,0.51,1.9,0.10,0.9,...
...
OVERALL,,,2010-01,2025-03,0.07,0.82,-0.10,0.14,0.51,2.0,0.09,0.85,...
```

## Usage

```bash
# Run walk-forward (requires regime classification and data)
python scripts/run_walk_forward.py
```

Or programmatically:

```python
from src.evaluation.walk_forward import run_walk_forward_evaluation

df = run_walk_forward_evaluation(
    min_train_months=60,
    test_months=12,
    expanding=True,
    output_path=Path("outputs/walk_forward_results.csv"),
)
```

## Assumptions and Risks

### Assumptions

1. **Regime labels are point-in-time:** Macro z-scores use only data available at each date.
2. **Monthly rebalance:** Strategy rebalances at month-end; no intra-month regime changes.
3. **Transaction costs:** 8 bps per dollar traded (one-way), applied in backtest.
4. **Benchmark availability:** SPY and IEF must be in the price data for 60/40 and Risk_On_Off.

### Risks

1. **Regime persistence:** If regimes are highly persistent, expanding-window results may be optimistic.
2. **Regime shift:** Structural breaks (e.g., post-2020) may make historical allocations less relevant.
3. **Short test periods:** 12-month test windows have high variance; consider longer test periods for stability.
4. **Survivorship bias:** Asset universe is fixed; no delisted tickers.
5. **Execution:** Paper trading may differ from backtest (slippage, timing).
