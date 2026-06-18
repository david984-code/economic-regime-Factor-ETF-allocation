"""Paired block bootstrap significance test: strategy Sharpe vs monthly-rebalanced 60/40.

Method:
  - Stitched walk-forward OOS daily strategy returns (exp_f_walkforward)
  - 60% SPY / 40% IEF benchmark, rebalanced at each month start
  - Monthly paired returns -> circular block bootstrap (default block_size=6 months)
  - H0: delta_Sharpe = Sharpe(strategy) - Sharpe(60/40) = 0
  - Two-sided p-value via |boot_delta| >= |obs_delta|
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROJECT_ROOT, RF_MONTHLY, START_DATE, get_end_date
from src.data.market_ingestion import fetch_prices
from src.research.exp_f_walkforward import exp_f_walkforward

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE_MONTHS = 6
DEFAULT_N_ITERATIONS = 10_000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class BootstrapResult:
    observed_delta_sharpe: float
    p_value_two_sided: float
    ci_95_low: float
    ci_95_high: float
    observed_strategy_sharpe: float
    observed_benchmark_sharpe: float
    n_months: int
    n_iterations: int
    block_size: int
    alpha: float
    verdict: str
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def daily_to_monthly_returns(daily: pd.Series) -> pd.Series:
    """Compound daily returns to calendar month-end returns."""
    daily = daily.dropna().sort_index()
    if daily.empty:
        return pd.Series(dtype=float)
    monthly = (1.0 + daily).groupby(daily.index.to_period("M")).prod() - 1.0
    monthly.index = monthly.index.to_timestamp("M")
    return monthly.astype(float)


def monthly_sharpe(monthly_rets: pd.Series, rf_monthly: float = RF_MONTHLY) -> float:
    """Annualized Sharpe from monthly excess returns."""
    rets = monthly_rets.dropna()
    if len(rets) < 2:
        return float("nan")
    excess = rets - rf_monthly
    std = float(excess.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float((excess.mean() / std) * np.sqrt(12.0))


def compute_monthly_rebalanced_6040(
    spy_daily: pd.Series,
    ief_daily: pd.Series,
    equity_weight: float = 0.6,
    bond_weight: float = 0.4,
) -> pd.Series:
    """Monthly-rebalanced 60/40 from daily SPY and IEF over aligned calendar."""
    df = pd.DataFrame({"SPY": spy_daily, "IEF": ief_daily}).dropna().sort_index()
    if df.empty:
        return pd.Series(dtype=float)

    target = np.array([equity_weight, bond_weight], dtype=float)
    target = target / target.sum()
    monthly: list[tuple[pd.Timestamp, float]] = []

    for _, group in df.groupby(df.index.to_period("M")):
        if group.empty:
            continue
        weights = target.copy()
        wealth = 1.0
        for day_rets in group[["SPY", "IEF"]].to_numpy(dtype=float):
            port_r = float(np.dot(weights, day_rets))
            wealth *= 1.0 + port_r
            weights = weights * (1.0 + day_rets)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
        monthly.append((group.index[-1], wealth - 1.0))

    out = pd.Series(dict(monthly), dtype=float).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def align_paired_monthly_returns(
    strategy_daily: pd.Series,
    spy_daily: pd.Series,
    ief_daily: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Build aligned monthly (strategy, 60/40) return pairs on common month-ends."""
    strat_m = daily_to_monthly_returns(strategy_daily)
    bench_m = compute_monthly_rebalanced_6040(spy_daily, ief_daily)

    strat_m.index = pd.to_datetime(strat_m.index).to_period("M")
    bench_m.index = pd.to_datetime(bench_m.index).to_period("M")
    common = strat_m.index.intersection(bench_m.index)
    if len(common) < 2:
        raise ValueError(f"Need >=2 overlapping monthly observations; got {len(common)}.")
    common = common.sort_values()
    return strat_m.loc[common], bench_m.loc[common]


def _circular_block_resample_paired(
    strategy: np.ndarray,
    benchmark: np.ndarray,
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample n paired observations using circular blocks of length block_size."""
    boot_s = np.empty(n, dtype=float)
    boot_b = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        for j in range(block_size):
            if filled >= n:
                break
            idx = (start + j) % n
            boot_s[filled] = strategy[idx]
            boot_b[filled] = benchmark[idx]
            filled += 1
    return boot_s, boot_b


def paired_block_bootstrap_delta_sharpe(
    strategy_monthly: pd.Series,
    benchmark_monthly: pd.Series,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE_MONTHS,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    rf_monthly: float = RF_MONTHLY,
) -> BootstrapResult:
    """Two-sided paired block bootstrap on delta Sharpe vs 60/40."""
    aligned = pd.concat(
        [strategy_monthly.rename("strategy"), benchmark_monthly.rename("benchmark")],
        axis=1,
    ).dropna()
    n = len(aligned)
    if n < max(2, block_size):
        raise ValueError(
            f"Need n_months >= max(2, block_size); got n={n}, block_size={block_size}."
        )

    strat = aligned["strategy"].to_numpy(dtype=float)
    bench = aligned["benchmark"].to_numpy(dtype=float)

    obs_strat = monthly_sharpe(aligned["strategy"], rf_monthly=rf_monthly)
    obs_bench = monthly_sharpe(aligned["benchmark"], rf_monthly=rf_monthly)
    obs_delta = obs_strat - obs_bench

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        boot_s, boot_b = _circular_block_resample_paired(strat, bench, n, block_size, rng)
        boot_deltas[i] = monthly_sharpe(pd.Series(boot_s), rf_monthly=rf_monthly) - monthly_sharpe(
            pd.Series(boot_b), rf_monthly=rf_monthly
        )

    # Center bootstrap distribution before the H0:delta=0 test. Under H_alt the
    # bootstrap is centered at obs_delta, so an uncentered |boot| >= |obs| test
    # would mostly measure distributional symmetry, not deviation from zero
    # (Politis & Romano 1994; Lahiri 2003).
    abs_obs = abs(obs_delta)
    centered = boot_deltas - boot_deltas.mean()
    p_two_sided = float(np.mean(np.abs(centered) >= abs_obs))
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    verdict = "reject_H0" if p_two_sided < alpha else "fail_to_reject_H0"

    return BootstrapResult(
        observed_delta_sharpe=float(obs_delta),
        p_value_two_sided=p_two_sided,
        ci_95_low=float(ci_low),
        ci_95_high=float(ci_high),
        observed_strategy_sharpe=float(obs_strat),
        observed_benchmark_sharpe=float(obs_bench),
        n_months=n,
        n_iterations=n_iterations,
        block_size=block_size,
        alpha=alpha,
        verdict=verdict,
        seed=seed,
    )


def run_bootstrap_significance(
    *,
    start: str | None = None,
    end: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE_MONTHS,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    fast_mode: bool = False,
    max_segments: int | None = None,
) -> BootstrapResult:
    """End-to-end: WF OOS strategy returns + 60/40 benchmark + block bootstrap."""
    strategy_daily = exp_f_walkforward(
        start=start,
        end=end,
        fast_mode=fast_mode,
        max_segments=max_segments,
    )
    if strategy_daily.empty:
        raise ValueError("exp_f_walkforward returned empty OOS daily returns.")

    idx_start = strategy_daily.index.min().strftime("%Y-%m-%d")
    idx_end = strategy_daily.index.max().strftime("%Y-%m-%d")
    prices = fetch_prices(tickers=["SPY", "IEF"], start=idx_start, end=idx_end)
    spy_daily = prices["SPY"].pct_change().reindex(strategy_daily.index)
    ief_daily = prices["IEF"].pct_change().reindex(strategy_daily.index)

    strat_m, bench_m = align_paired_monthly_returns(strategy_daily, spy_daily, ief_daily)
    return paired_block_bootstrap_delta_sharpe(
        strat_m,
        bench_m,
        block_size=block_size,
        n_iterations=n_iterations,
        seed=seed,
        alpha=alpha,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired block bootstrap: strategy Sharpe vs monthly-rebalanced 60/40."
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE_MONTHS)
    parser.add_argument("--n-iterations", type=int, default=DEFAULT_N_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "bootstrap_significance.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info(
        "Defaults: start=%s end=%s block_size=%d n_iterations=%d",
        args.start or START_DATE,
        args.end or get_end_date(),
        args.block_size,
        args.n_iterations,
    )

    result = run_bootstrap_significance(
        start=args.start,
        end=args.end,
        block_size=args.block_size,
        n_iterations=args.n_iterations,
        seed=args.seed,
        alpha=args.alpha,
        fast_mode=args.fast_mode,
        max_segments=args.max_segments,
    )
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
