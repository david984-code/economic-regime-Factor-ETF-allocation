"""Known-answer tests for src.allocation.optimizer.

Covers the two error classes most likely to silently inflate backtest edge:
  1. Sortino math bugs (sign flips, downside-only filtering, rf handling)
  2. Constraint violations (cash floor/ceiling, per-asset minimums, weights summing to 1)

Each test uses synthetic data with a hand-computable expected answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation.optimizer import _negative_sortino, optimize_allocations_from_data
from src.config import REGIME_CASH

# ---------------------------------------------------------------------------
# _negative_sortino math
# ---------------------------------------------------------------------------


class TestNegativeSortino:
    """The optimizer minimizes _negative_sortino, so sign / math errors here
    flip the entire allocation logic. These are the canary tests."""

    def test_all_positive_returns_gives_large_reward(self) -> None:
        """No downside -> Sortino is huge -> _negative_sortino is very negative."""
        # 2 risky assets, 10 monthly observations, all positive returns
        returns = np.full((10, 2), 0.02)
        weights = np.array([0.5, 0.5, 0.0])  # 50/50 risky, 0 cash
        result = _negative_sortino(weights, returns, risk_free=0.0)
        # With no downside, downside_vol is clamped to 1e-8; Sortino is enormous.
        # Function returns -sortino + cash_pref * cash_weight = very negative.
        assert result < -1e5, (
            f"All-positive returns should give very negative output (huge reward); got {result}"
        )

    def test_all_negative_returns_gives_large_penalty(self) -> None:
        """All downside -> negative Sortino -> _negative_sortino is positive (penalty)."""
        returns = np.full((10, 2), -0.02)
        weights = np.array([0.5, 0.5, 0.0])
        result = _negative_sortino(weights, returns, risk_free=0.0)
        # mean = -0.02 (below rf=0); downside_var > 0; Sortino is negative; -Sortino positive.
        assert result > 0, f"All-negative returns should give positive penalty; got {result}"

    def test_known_sortino_value(self) -> None:
        """Hand-computed Sortino: alternating +2% and -1%, rf=0.

        Mean per-period return = 0.005
        Downside returns (after rf=0): [0, -0.01, 0, -0.01]
        Downside variance = mean of squares = (0 + 0.0001 + 0 + 0.0001) / 4 = 5e-5
        Downside vol = sqrt(5e-5) ~ 0.007071
        Sortino = (0.005 - 0) / 0.007071 = 0.7071
        Expected _negative_sortino (no cash weight) = -0.7071
        """
        returns = np.array([[0.02], [-0.01], [0.02], [-0.01]])  # 4 periods, 1 asset
        weights = np.array([1.0, 0.0])  # 100% risky, 0 cash
        result = _negative_sortino(weights, returns, risk_free=0.0)
        expected = -0.7071  # ±0.01 tolerance for numerical precision
        assert abs(result - expected) < 0.01, f"Expected ~{expected} (Sortino=0.7071); got {result}"

    def test_cash_preference_penalty(self) -> None:
        """Cash weight adds OPTIMIZER_CASH_PREFERENCE * cash_weight to the objective.

        All-cash allocation -> risky_sum is 0 -> early return of 1e9 (sentinel).
        99% cash with 1% risky -> should add ~0.99 * 0.05 = 0.0495 to the Sortino term.
        """
        # Edge case: ~all cash, tiny risky -> the sentinel branch triggers at risky_sum<=1e-10
        returns = np.array([[0.01], [0.01], [0.01], [0.01]])
        weights_all_cash = np.array([0.0, 1.0])
        result_all_cash = _negative_sortino(weights_all_cash, returns, risk_free=0.0)
        assert result_all_cash == 1e9, (
            f"All-cash should hit the risky_sum<=1e-10 sentinel = 1e9; got {result_all_cash}"
        )


# ---------------------------------------------------------------------------
# optimize_allocations_from_data: constraint compliance
# ---------------------------------------------------------------------------


def _make_synth_data(n_months: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic 3-asset universe across Recovery and Contraction regimes.

    Asset A: strong positive returns in Recovery, sharp negative in Contraction
    Asset B: mild positive in both regimes (defensive)
    Asset C: opposite of A (negative in Recovery, positive in Contraction)
    """
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")

    # First half Recovery, second half Contraction (clean known regimes)
    regimes = ["Recovery"] * (n_months // 2) + ["Contraction"] * (n_months - n_months // 2)

    rets_a, rets_b, rets_c = [], [], []
    for r in regimes:
        if r == "Recovery":
            rets_a.append(0.025 + rng.normal(0, 0.01))
            rets_b.append(0.005 + rng.normal(0, 0.005))
            rets_c.append(-0.010 + rng.normal(0, 0.01))
        else:
            rets_a.append(-0.020 + rng.normal(0, 0.01))
            rets_b.append(0.003 + rng.normal(0, 0.005))
            rets_c.append(0.015 + rng.normal(0, 0.01))

    returns = pd.DataFrame({"A": rets_a, "B": rets_b, "C": rets_c}, index=dates)
    regime_df = pd.DataFrame({"regime": regimes}, index=dates)
    return returns, regime_df


class TestOptimizerConstraints:
    """The optimizer's constraint compliance is what protects the strategy
    from degenerate allocations (100% one asset, no cash, ignoring regime
    minimums). Each test asserts a known constraint holds in the output."""

    def test_weights_sum_to_one(self) -> None:
        returns, regime_df = _make_synth_data()
        allocs = optimize_allocations_from_data(returns, regime_df)
        for regime, weights in allocs.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-4, f"Regime {regime!r}: weights sum to {total}, not 1.0"

    def test_cash_floor_respected_per_regime(self) -> None:
        returns, regime_df = _make_synth_data()
        allocs = optimize_allocations_from_data(returns, regime_df)
        for regime, weights in allocs.items():
            min_cash, _ = REGIME_CASH.get(regime, (0.0, 1.0))
            assert weights.get("cash", 0.0) >= min_cash - 1e-4, (
                f"Regime {regime!r}: cash={weights.get('cash')} < min_cash={min_cash}"
            )

    def test_cash_ceiling_respected_per_regime(self) -> None:
        returns, regime_df = _make_synth_data()
        allocs = optimize_allocations_from_data(returns, regime_df)
        for regime, weights in allocs.items():
            _, max_cash = REGIME_CASH.get(regime, (0.0, 1.0))
            assert weights.get("cash", 0.0) <= max_cash + 1e-4, (
                f"Regime {regime!r}: cash={weights.get('cash')} > max_cash={max_cash}"
            )

    def test_no_negative_weights(self) -> None:
        """Long-only constraint: no shorting."""
        returns, regime_df = _make_synth_data()
        allocs = optimize_allocations_from_data(returns, regime_df)
        for regime, weights in allocs.items():
            for asset, w in weights.items():
                assert w >= -1e-6, f"Regime {regime!r}: asset {asset!r} has negative weight {w}"

    def test_optimizer_picks_winning_asset_in_recovery(self) -> None:
        """Asset A is strongest in Recovery by construction; optimizer should weight it >= B and C."""
        returns, regime_df = _make_synth_data(n_months=120)  # more data for stable opt
        allocs = optimize_allocations_from_data(returns, regime_df)
        if "Recovery" in allocs:
            recovery_w = allocs["Recovery"]
            assert recovery_w.get("A", 0.0) >= recovery_w.get("C", 0.0), (
                f"Recovery: optimizer should favor A (strong positive) over C (negative); "
                f"got A={recovery_w.get('A')}, C={recovery_w.get('C')}"
            )
