"""Known-answer tests for src.backtest.engine internal helpers.

Targets the bug classes the README's "Methodology and known caveats" section
flags as silent edge-inflation risks: weight blending math, regime smoothing
off-by-ones, equal-weight normalization.

Each test constructs data with a hand-computable expected answer and asserts
the helper matches. A test that just "exercises code" without a known answer
would pass even after a regression -- by design these don't.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import _blend_alloc, _equal_weight_alloc, _smooth_regime_labels

# ---------------------------------------------------------------------------
# _blend_alloc: weight blending between risk-on and risk-off allocations
# ---------------------------------------------------------------------------


class TestBlendAlloc:
    """The single most-called function in the live path. A sign-flip or
    alpha-vs-(1-alpha) bug here silently changes every weight every month."""

    def test_alpha_zero_returns_risk_off(self) -> None:
        """alpha=0 means fully risk-off; output should equal w_off (after re-normalization)."""
        w_off = {"SPY": 0.2, "IEF": 0.8}
        w_on = {"SPY": 0.9, "IEF": 0.1}
        out = _blend_alloc(w_off, w_on, alpha=0.0, assets=["SPY", "IEF"])
        # Both w_off and w_on already sum to 1, so renormalization is a no-op
        assert out["SPY"] == pytest.approx(0.2, abs=1e-9)
        assert out["IEF"] == pytest.approx(0.8, abs=1e-9)

    def test_alpha_one_returns_risk_on(self) -> None:
        """alpha=1 means fully risk-on; output should equal w_on."""
        w_off = {"SPY": 0.2, "IEF": 0.8}
        w_on = {"SPY": 0.9, "IEF": 0.1}
        out = _blend_alloc(w_off, w_on, alpha=1.0, assets=["SPY", "IEF"])
        assert out["SPY"] == pytest.approx(0.9, abs=1e-9)
        assert out["IEF"] == pytest.approx(0.1, abs=1e-9)

    def test_alpha_half_is_midpoint(self) -> None:
        """alpha=0.5 is the arithmetic mean of w_off and w_on."""
        w_off = {"SPY": 0.2, "IEF": 0.8}
        w_on = {"SPY": 0.9, "IEF": 0.1}
        out = _blend_alloc(w_off, w_on, alpha=0.5, assets=["SPY", "IEF"])
        # midpoint: SPY=(0.2+0.9)/2=0.55, IEF=(0.8+0.1)/2=0.45. Both sum to 1.0 already.
        assert out["SPY"] == pytest.approx(0.55, abs=1e-9)
        assert out["IEF"] == pytest.approx(0.45, abs=1e-9)

    def test_output_sums_to_one(self) -> None:
        """Re-normalization guarantee: any blend should sum to 1.0."""
        w_off = {"SPY": 0.3, "IEF": 0.5, "GLD": 0.2}
        w_on = {"SPY": 0.7, "IEF": 0.1, "GLD": 0.2}
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out = _blend_alloc(w_off, w_on, alpha=alpha, assets=["SPY", "IEF", "GLD"])
            assert abs(sum(out.values()) - 1.0) < 1e-9, (
                f"alpha={alpha}: output sums to {sum(out.values())}, not 1.0"
            )

    def test_alpha_clamped_above_one(self) -> None:
        """alpha > 1 should be clamped to 1 (not extrapolate past risk-on)."""
        w_off = {"SPY": 0.2}
        w_on = {"SPY": 0.9}
        out = _blend_alloc(w_off, w_on, alpha=1.5, assets=["SPY"])
        assert out["SPY"] == pytest.approx(1.0, abs=1e-9)  # single-asset normalizes to 1.0

    def test_alpha_clamped_below_zero(self) -> None:
        """alpha < 0 should be clamped to 0."""
        w_off = {"SPY": 0.2}
        w_on = {"SPY": 0.9}
        out = _blend_alloc(w_off, w_on, alpha=-0.5, assets=["SPY"])
        assert out["SPY"] == pytest.approx(1.0, abs=1e-9)  # single-asset normalizes to 1.0


# ---------------------------------------------------------------------------
# _equal_weight_alloc: the default fallback
# ---------------------------------------------------------------------------


class TestEqualWeightAlloc:
    def test_risk_on_sleeve_only(self) -> None:
        """Custom risk-on sleeve of 4 assets -> each gets 1/4=0.25."""
        out = _equal_weight_alloc(
            assets=["A", "B", "C", "D", "E"],
            is_risk_on=True,
            risk_on_sleeve=["A", "B", "C", "D"],
            risk_off_sleeve=["E"],
        )
        for a in ["A", "B", "C", "D"]:
            assert out[a] == pytest.approx(0.25, abs=1e-9)
        assert out["E"] == 0.0

    def test_risk_off_sleeve_only(self) -> None:
        """is_risk_on=False -> risk-off sleeve gets equal weight; risk-on zero."""
        out = _equal_weight_alloc(
            assets=["A", "B", "C"],
            is_risk_on=False,
            risk_on_sleeve=["A"],
            risk_off_sleeve=["B", "C"],
        )
        assert out["A"] == 0.0
        assert out["B"] == pytest.approx(0.5, abs=1e-9)
        assert out["C"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# _smooth_regime_labels: the rolling-mode smoother
# ---------------------------------------------------------------------------


class TestSmoothRegimeLabels:
    """A regime smoothing off-by-one or window-direction bug would either
    leak future regime info backward (lookahead) or fail to smooth single-month
    flips. Both meaningfully change reported edge."""

    def test_isolated_one_month_flip_is_smoothed(self) -> None:
        """Sequence [Recovery, Recovery, Stagflation, Recovery, Recovery]
        with window=3 should smooth the middle flip back to Recovery."""
        idx = pd.date_range("2024-01-31", periods=5, freq="ME")
        df = pd.DataFrame(
            {"regime": ["Recovery", "Recovery", "Stagflation", "Recovery", "Recovery"]},
            index=idx,
        )
        out = _smooth_regime_labels(df, window=3)
        # Month 3 (index=2, the Stagflation): window includes months 1,2,3 (R,R,S).
        # Mode of (R,R,S) = R. So Stagflation gets smoothed to Recovery.
        assert out.iloc[2]["regime"] == "Recovery", (
            f"Isolated 1-month Stagflation flip should be smoothed; got {out.iloc[2]['regime']!r}"
        )

    def test_sustained_regime_change_preserved(self) -> None:
        """A real 3+ month regime change should NOT be smoothed away."""
        idx = pd.date_range("2024-01-31", periods=6, freq="ME")
        df = pd.DataFrame(
            {
                "regime": [
                    "Recovery",
                    "Recovery",
                    "Stagflation",
                    "Stagflation",
                    "Stagflation",
                    "Stagflation",
                ]
            },
            index=idx,
        )
        out = _smooth_regime_labels(df, window=3)
        # By month 5 (4 consecutive Stagflations), the mode of any 3-window must be Stagflation.
        assert out.iloc[-1]["regime"] == "Stagflation", (
            f"Sustained Stagflation should persist; got {out.iloc[-1]['regime']!r}"
        )

    def test_window_one_is_no_op(self) -> None:
        """Window of 1 means no smoothing at all (mode of 1 = the value itself)."""
        idx = pd.date_range("2024-01-31", periods=4, freq="ME")
        df = pd.DataFrame(
            {"regime": ["Recovery", "Stagflation", "Recovery", "Contraction"]},
            index=idx,
        )
        out = _smooth_regime_labels(df, window=1)
        for i in range(len(df)):
            assert out.iloc[i]["regime"] == df.iloc[i]["regime"]
