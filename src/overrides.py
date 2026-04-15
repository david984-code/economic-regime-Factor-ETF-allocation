"""
Market-condition override filters for the regime allocator.

All functions are pure (no side-effects, no global state) so they are safe
to call from both the live backtest and from test harnesses.

Bias-safety notes
-----------------
* VIX override uses the close of the *rebalance date*.  Because the existing
  backtest sets weights on the first trading day of the month and uses those
  same-day closes for that day's return (matching the existing pipeline design),
  using same-day VIX is consistent and introduces no additional lookahead.
* MA filter computes the 200-day average from prices[:date] — strictly trailing.
* HYG credit trigger uses trailing N-day returns — strictly trailing, no lookahead.
* Momentum tilt uses trailing N-day returns — strictly trailing, no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

_CONFIG_IMPORTS = [
    "VIX_THRESHOLD",
    "VIX_RISK_ON_CAP",
    "MA_LOOKBACK",
    "MA_EQUITY_CAP",
    "VIX_INTRAMONTH_THRESHOLD",
    "SPY_WEEKLY_DROP_THRESHOLD",
    "ENABLE_CREDIT_TRIGGER",
    "HYG_LOOKBACK_DAYS",
    "HYG_STRESS_THRESHOLD",
    "HYG_RECOVERY_THRESHOLD",
    "HYG_STRESS_MEMORY_DAYS",
    "ENABLE_MOMENTUM_TILT",
    "MOMENTUM_LOOKBACK_DAYS",
    "MOMENTUM_STRENGTH",
    "MOMENTUM_MAX_TILT",
    "ENABLE_CREDIT_ZSCORE",
    "CREDIT_ZSCORE_LOOKBACK",
    "CREDIT_ZSCORE_STRESS",
    "CREDIT_ZSCORE_RECOVERY",
    "USE_CREDIT_STANDALONE",
]

try:
    from src.config import (
        CREDIT_ZSCORE_LOOKBACK,
        CREDIT_ZSCORE_RECOVERY,
        CREDIT_ZSCORE_STRESS,
        ENABLE_CREDIT_TRIGGER,
        ENABLE_CREDIT_ZSCORE,
        ENABLE_MOMENTUM_TILT,
        HYG_LOOKBACK_DAYS,
        HYG_RECOVERY_THRESHOLD,
        HYG_STRESS_MEMORY_DAYS,
        HYG_STRESS_THRESHOLD,
        MA_EQUITY_CAP,
        MA_LOOKBACK,
        MOMENTUM_LOOKBACK_DAYS,
        MOMENTUM_MAX_TILT,
        MOMENTUM_STRENGTH,
        SPY_WEEKLY_DROP_THRESHOLD,
        USE_CREDIT_STANDALONE,
        VIX_INTRAMONTH_THRESHOLD,
        VIX_RISK_ON_CAP,
        VIX_THRESHOLD,
    )
except ImportError:
    from config import (  # noqa: F401 (direct-run fallback)
        CREDIT_ZSCORE_LOOKBACK,
        CREDIT_ZSCORE_RECOVERY,
        CREDIT_ZSCORE_STRESS,
        ENABLE_CREDIT_TRIGGER,
        ENABLE_CREDIT_ZSCORE,
        ENABLE_MOMENTUM_TILT,
        HYG_LOOKBACK_DAYS,
        HYG_RECOVERY_THRESHOLD,
        HYG_STRESS_MEMORY_DAYS,
        HYG_STRESS_THRESHOLD,
        MA_EQUITY_CAP,
        MA_LOOKBACK,
        MOMENTUM_LOOKBACK_DAYS,
        MOMENTUM_MAX_TILT,
        MOMENTUM_STRENGTH,
        SPY_WEEKLY_DROP_THRESHOLD,
        USE_CREDIT_STANDALONE,
        VIX_INTRAMONTH_THRESHOLD,
        VIX_RISK_ON_CAP,
        VIX_THRESHOLD,
    )


# ── Data fetch ────────────────────────────────────────────────────────────────


def fetch_market_filters(start: str, end: str) -> pd.DataFrame:
    """Download daily VIX, SPY, HYG, and LQD closes needed by the override filters.

    Parameters
    ----------
    start, end : str  ISO date strings passed to yfinance.

    Returns
    -------
    DataFrame  columns=['VIX_close', 'SPY_close', 'HYG_close', 'HYG_10d_ret',
               'LQD_close', 'HYG_LQD_zscore'], DatetimeIndex of trading days.
    """
    raw_vix = yf.download(
        "^VIX", start=start, end=end, progress=False, auto_adjust=True
    )
    raw_spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    raw_hyg = yf.download("HYG", start=start, end=end, progress=False, auto_adjust=True)
    raw_lqd = yf.download("LQD", start=start, end=end, progress=False, auto_adjust=True)

    if raw_vix.empty:
        raise RuntimeError(
            "VIX data unavailable from yfinance (ticker: ^VIX). "
            "Check your network connection or yfinance version. "
            "Set USE_VIX_OVERRIDE=False in src/config.py to disable this check."
        )
    if raw_spy.empty:
        raise RuntimeError(
            "SPY data unavailable from yfinance (ticker: SPY). "
            "Check your network connection. "
            "Set USE_MA_FILTER=False in src/config.py to disable this check."
        )

    def _close(raw: pd.DataFrame, ticker: str) -> pd.Series:
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"][ticker]
        return raw["Close"]

    cols = {
        "VIX_close": _close(raw_vix, "^VIX"),
        "SPY_close": _close(raw_spy, "SPY"),
    }

    if not raw_hyg.empty:
        hyg_close = _close(raw_hyg, "HYG")
        cols["HYG_close"] = hyg_close
        cols["HYG_10d_ret"] = hyg_close.pct_change(HYG_LOOKBACK_DAYS)

        if not raw_lqd.empty:
            lqd_close = _close(raw_lqd, "LQD")
            cols["LQD_close"] = lqd_close
            ratio = hyg_close / lqd_close
            ratio_ma = ratio.rolling(
                CREDIT_ZSCORE_LOOKBACK, min_periods=CREDIT_ZSCORE_LOOKBACK // 2
            ).mean()
            ratio_std = ratio.rolling(
                CREDIT_ZSCORE_LOOKBACK, min_periods=CREDIT_ZSCORE_LOOKBACK // 2
            ).std()
            cols["HYG_LQD_zscore"] = (ratio - ratio_ma) / ratio_std.replace(0, np.nan)

    df = pd.DataFrame(cols)
    return df.dropna(how="all")


# ── VIX override ─────────────────────────────────────────────────────────────


def apply_vix_override(
    alpha: float,
    mkt_filters: pd.DataFrame,
    date: pd.Timestamp,
    threshold: float = VIX_THRESHOLD,
    cap: float = VIX_RISK_ON_CAP,
) -> float:
    """Cap risk_on `alpha` at `cap` when VIX exceeds `threshold`.

    If `date` is not in `mkt_filters` (e.g. market holiday) the original
    alpha is returned unchanged — we do NOT silently assume VIX is low.
    The caller is responsible for ensuring mkt_filters covers the date range.
    """
    if date not in mkt_filters.index:
        return alpha
    vix_val = mkt_filters.loc[date, "VIX_close"]
    if pd.isna(vix_val):
        return alpha
    if float(vix_val) > threshold:
        return min(float(alpha), cap)
    return float(alpha)


# ── 200-day MA filter ─────────────────────────────────────────────────────────


def apply_ma_filter(
    weights: dict,
    spy_prices_to_date: pd.Series,
    equity_tickers: list[str],
    lookback: int = MA_LOOKBACK,
    equity_cap: float = MA_EQUITY_CAP,
) -> dict:
    """Cap total equity weight at `equity_cap` when SPY is below its MA.

    Parameters
    ----------
    weights            : dict asset->weight (must sum to ~1, may include 'cash').
    spy_prices_to_date : SPY close price series through the rebalance date
                         (trailing data only, no lookahead).
    equity_tickers     : list of tickers that count as equity for the cap.
    lookback           : trading days for the moving average window.
    equity_cap         : maximum allowed total equity weight when triggered.

    Returns
    -------
    New weight dict.  If the filter does not fire, `weights` is returned
    unchanged (same object reference).
    """
    spy_hist = spy_prices_to_date.dropna()
    if len(spy_hist) < lookback:
        return weights  # insufficient history — filter cannot fire yet

    spy_today = float(spy_hist.iloc[-1])
    spy_ma = float(spy_hist.iloc[-lookback:].mean())

    if spy_today >= spy_ma:
        return weights  # trend is up, filter does not fire

    # Trend filter fired — enforce equity cap
    total_equity = sum(float(weights.get(t, 0.0)) for t in equity_tickers)
    if total_equity <= equity_cap + 1e-9:
        return weights  # already within cap

    scale = equity_cap / total_equity
    equity_cut = total_equity - equity_cap  # weight freed up

    new_w = {k: float(v) for k, v in weights.items()}
    for t in equity_tickers:
        new_w[t] = float(weights.get(t, 0.0)) * scale

    # Freed weight goes to cash (defensive)
    new_w["cash"] = float(weights.get("cash", 0.0)) + equity_cut

    # Renormalize to ensure weights sum to 1.0 exactly
    total = sum(new_w.values())
    if total > 0:
        new_w = {k: v / total for k, v in new_w.items()}

    return new_w


# ── Intramonth trigger ────────────────────────────────────────────────────────


def check_intramonth_trigger(
    mkt_filters: pd.DataFrame,
    date: pd.Timestamp,
    vix_threshold: float = VIX_INTRAMONTH_THRESHOLD,
    spy_drop_threshold: float = SPY_WEEKLY_DROP_THRESHOLD,
) -> tuple[bool, str]:
    """Evaluate whether a mid-month rebalance should fire on `date` (a Friday).

    Two independent conditions — either alone is sufficient to trigger:
      1. VIX close > ``vix_threshold`` (default 30)
      2. SPY Friday-to-Friday weekly return < -``spy_drop_threshold`` (default 3 %)

    Both conditions use data available at ``date``'s close — no lookahead.

    Parameters
    ----------
    mkt_filters      : DataFrame with 'VIX_close' and 'SPY_close' columns.
    date             : The Friday close being evaluated.
    vix_threshold    : VIX level that fires the trigger (default from config).
    spy_drop_threshold : Absolute weekly return drop threshold (default from config).

    Returns
    -------
    (triggered: bool, reason: str)

    Raises
    ------
    ValueError
        If VIX data is missing or NaN for ``date``.  Per spec, we never
        silently fall back — the caller must handle or disable the trigger.
    """
    if date not in mkt_filters.index:
        raise ValueError(
            f"Market data missing for {date.date()} in mkt_filters. "
            "Ensure fetch_market_filters() covers this date range, or set "
            "ENABLE_INTRAMONTH_TRIGGER=False in src/config.py."
        )

    # ── Credit standalone mode: skip VIX, use z-score + HYG momentum ─────
    if USE_CREDIT_STANDALONE and ENABLE_CREDIT_ZSCORE:
        if "HYG_LQD_zscore" in mkt_filters.columns:
            triggered, reason = check_credit_zscore_trigger(mkt_filters, date)
            if triggered:
                return True, reason
        if ENABLE_CREDIT_TRIGGER and "HYG_10d_ret" in mkt_filters.columns:
            triggered, reason = check_credit_trigger(mkt_filters, date)
            if triggered:
                return True, reason
        return False, "no credit trigger conditions met"

    # ── VIX check (legacy mode when USE_CREDIT_STANDALONE=False) ──────────
    vix_val = mkt_filters.loc[date, "VIX_close"]
    if pd.isna(vix_val):
        raise ValueError(
            f"VIX close is NaN for {date.date()} — stale or missing data. "
            "Set ENABLE_INTRAMONTH_TRIGGER=False in src/config.py to disable."
        )

    if float(vix_val) > vix_threshold:
        return True, f"VIX={float(vix_val):.1f} > threshold {vix_threshold}"

    # ── SPY weekly drop check ────────────────────────────────────────────────
    # NOTE: SPY weekly drop trigger is effectively disabled in production config
    # (SPY_WEEKLY_DROP_THRESHOLD = 0.99). Sensitivity sweep showed 4 false
    # positives/year in 2023-2024, all triggered into recoveries — unacceptable.
    # VIX-only trigger is retained. To re-enable, set SPY_WEEKLY_DROP_THRESHOLD
    # < 0.10 in src/config.py.  Do NOT delete this code — it remains gated by
    # the config value.
    spy_series = mkt_filters["SPY_close"].loc[:date].dropna()
    spy_weekly = spy_series.resample("W-FRI").last().dropna()

    if len(spy_weekly) < 2:
        return False, "insufficient SPY weekly history"

    prev_close = float(spy_weekly.iloc[-2])
    this_close = float(spy_weekly.iloc[-1])

    if prev_close <= 0:
        return False, "invalid prior week SPY close"

    weekly_ret = this_close / prev_close - 1.0
    if weekly_ret < -spy_drop_threshold:
        return True, (
            f"SPY weekly return {weekly_ret:.2%} < -{spy_drop_threshold:.0%} threshold"
        )

    # ── HYG credit stress check (momentum-based) ────────────────────────────
    if ENABLE_CREDIT_TRIGGER and "HYG_10d_ret" in mkt_filters.columns:
        triggered, reason = check_credit_trigger(mkt_filters, date)
        if triggered:
            return True, reason

    # ── HYG/LQD z-score check (not in standalone mode — that's handled above)
    if (
        ENABLE_CREDIT_ZSCORE
        and not USE_CREDIT_STANDALONE
        and "HYG_LQD_zscore" in mkt_filters.columns
    ):
        triggered, reason = check_credit_zscore_trigger(mkt_filters, date)
        if triggered:
            return True, reason

    return False, "no trigger conditions met"


# ── HYG credit stress trigger ────────────────────────────────────────────────


def check_credit_trigger(
    mkt_filters: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = HYG_LOOKBACK_DAYS,
    stress_threshold: float = HYG_STRESS_THRESHOLD,
    recovery_threshold: float = HYG_RECOVERY_THRESHOLD,
    memory_days: int = HYG_STRESS_MEMORY_DAYS,
) -> tuple[bool, str]:
    """Evaluate whether a credit-stress mid-month rebalance should fire.

    Two conditions (either fires):
      1. HYG trailing return < stress_threshold  (credit deterioration)
      2. HYG trailing return > recovery_threshold AND HYG was recently stressed
         (state-transition recovery — not a level trigger)

    Uses only data available at ``date`` close — no lookahead.
    """
    if date not in mkt_filters.index:
        return False, "no HYG data"
    if "HYG_10d_ret" not in mkt_filters.columns:
        return False, "HYG_10d_ret column missing"

    hyg_ret = mkt_filters.loc[date, "HYG_10d_ret"]
    if pd.isna(hyg_ret):
        return False, "HYG return NaN"
    hyg_ret = float(hyg_ret)

    if hyg_ret < stress_threshold:
        return True, (
            f"HYG {lookback}d return {hyg_ret:.2%} < "
            f"{stress_threshold:.0%} (credit stress)"
        )

    if hyg_ret > recovery_threshold:
        trailing = mkt_filters["HYG_10d_ret"].loc[:date].tail(memory_days)
        if (trailing < stress_threshold).any():
            return True, (
                f"HYG {lookback}d return {hyg_ret:.2%} — "
                f"recovery after recent credit stress"
            )

    return False, "no credit trigger"


# ── HYG/LQD z-score credit spread trigger ────────────────────────────────────


def check_credit_zscore_trigger(
    mkt_filters: pd.DataFrame,
    date: pd.Timestamp,
    stress_z: float = CREDIT_ZSCORE_STRESS,
    recovery_z: float = CREDIT_ZSCORE_RECOVERY,
    memory_days: int = HYG_STRESS_MEMORY_DAYS,
) -> tuple[bool, str]:
    """Evaluate HYG/LQD ratio z-score for credit stress / recovery.

    The ratio falls when high-yield underperforms investment-grade (spreads widen).
    A deeply negative z-score signals credit deterioration.

    Walk-forward validated (Exp F): Sharpe +0.070, CAGR +0.93%, 7/10 win rate.
    """
    if date not in mkt_filters.index:
        return False, "no data"
    if "HYG_LQD_zscore" not in mkt_filters.columns:
        return False, "HYG_LQD_zscore column missing"

    z = mkt_filters.loc[date, "HYG_LQD_zscore"]
    if pd.isna(z):
        return False, "z-score NaN"
    z = float(z)

    if z < stress_z:
        return True, f"HYG/LQD z={z:.2f} < {stress_z} (credit stress)"

    if z > recovery_z:
        trailing_z = mkt_filters["HYG_LQD_zscore"].loc[:date].tail(memory_days)
        if (trailing_z < stress_z).any():
            return True, f"HYG/LQD z={z:.2f} — recovery after credit stress"

    return False, "no credit z-score trigger"


# ── Cross-sectional momentum tilt ────────────────────────────────────────────


def apply_momentum_tilt(
    weights: dict,
    prices_df: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = MOMENTUM_LOOKBACK_DAYS,
    strength: float = MOMENTUM_STRENGTH,
    max_tilt: float = MOMENTUM_MAX_TILT,
) -> dict:
    """Tilt portfolio weights toward recent relative outperformers.

    For each non-cash asset with weight > 0, computes trailing ``lookback``-day
    return, cross-sectional z-score, then applies:
        w_new = w_base * (1 + strength * z_momentum)

    Capped so no asset exceeds ``max_tilt`` * its base weight.
    Cash weight is untouched; risky weights renormalise to preserve total.
    Uses only trailing prices — no lookahead.
    """
    w = {k: float(v) for k, v in weights.items()}
    cash_w = w.get("cash", 0.0)

    risky = [a for a in w if a != "cash" and w[a] > 1e-6 and a in prices_df.columns]
    if len(risky) < 2:
        return w

    trailing = prices_df[risky].loc[:date].tail(lookback + 1)
    if len(trailing) < lookback + 1:
        return w

    mom = (trailing.iloc[-1] / trailing.iloc[0] - 1).to_dict()
    vals = np.array([mom[a] for a in risky])
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-10:
        return w

    for a in risky:
        z = (mom[a] - mu) / sigma
        factor = max(1.0 / max_tilt, min(max_tilt, 1.0 + strength * z))
        w[a] *= factor

    risky_new = sum(w[a] for a in risky)
    risky_orig = 1.0 - cash_w
    if risky_new > 0 and risky_orig > 0:
        scale = risky_orig / risky_new
        for a in risky:
            w[a] *= scale

    w["cash"] = cash_w
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w
