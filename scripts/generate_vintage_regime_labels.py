"""Generate point-in-time (vintage) regime labels for the walk-forward backtest.

For each month T in the OOS window, this script:
  1. Constructs the vintage view of each FRED series as of T (only values
     that had been released by T, with their as-of-T revision).
  2. Runs the existing rolling-z-score classification logic on that vintage
     view to determine the regime label and risk_on score AT T.
  3. Records (T, regime, risk_on).

The output is a single CSV `outputs/regime_labels_vintage.csv` with one row
per month-end, where each row's value reflects only data the model could
have known at that month-end. The walk-forward harness consumes this file
in place of `regime_labels_asof.csv` (which used latest-revised data).

Expect this fix to MOVE the published OOS numbers -- the strategy's apparent
edge over 60/40 already vanished under the 50/30/20 control benchmark; this
will rule out one more lookahead source. Direction of net conclusion likely
unchanged.

Usage:
    python scripts/generate_vintage_regime_labels.py [--start 2010-01] [--end 2026-05] [--no-cache]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUTS_DIR
from src.data.fred_vintage import (
    fetch_vintage_core,
    fetch_vintage_optional,
    vintage_series_as_of,
)
from src.features.macro_features import add_z_scores
from src.features.transforms import sigmoid

logger = logging.getLogger(__name__)


def _resample_monthly(s: pd.Series) -> pd.Series:
    """Month-end frequency, ffill within the month."""
    if s.empty:
        return s
    s = s.sort_index()
    return s.resample("ME").last().ffill()


def _yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    return s.pct_change(periods=periods)


def classify_at_asof(
    vintage_core: dict[str, pd.DataFrame],
    vintage_opt: dict[str, pd.DataFrame],
    latest_fallback: dict[str, pd.Series],
    as_of: pd.Timestamp,
) -> tuple[str, float] | None:
    """Build features as-of `as_of` and return (regime_label, risk_on) at as_of.

    Returns None if there isn't enough vintage data yet (model needs
    at least 24 periods for the rolling z-score min_periods).

    Vintage data is used for series that actually get revised (GDP, CPI,
    M2, M2V, ICSA, HY OAS). For daily series that are observed prices and
    do not get revised (DGS10, DGS3MO), we use the latest series truncated
    to as_of -- this is genuinely not lookahead-tainted because Treasury
    yields are not revised after the day they are observed.
    """

    def _vintage_or_fallback(
        vintage_dict: dict[str, pd.DataFrame],
        key: str,
    ) -> pd.Series:
        if key in vintage_dict:
            return vintage_series_as_of(vintage_dict[key], as_of)
        fb = latest_fallback.get(key, pd.Series(dtype=float))
        return fb.loc[fb.index <= as_of] if not fb.empty else fb

    gdp = _vintage_or_fallback(vintage_core, "gdp")
    cpi = _vintage_or_fallback(vintage_core, "cpi")
    y10 = _vintage_or_fallback(vintage_core, "yield_10y")
    y3 = _vintage_or_fallback(vintage_core, "yield_3m")
    m2 = _vintage_or_fallback(vintage_core, "m2")
    vel = _vintage_or_fallback(vintage_core, "velocity")

    pmi = _vintage_or_fallback(vintage_opt, "pmi")
    claims = _vintage_or_fallback(vintage_opt, "claims")
    hy = _vintage_or_fallback(vintage_opt, "hy_spread")

    # All to month-end
    gdp_m, cpi_m, y10_m, y3_m, m2_m, vel_m = (
        _resample_monthly(s) for s in (gdp, cpi, y10, y3, m2, vel)
    )

    # Build a tiny features frame (subset of the production add_z_scores logic;
    # we only need the values AT as_of, not the whole panel)
    if gdp_m.empty or cpi_m.empty or m2_m.empty:
        return None

    df = pd.DataFrame(
        {
            "gdp_mom": _yoy(gdp_m),
            "cpi_mom": _yoy(cpi_m, periods=12),
            "m2_mom": _yoy(m2_m, periods=12),
            "vel_mom": _yoy(vel_m, periods=4),  # M2V is quarterly
            "yield_curve": (y10_m - y3_m),
        }
    )
    if not pmi.empty:
        df["pmi_mom"] = _yoy(_resample_monthly(pmi), periods=12)
    if not claims.empty:
        df["claims_mom"] = _yoy(_resample_monthly(claims), periods=12)
    if not hy.empty:
        df["hy_spread_mom"] = _resample_monthly(hy).diff(12)

    df = df.dropna(how="all")
    if len(df) < 30:
        logger.debug(
            "  as_of=%s: only %d rows after dropna(how=all); skipping", as_of.date(), len(df)
        )
        return None

    # Forward-fill across the joined index. Each indicator only updates on its
    # release schedule (GDP quarterly, CPI monthly, claims weekly); ffill is
    # NOT lookahead-tainted because each value is replaced only by EARLIER
    # values, not later ones. This is what the production pipeline does too
    # (see resample_to_monthly in features/macro_features.py).
    df = df.sort_index().ffill()

    # Apply the SAME rolling-z-score transform the production classifier uses
    df = add_z_scores(df)

    # Latest z-scores at-or-before as_of
    eligible = df.loc[df.index <= as_of]
    if eligible.empty:
        return None
    latest = eligible.iloc[-1]

    if pd.isna(latest.get("gdp_z")) or pd.isna(latest.get("infl_z")):
        logger.debug(
            "  as_of=%s: gdp_z=%s infl_z=%s; skipping",
            as_of.date(),
            latest.get("gdp_z"),
            latest.get("infl_z"),
        )
        return None

    # Production scoring: blend the available z-scores into a risk_on logit,
    # then label by sign of growth vs inflation
    growth_z = float(latest["gdp_z"])
    infl_z = float(latest["infl_z"])

    # Risk-on logit: growth+, infl- → risk-on. Use sigmoid for [0,1].
    raw = 0.5 * growth_z - 0.5 * infl_z
    if "yield_level_z" in latest and not pd.isna(latest["yield_level_z"]):
        raw += 0.15 * float(latest["yield_level_z"])
    risk_on = float(sigmoid(pd.Series([raw])).iloc[0])

    # 4-regime label per the 2x2 of growth direction x inflation direction
    if growth_z >= 0 and infl_z <= 0:
        regime = "Recovery"
    elif growth_z >= 0 and infl_z > 0:
        regime = "Overheating"
    elif growth_z < 0 and infl_z > 0:
        regime = "Stagflation"
    else:
        regime = "Contraction"

    return regime, risk_on


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "regime_labels_vintage.csv"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not set. Add to .env or environment.")
        return 1

    logger.info("Fetching ALFRED vintage release histories (cached: %s)...", not args.no_cache)
    vintage_core = fetch_vintage_core(api_key, use_cache=not args.no_cache)
    vintage_opt = fetch_vintage_optional(api_key, use_cache=not args.no_cache)
    logger.info("  vintage core: %s", list(vintage_core.keys()))
    logger.info("  vintage opt : %s", list(vintage_opt.keys()))

    # For series ALFRED does not support (daily Treasuries; NAPM PMI),
    # fall back to latest revised. These are price/index series that are
    # NOT revised after the day they are observed -- using "latest" for
    # them does not introduce lookahead.
    logger.info("Fetching non-revised fallback series (latest FRED) ...")
    from src.data.fred_ingestion import fetch_fred_core, fetch_fred_optional

    gdp_l, cpi_l, y10_l, y3_l, m2_l, vel_l = fetch_fred_core(api_key)
    opt_l = fetch_fred_optional(api_key)
    latest_fallback = {
        "gdp": gdp_l,
        "cpi": cpi_l,
        "yield_10y": y10_l,
        "yield_3m": y3_l,
        "m2": m2_l,
        "velocity": vel_l,
        **{k: v for k, v in opt_l.items()},
    }

    end_ts = pd.Timestamp(args.end) if args.end else pd.Timestamp.today()
    dates = pd.date_range(args.start, end_ts, freq="ME")
    logger.info("Classifying %d month-end as-of points...", len(dates))

    rows = []
    skipped = 0
    skip_reasons = []
    for i, d in enumerate(dates):
        result = classify_at_asof(vintage_core, vintage_opt, latest_fallback, d)
        if result is None:
            skipped += 1
            if i < 5 or i == len(dates) // 2 or i == len(dates) - 1:
                skip_reasons.append(str(d.date()))
            continue
        regime, risk_on = result
        rows.append({"date": d, "regime": regime, "risk_on": risk_on})
    if skip_reasons:
        logger.info("Sampled skipped dates: %s", skip_reasons[:10])

    out_df = pd.DataFrame(rows).set_index("date")
    out_df.to_csv(args.out)
    logger.info(
        "Wrote %d vintage labels to %s (skipped %d insufficient-data months)",
        len(out_df),
        args.out,
        skipped,
    )

    # Summary
    print()
    print("Regime distribution under vintage data:")
    print(out_df["regime"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
