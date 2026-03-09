"""Run walk-forward evaluation. Requires regime classification and data first."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.walk_forward import run_walk_forward_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stagflation-override", action="store_true",
                        help="Use risk_on blending in Stagflation (baseline behavior)")
    parser.add_argument("--stagflation-risk-on-cap", action="store_true",
                        help="Cap risk_on at 0.2 when regime==Stagflation (experiment)")
    parser.add_argument("--stagflation-cap-value", type=float, default=0.2,
                        help="Max risk_on when Stagflation (default 0.2)")
    args = parser.parse_args()
    use_override = not args.no_stagflation_override and not args.stagflation_risk_on_cap
    use_cap = args.stagflation_risk_on_cap
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = run_walk_forward_evaluation(
        min_train_months=60,
        test_months=12,
        expanding=True,
        use_stagflation_override=use_override,
        use_stagflation_risk_on_cap=use_cap,
        stagflation_risk_on_cap=args.stagflation_cap_value,
    )
    if df.empty:
        sys.exit(1)
    print("\n[WALK-FORWARD] Sample results (first 3 segments):")
    print(df.head(3).to_string())
    print("\n[WALK-FORWARD] OVERALL (averaged across segments):")
    overall = df[df["segment"] == "OVERALL"]
    if not overall.empty:
        print(overall.T.to_string())
