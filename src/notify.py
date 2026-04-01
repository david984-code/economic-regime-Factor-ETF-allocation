"""Post-pipeline SMS notifier via Twilio.

Reads today's state from allocations.db and sends a ~160-char summary.
Requires env vars: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
                   TWILIO_FROM_NUMBER, NOTIFY_TO_NUMBER.
"""

from __future__ import annotations

import logging
import os

from src.utils.database import Database

logger = logging.getLogger(__name__)

SMS_MAX_LEN = 160


def _get_twilio_config() -> dict[str, str]:
    """Read Twilio credentials from environment."""
    keys = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "NOTIFY_TO_NUMBER",
    ]
    config: dict[str, str] = {}
    missing: list[str] = []
    for k in keys:
        val = os.environ.get(k, "")
        if not val:
            missing.append(k)
        config[k] = val
    if missing:
        raise OSError(
            f"Missing Twilio env vars: {', '.join(missing)}. "
            "Set them in .env or environment."
        )
    return config


def build_sms_body() -> str:
    """Query allocations.db and build a concise SMS body.

    Format (≤160 chars):
        Regime: Recovery | risk_on: 0.72
        Top: SPY 28% MTUM 18% QUAL 15%
        Sharpe: 0.51 | CAGR: 7.5% | DD: -7.5%
    """
    db = Database()
    try:
        # Latest regime
        regime_df = db.load_regime_labels()
        if regime_df.empty:
            return "Pipeline ran but no regime data found."
        latest = regime_df.iloc[-1]
        regime = latest["regime"]
        risk_on = float(latest["risk_on"])

        # Latest allocations for current regime
        allocations = db.load_optimal_allocations()
        regime_alloc = allocations.get(regime, {})
        top_3 = sorted(
            ((a, w) for a, w in regime_alloc.items() if a != "cash"),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        top_str = " ".join(f"{a} {w:.0%}" for a, w in top_3)

        # Latest backtest
        bt = db.get_latest_backtest_results()
        if bt:
            p = bt["portfolio"]
            metrics_str = (
                f"Sharpe:{p['Sharpe']:.2f} "
                f"CAGR:{p['CAGR']:.1%} "
                f"DD:{p['Max Drawdown']:.1%}"
            )
        else:
            metrics_str = "No backtest"

        body = f"{regime}|ro:{risk_on:.2f}\n{top_str}\n{metrics_str}"

        if len(body) > SMS_MAX_LEN:
            body = body[:SMS_MAX_LEN]
        return body
    finally:
        db.close()


def send_sms(body: str) -> str:
    """Send SMS via Twilio. Returns the message SID."""
    from twilio.rest import Client  # type: ignore[import-untyped]

    config = _get_twilio_config()
    client = Client(config["TWILIO_ACCOUNT_SID"], config["TWILIO_AUTH_TOKEN"])
    message = client.messages.create(
        body=body,
        from_=config["TWILIO_FROM_NUMBER"],
        to=config["NOTIFY_TO_NUMBER"],
    )
    logger.info("SMS sent: SID=%s", message.sid)
    return str(message.sid)


def notify() -> None:
    """Build and send the daily SMS notification.

    Logs warnings on failure but does not raise — this is a non-critical step.
    """
    try:
        body = build_sms_body()
        logger.info("SMS body (%d chars): %s", len(body), body)
        send_sms(body)
    except OSError:
        logger.warning("Twilio not configured — skipping SMS notification.")
    except Exception as e:
        logger.warning("SMS notification failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    notify()
