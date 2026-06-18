"""Shared execution safety checks and run-guard utilities.

Single source of truth for:
- target-weight integrity checks
- pre-trade order safety limits
- duplicate-run protection markers
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from src.execution.create_orders import OrderPreviewRow, RebalancePreview, validate_target_weights_for_execution

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyConfig:
    allow_leverage: bool = False
    target_weight_sum_tolerance: float = 0.02
    max_weight_delta: float = 0.25
    max_notional_per_order: float = 50_000.0
    max_total_turnover: float = 1.0


def safety_config_from_paper_config(config: dict[str, Any]) -> SafetyConfig:
    rebal = config.get("rebalance") or {}
    exec_cfg = config.get("execution") or {}
    return SafetyConfig(
        allow_leverage=bool(rebal.get("allow_leverage", False)),
        target_weight_sum_tolerance=float(rebal.get("target_weight_sum_tolerance", 0.02)),
        max_weight_delta=float(rebal.get("max_weight_delta", 0.25)),
        max_notional_per_order=float(
            exec_cfg.get("max_notional_per_order", rebal.get("max_notional", 50_000.0))
        ),
        max_total_turnover=float(exec_cfg.get("max_total_turnover", 1.0)),
    )


def pre_trade_checks(
    preview: RebalancePreview,
    *,
    trading_enabled: bool,
    paper_only: bool,
    cfg: SafetyConfig,
) -> None:
    """Raise ValueError if submission is not safe.

    This is the shared replacement for monthly_rebalance_runner._pre_trade_checks.
    """
    if not trading_enabled:
        raise ValueError(
            "Submission refused: trading_enabled is false in config. Set to true to allow paper submission."
        )
    if not paper_only:
        raise ValueError("Submission refused: paper_only must be true. Live trading is not supported.")
    if preview.symbols_missing_price:
        raise ValueError(
            "Submission refused: missing prices for "
            + ", ".join(sorted(preview.symbols_missing_price))
            + ". Provide prices file or broker data."
        )

    validate_target_weights_for_execution(
        preview.target_weights,
        allow_leverage=cfg.allow_leverage,
        tolerance=cfg.target_weight_sum_tolerance,
    )

    for o in preview.proposed_orders:
        if abs(o.weight_delta) > cfg.max_weight_delta:
            raise ValueError(
                f"Submission refused: order {o.side} {o.symbol} weight_delta {o.weight_delta:.4f} "
                f"exceeds max_weight_delta {cfg.max_weight_delta}."
            )
        if o.approximate_dollar > cfg.max_notional_per_order:
            raise ValueError(
                f"Submission refused: order {o.side} {o.symbol} notional ${o.approximate_dollar:.2f} "
                f"exceeds max_notional_per_order {cfg.max_notional_per_order}."
            )
    if preview.turnover_one_way > cfg.max_total_turnover:
        raise ValueError(
            f"Submission refused: turnover_one_way {preview.turnover_one_way:.4f} exceeds "
            f"max_total_turnover {cfg.max_total_turnover:.4f}."
        )


def describe_order_offenders(
    proposed_orders: Iterable[OrderPreviewRow],
    *,
    cfg: SafetyConfig,
) -> dict[str, Any]:
    """Return offender lists without raising (for dry-run reporting)."""
    notional = [
        asdict(o)
        for o in proposed_orders
        if float(o.approximate_dollar) > float(cfg.max_notional_per_order)
    ]
    weight_delta = [
        asdict(o)
        for o in proposed_orders
        if abs(float(o.weight_delta)) > float(cfg.max_weight_delta)
    ]
    return {"notional_offenders": notional, "weight_delta_offenders": weight_delta}


def duplicate_run_refuse(marker_path: Path) -> bool:
    """Return True only if we already submitted real orders today (not dry-run markers)."""
    if not marker_path.exists():
        return False
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if (data.get("date") or "") != datetime.now(UTC).strftime("%Y-%m-%d"):
        return False
    if data.get("dry_run") is True:
        return False
    return int(data.get("submitted_order_count", 0) or 0) > 0


def write_submission_marker(marker_path: Path, *, run_id: str, submitted_order_count: int, dry_run: bool) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "run_id": run_id,
        "submitted_order_count": int(submitted_order_count),
        "dry_run": bool(dry_run),
    }
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote submission marker: %s", marker_path)

