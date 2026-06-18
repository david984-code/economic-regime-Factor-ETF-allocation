"""Shared structured run reporting for operators and audit trails."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    run_id: str
    runner: str
    mode: str  # e.g. "dry_run" | "live" | "safety_fail" | "error"
    status: str  # "ok" | "blocked" | "error"
    started_utc: str
    finished_utc: str | None = None
    message: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def utc_now_str() -> str:
    return datetime.now(UTC).isoformat()


def write_run_summary(report_dir: Path, summary: RunSummary) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"run_summary_{summary.run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, indent=2)
    logger.info("Wrote run summary: %s", path)
    return path
