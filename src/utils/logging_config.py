"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger and return it.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path to log file.
        format_string: Optional format string.

    Returns:
        Root logger instance.
    """
    fmt = format_string or "%(asctime)s - %(levelname)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger()
