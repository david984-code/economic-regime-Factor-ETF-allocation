"""Retry utilities for transient failures."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_logger = logging.getLogger(__name__)


def retry_on_permission_error(
    func: Callable[[], T],
    max_attempts: int = 4,
    base_wait: float = 0.5,
    logger: object | None = None,
) -> T:
    """Retry func with exponential backoff on PermissionError.

    Args:
        func: Callable with no args that may raise PermissionError.
        max_attempts: Maximum number of attempts.
        base_wait: Base wait time in seconds (doubles each retry).
        logger: Optional logger; if None, no log on retry.

    Returns:
        Result of func().

    Raises:
        PermissionError: If all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except PermissionError as e:
            last_exc = e
            if attempt < max_attempts - 1:
                wait = base_wait * (2**attempt)
                if logger and hasattr(logger, "warning"):
                    logger.warning(
                        "File locked, retrying in %.1fs (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_attempts,
                    )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_wait: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry func with exponential backoff (1s, 2s, 4s) on network/transient errors.

    Args:
        func: Callable that may raise a transient error.
        max_attempts: Total attempts (default 3).
        base_wait: Initial wait in seconds (doubles each retry).
        exceptions: Tuple of exception types to catch and retry.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exc = e
            if attempt < max_attempts - 1:
                wait = base_wait * (2**attempt)
                _logger.warning(
                    "Attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    max_attempts,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                _logger.error(
                    "All %d attempts failed: %s",
                    max_attempts,
                    e,
                )
    raise last_exc  # type: ignore[misc]
