"""Retry utilities for transient failures."""

import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def retry_on_permission_error(
    func: Callable[[], T],
    max_attempts: int = 4,
    base_wait: float = 0.5,
    logger: Optional[object] = None,
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
