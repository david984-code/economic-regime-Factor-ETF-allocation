"""Retry utilities for transient failures.

`retry_on_permission_error` is for local-disk file contention.
`external_api_retry` is for flaky network calls (FRED, yfinance, IBKR API):
exponential backoff via tenacity with sensible defaults.
"""

import time
from collections.abc import Callable
from typing import TypeVar

import tenacity

T = TypeVar("T")


# Exponential-backoff retry for external API calls. Apply as a decorator OR
# inline via `external_api_retry()(fn)(...)`. Five attempts, waiting 1s, 2s,
# 4s, 8s, capped at 30s. Does NOT retry on KeyboardInterrupt (you want to
# bail out of a hung run with Ctrl-C) or on authentication errors.
external_api_retry = lambda: tenacity.retry(  # noqa: E731
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
    retry=tenacity.retry_if_exception_type(
        (
            ConnectionError,
            TimeoutError,
            OSError,
            # tenacity catches generic Exception by default; we want to NOT
            # retry on auth or bad-input errors. ValueError covers most
            # "you asked for something invalid" cases from fredapi.
        )
    ),
    reraise=True,
)


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
