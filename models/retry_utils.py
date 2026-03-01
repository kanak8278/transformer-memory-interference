"""
Shared retry utility for all model wrappers.

Provides exponential backoff with full jitter — the recommended strategy
when many concurrent workers (e.g. 40 threads) may hit rate limits
simultaneously. Full jitter spreads retries randomly across the backoff
window, preventing thundering herd.

Backoff formula:
    wait = random.uniform(0, base_delay * 2^attempt)

    attempt 0 (1st retry): 0 – base_delay         (e.g. 0–2s)
    attempt 1 (2nd retry): 0 – 2×base_delay        (e.g. 0–4s)
    attempt 2 (3rd retry): 0 – 4×base_delay        (e.g. 0–8s)
    attempt 3 (4th retry): 0 – 8×base_delay        (e.g. 0–16s)

Usage:
    from .retry_utils import call_with_retry

    def _call():
        return client.some_api_method(...)

    result = call_with_retry(_call, label="claude-haiku")
"""

import time
import random
from typing import Callable, Any


# Error signals that are worth retrying (transient, not permanent failures)
_RETRYABLE_SIGNALS = [
    "rate_limit",
    "rate limit",
    "429",           # HTTP Too Many Requests
    "529",           # Anthropic overload
    "500",           # Internal server error
    "502",           # Bad gateway
    "503",           # Service unavailable
    "quota",
    "overloaded",
    "timeout",
    "timed out",
    "connection",    # transient connection errors
]


def is_retryable(error_str: str) -> bool:
    """Return True if the error is transient and worth retrying."""
    error_lower = error_str.lower()
    return any(signal in error_lower for signal in _RETRYABLE_SIGNALS)


def call_with_retry(
    fn: Callable,
    max_retries: int = 5,
    base_delay: float = 2.0,
    label: str = "API",
) -> Any:
    """
    Call fn() with exponential backoff + full jitter on retryable errors.

    Args:
        fn:          Zero-argument callable that makes the API call.
        max_retries: Total attempts (1 original + max_retries-1 retries).
        base_delay:  Base delay in seconds for backoff calculation.
        label:       Model name / label for log messages.

    Returns:
        Whatever fn() returns on success.

    Raises:
        The last exception if all retries are exhausted, or immediately
        for non-retryable errors.
    """
    last_exc = None

    for attempt in range(max_retries):
        try:
            return fn()

        except Exception as e:
            last_exc = e
            error_str = str(e)

            if is_retryable(error_str) and attempt < max_retries - 1:
                # Full jitter: spread retries randomly across the window
                # This prevents all 40 workers waking up at the same time
                wait = random.uniform(0, base_delay * (2 ** attempt))
                print(f"  ⚠ {label} retry {attempt + 1}/{max_retries - 1} "
                      f"in {wait:.1f}s — {error_str[:80]}")
                time.sleep(wait)
            else:
                # Non-retryable or final attempt — re-raise immediately
                raise

    # Should not reach here, but just in case
    raise last_exc
