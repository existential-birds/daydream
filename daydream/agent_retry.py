"""Resolve retry limits and account for useful work versus recovery overhead."""

import math
import os
import random

from daydream.backends import Backend, RetryPolicy, retry_policy_from_environment
from daydream.diagnostics import exception_text
from daydream.retry_policy import parse_message_retry_hint


def _sample_retry_delay(cap: float) -> float:
    """Sample unseeded full jitter; callers clamp again against hostile samples."""
    if cap <= 0:
        return 0.0
    return random.uniform(0.0, cap)


def _resolve_retry_settings(backend: Backend) -> RetryPolicy:
    """Resolve limits before dispatch; inverted delays raise ValueError.

    A declared policy suppresses ambient reads; otherwise backend attributes win
    over the one ambient retry parser.
    """
    retry_policy = getattr(backend, "retry_policy", None)
    if retry_policy is not None:
        max_attempts = retry_policy.attempts
        base_delay = retry_policy.base_delay_s
        max_delay = retry_policy.max_delay_s
    else:
        declared = [getattr(backend, name, None) for name in ("retry_attempts", "retry_base_delay_s",
                                                               "retry_max_delay_s")]
        if None in declared:
            # Only settings the backend leaves unset come from the one ambient parser.
            ambient = retry_policy_from_environment(os.environ, backend="default")
            defaults = (ambient.attempts, ambient.base_delay_s, ambient.max_delay_s)
            declared = [default if value is None else value for value, default in zip(declared, defaults, strict=True)]
        max_attempts, base_delay, max_delay = declared
    if max_attempts < 0:
        raise ValueError("retry attempts must be >= 0")
    for label, value in (("base", base_delay), ("max", max_delay)):
        if not math.isfinite(value):
            raise ValueError(f"retry {label} delay must be finite")
        if value < 0:
            raise ValueError(f"retry {label} delay must be >= 0")

    # Two valid delays can still contradict one another; refuse before dispatch.
    if base_delay > max_delay:
        raise ValueError(
            f"retry_base_delay_s ({base_delay}) must not exceed "
            f"retry_max_delay_s ({max_delay})"
        )
    return RetryPolicy(attempts=max_attempts, base_delay_s=base_delay, max_delay_s=max_delay)


def _plan_retry_delay(
    *,
    attempt: int,
    base_delay_s: float,
    max_delay_s: float,
    deadline_remaining_s: float | None,
    hint: float | None,
) -> float | None:
    """Return a bounded hint/jitter delay, or None when a server hint outlasts the deadline.

    A server hint is evaluated against the remaining deadline only; an admitted
    hint is waited in full and never shortened by the exponential cap or
    configured maximum jitter delay. A hint longer than the remaining deadline
    stops the ladder. The exponential cap and maximum delay bound the fallback
    jitter path only.
    """
    if hint is not None:
        if deadline_remaining_s is not None and hint > deadline_remaining_s:
            return None
        # A spent-but-not-none bound cannot turn an admitted hint negative.
        return max(hint, 0.0)
    cap = min(base_delay_s * (2 ** attempt), max_delay_s)
    if deadline_remaining_s is not None:
        cap = min(cap, deadline_remaining_s)
    cap = max(cap, 0.0)
    return float(min(_sample_retry_delay(cap), cap))


def _retry_hint(exc: BaseException) -> float | None:
    """Read a finite nonnegative retry_after attribute, else parse the message token.

    A present malformed attribute suppresses message fallback and uses jitter; zero
    is valid. Never raises.
    """
    attribute = getattr(exc, "retry_after", None)
    if attribute is None:
        return parse_message_retry_hint(exception_text(exc) or "")
    if isinstance(attribute, bool) or not isinstance(attribute, (int, float)):
        return None
    value = float(attribute)
    if not math.isfinite(value) or value < 0:
        return None
    return value
