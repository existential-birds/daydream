"""Resolve retry limits and account for useful work versus recovery overhead."""

import math
import os
import random
from dataclasses import dataclass

from daydream.backends import Backend
from daydream.diagnostics import exception_text
from daydream.retry_policy import parse_message_retry_hint


def _sample_retry_delay(cap: float) -> float:
    """Sample unseeded full jitter; callers clamp again against hostile samples."""
    if cap <= 0:
        return 0.0
    return random.uniform(0.0, cap)


def _retry_delay_from_env(name: str, default: float) -> float:
    """Read one non-negative finite delay knob from the environment, else *default*."""
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0 else default


@dataclass(frozen=True)
class _ResolvedRetrySettings:
    """One invocation's fully-resolved retry ladder settings."""

    max_attempts: int
    base_delay_s: float
    max_delay_s: float


def _resolve_retry_settings(backend: Backend) -> _ResolvedRetrySettings:
    """Resolve limits before dispatch; inverted delays raise ValueError.

    A declared policy suppresses ambient reads.
    """
    retry_policy = getattr(backend, "retry_policy", None)
    if retry_policy is not None:
        max_attempts = retry_policy.attempts
        base_delay = retry_policy.base_delay_s
        max_delay = retry_policy.max_delay_s
    else:
        try:
            default_attempts = int(os.environ.get("DAYDREAM_PI_RETRY_ATTEMPTS", "20"))
        except ValueError:
            default_attempts = 20
        if default_attempts < 0:
            default_attempts = 20
        max_attempts = getattr(backend, "retry_attempts", default_attempts)
        base_delay = getattr(
            backend,
            "retry_base_delay_s",
            _retry_delay_from_env("DAYDREAM_PI_RETRY_BASE_DELAY_S", 2.0),
        )
        max_delay = getattr(
            backend,
            "retry_max_delay_s",
            _retry_delay_from_env("DAYDREAM_PI_RETRY_MAX_DELAY_S", 120.0),
        )
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
    return _ResolvedRetrySettings(
        max_attempts=max_attempts,
        base_delay_s=base_delay,
        max_delay_s=max_delay,
    )


def _plan_retry_delay(
    *,
    attempt: int,
    base_delay_s: float,
    max_delay_s: float,
    deadline_remaining_s: float | None,
    hint: float | None,
) -> tuple[float, str | None]:
    """Return a bounded hint/jitter delay, or retry_hint_exceeds_budget.

    A server hint is evaluated against the remaining deadline only; an admitted
    hint is waited in full and never shortened by the exponential cap or
    configured maximum jitter delay. A hint longer than the remaining deadline
    stops the ladder with ``retry_hint_exceeds_budget``. The exponential cap and
    maximum delay bound the fallback jitter path only.
    """
    if hint is not None:
        if deadline_remaining_s is not None and hint > deadline_remaining_s:
            return 0.0, "retry_hint_exceeds_budget"
        # A spent-but-not-none bound cannot turn an admitted hint negative.
        return max(hint, 0.0), None
    cap = min(base_delay_s * (2 ** attempt), max_delay_s)
    if deadline_remaining_s is not None:
        cap = min(cap, deadline_remaining_s)
    cap = max(cap, 0.0)
    return min(_sample_retry_delay(cap), cap), None


@dataclass
class _RetryTelemetry:
    """Account for dispatched attempts separately from retry overhead.

    The opening useful-work attempt contributes to backend_s only. Retries
    charge retry_backend_s and backoff_s; cleanup is excluded from backend time.
    This keeps backend_s + backoff_s within elapsed invocation time.
    """

    attempts_dispatched: int = 0
    retry_attempts: int = 0
    backend_s: float = 0.0
    retry_backend_s: float = 0.0
    backoff_s: float = 0.0
    attempt_started_at: float | None = None
    attempt_is_retry: bool = False

    def start_attempt(self, now: float, *, retry: bool) -> None:
        """Record one dispatched attempt; *retry* marks it as retry overhead."""
        self.attempts_dispatched += 1
        self.attempt_is_retry = retry
        if retry:
            self.retry_attempts += 1
        self.attempt_started_at = now

    def charge_attempt(self, now: float, *, cleanup_elapsed_s: float = 0.0) -> float:
        """Close an attempt, excluding bounded cleanup time; return zero when none is in flight."""
        if self.attempt_started_at is None:
            return 0.0
        delta = now - self.attempt_started_at - cleanup_elapsed_s
        self.backend_s += delta
        if self.attempt_is_retry:
            self.retry_backend_s += delta
        self.attempt_started_at = None
        return delta

    def pending_s(self, now: float) -> float:
        """The in-flight attempt's elapsed time, or ``0.0`` when none is running."""
        if self.attempt_started_at is None:
            return 0.0
        return max(now - self.attempt_started_at, 0.0)

    def pending_retry_s(self, now: float) -> float:
        """The in-flight attempt's elapsed time when it is a retry, else ``0.0``."""
        return self.pending_s(now) if self.attempt_is_retry else 0.0

    @property
    def spent_retry_overhead(self) -> bool:
        """Count backoff as recovery even when the deadline prevents its retry from dispatching."""
        return self.retry_attempts > 0 or self.backoff_s > 0.0


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
