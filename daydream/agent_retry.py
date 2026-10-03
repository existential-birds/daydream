"""Resolve retry limits and account for useful work versus recovery overhead."""

import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any

from daydream.backends import Backend
from daydream.config import DEFAULT_RETRY_RECOVERY_ALLOWANCE_S
from daydream.diagnostics import exception_text
from daydream.retry_policy import (
    decode_retry_recovery_allowance,
    parse_message_retry_hint,
    undeclared_retry_allowance_message,
)

_logger = logging.getLogger(__name__)


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
    allowance_s: float


def _allowance_source(backend: Backend, policy: Any, explicit: float | None) -> tuple[str, Any] | None:
    """Return the first declared allowance without reading lower-priority sources."""
    key = "retry_recovery_allowance_s"
    for owner, prefix in ((policy, "RetryPolicy."), (backend, "")):
        value = getattr(owner, key, None)
        if value is not None:
            return prefix + key, value
    if explicit is not None:
        return key, explicit
    if policy is None:
        key = "DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S"
        value = os.environ.get(key)
        if value is not None:
            return key, value
    return None


def _resolve_retry_settings(
    backend: Backend, retry_recovery_allowance_s: float | None
) -> _ResolvedRetrySettings:
    """Resolve allowance precedence: policy > backend > argument > ambient Pi env > default.
    Policies suppress ambient reads. Invalid allowances warn/fall back; inverted delays or
    explicit nonzero allowance with disabled retries raise ValueError before dispatch.
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

    allowance_source = _allowance_source(backend, retry_policy, retry_recovery_allowance_s)
    resolved_allowance = DEFAULT_RETRY_RECOVERY_ALLOWANCE_S
    declared_allowance = False
    if allowance_source is not None:
        parsed_allowance = _coerce_retry_recovery_allowance(
            allowance_source[1], allowance_source[0]
        )
        if parsed_allowance is not None:
            resolved_allowance = parsed_allowance
            declared_allowance = True

    # Two declared values can each be valid and still contradict one another.
    # Refuse the combination here, before any dispatch, with a message naming both
    # keys -- never coerce it into a plausible bound. The allowance check fires
    # only for a *declared* non-zero value: the default 300s must not turn
    # ``retry_attempts = 0`` (a legitimate "no retries" declaration) into an error.
    if base_delay > max_delay:
        raise ValueError(
            f"retry_base_delay_s ({base_delay}) must not exceed "
            f"retry_max_delay_s ({max_delay})"
        )
    if declared_allowance and resolved_allowance > 0 and max_attempts == 0:
        raise ValueError(
            "retry_recovery_allowance_s "
            f"({resolved_allowance}) cannot be non-zero while retries are "
            "disabled (max_attempts == 0); set retry_recovery_allowance_s = 0 "
            "to disable retry recovery explicitly"
        )
    return _ResolvedRetrySettings(
        max_attempts=max_attempts,
        base_delay_s=base_delay,
        max_delay_s=max_delay,
        allowance_s=resolved_allowance,
    )


def _plan_retry_delay(
    *,
    attempt: int,
    base_delay_s: float,
    max_delay_s: float,
    allowance_remaining_s: float | None,
    deadline_remaining_s: float | None,
    hint: float | None,
) -> tuple[float, str | None]:
    """Bound hint/jitter by exponential growth, maximum delay, recovery allowance, and deadline.
    Server hints replace jitter; hints exceeding remaining budgets return retry_hint_exceeds_budget.
    """
    bounds = [
        bound
        for bound in (
            allowance_remaining_s,
            # ``max(..., 0.0)`` twice: a spent deadline is a spent bound, never a
            # negative one that would invert the comparison below.
            deadline_remaining_s,
        )
        if bound is not None
    ]
    cap = min(base_delay_s * (2 ** attempt), max_delay_s)
    if bounds:
        cap = min(cap, min(bounds))
    cap = max(cap, 0.0)
    if hint is not None and bounds and hint > min(bounds):
        return 0.0, "retry_hint_exceeds_budget"
    return min(hint if hint is not None else _sample_retry_delay(cap), cap), None


@dataclass
class _RetryTelemetry:
    """Initial useful work charges backend_s; retries also charge retry_backend_s and backoff_s.
    Cleanup is excluded so backend_s + backoff_s stays within elapsed invocation time.
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


def _coerce_retry_recovery_allowance(raw: Any, source: str) -> float | None:
    """Decode shared config/env/backend allowance; zero disables recovery, invalid input warns/falls back."""
    value = decode_retry_recovery_allowance(raw)
    if value is None:
        _logger.warning(
            "daydream: %s; using default %s",
            undeclared_retry_allowance_message(source, raw),
            DEFAULT_RETRY_RECOVERY_ALLOWANCE_S,
        )
    return value


def _retry_hint(exc: BaseException) -> float | None:
    """Read finite nonnegative retry_after (zero allowed), or parse the message when absent.
    Malformed present attributes suppress fallback and use jitter; never raises.
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
