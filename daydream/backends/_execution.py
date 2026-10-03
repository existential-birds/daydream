"""Run-owned backend environment and retry settings."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from daydream.retry_policy import decode_retry_recovery_allowance, undeclared_retry_allowance_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Complete retry settings owned by one constructed backend."""

    attempts: int
    base_delay_s: float
    max_delay_s: float
    #: Invocation retry overhead in seconds; zero disables recovery, None defers.
    #: A declared RetryPolicy suppresses ambient PI retry settings. Embedded runs
    #: materialize environment overrides into this policy during construction.
    retry_recovery_allowance_s: float | None = None


def _parsed_nonnegative_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid integer; using default %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%r is negative; using default %d", name, raw, default)
        return default
    return value


def _parsed_nonnegative_float(environment: Mapping[str, str], name: str, default: float) -> float:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid float; using default %g", name, raw, default)
        return default
    if not math.isfinite(value):
        logger.warning("%s=%r is not finite; using default %g", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%r is negative; using default %g", name, raw, default)
        return default
    return value


def _parsed_optional_retry_allowance(environment: Mapping[str, str], name: str) -> float | None:
    """Decode an optional allowance; absent/invalid values defer to the caller, with warnings on invalid input."""
    raw = environment.get(name)
    if raw is None:
        return None
    value = decode_retry_recovery_allowance(raw)
    if value is None:
        logger.warning("%s", undeclared_retry_allowance_message(name, raw))
    return value


def _parsed_positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    value = _parsed_nonnegative_int(environment, name, default)
    if value == 0:
        logger.warning("%s must be positive; using default %d", name, default)
        return default
    return value


@dataclass(frozen=True)
class BackendExecutionInput:
    """Run-owned settings with an immutable copied environment; each transport receives a fresh mutable copy."""

    _environment: Mapping[str, str] = field(repr=False, compare=False)
    retry_policy: RetryPolicy
    fanout_concurrency: int
    pi_provider: str | None
    pi_thinking: str | None
    pi_agent_dir: Path | None
    stream_idle_timeout_s: float | None
    pi_response_idle_timeout_s: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_environment", MappingProxyType(dict(self._environment)))

    @classmethod
    def from_environment(cls, environment: Mapping[str, str], *, backend: str) -> BackendExecutionInput:
        """Parse one complete environment without consulting process globals."""
        from daydream.backends._subprocess import (
            DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S,
            DEFAULT_STREAM_IDLE_TIMEOUT_S,
        )

        if backend not in {"pi", "codex", "claude"}:
            if backend == "osprey":
                raise ValueError("explicit BackendExecutionInput is not supported for osprey")
            raise ValueError(f"unsupported backend for execution input: {backend!r}")

        copied = dict(environment)
        base_delay_default = 10.0 if backend == "pi" else 2.0
        fanout_name = "DAYDREAM_PI_FANOUT_CONCURRENCY" if backend == "pi" else "DAYDREAM_FANOUT_CONCURRENCY"
        fanout_default = 10 if backend == "pi" else 8
        idle = _parsed_nonnegative_float(copied, "DAYDREAM_STREAM_IDLE_TIMEOUT_S", DEFAULT_STREAM_IDLE_TIMEOUT_S)
        response_idle = _parsed_nonnegative_float(
            copied,
            "DAYDREAM_STREAM_IDLE_TIMEOUT_S",
            DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S,
        )
        home = copied.get("HOME")
        agent_dir_raw = copied.get("PI_CODING_AGENT_DIR")
        pi_agent_dir = Path(agent_dir_raw) if agent_dir_raw else Path(home) / ".pi" / "agent" if home else None

        return cls(
            copied,
            RetryPolicy(
                attempts=_parsed_nonnegative_int(copied, "DAYDREAM_PI_RETRY_ATTEMPTS", 20),
                base_delay_s=_parsed_nonnegative_float(copied, "DAYDREAM_PI_RETRY_BASE_DELAY_S", base_delay_default),
                max_delay_s=_parsed_nonnegative_float(copied, "DAYDREAM_PI_RETRY_MAX_DELAY_S", 120.0),
                # Parsed here, not in run_agent: this is the construction path
                # every embedded caller uses, and ``run_agent`` resolves the
                # documented top precedence tier (RetryPolicy) before the env, so
                # the operator knob must be materialised into the policy or it
                # would be silently dropped on this path only.
                retry_recovery_allowance_s=_parsed_optional_retry_allowance(
                    copied, "DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S"
                ),
            ),
            _parsed_positive_int(copied, fanout_name, fanout_default),
            copied.get("PI_PROVIDER") or None,
            copied.get("PI_THINKING") or None,
            pi_agent_dir,
            None if idle == 0 else idle,
            None if response_idle == 0 else response_idle,
        )

    def child_environment(self) -> dict[str, str]:
        """Return a fresh complete environment for one native transport."""
        return dict(self._environment)
