"""Single source of truth for classifying backend failures.

Before this module the retry decision in :func:`daydream.agent.run_agent` and the
Pi backend's diagnostic category disagreed: the retry branch trusted an opt-in
``retryable`` flag (derived from a message heuristic) and never consulted
``category``, so a permanent condition that happened to mention a transient
token (``"model not found: gpt-5 (503)"``) retried. :func:`classify_failure`
collapses both views into one total, single-valued decision in which permanent
conditions win ties.

The classifier is deliberately pure: no I/O, no clock reads, and it never
raises. Every attribute access is guarded and ``str(exc)`` is bounded, so a
hostile exception cannot make classification itself a failure.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FailureClass(StrEnum):
    """The outcome classes a backend failure can fall into."""

    RATE_LIMIT = "RATE_LIMIT"
    SERVER_ERROR = "SERVER_ERROR"
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"
    AUTH_CONFIG = "AUTH_CONFIG"
    SCHEMA = "SCHEMA"
    TOOL_POLICY = "TOOL_POLICY"
    PERMANENT = "PERMANENT"
    NOT_RETRYABLE = "NOT_RETRYABLE"


#: Classes for which a retry can never help. Membership is the single definition
#: of "permanent"; :func:`classify_failure` derives ``retries_allowed`` from it.
_PERMANENT_CLASSES = frozenset(
    {
        FailureClass.AUTH_CONFIG,
        FailureClass.SCHEMA,
        FailureClass.TOOL_POLICY,
        FailureClass.PERMANENT,
        FailureClass.NOT_RETRYABLE,
    }
)

#: Transient categories (backend diagnostic vocabulary) mapped to a class. The
#: stream families are transport failures; anything unrecognised is a generic
#: server error, matching the pre-collapse fallback.
_TRANSIENT_CATEGORY_CLASSES: dict[str, FailureClass] = {
    "RATE_LIMIT": FailureClass.RATE_LIMIT,
    "SERVER_ERROR": FailureClass.SERVER_ERROR,
    "TIMEOUT": FailureClass.TIMEOUT,
    "TRANSPORT": FailureClass.TRANSPORT,
    "STREAM_DROP": FailureClass.TRANSPORT,
    "STREAM_TRUNCATION": FailureClass.TRANSPORT,
}

#: Message substrings that name a permanent condition. These win over any
#: transient token in the same message: a 503 is irrelevant if the model does
#: not exist. A bare "provider" is deliberately *not* listed: it is a generic
#: noun that appears in transient failures too ("provider rate limit"), so a
#: real provider config failure must arrive as the ``AUTH_CONFIG`` category
#: (which :func:`classify_failure` honours) or carry an explicit
#: "not configured"/"not authenticated" marker.
_PERMANENT_CONDITION_RE = re.compile(
    r"credential"
    r"|api[ _-]?key"
    r"|not configured"
    r"|model not found"
    r"|schema validation"
    r"|additionalproperties"
    r"|not authenticated",
    re.IGNORECASE,
)

#: Numeric-seconds ``retry[- ]after[: ]N`` token in a failure message. Shared
#: by the agent retry branch and any backend that carries a hint in text rather
#: than on the exception attribute.
_RETRY_HINT_RE = re.compile(r"retry[- ]after[:\s]+([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_message_retry_hint(message: str) -> float | None:
    """Extract a numeric-seconds server retry hint from *message*.

    Returns ``None`` for an absent, non-numeric, negative, or non-finite hint;
    an unparseable hint degrades to jitter rather than to a fabricated delay.
    ``0`` is a valid hint. Never raises.
    """
    if not message:
        return None
    match = _RETRY_HINT_RE.search(message)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


@dataclass(frozen=True)
class RetryDecision:
    """The classifier's total verdict for one failure."""

    failure_class: FailureClass
    retries_allowed: bool


def decode_retry_recovery_allowance(raw: Any) -> float | None:
    """Decode one declared retry-recovery allowance value, or ``None`` if refused.

    The single decode rule shared by every allowance source -- a backend
    ``RetryPolicy`` field, a backend attribute, ``run_agent``'s explicit argument,
    the ``DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S`` env var, and the repo
    ``retry_recovery_allowance_s`` config key -- so one value can never be
    accepted on one path and refused on another. A numeric string (the env
    shape) is accepted; a bool, a non-number, a non-finite value, or a negative
    value is refused as ``None``, which means "not declared" and never becomes an
    effective bound; ``0`` is a real declaration that disables retry recovery.

    Callers keep their own logging surface, but should render the refusal with
    :func:`undeclared_retry_allowance_message` so one rule reads the same way
    everywhere. Never raises.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def undeclared_retry_allowance_message(source: str, raw: Any) -> str:
    """One warning shape for a refused allowance value, shared by every source.

    Names the source and the raw value and states the consequence: the value
    stays undeclared, so the caller's own default applies downstream. The refused
    value is never restated as if it were a bound.
    """
    return (
        f"{source}={raw!r} is not a finite non-negative number; "
        "the retry-recovery allowance stays undeclared"
    )


@dataclass
class RetryRecoveryBudget:
    """Cumulative retry-overhead allowance for one invocation's ladder.

    A retry may spend only the recovery budget it was given, never the
    invocation's useful-work time. The budget is activated on the *first*
    retryable failure; at that instant its ``allowance_s`` is clamped once to
    the remaining effective deadline so it composes with the single invocation
    deadline by clamping, never by re-basing. Every later failure consults the
    same cumulative accumulator, so a retry storm cannot reset the allowance.

    Pure state: the caller supplies every clock value. Charges are cumulative
    and independent of ``backend_s``/``backoff_s``; :meth:`remaining` never
    returns a negative value.
    """

    allowance_s: float
    _activated_at: float | None = field(default=None, init=False)
    _spent_s: float = field(default=0.0, init=False)

    def activate(self, now: float, effective_deadline: float | None = None) -> None:
        """Start the allowance clock; idempotent on every later failure.

        On the first call the activation instant is recorded and ``allowance_s``
        is clamped to whatever the effective deadline leaves (never increased),
        once, never re-based. Later calls are no-ops.
        """
        if self._activated_at is not None:
            return
        self._activated_at = now
        if effective_deadline is not None:
            remaining_deadline_s = effective_deadline - now
            if remaining_deadline_s < self.allowance_s:
                self.allowance_s = max(remaining_deadline_s, 0.0)

    @property
    def active(self) -> bool:
        """Whether a retryable failure has activated the budget."""
        return self._activated_at is not None

    @property
    def spent_s(self) -> float:
        """Cumulative retry overhead charged so far."""
        return self._spent_s

    def charge(self, seconds: float) -> None:
        """Charge time to the cumulative retry allowance."""
        if seconds > 0:
            self._spent_s += seconds

    def remaining(self) -> float:
        """Seconds of recovery allowance left, clamped at zero."""
        return max(0.0, self.allowance_s - self._spent_s)


def _numeric(value: Any) -> float:
    """Coerce one serialized numeric field, defaulting malformed values to zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    as_float = float(value)
    return as_float if math.isfinite(as_float) else 0.0


def derive_retry_summary(phase_events: Any) -> dict[str, Any] | None:
    """Reduce frozen phase events into a retry/circuit summary.

    Counts every retry-ladder stop's ``retry_stop_reason`` (omitting ``None``),
    sums the duration/count fields over those events only, and collects the
    distinct non-``None`` ``circuit_state`` values in first-seen order. A
    deadline stop carries an ``agent_budget_stop`` payload but no retry reason,
    so it is skipped entirely: its ``attempts``/``backend_s`` are useful-work
    time, not retry overhead. Returns ``None`` when no event carries a
    retry-ladder stop reason, so a run without retry activity produces no
    summary and its manifest stays byte-identical.

    The reducer is total: a malformed container or payload degrades to
    ``None``/zero contributions instead of raising on the archive write path.
    Only durations, counts and reason/state codes are emitted -- never an
    absolute or monotonic deadline value.
    """
    if not isinstance(phase_events, Sequence) or isinstance(phase_events, (str, bytes, bytearray)):
        return None

    stops: dict[str, int] = {}
    circuit_states: list[str] = []
    attempts = 0
    backoff_s = 0.0
    backend_s = 0.0
    retry_recovery_spent_s = 0.0
    for event in phase_events:
        if not isinstance(event, Mapping) or event.get("event") != "agent_budget_stop":
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        reason = metadata.get("retry_stop_reason")
        if not isinstance(reason, str) or not reason:
            # Not a retry-ladder stop (e.g. a plain deadline stop): its attempts,
            # backend_s and circuit_state describe useful work, so folding them
            # in would let non-retry time dominate the retry totals.
            continue
        stops[reason] = stops.get(reason, 0) + 1
        state = metadata.get("circuit_state")
        if isinstance(state, str) and state and state not in circuit_states:
            circuit_states.append(state)
        attempts += int(_numeric(metadata.get("attempts")))
        backoff_s += _numeric(metadata.get("backoff_s"))
        backend_s += _numeric(metadata.get("backend_s"))
        retry_recovery_spent_s += _numeric(metadata.get("retry_recovery_spent_s"))

    if not stops:
        return None
    return {
        "stops": stops,
        "attempts": attempts,
        "backoff_s": round(backoff_s, 6),
        "backend_s": round(backend_s, 6),
        "retry_recovery_spent_s": round(retry_recovery_spent_s, 6),
        "circuit_states": circuit_states,
    }


def _decide(failure_class: FailureClass) -> RetryDecision:
    return RetryDecision(
        failure_class=failure_class,
        retries_allowed=failure_class not in _PERMANENT_CLASSES,
    )


def _coerce_class(value: Any) -> FailureClass | None:
    """Best-effort conversion of a declared ``failure_class`` attribute."""
    if isinstance(value, FailureClass):
        return value
    if isinstance(value, str):
        try:
            return FailureClass(value)
        except ValueError:
            try:
                return FailureClass[value]
            except KeyError:
                return None
    return None


def _transient_class(category: str | None) -> FailureClass:
    if category is None:
        return FailureClass.SERVER_ERROR
    return _TRANSIENT_CATEGORY_CLASSES.get(category.upper(), FailureClass.SERVER_ERROR)


def classify_failure(exc: BaseException) -> RetryDecision:
    """Classify *exc* into exactly one :class:`RetryDecision`.

    Order (first match wins): a declared ``failure_class`` attribute; a truthy
    ``tool_policy_stop``; the permanent categories ``AUTH_CONFIG``/``SCHEMA``; a
    truthy ``permanent`` attribute; a permanent-condition match in the message;
    a ``retryable`` flag or transient category; otherwise ``NOT_RETRYABLE``.
    """
    declared = _coerce_class(getattr(exc, "failure_class", None))
    if declared is not None:
        return _decide(declared)

    if getattr(exc, "tool_policy_stop", False):
        return _decide(FailureClass.TOOL_POLICY)

    raw_category = getattr(exc, "category", None)
    category = raw_category if isinstance(raw_category, str) else None
    if category is not None:
        upper = category.upper()
        if upper == FailureClass.AUTH_CONFIG:
            return _decide(FailureClass.AUTH_CONFIG)
        if upper == FailureClass.SCHEMA:
            return _decide(FailureClass.SCHEMA)

    if getattr(exc, "permanent", False):
        return _decide(FailureClass.PERMANENT)

    try:
        message = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ must not break classification
        message = ""
    if _PERMANENT_CONDITION_RE.search(message):
        return _decide(FailureClass.PERMANENT)

    if getattr(exc, "retryable", False):
        return _decide(_transient_class(category))
    if not hasattr(exc, "retryable") and category is not None and category.upper() in _TRANSIENT_CATEGORY_CLASSES:
        return _decide(_TRANSIENT_CATEGORY_CLASSES[category.upper()])
    return _decide(FailureClass.NOT_RETRYABLE)
