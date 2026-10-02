"""Pure backend-failure classification and cumulative retry-recovery accounting.

Explicit permanent conditions take precedence over transient message hints.
Exception messages use the shared safe conversion boundary."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from daydream.diagnostics import exception_text
from daydream.json_utils import extract_json


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
#: than on the exception attribute. Structured payloads are read first (see
#: :func:`_structured_retry_hint`); this regex is the fallback for plain text.
_RETRY_HINT_RE = re.compile(r"retry[- ]after[:\s]+([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

#: Normalised header names admitted from a structured ``metadata.headers`` map.
#: Matching on the lowercased, stripped key is what makes the read case-insensitive.
_STRUCTURED_RETRY_HINT_KEY: frozenset[str] = frozenset({"retry-after"})


def _structured_retry_hint(message: str) -> float | None:
    """Read a hint from a provider JSON payload, guarded at every level.

    The decoded payload, its ``metadata``, and its ``headers`` must each be a
    mapping before any key lookup; the first admitted header wins. Anything else
    yields no hint so the text fallback stays live. Values pass through the
    shared decoder, so no second numeric rule is introduced here."""
    payload = extract_json(message)
    if not isinstance(payload, Mapping):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    headers = metadata.get("headers")
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if not isinstance(key, str) or key.strip().lower() not in _STRUCTURED_RETRY_HINT_KEY:
            continue
        hint = decode_retry_recovery_allowance(value)
        if hint is not None:
            return hint
    return None


def parse_message_retry_hint(message: str) -> float | None:
    """Return finite nonnegative retry-after seconds, including zero; else None.

    A structured ``metadata.headers.Retry-After`` value wins when it decodes; the
    legacy plain-text token is the fallback for messages without one."""
    if not message:
        return None
    structured = _structured_retry_hint(message)
    if structured is not None:
        return structured
    match = _RETRY_HINT_RE.search(message)
    return decode_retry_recovery_allowance(match.group(1)) if match else None


@dataclass(frozen=True)
class RetryDecision:
    """The classifier's total verdict for one failure."""

    failure_class: FailureClass
    retries_allowed: bool


def decode_retry_recovery_allowance(raw: Any) -> float | None:
    """Decode a shared backend/config/environment allowance, or None if undeclared.

    Numeric strings are accepted; bools, nonnumbers, negatives, and nonfinite values
    are refused. Zero disables recovery. Callers own logging via the shared warning."""
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
    """Name a refused value and source without presenting it as an effective bound."""
    return (
        f"{source}={raw!r} is not a finite non-negative number; "
        "the retry-recovery allowance stays undeclared"
    )


@dataclass
class RetryRecoveryBudget:
    """Accumulate invocation retry overhead independently of useful-work time.

    The first retryable failure clamps the allowance once to the remaining deadline.
    Later failures cannot reset it. Callers supply clocks; remaining time is nonnegative."""

    allowance_s: float
    _activated_at: float | None = field(default=None, init=False)
    _spent_s: float = field(default=0.0, init=False)

    def activate(self, now: float, effective_deadline: float | None = None) -> None:
        """Activate once and clamp the allowance to the remaining effective deadline."""
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
    """Reduce retry-ladder stops to reason counts, durations, and ordered circuit states.

    Skip ordinary deadline stops: they describe useful work. Missing retry activity
    returns None, preserving old manifest shapes. Malformed containers contribute
    nothing; absolute or monotonic deadlines are never emitted."""
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

    message = exception_text(exc) or ""
    if _PERMANENT_CONDITION_RE.search(message):
        return _decide(FailureClass.PERMANENT)

    if getattr(exc, "retryable", False):
        return _decide(_transient_class(category))
    if not hasattr(exc, "retryable") and category is not None and category.upper() in _TRANSIENT_CATEGORY_CLASSES:
        return _decide(_TRANSIENT_CATEGORY_CLASSES[category.upper()])
    return _decide(FailureClass.NOT_RETRYABLE)
