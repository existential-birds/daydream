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

import re
from dataclasses import dataclass
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
#: not exist.
_PERMANENT_CONDITION_RE = re.compile(
    r"credential"
    r"|api[ _-]?key"
    r"|not configured"
    r"|provider"
    r"|model not found"
    r"|schema validation"
    r"|additionalproperties"
    r"|not authenticated",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetryDecision:
    """The classifier's total verdict for one failure."""

    failure_class: FailureClass
    retries_allowed: bool


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
