"""Unit tests for the single failure-classification source of truth.

``classify_failure`` collapses the two views that used to disagree (an opt-in
message heuristic and a diagnostic category) into one total decision. These
tests pin the classifier's contract directly; ``test_agent_retry.py`` proves the
production retry branch consumes it.
"""
from __future__ import annotations

import pytest

from daydream.agent import _ToolSupervisorFailure
from daydream.backends.pi import PiError
from daydream.retry_policy import FailureClass, classify_failure


def test_tool_policy_veto_is_permanent_and_never_retryable() -> None:
    decision = classify_failure(_ToolSupervisorFailure(RuntimeError("veto:Write")))

    assert decision.failure_class is FailureClass.TOOL_POLICY
    assert decision.retries_allowed is False


def test_permanent_condition_beats_transient_token_in_same_message() -> None:
    """A 503 in the message must not rescue a "model not found" failure."""
    decision = classify_failure(
        PiError("model not found: gpt-5 (503)", retryable=True, category="SERVER_ERROR")
    )

    assert decision.failure_class is FailureClass.PERMANENT
    assert decision.retries_allowed is False


def test_rate_limit_message_mentioning_provider_stays_transient() -> None:
    """A bare 'provider' token is not a permanent condition.

    The improve plan-writer's real-path rate limit (``category=RATE_LIMIT``,
    ``retryable=True``) carries the message "provider rate limit"; a generic
    'provider' substring must not veto an explicitly transient category and
    strand the retry ladder on its first failure.
    """
    decision = classify_failure(
        PiError("provider rate limit", retryable=True, category="RATE_LIMIT")
    )

    assert decision.failure_class is FailureClass.RATE_LIMIT
    assert decision.retries_allowed is True


def test_tool_policy_stop_attribute_is_permanent_and_never_retryable() -> None:
    class _ToolPolicyStop(Exception):
        tool_policy_stop = True

    decision = classify_failure(_ToolPolicyStop("veto:Write"))

    assert decision.failure_class is FailureClass.TOOL_POLICY
    assert decision.retries_allowed is False


def test_declared_class_wins_over_message_and_category() -> None:
    class _Declared(Exception):
        failure_class = FailureClass.PERMANENT

    decision = classify_failure(_Declared("429 rate limit exceeded"))

    assert decision.failure_class is FailureClass.PERMANENT
    assert decision.retries_allowed is False


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (PiError("429 rate limit exceeded", retryable=True, category="RATE_LIMIT"), FailureClass.RATE_LIMIT),
        (PiError("503 service unavailable", retryable=True, category="SERVER_ERROR"), FailureClass.SERVER_ERROR),
        (PiError("upstream timed out", retryable=True, category="TIMEOUT"), FailureClass.TIMEOUT),
        (PiError("stream terminated", retryable=True, category="STREAM_DROP"), FailureClass.TRANSPORT),
        (PiError("boom", retryable=True, category="UNKNOWN"), FailureClass.SERVER_ERROR),
    ],
)
def test_transient_failures_map_to_their_family(error: PiError, expected: FailureClass) -> None:
    decision = classify_failure(error)

    assert decision.failure_class is expected
    assert decision.retries_allowed is True


def test_permanent_attribute_beats_retryable_flag_and_transient_category() -> None:
    class _Permanent(Exception):
        permanent = True

    decision = classify_failure(_Permanent("429 rate limit exceeded"))

    assert decision.failure_class is FailureClass.PERMANENT
    assert decision.retries_allowed is False


def test_category_only_transient_failure_without_retryable_flag_is_retryable() -> None:
    class _CategoryOnly(Exception):
        category = "RATE_LIMIT"

    decision = classify_failure(_CategoryOnly("slow down"))

    assert decision.failure_class is FailureClass.RATE_LIMIT
    assert decision.retries_allowed is True


def test_plain_exception_is_not_retryable() -> None:
    decision = classify_failure(RuntimeError("something broke"))

    assert decision.failure_class is FailureClass.NOT_RETRYABLE
    assert decision.retries_allowed is False


def test_derive_retry_summary_ignores_deadline_only_stops() -> None:
    """A deadline stop's attempts/backend_s are useful work, not retry overhead."""
    from daydream.retry_policy import derive_retry_summary

    events = [
        {
            "event": "agent_budget_stop",
            "metadata": {
                "retry_stop_reason": None,
                "attempts": 1,
                "backend_s": 1_700.0,
                "backoff_s": 0.0,
                "circuit_state": "closed",
            },
        },
        {
            "event": "agent_budget_stop",
            "metadata": {
                "retry_stop_reason": "retry_recovery_allowance_exhausted",
                "attempts": 4,
                "backend_s": 12.0,
                "backoff_s": 30.0,
                "retry_recovery_spent_s": 42.0,
                "circuit_state": "open",
            },
        },
    ]

    summary = derive_retry_summary(events)

    assert summary == {
        "stops": {"retry_recovery_allowance_exhausted": 1},
        "attempts": 4,
        "backoff_s": 30.0,
        "backend_s": 12.0,
        "retry_recovery_spent_s": 42.0,
        "circuit_states": ["open"],
    }


def test_derive_retry_summary_is_none_when_only_a_deadline_stopped() -> None:
    """No retry-ladder stop means no summary, so a legacy manifest stays byte-identical."""
    from daydream.retry_policy import derive_retry_summary

    events = [
        {
            "event": "agent_budget_stop",
            "metadata": {
                "retry_stop_reason": None,
                "attempts": 1,
                "backend_s": 1_700.0,
                "backoff_s": 0.0,
            },
        }
    ]

    assert derive_retry_summary(events) is None


def test_classify_failure_never_raises_on_non_string_category_or_message() -> None:
    class _HostileError(Exception):
        category = 123

        def __str__(self) -> str:
            raise RuntimeError("no string for you")

    decision = classify_failure(_HostileError())

    assert decision.failure_class is FailureClass.NOT_RETRYABLE
    assert decision.retries_allowed is False
