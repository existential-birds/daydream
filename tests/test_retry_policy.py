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


def test_classify_failure_never_raises_on_non_string_category_or_message() -> None:
    class _HostileError(Exception):
        category = 123

        def __str__(self) -> str:
            raise RuntimeError("no string for you")

    decision = classify_failure(_HostileError())

    assert decision.failure_class is FailureClass.NOT_RETRYABLE
    assert decision.retries_allowed is False
