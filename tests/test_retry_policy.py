"""Unit tests for the single failure-classification source of truth.

``classify_failure`` collapses the two views that used to disagree (an opt-in
message heuristic and a diagnostic category) into one total decision. These
tests pin the classifier's contract directly; ``test_agent_retry.py`` proves the
production retry branch consumes it.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

from daydream.agent import _coerce_retry_recovery_allowance, _ToolSupervisorFailure
from daydream.backends import BackendExecutionInput
from daydream.backends.pi import PiError
from daydream.config_file import _coerce_retry_recovery_allowance as file_coerce
from daydream.retry_policy import FailureClass, classify_failure, decode_retry_recovery_allowance, derive_retry_summary


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5, 5.0),
        (0, 0.0),
        (42.5, 42.5),
        ("42", 42.0),
        ("0", 0.0),
        (True, None),
        (False, None),
        (-1, None),
        ("-1", None),
        ("nonsense", None),
        (float("nan"), None),
        (float("inf"), None),
        (None, None),
        ([], None),
    ],
)
def test_the_shared_allowance_decoder_is_one_rule(raw: Any, expected: float | None) -> None:
    """One decode rule: numeric strings accepted, everything invalid refused."""

    assert decode_retry_recovery_allowance(raw) == expected


def test_every_allowance_source_decodes_with_the_same_rule() -> None:
    """The three decode sites cannot disagree about one value.

    The config-file key, the explicit argument and the env var each used to carry
    their own copy of the rule -- with three different warning texts and a real
    drift hazard -- so agreement is pinned here rather than assumed.
    """

    for raw in (5, 0, 42.5, "42", "0", True, False, -1, "-1", "nonsense", None):
        argument = _coerce_retry_recovery_allowance(raw, "retry_recovery_allowance_s")
        file_value = file_coerce({"retry_recovery_allowance_s": raw})
        assert argument == file_value, raw

    # The env source only ever sees strings; it must agree with them too.
    for raw in ("5", "0", "42.5", "nonsense", "-1", "nan", "inf"):
        embedded = BackendExecutionInput.from_environment(
            {"DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S": raw}, backend="pi"
        ).retry_policy.retry_recovery_allowance_s
        assert embedded == _coerce_retry_recovery_allowance(raw, "DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S"), raw


def test_a_refused_allowance_warning_names_the_source_and_the_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One shared warning shape: the source, the raw value, and no restated bound."""


    with caplog.at_level(logging.WARNING):
        assert _coerce_retry_recovery_allowance(-1, "retry_recovery_allowance_s") is None

    message = caplog.records[-1].message
    assert "retry_recovery_allowance_s=-1" in message
    assert "stays undeclared" in message
