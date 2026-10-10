"""Pin shared failure classification; test_agent_retry covers its production consumer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import _ToolSupervisorFailure
from daydream.backends import BackendExecutionInput
from daydream.backends.pi import PiError
from daydream.config_file import load_file_config
from daydream.retry_policy import (
    FailureClass,
    classify_failure,
    coerce_declared_retry_allowance,
    decode_retry_recovery_allowance,
    derive_retry_summary,
    parse_message_retry_hint,
)


def test_tool_policy_veto_is_permanent_and_never_retryable() -> None:
    decision = classify_failure(_ToolSupervisorFailure(RuntimeError("veto:Write")))

    assert decision.failure_class is FailureClass.TOOL_POLICY
    assert decision.retries_allowed is False


def test_rate_limit_message_mentioning_provider_stays_transient() -> None:
    """Generic "provider" text must not override an explicitly transient rate-limit category."""
    decision = classify_failure(PiError("provider rate limit", retryable=True, category="RATE_LIMIT"))

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


def test_derive_retry_summary_ignores_deadline_only_stops() -> None:
    """A deadline stop's attempts/backend_s are useful work, not retry overhead."""

    events = [{"event": "agent_budget_stop",
            "metadata": {"retry_stop_reason": None, "attempts": 1, "backend_s": 1_700.0, "backoff_s": 0.0,
                "circuit_state": "closed",
            },
        }, {"event": "agent_budget_stop",
            "metadata": {"retry_stop_reason": "retry_recovery_allowance_exhausted", "attempts": 4, "backend_s": 12.0,
                "backoff_s": 30.0, "retry_recovery_spent_s": 42.0, "circuit_state": "open",
            },
        },
    ]

    summary = derive_retry_summary(events)

    assert summary == {
        "stops": {"retry_recovery_allowance_exhausted": 1}, "attempts": 4, "backoff_s": 30.0, "backend_s": 12.0,
        "retry_recovery_spent_s": 42.0, "circuit_states": ["open"],
    }

def test_derive_retry_summary_is_none_when_only_a_deadline_stopped() -> None:
    """No retry-ladder stop means no summary, so a legacy manifest stays byte-identical."""

    events = [{"event": "agent_budget_stop",
            "metadata": {"retry_stop_reason": None, "attempts": 1, "backend_s": 1_700.0, "backoff_s": 0.0},
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

@pytest.mark.parametrize(("raw", "expected"),
    [(5, 5.0), (0, 0.0), (42.5, 42.5), ("42", 42.0), ("0", 0.0), (True, None), (False, None), (-1, None),
        ("-1", None), ("nonsense", None), (float("nan"), None), (float("inf"), None), (None, None), ([], None),
    ],
)
def test_the_shared_allowance_decoder_is_one_rule(raw: Any, expected: float | None) -> None:
    """One decode rule: numeric strings accepted, everything invalid refused."""

    assert decode_retry_recovery_allowance(raw) == expected

def test_the_shared_allowance_helper_is_the_one_declared_value_rule() -> None:
    """One declared-value rule: what is admitted, and what falls through to the default."""

    assert coerce_declared_retry_allowance(42, "retry_recovery_allowance_s") == 42.0
    assert coerce_declared_retry_allowance("0", "retry_recovery_allowance_s") == 0.0
    assert coerce_declared_retry_allowance(-1, "retry_recovery_allowance_s") is None
    assert coerce_declared_retry_allowance("nonsense", "retry_recovery_allowance_s") is None

def test_an_absent_allowance_declaration_is_silent_on_every_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An absent key is not a declaration: none of the None-bearing sources warns.

    ``_resolve_retry_settings`` passes only a source ``_allowance_source`` actually
    found, so these two plus the helper cover every ``None`` caller of the rule.
    """
    with caplog.at_level(logging.WARNING):
        assert coerce_declared_retry_allowance(None, "retry_recovery_allowance_s") is None
        assert load_file_config(tmp_path).retry_recovery_allowance_s is None
        undeclared = BackendExecutionInput.from_environment({}, backend="pi")
        assert undeclared.retry_policy.retry_recovery_allowance_s is None
    assert not [record for record in caplog.records if "retry_recovery_allowance" in record.message]

def test_a_refused_allowance_warning_names_the_source_and_the_value(caplog: pytest.LogCaptureFixture,) -> None:
    """One shared warning shape: the source, the raw value, and no restated bound."""


    with caplog.at_level(logging.WARNING):
        assert coerce_declared_retry_allowance(-1, "retry_recovery_allowance_s") is None

    message = caplog.records[-1].message
    assert "retry_recovery_allowance_s=-1" in message
    assert "stays undeclared" in message

@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # Structured, status-prefixed (the issue's sanitized payload).
        pytest.param(
            '429: {"message":"Temporary admission failure","code":429,'
            '"metadata":{"headers":{"Retry-After":"10"}}}', 10.0, id="issue-429",
        ),
        pytest.param('429: {"metadata":{"headers":{"retry-after":"7"}}}', 7.0, id="lowercase-key"),
        pytest.param('429: {"metadata":{"headers":{"RETRY-AFTER":"7"}}}', 7.0, id="upper-key"),
        pytest.param('429: {"metadata":{"headers":{"Retry-After":2.5}}}', 2.5, id="fractional-number"),
        pytest.param('429: {"metadata":{"headers":{"Retry-After":10}}}', 10.0, id="bare-number"),
        pytest.param('429: {"metadata":{"headers":{"Retry-After":"0"}}}', 0.0, id="zero-string"),
        # Refusal: the shared decoder governs; never a second numeric rule.
        pytest.param('429: {"metadata":{"headers":{"Retry-After":"-5"}}}', None, id="negative"),
        pytest.param('429: {"metadata":{"headers":{"Retry-After":true}}}', None, id="boolean"),
        pytest.param('429: {"metadata":{"headers":{"Retry-After":"nonsense"}}}', None, id="text"),
        pytest.param(
            '429: {"metadata":{"headers":{"Retry-After":"Wed, 21 Oct 2026 07:28:00 GMT"}}}',
            None, id="http-date-out-of-scope",
        ),
        # Guarded shapes: no hint, no raise, text fallback stays live.
        pytest.param('429: {"metadata":{"headers":"nope"}}', None, id="headers-not-mapping"),
        pytest.param('429: {"metadata":"nope"}', None, id="metadata-not-mapping"),
        pytest.param('429: {"code":429}', None, id="no-headers"),
        pytest.param('429: {"metadata":{"headers":{"Retry-After":"10"}', None, id="malformed-json"),
        # Preserved plain-text format (req 6): regex fallback.
        pytest.param("503 Service Unavailable; retry-after: 30", 30.0, id="plain-text"),
        pytest.param("503 retry-after 45", 45.0, id="plain-whitespace"),
        pytest.param("503 Service Unavailable", None, id="no-hint-plain"),
        pytest.param("", None, id="empty"),
        # Structured precedence (Should-Have 2): structured wins over text, deterministically.
        pytest.param(
            '429: {"metadata":{"headers":{"Retry-After":"10"}}} retry-after: 99', 10.0, id="structured-wins",
        ),
    ],
)
def test_parse_message_retry_hint_reads_structured_headers(message: str, expected: float | None) -> None:
    assert parse_message_retry_hint(message) == expected
