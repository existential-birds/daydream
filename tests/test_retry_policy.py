"""Pin shared failure classification; test_agent_retry covers its production consumer."""
from __future__ import annotations

from daydream.retry_policy import (
    FailureClass,
    classify_failure,
    decode_retry_seconds,
    derive_retry_summary,
    parse_message_retry_hint,
)


def test_host_attributes_win_over_transient_message_and_are_never_retryable() -> None:
    class _ToolPolicyStop(Exception):
        tool_policy_stop = True

    class _Declared(Exception):
        failure_class = FailureClass.PERMANENT

    class _Permanent(Exception):
        permanent = True

    for error, expected in [(_ToolPolicyStop("veto:Write"), FailureClass.TOOL_POLICY),
                            (_Declared("429 rate limit exceeded"), FailureClass.PERMANENT),
                            (_Permanent("429 rate limit exceeded"), FailureClass.PERMANENT)]:
        decision = classify_failure(error)
        assert decision.failure_class is expected
        assert decision.retries_allowed is False


def test_retry_hints_allowances_and_summaries_refuse_invalid_evidence() -> None:
    # No retry-ladder stop means no summary, so a legacy manifest stays byte-identical.
    events = [{"event": "agent_budget_stop",
               "metadata": {"retry_stop_reason": None, "attempts": 1, "backend_s": 1_700.0, "backoff_s": 0.0}}]
    assert derive_retry_summary(events) is None
    # One decode rule: numeric strings accepted, everything invalid refused.
    for raw in (True, "nonsense", None):
        assert decode_retry_seconds(raw) is None
    for message in ('429: {"metadata":{"headers":{"Other":"5","Retry-After":-1}}}',
                    '429: {"metadata":{"headers":"nope"}}',
                    '429: {"metadata":{"headers":{"Retry-After":"10"}'):
        assert parse_message_retry_hint(message) is None
