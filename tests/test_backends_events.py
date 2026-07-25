"""Tests for the ``AgentEvent`` dataclasses in ``daydream/backends/__init__.py``.

Field values, nullable defaults, and the union/export surface all live here;
``tests/test_backends_init.py`` covers the ``Backend`` protocol and the
``create_backend`` factory. The two files previously asserted the same event
field defaults twice.

Covers Plan 02-02 of phase 02-recorder-core-event-enrichment-mapping:

- Every event dataclass carries a ``timestamp: str`` field defaulted via
  ``now_iso()`` (Pitfall 2 single source of truth).
- The ``MetricsEvent`` dataclass exists and uses the EVNT-02 verbatim
  field names (``prompt_tokens``, ``completion_tokens``, NOT
  ``input_tokens`` / ``output_tokens`` — those are the SDK boundary keys
  that backends rename when emitting MetricsEvent).
- ``CostEvent`` carries the ``cached_tokens`` field (default ``None``
  for backward compatibility with the existing 3-positional-arg call sites
  in ``backends/claude.py:124`` and ``backends/codex.py:310``).
- ``MetricsEvent`` is part of the ``AgentEvent`` TypeAlias union and is
  exported in ``__all__``.
"""

from __future__ import annotations

from typing import Any

import pytest

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)


def _assert_fields(event: Any, expected: dict[str, Any]) -> None:
    """Assert each expected field, by identity for ``None`` / bools."""
    for name, want in expected.items():
        got = getattr(event, name)
        if want is None or isinstance(want, bool):
            assert got is want, name
        else:
            assert got == want, name


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    [
        pytest.param(TextEvent, {"text": "hello"}, {"text": "hello"}, id="text-event"),
        pytest.param(
            ThinkingEvent, {"text": "reasoning..."}, {"text": "reasoning..."}, id="thinking-event"
        ),
        pytest.param(
            ToolStartEvent,
            {"id": "t1", "name": "Bash", "input": {"command": "ls"}},
            {"id": "t1", "name": "Bash", "input": {"command": "ls"}},
            id="tool-start-event",
        ),
        pytest.param(
            ToolResultEvent,
            {"id": "t1", "output": "file.py", "is_error": False},
            {"id": "t1", "output": "file.py", "is_error": False},
            id="tool-result-event",
        ),
        pytest.param(
            CostEvent,
            {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 50},
            {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 50, "cached_tokens": None},
            id="cost-event",
        ),
        pytest.param(
            CostEvent,
            {"cost_usd": None, "input_tokens": None, "output_tokens": None},
            {"cost_usd": None, "input_tokens": None, "output_tokens": None},
            id="cost-event-nullable",
        ),
        pytest.param(
            CostEvent,
            {"cost_usd": 0.5, "input_tokens": 10, "output_tokens": 20, "cached_tokens": 3},
            {"cached_tokens": 3},
            id="cost-event-cached-tokens",
        ),
        # Backward compat: the existing 3-arg call sites still work; cached_tokens defaults to None.
        pytest.param(
            CostEvent,
            {"cost_usd": 0.5, "input_tokens": 10, "output_tokens": 20},
            {"cached_tokens": None},
            id="cost-event-cached-tokens-default-none",
        ),
        pytest.param(
            ResultEvent,
            {"structured_output": None, "continuation": None},
            {"structured_output": None, "continuation": None},
            id="result-event-nullable",
        ),
        pytest.param(
            MetricsEvent,
            {
                "message_id": "msg_01",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 5,
                "cost_usd": 0.001,
            },
            {
                "message_id": "msg_01",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 5,
                "cost_usd": 0.001,
            },
            id="metrics-event",
        ),
    ],
)
def test_event_field_values(cls: type, kwargs: dict[str, Any], expected: dict[str, Any]) -> None:
    """Each event dataclass exposes the constructed values on the documented field names."""
    _assert_fields(cls(**kwargs), expected)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        pytest.param(TextEvent, {"text": "hi"}, id="text-event"),
        pytest.param(ThinkingEvent, {"text": "reasoning"}, id="thinking-event"),
        pytest.param(
            ToolStartEvent, {"id": "abc", "name": "Read", "input": {"file_path": "/tmp/a"}}, id="tool-start-event"
        ),
        pytest.param(ToolResultEvent, {"id": "abc", "output": "ok", "is_error": False}, id="tool-result-event"),
        pytest.param(
            CostEvent,
            {"cost_usd": 0.5, "input_tokens": 10, "output_tokens": 20, "cached_tokens": 3},
            id="cost-event",
        ),
        pytest.param(
            MetricsEvent,
            {
                "message_id": "msg_01",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 5,
                "cost_usd": 0.001,
            },
            id="metrics-event",
        ),
        pytest.param(ResultEvent, {"structured_output": None, "continuation": None}, id="result-event"),
        pytest.param(TurnEndEvent, {}, id="turn-end-event"),
    ],
)
def test_event_has_default_z_timestamp(cls: type, kwargs: dict[str, Any]) -> None:
    """Every member of the AgentEvent union defaults ``timestamp`` to a Z-suffixed stamp."""
    event = cls(**kwargs)
    assert isinstance(event.timestamp, str)
    assert event.timestamp.endswith("Z"), f"timestamp must end with Z: {event.timestamp!r}"


def test_result_event_carries_the_continuation_token() -> None:
    """ResultEvent holds the exact ContinuationToken instance it was given."""
    token = ContinuationToken(backend="codex", data={})
    event = ResultEvent(structured_output={"key": "val"}, continuation=token)
    assert event.structured_output == {"key": "val"}
    assert event.continuation is token


def test_metrics_event_in_agent_event_union() -> None:
    def accept(_e: AgentEvent) -> None:
        return None

    accept(
        MetricsEvent(
            message_id="x",
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=None,
            cost_usd=None,
        )
    )


def test_metrics_event_in_all_export() -> None:
    from daydream import backends

    assert "MetricsEvent" in backends.__all__


def test_turn_end_event_is_in_agent_event_union() -> None:
    """TurnEndEvent is a recognized AgentEvent so trajectory.py can dispatch."""
    ev = TurnEndEvent()
    ev2 = TurnEndEvent(message_id="msg_abc123")
    assert ev2.message_id == "msg_abc123"
    assert isinstance(ev.timestamp, str) and ev.timestamp.endswith("Z")
    # Runtime confirmation that TurnEndEvent is part of the AgentEvent union.
    assert isinstance(ev, AgentEvent)
