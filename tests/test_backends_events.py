"""Smoke tests for AgentEvent enrichment (EVNT-01..03).

These tests cover Plan 02-02 of phase 02-recorder-core-event-enrichment-mapping:

- Every event dataclass in ``daydream/backends/__init__.py`` carries a
  ``timestamp: str`` field defaulted via ``now_iso()`` (Pitfall 2 single
  source of truth).
- The new ``MetricsEvent`` dataclass exists and uses the EVNT-02 verbatim
  field names (``prompt_tokens``, ``completion_tokens``, NOT
  ``input_tokens`` / ``output_tokens`` — those are the SDK boundary keys
  that backends rename when emitting MetricsEvent).
- ``CostEvent`` carries the new ``cached_tokens`` field (default ``None``
  for backward compatibility with the existing 3-positional-arg call sites
  in ``backends/claude.py:124`` and ``backends/codex.py:310``; Plans 03/04
  update those sites to populate the new field).
- ``MetricsEvent`` is part of the ``AgentEvent`` TypeAlias union and is
  exported in ``__all__``.
"""

from __future__ import annotations

from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)


def test_text_event_has_default_z_timestamp() -> None:
    e = TextEvent(text="hi")
    assert e.timestamp.endswith("Z"), f"timestamp must end with Z: {e.timestamp!r}"


def test_thinking_event_has_default_z_timestamp() -> None:
    e = ThinkingEvent(text="reasoning")
    assert e.timestamp.endswith("Z")


def test_tool_start_event_default_timestamp() -> None:
    e = ToolStartEvent(id="abc", name="Read", input={"file_path": "/tmp/a"})
    assert e.timestamp.endswith("Z")


def test_tool_result_event_default_timestamp() -> None:
    e = ToolResultEvent(id="abc", output="ok", is_error=False)
    assert e.timestamp.endswith("Z")


def test_metrics_event_construction() -> None:
    e = MetricsEvent(
        message_id="msg_01",
        prompt_tokens=10,
        completion_tokens=20,
        cached_tokens=5,
        cost_usd=0.001,
    )
    assert e.message_id == "msg_01"
    assert e.prompt_tokens == 10
    assert e.completion_tokens == 20
    assert e.cached_tokens == 5
    assert e.cost_usd == 0.001
    assert e.timestamp.endswith("Z")


def test_cost_event_has_cached_tokens() -> None:
    e = CostEvent(cost_usd=0.5, input_tokens=10, output_tokens=20, cached_tokens=3)
    assert e.cached_tokens == 3
    assert e.timestamp.endswith("Z")


def test_cost_event_cached_tokens_default_none() -> None:
    """Backward compat: existing 3-arg call sites still work; cached_tokens defaults to None."""
    e = CostEvent(cost_usd=0.5, input_tokens=10, output_tokens=20)
    assert e.cached_tokens is None
    assert e.timestamp.endswith("Z")


def test_result_event_default_timestamp() -> None:
    e = ResultEvent(structured_output=None, continuation=None)
    assert e.timestamp.endswith("Z")


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
