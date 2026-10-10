"""Missing or partial Claude usage and usage-only terminal billing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    CostEvent,
    MetricsEvent,
    TextEvent,
)
from daydream.backends.claude import ClaudeBackend
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    patch_claude_sdk,
    scripted_client,
)


async def _collect_events(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> list[Any]:
    patch_claude_sdk(monkeypatch, scripted_client(messages))
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), "test prompt"):
        events.append(event)
    return events


async def test_no_metrics_event_when_usage_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessage(content=[MockTextBlock(text="ok")], message_id="msg_03", usage=None),
            MockResultMessage(total_cost_usd=0.001, structured_output=None, usage=None),
        ],
    )
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 0
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(text_events) == 1
    assert text_events[0].text == "ok"

async def test_partial_usage_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing input/output_tokens => no MetricsEvent (EVNT-02 types prompt/completion as int)."""
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessage(
                content=[MockTextBlock(text="ok")], message_id="msg_04",
                usage={"input_tokens": 100},  # output_tokens missing
            ), MockResultMessage(total_cost_usd=0.001, structured_output=None, usage={"input_tokens": 100}),
        ],
    )
    # No MetricsEvent because EVNT-02 requires both prompt_tokens and completion_tokens.
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 0
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert cost.input_tokens == 100
    assert cost.output_tokens is None

async def test_cost_event_emitted_on_usage_only(monkeypatch: pytest.MonkeyPatch) -> None:
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessage(content=[MockTextBlock(text="ok")], message_id="msg_05", usage=None),
            MockResultMessage(
                total_cost_usd=None, structured_output=None,
                usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
            ),
        ],
    )
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert cost.cost_usd is None
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50
