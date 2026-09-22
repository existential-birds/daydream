"""Tests for the shared Claude-SDK mock/patch harness.

The harness stands in for the SDK types ``ClaudeBackend`` dispatches on, so its
own contract — the eight patched names, message replay, options capture, and the
``assistant_message`` / ``result_message`` override seam the metrics tests rely
on — is pinned here rather than inferred from its consumers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.backends import CostEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.claude import ClaudeBackend
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_claude_sdk,
    scripted_client,
)


async def _drive(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> list[Any]:
    patch_claude_sdk(monkeypatch, scripted_client(messages))
    backend = ClaudeBackend(model="opus")
    return [event async for event in backend.execute(Path("/tmp"), "go")]


@pytest.mark.asyncio
async def test_patched_blocks_reach_the_backend_as_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every mock block type the harness patches maps onto its AgentEvent."""
    events = await _drive(
        monkeypatch,
        [
            MockAssistantMessage(
                content=[
                    MockTextBlock(text="hello"),
                    MockToolUseBlock(id="t1", name="Read", input={"file_path": "a.py"}),
                ]
            ),
            MockUserMessage(content=[MockToolResultBlock(tool_use_id="t1", content="contents")]),
            MockResultMessage(total_cost_usd=0.02),
        ],
    )

    assert [e.text for e in events if isinstance(e, TextEvent)] == ["hello"]
    starts = [e for e in events if isinstance(e, ToolStartEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert (starts[0].name, starts[0].id) == ("Read", "t1")
    assert (results[0].id, results[0].output) == ("t1", "contents")
    assert [e.cost_usd for e in events if isinstance(e, CostEvent)] == [0.02]
    assert len([e for e in events if isinstance(e, ResultEvent)]) == 1
