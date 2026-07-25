"""Tests for Claude backend token extraction and MetricsEvent emission (EVNT-04..06).

Covers Phase 2 / Plan 02-03:
  * EVNT-04: dropped-token bug fix in `daydream/backends/claude.py` lines 120-128
  * EVNT-05: CostEvent now carries `cached_tokens` from `cache_read_input_tokens`
  * EVNT-06: MetricsEvent emitted per AssistantMessage with EVNT-02 verbatim
    field names (`prompt_tokens` / `completion_tokens`); the SDK boundary
    keys (`input_tokens` / `output_tokens`) are renamed at emission time.

Reuses the shared mock-block dataclasses in tests/harness/claude_sdk.py and
extends MockAssistantMessage / MockResultMessage with the `usage` and
`message_id` fields needed by Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# Phase 2 extensions: AssistantMessage needs message_id + usage; ResultMessage needs usage.
@dataclass
class MockAssistantMessageWithUsage(MockAssistantMessage):
    """Shared MockAssistantMessage plus EVNT-06 fields (message_id, usage)."""

    message_id: str = ""
    usage: dict[str, Any] | None = None


@dataclass
class MockResultMessageWithUsage(MockResultMessage):
    """Shared MockResultMessage plus the EVNT-04/05 usage field."""

    usage: dict[str, Any] | None = None


async def _collect_events(monkeypatch, messages: list[Any]) -> list[Any]:
    """Drive ClaudeBackend.execute with a canned message sequence; return events."""
    patch_claude_sdk(
        monkeypatch,
        scripted_client(messages),
        assistant_message=MockAssistantMessageWithUsage,
        result_message=MockResultMessageWithUsage,
    )
    backend = ClaudeBackend(model="opus")
    events: list[Any] = []
    async for event in backend.execute(Path("/tmp"), "test prompt"):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_dropped_token_bug_fixed(monkeypatch):
    """ResultMessage with usage produces CostEvent with non-None tokens (EVNT-04, EVNT-05)."""
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(
                content=[MockTextBlock(text="Reviewing")],
                message_id="msg_01",
                usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 30},
            ),
            MockResultMessageWithUsage(
                total_cost_usd=0.001,
                structured_output=None,
                usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 30},
            ),
        ],
    )
    cost = next(e for e in events if isinstance(e, CostEvent))
    assert cost.input_tokens == 130  # 100 uncached + 30 read folded into the total
    assert cost.output_tokens == 50
    assert cost.cached_tokens == 30
    assert cost.cost_usd == 0.001


@pytest.mark.asyncio
async def test_metrics_event_emitted_per_assistant_message(monkeypatch):
    """AssistantMessage with usage produces a MetricsEvent with EVNT-02 field names (EVNT-06)."""
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(
                content=[MockTextBlock(text="Hi")],
                message_id="msg_01",
                usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 30},
            ),
            MockResultMessageWithUsage(total_cost_usd=0.001, structured_output=None, usage=None),
        ],
    )
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 1
    m = metrics[0]
    assert m.message_id == "msg_01"
    assert m.prompt_tokens == 130  # EVNT-02 verbatim name; 100 uncached + 30 read folded in
    assert m.completion_tokens == 50  # EVNT-02 verbatim name (NOT output_tokens)
    assert m.cached_tokens == 30
    assert m.cost_usd is None  # AssistantMessage carries no per-message cost


@pytest.mark.asyncio
async def test_prompt_tokens_include_cache_read_and_creation(monkeypatch):
    """prompt_tokens folds input + cache_read + cache_creation into the true total input."""
    # Fully-cached turn: raw input_tokens is the uncached remainder (22); the total
    # input the model actually processed is 22 + 20000 read = 20022.
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(
                content=[MockTextBlock(text="cached")],
                message_id="msg_cached",
                usage={
                    "input_tokens": 22,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 20000,
                    "cache_creation_input_tokens": 0,
                },
            ),
            MockResultMessageWithUsage(
                total_cost_usd=0.002,
                structured_output=None,
                usage={
                    "input_tokens": 22,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 20000,
                    "cache_creation_input_tokens": 0,
                },
            ),
        ],
    )
    metrics = [e for e in events if isinstance(e, MetricsEvent)][0]
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert metrics.prompt_tokens == 20022
    assert metrics.cached_tokens == 20000
    assert metrics.completion_tokens == 100
    assert cost.input_tokens == 20022
    assert cost.cached_tokens == 20000

    # Cache-write turn: creation tokens fold in too; a write is not a read hit, so
    # cached_tokens stays 0.
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(
                content=[MockTextBlock(text="write")],
                message_id="msg_write",
                usage={
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 15000,
                },
            ),
            MockResultMessageWithUsage(
                total_cost_usd=0.003,
                structured_output=None,
                usage={
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 15000,
                },
            ),
        ],
    )
    metrics = [e for e in events if isinstance(e, MetricsEvent)][0]
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert metrics.prompt_tokens == 15050
    assert metrics.cached_tokens == 0
    assert cost.input_tokens == 15050
    assert cost.cached_tokens == 0

    # Read-and-write turn: the buckets are not mutually exclusive. One breakpoint
    # is read (18000) while another is written (12000); both fold into the total
    # (40 + 18000 + 12000 = 30040), but cached_tokens reflects only the read hit.
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(
                content=[MockTextBlock(text="both")],
                message_id="msg_both",
                usage={
                    "input_tokens": 40,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 18000,
                    "cache_creation_input_tokens": 12000,
                },
            ),
            MockResultMessageWithUsage(
                total_cost_usd=0.004,
                structured_output=None,
                usage={
                    "input_tokens": 40,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 18000,
                    "cache_creation_input_tokens": 12000,
                },
            ),
        ],
    )
    metrics = [e for e in events if isinstance(e, MetricsEvent)][0]
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert metrics.prompt_tokens == 30040
    assert metrics.cached_tokens == 18000
    assert cost.input_tokens == 30040
    assert cost.cached_tokens == 18000


@pytest.mark.asyncio
async def test_prompt_tokens_is_total_of_all_input_buckets(monkeypatch):
    """prompt_tokens is the total (uncached + read + creation); cached_tokens is the read subset."""
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(
                content=[MockTextBlock(text="ok")],
                message_id="msg_02",
                usage={"input_tokens": 500, "output_tokens": 200, "cache_read_input_tokens": 300},
            ),
            MockResultMessageWithUsage(
                total_cost_usd=0.005,
                structured_output=None,
                usage={"input_tokens": 500, "output_tokens": 200, "cache_read_input_tokens": 300},
            ),
        ],
    )
    metrics = [e for e in events if isinstance(e, MetricsEvent)][0]
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert metrics.prompt_tokens == 800  # 500 uncached + 300 read
    assert metrics.cached_tokens == 300
    assert cost.input_tokens == 800
    assert cost.cached_tokens == 300


@pytest.mark.asyncio
async def test_no_metrics_event_when_usage_is_none(monkeypatch):
    """AssistantMessage.usage None => no MetricsEvent emitted, but TextEvent still flows."""
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(content=[MockTextBlock(text="ok")], message_id="msg_03", usage=None),
            MockResultMessageWithUsage(total_cost_usd=0.001, structured_output=None, usage=None),
        ],
    )
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 0
    # TextEvent flow continues normally
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(text_events) == 1
    assert text_events[0].text == "ok"


@pytest.mark.asyncio
async def test_partial_usage_data(monkeypatch):
    """Missing input/output_tokens => no MetricsEvent (EVNT-02 types prompt/completion as int)."""
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(
                content=[MockTextBlock(text="ok")],
                message_id="msg_04",
                usage={"input_tokens": 100},  # output_tokens missing
            ),
            MockResultMessageWithUsage(
                total_cost_usd=0.001,
                structured_output=None,
                usage={"input_tokens": 100},
            ),
        ],
    )
    # No MetricsEvent because EVNT-02 requires both prompt_tokens and completion_tokens.
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 0
    # CostEvent still emitted with output_tokens=None (CostEvent fields are Optional).
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert cost.input_tokens == 100
    assert cost.output_tokens is None


@pytest.mark.asyncio
async def test_cost_event_emitted_on_usage_only(monkeypatch):
    """ResultMessage with usage but total_cost_usd=None still emits CostEvent."""
    events = await _collect_events(
        monkeypatch,
        [
            MockAssistantMessageWithUsage(content=[MockTextBlock(text="ok")], message_id="msg_05", usage=None),
            MockResultMessageWithUsage(
                total_cost_usd=None,
                structured_output=None,
                usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0},
            ),
        ],
    )
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert cost.cost_usd is None
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50
