"""Task 1 real-path tests: run_agent must not echo structured-output JSON.

Verified terminal-render harness (from Task 0):
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    ... drive run_agent ...
    rec.export_text()  # captures the rendered agent text

run_agent requires the keyword-only `phase=` argument (DaydreamPhase),
imported from daydream.trajectory. MockBackend mirrors the dataclass at
tests/test_agent_recorder_integration.py:61-96.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import DaydreamPhase

RAW = '{"conventions": [{"name": "OpenAPI First", "description": "x", "source": "CLAUDE.md"}]}'
PAYLOAD = {"conventions": [{"name": "OpenAPI First", "description": "x", "source": "CLAUDE.md"}]}


@dataclass
class MockBackend:
    """Minimal Backend that replays a canned event list (mirrors the protocol)."""

    model = "mock-model"
    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        events = self.events

        async def _gen() -> AsyncIterator[AgentEvent]:
            for event in events:
                yield event

        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_structured_output_text_is_not_rendered(monkeypatch, tmp_path):
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    backend = MockBackend([TextEvent(text=RAW), ResultEvent(structured_output=PAYLOAD, continuation=None)])
    result, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"}
    )
    out = rec.export_text()
    assert result == PAYLOAD  # canonical structured result still returned
    assert "OpenAPI First" not in out  # raw JSON content NOT on the terminal
    assert "{" not in out


async def test_plain_text_still_renders(monkeypatch, tmp_path):
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    backend = MockBackend(
        [TextEvent(text="narration here"), ResultEvent(structured_output=None, continuation=None)]
    )
    result, _ = await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)  # no output_schema
    assert "narration here" in rec.export_text()
