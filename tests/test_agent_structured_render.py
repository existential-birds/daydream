"""Agent terminal rendering, log redaction, and structured-result/fallback precedence."""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.agent import run_agent
from daydream.backends import (
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.run_context import InteractionPolicy, RunContext
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend


def _scripted(events: list[Any]) -> ScriptedBackend:
    """ScriptedBackend whose turn ends with an empty structured result."""
    return ScriptedBackend(events=[*events, ResultEvent(structured_output=None, continuation=None)], model="mock-model")

@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Install a recording console as ``daydream.agent.console`` and return it."""
    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", console)
    return console

async def test_log_mode_emission_redacts_sentinels_on_agent_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Redact every logged event type while returning the original structured result."""
    sentinel = "ghp_" + "x" * 16
    payload = {
        "status": "complete", "token": sentinel,
        "conventions": [{"name": "OpenAPI First", "description": "x", "source": "AGENTS.md"}],
        "nested": {"path": f"/Users/{sentinel}"},
    }
    backend = ScriptedBackend(
        events=[
            TextEvent(text=f"All source artifacts are empty: `stack-python-records.json` is `[]`. token={sentinel}"),
            ThinkingEvent(text=f"thinking about {sentinel}"),
            ToolStartEvent(id="t", name="bash", input={"command": f"echo {sentinel}"}),
            ToolResultEvent(id="t", output=f"token={sentinel}", is_error=False),
            ResultEvent(structured_output=payload, continuation=None),
        ], model="mock-model",
    )
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"},
        run_context=RunContext(InteractionPolicy(log_mode=True)),
    )
    assert result == payload          # returned object is raw/unchanged
    out = capsys.readouterr().out
    assert "[REDACTED_API_KEY]" in out  # markers present
    assert sentinel not in out          # no raw leak through the agent's log-mode emission
    assert out.count("[result]") == 1 and "OpenAPI First" in out

@pytest.mark.parametrize("primary", [{"issues": "invalid", "description": "REJECTED_RESULT"}])
async def test_log_presents_only_resolved_result_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], primary: Any,
) -> None:
    sentinel = "ghp_" + "x" * 16
    payload = {"issues": [{"id": 1, "description": "Accepted " + sentinel}]}
    backend = ScriptedBackend(events=[TextEvent(text=json.dumps(payload)),
                                      ResultEvent(structured_output=primary, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "review", phase=DaydreamPhase.REVIEW,
        output_schema={"type": "object", "required": ["issues"], "properties": {"issues": {"type": "array"}}},
        run_context=RunContext(InteractionPolicy(log_mode=True)),
    )
    assert result == payload
    output = capsys.readouterr().out
    assert output.count("[result]") == 1 and "[result] {\"issues\": [" in output
    assert "REJECTED_RESULT" not in output and sentinel not in output
    assert "[REDACTED_API_KEY]" in output


async def test_salvageable_text_requires_lists_in_required_array_slots(tmp_path: Path) -> None:
    """Without full validation, prose JSON is admitted only when every required array slot holds a list."""
    schema = {"type": "object", "required": ["issues", "summary"],
              "properties": {"issues": {"type": "array"}, "summary": {"type": "string"}}}
    salvageable = '{"issues": [], "summary": 5}'
    result, _, _ = await run_agent(_scripted([TextEvent(text=salvageable)]), tmp_path, "review",
                                   phase=DaydreamPhase.REVIEW, output_schema=schema)
    assert result == {"issues": [], "summary": 5}
    result, _, _ = await run_agent(_scripted([TextEvent(text='{"issues": "none", "summary": "ok"}')]), tmp_path,
                                   "review", phase=DaydreamPhase.REVIEW, output_schema=schema)
    assert result != {"issues": "none", "summary": "ok"}
