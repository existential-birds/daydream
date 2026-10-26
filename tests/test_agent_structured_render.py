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

RAW = '{"conventions": [{"name": "OpenAPI First", "description": "x", "source": "AGENTS.md"}]}'
PAYLOAD = {"conventions": [{"name": "OpenAPI First", "description": "x", "source": "AGENTS.md"}]}

def _scripted(events: list[Any]) -> ScriptedBackend:
    """ScriptedBackend whose turn ends with an empty structured result."""
    return ScriptedBackend(events=[*events, ResultEvent(structured_output=None, continuation=None)], model="mock-model")

@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Install a recording console as ``daydream.agent.console`` and return it."""
    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", console)
    return console

async def test_structured_output_text_is_not_rendered(rec: Console, tmp_path: Path) -> None:
    backend = ScriptedBackend(
        events=[TextEvent(text=RAW), ResultEvent(structured_output=PAYLOAD, continuation=None)], model="mock-model",
    )
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"}
    )
    out = rec.export_text()
    assert result == PAYLOAD  # canonical structured result still returned
    assert "OpenAPI First" not in out  # raw JSON content NOT on the terminal
    assert "{" not in out


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

async def test_structured_fallback_recon_not_gated_all_or_nothing(rec: Console, tmp_path: Path) -> None:
    """Explicit validation opt-out lets recon salvage commands despite missing top-level fields.

    Ordinary callers retain validation; the opt-out is not implicit in the phase.
    """
    schema = {
        "type": "object", "required": ["languages", "commands", "conventions", "intent_docs"],
        "properties": {
            "languages": {"type": "array", "items": {"type": "string"}},
            "commands": {"type": "array", "items": {"type": "object"}},
            "conventions": {"type": "array", "items": {"type": "string"}},
            "intent_docs": {"type": "array", "items": {"type": "string"}},
        },
    }
    backend = _scripted([ TextEvent(text='{"commands": [{"command": "make test"}]}'), ])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.RECON, output_schema=schema, validate_structured_output=False,
    )
    assert result == {"commands": [{"command": "make test"}]}
    assert isinstance(result, dict)

async def test_structured_fallback_salvages_partial_dict(rec: Console, tmp_path: Path) -> None:
    """A valid top-level field reaches the consumer even if some nested records are invalid.

    Consumers discard bad records individually, preserving valid siblings.
    """
    schema = {
        "type": "object", "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object", "required": ["issue_id", "verdict", "evidence"],
                    "properties": {
                        "issue_id": {"type": "integer"}, "verdict": {"type": "string"}, "evidence": {"type": "string"},
                    },
                },
            }
        },
    }
    partial = {
        "verdicts": [
            {"issue_id": 1, "verdict": "consistent", "evidence": "matches"},
            {"issue_id": 2, "verdict": "bogus"},  # missing required "evidence"
        ]
    }
    backend = _scripted([ TextEvent(text=json.dumps(partial)), ])
    result, _, _ = await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.VERIFY, output_schema=schema)
    assert result == partial  # the partial dict reaches the salvage path
    assert isinstance(result, dict)

async def test_structured_fallback_requires_merge_envelope(rec: Console, tmp_path: Path) -> None:
    """A bare array remains text and cannot establish structured output."""
    schema = {
        "type": "object", "required": ["items"],
        "properties": {"items": {"type": "array", "items": {"type": "object"}}},
    }
    items = [{"id": 1, "description": "x"}]
    backend = _scripted([TextEvent(text=json.dumps(items))])
    result, _, _ = await run_agent(backend, tmp_path, "merge", phase=DaydreamPhase.DEEP, output_schema=schema)
    assert result == json.dumps(items)
    assert isinstance(result, str)


@pytest.mark.parametrize("primary", [None, {"issues": "invalid", "description": "REJECTED_RESULT"}])
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
