"""Task 1 real-path tests: run_agent must not echo structured-output JSON.

Verified terminal-render harness (from Task 0):
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    ... drive run_agent ...
    rec.export_text()  # captures the rendered agent text

run_agent requires the keyword-only `phase=` argument (DaydreamPhase),
imported from daydream.trajectory. MockBackend is imported from
tests.test_agent_recorder_integration (the single canonical definition).
"""
from __future__ import annotations

import copy
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
from daydream.trajectory import DaydreamPhase
from tests.test_agent_recorder_integration import MockBackend

RAW = '{"conventions": [{"name": "OpenAPI First", "description": "x", "source": "CLAUDE.md"}]}'
PAYLOAD = {"conventions": [{"name": "OpenAPI First", "description": "x", "source": "CLAUDE.md"}]}


def test_redact_log_value_recurses_without_mutating() -> None:
    """_redact_log_value redacts string keys AND values recursively, in fresh
    containers, and never mutates its argument."""
    from daydream.agent import _redact_log_value

    sentinel = "ghp_" + "x" * 16
    payload = {
        "token": sentinel,
        sentinel: "key-that-is-a-secret",
        "nested": {"path": f"/Users/{sentinel}"},
        "items": [sentinel, 42, None],
        "flag": True,
        1: "non-string-key",
    }
    original = copy.deepcopy(payload)
    out = _redact_log_value(payload)

    assert payload == original                    # argument never mutated
    assert out is not payload                     # fresh top-level container
    assert out["nested"] is not payload["nested"]  # fresh nested container
    assert sentinel not in json.dumps(out)        # redacted in values AND keys
    assert "[REDACTED" in json.dumps(out)         # a marker replaced it
    assert out["items"] == ["[REDACTED_API_KEY]", 42, None]  # scalars preserved
    assert out["flag"] is True
    assert out[1] == "non-string-key"             # non-string keys untouched


async def test_structured_output_text_is_not_rendered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    backend = MockBackend([TextEvent(text=RAW), ResultEvent(structured_output=PAYLOAD, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"}
    )
    out = rec.export_text()
    assert result == PAYLOAD  # canonical structured result still returned
    assert "OpenAPI First" not in out  # raw JSON content NOT on the terminal
    assert "{" not in out


async def test_plain_text_still_renders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    backend = MockBackend(
        [TextEvent(text="narration here"), ResultEvent(structured_output=None, continuation=None)]
    )
    result, _, _ = await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)  # no output_schema
    assert "narration here" in rec.export_text()


async def test_log_mode_emission_redacts_sentinels_on_agent_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Real-path sentinel-absence check at the agent boundary (no phases.py
    prints): every --log event type is emitted redacted — markers present, raw
    sentinel absent — while the returned structured result stays raw."""
    sentinel = "ghp_" + "x" * 16
    payload = {"status": "complete", "token": sentinel}
    backend = MockBackend(
        [
            TextEvent(text=f"token={sentinel}"),
            ThinkingEvent(text=f"thinking about {sentinel}"),
            ToolStartEvent(id="t", name="bash", input={"command": f"echo {sentinel}"}),
            ToolResultEvent(id="t", output=f"token={sentinel}", is_error=False),
            ResultEvent(structured_output=payload, continuation=None),
        ]
    )
    monkeypatch.setattr("daydream.agent._state.log_mode", True)
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"}
    )
    assert result == payload          # returned object is raw/unchanged
    out = capsys.readouterr().out
    assert "[REDACTED_API_KEY]" in out  # markers present
    assert sentinel not in out          # no raw leak through the agent's log-mode emission


async def test_log_mode_captures_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Under --log, the structured result is still captured, NOT just printed —
    and the printed projection is redacted while the returned object stays raw."""
    sentinel = "ghp_" + "x" * 16
    payload = {
        "conventions": [{"name": "OpenAPI First", "description": "x", "source": "CLAUDE.md"}],
        "token": sentinel,
        "nested": {"path": f"/Users/{sentinel}"},
    }
    monkeypatch.setattr("daydream.agent._state.log_mode", True)
    backend = MockBackend([ResultEvent(structured_output=payload, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"}
    )
    assert result == payload        # captured AND returned raw (never redacted)
    out = capsys.readouterr().out
    assert "[result]" in out        # log-mode print is additive, still happens
    assert sentinel not in out      # the stdout projection is redacted
    assert "OpenAPI First" in out   # benign content still serialized


async def test_log_mode_structured_result_wins_over_prose_stray_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Under --log, prose containing stray JSON must not be scraped over the real result.

    Regression for the deep cross-stack merge crash ("Cross-stack merge returned
    no item list (got list)"): the merge agent narrates in prose while emitting a
    ``{"items": [...]}`` structured result. When --log dropped the captured
    structured result (the if/elif bug), run_agent fell through to the JSON
    fallback, which scraped the stray ``[]`` out of prose like
    "all source artifacts are empty: `[]`" and returned a bare list. The merge
    phase's ``isinstance(result, dict)`` check then failed with "got list".

    With the fix the captured structured result wins, so the payload survives as a
    dict and the fallback never runs.
    """
    monkeypatch.setattr("daydream.agent._state.log_mode", True)
    merge_prose = "All source artifacts are empty: `stack-python-records.json` is `[]`. Nothing to merge."
    payload: dict[str, Any] = {"items": []}
    backend = MockBackend(
        [
            TextEvent(text=merge_prose),
            ResultEvent(structured_output=payload, continuation=None),
        ]
    )
    result, _, _ = await run_agent(
        backend, tmp_path, "merge", phase=DaydreamPhase.DEEP, output_schema={"type": "object"}
    )
    assert result == payload  # the captured dict, NOT the stray [] scraped from prose
    assert isinstance(result, dict)  # the exact type the merge phase gate requires


async def test_structured_fallback_validates_against_output_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Must-haves #4/#5: with output_schema set and structured output failing,
    (a) valid-schema JSON is returned as structured output, and (b) invalid-
    schema JSON falls through to the plain-text return."""
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    schema = {
        "type": "object",
        "required": ["file"],
        "properties": {"file": {"type": "string"}},
    }

    # (a) valid-schema raw JSON -> returned as structured output
    valid_backend = MockBackend([
        TextEvent(text='{"file": "src/a.py"}'),
        ResultEvent(structured_output=None, continuation=None),
    ])
    result, _, _ = await run_agent(
        valid_backend, tmp_path, "go", phase=DaydreamPhase.REVIEW, output_schema=schema
    )
    assert result == {"file": "src/a.py"}

    # (b) invalid-schema raw JSON (missing required "file") -> plain-text fallthrough
    invalid_backend = MockBackend([
        TextEvent(text='{"line": 3}'),
        ResultEvent(structured_output=None, continuation=None),
    ])
    result2, _, _ = await run_agent(
        invalid_backend, tmp_path, "go", phase=DaydreamPhase.REVIEW, output_schema=schema
    )
    assert result2 == '{"line": 3}'
    assert isinstance(result2, str)


async def test_structured_fallback_recon_not_gated_all_or_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RECON's fallback skips the structured-output gate via the caller-declared
    ``validate_structured_output=False`` kwarg (not a phase carve-out): the
    top-level RECON schema requires languages/commands/conventions/intent_docs,
    but the orchestrator salvages per-command records downstream via
    validate_recon_commands() and defaults missing model fields to []. An
    extracted response with a valid commands list but a missing top-level field
    must still reach that salvage path instead of falling through to plain text.
    Callers that do not opt out keep the gate (see the REVIEW assertions
    above)."""
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    schema = {
        "type": "object",
        "required": ["languages", "commands", "conventions", "intent_docs"],
        "properties": {
            "languages": {"type": "array", "items": {"type": "string"}},
            "commands": {"type": "array", "items": {"type": "object"}},
            "conventions": {"type": "array", "items": {"type": "string"}},
            "intent_docs": {"type": "array", "items": {"type": "string"}},
        },
    }
    backend = MockBackend([
        TextEvent(text='{"commands": [{"command": "make test"}]}'),
        ResultEvent(structured_output=None, continuation=None),
    ])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.RECON, output_schema=schema,
        validate_structured_output=False,
    )
    assert result == {"commands": [{"command": "make test"}]}
    assert isinstance(result, dict)


async def test_structured_fallback_salvages_partial_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fallback gate is salvage-tolerant, not all-or-nothing: a dict whose
    required top-level field is present but whose nested records contain one
    schema-invalid item still reaches the consumer. The recommendation verifier
    (and the per-stack parse) drop invalid records rather than losing the whole
    payload, so an all-or-nothing gate here would starve every valid record."""
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    schema = {
        "type": "object",
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["issue_id", "verdict", "evidence"],
                    "properties": {
                        "issue_id": {"type": "integer"},
                        "verdict": {"type": "string"},
                        "evidence": {"type": "string"},
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
    backend = MockBackend([
        TextEvent(text=json.dumps(partial)),
        ResultEvent(structured_output=None, continuation=None),
    ])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.VERIFY, output_schema=schema
    )
    assert result == partial  # the partial dict reaches the salvage path
    assert isinstance(result, dict)


async def test_structured_fallback_bare_array_reaches_merge_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bare JSON array can never validate against an object-typed schema
    (MERGED_ITEMS_SCHEMA is ``type: object``), but phase_cross_stack_merge
    normalizes a bare array to its item list. The gate must let it through
    instead of falling back to plain text, which would raise
    CrossStackMergeError downstream and abort the run."""
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {"items": {"type": "array", "items": {"type": "object"}}},
    }
    items = [{"id": 1, "description": "x"}]
    backend = MockBackend([
        TextEvent(text=json.dumps(items)),
        ResultEvent(structured_output=None, continuation=None),
    ])
    result, _, _ = await run_agent(
        backend, tmp_path, "merge", phase=DaydreamPhase.DEEP, output_schema=schema
    )
    assert result == items
    assert isinstance(result, list)
