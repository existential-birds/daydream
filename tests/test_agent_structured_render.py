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


async def test_structured_output_text_is_not_rendered(monkeypatch, tmp_path):
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


async def test_plain_text_still_renders(monkeypatch, tmp_path):
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    backend = MockBackend(
        [TextEvent(text="narration here"), ResultEvent(structured_output=None, continuation=None)]
    )
    result, _, _ = await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)  # no output_schema
    assert "narration here" in rec.export_text()


async def test_log_mode_emission_redacts_sentinels_on_agent_path(
    monkeypatch, tmp_path, capsys
):
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


async def test_log_mode_captures_structured_output(monkeypatch, tmp_path, capsys):
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


async def test_log_mode_structured_result_wins_over_prose_stray_json(monkeypatch, tmp_path):
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
    payload = {"items": []}
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


_FALLBACK_SCHEMA = {
    "type": "object",
    "required": ["status"],
    "properties": {"status": {"type": "string"}},
}


async def test_fallback_valid_schema_json_is_returned_as_structured(tmp_path) -> None:
    backend = MockBackend(
        [TextEvent(text='{"status": "complete"}'),
         ResultEvent(structured_output=None, continuation=None)]
    )
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW,
        output_schema=_FALLBACK_SCHEMA,
    )
    assert result == {"status": "complete"}


async def test_fallback_invalid_schema_json_falls_through_to_plain_text(tmp_path) -> None:
    backend = MockBackend(
        [TextEvent(text='{"status": 42}'),
         ResultEvent(structured_output=None, continuation=None)]
    )
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW,
        output_schema=_FALLBACK_SCHEMA,
    )
    # `{"status": 42}` parses via extract_json but violates the schema
    # (status must be a string), so it must NOT be returned as structured
    # output — it falls through to the plain-text return.
    assert result == '{"status": 42}'
