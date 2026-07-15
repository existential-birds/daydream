"""Task 1 real-path tests: run_agent must not echo structured-output JSON.

Verified terminal-render harness (from Task 0):
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    ... drive run_agent ...
    rec.export_text()  # captures the rendered agent text

run_agent requires the keyword-only `phase=` argument (DaydreamPhase),
imported from daydream.trajectory. MockBackend is imported from
tests.test_agent_recorder_integration (the single canonical definition).
"""

from __future__ import annotations

from rich.console import Console

from daydream.agent import run_agent
from daydream.backends import (
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import DaydreamPhase
from tests.test_agent_recorder_integration import MockBackend

RAW = '{"conventions": [{"name": "OpenAPI First", "description": "x", "source": "CLAUDE.md"}]}'
PAYLOAD = {"conventions": [{"name": "OpenAPI First", "description": "x", "source": "CLAUDE.md"}]}


async def test_structured_output_text_is_not_rendered(monkeypatch, tmp_path):
    rec = Console(record=True, force_terminal=True, width=100)
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
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.agent.console", rec)
    backend = MockBackend(
        [TextEvent(text="narration here"), ResultEvent(structured_output=None, continuation=None)]
    )
    result, _, _ = await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)  # no output_schema
    assert "narration here" in rec.export_text()


async def test_log_mode_captures_structured_output(monkeypatch, tmp_path, capsys):
    """Under --log, the structured result must still be captured, not just printed.

    Regression: the log-mode print and the structured-result capture were an
    if/elif, so --log dropped every structured result (run_agent returned "").
    Downstream this emptied exploration conventions/dependencies and review
    findings across the deep pipeline.
    """
    monkeypatch.setattr("daydream.agent._state.log_mode", True)
    backend = MockBackend([ResultEvent(structured_output=PAYLOAD, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "scan", phase=DaydreamPhase.REVIEW, output_schema={"type": "object"}
    )
    assert result == PAYLOAD  # captured, not discarded
    out = capsys.readouterr().out
    assert "[result]" in out  # log-mode print is additive, still happens
    assert "OpenAPI First" in out  # the serialized payload, not just the marker


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
