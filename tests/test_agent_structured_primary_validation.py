"""Real-path tests: the run_agent primary structured-output return path must
gate the backend-supplied result through the same ``_salvageable`` check the
extraction fallback applies (honoring ``validate_fallback_schema``).

Pins the agent.py:864 asymmetry: before this module, the primary path returned
``structured_result`` unvalidated whenever ``output_schema`` was set and the
backend supplied a non-None result — so a codex/pi payload that was valid JSON
but violated ``output_schema`` leaked out unvalidated. These tests drive
``run_agent`` (the single-agent production entrypoint) with the shared
``MockBackend``, mocking only the ``Backend`` protocol seam — the established
real-path pattern in ``test_agent_structured_render.py`` /
``test_agent_recorder_integration.py``.
"""

from __future__ import annotations

import json

from daydream.agent import run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.trajectory import DaydreamPhase
from tests.test_agent_recorder_integration import MockBackend

_FILE_SCHEMA = {
    "type": "object",
    "required": ["file"],
    "properties": {"file": {"type": "string"}},
}


async def test_primary_path_schema_violation_degrades_to_fallback(tmp_path) -> None:
    backend = MockBackend([
        TextEvent(text='{"file": "src/a.py"}'),
        ResultEvent(structured_output={"line": 3}, continuation=None),
    ])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.REVIEW, output_schema=_FILE_SCHEMA
    )
    # {"line": 3} is missing required "file" -> rejected; the fallback
    # re-extracts the valid {"file": "src/a.py"} from the agent text.
    assert result == {"file": "src/a.py"}
    assert isinstance(result, dict)


async def test_primary_path_salvages_partial_dict(tmp_path) -> None:
    schema = {"type": "object", "required": ["verdicts"], "properties": {
        "verdicts": {"type": "array", "items": {
            "type": "object", "required": ["issue_id", "verdict", "evidence"]}}}}
    partial = {"verdicts": [
        {"issue_id": 1, "verdict": "consistent", "evidence": "matches"},
        {"issue_id": 2, "verdict": "bogus"},  # nested item missing "evidence"
    ]}
    backend = MockBackend([TextEvent(text=json.dumps(partial)),
                           ResultEvent(structured_output=partial, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.VERIFY, output_schema=schema)
    assert result == partial          # salvage-tolerant: nested validity not gated
    assert isinstance(result, dict)


async def test_primary_path_bare_array_reaches_merge_shape(tmp_path) -> None:
    schema = {"type": "object", "required": ["items"], "properties": {
        "items": {"type": "array", "items": {"type": "object"}}}}
    items = [{"id": 1, "description": "x"}]
    backend = MockBackend([TextEvent(text=json.dumps(items)),
                           ResultEvent(structured_output=items, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "merge", phase=DaydreamPhase.DEEP, output_schema=schema)
    assert result == items            # bare array is a salvageable form
    assert isinstance(result, list)


async def test_primary_path_respects_validate_fallback_schema_false(tmp_path) -> None:
    backend = MockBackend([ResultEvent(structured_output={"line": 3}, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.RECON,
        output_schema=_FILE_SCHEMA, validate_fallback_schema=False)
    assert result == {"line": 3}      # opt-out passes through unvalidated, as today
    assert isinstance(result, dict)


async def test_primary_path_valid_structured_output_returned(tmp_path) -> None:
    schema = {"type": "object", "required": ["issues"], "properties": {"issues": {"type": "array"}}}
    payload = {"issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]}
    backend = MockBackend([ResultEvent(structured_output=payload, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "Parse", phase=DaydreamPhase.REVIEW, output_schema=schema)
    assert result == payload          # valid codex-shaped output returned unchanged
