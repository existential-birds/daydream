"""Real-path tests: the run_agent primary structured-output return path must
gate the backend-supplied result through the same ``_salvageable`` check the
extraction fallback applies (honoring ``validate_structured_output``).

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


async def test_primary_path_unusable_text_and_structured_degrade_to_plain_text(tmp_path) -> None:
    # The reachable codex/pi route: structured output is parsed from the agent
    # text itself (codex.py:647-648, pi.py:910), so text and structured carry
    # the same payload. When that payload violates output_schema, both gates
    # reject and run_agent degrades to the plain-text string — not the
    # unvalidated dict that leaked out before the primary gate existed.
    payload = {"line": 3}
    backend = MockBackend([TextEvent(text=json.dumps(payload)),
                           ResultEvent(structured_output=payload, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.REVIEW, output_schema=_FILE_SCHEMA)
    assert result == '{"line": 3}'   # identical invalid JSON in text and structured
    assert isinstance(result, str)


async def test_primary_path_salvages_partial_dict(tmp_path) -> None:
    schema = {"type": "object", "required": ["verdicts"], "properties": {
        "verdicts": {"type": "array", "items": {
            "type": "object", "required": ["issue_id", "verdict", "evidence"]}}}}
    partial = {"verdicts": [
        {"issue_id": 1, "verdict": "consistent", "evidence": "matches"},
        {"issue_id": 2, "verdict": "bogus"},  # nested item missing "evidence"
    ]}
    # No TextEvent: a mirrored text would let the fallback re-extract the same
    # value, so this would pass even without the primary gate. Only the primary
    # return path can yield ``partial`` here.
    backend = MockBackend([ResultEvent(structured_output=partial, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.VERIFY, output_schema=schema)
    assert result == partial          # salvage-tolerant: nested validity not gated
    assert isinstance(result, dict)


async def test_primary_path_bare_array_reaches_merge_shape(tmp_path) -> None:
    schema = {"type": "object", "required": ["items"], "properties": {
        "items": {"type": "array", "items": {"type": "object"}}}}
    items = [{"id": 1, "description": "x"}]
    # No TextEvent: a mirrored text would let the fallback re-extract the same
    # value, so this would pass even without the primary gate. Only the primary
    # return path can yield ``items`` here.
    backend = MockBackend([ResultEvent(structured_output=items, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "merge", phase=DaydreamPhase.DEEP, output_schema=schema)
    assert result == items            # bare array is a salvageable form
    assert isinstance(result, list)


async def test_primary_path_respects_validate_structured_output_false(tmp_path) -> None:
    backend = MockBackend([ResultEvent(structured_output={"line": 3}, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "go", phase=DaydreamPhase.RECON,
        output_schema=_FILE_SCHEMA, validate_structured_output=False)
    assert result == {"line": 3}      # opt-out passes through unvalidated, as today
    assert isinstance(result, dict)


async def test_primary_path_valid_structured_output_returned(tmp_path) -> None:
    schema = {"type": "object", "required": ["issues"], "properties": {"issues": {"type": "array"}}}
    payload = {"issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]}
    backend = MockBackend([ResultEvent(structured_output=payload, continuation=None)])
    result, _, _ = await run_agent(
        backend, tmp_path, "Parse", phase=DaydreamPhase.REVIEW, output_schema=schema)
    assert result == payload          # valid codex-shaped output returned unchanged
