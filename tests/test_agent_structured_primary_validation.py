"""Real agent execution retains validated fallback, salvage, and failure witnesses."""
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import StructuredOutputFailure, run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend

_FILE = {"type": "object", "required": ["file"], "properties": {"file": {"type": "string"}}}
_VERDICTS = {"type": "object", "required": ["verdicts"], "properties": {
    "verdicts": {"type": "array", "items": {"type": "object", "required": ["issue_id", "verdict", "evidence"]}}}}
_ITEMS = {"type": "object", "required": ["items"], "properties": {
    "items": {"type": "array", "items": {"type": "object"}}}}
_ISSUES = {"type": "object", "required": ["issues"], "properties": {"issues": {"type": "array"}}}
_PARTIAL = {"verdicts": [{"issue_id": 1, "verdict": "consistent", "evidence": "matches"},
                         {"issue_id": 2, "verdict": "bogus"}]}
_BARE = [{"id": 1, "description": "x"}]
_VALID = {"issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]}


@pytest.mark.parametrize("schema,payload,text,phase,options,expected,reason", [
    pytest.param(_FILE, {"line": 3}, '{"file": "src/a.py"}', "REVIEW", {}, {"file": "src/a.py"}, None, id="fallback"),
    pytest.param(_FILE, {"line": 3}, '{"line": 3}', "REVIEW", {}, '{"line": 3}', None, id="invalid-text"),
    pytest.param(_VERDICTS, _PARTIAL, None, "VERIFY", {}, _PARTIAL, None, id="partial-salvage"),
    pytest.param(_ITEMS, _BARE, None, "DEEP", {"require_full_schema": True}, "malformed_output", None,
                 id="bare-array-rejected"),
    pytest.param(_FILE, {"line": 3}, None, "RECON", {"validate_structured_output": False}, {"line": 3}, None,
                 id="opt-out"),
    pytest.param(_ISSUES, _VALID, None, "REVIEW", {}, _VALID, None, id="valid-primary"),
    pytest.param(_FILE, {"line": 3}, None, "REVIEW", {"require_full_schema": True}, "malformed_output", None,
                 id="malformed-witness"),
    pytest.param(_FILE, None, None, "REVIEW", {"require_full_schema": True}, "missing_output", None,
                 id="missing-witness"),
    pytest.param(_FILE, {"line": 3}, '{"file": "src/a.py"}', "REVIEW",
                 {"require_full_schema": True}, {"file": "src/a.py"}, None,
                 id="strict-fallback"),
])
async def test_primary_validation_and_host_witness(
    tmp_path: Path, schema: dict[str, Any], payload: Any, text: str | None, phase: str,
    options: dict[str, Any], expected: Any, reason: str | None,
) -> None:
    events: list[Any] = []
    if text is not None:
        events.append(TextEvent(text=text))
    events.append(ResultEvent(structured_output=payload, continuation=None))
    result, _, budget_reason = await run_agent(ScriptedBackend(events=events, model="mock-model"), tmp_path,
                                               "review", phase=DaydreamPhase[phase], output_schema=schema, **options)
    if expected in ("malformed_output", "missing_output"):
        assert isinstance(result, StructuredOutputFailure) and result.reason == expected
    else:
        assert result == expected and isinstance(result, type(expected))
    assert budget_reason == reason
