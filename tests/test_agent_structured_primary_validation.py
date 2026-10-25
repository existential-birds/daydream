"""Real agent execution retains validated fallback, salvage, and failure witnesses."""
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import StructuredOutputFailure, run_agent
from daydream.backends import ResultEvent, TextEvent
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.table_cases import isolated_case

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


async def test_primary_validation_and_host_witness(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    cases: list[tuple[dict[str, Any], Any, str | None, str, dict[str, Any], Any, str | None]] = [
        (_FILE, {"line": 3}, '{"file": "src/a.py"}', "REVIEW", {}, {"file": "src/a.py"}, None),
        (_FILE, {"line": 3}, '{"line": 3}', "REVIEW", {}, '{"line": 3}', None),
        (_VERDICTS, _PARTIAL, None, "VERIFY", {}, _PARTIAL, None),
        (_ITEMS, _BARE, None, "DEEP", {"require_full_schema": True}, "malformed_output", None),
        (_FILE, {"line": 3}, None, "RECON", {"validate_structured_output": False}, {"line": 3}, None),
        (_ISSUES, _VALID, None, "REVIEW", {}, _VALID, None),
        (_FILE, {"line": 3}, None, "REVIEW", {"require_full_schema": True}, "malformed_output", None),
        (_FILE, None, None, "REVIEW", {"require_full_schema": True}, "missing_output", None),
        (
            _FILE,
            {"line": 3},
            '{"file": "src/a.py"}',
            "REVIEW",
            {"require_full_schema": True},
            {"file": "src/a.py"},
            None,
        ),
    ]
    for schema, payload, text, phase, options, expected, reason in cases:
        row_index += 1
        with subtests.test(text=text):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                events: list[Any] = []
                if text is not None:
                    events.append(TextEvent(text=text))
                events.append(ResultEvent(structured_output=payload, continuation=None))
                result, _, budget_reason = await run_agent(
                    ScriptedBackend(events=events, model="mock-model"),
                    tmp_path,
                    "review",
                    phase=DaydreamPhase[phase],
                    output_schema=schema,
                    **options,
                )
                if expected in ("malformed_output", "missing_output"):
                    assert isinstance(result, StructuredOutputFailure) and result.reason == expected
                else:
                    assert result == expected and isinstance(result, type(expected))
                assert budget_reason == reason
