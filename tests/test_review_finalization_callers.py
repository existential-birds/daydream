"""Production phase callers retain explicit inputs at review cutoffs."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import phases
from daydream.backends import ResultEvent, ToolStartEvent
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.review_budget import ReviewLimits
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.review_result import record_pool, review_scopes


def _stop_then_finalize(result: dict[str, Any]) -> ScriptedBackend:
    return ScriptedBackend(script=[[ToolStartEvent(id="limit", name="read", input={"path": "unused.py"})],
        [ResultEvent(structured_output=result, continuation=None)],
    ])

async def test_structural_finalizer_captures_prioritized_diff_without_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
) -> None:
    backend = _stop_then_finalize({"issues": []})
    monkeypatch.setattr("daydream.phases.review.ReviewLimits", lambda *a, **kw: ReviewLimits(10, 2, 0))
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/api.py b/api.py\n+FOUNDATIONAL_DIFF\n")
    intent = tmp_path / "intent.md"
    intent.write_text("PRESERVE_AUTHOR_INTENT")
    alternatives = tmp_path / "alternatives.json"
    alternatives.write_text("ADVISORY " * 4000)
    coverage = await review_scopes(backend, make_work(tmp_path),
        [StackAssignment(stack_name=STRUCTURE_STACK_NAME, files=["api.py"], is_docs_only=False)],
        diff_path=diff, intent_path=intent, alternatives_path=alternatives,
        intent_authoritative=True, allow_standalone=True,
    )
    assert "FOUNDATIONAL_DIFF" in backend.last_prompt
    assert "PRESERVE_AUTHOR_INTENT" in backend.last_prompt
    assert "AUTHORITATIVE" in backend.last_prompt
    assert "api.py" in backend.last_prompt
    assert "Investigation allowance" not in backend.last_prompt
    assert "budget exhausted" in coverage.unfinished_scopes[STRUCTURE_STACK_NAME]
    assert coverage.scopes[STRUCTURE_STACK_NAME]["status"] == "incomplete"
    records = next(tmp_path.rglob("stack-structure-records.json"))
    saved = json.loads(records.read_text())
    assert saved["incomplete"] is True
    assert saved["issues"] == []

async def test_merge_finalizer_prioritizes_records_and_keeps_budget_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
) -> None:
    backend = _stop_then_finalize({"items": []})
    monkeypatch.setattr("daydream.phases.merge.ReviewLimits", lambda *a, **kw: ReviewLimits(10, 2, 0))
    records = tmp_path / "stack-python-records.json"
    records.write_text(json.dumps({"issues": [{"description": "ESTABLISHED_RECORD"}]}))
    intent = tmp_path / "intent.md"
    intent.write_text("Preserve contracts")
    alternatives = tmp_path / "alternatives.json"
    alternatives.write_text("ADVISORY " * 4000)
    dedup = tmp_path / "dedup.json"
    dedup.write_text("ADVISORY " * 4000)
    with pytest.raises(phases.CrossStackMergeError, match="budget exhausted"):
        await phases.phase_cross_stack_merge(backend, make_work(tmp_path),
            record_pool=record_pool(tmp_path, json.loads(records.read_text())["issues"], paths=[records]),
            intent_path=intent, alternatives_path=alternatives, dedup_candidates_path=dedup,
            failed_stacks={"react": "budget exhausted"}, allow_standalone=True,
        )
    assert "ESTABLISHED_RECORD" in backend.last_prompt
    assert "budget exhausted" in backend.last_prompt
    assert "Investigation allowance" not in backend.last_prompt

@pytest.mark.parametrize("payload", [{}, {"issues": "bad"}, {"issues": [{"description": "unfinished"}]}])
async def test_ordinary_reviewer_rejects_invalid_envelope(
    tmp_path: Path, make_work: Callable[..., WorkContext], payload: dict[str, Any],
) -> None:
    backend = ScriptedBackend(events=[ResultEvent(structured_output=payload, continuation=None)])
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/api.py b/api.py\n+change\n")
    intent = tmp_path / "intent.md"
    intent.write_text("intent")
    alternatives = tmp_path / "alternatives.json"
    alternatives.write_text("[]")
    coverage = await review_scopes(backend, make_work(tmp_path),
        [StackAssignment(stack_name="python", files=["api.py"], is_docs_only=False)],
        diff_path=diff, intent_path=intent, alternatives_path=alternatives, allow_standalone=True)
    assert coverage.scopes["python"]["status"] == "failed"
    assert coverage.unfinished_scopes["python"] == "malformed_output: reviewer response did not satisfy its schema"
    assert not list(tmp_path.rglob("stack-python-records.json"))

async def test_intent_requires_nonempty_provider_evidence(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    from daydream.phases.review import ReviewOutputError
    backend = ScriptedBackend(events=[ResultEvent(structured_output=None, continuation=None)])
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/api.py b/api.py\n+change\n")
    with pytest.raises(ReviewOutputError) as raised:
        await phases.phase_understand_intent(backend, make_work(tmp_path), diff, "log", "branch")
    assert raised.value.reason_code == "missing_output"
