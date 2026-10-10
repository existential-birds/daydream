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
from tests.harness.review_result import review_scopes


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
    results, failures = await review_scopes(backend, make_work(tmp_path),
        [StackAssignment(stack_name=STRUCTURE_STACK_NAME, files=["api.py"], is_docs_only=False)],
        diff_path=diff, intent_path=intent, alternatives_path=alternatives,
        intent_authoritative=True, allow_standalone=True,
    )
    assert "FOUNDATIONAL_DIFF" in backend.last_prompt
    assert "PRESERVE_AUTHOR_INTENT" in backend.last_prompt
    assert "AUTHORITATIVE" in backend.last_prompt
    assert "api.py" in backend.last_prompt
    assert "Investigation allowance" not in backend.last_prompt
    assert "budget exhausted" in failures[STRUCTURE_STACK_NAME]
    assert STRUCTURE_STACK_NAME in results
    records = next(tmp_path.rglob("stack-structure-records.json"))
    saved = json.loads(records.read_text())
    assert saved["incomplete"] is True
    assert saved["issues"] == []

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
