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
from daydream.hunk_index import write_hunk_index
from daydream.workspace import WorkContext
from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.review_result import review_scopes
from tests.harness.stub_backend import review_stage_state
from tests.test_deep_orchestrator import _sanctioned_inputs


def _stop_then_finalize(result: dict[str, Any]) -> ScriptedBackend:
    return ScriptedBackend(script=[[ToolStartEvent(id="limit", name="read", input={"path": "unused.py"})],
        [ResultEvent(structured_output=result, continuation=None)],
    ])

async def test_structural_stage_cutoff_publishes_without_model_finalizer(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    intent_reads: list[str] = []

    def exhaust(_cwd: Path, prompt: str, *_args: Any) -> list[Any]:
        stage = review_stage_state(prompt)
        assert stage is not None and stage['stage'] == 'integration'
        text = supporting_contents(prompt)['intent']
        intent_reads.append(text)
        return [ToolStartEvent(id=f"limit-{index}", name="read", input={"path": "api.py"})
                for index in range(stage['remaining_tool_calls'] + 1)]

    backend = ScriptedBackend(responder=exhaust)
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/api.py b/api.py\n--- a/api.py\n+++ b/api.py\n"
                    "@@ -1 +1 @@\n-VALUE = 0\n+FOUNDATIONAL_DIFF\n")
    (tmp_path / "api.py").write_text("FOUNDATIONAL_DIFF\n")
    seed_feature_branch(tmp_path, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': 'FOUNDATIONAL_DIFF\n'})
    write_hunk_index(tmp_path, diff.read_text())
    intent = tmp_path / "intent.md"
    intent.write_text("PRESERVE_AUTHOR_INTENT")
    alternatives = tmp_path / "alternatives.json"
    alternatives.write_text("ADVISORY " * 4000)
    work = make_work(tmp_path, base_sha=git(tmp_path, "rev-parse", "main"),
                     head_sha=git(tmp_path, "rev-parse", "HEAD"), head_branch="feature")
    results, failures = await review_scopes(backend, work,
        [StackAssignment(stack_name=STRUCTURE_STACK_NAME, files=["api.py"], is_docs_only=False)],
        diff_path=diff, intent_path=intent, alternatives_path=alternatives,
        intent_authoritative=True, allow_standalone=True,
    )
    assert str(diff) not in backend.last_prompt
    assert 'review-assignment' in _sanctioned_inputs(backend.last_prompt)
    assert intent_reads == ["PRESERVE_AUTHOR_INTENT"]
    assert 'PRESERVE_AUTHOR_INTENT' in backend.last_prompt
    assert "AUTHORITATIVE" in backend.last_prompt
    assert "api.py" in backend.last_prompt
    assert "Hard reviewer allowance" in backend.last_prompt
    assert "Host review stage:" in backend.last_prompt
    assert backend.call_count == 1
    assert "tool_call_budget_exceeded" in failures[STRUCTURE_STACK_NAME]
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
    diff.write_text("diff --git a/api.py b/api.py\n--- a/api.py\n+++ b/api.py\n"
                    "@@ -1 +1 @@\n-VALUE = 0\n+VALUE = 1\n")
    (tmp_path / "api.py").write_text("VALUE = 1\n")
    write_hunk_index(tmp_path, diff.read_text())
    with pytest.raises(ReviewOutputError) as raised:
        await phases.phase_understand_intent(backend, make_work(tmp_path), diff, "log", "branch")
    assert raised.value.reason_code == "missing_output"
