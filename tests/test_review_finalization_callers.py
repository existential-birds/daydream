"""Production phase callers retain explicit inputs at review cutoffs."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import phases
from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.review_budget import ReviewLimits
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
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
        pointer = _sanctioned_inputs(prompt)['intent']
        text = pointer.read_text()
        intent_reads.append(text)
        return [ToolStartEvent(id='intent-context', name='read', input={'path': str(pointer)}),
                ToolResultEvent(id='intent-context', output=text, is_error=False),
                *[ToolStartEvent(id=f"limit-{index}", name="read", input={"path": "api.py"})
                  for index in range(stage['remaining_tool_calls'])]]

    backend = ScriptedBackend(responder=exhaust)
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
    assert str(diff) in backend.last_prompt
    assert intent_reads == ["PRESERVE_AUTHOR_INTENT"]
    assert str(intent) in backend.last_prompt
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
    diff.write_text("diff --git a/api.py b/api.py\n+change\n")
    with pytest.raises(ReviewOutputError) as raised:
        await phases.phase_understand_intent(backend, make_work(tmp_path), diff, "log", "branch")
    assert raised.value.reason_code == "missing_output"
