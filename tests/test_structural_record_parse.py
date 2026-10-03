"""Structural findings remain partitioned through review and resume."""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.deep.review_steps import _per_stack_body, _step_per_stack_parse
from daydream.extensions import Registry, get_registry
from daydream.flows.engine import FlowContext
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.review_result import records_artifact, review_coverage


@pytest.mark.parametrize("start_at", [None, "merge", "fix"])
@pytest.mark.parametrize("uids", [[], ["structure:1", "structure:3"]])
async def test_parse_preserves_structural_partition_on_resume(
    tmp_path: Path, make_config: Any, make_work: Any, start_at: str | None, uids: list[str],
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    coverage = review_coverage(files=("api.py",), phases=())
    primary = records_artifact(coverage, "python")
    (dd / "stack-python-records.json").write_text(json.dumps(primary))
    issues = [{"id": 1, "uid": uid, "file": "api.py", "line": 1, "description": "Boundary mismatch",
               "severity": "high", "confidence": "HIGH", "rationale": "Shared contract", "evidence": "api.py:1"}
              for uid in uids]
    structural = records_artifact(coverage, "structure", issues)
    path = dd / "stack-structure-records.json"
    path.write_text(json.dumps(structural))
    ctx = FlowContext(config=make_config(tmp_path, start_at=start_at), work=make_work(tmp_path), registry=Registry(),
        data={"dd": dd, "review_coverage": coverage, "stacks": [StackAssignment("python", ["api.py"]),
                                    StackAssignment("structure", ["api.py"])], "failed_stacks": {}},
    )
    assert await _step_per_stack_parse(ctx) is None
    assert json.loads(path.read_text()) == structural
    assert ctx.data["record_pool"].structural == issues
    assert ctx.data["record_pool"].language == []

@pytest.mark.parametrize("start_at", [None, "per-stack"])
async def test_per_stack_rerun_clears_stale_structural_outputs_before_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, start_at: str | None,
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    artifacts = [dd / name for name in ("stack-structure-records.json", "stack-structure-review.md",)]
    for path in artifacts:
        path.write_text("STALE")
    (tmp_path / "api.py").write_text("value = 1\n")
    diff = dd / "diff.patch"
    diff.write_text("diff --git a/api.py b/api.py\n--- a/api.py\n+++ b/api.py\n"
                    "@@ -1 +1 @@\n-value = 0\n+value = 1\n")
    intent = dd / "intent.md"
    intent.write_text("Preserve behavior")
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    attempted: list[str] = []

    async def review(*args: Any, **kwargs: Any) -> Any:
        assert not artifacts[0].exists()
        assert "STALE" not in args[2]
        attempted.append("structure" if "repository-wide interactions" in args[2] else "primary")
        return {"issues": []}, None, None

    monkeypatch.setattr("daydream.agent.run_agent", review)
    backend = PiBackend(model="test", reasoning_effort="high")
    ctx = FlowContext(
        config=make_config(tmp_path, start_at=start_at), work=make_work(tmp_path), registry=get_registry(),
        allow_standalone_artifacts=True, run_context=RunContext(InteractionPolicy(interactive=False)),
        _backend_factory=lambda *_: backend,
        data={"dd": dd, "review_coverage": review_coverage(files=("api.py",), phases=()),
              "diff_path": diff, "diff": diff.read_text(), "intent_path": intent,
              "alts_path": alternatives, "exploration_dir": None, "failed_stacks": {},
              "stacks": [StackAssignment("python", ["api.py"]), StackAssignment("structure", ["api.py"])]},
    )
    await _per_stack_body(ctx, include_alternatives=False)
    assert sorted(attempted) == ["primary", "structure"]
    assert json.loads(artifacts[0].read_text())["issues"] == []
    assert artifacts[1].read_text().startswith("# Review")
    assert ctx.data["failed_stacks"] == {}


@pytest.mark.parametrize('ordinal', ['²', '١', '01', '0', '-1'])
def test_record_artifact_rejects_noncanonical_ordinals(ordinal: str) -> None:
    from daydream.phases.review import valid_record_artifact
    from tests.harness.review_result import records_artifact, review_coverage

    coverage = review_coverage()
    artifact = records_artifact(coverage, 'python', [{'uid': f'python:{ordinal}'}])
    assert not valid_record_artifact(artifact, scope_id='python', analyzed_revision=coverage.revision.to_dict())
