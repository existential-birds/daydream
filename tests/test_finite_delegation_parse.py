"""Host structural delegation remains explicit through parse and resume."""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.deep.finite_review import FiniteResult
from daydream.deep.review_steps import _per_stack_body, _step_per_stack_parse
from daydream.extensions import Registry, get_registry
from daydream.flows.engine import FlowContext
from daydream.run_context import InteractionPolicy, RunContext


@pytest.mark.parametrize("start_at", [None, "merge", "fix"])
@pytest.mark.parametrize("uids", [[], ["structure:1", "structure:3"]])
async def test_parse_preserves_confirmed_structural_delegation(
    tmp_path: Path, make_config: Any, make_work: Any, start_at: str | None, uids: list[str],
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    primary = {"issues": [], "verdicts": [], "source_evidence": [{"path": "api.py", "sha256": "abc", "lines": 1}]}
    (dd / "stack-python-records.json").write_text(json.dumps(primary))
    issues = [{"id": 1, "uid": uid, "file": "api.py", "line": 1, "description": "Boundary mismatch",
               "severity": "high", "confidence": "HIGH", "rationale": "Shared contract", "evidence": "api.py:1"}
              for uid in uids]
    delegated = {"issues": issues, "verdicts": [], "delegated_to": ["python"]}
    path = dd / "stack-structure-records.json"
    path.write_text(json.dumps(delegated))
    (dd / "structural-delegation.json").write_text(json.dumps({
        "structural_files": ["api.py"], "primary_scopes": {"python": ["api.py"]},
        "status": "delegated; completion is recorded in each primary review",
    }))
    ctx = FlowContext(
        config=make_config(tmp_path, start_at=start_at), work=make_work(tmp_path), registry=Registry(),
        data={"dd": dd, "stacks": [StackAssignment("python", ["api.py"]),
                                    StackAssignment("structure", ["api.py"])], "failed_stacks": {}},
    )
    assert await _step_per_stack_parse(ctx) is None
    assert json.loads(path.read_text()) == delegated
    assert json.loads((dd / "stack-python-records.json").read_text())["source_evidence"] == primary["source_evidence"]
    assert ctx.data["structural_records"] == issues
    assert ctx.data["records"] == []


@pytest.mark.parametrize("sidecar", [
    None, {"structural_files": ["other.py"], "primary_scopes": {"python": ["api.py"]}},
    {"structural_files": ["api.py"], "primary_scopes": {"python": ["other.py"]}},
])
async def test_parse_does_not_trust_model_delegation_or_stale_scope(
    tmp_path: Path, make_config: Any, make_work: Any, sidecar: dict[str, Any] | None,
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    (dd / "stack-python-records.json").write_text('{"issues": [], "verdicts": []}')
    path = dd / "stack-structure-records.json"
    path.write_text('{"issues": [], "verdicts": [], "delegated_to": ["python"]}')
    if sidecar is not None:
        (dd / "structural-delegation.json").write_text(json.dumps(sidecar))
    ctx = FlowContext(
        config=make_config(tmp_path), work=make_work(tmp_path), registry=Registry(),
        data={"dd": dd, "stacks": [StackAssignment("python", ["api.py"]),
                                    StackAssignment("structure", ["api.py"])], "failed_stacks": {}},
    )
    assert await _step_per_stack_parse(ctx) is None
    saved = json.loads(path.read_text())
    assert "delegated_to" not in saved
    assert saved["verdicts"][0]["verdict"] == "not_reviewed"


@pytest.mark.parametrize("start_at", [None, "per-stack"])
async def test_per_stack_rerun_clears_stale_delegation_before_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
    start_at: str | None,
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    artifacts = [dd / name for name in (
        "structural-delegation.json", "stack-structure-records.json", "stack-structure-review.md",
    )]
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

    async def finite(*args: Any, **kwargs: Any) -> FiniteResult:
        assert not any(path.exists() for path in artifacts)
        attempted.append("primary")
        return FiniteResult(
            {"issues": [], "verdicts": [{"path": "api.py", "lines_read": 1,
                "verdict": "not_reviewed", "n_findings": 0}]},
            "evidence_incomplete", frozenset(), tuple(source.metadata() for source in args[2].sources),
        )

    async def fallback(*args: Any, **kwargs: Any) -> Any:
        assert attempted == ["primary"]
        attempted.append("fallback")
        return {"issues": [], "verdicts": []}, None, None

    monkeypatch.setattr("daydream.deep.finite_review.run_finite_review", finite)
    monkeypatch.setattr("daydream.phases.run_agent", fallback)
    backend = PiBackend(model="test", reasoning_effort="high")
    ctx = FlowContext(
        config=make_config(tmp_path, start_at=start_at), work=make_work(tmp_path), registry=get_registry(),
        allow_standalone_artifacts=True,
        run_context=RunContext(InteractionPolicy(interactive=False)),
        _backend_factory=lambda *_: backend,
        data={"dd": dd, "diff_path": diff, "diff": diff.read_text(), "intent_path": intent,
              "alts_path": alternatives, "exploration_dir": None, "failed_stacks": {},
              "stacks": [StackAssignment("python", ["api.py"]), StackAssignment("structure", ["api.py"])]},
    )
    await _per_stack_body(ctx, include_alternatives=False)
    assert attempted == ["primary", "fallback"]
    assert not artifacts[0].exists()
    assert "delegated_to" not in json.loads(artifacts[1].read_text())
    assert artifacts[2].read_text().startswith("# Review")
    assert set(ctx.data["failed_stacks"]) == {"python"}


def _mark_delegated_artifacts(deep: Path, scopes: dict[str, list[str]]) -> None:
    """Add the committed host metadata to a primed structural record fixture."""
    from daydream.deep.records import stamp_record_uids

    path = deep / "stack-structure-records.json"
    loaded = json.loads(path.read_text())
    issues = loaded if isinstance(loaded, list) else loaded["issues"]
    stamp_record_uids(issues, "structure")
    path.write_text(json.dumps({"issues": issues, "verdicts": [], "delegated_to": list(scopes)}))
    (deep / "structural-delegation.json").write_text(json.dumps({
        "primary_scopes": scopes, "structural_files": [file for files in scopes.values() for file in files],
    }))


@pytest.mark.parametrize("change", [
    {"uid": "python:1"}, {"uid": "structure:0"}, {"uid": "structure:bad"},
    {"lens": "structural"}, {"delegated_to": ["react"]},
])
async def test_parse_rejects_inconsistent_delegated_record_partition(
    tmp_path: Path, make_config: Any, make_work: Any, change: dict[str, Any],
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    (dd / "stack-python-records.json").write_text('{"issues": [], "verdicts": []}')
    issue = {"id": 1, "file": "api.py", "line": 1, "description": "Boundary mismatch", "uid": "structure:1",
             "severity": "high", "confidence": "HIGH", "rationale": "Shared contract", "evidence": "api.py:1"}
    issue.update({key: value for key, value in change.items() if key != "delegated_to"})
    path = dd / "stack-structure-records.json"
    path.write_text(json.dumps({"issues": [issue], "verdicts": []}))
    _mark_delegated_artifacts(dd, {"python": ["api.py"]})
    if "delegated_to" in change:
        loaded = json.loads(path.read_text())
        loaded["delegated_to"] = change["delegated_to"]
        path.write_text(json.dumps(loaded))
    ctx = FlowContext(
        config=make_config(tmp_path), work=make_work(tmp_path), registry=Registry(),
        data={"dd": dd, "stacks": [StackAssignment("python", ["api.py"]),
                                  StackAssignment("structure", ["api.py"])], "failed_stacks": {}},
    )
    assert await _step_per_stack_parse(ctx) is None
    saved = json.loads(path.read_text())
    assert "delegated_to" not in saved
    assert saved["issues"] == [issue]
    assert saved["verdicts"][0]["verdict"] == "has_findings"
    assert saved["verdicts"][0]["n_findings"] == 1
