"""Host structural delegation remains explicit through parse and resume."""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.detection import StackAssignment
from daydream.deep.review_steps import _step_per_stack_parse
from daydream.extensions import Registry
from daydream.flows.engine import FlowContext


@pytest.mark.parametrize("start_at", [None, "merge", "fix"])
async def test_parse_preserves_confirmed_structural_delegation(
    tmp_path: Path, make_config: Any, make_work: Any, start_at: str | None,
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    primary = {"issues": [], "verdicts": [], "source_evidence": [{"path": "api.py", "sha256": "abc", "lines": 1}]}
    (dd / "stack-python-records.json").write_text(json.dumps(primary))
    delegated = {"issues": [], "verdicts": [], "delegated_to": ["python"]}
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
    assert ctx.data["structural_records"] == []


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
