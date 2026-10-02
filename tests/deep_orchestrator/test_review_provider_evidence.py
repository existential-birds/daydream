"""Provider checkpoints and loaded artifact binding through public review export."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, MaxTurnsError, ResultEvent
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.deep_orchestrator.test_review_completion import _export, _scopes
from tests.test_deep_orchestrator import _record


@pytest.mark.parametrize("artifact_fault", [None, "missing", "corrupt"])
@pytest.mark.parametrize("nonempty", [False, True])
async def test_model_turn_budget_preserves_valid_checkpoint(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonempty: bool,
    artifact_fault: str | None,
) -> None:
    record = _record(description="Checkpoint defect", file="api.py", line=2)

    class CheckpointBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if "you are reviewing the " in prompt.lower() or "you are the structural reviewer" in prompt.lower():
                self.calls.append({"prompt": prompt, "model": self.model})
                yield ResultEvent(structured_output={"issues": [record] if nonempty else []}, continuation=None)
                raise MaxTurnsError("provider exhausted its configured turns")
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    if artifact_fault == "missing":
        original_is_file = Path.is_file
        monkeypatch.setattr(Path, "is_file", lambda path: False if path.name == "stack-python-records.json"
                            else original_is_file(path))
    elif artifact_fault == "corrupt":
        original_read = Path.read_text
        monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                            "{" if path.name == "stack-python-records.json" else original_read(path, *args, **kwargs))
    backend = CheckpointBackend(multi_stack_target, forbid_merge=False, forbid_supervise=False)
    backend.merge_echo_records = True
    code, artifact = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert code == (1 if artifact_fault else 0)
    result = artifact["terminal_result"]
    assert result["analysis_state"] == "incomplete"
    assert result["completed_stacks"] == []
    for name, scope in _scopes(artifact).items():
        damaged = artifact_fault is not None and name == "python"
        assert scope["status"] == ("failed" if damaged else "incomplete")
        assert scope["partial_evidence"] is not damaged
        expected = ["model_budget_exhaustion"]
        if damaged:
            expected.append("missing_artifact" if artifact_fault == "missing" else "malformed_artifact")
        assert scope["reason_codes"] == sorted(expected)
    assert bool(artifact["findings"]) is nonempty


@pytest.mark.parametrize("binding", ["revision", "scope", "origin"])
async def test_loaded_review_artifact_rejects_foreign_binding(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str,
) -> None:
    original = Path.read_text

    def corrupt_read(path: Path, *args: Any, **kwargs: Any) -> str:
        content = original(path, *args, **kwargs)
        if path.name == "stack-python-records.json":
            data = json.loads(content)
            if binding == "revision":
                data["analyzed_revision"]["head_sha"] = "foreign-head"
            elif binding == "scope":
                data["scope_id"] = "foreign-scope"
            else:
                data.pop("originating_run_id")
            return json.dumps(data)
        return content

    monkeypatch.setattr(Path, "read_text", corrupt_read)
    code, artifact = await _export(multi_stack_target, tmp_path, monkeypatch,
                                  EmptyReviewBackend(multi_stack_target))
    assert code == 1
    assert artifact["terminal_result"]["pipeline_state"] == "failed"
    assert artifact["terminal_result"]["analysis_state"] == "incomplete"
    assert artifact["terminal_result"]["projection_valid"] is True
    assert artifact["findings"] == []
    assert _scopes(artifact)["python"]["status"] == "failed"
    assert _scopes(artifact)["python"]["reason_codes"] == ["malformed_artifact"]
    assert all(scope["status"] == "complete" for name, scope in _scopes(artifact).items() if name != "python")
