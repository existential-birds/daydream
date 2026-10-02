"""Terminal review results through the production runner and public export."""
from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent
from daydream.findings import load_findings_artifact
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.test_deep_orchestrator import _pin_findings_pr, _record


async def _export(target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                  backend: EmptyReviewBackend, **overrides: Any) -> tuple[int, dict[str, Any]]:
    pr = _pin_findings_pr(monkeypatch, target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    output = tmp_path / "public-findings.json"
    rc = await runner.run(empty_review_config(target, tmp_path / "trajectory.json",
                                             findings_out=str(output), pr_number=pr.number, **overrides))
    artifact = load_findings_artifact(output, expected_repo="o/r", expected_pr_number=7,
                                     expected_head_sha=pr.head_sha)
    assert artifact.head_sha == pr.head_sha
    data = json.loads(output.read_text())
    assert data["schema_version"] == 2
    assert data["terminal_result"]["analyzed_revision"]["head_sha"] == pr.head_sha
    assert data["terminal_result"]["analyzed_revision"]["merge_base_sha"] == pr.base_sha
    return rc, data


def _scopes(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {outcome["scope_id"]: outcome for outcome in data["terminal_result"]["stack_outcomes"]}


async def test_complete_empty_review_exports_complete_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = EmptyReviewBackend(multi_stack_target)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert data["findings"] == []
    assert data["terminal_result"]["analysis_state"] == "complete"
    assert data["terminal_result"]["pipeline_state"] == "completed"
    assert set(_scopes(data)) == {"python", "react", "generic", "structure"}
    assert all(scope["status"] == "complete" for scope in _scopes(data).values())
    assert not any("cross-stack merge agent" in call["prompt"].lower() for call in backend.calls)
    assert not any("supervisor adjudication" in call["prompt"].lower() for call in backend.calls)


class ProviderAuthError(RuntimeError):
    category = "AUTH_CONFIG"


@pytest.mark.parametrize(("error", "reason"), [
    (RuntimeError("provider unavailable"), "backend_failure"),
    (ProviderAuthError("provider authentication rejected"), "authentication_failure"),
])
async def test_mixed_empty_review_exports_incomplete_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str,
) -> None:
    backend = EmptyReviewBackend(multi_stack_target, fail_stack="python", stack_error=error)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert data["findings"] == []
    assert data["terminal_result"]["analysis_state"] == "incomplete"
    assert data["terminal_result"]["pipeline_state"] == "completed"
    assert _scopes(data)["python"]["status"] == "failed"
    assert all(value["status"] == "complete" for key, value in _scopes(data).items() if key != "python")
    assert _scopes(data)["python"]["reason_codes"] == [reason]


@pytest.mark.parametrize("failed", [False, True])
async def test_mixed_nonempty_review_preserves_partial_findings(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool,
) -> None:
    record = _record(description="Boundary defect", file="api.py", line=2, severity="medium",
                     confidence="MEDIUM", rationale="Boundary is inconsistent", evidence="api.py:2")
    backend = EmptyReviewBackend(multi_stack_target, review_by_stack={"structure": [record]},
                                 forbid_supervise=False, fail_stack="python" if failed else None)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert len(data["findings"]) == 1
    assert data["findings"][0]["title"] == "Boundary defect"
    assert data["terminal_result"]["analysis_state"] == ("incomplete" if failed else "complete")


class FailedReviewBackend(EmptyReviewBackend):
    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if "you are reviewing the " in prompt.lower() or "you are the structural reviewer" in prompt.lower():
            raise RuntimeError("all reviewers unavailable")
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


async def test_all_reviewers_failed_exports_failed_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FailedReviewBackend(multi_stack_target, forbid_merge=False, forbid_supervise=False)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert data["findings"] == []
    assert data["terminal_result"]["analysis_state"] == "failed"
    assert all(scope["status"] == "failed" for scope in _scopes(data).values())


async def test_analyzed_snapshot_survives_live_head_change(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    calls = 0
    live = pr
    def _live_pr(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return live
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", _live_pr)
    class AdvancingBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal live
            live = replace(pr, head_sha=pr.base_sha)
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    record = _record(description="Head A defect", file="api.py", line=2, severity="medium",
                     confidence="MEDIUM", rationale="A evidence", evidence="api.py:2")
    backend = AdvancingBackend(multi_stack_target, forbid_supervise=False, review_by_stack={"structure": [record]})
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    output = tmp_path / "public.json"
    assert await runner.run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json",
                                               findings_out=str(output), pr_number=7)) == 0
    data = json.loads(output.read_text())
    assert git_ops.head_sha(multi_stack_target) == pr.head_sha
    assert data["head_sha"] == pr.head_sha
    assert data["terminal_result"]["analyzed_revision"]["head_sha"] == pr.head_sha
    assert live.head_sha == pr.base_sha
    assert git_ops.show(multi_stack_target, live.head_sha, "api.py")
    assert data["findings"][0]["placement"] == "inline"
    assert data["findings"][0]["line"] == 2
    assert data["terminal_result"]["analyzed_revision"]["merge_base_sha"] == pr.base_sha
    assert data["terminal_result"]["analyzed_revision"]["pr_base_sha"] == pr.base_sha
    assert calls == 1


async def test_dirty_commit_bound_export_rejected(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (multi_stack_target / "api.py").write_text("DIRTY = True\n")
    backend = EmptyReviewBackend(multi_stack_target)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 1
    assert data["terminal_result"]["analysis_state"] == "failed"
    assert "dirty_snapshot" in data["terminal_result"]["reason_codes"]
    assert backend.calls == []


async def test_no_diff_exports_complete_noop_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.git_helpers import git as _git
    _git(multi_stack_target, "checkout", "main")
    backend = EmptyReviewBackend(multi_stack_target)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert data["terminal_result"]["analysis_state"] == "complete"
    assert data["terminal_result"]["stack_outcomes"] == []
    assert data["terminal_result"]["phase_outcomes"][0]["noop"] is True
    assert data["terminal_result"]["run_id"]
    assert backend.calls == []


async def test_pipeline_budget_before_dispatch_exports_failed_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_deep_orchestrator import _profile_with_pipeline
    backend = EmptyReviewBackend(multi_stack_target)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend,
                             review_profile=_profile_with_pipeline(review_wall_budget_s=0))
    assert rc == 0
    assert backend.calls == []
    result = data["terminal_result"]
    assert result["analysis_state"] == "failed"
    assert "host_pipeline_budget_exhaustion" in result["reason_codes"]
    assert all(scope["status"] == "uncovered" for scope in _scopes(data).values())


async def test_model_turn_budget_exports_explicit_reason(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream.backends import MaxTurnsError
    backend = EmptyReviewBackend(multi_stack_target, fail_stack="python", stack_error=MaxTurnsError("spent turns"))
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert data["terminal_result"]["analysis_state"] == "incomplete"
    assert _scopes(data)["python"]["reason_codes"] == ["model_budget_exhaustion"]


@pytest.mark.parametrize(("mode", "reason", "complete"), [
    ("missing", "missing_output", False), ("malformed", "malformed_output", False),
    ("invalid-record", "malformed_output", False), ("fallback", None, True),
])
async def test_output_validation_preserves_failure_category(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mode: str, reason: str | None, complete: bool,
) -> None:
    from daydream.backends import ResultEvent, TextEvent
    class OutputBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if "you are reviewing the python stack" in prompt.lower():
                self.calls.append({"prompt": prompt, "model": self.model})
                payload: Any = None if mode == "missing" else {"unexpected": []}
                if mode == "invalid-record":
                    payload = {"issues": [{"id": 1}, "invalid"]}
                if mode == "fallback":
                    yield TextEvent(text='{"issues": []}')
                yield ResultEvent(structured_output=payload, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, OutputBackend(multi_stack_target))
    assert rc == 0
    assert data["terminal_result"]["analysis_state"] == ("complete" if complete else "incomplete")
    assert _scopes(data)["python"]["reason_codes"] == ([] if reason is None else [reason])


async def test_archived_and_public_result_preserve_same_coverage(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
) -> None:
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch,
                             EmptyReviewBackend(multi_stack_target, fail_stack="python"), archive=True)
    assert rc == 0
    result = data["terminal_result"]
    archived = archive_dir / "runs" / result["run_id"]
    manifests = json.loads((archived / "manifest.json").read_text())
    assert manifests["archive_status"] == "complete"
    archived_result = json.loads((archived / "findings.json").read_text())
    assert archived_result["terminal_result"] == result
    assert archived_result == data
    paths = list(archived.rglob("review-coverage.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text())
    assert saved["stack_outcomes"] == result["stack_outcomes"]
    assert saved["analyzed_revision"] == result["analyzed_revision"]
    assert json.loads((multi_stack_target / ".daydream/deep/review-coverage.json").read_text()) == saved


@pytest.mark.parametrize(("phase", "malformed"), [("intent", False), ("alternatives", False), ("merge", True)])
async def test_required_phase_failure_finalizes_terminal_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, malformed: bool,
) -> None:
    from tests.harness.review_profile import independent_alternatives_profile
    class PhaseFailureBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            lower = prompt.lower()
            if (phase == "intent" and "present your understanding concisely" in lower) or (
                phase == "alternatives" and "evaluate the implementation" in lower
            ):
                raise RuntimeError(f"{phase} unavailable")
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    record = _record(description="Surviving defect", file="api.py", line=2, severity="medium",
                     confidence="MEDIUM", rationale="Grounded inconsistency", evidence="api.py:2")
    backend = PhaseFailureBackend(multi_stack_target, forbid_merge=False, forbid_supervise=False,
                                  review_by_stack={"python": [record]})
    backend.merge_echo_records = True
    backend.merge_emit_str = "invalid merge output" if malformed else None
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    output = tmp_path / "public-findings.json"
    config = empty_review_config(multi_stack_target, tmp_path / "trajectory.json", findings_out=str(output),
                                 pr_number=7, review_profile=independent_alternatives_profile())
    if malformed:
        assert await runner.run(config) == 1
    else:
        with pytest.raises(RuntimeError, match=f"{phase} unavailable"):
            await runner.run(config)
    data = json.loads(output.read_text())
    assert data["head_sha"] == pr.head_sha
    assert data["terminal_result"]["pipeline_state"] == "failed"
    assert data["terminal_result"]["analysis_state"] == ("incomplete" if malformed else "failed")
    if malformed:
        assert len(data["findings"]) == 1
        assert "synthesis_failure" in data["terminal_result"]["reason_codes"]


@pytest.mark.parametrize("prior", [False, True])
async def test_public_install_failure_preserves_prior_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior: bool,
) -> None:
    backend = EmptyReviewBackend(multi_stack_target)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    output = tmp_path / "public-findings.json"
    config = empty_review_config(multi_stack_target, tmp_path / "trajectory.json",
                                 findings_out=str(output), pr_number=7)
    if prior:
        assert await runner.run(config) == 0
    original = output.read_bytes() if prior else None
    real_link = os.link
    failed = False
    def _link(source: Any, destination: Any, **kwargs: Any) -> None:
        nonlocal failed
        if Path(destination) == output and not failed:
            failed = True
            raise OSError("injected public findings install failure")
        real_link(source, destination, **kwargs)
    monkeypatch.setattr(os, "link", _link)
    assert await runner.run(config) == 1
    assert failed
    assert output.read_bytes() == original if prior else not output.exists()


@pytest.mark.parametrize(("budget", "nonempty"), [("tool", False), ("tool", True), ("wall", False), ("wall", True)])
async def test_host_budget_exports_valid_partial_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: str, nonempty: bool,
) -> None:
    from daydream.backends import ResultEvent, ToolStartEvent
    from tests.harness.fake_clock import FakeClock
    from tests.test_deep_orchestrator import _profile_with_pipeline
    fake = FakeClock().install(monkeypatch)
    record = _record(description="Budget partial defect", file="api.py", line=2, severity="medium",
                     confidence="MEDIUM", rationale="Grounded finding", evidence="api.py:2")
    class PartialBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            lower = prompt.lower()
            if "you are reviewing the " in lower or "you are the structural reviewer" in lower:
                self.calls.append({"prompt": prompt, "model": self.model})
                yield ResultEvent(structured_output={"issues": [record] if nonempty else []}, continuation=None)
                for index in range(3):
                    if budget == "wall":
                        fake.advance(601)
                    yield ToolStartEvent(id=f"budget-{index}", name="Read", input={"file_path": "api.py"})
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    monkeypatch.setattr("daydream.config.DEFAULT_TOOL_CALL_BUDGET", 1 if budget == "tool" else None)
    backend = PartialBackend(multi_stack_target, forbid_merge=False, forbid_supervise=False)
    backend.merge_echo_records = True
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend,
                             review_profile=_profile_with_pipeline(review_wall_budget_s=99999))
    assert rc == 0
    assert data["terminal_result"]["analysis_state"] == "incomplete"
    assert data["terminal_result"]["completed_stacks"] == []
    assert all(scope["partial_evidence"] for scope in _scopes(data).values())
    assert all(scope["status"] == "incomplete" for scope in _scopes(data).values())
    assert f"host_{budget}_budget_exhaustion" in data["terminal_result"]["reason_codes"]
    assert bool(data["findings"]) is nonempty


async def test_host_demoted_finding_remains_valid_projection(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(description="Distant citation", file="api.py", line=500, severity="medium",
                     confidence="MEDIUM", rationale="Grounded finding", evidence="api.py:500")
    backend = EmptyReviewBackend(multi_stack_target, review_by_stack={"structure": [record]}, forbid_supervise=False)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert data["terminal_result"]["analysis_state"] == "complete"
    assert len(data["findings"]) == 1
    assert data["findings"][0]["confidence"] == "LOW"
    assert data["findings"][0]["location_distrust"] is True


async def test_scheduled_shard_inventory_is_exported(
    shard_many_python_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc, data = await _export(shard_many_python_target, tmp_path, monkeypatch,
                             EmptyReviewBackend(shard_many_python_target), deep_shard_enabled=True,
                             deep_shard_max_files=1)
    assert rc == 0
    assert set(_scopes(data)) == {"python#0", "python#1", "python#2", "generic", "structure"}
    assert all(scope["status"] == "complete" for scope in _scopes(data).values())
    files = {file for name, scope in _scopes(data).items() if name != "structure" for file in scope["files"]}
    assert files == {"mod0.py", "mod1.py", "mod2.py", "README.md"}
    assert data["terminal_result"]["analysis_state"] == "complete"


async def test_dirty_interactive_review_remains_supported(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (multi_stack_target / "api.py").write_text("DIRTY = True\n")
    backend = EmptyReviewBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    assert await runner.run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json")) == 0
    assert backend.calls
    assert "DIRTY = True" in (multi_stack_target / ".daydream/diff.patch").read_text()


async def test_failure_before_snapshot_leaves_no_terminal_artifact(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", lambda *_args, **_kwargs:
                        replace(pr, head_sha=pr.base_sha))
    backend = EmptyReviewBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    output = tmp_path / "absent.json"
    assert await runner.run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json",
                                               findings_out=str(output), pr_number=7)) == 1
    assert not output.exists()
    assert backend.calls == []


async def test_terminal_writer_failure_surfaces_without_artifact(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream import json_utils
    _pin_findings_pr(monkeypatch, multi_stack_target)
    backend = EmptyReviewBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    output = tmp_path / "absent.json"
    real_stage = json_utils._stage_bytes
    def _stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        if b'"terminal_result"' in content:
            raise OSError("injected findings staging failure")
        return real_stage(path, content, **kwargs)
    monkeypatch.setattr(json_utils, "_stage_bytes", _stage)
    assert await runner.run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json",
                                               findings_out=str(output), pr_number=7)) == 1
    assert not output.exists()


async def test_dirty_no_commit_diff_cannot_export_complete_noop(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.git_helpers import git
    git(multi_stack_target, "checkout", "main")
    (multi_stack_target / "api.py").write_text("UNBOUND = True\n")
    backend = EmptyReviewBackend(multi_stack_target)
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 1
    assert data["terminal_result"]["analysis_state"] == "failed"
    assert "dirty_snapshot" in data["terminal_result"]["reason_codes"]
    assert backend.calls == []


async def test_pr_base_tip_is_distinct_from_analyzed_merge_base(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.git_helpers import git
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    git(multi_stack_target, "checkout", "main")
    (multi_stack_target / "base-only.txt").write_text("base advanced\n")
    git(multi_stack_target, "add", "base-only.txt")
    git(multi_stack_target, "commit", "-m", "advance base tip")
    tip = git_ops.head_sha(multi_stack_target)
    git(multi_stack_target, "checkout", "feature")
    captured = replace(pr, pr_base_sha=tip)
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", lambda *_args, **_kwargs: captured)
    backend = EmptyReviewBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    output = tmp_path / "bound.json"
    assert await runner.run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json",
                                               findings_out=str(output), pr_number=7)) == 0
    revision = json.loads(output.read_text())["terminal_result"]["analyzed_revision"]
    assert revision["merge_base_sha"] == pr.base_sha
    assert revision["pr_base_sha"] == tip
    assert revision["pr_base_sha"] != revision["merge_base_sha"]


async def test_complete_nonempty_review_exports_complete_result(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(description="Language defect", file="api.py", line=2, severity="medium",
                     confidence="MEDIUM", rationale="Grounded inconsistency", evidence="api.py:2")
    backend = EmptyReviewBackend(multi_stack_target, forbid_merge=False, forbid_supervise=False,
                                 review_by_stack={"python": [record]})
    backend.merge_echo_records = True
    rc, data = await _export(multi_stack_target, tmp_path, monkeypatch, backend)
    assert rc == 0
    assert data["terminal_result"]["analysis_state"] == "complete"
    assert data["terminal_result"]["reason_codes"] == []
    assert len(data["findings"]) == 1
    assert data["findings"][0]["title"] == "Language defect"
    canonical = json.loads((multi_stack_target / ".daydream/deep/merged-items.json").read_text())
    assert canonical["items"][0]["source_uids"] == ["python:1"]
    assert all(scope["status"] == "complete" for scope in _scopes(data).values())
