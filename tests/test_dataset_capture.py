"""New evidence capture through the real run and public local store seams."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.dataset import LocalRecordStore
from daydream.dataset.schema import (
    CapturedCorrection,
    FindingJudgmentPayload,
    ObservationRecord,
    semantic_evidence_digest,
)
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git
from tests.harness.remote_ci import NoCIRemote
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence
from tests.test_archive_data_capture import _ArchiveCaptureBackend
from tests.test_deep_orchestrator import _merge_item


async def test_review_captures_original_dirty_input_without_archive_or_findings_export(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, feature_branch_repo, pin_skill_availability=False)
    original_head = git_ops.head_sha(feature_branch_repo)
    original_base = git_ops.resolve_diff_merge_base(feature_branch_repo, "main", original_head)
    (feature_branch_repo / "main.py").write_text("# tracked dirty input\n")
    original_diff = git_ops.diff(feature_branch_repo, "main")
    store_path = tmp_path / "records"
    result = await run(RunConfig(
        target=str(feature_branch_repo), stack="python", quiet=True, cleanup=False,
        shallow=True, output_mode="review", archive=False, run_eval=False,
        dataset_capture=True, dataset_store_path=store_path,
    ))
    assert result == 0
    store = LocalRecordStore(store_path)
    records = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00"))
    assert len(records.runs) == 1
    captured = records.runs[0]
    task = captured.original_task.value
    assert isinstance(task, dict)
    assert task["diff"] == original_diff
    assert task["diff_sha256"] == hashlib.sha256(original_diff.encode()).hexdigest()
    assert task["analyzed_revision"]["head_sha"] == original_head
    assert task["analyzed_revision"]["merge_base_sha"] == original_base
    assert task["input_scope"] == "committed_and_tracked_worktree"
    assert task["dirty_tracked"] is True
    assert captured.outcome == "success"
    assert captured.findings.status == "available"
    assert captured.findings.value["items"]
    assert captured.findings.value["items"][0]["item_uid"]
    assert captured.findings.value["items"][0]["fingerprint"]
    assert captured.trajectories.status == "available"
    assert captured.trace_id is None
    assert captured.verification.status == "unproduced"
    evidence = {"source_reply_id": "reply:123", "disposition": "rejected"}
    observation = ObservationRecord(
        observation_id="correction:123", run_id=captured.run_id,
        item_uid=captured.findings.value["items"][0]["item_uid"],
        valid_at="2027-01-01T00:00:00+00:00", observed_at="2027-01-02T00:00:00+00:00",
        source="maintainer-reply", author="maintainer", role="rater",
        policy_version="policy:1", rubric_version="rubric:1", semantic_evidence=evidence,
        evidence_digest=semantic_evidence_digest(evidence),
        payload=FindingJudgmentPayload(disposition="rejected", rationale="The guard handles this case."),
        correction=CapturedCorrection(
            status="available", source_reply_id="reply:123", text="The guard handles this case.",
            captured_sha256="fa309782d1dc3aa476a6371505c3a6e9ba300daa70bd2cae9350a8762052f4f3",
            redaction_provenance={"policy": "shared-v1", "applied": False},
        ),
    )
    assert store.append_observation(observation).committed
    assert not store.append_observation(observation).committed
    pinned = store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00")
    assert store.read_snapshot(pinned).observations == (observation,)
    later = observation.model_copy(update={
        "observation_id": "suggestion:124", "observed_at": "2027-02-01T00:00:00+00:00",
        "role": "model-suggested", "author": "review-model", "review_required": True,
    })
    assert store.append_observation(later).committed
    assert store.read_snapshot(pinned.snapshot_id).observations == (observation,)
    newer = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00"))
    assert {item.observation_id for item in newer.observations} == {"correction:123", "suggestion:124"}
    assert next(item for item in newer.observations if item.observation_id == "suggestion:124").review_required


class _RetryTransportError(RuntimeError):
    retryable = True


class _RetryBackend(StubBackend):
    retried = False

    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if not self.retried:
            self.retried = True
            yield ToolStartEvent(id="attempt-one-tool", name="Read", input={"file_path": "main.py"})
            yield ToolResultEvent(id="attempt-one-tool", output="def hello(): return 'universe'", is_error=False)
            raise _RetryTransportError("temporary transport failure")
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


async def test_capture_preserves_failed_attempt_and_child_registration_without_tracing(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    silence(monkeypatch)
    backend = _RetryBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    store = LocalRecordStore(tmp_path / "records")
    assert await run(RunConfig(
        target=str(multi_stack_target), output_mode="review", archive=False, run_eval=False,
        dataset_capture=True, dataset_store_path=store.root,
        shallow_fanout_threshold=0, cleanup=False,
    )) == 0
    records = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00"))
    captured = records.runs[0]
    assert captured.trace_id is None
    documents = captured.trajectories.value["documents"]
    assert len(documents) > 1
    root = next(document for document in documents if document["trajectory_id"] == captured.run_id)
    invocations = [item for item in root["extra"]["subtrajectories"] if "invocation_id" in item]
    assert len({item["invocation_id"] for item in invocations}) >= 2
    first_invocation, second_invocation = invocations[:2]
    assert first_invocation["step_ids"][-1] < second_invocation["step_ids"][0]
    first_steps = [step for step in root["steps"] if step["step_id"] in first_invocation["step_ids"]]
    assert any(step.get("extra", {}).get("error") for step in first_steps)
    assert any(tool["tool_call_id"] == "attempt-one-tool" for step in first_steps
               for tool in step.get("tool_calls", []))
    forks = [item for item in root["extra"]["subtrajectories"] if "fork_id" in item]
    assert forks
    assert all(item["trajectory_id"] in {document["trajectory_id"] for document in documents} for item in forks)
    dispatches = [step for step in root["steps"] if "dispatch_id" in step.get("extra", {})]
    assert dispatches
    registration = dispatches[0]["observation"]["results"]
    assert [item["content"] for item in registration] == [
        "Dispatched to deep-python", "Dispatched to deep-react", "Dispatched to deep-generic",
        "Dispatched to deep-structure",
    ]
    assert [item["subagent_trajectory_ref"][0]["trajectory_id"] for item in registration] == [
        f"{captured.run_id}:dispatch:1:fork:{ordinal}" for ordinal in range(1, 5)
    ]


class _InterruptedBackend(StubBackend):
    execution_cwd: Path | None = None

    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        self.execution_cwd = Path(cwd)
        yield ToolStartEvent(id="interrupted-tool", name="Read", input={"file_path": "main.py"})
        raise asyncio.CancelledError


async def test_cooperative_interruption_retains_acquired_task_and_partial_tool(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    silence(monkeypatch)
    backend = _InterruptedBackend(feature_branch_repo)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    remote = bare_remote(tmp_path / "origin.git")
    git(feature_branch_repo, "remote", "add", "origin", str(remote))
    git(feature_branch_repo, "push", "origin", "main", "feature")
    original_diff = git_ops.diff(feature_branch_repo, "main")
    store = LocalRecordStore(tmp_path / "records")
    with pytest.raises(asyncio.CancelledError):
        await run(RunConfig(
            target=str(feature_branch_repo), shallow=True, output_mode="review", archive=False,
            run_eval=False, dataset_capture=True, dataset_store_path=store.root, cleanup=False, force_worktree=True,
        ))
    assert backend.execution_cwd != feature_branch_repo
    assert backend.execution_cwd is not None and not backend.execution_cwd.exists()
    captured = store.read_snapshot(store.select_snapshot(
        observed_before="2100-01-01T00:00:00+00:00",
    )).runs[0]
    assert captured.outcome == "interrupted"
    assert captured.original_task.value["diff"] == original_diff
    documents = captured.trajectories.value["documents"]
    assert documents[0]["extra"]["target_dir"] == str(feature_branch_repo)
    assert any(document.get("extra", {}).get("partial") for document in documents)
    assert any(result.get("extra", {}).get("status") == "interrupted" for document in documents
               for step in document["steps"] for result in step.get("observation", {}).get("results", []))
    assert captured.recommended_patch.status == "unproduced"


async def test_empty_input_and_capture_opt_out_are_distinct(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, feature_branch_repo, pin_skill_availability=False)
    config = RunConfig(
        target=str(feature_branch_repo), output_mode="review", archive=False, run_eval=False,
        dataset_capture=False, dataset_store_path=tmp_path / "records", ignore_paths=["main.py"], cleanup=False,
    )
    assert await run(config) == 0
    assert not (tmp_path / "records").exists()
    assert await run(replace(config, dataset_capture=True)) == 0
    store = LocalRecordStore(tmp_path / "records")
    captured = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00")).runs[0]
    assert captured.outcome == "success"
    assert captured.original_task.status == "available"
    assert captured.original_task.value["diff"] == ""
    assert captured.findings.value["items"] == []
    assert captured.scoring.status == "unproduced"


async def test_supported_pre_recorder_failure_has_absent_evidence_and_preserves_prior_outputs(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gh: FakeGh,
) -> None:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, feature_branch_repo, pin_skill_availability=False)
    prior_review = feature_branch_repo / REVIEW_OUTPUT_FILE
    prior_review.write_text("prior completed review\n")
    prior_artifact = feature_branch_repo / ".daydream" / "deep" / "history.json"
    prior_artifact.parent.mkdir(parents=True)
    prior_artifact.write_text('{"prior": true}\n')
    (prior_artifact.parent / "merged-items.json").write_text('{"items": [{"item_uid": "item:old"}]}\n')
    (prior_artifact.parent.parent / "recommended.patch").write_text("prior recommended patch\n")
    fake_gh.serve_pr_view({
        "number": 7, "state": "OPEN", "headRefName": "feature", "baseRefName": "main",
        "headRefOid": "0" * 40, "headRepository": {"nameWithOwner": "acme/widgets"},
        "headRepositoryOwner": {"login": "acme"}, "url": "https://github.com/acme/widgets/pull/7", "body": "",
    })
    store = LocalRecordStore(tmp_path / "records")
    assert await run(RunConfig(
        target=str(feature_branch_repo), output_mode="review", archive=False, run_eval=False,
        dataset_capture=True, dataset_store_path=store.root, cleanup=False,
        findings_out=str(tmp_path / "findings.json"), pr_number=7, pr_repo="acme/widgets",
    )) == 1
    captured = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00")).runs[0]
    assert captured.outcome == "failed"
    assert captured.original_task.status == "unavailable"
    assert captured.trajectories.status == "unproduced"
    assert captured.findings.status == "unproduced"
    assert captured.scoring.status == "unproduced"
    assert captured.recommended_patch.status == "unproduced"
    assert prior_review.read_text() == "prior completed review\n"
    assert prior_artifact.read_text() == '{"prior": true}\n'


async def test_committed_fix_keeps_original_revision_and_item_verifier_associations(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_ci_remote: NoCIRemote,
) -> None:
    silence(monkeypatch)
    original_head = git_ops.head_sha(multi_stack_target)
    original_base = git_ops.resolve_diff_merge_base(multi_stack_target, "main", original_head)
    original_diff = git_ops.diff(multi_stack_target, "main")
    remote = bare_remote(tmp_path / "origin.git")
    no_ci_remote.connect(multi_stack_target, remote)
    backend = _ArchiveCaptureBackend(multi_stack_target)
    backend.fix_edit_line = "# daydream recommended change\n"
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    backend.merge_items[0]["source_uids"] = ["python:1"]
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    store = LocalRecordStore(tmp_path / "records")
    assert await run(RunConfig(
        target=str(multi_stack_target), output_mode="loop", assume="yes", non_interactive=True,
        archive=False, run_eval=False, dataset_capture=True, dataset_store_path=store.root, cleanup=False,
        pr_number=no_ci_remote.pr_number, pr_repo=no_ci_remote.base_repository,
    )) == 0
    final_head = git_ops.head_sha(multi_stack_target)
    assert final_head != original_head
    captured = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00")).runs[0]
    assert captured.original_task.value["analyzed_revision"]["head_sha"] == original_head
    assert captured.original_task.value["analyzed_revision"]["merge_base_sha"] == original_base
    assert captured.original_task.value["diff"] == original_diff
    assert captured.final_state.value["head_sha"] == final_head
    patch = captured.recommended_patch.value["patch"]
    assert patch != original_diff
    assert "# daydream recommended change" in patch
    assert "# daydream recommended change" not in original_diff
    item = captured.findings.value["items"][0]
    assert item["item_uid"]
    assert item["source_uids"]
    verdicts = captured.verification.value["recommendation-verdicts.json"]
    assert verdicts["selection"]["decisions"][0]["item_uid"] == item["item_uid"]
    assert verdicts["verdicts"][0]["issue_id"] == item["id"]
    assert captured.verification.value["fix-outcomes.json"]["outcomes"][item["item_uid"]]["verdict"] == "resolved"
    assert captured.scoring.value["persisted_breakdown"]["correctness_per_finding"] == [1.0]


class _MalformedProducerBackend(StubBackend):
    malformed = "{ malformed producer record"

    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if "cross-stack merge agent" in prompt.lower():
            records_path = Path(re.findall(r"  - (\S+-records\.json)", prompt)[0])
            records_path.write_text(self.malformed)
            raise RuntimeError("external backend failed after malformed structured evidence")
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


@pytest.mark.parametrize(("malformed", "reason", "format_valid"), [
    ("{ malformed producer record", "malformed_json", False),
    ('{"issues": "invalid"}', "malformed_shape", True),
])
async def test_failed_run_preserves_malformed_producer_evidence_and_uncomputable_reward(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    malformed: str, reason: str, format_valid: bool,
) -> None:
    silence(monkeypatch)
    backend = _MalformedProducerBackend(multi_stack_target)
    backend.malformed = malformed
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    store = LocalRecordStore(tmp_path / "records")
    with pytest.raises(RuntimeError, match="external backend failed"):
        await run(RunConfig(
            target=str(multi_stack_target), output_mode="review", archive=False, run_eval=False,
            dataset_capture=True, dataset_store_path=store.root, shallow_fanout_threshold=0, cleanup=False,
        ))
    captured = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00+00:00")).runs[0]
    assert captured.outcome == "failed"
    assert captured.original_task.status == "available"
    assert captured.trajectories.status == "available"
    assert captured.completeness["artifact_acquisition"] == "failed"
    diagnostics = captured.provenance["collection_diagnostics"]
    assert any(item["reason"] == reason for item in diagnostics.values())
    if format_valid:
        assert any(item.get("raw_json") == {"issues": "invalid"} for item in diagnostics.values())
    else:
        assert any(item.get("raw_text") == "{ malformed producer record" for item in diagnostics.values())
    assert captured.scoring.value["format_valid"] is format_valid
    assert captured.scoring.value["persisted_breakdown"]["correctness_per_finding"] is None
    assert captured.scoring.value["persisted_breakdown"]["composite"] == (None if format_valid else 0.0)
