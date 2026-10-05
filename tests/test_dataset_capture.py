"""Real run -> raw record -> typed history, including producer failures."""
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
from daydream.dataset import LocalRecordStore, parse_observation, serialize_record
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.backend import ScriptedBackend
from tests.harness.dataset import observation, read_records
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence


@pytest.fixture
def config(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target, pin_skill_availability=False)
    return RunConfig(target=str(multi_stack_target), output_mode="review", archive=False, run_eval=False,
        dataset_capture=True, dataset_store_path=tmp_path / "records", shallow_fanout_threshold=0, cleanup=False)


def captured(config: RunConfig) -> dict[str, Any]:
    assert config.dataset_store_path is not None
    records = read_records(LocalRecordStore(config.dataset_store_path))
    assert len(records.runs) == 1
    return serialize_record(records.runs[0])


@pytest.mark.parametrize("shallow", [False, True])
async def test_dirty_original_input_and_history_without_archive_or_findings_export(
    config: RunConfig, shallow: bool,
) -> None:
    config = replace(config, shallow=shallow, stack="python" if shallow else None)
    repo = Path(config.target or "")
    head = git_ops.head_sha(repo)
    base = git_ops.resolve_diff_merge_base(repo, "main", head)
    (repo / "api.py").write_text("# tracked dirty input\n")
    diff = git_ops.diff(repo, "main")
    assert await run(config) == 0
    raw = captured(config)
    task = raw["original_task"]["value"]
    assert task["diff"] == diff and task["diff_sha256"] == hashlib.sha256(diff.encode()).hexdigest()
    assert (task["analyzed_revision"]["head_sha"], task["analyzed_revision"]["merge_base_sha"]) == (head, base)
    assert task["input_scope"] == "committed_and_tracked_worktree" and task["dirty_tracked"]
    assert raw["outcome"] == "success" and raw["trace_id"] is None
    item = raw["findings"]["value"]["items"][0]
    assert item["item_uid"] and item["fingerprint"] and raw["trajectories"]["status"] == "available"
    assert raw["verification"]["status"] == "unproduced"
    text = "Use [REDACTED] through the environment."
    correction = parse_observation(observation(run_id=raw["run_id"], item_uid=item["item_uid"], correction={
        "status": "available", "source_reply_id": "reply:123", "text": text, "body_sha256": "a" * 64,
        "captured_sha256": hashlib.sha256(text.encode()).hexdigest(), "redaction_provenance": {"policy": "shared-v1"}}))
    assert config.dataset_store_path is not None
    store = LocalRecordStore(config.dataset_store_path)
    assert store.append_observation(correction).committed and not store.append_observation(correction).committed
    pinned = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    assert store.read_snapshot(pinned).observations == (correction,)
    later = correction.model_copy(update={"observation_id": "suggestion", "role": "model-suggested", "author": "model"})
    store.append_observation(later)
    assert store.read_snapshot(pinned.snapshot_id).observations == (correction,)
    newer = read_records(store)
    assert {obs.observation_id for obs in newer.observations} == {correction.observation_id, "suggestion"}
    assert next(obs for obs in newer.observations if obs.observation_id == "suggestion").review_required


class _RetryTransportError(RuntimeError):
    retryable = True


class _RetryBackend(StubBackend):
    retried = False
    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if not self.retried:
            self.retried = True
            yield ToolStartEvent(id="attempt-one-tool", name="Read", input={"file_path": "api.py"})
            yield ToolResultEvent(id="attempt-one-tool", output="partial attempt", is_error=False)
            raise _RetryTransportError("temporary transport failure")
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


async def test_failed_attempt_and_child_registration_without_tracing(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RetryBackend(Path(config.target or ""))
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    assert await run(config) == 0
    raw = captured(config)
    assert raw["trace_id"] is None
    documents = raw["trajectories"]["value"]["documents"]
    root = next(doc for doc in documents if doc["trajectory_id"] == raw["run_id"])
    invocations = [item for item in root["extra"]["subtrajectories"] if "invocation_id" in item]
    assert len({item["invocation_id"] for item in invocations}) >= 2
    assert invocations[0]["step_ids"][-1] < invocations[1]["step_ids"][0]
    first = [step for step in root["steps"] if step["step_id"] in invocations[0]["step_ids"]]
    assert any(step.get("extra", {}).get("error") for step in first)
    assert any(tool["tool_call_id"] == "attempt-one-tool" for step in first for tool in step.get("tool_calls", []))
    forks = [item for item in root["extra"]["subtrajectories"] if "fork_id" in item]
    assert forks and all(item["trajectory_id"] in {doc["trajectory_id"] for doc in documents} for item in forks)
    dispatch = next(step for step in root["steps"] if "dispatch_id" in step.get("extra", {}))["observation"]["results"]
    assert [item["content"] for item in dispatch] == [f"Dispatched to deep-{s}" for s in
        ("python", "react", "generic", "structure")]
    assert [item["subagent_trajectory_ref"][0]["trajectory_id"] for item in dispatch] == [
        f"{raw['run_id']}:dispatch:1:fork:{i}" for i in range(1, 5)]


async def test_interruption_retains_task_and_partial_tool_after_worktree_teardown(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = Path(config.target or "")
    backend = ScriptedBackend(events=[ToolStartEvent("interrupted-tool", "Read", {"file_path": "api.py"}),
                                      asyncio.CancelledError()])
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    remote = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "origin", "main", "feature")
    diff = git_ops.diff(repo, "main")
    with pytest.raises(asyncio.CancelledError):
        await run(replace(config, shallow=True, stack="python", force_worktree=True))
    cwd = backend.calls[0]["cwd"]
    assert cwd != repo and not cwd.exists()
    raw = captured(config)
    assert raw["outcome"] == "interrupted" and raw["original_task"]["value"]["diff"] == diff
    documents = raw["trajectories"]["value"]["documents"]
    assert documents[0]["extra"]["target_dir"] == str(repo)
    assert any(doc.get("extra", {}).get("partial") for doc in documents)
    assert any(result.get("extra", {}).get("status") == "interrupted" for doc in documents
        for step in doc["steps"] for result in step.get("observation", {}).get("results", []))
    assert raw["recommended_patch"]["status"] == "unproduced"


async def test_empty_input_and_capture_opt_out_are_distinct(config: RunConfig) -> None:
    config = replace(config, dataset_capture=False,
                     ignore_paths=git_ops.diff_name_only(Path(config.target or ""), "main"))
    assert await run(config) == 0 and config.dataset_store_path is not None
    assert not config.dataset_store_path.exists()
    assert await run(replace(config, dataset_capture=True)) == 0
    raw = captured(config)
    assert raw["outcome"] == "success" and raw["original_task"]["status"] == "available"
    assert raw["original_task"]["value"]["diff"] == "" and raw["findings"]["value"]["items"] == []
    assert raw["scoring"]["status"] == "unproduced"


async def test_pre_recorder_failure_has_absent_evidence_and_preserves_prior_outputs(
    config: RunConfig, fake_gh: FakeGh, tmp_path: Path,
) -> None:
    repo = Path(config.target or "")
    prior = repo / ".review-output.md"
    prior.write_text("prior completed review\n")
    artifacts = repo / ".daydream" / "deep"
    artifacts.mkdir(parents=True)
    (artifacts / "history.json").write_text('{"prior": true}\n')
    (artifacts / "merged-items.json").write_text('{"items": [{"item_uid": "item:old"}]}\n')
    (artifacts.parent / "recommended.patch").write_text("prior recommended patch\n")
    fake_gh.serve_pr_view({"number": 7, "state": "OPEN", "headRefName": "feature", "baseRefName": "main",
        "headRefOid": "0" * 40, "headRepository": {"nameWithOwner": "acme/widgets"},
        "headRepositoryOwner": {"login": "acme"}, "url": "https://github.com/acme/widgets/pull/7", "body": ""})
    assert await run(replace(config, findings_out=str(tmp_path / "findings.json"), pr_number=7,
                             pr_repo="acme/widgets")) == 1
    raw = captured(config)
    assert raw["outcome"] == "failed" and raw["original_task"]["status"] == "unavailable"
    assert all(raw[name]["status"] == "unproduced"
               for name in ("trajectories", "findings", "scoring", "recommended_patch"))
    assert prior.read_text() == "prior completed review\n"
    assert (artifacts / "history.json").read_text() == '{"prior": true}\n'


class _MalformedProducerBackend(StubBackend):
    malformed = "{ malformed producer record"
    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if "cross-stack merge agent" in prompt.lower():
            Path(re.findall(r"  - (\S+-records\.json)", prompt)[0]).write_text(self.malformed)
            raise RuntimeError("external backend failed")
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


@pytest.mark.parametrize(("malformed", "reason", "valid"), [
    ("{ malformed producer record", "malformed_json", False), ('{"issues": "invalid"}', "malformed_shape", True)])
async def test_malformed_producer_evidence_and_uncomputable_reward(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, malformed: str, reason: str, valid: bool,
) -> None:
    backend = _MalformedProducerBackend(Path(config.target or ""))
    backend.malformed = malformed
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    with pytest.raises(RuntimeError, match="external backend failed"):
        await run(config)
    raw = captured(config)
    assert raw["outcome"] == "failed" and raw["completeness"]["artifact_acquisition"] == "failed"
    assert all(raw[name]["status"] == "available" for name in ("original_task", "trajectories"))
    diagnostics = raw["provenance"]["collection_diagnostics"].values()
    assert any(item["reason"] == reason for item in diagnostics)
    assert any(item.get("raw_json" if valid else "raw_text") == ({"issues": "invalid"} if valid else malformed)
               for item in diagnostics)
    scoring = raw["scoring"]["value"]
    assert scoring["format_valid"] is valid and scoring["persisted_breakdown"]["correctness_per_finding"] is None
    assert scoring["persisted_breakdown"]["composite"] == (None if valid else 0.0)
