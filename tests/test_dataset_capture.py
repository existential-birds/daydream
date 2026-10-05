"""Real run -> raw record -> typed history, including producer failures."""
import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from daydream.dataset import LocalRecordStore, StoreError, parse_observation
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.training.labeler_versions import reply_evidence_digest
from tests.harness.backend import ScriptedBackend
from tests.harness.dataset import observation, read_records
from tests.harness.git_helpers import bare_remote, git
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence

_SENSITIVE = "Review credential exposure: sk-offlineplaceholder0123456789"


@pytest.fixture
def config(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    silence(monkeypatch)
    backend = install_stub_backend(monkeypatch, multi_stack_target, pin_skill_availability=False)
    backend.merge_echo_records = True
    backend.parse_by_stack = {"python": {"severity": "high", "confidence": "HIGH", "file": "api.py", "line": 1,
                                        "description": _SENSITIVE}}
    return RunConfig(target=str(multi_stack_target), output_mode="review", archive=False, run_eval=False,
        dataset_capture=True, dataset_store_path=tmp_path / "records", shallow_fanout_threshold=0, cleanup=False)


def captured(config: RunConfig) -> dict[str, Any]:
    assert config.dataset_store_path is not None
    records = read_records(LocalRecordStore(config.dataset_store_path))
    assert len(records.runs) == 1
    return records.runs[0]


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
    review = (repo / ".review-output.md").read_text()
    scoring = raw["scoring"]["value"]
    assert _SENSITIVE in review and "sk-offlineplaceholder" not in scoring["review_text"]
    assert "[REDACTED_API_KEY]" in scoring["review_text"] and scoring["review_text_redaction"]["applied"]
    assert scoring["length"] == len(review) != len(scoring["review_text"])
    assert scoring["source_review_sha256"] == hashlib.sha256(review.encode()).hexdigest()
    assert scoring["persisted_breakdown"]["composite"] is None
    assert scoring["persisted_breakdown"]["correctness_per_finding"] is None and scoring["posterior_cost"] is None
    text = "Use [REDACTED] through the environment."
    replies = [{"reply_id": "reply:123", "body_sha256": "a" * 64, "disposition": "rejected"}]
    correction = parse_observation(observation(run_id=raw["run_id"], item_uid=item["item_uid"],
        semantic_evidence=replies, evidence_digest=reply_evidence_digest(replies),
        evidence_digest_scheme="reply-evidence-v1", correction={
        "status": "available", "source_reply_id": "reply:123", "text": text, "body_sha256": "a" * 64,
        "captured_sha256": hashlib.sha256(text.encode()).hexdigest(), "redaction_provenance": {"policy": "shared-v1"}}))
    assert config.dataset_store_path is not None
    store = LocalRecordStore(config.dataset_store_path)
    assert store.append_observation(correction).committed and not store.append_observation(correction).committed
    restored = read_records(store).observations[0]
    assert restored["semantic_evidence"] == replies and restored["evidence_digest"] == reply_evidence_digest(replies)
    assert restored["correction"]["text"] == text and restored["correction"]["body_sha256"] == "a" * 64
    assert restored["correction"]["captured_sha256"] != restored["correction"]["body_sha256"]
    for changed in ({"correction": {**correction["correction"], "text": "changed"}},
                    {"semantic_evidence": [{"reply_id": "changed"}]}):
        with pytest.raises(StoreError, match="invalid_or_unknown_record_schema"):
            store.append_observation({**correction, **changed})
    pinned = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    assert store.read_snapshot(pinned).observations == (correction,)
    later = {**correction, "observation_id": "suggestion", "role": "model-suggested", "author": "model"}
    store.append_observation(later)
    assert store.read_snapshot(pinned["snapshot_id"]).observations == (correction,)
    newer = read_records(store)
    assert {obs["observation_id"] for obs in newer.observations} == {correction["observation_id"], "suggestion"}
    assert next(obs for obs in newer.observations if obs["observation_id"] == "suggestion")["review_required"]


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
