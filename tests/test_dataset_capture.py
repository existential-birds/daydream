"""Real run -> raw record -> typed history, including producer failures."""
import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from daydream import git_ops, json_utils, runner
from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from daydream.dataset import LocalRecordStore, StoreError, parse_observation
from daydream.extensions import Registry, build_registry
from daydream.observability.config import ObservabilityConfig
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.training.labeler_versions import reply_evidence_digest
from tests.harness.backend import ScriptedBackend
from tests.harness.dataset import observation, read_records, run_record
from tests.harness.git_helpers import bare_remote, git
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence

_SENSITIVE = "Review credential exposure: sk-offlineplaceholder0123456789"


@pytest.mark.parametrize("text", ["", "Applied.\n\nCorrection: café 🦉\r\n保留\n"])
def test_source_bound_reply_captures_preserve_exact_text_and_distinct_hashes(tmp_path: Path, text: str) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    evidence = [{"reply_id": 123, "body_sha256": "a" * 64}, {"reply_id": 124, "body_sha256": "b" * 64}]
    captures = [{"status": "available", "source_reply_id": "123", "body_sha256": "a" * 64,
                 "text": text, "captured_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "created_at": "2026-10-04T11:00:00Z", "updated_at": "2026-10-04T12:00:00Z",
                 "html_url": "https://github.com/owner/repo/pull/7#discussion_r123", "in_reply_to_id": "100"},
                {"status": "unavailable", "source_reply_id": "124", "body_sha256": "b" * 64,
                 "reason": "Reply body missing from acquired source"}]
    record = observation(semantic_evidence=evidence, evidence_digest=reply_evidence_digest(evidence),
                         evidence_digest_scheme="reply-evidence-v1", reply_captures=captures)
    assert store.append_observation(record).committed
    assert not store.append_observation(record).committed
    pinned = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    assert store.read_snapshot(pinned).observations[0]["reply_captures"] == captures
    assert captures[0]["captured_sha256"] != captures[0]["body_sha256"]
    assert "text" not in captures[1]


@pytest.mark.parametrize("fault", ["captured_hash", "source_hash", "source_id", "duplicate", "absent_text",
                                  "missing_source_hash", "wrong_scheme", "run_target"])
def test_invalid_reply_captures_never_enter_the_store(tmp_path: Path, fault: str) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    evidence = [{"reply_id": 123, "body_sha256": "a" * 64}]
    capture = {"status": "available", "source_reply_id": "123", "body_sha256": "a" * 64,
               "text": "Applied.", "captured_sha256": hashlib.sha256(b"Applied.").hexdigest()}
    record = observation(semantic_evidence=evidence, evidence_digest=reply_evidence_digest(evidence),
                         evidence_digest_scheme="reply-evidence-v1", reply_captures=[capture])
    if fault == "captured_hash":
        capture["captured_sha256"] = "c" * 64
    elif fault == "source_hash":
        capture["body_sha256"] = "b" * 64
    elif fault == "source_id":
        capture["source_reply_id"] = "124"
    elif fault == "duplicate":
        record["reply_captures"].append(dict(capture))
    elif fault == "absent_text":
        capture["status"] = "unavailable"
    elif fault == "missing_source_hash":
        del capture["body_sha256"]
    elif fault == "wrong_scheme":
        del record["evidence_digest_scheme"]
        from daydream.dataset import semantic_evidence_digest
        record["evidence_digest"] = semantic_evidence_digest(evidence)
    else:
        record.update(item_uid=None, payload={"type": "run-label", "label": "accepted"})
    with pytest.raises(StoreError, match="invalid_or_unknown_record_schema"):
        store.append_observation(record)
    assert store.read_records()["observations"] == ()


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


@pytest.fixture
def memory_exporters(monkeypatch: pytest.MonkeyPatch) -> list[InMemorySpanExporter]:
    exporters: list[InMemorySpanExporter] = []

    def factory(config: ObservabilityConfig) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        exporters.append(exporter)
        return exporter

    def registry() -> Registry:
        result = build_registry()
        result.register_trace_exporter("memory", factory)
        return result

    monkeypatch.setattr(runner, "build_registry", registry)
    return exporters


def owned_trace_id(exporter: InMemorySpanExporter, run_id: str) -> str:
    root, = [span for span in exporter.get_finished_spans() if span.name == "daydream.run"]
    assert root.context is not None and root.parent is None
    assert (root.attributes or {})["daydream.session.id"] == run_id
    identity = format(root.context.trace_id, "032x")
    assert re.fullmatch("[0-9a-f]{32}", identity) and identity != "0" * 32
    return identity


@pytest.mark.parametrize("tracing", [False, True])
@pytest.mark.parametrize("shallow", [False, True])
async def test_dirty_original_input_and_history_without_archive_or_findings_export(
    config: RunConfig, shallow: bool, tracing: bool, memory_exporters: list[InMemorySpanExporter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(config, shallow=shallow, stack="python" if shallow else None,
                     observability=ObservabilityConfig(destinations=("memory",) if tracing else ()))
    repo = Path(config.target or "")
    head = git_ops.head_sha(repo)
    base = git_ops.resolve_diff_merge_base(repo, "main", head)
    (repo / "api.py").write_text(f"# tracked dirty input: {_SENSITIVE}\n")
    diff = git_ops.diff(repo, "main")
    ambient = NonRecordingSpan(SpanContext(123, 456, False, TraceFlags(TraceFlags.SAMPLED)))
    stage = json_utils._stage_bytes
    findings_reports: list[str] = []
    def observe_context_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        if path.name == 'review-report-context.json':
            before_context = (path.parent / 'review-output.md').read_text()
            # This complete fixture's earlier supervision warning predates its own completion.
            findings_reports.append(before_context[before_context.index('# Review\n'):])
        return stage(path, content, **kwargs)
    monkeypatch.setattr(json_utils, '_stage_bytes', observe_context_stage)
    with trace.use_span(ambient):
        assert await run(config) == 0
    raw = captured(config)
    task = raw["original_task"]["value"]
    assert task["diff"] == diff and task["diff_sha256"] == hashlib.sha256(diff.encode()).hexdigest()
    assert "redaction" not in task and "source_diff_sha256" not in task
    assert (task["analyzed_revision"]["head_sha"], task["analyzed_revision"]["merge_base_sha"]) == (head, base)
    assert task["input_scope"] == "committed_and_tracked_worktree" and task["dirty_tracked"]
    expected_trace = owned_trace_id(memory_exporters[0], raw["run_id"]) if tracing else None
    assert expected_trace != format(123, "032x")
    assert raw["outcome"] == "success" and raw["trace_id"] == expected_trace
    item = raw["findings"]["value"]["items"][0]
    assert item["item_uid"] and item["fingerprint"] and raw["trajectories"]["status"] == "available"
    assert raw["verification"]["status"] == "unproduced"
    review = (repo / ".review-output.md").read_text()
    scoring = raw["scoring"]["value"]
    assert _SENSITIVE in review and scoring["review_text"] == review
    assert len(findings_reports) == 1 and _SENSITIVE in findings_reports[0]
    assert '## Review outcome' in review and scoring['length'] == len(findings_reports[0]) < len(review)
    assert "review_text_redaction" not in scoring and "source_review_sha256" not in scoring
    assert scoring["persisted_breakdown"]["composite"] is None
    assert scoring["persisted_breakdown"]["correctness_per_finding"] is None and scoring["posterior_cost"] is None
    text = f"Use {_SENSITIVE} through the environment."
    replies = [{"reply_id": "reply:123", "body_sha256": "a" * 64, "disposition": "rejected"}]
    correction = parse_observation(observation(run_id=raw["run_id"], item_uid=item["item_uid"],
        semantic_evidence=replies, evidence_digest=reply_evidence_digest(replies),
        evidence_digest_scheme="reply-evidence-v1", correction={
        "status": "available", "source_reply_id": "reply:123", "text": text, "body_sha256": "a" * 64,
        "captured_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    assert config.dataset_store_path is not None
    store = LocalRecordStore(config.dataset_store_path)
    disk_record, = (store.root / "runs").glob("*.jsonl")
    assert json.loads(disk_record.read_bytes())["trace_id"] == expected_trace
    assert store.read_records()["runs"][0]["trace_id"] == expected_trace
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
    assert store.read_snapshot(pinned["snapshot_id"]).runs[0]["trace_id"] == expected_trace
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


@pytest.mark.parametrize("tracing", [False, True])
async def test_interruption_retains_task_and_partial_tool_after_worktree_teardown(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    tracing: bool, memory_exporters: list[InMemorySpanExporter],
) -> None:
    config = replace(config, observability=ObservabilityConfig(destinations=("memory",) if tracing else ()))
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
    assert raw["trace_id"] == (owned_trace_id(memory_exporters[0], raw["run_id"]) if tracing else None)
    documents = raw["trajectories"]["value"]["documents"]
    assert documents[0]["extra"]["target_dir"] == str(repo)
    assert any(doc.get("extra", {}).get("partial") for doc in documents)
    assert any(result.get("extra", {}).get("status") == "interrupted" for doc in documents
        for step in doc["steps"] for result in step.get("observation", {}).get("results", []))
    assert raw["recommended_patch"]["status"] == "unproduced"


async def test_terminal_failure_retains_owned_trace_and_partial_evidence(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, memory_exporters: list[InMemorySpanExporter],
) -> None:
    backend = ScriptedBackend(events=[ToolStartEvent("failed-tool", "Read", {"file_path": "api.py"}),
                                      RuntimeError("terminal offline failure")])
    monkeypatch.setattr(runner, "create_backend", lambda *args, **kwargs: backend)
    config = replace(config, shallow=True, stack="python", observability=ObservabilityConfig(destinations=("memory",)))
    with pytest.raises(RuntimeError, match="terminal offline failure"):
        await run(config)
    raw = captured(config)
    assert raw["outcome"] == "failed"
    assert raw["trace_id"] == owned_trace_id(memory_exporters[0], raw["run_id"])
    assert raw["original_task"]["status"] == "available"
    documents = raw["trajectories"]["value"]["documents"]
    assert any(tool["tool_call_id"] == "failed-tool" for doc in documents for step in doc["steps"]
               for tool in step.get("tool_calls", []))
    assert any(doc.get("extra", {}).get("partial") for doc in documents)


@pytest.mark.parametrize("fault", ["start", "invalid", "context"])
async def test_unavailable_owned_trace_captures_null_despite_ambient_context(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, memory_exporters: list[InMemorySpanExporter], fault: str,
) -> None:
    start_span = Tracer.start_span

    class BrokenContext(NonRecordingSpan):
        def get_span_context(self) -> SpanContext:
            raise RuntimeError("owned context unavailable")

    def start(self: Tracer, name: str, *args: Any, **kwargs: Any) -> trace.Span:
        if name == "daydream.run":
            if fault == "start":
                raise RuntimeError("owned root unavailable")
            context = SpanContext(0, 0, False)
            return BrokenContext(context) if fault == "context" else NonRecordingSpan(context)
        return start_span(self, name, *args, **kwargs)

    monkeypatch.setattr(Tracer, "start_span", start)
    ambient = NonRecordingSpan(SpanContext(123, 456, False, TraceFlags(TraceFlags.SAMPLED)))
    with trace.use_span(ambient):
        assert await run(replace(config, shallow=True, stack="python",
                                 observability=ObservabilityConfig(destinations=("memory",)))) == 0
    assert len(memory_exporters) == 1
    raw = captured(config)
    assert raw["trace_id"] is None and raw["outcome"] == "success"
    assert _SENSITIVE in (Path(config.target or "") / ".review-output.md").read_text()


async def test_concurrent_captures_keep_owned_roots_under_ambient_trace(
    config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    memory_exporters: list[InMemorySpanExporter],
) -> None:
    repo = Path(config.target or "")
    other = tmp_path / "other-repo"
    git(tmp_path, "clone", "--quiet", str(repo), str(other))
    git(other, "branch", "main", "origin/main")
    configs = [replace(config, target=str(target), dataset_store_path=tmp_path / f"records-{index}",
                       base="main", shallow=True, stack="python",
                       observability=ObservabilityConfig(destinations=("memory",)))
               for index, target in enumerate((repo, other))]
    active = 0
    both_active = asyncio.Event()

    class ConcurrentBackend(StubBackend):
        async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal active
            active += 1
            if active == 2:
                both_active.set()
            await asyncio.wait_for(both_active.wait(), 10)
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    monkeypatch.setattr(runner, "create_backend", lambda *args, **kwargs: ConcurrentBackend(repo))
    ambient = NonRecordingSpan(SpanContext(123, 456, False, TraceFlags(TraceFlags.SAMPLED)))
    with trace.use_span(ambient):
        assert await asyncio.gather(*(run(candidate) for candidate in configs)) == [0, 0]
    roots = {}
    for exporter in memory_exporters:
        root, = [span for span in exporter.get_finished_spans() if span.name == "daydream.run"]
        run_id = (root.attributes or {})["daydream.session.id"]
        assert isinstance(run_id, str)
        roots[run_id] = owned_trace_id(exporter, run_id)
    records = [captured(candidate) for candidate in configs]
    assert len(roots) == 2 and len(set(roots.values())) == 2
    for raw in records:
        assert raw["trace_id"] == roots[raw["run_id"]] != format(123, "032x")
