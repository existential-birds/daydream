"""Real run -> raw record -> typed history, including producer failures."""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from daydream import git_ops, runner
from daydream.dataset import LocalRecordStore, StoreError
from daydream.deep import review_terminal
from daydream.extensions import Registry, build_registry
from daydream.observability.config import ObservabilityConfig
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.training.labeler_versions import reply_evidence_digest
from tests.harness.dataset import observation, read_records, run_record
from tests.harness.git_helpers import git
from tests.harness.stub_backend import install_stub_backend, silence


def test_invalid_reply_captures_never_enter_the_store(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    evidence = [{"reply_id": 123, "body_sha256": "a" * 64}]
    for fault in ("captured_hash", "source_id", "duplicate", "absent_text", "wrong_scheme", "run_target"):
        capture = {"status": "available", "source_reply_id": "123", "body_sha256": "a" * 64,
                   "text": "Applied.", "captured_sha256": hashlib.sha256(b"Applied.").hexdigest()}
        record = observation(semantic_evidence=evidence, evidence_digest=reply_evidence_digest(evidence),
                             evidence_digest_scheme="reply-evidence-v1", reply_captures=[capture])
        if fault == "captured_hash":
            capture["captured_sha256"] = "c" * 64
        elif fault == "source_id":
            capture["source_reply_id"] = "124"
        elif fault == "duplicate":
            record["reply_captures"].append(dict(capture))
        elif fault == "absent_text":
            capture["status"] = "unavailable"
        elif fault == "wrong_scheme":
            del record["evidence_digest_scheme"]
            from daydream.dataset import semantic_evidence_digest
            record["evidence_digest"] = semantic_evidence_digest(evidence)
        else:
            record.update(item_uid=None, payload={"type": "run-label", "label": "accepted"})
        with pytest.raises(StoreError, match="invalid_or_unknown_record_schema"):
            store.append_observation(record)
        assert store.read_records()["observations"] == (), fault


@pytest.fixture
def config(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target, pin_skill_availability=False)
    return RunConfig(target=str(multi_stack_target), output_mode="review", archive=False, run_eval=False,
        dataset_capture=True, dataset_store_path=tmp_path / "records", cleanup=False)


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


async def test_terminal_failure_retains_owned_trace_and_partial_evidence(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, memory_exporters: list[InMemorySpanExporter],
) -> None:
    repo = Path(config.target or "")
    original_head, original_diff = git(repo, "rev-parse", "HEAD"), git_ops.diff(repo, "main")
    backend = install_stub_backend(monkeypatch, repo, pin_skill_availability=False)
    backend.review_failure = RuntimeError("terminal offline failure")
    # Evidence that cannot be decoded is retained as text with its reason, never dropped.
    finalize = review_terminal.finalize_review

    def with_malformed_evidence(ctx: Any, *args: Any, **kwargs: Any) -> int:
        (ctx.data["dd"] / "fix-failures.json").write_text("{malformed")
        return finalize(ctx, *args, **kwargs)

    monkeypatch.setattr(review_terminal, "finalize_review", with_malformed_evidence)
    config = replace(config, observability=ObservabilityConfig(destinations=("memory",)))
    # The reviewer retains a provider failure as unfinished work; a review with no usable evidence fails the run.
    assert await run(config) == 1
    raw = captured(config)
    assert raw["outcome"] == "failed"
    coverage = raw["findings"]["value"]["terminal_coverage"]
    assert [scope["status"] for scope in coverage["stack_outcomes"]] == ["incomplete"]
    assert coverage["stack_outcomes"][0]["reason_codes"] == ["backend_failure", "evidence_incomplete"]
    assert raw["findings"]["value"]["items"] == []
    sealed = json.loads((repo / ".daydream/deep/committed-review.json").read_text())
    assert sealed["terminal_result"]["analysis_state"] == "failed" and sealed["run_id"] == raw["run_id"]
    assert raw["trace_id"] == owned_trace_id(memory_exporters[0], raw["run_id"])
    assert raw["original_task"]["status"] == "available"
    assert raw["provenance"]["collection_diagnostics"]["fix-failures.json"] == {
        "status": "failed", "reason": "malformed_json", "raw_text": "{malformed"}
    assert [raw["provenance"][key] for key in ("latency_profile", "latency_route", "latency_routing")] == [None] * 3
    documents = raw["trajectories"]["value"]["documents"]
    assert any(tool["tool_call_id"] == "failed-tool" for doc in documents for step in doc["steps"]
               for tool in step.get("tool_calls", []))
    assert raw["original_task"]["value"]["diff"] == original_diff
    assert raw["original_task"]["value"]["analyzed_revision"]["head_sha"] == original_head
    assert all(document["session_id"] == raw["run_id"] for document in documents)
