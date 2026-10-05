"""Public persistence, history, recovery and tamper checks on real local storage."""
import hashlib
import json
import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from daydream.dataset import LocalRecordStore, StoreError, parse_run, run_record_schema
from daydream.dataset_capture import capture_scoring
from daydream.training.reward import ScoringInputs
from tests.harness.dataset import observation, read_records, run_record


@pytest.fixture
def store(tmp_path: Path) -> LocalRecordStore:
    result = LocalRecordStore(tmp_path / "records")
    assert result.commit_run(run_record()).committed
    return result


def test_history_pins_temporal_membership_and_preserves_typed_human_decisions(store: LocalRecordStore) -> None:
    assert not store.commit_run(run_record()).committed
    assert store.append_observation(observation()).committed
    assert not store.append_observation(observation()).committed
    store.append_observation(observation("future", valid_at="2026-10-05T11:00:00Z"))
    store.append_observation(observation("model", role="model-suggested", author="model",
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "suggestion"}))
    label = observation("label", item_uid=None, payload={"type": "run-label", "label": "contested",
        "reviewer_logins": ["alice"], "outcome_prior": 0.25, "outcome_prior_n": 4, "rubric": {"version": "v1"}})
    license_evidence = observation("license", item_uid=None, payload={"type": "enrichment", "kind": "license",
        "evidence": {"status": "unavailable", "reason": "not acquired"}})
    for raw in (label, license_evidence):
        store.append_observation(raw)
    snapshot = store.select_snapshot(observed_before="2026-10-04T13:00:00Z", valid_before="2026-10-04T11:30:00Z")
    read = LocalRecordStore(store.root).read_snapshot(snapshot["snapshot_id"])
    assert [run["run_id"] for run in read.runs] == ["run-1"] and read.snapshot == snapshot
    assert {obs["observation_id"] for obs in read.observations} == {"judgment-1", "future", "model", "label", "license"}
    assert {obs["observation_id"] for obs in read.eligible_observations} == {"judgment-1", "model", "label", "license"}
    history = {obs["observation_id"]: obs for obs in read.observations}
    assert history["model"]["review_required"]
    assert history["label"]["payload"]["outcome_prior"] == 0.25
    assert history["license"]["payload"]["evidence"]["status"] == "unavailable"
    judgment = read.effective_judgment("run-1", "item:1")
    assert judgment["disposition"] == "accepted" and judgment["role"] == "rater"
    store.append_observation(observation("bob", author="bob", observed_at="2026-10-04T14:00:00Z",
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "not reproducible",
                 "record_id": "c" * 64}))
    assert store.read_snapshot(snapshot) == read
    newer = read_records(store)
    judgment = newer.effective_judgment("run-1", "item:1")
    assert judgment["conflict"] and not judgment["gold_eligible"]
    assert next(obs for obs in newer.observations if obs["observation_id"] == "bob")["payload"]["record_id"]


@pytest.mark.parametrize(("kind", "raw", "diagnostic"), [
    ("run", run_record(outcome="failed"), "immutable_identity_conflict"),
    ("run", run_record(scoring={"status": "available", "value": {
        **capture_scoring(ScoringInputs(None, True, 6), "review"),
        "reward_policy": {"version": "missing", "configuration": {}}}}), "invalid_or_unknown_record_schema"),
    ("run", b"{private-malformed", "invalid_or_unknown_record_schema"),
    ("run", run_record(schema_version="daydream.run.v99"), "invalid_or_unknown_record_schema"),
    ("run", {k: v for k, v in run_record().items() if k != "schema_version"}, "invalid_or_unknown_record_schema"),
    ("run", run_record(**{"PRIVATE-CREDENTIAL-KEY": "PRIVATE-CREDENTIAL-VALUE"}), "invalid_or_unknown_record_schema"),
    *(('observation', observation(author='gpt-6', role=role), 'invalid_or_unknown_record_schema')
      for role in ('rater', 'adjudicator')),
    ("run", run_record("dirty", provenance={"url": "https://user:password@github.com/owner/repo"}), "privacy_refused"),
    ("observation", observation("unknown", run_id="unknown"), "unknown_run_reference"),
    ("observation", observation("orphan", item_uid="item:99"), "orphan_finding_reference"),
    ("observation", observation(author="bob"), "immutable_identity_conflict"),
])
def test_invalid_mutations_are_withheld(store: LocalRecordStore, kind: str, raw: Any, diagnostic: str) -> None:
    store.append_observation(observation())
    with pytest.raises(StoreError, match=diagnostic) as error:
        (store.commit_run if kind == "run" else store.append_observation)(raw)
    assert "password" not in str(error.value).lower() and "private" not in str(error.value).lower()
    read = read_records(store)
    assert len(read.runs) == len(read.observations) == 1


@pytest.mark.parametrize("unsafe", ["alias", "missing/.."])
def test_unsafe_storage_is_refused(tmp_path: Path, store: LocalRecordStore, unsafe: str) -> None:
    with pytest.raises(StoreError, match="record_too_large"):
        LocalRecordStore(tmp_path / "small", max_record_bytes=32).commit_run(run_record())
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(StoreError, match="unsafe_storage_path"):
        LocalRecordStore(tmp_path / unsafe / "records").commit_run(run_record())
    assert not (real / "records").exists() and not (tmp_path / "missing").exists()
    (real / "operator.tmp").write_text("preserve me")
    (store.root / "unowned").symlink_to(real, target_is_directory=True)
    store.commit_run(run_record("retry"))
    assert (real / "operator.tmp").read_text() == "preserve me"


def _write_concurrently(root: Path, index: int) -> None:
    for _ in range(2):
        LocalRecordStore(root).commit_run(run_record(f"run-{index}"))


def test_serialized_process_writers_preserve_all_unique_records(tmp_path: Path) -> None:
    root = tmp_path / "records"
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_write_concurrently, args=(root, index)) for index in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert {record["run_id"] for record in read_records(LocalRecordStore(root)).runs} == {f"run-{i}" for i in range(6)}


def test_failed_commit_is_not_visible_and_abandoned_staging_recovers_privately(
    store: LocalRecordStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync, record_synced = os.fsync, False
    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal record_synced
        metadata = os.fstat(descriptor)
        if record_synced and stat.S_ISDIR(metadata.st_mode):
            raise OSError("credentials must not enter diagnostics")
        original_fsync(descriptor)
        record_synced |= stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0
    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", fail_directory_fsync)
        with pytest.raises(StoreError, match="persistence_failed") as error:
            store.commit_run(run_record("retry"))
        assert "credentials" not in str(error.value)
    assert [run["run_id"] for run in read_records(store).runs] == ["run-1"]
    abandoned = store.root / "runs" / "record-abandoned.tmp"
    abandoned.write_bytes(b'{"partial":')
    assert store.commit_run(run_record("retry")).diagnostics == ("recovered_interrupted_write",)
    assert {run["run_id"] for run in read_records(store).runs} == {"run-1", "retry"}
    assert not abandoned.exists() and stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in (store.root / "runs").iterdir())


@pytest.mark.parametrize(("fault", "diagnostic"), [
    ("partial", "malformed_or_interrupted_record"), ("changed", "snapshot_record_digest_mismatch"),
    ("duplicate", "malformed_snapshot"), ("null", "malformed_snapshot"), ("later", None),
])
def test_snapshot_reads_validate_only_pinned_shards(
    store: LocalRecordStore, fault: str, diagnostic: str | None,
) -> None:
    snapshot = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    shard = store.root / "runs" / snapshot["runs"][0]["shard"]
    identity = snapshot["snapshot_id"]
    if fault == "later":
        store.commit_run(run_record("later"))
        shard = next(path for path in shard.parent.iterdir() if path != shard)
        shard.write_bytes(b"unfinished later content")
    elif fault in {"partial", "changed"}:
        original = shard.read_bytes()
        shard.write_bytes(original[:-1] if fault == "partial" else original.replace(b'"success"', b'"failed"'))
    else:
        path = store.root / "snapshots" / (identity + ".jsonl")
        if fault == "null":
            path.write_text("null\n")
        else:
            document = json.loads(path.read_text())
            document.pop("snapshot_id")
            document["runs"].append(document["runs"][0])
            identity = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            document["snapshot_id"] = identity
            path.with_name(identity + ".jsonl").write_text(json.dumps(document, separators=(",", ":")) + "\n")
    if diagnostic:
        with pytest.raises(StoreError, match=diagnostic):
            store.read_snapshot(identity)
    else:
        assert [run["run_id"] for run in store.read_snapshot(snapshot).runs] == ["run-1"]


@pytest.mark.parametrize("field", ["original_task", "trajectories", "findings", "verification", "scoring"])
def test_available_run_sections_require_complete_semantic_payload(field: str) -> None:
    raw = parse_run(run_record())
    raw[field] = {"status": "available", "value": {}}
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run(raw)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, run_record_schema())
    raw[field] = {"status": "unavailable", "reason": "not acquired"}
    jsonschema.validate(parse_run(raw), run_record_schema())


@pytest.mark.parametrize("registered_child", [False, True])
def test_trajectory_membership_preserves_invocations_and_unproduced_registered_children(registered_child: bool) -> None:
    summary: dict[str, Any] = ({"trajectory_id": "run-1-child", "invocations": []} if registered_child else
               {"trajectory_id": "run-1", "invocation_id": "attempt-1", "step_ids": [1], "phase": "review"})
    raw = run_record(outcome="interrupted", trajectories={"status": "available", "value": {
        "root_trajectory_id": "run-1", "status": "partial", "cutoff_at": "2026-10-04T12:01:00Z",
        "documents": [{"schema_version": "ATIF-v1.7", "session_id": "run-1", "trajectory_id": "run-1",
                       "agent": {"name": "daydream", "version": "test"},
                       "steps": [{"step_id": 1, "source": "user", "message": "review"}],
                       "extra": {"subtrajectories": [summary]}}]}})
    restored = parse_run(raw)
    assert restored["outcome"] == "interrupted"
    assert restored["trajectories"]["value"]["documents"][0]["extra"]["subtrajectories"] == [summary]
    if not registered_child:
        summary["step_ids"] = [9]
        with pytest.raises(ValueError, match="invalid RunRecord"):
            parse_run(raw)
