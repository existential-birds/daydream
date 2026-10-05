"""The public local dataset store persists newly captured evidence directly."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.dataset.schema import semantic_evidence_digest
from daydream.dataset.store import LocalRecordStore


def run_record(run_id: str = "run-1") -> dict[str, Any]:
    return {
        "schema_version": "daydream.run.v1", "run_id": run_id,
        "captured_at": "2026-10-04T10:00:00Z", "outcome": "success",
        "original_task": {"status": "unavailable", "reason": "not acquired"},
        "final_state": {"status": "unproduced"},
        "recommended_patch": {"status": "unproduced"},
        "trajectories": {"status": "unproduced"},
        "findings": {"status": "available", "value": {"claims": [],
            "items": [{"item_uid": "item:1", "source_uids": [], "fingerprint": "a" * 64}],
            "derivation": {}, "terminal_coverage": None}},
        "verification": {"status": "unproduced"}, "scoring": {"status": "unproduced"},
        "provenance": {}, "completeness": {},
    }


def observation(observation_id: str = "judgment-1", **overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": "daydream.observation.v1", "observation_id": observation_id,
        "run_id": "run-1", "item_uid": "item:1", "valid_at": "2026-10-04T11:00:00Z",
        "observed_at": "2026-10-04T12:00:00Z", "source": "manual", "author": "alice",
        "role": "rater", "policy_version": "policy-1", "rubric_version": "rubric-1",
        "evidence_digest": semantic_evidence_digest({"reply": "confirmed"}),
        "semantic_evidence": {"reply": "confirmed"},
        "payload": {"type": "finding-judgment", "disposition": "accepted", "rationale": "reproduced"},
        **overrides,
    }


def test_committed_run_and_observation_remain_in_an_immutable_snapshot(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())
    snapshot = store.select_snapshot(observed_before="2026-10-04T13:00:00Z", valid_before="2026-10-04T11:30:00Z")
    store.append_observation(observation("judgment-2", observed_at="2026-10-04T14:00:00Z"))
    read = LocalRecordStore(tmp_path / "records").read_snapshot(snapshot.snapshot_id)
    assert [run.run_id for run in read.runs] == ["run-1"]
    assert [obs.observation_id for obs in read.observations] == ["judgment-1"]
    assert [obs.observation_id for obs in read.eligible_observations] == ["judgment-1"]
    assert read.snapshot == snapshot


def test_snapshot_keeps_future_valid_history_and_human_precedence(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())
    store.append_observation(observation("future", valid_at="2026-10-05T11:00:00Z"))
    store.append_observation(observation("model", role="model-suggested", author="model",
        observed_at="2026-10-04T12:30:00Z",
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "suggestion"}))
    snapshot = store.select_snapshot(observed_before="2026-10-04T13:00:00Z", valid_before="2026-10-04T12:00:00Z")
    records = store.read_snapshot(snapshot)
    assert {obs.observation_id for obs in records.observations} == {"judgment-1", "future", "model"}
    assert {obs.observation_id for obs in records.eligible_observations} == {"judgment-1", "model"}
    assert records.effective_judgment("run-1", "item:1")["disposition"] == "accepted"
    assert records.effective_judgment("run-1", "item:1")["role"] == "rater"
    assert next(obs for obs in records.observations if obs.observation_id == "model").review_required is True


def test_public_store_rejects_symlinked_storage_ancestors(tmp_path: Path) -> None:
    import pytest

    from daydream.dataset.store import StoreError

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(StoreError, match="unsafe_storage_path"):
        LocalRecordStore(alias / "records").commit_run(run_record())
    assert not (real / "records").exists()


def test_same_identity_retries_are_noops_and_conflicts_are_actionable(tmp_path: Path) -> None:
    import pytest

    from daydream.dataset.store import StoreError

    store = LocalRecordStore(tmp_path / "records")
    assert store.commit_run(run_record()).committed
    assert not store.commit_run(run_record()).committed
    changed = run_record()
    changed["outcome"] = "failed"
    with pytest.raises(StoreError, match="immutable_identity_conflict"):
        store.commit_run(changed)
    assert store.append_observation(observation()).committed
    assert not store.append_observation(observation()).committed
    with pytest.raises(StoreError, match="immutable_identity_conflict"):
        store.append_observation(observation(author="bob"))
    read = store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z"))
    assert len(read.runs) == len(read.observations) == 1


def test_unknown_runs_and_orphan_findings_cannot_enter_history(tmp_path: Path) -> None:
    import pytest

    from daydream.dataset.store import StoreError

    store = LocalRecordStore(tmp_path / "records")
    with pytest.raises(StoreError, match="unknown_run_reference"):
        store.append_observation(observation())
    store.commit_run(run_record())
    with pytest.raises(StoreError, match="orphan_finding_reference"):
        store.append_observation(observation(item_uid="item:99"))
    assert store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z")).observations == ()


def test_serialized_process_writers_preserve_all_unique_records(tmp_path: Path) -> None:
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    root = tmp_path / "records"
    processes = [ctx.Process(target=_write_concurrently, args=(root, index)) for index in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    store = LocalRecordStore(root)
    read = store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z"))
    assert {record.run_id for record in read.runs} == {f"run-{index}" for index in range(6)}


def _write_concurrently(root: Path, index: int) -> None:
    store = LocalRecordStore(root)
    store.commit_run(run_record(f"run-{index}"))
    store.commit_run(run_record(f"run-{index}"))


def test_invalid_oversized_and_private_payloads_are_withheld(tmp_path: Path) -> None:
    import pytest

    from daydream.dataset.store import StoreError

    store = LocalRecordStore(tmp_path / "records")
    for payload in (b"{secret-malformed", {**run_record(), "schema_version": "daydream.run.v99"}):
        with pytest.raises(StoreError, match="invalid_or_unknown_record_schema"):
            store.commit_run(payload)
    dirty = run_record()
    dirty["provenance"] = {"url": "https://user:password@github.com/owner/repo"}
    with pytest.raises(StoreError, match="privacy_refused") as error:
        store.commit_run(dirty)
    assert "password" not in str(error.value)
    with pytest.raises(StoreError, match="record_too_large"):
        LocalRecordStore(tmp_path / "records", max_record_bytes=32).commit_run(run_record())
    assert store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z")).runs == ()


def test_persistence_failure_never_reports_commit_and_retry_is_complete(tmp_path: Path, monkeypatch: Any) -> None:
    import os
    import stat

    import pytest

    from daydream.dataset.store import StoreError

    store = LocalRecordStore(tmp_path / "records")
    original_fsync = os.fsync
    record_synced = False

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal record_synced
        metadata = os.fstat(descriptor)
        if record_synced and stat.S_ISDIR(metadata.st_mode):
            raise OSError("private credentials must not enter diagnostics")
        original_fsync(descriptor)
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0:
            record_synced = True

    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", fail_directory_fsync)
        with pytest.raises(StoreError, match="persistence_failed") as error:
            store.commit_run(run_record())
        assert "credentials" not in str(error.value)
    empty = store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z"))
    assert empty.runs == ()
    assert store.commit_run(run_record()).committed
    assert len(store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z")).runs) == 1


def test_abandoned_staging_is_recovered_with_private_storage(tmp_path: Path) -> None:
    import stat

    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    staging = tmp_path / "records" / "staging"
    abandoned = staging / "record-abandoned"
    abandoned.write_bytes(b'{"partial":')
    result = store.commit_run(run_record("run-2"))
    assert result.diagnostics == ("recovered_interrupted_write",)
    assert not abandoned.exists()
    assert stat.S_IMODE((tmp_path / "records").stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in (tmp_path / "records" / "runs").iterdir())
    assert not list(staging.iterdir())


def test_snapshot_reads_detect_tampered_or_partial_shards(tmp_path: Path) -> None:
    import pytest

    from daydream.dataset.store import StoreError

    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    snapshot = store.select_snapshot(observed_before="2026-10-05T00:00:00Z")
    shard = tmp_path / "records" / "runs" / snapshot.runs[0].shard
    original = shard.read_bytes()
    shard.write_bytes(original[:-1])
    with pytest.raises(StoreError, match="malformed_or_interrupted_record"):
        store.read_snapshot(snapshot)
    shard.write_bytes(original.replace(b'"success"', b'"failed"'))
    with pytest.raises(StoreError, match="snapshot_record_digest_mismatch"):
        store.read_snapshot(snapshot)


def test_snapshot_schema_rejects_self_consistent_but_invalid_membership(tmp_path: Path) -> None:
    import hashlib
    import json

    import pytest

    from daydream.dataset.store import StoreError

    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    snapshot = store.select_snapshot(observed_before="2026-10-05T00:00:00Z")
    path = tmp_path / "records" / "snapshots" / (snapshot.snapshot_id + ".jsonl")
    document = json.loads(path.read_text())
    document.pop("snapshot_id")
    document["runs"].append(document["runs"][0])
    digest = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    document["snapshot_id"] = digest
    invalid = path.with_name(digest + ".jsonl")
    invalid.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(StoreError, match="malformed_snapshot"):
        store.read_snapshot(digest)


def test_typed_labels_enrichment_and_redacted_corrections_round_trip(tmp_path: Path) -> None:
    import hashlib

    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    text = "Use [REDACTED_API_KEY] only through the environment."
    correction = {
        "status": "available", "source_reply_id": "reply-77", "body_sha256": "b" * 64,
        "text": text, "captured_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "redaction_provenance": {"policy": "shared-redaction-v1"},
    }
    store.append_observation(observation(correction=correction))
    store.append_observation(observation("run-label", item_uid=None,
        payload={"type": "run-label", "label": "contested", "reviewer_logins": ["alice"],
                 "outcome_prior": 0.25, "outcome_prior_n": 4, "rubric": {"version": "v1"}}))
    store.append_observation(observation("license", item_uid=None,
        payload={"type": "enrichment", "kind": "license",
                 "evidence": {"status": "unavailable", "reason": "not acquired"}}))
    records = store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z"))
    history = {record.observation_id: record for record in records.observations}
    captured = history["judgment-1"].correction
    assert captured is not None
    assert captured.text == text and captured.body_sha256 == "b" * 64
    assert captured.captured_sha256 != captured.body_sha256
    assert history["judgment-1"].semantic_evidence == {"reply": "confirmed"}
    assert history["run-label"].payload.model_dump()["outcome_prior"] == 0.25
    assert history["license"].payload.model_dump()["evidence"]["status"] == "unavailable"


def test_old_snapshot_read_does_not_consume_later_unselected_shards(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    snapshot = store.select_snapshot(observed_before="2026-10-05T00:00:00Z")
    store.commit_run(run_record("later-run"))
    paths = list((tmp_path / "records" / "runs").iterdir())
    later = next(path for path in paths if path.name != snapshot.runs[0].shard)
    later.write_bytes(b'unknown, unfinished subsequent content')
    assert [run.run_id for run in store.read_snapshot(snapshot).runs] == ["run-1"]


def test_malformed_snapshot_objects_have_sanitized_diagnostics(tmp_path: Path) -> None:
    import pytest

    from daydream.dataset.store import StoreError

    store = LocalRecordStore(tmp_path / "records")
    store.select_snapshot(observed_before="2026-10-05T00:00:00Z")
    identity = "f" * 64
    path = tmp_path / "records" / "snapshots" / (identity + ".jsonl")
    path.write_text("null\n")
    with pytest.raises(StoreError, match="malformed_snapshot"):
        store.read_snapshot(identity)


def test_raw_finding_history_keeps_rater_conflicts_across_optional_corpus_ids(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())
    store.append_observation(observation("other-rater", author="bob", observed_at="2026-10-04T12:30:00Z",
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "not reproducible",
                 "record_id": "c" * 64}))
    read = store.read_snapshot(store.select_snapshot(observed_before="2026-10-05T00:00:00Z"))
    judgment = read.effective_judgment("run-1", "item:1")
    assert judgment["conflict"] is True
    assert judgment["gold_eligible"] is False
    assert next(record for record in read.observations if record.observation_id == "other-rater").payload.model_dump()[
        "record_id"
    ] == "c" * 64
