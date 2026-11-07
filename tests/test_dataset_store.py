"""Public persistence, history, recovery and tamper checks on real local storage."""
import fcntl
import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jsonschema
import pytest

from daydream.dataset import LocalRecordStore, StoreError, parse_run, run_record_schema
from daydream.dataset_capture import capture_run_record, capture_scoring
from daydream.dataset_scoring import assemble_scoring_inputs
from daydream.deep.render import insert_review_context
from daydream.run_config import RunConfig
from daydream.training.reward import ScoringInputs
from tests.harness.dataset import observation, read_records, run_record

LEGACY_REVIEW = '# Review\n\n## Issues\n1. [a.py:1] Existing finding.\n'
CONTEXT_REVIEW = insert_review_context(LEGACY_REVIEW, {
    'scopes': [], 'scopes_omitted': 0, 'verification': {'status': 'unproduced', 'value': None}, 'author_reported': None,
})
SCORING = ScoringInputs([{'verdict': 'consistent'}], True, len(LEGACY_REVIEW))


@pytest.fixture
def store(tmp_path: Path) -> LocalRecordStore:
    result = LocalRecordStore(tmp_path / "records")
    assert result.commit_run(run_record()).committed
    return result


def test_current_records_read_all_history_without_creating_snapshot(store: LocalRecordStore) -> None:
    trace_fields = {"trace_id": "0123456789abcdef0123456789abcdef"}
    store.commit_run(run_record("future-run", captured_at="2200-01-01T00:00:00Z", **trace_fields))
    projected = capture_scoring(SCORING, CONTEXT_REVIEW)
    historic = capture_scoring(replace(SCORING, length=len(CONTEXT_REVIEW)), CONTEXT_REVIEW)
    legacy = capture_scoring(SCORING, LEGACY_REVIEW)
    redacted = capture_scoring(replace(SCORING, length=len(LEGACY_REVIEW) + 7), LEGACY_REVIEW)
    redacted['review_text_redaction'] = {'applied': True, 'policy': 'retained-redaction-policy'}
    store.commit_run(run_record('scoring-projected', scoring={'status': 'available', 'value': projected}))
    store.commit_run(run_record('scoring-historic', scoring={'status': 'available', 'value': historic}))
    store.commit_run(run_record('scoring-legacy', scoring={'status': 'available', 'value': legacy}))
    store.commit_run(run_record('scoring-redacted', scoring={'status': 'available', 'value': redacted}))
    store.append_observation(observation("future-observation", observed_at="2200-01-02T00:00:00Z"))
    assert list((store.root / "snapshots").iterdir()) == []
    current = store.read_records()
    assert {record["run_id"] for record in current["runs"]} == {
        "run-1", "future-run", "scoring-projected", "scoring-historic", "scoring-legacy", "scoring-redacted",
    }
    assert [record["observation_id"] for record in current["observations"]] == ["future-observation"]
    assert {record["run_id"]: record["trace_id"] for record in current["runs"]} == {
        "run-1": None, "future-run": trace_fields.get("trace_id"),
        "scoring-projected": None, "scoring-historic": None, "scoring-legacy": None, "scoring-redacted": None,
    }
    assert {r['run_id']: (r['scoring']['value']['review_text'], r['scoring']['value']['length'],
                         r['scoring']['value'].get('review_text_redaction'))
            for r in current['runs'] if r['run_id'].startswith('scoring-')} == {
        'scoring-projected': (CONTEXT_REVIEW, len(LEGACY_REVIEW), None),
        'scoring-historic': (CONTEXT_REVIEW, len(CONTEXT_REVIEW), None),
        'scoring-legacy': (LEGACY_REVIEW, len(LEGACY_REVIEW), None),
        'scoring-redacted': (LEGACY_REVIEW, len(LEGACY_REVIEW) + 7,
                             {'applied': True, 'policy': 'retained-redaction-policy'}),
    }
    source = {"repository": "test-user/private-trajectories", "revision": "a" * 40}
    store.record_download_source(source, **current)
    path = store.root / "source.json"
    assert json.loads(path.read_bytes()) == source
    assert store.download_source() == source
    assert path.read_bytes().endswith(b"\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list((store.root / "snapshots").iterdir()) == []
    pinned = store.select_snapshot(observed_before="2300-01-01T00:00:00Z")
    assert pinned["source"] == {**source, "manifest_sha256": None}
    assert store.read_snapshot(pinned["snapshot_id"]).runs == current["runs"]


def test_current_record_reads_reject_orphaned_observations(store: LocalRecordStore) -> None:
    store.append_observation(observation())
    current = store.read_records()
    path = next((store.root / "runs").iterdir())
    path.unlink()
    diagnostic = "unknown_run_reference"
    with pytest.raises(StoreError, match=diagnostic):
        store.read_records()
    with pytest.raises(StoreError, match=diagnostic):
        store.record_download_source({"revision": "a" * 40}, **current)
    assert not (store.root / "source.json").exists()
    assert list((store.root / "snapshots").iterdir()) == []


def test_download_source_refuses_symlink_without_touching_target(store: LocalRecordStore, tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("operator data")
    (store.root / "source.json").symlink_to(outside)
    with pytest.raises(StoreError, match="unsafe_storage_path"):
        store.record_download_source({"revision": "a" * 40}, **store.read_records())
    assert outside.read_text() == "operator data"


def test_invalid_mutations_are_withheld(store: LocalRecordStore) -> None:
    store.append_observation(observation())
    for kind, raw, diagnostic in (
        ("run", run_record(scoring={'status': 'available', 'value': capture_scoring(
            replace(SCORING, length=len(LEGACY_REVIEW) + 7), LEGACY_REVIEW)}), "invalid_or_unknown_record_schema"),
        ("run", b"{private-malformed", "invalid_or_unknown_record_schema"),
        ('observation', observation(author='gpt-6', role='adjudicator'), 'invalid_or_unknown_record_schema'),
        ("observation", observation("orphan", item_uid="item:99"), "orphan_finding_reference"),
        ("observation", observation(author="bob"), "immutable_identity_conflict"),
    ):
        with pytest.raises(StoreError, match=diagnostic) as error:
            (store.commit_run if kind == "run" else store.append_observation)(raw)
        assert "password" not in str(error.value).lower() and "private" not in str(error.value).lower()
        read = read_records(store)
        assert len(read.runs) == len(read.observations) == 1


def test_captured_source_record_binding_is_validated_before_storage(store: LocalRecordStore) -> None:
    """Producer round trips miss a consumer accepting mismatched source/population identities."""
    for fault in ('missing-source', 'final-source', 'final-head', 'committed-final'):
        digest = hashlib.sha256(b'').hexdigest()
        task: dict[str, Any] = {
            'analyzed_revision': {'head_sha': 'h' * 40, 'merge_base_sha': 'b' * 40,
                                  'diff_key': digest, 'source_id': 'd' * 64},
            'diff': '', 'diff_sha256': digest,
            'repository': {'repo_slug': None, 'remote_url': None, 'host': None},
            'pr': {'number': None, 'repo': None}, 'changed_files': [],
            'input_scope': 'captured_working', 'dirty_tracked': None, 'untracked_files_included': True,
        }
        final = {'head_sha': 'h' * 40, 'source_id': 'd' * 64}
        record = run_record('captured-run', original_task={'status': 'available', 'value': task},
                            final_state={'status': 'available', 'value': final})
        parse_run(record)  # Each fault starts from an independently valid current-format record.
        if fault == 'missing-source':
            del task['analyzed_revision']['source_id']
        elif fault == 'final-source':
            final['source_id'] = 'e' * 64
        elif fault == 'final-head':
            final['head_sha'] = 'x' * 40
        else:
            del task['analyzed_revision']['source_id']
            task.update(input_scope='committed', untracked_files_included=False)
        before = {path: path.read_bytes() for path in store.root.rglob('*.jsonl')}
        with pytest.raises(StoreError, match='invalid_or_unknown_record_schema'):
            store.commit_run(record)
        assert {path: path.read_bytes() for path in store.root.rglob('*.jsonl')} == before
        assert len(store.read_records()['runs']) == 1


def test_unsafe_storage_is_refused(tmp_path: Path, store: LocalRecordStore) -> None:
    small = LocalRecordStore(tmp_path / "small", max_record_bytes=32)
    with pytest.raises(StoreError, match="record_too_large"):
        small.commit_run(run_record())
    # A streamed record is refused at the cap before its documents are pulled, leaving no shard or staging.
    documents: Iterator[dict[str, Any]] = iter([{}])
    with pytest.raises(StoreError, match="record_too_large"):
        small.commit_run(run_record(trajectories={"status": "available", "value": {
            "root_trajectory_id": "run-1", "status": "complete", "cutoff_at": "2026-10-04T12:01:00Z"}}), documents)
    assert next(documents) == {} and list((small.root / "runs").iterdir()) == []
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    for unsafe in ("alias", "missing/.."):
        with pytest.raises(StoreError, match="unsafe_storage_path"):
            LocalRecordStore(tmp_path / unsafe / "records").commit_run(run_record())
        assert not (real / "records").exists() and not (tmp_path / "missing").exists()
    (real / "operator.tmp").write_text("preserve me")
    (store.root / "unowned").symlink_to(real, target_is_directory=True)
    store.commit_run(run_record("retry"))
    assert (real / "operator.tmp").read_text() == "preserve me"
    # Capture refuses a store that overlaps the run's own source, repository or artifact namespace.
    source = tmp_path / "source"
    owned = SimpleNamespace(source=source, repo=source, root=tmp_path / "artifacts" / "root")
    with pytest.raises(ValueError, match="overlaps a run-owned namespace"):
        capture_run_record(artifacts=cast(Any, owned), artifact_session=cast(Any, None), selected=None,
                           original_task=None, identity=None, work=cast(Any, owned), outcome="success",
                           config=RunConfig(dataset_store_path=source / ".daydream" / "records"))


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

    def documents() -> Iterator[dict[str, Any]]:
        """Stream while the store lock is free; another committer's recovery keeps this live stage."""
        with open(store.root / ".lock", "rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live, = (store.root / "runs").glob("*.tmp")
        assert store.commit_run(run_record("concurrent")).diagnostics == () and live.exists()
        assert {run["run_id"] for run in read_records(store).runs} == {"run-1", "retry", "concurrent"}
        yield {"schema_version": "ATIF-v1.7", "session_id": "streamed", "trajectory_id": "streamed",
               "agent": {"name": "daydream", "version": "test"},
               "steps": [{"step_id": 1, "source": "user", "message": "review"}]}
    streamed = run_record("streamed", trajectories={"status": "available", "value": {
        "root_trajectory_id": "streamed", "status": "complete", "cutoff_at": "2026-10-04T12:01:00Z"}})
    assert store.commit_run(streamed, documents()).committed
    assert {run["run_id"] for run in read_records(store).runs} == {"run-1", "retry", "concurrent", "streamed"}
    assert not abandoned.exists() and stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in (store.root / "runs").iterdir())


def test_snapshot_reads_validate_only_pinned_shards(tmp_path: Path) -> None:
    for fault, diagnostic in (("changed", "snapshot_record_digest_mismatch"), ("duplicate", "malformed_snapshot"),
                              ("null", "malformed_snapshot")):
        store = LocalRecordStore(tmp_path / fault)
        assert store.commit_run(run_record()).committed
        snapshot = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
        shard = store.root / "runs" / snapshot["runs"][0]["shard"]
        identity = snapshot["snapshot_id"]
        path = store.root / "snapshots" / (identity + ".jsonl")
        if fault == "changed":
            shard.write_bytes(shard.read_bytes().replace(b'"success"', b'"failed"'))
        elif fault == "null":
            path.write_text("null\n")
        else:
            document = json.loads(path.read_text())
            document.pop("snapshot_id")
            document["runs"].append(document["runs"][0])
            identity = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            document["snapshot_id"] = identity
            path.with_name(identity + ".jsonl").write_text(json.dumps(document, separators=(",", ":")) + "\n")
        with pytest.raises(StoreError, match=diagnostic):
            store.read_snapshot(identity)


def test_run_sections_require_complete_semantic_payload_and_bound_membership(tmp_path: Path) -> None:
    field = "trajectories"
    raw = parse_run(run_record())
    raw[field] = {"status": "available", "value": {}}
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run(raw)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, run_record_schema())
    raw[field] = {"status": "unavailable", "reason": "not acquired"}
    jsonschema.validate(parse_run(raw), run_record_schema())
    summary: dict[str, Any] = {"trajectory_id": "run-1", "invocation_id": "attempt-1", "step_ids": [1],
                               "phase": "review"}
    agent = {"name": "daydream", "version": "test"}
    documents = [{"schema_version": "ATIF-v1.7", "session_id": "run-1", "trajectory_id": "run-1", "agent": agent,
                  "steps": [{"step_id": 1, "source": "user", "message": "review"}],
                  "extra": {"subtrajectories": [summary]}},
                 {"schema_version": "ATIF-v1.7", "session_id": "run-1", "trajectory_id": "run-1:child", "agent": agent,
                  "steps": [{"step_id": 1, "source": "agent", "message": "caf\u00e9 \U0001f600",
                             "metrics": {"cost_usd": 1.5e-05}}], "extra": {"absent": None}}]
    value = {"root_trajectory_id": "run-1", "status": "partial", "cutoff_at": "2026-10-04T12:01:00Z"}
    envelope = run_record(outcome="interrupted", trajectories={"status": "available", "value": value})
    raw = run_record(outcome="interrupted", trajectories={"status": "available", "value": {
        **value, "documents": documents}})
    restored = parse_run(raw)
    assert restored["outcome"] == "interrupted"
    assert restored["trajectories"]["value"]["documents"] == documents
    # Capture's one-shot document stream and a whole mapping commit the same canonical bytes.
    expected = json.dumps(restored, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    streamed, mapped = LocalRecordStore(tmp_path / "streamed"), LocalRecordStore(tmp_path / "mapped")
    assert streamed.commit_run(envelope, (document for document in documents)).committed
    assert mapped.commit_run(raw).committed
    shards = [next((store.root / "runs").iterdir()) for store in (streamed, mapped)]
    assert [shard.read_text(encoding="utf-8") for shard in shards] == [expected, expected]
    assert not streamed.commit_run(raw).committed
    with pytest.raises(StoreError, match="immutable_identity_conflict"):  # Same size, different digest.
        mapped.commit_run({**raw, "captured_at": "2026-10-04T10:00:01Z"})
    assert shards[1].read_text(encoding="utf-8") == expected
    size = len(expected.encode())
    assert LocalRecordStore(tmp_path / "exact", max_record_bytes=size).commit_run(envelope, iter(documents)).committed
    short = LocalRecordStore(tmp_path / "short", max_record_bytes=size - 1)
    with pytest.raises(StoreError, match="record_too_large"):
        short.commit_run(envelope, iter(documents))
    assert list((short.root / "runs").iterdir()) == []
    root, child = documents
    faulted = LocalRecordStore(tmp_path / "faulted")
    for record, streamed_documents in (
        (envelope, [root, child | {"session_id": "other"}]), (envelope, [root, child | {"trajectory_id": "run-1"}]),
        (envelope, [root, child | {"steps": []}]), (envelope, [root, child | {"extra": {"x": math.nan}}]),
        (envelope, [child]), (raw, documents), (run_record(), documents),
    ):
        with pytest.raises(StoreError, match="invalid_or_unknown_record_schema"):
            faulted.commit_run(record, iter(streamed_documents))
    summary["step_ids"] = [9]
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run(raw)
    with pytest.raises(StoreError, match="invalid_or_unknown_record_schema"):
        faulted.commit_run(envelope, iter(documents))
    assert list((faulted.root / "runs").iterdir()) == []


def test_download_source_read_refuses_unsafe_provenance(
    store: LocalRecordStore, tmp_path: Path,
) -> None:
    path = store.root / "source.json"
    path.write_text("[]")
    with pytest.raises(StoreError, match="invalid_download_source|unsafe_storage_path"):
        store.download_source()


def test_scoring_inputs_without_review_evidence_earn_no_credit(tmp_path: Path) -> None:
    """A run with no verdicts and no review output in either location has no correctness or length signal."""
    assert assemble_scoring_inputs(tmp_path) == ScoringInputs(None, True, None)
