"""Real JSONL queue, atomic publication and pinned download via an offline HF boundary."""
from __future__ import annotations

import hashlib
import importlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from daydream.dataset import LocalRecordStore, StoreError
from tests.harness.dataset import observation, read_records, run_record

TEST_REPO = "test-user/private-trajectories"


@pytest.fixture
def api() -> Any:
    try:
        return importlib.import_module("daydream.dataset_hub")
    except ModuleNotFoundError:
        return None


def test_offline_queue_retry_download_and_immutable_history(tmp_path: Path, api: Any) -> None:
    assert api is not None, "The JSONL HF publication API is not implemented"
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "source")
    store.commit_run(run_record())
    store.append_observation(observation())
    publisher = api.DatasetUploader(store, TEST_REPO, backend=hub)
    hub.fail_before = True
    failure = publisher.upload()
    assert (failure.queued, failure.failed, failure.published, failure.error) == (2, 2, 0, "network_failed")
    hub.fail_before = False
    publisher = api.DatasetUploader(LocalRecordStore(store.root), TEST_REPO, backend=hub)
    result = publisher.upload()
    assert (result.queued, result.published, result.failed) == (0, 2, 0)
    assert publisher.status() == result
    assert len(hub.commits) == 1
    tree = hub.trees[result.revision]
    manifest = json.loads(tree[api.MANIFEST_PATH])
    assert manifest["schema_version"] == "daydream.hub.v3"
    assert manifest["record_schemas"] == {"runs": ["daydream.run.v1"],
        "observations": ["daydream.observation.v3"]}
    for kind, shards in manifest["shards"].items():
        assert len(shards) == 1
        shard = shards[0]
        assert shard["path"].startswith(kind + "/")
        assert hashlib.sha256(tree[shard["path"]]).hexdigest() == shard["sha256"]
        assert shard["count"] == 1 and shard["bytes"] == len(tree[shard["path"]])
    downloaded = api.download_snapshot(TEST_REPO, result.revision, tmp_path / "download", backend=hub)
    before = read_records(downloaded)
    assert before.runs == read_records(store).runs and before.observations == read_records(store).observations
    assert before.effective_judgment("run-1", "item:1")["disposition"] == "accepted"
    assert stat.S_IMODE(downloaded.root.stat().st_mode) == 0o700
    store.append_observation(observation("later", observed_at="2026-10-04T14:00:00Z"))
    later = publisher.upload()
    assert later.published == 3 and len(hub.commits) == 2
    assert publisher.upload().revision == later.revision and len(hub.commits) == 2
    old = api.download_snapshot(TEST_REPO, result.revision, tmp_path / "old", backend=hub)
    assert read_records(old).observations == before.observations


def test_credentials_alone_do_not_construct_a_hub(tmp_path: Path, api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "private-do-not-export")
    def forbidden() -> None:
        pytest.fail("An unconfigured uploader must not contact HF")
    monkeypatch.setattr(api, "HfDatasetHub", forbidden)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    result = api.DatasetUploader(store).upload()
    assert (result.queued, result.published, result.failed) == (1, 0, 0)


def test_uncertain_success_is_recovered_without_republishing(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub
    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    hub.lose_response = True
    failure = api.DatasetUploader(store, TEST_REPO, backend=hub).upload()
    assert failure.queued == failure.failed == 1 and len(hub.commits) == 1
    result = api.DatasetUploader(LocalRecordStore(store.root), TEST_REPO, backend=hub).upload()
    assert result.published == 1 and result.queued == 0 and len(hub.commits) == 1


def test_concurrent_manifest_update_preserves_rival_records(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub
    hub = FakeDatasetHub()
    first, rival = LocalRecordStore(tmp_path / "first"), LocalRecordStore(tmp_path / "rival")
    first.commit_run(run_record("first"))
    rival.commit_run(run_record("rival"))
    hub.before_commit = lambda: api.DatasetUploader(rival, TEST_REPO, backend=hub).upload()
    result = api.DatasetUploader(first, TEST_REPO, backend=hub).upload()
    assert result.queued == 0 and len(hub.commits) == 2
    downloaded = api.download_snapshot(TEST_REPO, result.revision, tmp_path / "out", backend=hub)
    assert {record["run_id"] for record in read_records(downloaded).runs} == {"first", "rival"}


@pytest.mark.parametrize("fault", ["public", "secret", "reply_secret", "oversized", "partial", "schema", "path"])
def test_rejected_records_are_queued_with_safe_errors(tmp_path: Path, api: Any, fault: str) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub
    hub = FakeDatasetHub(private=fault != "public")
    store = LocalRecordStore(tmp_path / "records")
    secret = "-----BEGIN PRIVATE KEY-----\nprivate-sensitive-body\n-----END PRIVATE KEY-----"
    store.commit_run(run_record(provenance={"nested": secret} if fault == "secret" else {}))
    if fault == "reply_secret":
        from daydream.training.labeler_versions import reply_evidence_digest
        source_hash = hashlib.sha256(secret.encode()).hexdigest()
        evidence = [{"reply_id": 123, "body_sha256": source_hash}]
        store.append_observation(observation(semantic_evidence=evidence,
            evidence_digest=reply_evidence_digest(evidence), evidence_digest_scheme="reply-evidence-v1",
            reply_captures=[{"status": "available", "source_reply_id": "123", "body_sha256": source_hash,
                             "captured_sha256": source_hash, "text": secret}]))
    expected = {"public": "public_destination", "secret": "blocking_secret_findings",
                "reply_secret": "blocking_secret_findings",
                "oversized": "record_exceeds_shard_limit", "partial": "malformed_or_interrupted_record",
                "schema": "invalid_or_unknown_record_schema", "path": "unsafe_storage_path"}[fault]
    shard = next((store.root / "runs").iterdir())
    if fault == "partial":
        shard.write_bytes(shard.read_bytes()[:-1])
    if fault == "schema":
        shard.write_bytes(shard.read_bytes().replace(b'daydream.run.v1', b'daydream.run.v9'))
    if fault == "path":
        target = tmp_path / "outside"
        target.write_bytes(shard.read_bytes())
        shard.unlink()
        shard.symlink_to(target)
    result = api.DatasetUploader(store, TEST_REPO, backend=hub,
                                 max_shard_bytes=20 if fault == "oversized" else 1024 * 1024).upload()
    assert result.error == expected and result.published == 0
    assert not hub.commits and "private-sensitive-body" not in str(result)
    assert shard.exists()


def test_bounded_shards_keep_whole_records_and_identity_conflicts_fail(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub
    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "records")
    for i in range(5):
        store.commit_run(run_record(f"run-{i}"))
    result = api.DatasetUploader(store, TEST_REPO, backend=hub, max_shard_records=2).upload()
    manifest = json.loads(hub.trees[result.revision][api.MANIFEST_PATH])
    assert sorted(shard["count"] for shard in manifest["shards"]["runs"]) == [1, 2, 2]
    rival = LocalRecordStore(tmp_path / "rival")
    rival.commit_run(run_record("run-0", outcome="failed"))
    failure = api.DatasetUploader(rival, TEST_REPO, backend=hub).upload()
    assert failure.error == "immutable_identity_conflict" and len(hub.commits) == 1


@pytest.mark.parametrize("fault", ["digest", "count", "path", "schema", "duplicate", "missing", "partial", "record"])
def test_corrupt_pinned_snapshots_never_expose_records(tmp_path: Path, api: Any, fault: str) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub
    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    result = api.DatasetUploader(store, TEST_REPO, backend=hub).upload()
    tree = dict(hub.trees[result.revision])
    manifest = json.loads(tree[api.MANIFEST_PATH])
    shard = manifest["shards"]["runs"][0]
    if fault == "digest":
        tree[shard["path"]] += b"broken\n"
    elif fault == "count":
        shard["count"] += 1
    elif fault == "path":
        shard["path"] = "../outside.jsonl"
    elif fault == "schema":
        manifest["record_schemas"]["runs"] = "daydream.run.v99"
    elif fault == "duplicate":
        manifest["shards"]["runs"].append(shard)
    elif fault == "missing":
        del tree[shard["path"]]
    elif fault in {"partial", "record"}:
        data = tree.pop(shard["path"])
        data = data[:-1] if fault == "partial" else data.replace(b'"success"', b'"failed"')
        shard["sha256"] = hashlib.sha256(data).hexdigest()
        shard["bytes"] = len(data)
        shard["path"] = "runs/" + shard["sha256"] + ".jsonl"
        tree[shard["path"]] = data
    tree[api.MANIFEST_PATH] = json.dumps(manifest).encode()
    revision = hub.commit(TEST_REPO, tree, hub.revision)
    if fault == "missing":
        hub.trees[revision].pop(shard["path"], None)
    with pytest.raises(StoreError):
        api.download_snapshot(TEST_REPO, revision, tmp_path / "download", backend=hub)
    assert not (tmp_path / "download" / "runs").exists()


def test_download_requires_exact_commit(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub
    with pytest.raises(StoreError, match="invalid_revision"):
        api.download_snapshot(TEST_REPO, "main", tmp_path / "download", backend=FakeDatasetHub())


def test_interruption_after_remote_commit_recovers_from_content_identity(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    class InterruptedHub(FakeDatasetHub):
        interrupt = True

        def read_file(self, repo_id: str, path: str, revision: str) -> bytes | None:
            if self.commits and self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt
            return super().read_file(repo_id, path, revision)

    hub = InterruptedHub()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    with pytest.raises(KeyboardInterrupt):
        api.DatasetUploader(store, TEST_REPO, backend=hub).upload()
    restarted = api.DatasetUploader(LocalRecordStore(store.root), TEST_REPO, backend=hub)
    assert restarted.status().queued == 1 and restarted.status().published == 0
    assert restarted.upload().published == 1 and len(hub.commits) == 1


def test_failed_queue_write_keeps_original_and_retries_privately(
    tmp_path: Path, api: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    hub = FakeDatasetHub()
    original = api.atomic_write_bytes

    def fail_write(path: Path, data: bytes, **options: Any) -> None:
        if path.suffix == ".jsonl":
            raise OSError("private-do-not-export")
        original(path, data, **options)

    with monkeypatch.context() as fault:
        fault.setattr(api, "atomic_write_bytes", fail_write)
        status = api.DatasetUploader(store, TEST_REPO, backend=hub).upload()
    assert status.error == "queue_persistence_failed" and status.queued == 1
    assert not hub.commits and len(read_records(store).runs) == 1
    assert api.DatasetUploader(store, TEST_REPO, backend=hub).upload().published == 1


def test_download_refuses_unrelated_destination_data(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    source, destination = LocalRecordStore(tmp_path / "source"), LocalRecordStore(tmp_path / "destination")
    source.commit_run(run_record())
    destination.commit_run(run_record("unrelated"))
    result = api.DatasetUploader(source, TEST_REPO, backend=hub).upload()
    with pytest.raises(StoreError, match="download_destination_conflict"):
        api.download_snapshot(TEST_REPO, result.revision, destination.root, backend=hub)
    assert [record["run_id"] for record in read_records(destination).runs] == ["unrelated"]


def test_byte_limit_splits_only_between_complete_records(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "records")
    for i in range(3):
        store.commit_run(run_record(f"run-{i}"))
    largest = max(path.stat().st_size for path in (store.root / "runs").iterdir())
    result = api.DatasetUploader(store, TEST_REPO, backend=hub, max_shard_bytes=largest * 2).upload()
    tree = hub.trees[result.revision]
    shards = json.loads(tree[api.MANIFEST_PATH])["shards"]["runs"]
    assert sorted(shard["count"] for shard in shards) == [1, 2]
    assert all(shard["bytes"] <= largest * 2 for shard in shards)
    assert sum(len(tree[shard["path"]].splitlines()) for shard in shards) == 3


def test_local_upload_writers_serialize_acknowledgements(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())

    def upload() -> Any:
        return api.DatasetUploader(LocalRecordStore(store.root), TEST_REPO, backend=hub).upload()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: upload(), range(4)))
    assert all((result.queued, result.published, result.failed) == (0, 2, 0) for result in results)
    assert len(hub.commits) == 1
    uploader = api.DatasetUploader(store, TEST_REPO, backend=hub)
    assert uploader.status().revision == hub.revision
    uploads = store.root / "uploads"
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in uploads.iterdir())
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in uploads.rglob("*") if path.is_file())


def test_download_rejects_concurrent_destination_capture_before_pinning_source(
    tmp_path: Path, api: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    source = LocalRecordStore(tmp_path / "source")
    source.commit_run(run_record("pinned"))
    hub = FakeDatasetHub()
    result = api.DatasetUploader(source, TEST_REPO, backend=hub).upload()
    destination = tmp_path / "download"
    original = LocalRecordStore.commit_run

    def capture_during_install(store: LocalRecordStore, record: Any) -> Any:
        committed = original(store, record)
        if store.root == destination.absolute():
            original(store, run_record("concurrent-unrelated"))
        return committed

    monkeypatch.setattr(LocalRecordStore, "commit_run", capture_during_install)
    with pytest.raises(StoreError, match="download_destination_conflict"):
        api.download_snapshot(TEST_REPO, result.revision, destination, backend=hub)
    assert not (destination / "source.json").exists()


@pytest.mark.parametrize("version", ["daydream.hub.v1", "daydream.hub.v2"])
def test_supported_publication_refuses_historical_manifests_without_rewriting_history(
    tmp_path: Path, api: Any, version: str,
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    publisher = api.DatasetUploader(store, TEST_REPO, backend=hub)
    published = publisher.upload()
    previous = json.loads(hub.trees[published.revision][api.MANIFEST_PATH])
    previous["schema_version"] = version
    old_pin = hub.commit(TEST_REPO, {api.MANIFEST_PATH: json.dumps(previous).encode()}, hub.revision)
    before = dict(hub.trees[old_pin])
    with pytest.raises(StoreError):
        api.download_snapshot(TEST_REPO, old_pin, tmp_path / "old", backend=hub)
    store.append_observation(observation())
    result = publisher.upload()
    assert result.failed > 0
    assert hub.revision == old_pin
    assert hub.trees[old_pin] == before


def test_snapshot_record_version_must_match_its_manifest_declaration(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation(schema_version="daydream.observation.v3"))
    published = api.DatasetUploader(store, TEST_REPO, backend=hub).upload()
    manifest = json.loads(hub.trees[published.revision][api.MANIFEST_PATH])
    shard = manifest["shards"]["observations"][0]
    data = hub.trees[published.revision][shard["path"]].replace(b"daydream.observation.v3", b"daydream.observation.v9")
    shard["sha256"] = hashlib.sha256(data).hexdigest()
    shard["path"] = "observations/" + shard["sha256"] + ".jsonl"
    forged_pin = hub.commit(TEST_REPO, {api.MANIFEST_PATH: json.dumps(manifest).encode(), shard["path"]: data},
                           hub.revision)
    destination = tmp_path / "download"
    with pytest.raises(StoreError, match="invalid_or_unknown_record_schema"):
        api.download_snapshot(TEST_REPO, forged_pin, destination, backend=hub)
    assert not destination.exists()
