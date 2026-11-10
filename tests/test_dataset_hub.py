"""Real JSONL queue, atomic publication and pinned download via an offline HF boundary."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import stat
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from daydream.dataset import LocalRecordStore, StoreError, record_name
from tests.harness.dataset import observation, read_records, run_record

TEST_REPO = "test-user/private-trajectories"
SECRET = "-----BEGIN PRIVATE KEY-----\nprivate-sensitive-body\n-----END PRIVATE KEY-----"


@pytest.fixture
def api() -> Any:
    try:
        return importlib.import_module("daydream.dataset_hub")
    except ModuleNotFoundError:
        return None


def test_offline_queue_retry_download_and_immutable_history(
    tmp_path: Path, api: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    store.commit_run(run_record("leaked", provenance={"key": SECRET}))
    with monkeypatch.context() as earlier_scanner:  # a published record that today's scanner flags
        earlier_scanner.setattr(api, "_scan", lambda record: None)
        later = publisher.upload()
    assert later.published == 4 and len(hub.commits) == 2
    hub.reads.clear()
    assert publisher.upload().revision == later.revision and len(hub.commits) == 2
    assert set(hub.reads) == {api.MANIFEST_PATH}
    leaked = api.download_snapshot(TEST_REPO, later.revision, tmp_path / "leaked", backend=hub)
    assert len(read_records(leaked).runs) == 2
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


def test_rejected_records_are_skipped_with_safe_errors(tmp_path: Path, api: Any) -> None:
    """A refused record and the observation that targets it stay local; other records still publish.

    The observation reports its run's local refusal, or ``unknown_run_reference`` for a valid run refused only
    for publication.
    """
    from tests.harness.dataset_hub import FakeDatasetHub
    for fault, expected in (("oversized", "record_exceeds_shard_limit"), ("secret", "blocking_secret_findings"),
                            ("schema", "invalid_or_unknown_record_schema"),
                            ("partial", "malformed_or_interrupted_record"), ("path", "unsafe_storage_path"),
                            ("unknown", "unknown_shard")):
        root = tmp_path / fault
        hub = FakeDatasetHub()
        store = LocalRecordStore(root / "records")
        store.commit_run(run_record("kept"))
        store.commit_run(run_record(provenance={"nested": SECRET} if fault == "secret" else
                                    {"pad": "x" * 4096} if fault == "oversized" else {}))
        store.append_observation(observation())
        shard = store.root / "runs" / record_name("run-1")
        if fault == "schema":
            shard.write_bytes(shard.read_bytes().replace(b'daydream.run.v1', b'daydream.run.v9'))
        if fault == "partial":
            shard.write_bytes(shard.read_bytes()[:-1])
        if fault == "path":
            target = root / "outside"
            target.write_bytes(shard.read_bytes())
            shard.unlink()
            shard.symlink_to(target)
        if fault == "unknown":
            (store.root / "runs" / "notes.txt").write_text("not a record\n")
        publisher = api.DatasetUploader(store, TEST_REPO, backend=hub, max_shard_bytes=2048)
        result = publisher.upload()
        assert shard.exists() and "private-sensitive-body" not in str(result)
        if fault in ("path", "unknown"):
            assert result.error == expected and result.published == 0 and not hub.commits
            continue
        assert (result.queued, result.published, result.failed, result.error) == (2, 1, 2, None)
        assert result.skipped == (
            api.SkippedRecord("observations", hashlib.sha256(b"judgment-1").hexdigest(),
                              "unknown_run_reference" if fault in ("oversized", "secret") else expected),
            api.SkippedRecord("runs", hashlib.sha256(b"run-1").hexdigest(), expected),
        )
        manifest = json.loads(hub.trees[hub.revision][api.MANIFEST_PATH])
        assert [record["identity"] for kind in ("runs", "observations") for shard_entry in manifest["shards"][kind]
                for record in shard_entry["records"]] == ["kept"]
        assert publisher.upload().skipped == result.skipped and len(hub.commits) == 1


def test_bounded_publication_keeps_whole_records_and_identity_conflicts_fail(tmp_path: Path, api: Any) -> None:
    """Shards and commits stay within their limits, memory stays flat as stored history grows,
    and an identity conflict fails before any commit."""
    from tests.harness.dataset_hub import FakeDatasetHub
    between = [LocalRecordStore(tmp_path / "between")]
    between[0].commit_run(run_record("between"))

    class RivalBetweenCommits(FakeDatasetHub):
        def private_revision(self, repo_id: str) -> str:
            if len(self.commits) == 1 and between:
                api.DatasetUploader(between.pop(), TEST_REPO, backend=self).upload()
            return super().private_revision(repo_id)

    hub = RivalBetweenCommits()
    store = LocalRecordStore(tmp_path / "records")
    for i in range(5):
        store.commit_run(run_record(f"run-{i}"))
    store.append_observation(observation())
    bound = 2 * max(path.stat().st_size for path in (store.root / "runs").iterdir())
    empty = hub.revision
    result = api.DatasetUploader(store, TEST_REPO, backend=hub, max_shard_records=2, max_shard_bytes=bound,
                                 max_commit_bytes=bound).upload()
    manifest = json.loads(hub.trees[result.revision][api.MANIFEST_PATH])
    assert sorted(shard["count"] for shard in manifest["shards"]["runs"]) == [1, 1, 2, 2]
    assert result.published == 6 and len(hub.commits) == 4
    for parent, child in itertools.pairwise([empty, *hub.commits]):
        added = set(hub.trees[child]) - set(hub.trees[parent]) - {api.MANIFEST_PATH}
        assert 0 < sum(len(hub.trees[child][path]) for path in added) <= bound
    rival = LocalRecordStore(tmp_path / "rival")
    rival.commit_run(run_record("run-0", outcome="failed"))
    failure = api.DatasetUploader(rival, TEST_REPO, backend=hub).upload()
    assert failure.error == "immutable_identity_conflict" and len(hub.commits) == 4

    def publication_peak(history: int) -> int:
        stored = LocalRecordStore(tmp_path / f"history-{history}")
        remote = FakeDatasetHub()
        for i in range(history):
            stored.commit_run(run_record(f"stored-{i}", provenance={"pad": "x" * 32768}))
        api.DatasetUploader(stored, TEST_REPO, backend=remote, max_shard_records=1).upload()
        stored.commit_run(run_record("new", provenance={"pad": "x" * 32768}))
        tracemalloc.start()
        try:
            assert api.DatasetUploader(stored, TEST_REPO, backend=remote).upload().published == history + 1
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    few = publication_peak(2)  # first, so one-time warm-up cannot inflate the larger history
    assert publication_peak(8) - few < 32768


def test_corrupt_pinned_snapshots_never_expose_records(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub
    for fault in ("digest", "count", "schema", "duplicate", "missing", "partial", "record", "orphan"):
        root = tmp_path / fault
        hub = FakeDatasetHub()
        store = LocalRecordStore(root / "records")
        store.commit_run(run_record())
        if fault == "orphan":
            store.append_observation(observation())
        result = api.DatasetUploader(store, TEST_REPO, backend=hub).upload()
        tree = dict(hub.trees[result.revision])
        manifest = json.loads(tree[api.MANIFEST_PATH])
        shard = manifest["shards"]["runs"][0]
        if fault == "digest":
            tree[shard["path"]] += b"broken\n"
        elif fault == "count":
            shard["count"] += 1
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
        elif fault == "orphan":
            manifest["shards"]["runs"] = []
        tree[api.MANIFEST_PATH] = json.dumps(manifest).encode()
        revision = hub.commit(TEST_REPO, tree, hub.revision)
        if fault == "missing":
            hub.trees[revision].pop(shard["path"], None)
        with pytest.raises(StoreError):
            api.download_snapshot(TEST_REPO, revision, root / "download", backend=hub)
        assert not (root / "download" / "runs").exists()


def test_download_requires_exact_commit(tmp_path: Path, api: Any) -> None:
    """The public download API must reject a moving pin before opening its backend."""
    from tests.harness.dataset_hub import FakeDatasetHub

    class ForbiddenHub(FakeDatasetHub):
        def private_revision(self, repo_id: str) -> str:
            raise AssertionError("invalid download pin reached the network")

    destination = tmp_path / "download"
    with pytest.raises(StoreError, match="invalid_revision"):
        api.download_snapshot(TEST_REPO, "main", destination, backend=ForbiddenHub())
    assert not destination.exists()


def test_interruption_after_remote_commit_recovers_from_content_identity(tmp_path: Path, api: Any) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    class InterruptedHub(FakeDatasetHub):
        interrupt = True

        def read_file(self, repo_id: str, path: str, revision: str) -> bytes | None:
            if self.commits and self.interrupt:
                self.interrupt = False
                return None  # the committed manifest cannot be read back
            return super().read_file(repo_id, path, revision)

    hub = InterruptedHub()
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    assert api.DatasetUploader(store, TEST_REPO, backend=hub).upload().error == "publication_not_confirmed"
    restarted = api.DatasetUploader(LocalRecordStore(store.root), TEST_REPO, backend=hub)
    assert restarted.status().queued == 1 and restarted.status().published == 0
    assert restarted.upload().published == 1 and len(hub.commits) == 1
    published = store.root / "runs" / record_name("run-1")
    original = published.read_bytes()
    for index, (damaged, code) in enumerate((
        (original[:-1], "malformed_or_interrupted_record"),
        (original.replace(b"daydream.run.v1", b"daydream.run.v9"), "invalid_or_unknown_record_schema"),
    )):
        published.write_bytes(damaged)
        store.commit_run(run_record(f"after-{index}"))
        after = restarted.upload()
        assert (after.published, after.error, len(hub.commits)) == (index + 1, None, index + 2)
        assert after.skipped == (api.SkippedRecord("runs", hashlib.sha256(b"run-1").hexdigest(), code),)


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


def test_supported_publication_refuses_historical_manifests_without_rewriting_history(
    tmp_path: Path, api: Any,
) -> None:
    version = "daydream.hub.v2"
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
