"""The public dataset commands read persisted records without archive reconstruction."""

import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore, record_name
from tests.harness.dataset import observation, run_record
from tests.harness.scripts import cli_main


@pytest.fixture(autouse=True)
def operator_hub_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "test-user/private-trajectories")


def test_dataset_commands_require_operator_destination_before_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    operation = "download"
    environment = ""
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", environment)
    monkeypatch.setenv("HF_TOKEN", "hf_offline_fixture_token")

    def forbidden_network() -> None:
        pytest.fail("Unconfigured dataset commands must not contact HF")

    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", forbidden_network)
    destination = tmp_path / "records"
    arguments = (["--revision", "a" * 40, "--output", str(destination)])
    assert cli_main(["corpus", "dataset", operation, *arguments]) == 2
    output = capsys.readouterr()
    assert "--trajectory-hub-repo" in output.err
    assert not destination.exists()


def test_dataset_download_rejects_moving_revision_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    destination = tmp_path / "download"

    def forbidden_hub() -> None:
        pytest.fail("Moving revisions must be refused before constructing a Hub client")

    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", forbidden_hub)
    assert cli_main(["corpus", "dataset", "download", "--revision", "main", "--output", str(destination)]) == 1
    output = capsys.readouterr()
    assert "revision" in (output.out + output.err).lower()
    assert not destination.exists()


def test_cli_retries_lost_publication_response_without_duplicate_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    hub.lose_response = True
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())
    publish = ["corpus", "dataset", "publish", "--store", str(store.root)]
    assert cli_main(publish) == 1
    failed = json.loads(capsys.readouterr().out)
    assert (failed["queued"], failed["published"], failed["failed"]) == (2, 0, 2)
    committed = hub.revision
    assert len(hub.commits) == 1

    assert cli_main(publish) == 0
    retried = json.loads(capsys.readouterr().out)
    assert (retried["queued"], retried["published"], retried["failed"]) == (0, 2, 0)
    assert retried["revision"] == committed and len(hub.commits) == 1
    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", "--revision", committed,
                     "--output", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["observations"] == 1
    assert LocalRecordStore(destination).read_records() == store.read_records()


def test_cli_rival_manifest_commit_preserves_both_record_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    first = LocalRecordStore(tmp_path / "first")
    rival = LocalRecordStore(tmp_path / "rival")
    first.commit_run(run_record("first"))
    rival.commit_run(run_record("rival"))
    first.append_observation(observation("first-judgment", run_id="first"))
    rival.append_observation(observation("rival-judgment", run_id="rival"))
    first.commit_run(run_record("refused", provenance={
        "key": "-----BEGIN PRIVATE KEY-----\nprivate-sensitive-body\n-----END PRIVATE KEY-----"}))
    blocked: list[bool] = []
    read_bytes = LocalRecordStore._read_bytes

    def commit_while_publish_reads(store: LocalRecordStore, path: Path) -> bytes:
        with open(first.root / ".lock", "rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                blocked.append(True)
        if not blocked and not (first.root / "runs" / record_name("later")).exists():
            first.commit_run(run_record("later"))
        return read_bytes(store, path)

    def publish_rival() -> None:
        assert cli_main(["corpus", "dataset", "publish", "--store", str(rival.root)]) == 0
        assert json.loads(capsys.readouterr().out)["published"] == 2

    monkeypatch.setattr(LocalRecordStore, "_read_bytes", commit_while_publish_reads)
    hub.before_commit = publish_rival
    assert cli_main(["corpus", "dataset", "publish", "--store", str(first.root)]) == 1
    published = json.loads(capsys.readouterr().out)
    assert not blocked, "the publisher held the store lock while reading a record"
    assert (published["published"], published["queued"], len(hub.commits)) == (2, 1, 2)
    assert published["skipped"] == [{"error": "blocking_secret_findings", "identity_sha256":
                                     hashlib.sha256(b"refused").hexdigest(), "kind": "runs"}]
    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", "--revision", hub.revision,
                     "--output", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 2
    records = LocalRecordStore(destination).read_records()
    assert {record["run_id"] for record in records["runs"]} == {"first", "rival"}
    assert {record["observation_id"] for record in records["observations"]} == {"first-judgment", "rival-judgment"}
    for store, queued, error in ((first, 2, "records_skipped"), (rival, 0, None)):
        assert cli_main(["corpus", "dataset", "status", "--store", str(store.root)]) == (1 if error else 0)
        status = json.loads(capsys.readouterr().out)
        assert (status["published"], status["queued"], status["error"]) == (2, queued, error)
    assert cli_main(["corpus", "dataset", "publish", "--store", str(first.root)]) == 1
    later = json.loads(capsys.readouterr().out)
    assert (later["published"], later["skipped"], len(hub.commits)) == (3, published["skipped"], 3)


def test_cli_rejects_corrupt_pinned_snapshot_before_exposing_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    assert cli_main(["corpus", "dataset", "publish", "--store", str(store.root)]) == 0
    capsys.readouterr()
    revision = hub.revision
    tree = hub.trees[revision]
    manifest = json.loads(tree["manifest.json"])
    manifest["schema_version"] = "unsupported"
    tree["manifest.json"] = json.dumps(manifest).encode()
    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", "--revision", revision,
                     "--output", str(destination)]) == 1
    output = capsys.readouterr()
    assert "Data Collection" in output.out
    assert "SECRET_CORRUPT_SNAPSHOT" not in output.out + output.err
    assert not destination.exists()
