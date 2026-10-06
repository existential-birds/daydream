"""The public dataset commands read persisted records without archive reconstruction."""
import json
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore
from tests.harness.dataset import observation, run_record
from tests.harness.scripts import cli_main


@pytest.fixture(autouse=True)
def operator_hub_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "test-user/private-trajectories")


@pytest.mark.parametrize(("cli_repo", "expected_repo"), [
    (None, "operator/env-trajectories"),
    ("operator/cli-trajectories", "operator/cli-trajectories"),
])
def test_operator_destination_selects_publication_and_pinned_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    cli_repo: str | None, expected_repo: str,
) -> None:
    from daydream.dataset_hub_client import HubError
    from tests.harness.dataset_hub import FakeDatasetHub

    class OperatorHub(FakeDatasetHub):
        def private_revision(self, repo_id: str) -> str:
            if repo_id != expected_repo:
                raise HubError("network_failed")
            return super().private_revision(repo_id)

    hub = OperatorHub()
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "operator/env-trajectories")
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record("operator-run"))
    arguments = [] if cli_repo is None else ["--trajectory-hub-repo", cli_repo]
    assert cli_main(["corpus", "dataset", "publish", *arguments, "--store", str(store.root)]) == 0
    assert json.loads(capsys.readouterr().out)["published"] == 1
    assert cli_main(["corpus", "dataset", "status", *arguments, "--store", str(store.root)]) == 0
    assert json.loads(capsys.readouterr().out)["published"] == 1
    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", *arguments, "--revision", hub.revision,
                     "--output", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 1
    assert LocalRecordStore(destination).read_records()["runs"][0]["run_id"] == "operator-run"


@pytest.mark.parametrize("operation", ["publish", "status", "download"])
@pytest.mark.parametrize("environment", [None, ""])
def test_dataset_commands_require_operator_destination_before_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    operation: str, environment: str | None,
) -> None:
    if environment is None:
        monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    else:
        monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", environment)
    monkeypatch.setenv("HF_TOKEN", "hf_offline_fixture_token")

    def forbidden_network() -> None:
        pytest.fail("Unconfigured dataset commands must not contact HF")

    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", forbidden_network)
    destination = tmp_path / "records"
    arguments = (["--revision", "a" * 40, "--output", str(destination)] if operation == "download"
                 else ["--store", str(destination)])
    assert cli_main(["corpus", "dataset", operation, *arguments]) == 2
    output = capsys.readouterr()
    assert "--trajectory-hub-repo" in output.err
    assert "--repo" not in output.out + output.err
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["publish", "status", "download"])
@pytest.mark.parametrize("legacy_arguments", [["--repo", "OWNER/REPO"], ["--repo=OWNER/REPO"]])
def test_dataset_commands_reject_repo_flag_before_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    operation: str, legacy_arguments: list[str],
) -> None:
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "OWNER/REPO")
    monkeypatch.setenv("HF_TOKEN", "hf_offline_fixture_token")

    def forbidden_network() -> None:
        pytest.fail("Rejected dataset arguments must not contact HF")

    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", forbidden_network)
    destination = tmp_path / "records"
    arguments = (["--revision", "a" * 40, "--output", str(destination)] if operation == "download"
                 else ["--store", str(destination)])
    assert cli_main(["corpus", "dataset", operation, *legacy_arguments, *arguments]) == 2
    output = capsys.readouterr()
    assert "unrecognized arguments: " + " ".join(legacy_arguments) in output.err
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["publish", "status", "download"])
def test_dataset_command_help_exposes_canonical_destination(
    capsys: pytest.CaptureFixture[str], operation: str,
) -> None:
    assert cli_main(["corpus", "dataset", operation, "--help"]) == 0
    output = capsys.readouterr()
    assert "--trajectory-hub-repo OWNER/REPO" in output.out
    assert "DAYDREAM_TRAJECTORY_HUB_REPO" in output.out
    assert "--repo" not in output.out + output.err


def test_dataset_status_reads_local_queue_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())
    assert cli_main(["corpus", "dataset", "status", "--trajectory-hub-repo", "test-user/private-trajectories",
                     "--store", str(store.root)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["queued"] == 2 and summary["published"] == 0 and summary["failed"] == 0


def test_dataset_download_rejects_moving_revision_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", "--revision", "main", "--output", str(destination)]) == 1
    output = capsys.readouterr()
    assert "revision" in (output.out + output.err).lower()
    assert not destination.exists()


def test_dataset_namespace_lists_its_operations(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["corpus", "dataset"]) == 2
    output = capsys.readouterr()
    assert all(operation in output.out for operation in ("publish", "status", "download"))


def test_real_review_explicit_capture_opt_out(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.stub_backend import install_stub_backend, silence

    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target)
    store = tmp_path / "records"
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "test-user/private-trajectories")
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda _repo, **_kwargs: None)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda _repo, _branch, **_kwargs: None)
    assert cli_main(["--review", "--stack", "python", "--no-archive", "--no-eval", "--dataset-store", str(store),
                     "--no-capture-data", str(multi_stack_target)]) == 0
    assert not store.exists()
    assert (multi_stack_target / ".review-output.md").is_file()


def test_dataset_publish_download_round_trip_through_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.harness.dataset import read_records
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())
    arguments = ["--trajectory-hub-repo", "OWNER/REPO"]
    assert cli_main(["corpus", "dataset", "publish", *arguments, "--store", str(store.root)]) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["published"] == 2 and published["queued"] == 0
    assert published["revision"] == hub.revision
    assert cli_main(["corpus", "dataset", "status", *arguments, "--store", str(store.root)]) == 0
    assert json.loads(capsys.readouterr().out)["published"] == 2

    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", *arguments, "--revision", hub.revision,
                     "--output", str(destination)]) == 0
    downloaded = json.loads(capsys.readouterr().out)
    assert downloaded == {"revision": hub.revision, "runs": 1, "observations": 1}
    assert read_records(LocalRecordStore(destination)) == read_records(store)
    assert not (destination / "index.db").exists()


def test_dataset_publish_failed_content_keeps_queue_with_sanitized_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub(private=False)
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    assert cli_main(["corpus", "dataset", "publish", "--store", str(store.root)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["queued"] == 1 and summary["failed"] == 1 and summary["published"] == 0
    assert summary["error"] == "public_destination"
    assert hub.commits == []


def test_dataset_download_counts_all_records_without_temporal_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record(captured_at="9999-12-31T23:59:59.500000+00:00"))
    assert cli_main(["corpus", "dataset", "publish", "--store", str(store.root)]) == 0
    capsys.readouterr()
    assert cli_main(["corpus", "dataset", "download", "--revision", hub.revision,
                     "--output", str(tmp_path / "download")]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 1


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

    def publish_rival() -> None:
        assert cli_main(["corpus", "dataset", "publish", "--store", str(rival.root)]) == 0
        assert json.loads(capsys.readouterr().out)["published"] == 1

    hub.before_commit = publish_rival
    assert cli_main(["corpus", "dataset", "publish", "--store", str(first.root)]) == 0
    assert json.loads(capsys.readouterr().out)["published"] == 1
    assert len(hub.commits) == 2
    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", "--revision", hub.revision,
                     "--output", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 2
    records = LocalRecordStore(destination).read_records()
    assert {record["run_id"] for record in records["runs"]} == {"first", "rival"}
    for store in (first, rival):
        assert cli_main(["corpus", "dataset", "status", "--store", str(store.root)]) == 0
        assert json.loads(capsys.readouterr().out)["published"] == 1


@pytest.mark.parametrize("fault", ["manifest", "shard"])
def test_cli_rejects_corrupt_pinned_snapshot_before_exposing_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fault: str,
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
    if fault == "manifest":
        manifest = json.loads(tree["manifest.json"])
        manifest["schema_version"] = "unsupported"
        tree["manifest.json"] = json.dumps(manifest).encode()
    else:
        path = next(path for path in tree if path.endswith(".jsonl"))
        tree[path] = b"SECRET_CORRUPT_SNAPSHOT"
    destination = tmp_path / "download"
    assert cli_main(["corpus", "dataset", "download", "--revision", revision,
                     "--output", str(destination)]) == 1
    output = capsys.readouterr()
    assert "Data Collection" in output.out
    assert "SECRET_CORRUPT_SNAPSHOT" not in output.out + output.err
    assert not destination.exists()
