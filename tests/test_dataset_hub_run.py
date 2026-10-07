"""Real review lifecycle retains new JSONL evidence across offline HF publication."""
from dataclasses import replace
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.dataset import read_records
from tests.harness.stub_backend import install_stub_backend, silence


@pytest.fixture
def config(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target, pin_skill_availability=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    return RunConfig(target=str(multi_stack_target), output_mode="review", archive=False, run_eval=False,
        dataset_store_path=tmp_path / "records", trajectory_hub_repo="test-user/private-trajectories",
        shallow_fanout_threshold=0, cleanup=False)



async def test_review_publishes_raw_records_and_other_machine_reads_pinned_commit(
    config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream import dataset_hub
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", lambda: hub)
    assert await run(config) == 0
    assert config.dataset_store_path is not None and config.trajectory_hub_repo is not None
    local = LocalRecordStore(config.dataset_store_path)
    records = read_records(local)
    assert len(hub.commits) == 1
    status = dataset_hub.DatasetUploader(local, config.trajectory_hub_repo, backend=hub).status()
    assert (status.queued, status.published, status.failed) == (0, 1, 0)
    downloaded = dataset_hub.download_snapshot(
        config.trajectory_hub_repo, hub.revision, tmp_path / "other-machine", backend=hub)
    assert read_records(downloaded).runs == records.runs
    assert records.runs[0]["run_id"] not in {path.name for path in downloaded.root.iterdir()}


@pytest.mark.parametrize("failure", ["offline", "public", "secret"])
async def test_review_stays_successful_with_durable_failed_uploads(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    from daydream import dataset_hub
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub(private=failure != "public")
    hub.fail_before = failure == "offline"
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", lambda: hub)
    credential = "sk-offlineplaceholder0123456789"
    if failure == "secret":
        Path(config.target or "").joinpath("api.py").write_text(f"# operator credential: {credential}\n")
    assert await run(config) == 0
    assert config.dataset_store_path is not None and config.trajectory_hub_repo is not None
    store = LocalRecordStore(config.dataset_store_path)
    captured = read_records(store)
    assert len(captured.runs) == 1 and captured.runs[0]["outcome"] == "success"
    if failure == "offline":
        assert captured.runs[0]["original_task"]["status"] == "available"
        assert captured.runs[0]["trajectories"]["status"] == "available"
    uploader = dataset_hub.DatasetUploader(store, config.trajectory_hub_repo, backend=hub)
    status = uploader.status()
    assert status.queued == 1 and status.published == 0 and status.failed == 1
    assert credential not in str(status.error)
    assert hub.commits == []
    if failure == "secret":
        assert credential in captured.runs[0]["original_task"]["value"]["diff"]
    else:
        hub.fail_before = False
        hub.private = True
        retried = uploader.upload()
        assert (retried.queued, retried.published, retried.failed) == (0, 1, 0)
        assert read_records(store).runs == captured.runs


async def test_credentials_without_destination_do_not_upload(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream import dataset_hub
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setenv("HF_TOKEN", "hf_offline_fixture_token")
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", lambda: hub)
    assert await run(replace(config, trajectory_hub_repo=None, dataset_capture=True)) == 0
    assert config.dataset_store_path is not None
    assert len(read_records(LocalRecordStore(config.dataset_store_path)).runs) == 1
    assert hub.commits == []


async def test_explicit_capture_disable_overrides_hub_destination(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream import dataset_hub
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", lambda: hub)
    assert await run(replace(config, dataset_capture=True, dataset_capture_disabled=True)) == 0
    assert config.dataset_store_path is not None
    assert not config.dataset_store_path.exists()
    assert hub.commits == []


@pytest.mark.parametrize(("cli_repo", "env_repo", "expected_repo"), [
    ("operator/cli", "operator/env", "operator/cli"),
    ("operator/cli", None, "operator/cli"),
    ("", "operator/env", "operator/env"),
    (None, "operator/env", "operator/env"),
    (None, "", None),
    (None, None, None),
])
async def test_operator_destination_precedence_controls_real_review_publication(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch,
    cli_repo: str | None, env_repo: str | None, expected_repo: str | None,
) -> None:
    from daydream import dataset_hub
    from tests.harness.dataset_hub import FakeDatasetHub

    class DestinationHub(FakeDatasetHub):
        destinations: list[str] = []

        def private_revision(self, repo_id: str) -> str:
            self.destinations.append(repo_id)
            return super().private_revision(repo_id)

    hub = DestinationHub()
    monkeypatch.setenv("HF_TOKEN", "hf_offline_fixture_token")
    if env_repo is not None:
        monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", env_repo)
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", lambda: hub)
    assert await run(replace(config, trajectory_hub_repo=cli_repo)) == 0
    assert config.dataset_store_path is not None
    if expected_repo is None:
        assert not config.dataset_store_path.exists()
        assert hub.commits == [] and hub.destinations == []
    else:
        assert len(hub.commits) == 1 and set(hub.destinations) == {expected_repo}
        assert len(read_records(LocalRecordStore(config.dataset_store_path)).runs) == 1


@pytest.mark.parametrize("filename,body", [
    ("pyproject.toml", '[tool.daydream]\ntrajectory_hub_repo = "evil/repo"\n'),
    (".daydream.toml", 'trajectory_hub_repo = "evil/repo"\n'),
])
async def test_hostile_checkout_config_cannot_enable_record_uploads(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, filename: str, body: str,
) -> None:
    from daydream import dataset_hub
    from daydream.config_file import load_file_config
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    target = Path(config.target or "")
    (target / filename).write_text(body, encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_offline_fixture_token")
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", lambda: hub)
    assert await run(replace(config, trajectory_hub_repo=None, file_config=load_file_config(target))) == 0
    assert config.dataset_store_path is not None
    assert not config.dataset_store_path.exists()
    assert hub.commits == []
    assert (target / ".review-output.md").is_file()
