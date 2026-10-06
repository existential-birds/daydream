"""Real capture, CLI annotation, canonical HF publication, and offline projection."""
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import dataset_hub, git_ops
from daydream.dataset import LocalRecordStore
from daydream.pr_review import DAYDREAM_FOOTER, finding_marker
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.training.license_evidence import EnrichedEvidence, GithubLicenseResolver
from tests.harness.dataset import read_records
from tests.harness.dataset_hub import FakeDatasetHub
from tests.harness.git_helpers import git
from tests.harness.record_projection import policy_file
from tests.harness.scripts import cli_main
from tests.test_dataset_hub_run import config as config


def test_capture_publish_download_annotate_republish_and_build_offline(
    config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Path(config.target or "")
    git(target, "remote", "add", "origin", "https://github.com/owner/repo")
    hub = FakeDatasetHub()
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", lambda: hub)
    assert anyio.run(run, replace(config, trajectory_hub_repo=None, dataset_capture=True)) == 0
    assert config.dataset_store_path is not None
    store = LocalRecordStore(config.dataset_store_path)
    captured = read_records(store).runs[0]
    original = json.dumps(captured, sort_keys=True)
    items = captured["findings"]["value"]["items"]
    assert items
    repo = "test-user/private-trajectories"
    assert cli_main(["corpus", "dataset", "publish", "--store", str(store.root),
                     "--trajectory-hub-repo", repo]) == 0
    first_revision = hub.revision
    downloaded_path = tmp_path / "downloaded"
    assert cli_main(["corpus", "dataset", "download", "--trajectory-hub-repo", repo,
                     "--revision", first_revision, "--output", str(downloaded_path)]) == 0
    downloaded = LocalRecordStore(downloaded_path)
    initial = downloaded.select_snapshot(observed_before="2100-01-01T00:00:00Z")

    comments = []
    for index, item in enumerate(items, start=1):
        comments += [{"id": index * 2, "user": {"login": "daydream-runner"},
                      "body": f"finding\n{finding_marker(item['fingerprint'])}\n{DAYDREAM_FOOTER}"},
                     {"id": index * 2 + 1, "in_reply_to_id": index * 2, "user": {"login": "reviewer"},
                      "author_association": "OWNER", "body": "Needs investigation." if index == 1 else "Applied.",
                      "created_at": "2026-10-06T10:00:00Z"}]

    def github(repo_path: Path, endpoint: str, **kwargs: Any) -> Any:
        if endpoint.endswith("/pulls") and "/commits/" in endpoint:
            return [{"number": 7, "head": {"sha": captured["original_task"]["value"]["analyzed_revision"]["head_sha"],
                                          "repo": {"full_name": "owner/repo"}}}]
        if endpoint.endswith("/pulls/7"):
            return {"merged": True, "merged_at": "2026-10-06T11:00:00Z", "user": {"login": "author"}}
        if endpoint.endswith("/reviews"):
            return [{"user": {"login": "reviewer"}}]
        if endpoint.endswith("/comments"):
            return comments
        raise AssertionError(endpoint)

    def unavailable_clone(*args: Any, **kwargs: Any) -> None:
        raise git_ops.GitError("offline Git remote")

    monkeypatch.setattr(git_ops, "gh_api", github)
    monkeypatch.setattr(git_ops, "clone_with_token", unavailable_clone)
    monkeypatch.setattr(GithubLicenseResolver, "resolve",
                        lambda self, slug, *, repo_commit:
                        EnrichedEvidence("MIT", f"github:{slug}@{repo_commit}", repo_commit))
    assert cli_main(["corpus", "harvest", "--store", str(downloaded.root),
                     "--snapshot-id", initial["snapshot_id"], "--cache-dir", str(tmp_path / "cache"),
                     "--gh-spacing-sec", "0"]) == 0
    enriched = downloaded.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    state = tmp_path / "queue"
    assert cli_main(["corpus", "adjudicate", "build", "--store", str(downloaded.root),
                     "--snapshot-id", enriched["snapshot_id"], "--state-dir", str(state)]) == 0
    assert cli_main(["corpus", "adjudicate", "label", "--state-dir", str(state), "--batch", str(len(items)),
                     "--disposition", "accepted", "--rationale", "Human confirmed the frozen evidence",
                     "--labeler", "alice"]) == 0
    assert cli_main(["corpus", "dataset", "publish", "--store", str(downloaded.root),
                     "--trajectory-hub-repo", repo]) == 0
    final_revision = hub.revision
    assert final_revision != first_revision
    final_path = tmp_path / "final-records"
    assert cli_main(["corpus", "dataset", "download", "--trajectory-hub-repo", repo,
                     "--revision", final_revision, "--output", str(final_path)]) == 0
    final = LocalRecordStore(final_path)
    frozen = final.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    records = final.read_snapshot(frozen["snapshot_id"])
    assert json.dumps(records.runs[0], sort_keys=True) == original
    assert any(o["payload"]["type"] == "harvest-annotation" for o in records.observations)
    assert any(o["role"] == "rater" and o["source"] == "adjudication" for o in records.observations)
    assert downloaded.read_snapshot(initial["snapshot_id"]).observations == ()

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("projection attempted network access")

    monkeypatch.setattr(git_ops, "gh_api", no_network)
    monkeypatch.setattr(GithubLicenseResolver, "resolve", no_network)
    monkeypatch.setattr(dataset_hub, "HfDatasetHub", no_network)
    for name in ("out-a", "out-b"):
        assert cli_main(["corpus", "build", "--store", str(final.root), "--snapshot-id", frozen["snapshot_id"],
                         "--license-policy", str(policy_file(tmp_path)),
                         "--out", str(tmp_path / name / "corpus.jsonl")]) == 0
    first = tmp_path / "out-a"
    second = tmp_path / "out-b"
    assert (first / "corpus.jsonl").read_bytes()
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    source = final.download_source()
    assert source is not None and source["revision"] == final_revision
