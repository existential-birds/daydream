"""Actual CLI recovery from total VM loss, with only external Hub/license fakes."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from daydream import cli
from daydream.archive import hydrate, license_enrich
from daydream.archive.hydrate_client import FakeHub
from daydream.archive.index import append_label_observation, label_observation_history, query_runs, upsert_run
from daydream.archive.manifest import Manifest
from daydream.training.adjudication import cli as adjudication_cli
from daydream.training.adjudication.final_bundle import final_snapshot_id
from daydream.training.adjudication.publish import publish_final_annotation_bundle
from tests.fixtures.training.build_hub_snapshot import PublicationHubs, build_publication_hubs


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as result:
        cli.main(["corpus", *argv])
    captured = capsys.readouterr()
    assert result.value.code == 0, f"{argv!r}\n{captured.out}\n{captured.err}"
    return captured.out + captured.err


def _hydrate_vm(root: Path, hubs: PublicationHubs, capsys: pytest.CaptureFixture[str]) -> tuple[Path, str]:
    stage = root / "hydrated"
    _run_cli([
        "hydrate-hub", "--source-repo", hubs.source.repo_id,
        "--source-revision", hubs.source_revision,
        "--destination-repo", hubs.source.repo_id,
        "--stage-dir", str(stage), "--license-policy", str(hubs.policy_path),
    ], capsys)
    manifests = list((stage / "curated").glob("*/curation-manifest.json"))
    assert len(manifests) == 1
    curation_id = str(json.loads(manifests[0].read_text())["curation_id"])
    return stage, curation_id


def _materialize_vm(
    root: Path, stage: Path, pin: dict[str, Any], capsys: pytest.CaptureFixture[str],
) -> Path:
    materialized = root / "materialized"
    arguments = [
        "adjudicate", "materialize", "--index-root", str(stage), "--out-dir", str(materialized),
    ]
    for name in (
        "curation_id", "sanitized_hub_commit", "source_hub_commit",
        "archive_index_digest", "evidence_observed_at", "as_of",
    ):
        if pin.get(name):
            arguments.extend(["--" + name.replace("_", "-"), str(pin[name])])
    _run_cli(arguments, capsys)
    return materialized


def _import_backup(
    root: Path, stage: Path, state: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Seed a real external-backup producer index, then import through the CLI.
    # The special history row exists only in SQLite, never state observations.
    backup = root / "local-backup"
    row = query_runs(stage, "session_id = ?", ("sess-a",))[0]
    upsert_run(backup, Manifest(**{field.name: row[field.name] for field in fields(Manifest) if field.name in row}))
    shutil.copytree(stage / "runs" / "sess-a", backup / "runs" / "sess-a")
    assert append_label_observation(
        backup, "sess-a", labels=["accepted"], pr_state=None,
        labeler_version="980-rubric-r2", evidence_sha=str(row["head_sha"]),
        reward_version="vm-loss-sqlite-canary",
        source="human", observed_at="2026-06-01T00:00:00+00:00",
    )
    _run_cli([
        "adjudicate", "import-local-observations", "--archive-root", str(backup),
        "--index-root", str(stage), "--archive-dir", str(state), "--state-dir", str(state),
    ], capsys)
    assert any(row["reward_version"] == "vm-loss-sqlite-canary" for row in label_observation_history(state, "sess-a"))
    assert "vm-loss-sqlite-canary" not in (state / "observations.jsonl").read_text()


def _publish_first_vm(
    root: Path, hubs: PublicationHubs, capsys: pytest.CaptureFixture[str], *, import_history: bool,
) -> str:
    """Return only an operator-known identity; every other input dies with VM 1."""
    root.mkdir()
    stage, curation_id = _hydrate_vm(root, hubs, capsys)
    pin = {
        "curation_id": curation_id,
        "sanitized_hub_commit": hubs.source_revision,
        "source_hub_commit": hubs.source_revision,
        "archive_index_digest": hashlib.sha256((stage / "index.db").read_bytes()).hexdigest(),
        "evidence_observed_at": "2026-06-01T00:00:00+00:00",
    }
    materialized = _materialize_vm(root, stage, pin, capsys)
    state = root / "state"
    _run_cli(["adjudicate", "build", "--index-root", str(materialized), "--state-dir", str(state)], capsys)
    _run_cli([
        "adjudicate", "label", "--state-dir", str(state), "--batch", "1",
        "--disposition", "accepted", "--rationale", "verified-against-diff-context",
        "--labeler", "alice", "--valid-at", "2026-06-01T00:00:00+00:00",
    ], capsys)
    _run_cli([
        "adjudicate", "export", "--index-root", str(materialized),
        "--state-dir", str(state), "--dry-run",
    ], capsys)
    if import_history:
        _import_backup(root, stage, state, capsys)
    else:
        assert not (state / "index.db").exists()
    _run_cli([
        "adjudicate", "publish-state", "--state-dir", str(state),
        "--manifest", str(materialized / "preview-manifest.json"),
        "--hub-repo", hubs.annotations.repo_id,
    ], capsys)
    pointer = f"annotations/{curation_id}/checkpoints/batch-latest.json"
    commit = hubs.annotations.commit_order[-1]
    assert pointer in commit["contains"]
    assert any(path.endswith("/index.db") for path in commit["contains"]) is import_history
    assert any(path.endswith("/observations.jsonl") for path in commit["contains"])
    return curation_id


@pytest.mark.parametrize("import_history", [False, True], ids=["ordinary-no-backup", "sqlite-backup"])
def test_ordinary_checkpoint_survives_total_vm_loss_and_final_cli_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    import_history: bool,
) -> None:
    hubs = build_publication_hubs()

    def external_client(repo_id: str, **_kwargs: Any) -> FakeHub:
        return {hubs.source.repo_id: hubs.source, hubs.annotations.repo_id: hubs.annotations}[repo_id]

    class ExternalLicenseResolver:
        def resolve(self, repo_slug: str, repo_commit: str | None) -> license_enrich.EnrichedEvidence:
            commit = repo_commit or "a" * 40
            return license_enrich.EnrichedEvidence("MIT", f"github:{repo_slug}@{commit}", commit)

    monkeypatch.setattr(hydrate, "_make_client", external_client)
    monkeypatch.setattr(adjudication_cli, "_make_client", external_client)
    monkeypatch.setattr(license_enrich, "_make_license_resolver", ExternalLicenseResolver)
    monkeypatch.setenv("HF_TOKEN", "offline-fixture-token")
    monkeypatch.setenv("GITHUB_TOKEN", "offline-fixture-token")
    vm1 = tmp_path / "first disposable VM"
    curation_id = _publish_first_vm(vm1, hubs, capsys, import_history=import_history)
    shutil.rmtree(vm1)
    assert not vm1.exists()

    vm2 = tmp_path / "second empty VM"
    vm2.mkdir()
    state = vm2 / "state"
    checkpoint_revision = hubs.annotations.repo_info("main").sha
    hubs.annotations.downloaded_revision_log.clear()
    _run_cli([
        "adjudicate", "resume-state", "--curation-id", curation_id,
        "--destination", str(state), "--hub-repo", hubs.annotations.repo_id,
    ], capsys)
    assert hubs.annotations.downloaded_revision_log
    assert {revision for _, revision in hubs.annotations.downloaded_revision_log} == {checkpoint_revision}
    if import_history:
        assert any(
            row["reward_version"] == "vm-loss-sqlite-canary"
            for row in label_observation_history(state, "sess-a")
        )
    else:
        assert not (state / "index.db").exists()
    pin = json.loads((state / "preview-manifest.json").read_text())
    stage, rebuilt_curation = _hydrate_vm(vm2, hubs, capsys)
    assert rebuilt_curation == curation_id
    materialized = _materialize_vm(vm2, stage, pin, capsys)
    assert json.loads((materialized / "preview-manifest.json").read_text()) == pin
    # Literal runbook branch: restored history owns the archive when present;
    # otherwise the exact rehydrated source index supplies the run identities.
    archive = state if (state / "index.db").is_file() else stage
    assert (archive == state) is import_history
    _run_cli([
        "adjudicate", "harvest-snapshot", "--index-root", str(stage),
        "--materialize-dir", str(materialized), "--archive-dir", str(archive), "--state-dir", str(state),
    ], capsys)
    final_arguments = [
        "adjudicate", "publish-final", "--index-root", str(stage),
        "--materialize-dir", str(materialized), "--archive-dir", str(archive), "--state-dir", str(state),
        "--curation-bundle-dir", str(stage / "curated" / curation_id),
        "--hub-repo", hubs.annotations.repo_id,
    ]
    commit_count = len(hubs.annotations.commit_order)
    dry_output = _run_cli([*final_arguments, "--dry-run"], capsys)
    bundle = materialized / "final-bundle"
    expected_final_id, _digests = final_snapshot_id(bundle)
    assert expected_final_id in dry_output
    assert len(hubs.annotations.commit_order) == commit_count
    # These are genuine CLI-produced semantic files. Re-hashing a changed
    # bundle must not turn contradictory report/lineage claims into valid data.
    for filename in ("coverage-report.json", "lineage.json"):
        path = bundle / filename
        original = path.read_bytes()
        changed = json.loads(original)
        if filename == "coverage-report.json":
            changed["outcome_coverage"] = {"adjudicated": 0, "total": 100}
            changed["admission_gate"]["passes_80pct"] = True
        else:
            changed["snapshot_id"] = "f" * 64
        path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        try:
            assert final_snapshot_id(bundle)[0] != expected_final_id
            with pytest.raises(ValueError, match=filename.replace(".", r"\.")):
                publish_final_annotation_bundle(hubs.annotations, bundle)
            assert len(hubs.annotations.commit_order) == commit_count
        finally:
            path.write_bytes(original)
    published_output = _run_cli(final_arguments, capsys)
    success_commit = hubs.annotations.commit_order[-1]
    revision = str(success_commit["sha"])
    assert len(success_commit["contains"]) == 1
    success_path = str(success_commit["contains"][0])
    assert success_path.endswith("/_SUCCESS")
    marker = json.loads(hubs.annotations.download_file(success_path, revision))
    final_id = str(marker["final_snapshot_id"])
    assert final_id == expected_final_id
    assert revision in published_output
    assert final_id in published_output
    downloaded = vm2 / "clean final download"
    _run_cli([
        "adjudicate", "download-final", "--curation-id", curation_id,
        "--snapshot-id", final_id, "--revision", revision, "--destination", str(downloaded),
        "--hub-repo", hubs.annotations.repo_id,
    ], capsys)
    assert {path.name for path in downloaded.iterdir()} == {
        "annotations.jsonl", "sessions.jsonl", "label-observations.jsonl", "coverage-report.json",
        "lineage.json", "preview-manifest.json", "policy-binding.json", "publication-manifest.json",
        "SHA256SUMS", "_SUCCESS",
    }
    for line in (downloaded / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((downloaded / name).read_bytes()).hexdigest() == expected
    assert json.loads((downloaded / "_SUCCESS").read_text()) == marker
    history = [json.loads(line) for line in (downloaded / "label-observations.jsonl").read_text().splitlines()]
    assert any(row["reward_version"] == "vm-loss-sqlite-canary" for row in history) is import_history
    assert not vm1.exists()
