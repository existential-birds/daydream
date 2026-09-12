import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.hydrate import HubDownloadError, HydrationError, PublicDestinationError
from daydream.training.adjudication.publish import (
    download_final_annotation_bundle,
    publish_annotation_state,
    publish_final_annotation_bundle,
    resume_annotation_state,
)
from tests.fixtures.training.build_hub_snapshot import AnnotationsHub

# M6: production manifests always pin index_revision (materialize writes it),
# so the Hub-verified 40-hex branch — not the synthetic digest fallback — is
# the path that must be exercised.
INDEX_REVISION = "a" * 40

# P14 immutable checkpoint protocol -------------------------------------------------

_CID = "cur-1"
_SID = "e" * 64
_STABLE_POINTER = f"annotations/{_CID}/checkpoints/batch-latest.json"
_STATE_NAMES = ("queue.json", "observations.jsonl", "preview-ledger.json", "preview-manifest.json", "index.db")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _observation(
    *, record_id: str, observed_at: str, disposition: str, rationale: str
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "observed_at": observed_at,
        "disposition": disposition,
        "rationale": rationale,
        "role": "rater",
    }


def _manifest_data(*, snapshot_id: str = _SID, note: str = "base") -> dict[str, Any]:
    return {
        "curation_id": _CID,
        "snapshot_id": snapshot_id,
        "index_revision": INDEX_REVISION,
        "note": note,
    }


def _state_v2(
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    state = tmp_path / "state"
    state.mkdir(parents=True)
    (state / "queue.json").write_bytes(_canonical_bytes([{"record_id": "r1", "state": "base"}]))
    observations = rows or [
        _observation(
            record_id="r1",
            observed_at="2026-01-01T00:00:00Z",
            disposition="accepted",
            rationale="first",
        )
    ]
    (state / "observations.jsonl").write_bytes(b"".join(_canonical_bytes(row) for row in observations))
    (state / "preview-ledger.json").write_bytes(_canonical_bytes({"cursor": "base"}))
    (state / "index.db").write_bytes(b"SQLite format 3\x00base-index")
    return state, manifest or _manifest_data()


def _checkpoint_files(payloads: dict[str, bytes], *, manifest: dict[str, Any]) -> tuple[str, dict[str, bytes]]:
    import hashlib

    entries = [
        {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(payloads.items())
    ]
    identity = _canonical_bytes({"files": entries})
    batch_id = hashlib.sha256(identity).hexdigest()
    batch_prefix = f"annotations/{_CID}/checkpoints/batches/{batch_id}/"
    observed = [
        json.loads(line)
        for line in payloads["observations.jsonl"].decode().splitlines()
        if line.strip()
    ]
    pointer = {
        "schema_version": "annotation-checkpoint/v1",
        "curation_id": _CID,
        "snapshot_id": manifest["snapshot_id"],
        "batch_id": batch_id,
        "batch_prefix": batch_prefix,
        "files": entries,
        "observation_count": len(observed),
        "latest_observed_at": max((row.get("observed_at") for row in observed), default=None),
    }
    remote = {f"{batch_prefix}{name}": data for name, data in payloads.items()}
    remote[_STABLE_POINTER] = _canonical_bytes(pointer)
    return batch_id, remote


def _published_payloads(hub: AnnotationsHub, revision: str) -> dict[str, bytes]:
    tree = hub.revision_files(revision)
    pointer = json.loads(tree[_STABLE_POINTER])
    return {
        entry["path"]: tree[f"{pointer['batch_prefix']}{entry['path']}"]
        for entry in pointer["files"]
    }


def test_mandatory_checkpoint_commits_immutable_batch_and_stable_pointer(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)

    result = publish_annotation_state(hub, state, manifest=manifest)

    assert len(hub.commit_order) == 1
    commit = hub.commit_order[0]
    assert result["checkpoint_revision"] == commit["sha"]
    assert result["batch_prefix"] == f"annotations/{_CID}/checkpoints/batches/{result['batch_id']}/"
    assert _STABLE_POINTER in commit["contains"]
    assert f"{result['batch_prefix']}observations.jsonl" in commit["contains"]
    assert hub.info_revision_log == ["main"]
    assert hub.listed_revision_log == [hub.atomic_attempt_log[0]["parent_commit"]]
    assert hub.atomic_attempt_log[0]["branch"] == "main"


def test_changed_same_time_observation_survives_and_exact_row_dedupes(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    first = _observation(
        record_id="r1", observed_at="2026-01-01T00:00:00Z", disposition="accepted", rationale="first"
    )
    changed = _observation(
        record_id="r1", observed_at="2026-01-01T00:00:00Z", disposition="rejected", rationale="changed"
    )
    second = _observation(
        record_id="r2", observed_at="2026-01-01T00:01:00Z", disposition="accepted", rationale="second"
    )
    state, manifest = _state_v2(tmp_path, rows=[first])
    publish_annotation_state(hub, state, manifest=manifest)
    (state / "observations.jsonl").write_bytes(
        b"".join(_canonical_bytes(row) for row in [first, first, changed, second])
    )

    result = publish_annotation_state(hub, state, manifest=manifest)
    rows = [
        json.loads(line)
        for line in _published_payloads(hub, result["checkpoint_revision"])[
            "observations.jsonl"
        ].splitlines()
    ]

    assert [(row["record_id"], row["disposition"]) for row in rows] == [
        ("r1", "accepted"),
        ("r1", "rejected"),
        ("r2", "accepted"),
    ]
    commits = len(hub.commit_order)
    repeated = publish_annotation_state(hub, state, manifest=manifest)
    assert {key: value for key, value in repeated.items() if key != "uploaded"} == {
        key: value for key, value in result.items() if key != "uploaded"
    }
    assert repeated["uploaded"] == []
    assert len(hub.commit_order) == commits


@pytest.mark.parametrize("name", ["queue.json", "preview-ledger.json", "preview-manifest.json", "index.db"])
@pytest.mark.parametrize("mode", ["identical", "local-only", "remote-only", "divergent"])
def test_three_way_checkpoint_non_observation_files(
    name: str, mode: str, tmp_path: Path
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    baseline = publish_annotation_state(hub, state, manifest=manifest)
    base_payloads = _published_payloads(hub, baseline["checkpoint_revision"])
    local_payloads = dict(base_payloads)
    remote_payloads = dict(base_payloads)
    if name == "preview-manifest.json":
        local_value = _canonical_bytes(_manifest_data(note="local"))
        remote_value = _canonical_bytes(_manifest_data(note="remote"))
    elif name == "index.db":
        local_value, remote_value = b"SQLite format 3\x00local", b"SQLite format 3\x00remote"
    else:
        local_value = _canonical_bytes({"side": "local"})
        remote_value = _canonical_bytes({"side": "remote"})
    if mode in {"local-only", "divergent"}:
        local_payloads[name] = local_value
        if name == "preview-manifest.json":
            manifest = _manifest_data(note="local")
        else:
            (state / name).write_bytes(local_value)
    if mode in {"remote-only", "divergent"}:
        remote_payloads[name] = remote_value
        rival_manifest = (
            _manifest_data(note="remote")
            if name == "preview-manifest.json"
            else manifest
        )
        _, rival = _checkpoint_files(remote_payloads, manifest=rival_manifest)
        # Force a local commit attempt so the fixture can introduce the rival
        # after the publication's stable base read.
        local_observation = _observation(
            record_id="local-race",
            observed_at="2026-01-01T00:02:00Z",
            disposition="accepted",
            rationale="force CAS",
        )
        with (state / "observations.jsonl").open("ab") as handle:
            handle.write(_canonical_bytes(local_observation))
    else:
        rival = {"unrelated/race.txt": mode.encode()}
    hub.queue_concurrent_commit("batch", rival)

    before = len(hub.commit_order)
    if mode == "divergent":
        with pytest.raises(HydrationError, match="concurrent state conflict"):
            publish_annotation_state(hub, state, manifest=manifest)
        assert len(hub.commit_order) == before + 1
        return

    result = publish_annotation_state(hub, state, manifest=manifest)
    resolved = _published_payloads(hub, result["checkpoint_revision"])
    expected = local_value if mode == "local-only" else remote_value if mode == "remote-only" else base_payloads[name]
    assert resolved[name] == expected


def test_superseded_manifest_identity_fails_instead_of_repointing(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    baseline = publish_annotation_state(hub, state, manifest=manifest)
    remote = _published_payloads(hub, baseline["checkpoint_revision"])
    superseded = _manifest_data(snapshot_id="f" * 64)
    remote["preview-manifest.json"] = _canonical_bytes(superseded)
    _, rival = _checkpoint_files(remote, manifest=superseded)
    hub.queue_concurrent_commit("batch", rival)
    with (state / "observations.jsonl").open("ab") as handle:
        handle.write(
            _canonical_bytes(
                _observation(
                    record_id="local-race",
                    observed_at="2026-01-01T00:02:00Z",
                    disposition="accepted",
                    rationale="force CAS",
                )
            )
        )

    with pytest.raises(HydrationError, match="superseded"):
        publish_annotation_state(hub, state, manifest=manifest)

    assert json.loads(hub.revision_files(hub.repo_info().sha)[_STABLE_POINTER])["snapshot_id"] == "f" * 64


def test_concurrent_checkpoint_pointer_deletion_is_not_recreated(tmp_path: Path) -> None:
    class PointerDeletingHub(AnnotationsHub):
        delete_on_batch = False

        def commit_files_atomic(
            self,
            mapping: dict[str | Path, Path],
            commit_message: str,
            *,
            parent_commit: str,
            branch: str,
        ) -> str:
            if self.delete_on_batch and any(str(path).endswith("/batch-latest.json") for path in mapping):
                self.delete_on_batch = False
                tree = self.revision_files(self.repo_info("main").sha)
                tree.pop(_STABLE_POINTER)
                self.files = tree
                revision = hashlib.sha256(
                    _canonical_bytes({path: hashlib.sha256(data).hexdigest() for path, data in sorted(tree.items())})
                ).hexdigest()[:40]
                self.commit_revision(revision, ref="main")
            return super().commit_files_atomic(
                mapping,
                commit_message,
                parent_commit=parent_commit,
                branch=branch,
            )

    hub = PointerDeletingHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    publish_annotation_state(hub, state, manifest=manifest)
    with (state / "observations.jsonl").open("ab") as handle:
        handle.write(
            _canonical_bytes(
                _observation(
                    record_id="local-race",
                    observed_at="2026-01-01T00:02:00Z",
                    disposition="accepted",
                    rationale="force CAS",
                )
            )
        )
    hub.delete_on_batch = True

    with pytest.raises(HydrationError, match="checkpoint pointer disappeared"):
        publish_annotation_state(hub, state, manifest=manifest)

    assert _STABLE_POINTER not in hub.revision_files(hub.repo_info("main").sha)


def test_resume_stable_bootstrap_is_pinned_and_byte_identical(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    published = publish_annotation_state(hub, state, manifest=manifest)
    hub.info_revision_log.clear()
    hub.listed_revision_log.clear()
    hub.downloaded_revision_log.clear()
    destination = tmp_path / "fresh"

    restored = resume_annotation_state(hub, curation_id=_CID, destination=destination)

    assert restored["checkpoint_revision"] == published["checkpoint_revision"]
    assert restored["snapshot_id"] == _SID
    assert hub.info_revision_log == ["main"]
    assert hub.listed_revision_log == [published["checkpoint_revision"]]
    assert all(revision == published["checkpoint_revision"] for _, revision in hub.downloaded_revision_log)
    for name, data in _published_payloads(hub, published["checkpoint_revision"]).items():
        assert (destination / name).read_bytes() == data


def test_resume_missing_checkpoint_is_an_error_not_empty_state(tmp_path: Path) -> None:
    destination = tmp_path / "fresh"
    with pytest.raises(HydrationError, match="no published checkpoint"):
        resume_annotation_state(
            AnnotationsHub(repo_id="org/private-annotations"),
            curation_id=_CID,
            destination=destination,
        )
    assert not destination.exists()


def test_resume_auth_error_is_not_treated_as_missing(tmp_path: Path) -> None:
    class UnavailableHub(AnnotationsHub):
        def list_repo_files(self, revision: str | None = None) -> list[str]:
            raise HubDownloadError("auth failed https://user:hf_secret_token@huggingface.co/private")

    destination = tmp_path / "fresh"
    with pytest.raises(HydrationError) as excinfo:
        resume_annotation_state(
            UnavailableHub(repo_id="org/private-annotations"),
            curation_id=_CID,
            destination=destination,
        )
    assert "hf_secret_token" not in str(excinfo.value)
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_resume_requires_fresh_destination(kind: str, tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    destination = tmp_path / "fresh"
    if kind == "file":
        destination.write_text("existing")
    elif kind == "directory":
        destination.mkdir()
    else:
        target = tmp_path / "target"
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not exist"):
        resume_annotation_state(hub, curation_id=_CID, destination=destination)
    assert hub.info_revision_log == []


def test_publish_path_guard_rejects_secret_bearing_outward_symlink(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    secret = tmp_path / "outside-secret"
    secret.write_text("hf_abc123secret")
    (state / "queue.json").unlink()
    (state / "queue.json").symlink_to(secret)

    with pytest.raises(PublicDestinationError, match="symlink"):
        publish_annotation_state(hub, state, manifest=manifest)
    assert hub.commit_order == []


def test_publish_accepts_manifest_under_symlinked_ancestor(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    outside = tmp_path / "outside-manifest"
    outside.mkdir()
    (outside / "preview-manifest.json").write_bytes(_canonical_bytes(manifest))
    linked_parent = tmp_path / "linked-manifest"
    linked_parent.symlink_to(outside, target_is_directory=True)

    result = publish_annotation_state(
        hub,
        state,
        manifest=linked_parent / "preview-manifest.json",
    )

    assert result["checkpoint_revision"] == hub.repo_info().sha


def test_publish_accepts_real_state_root_under_symlinked_ancestor(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    state, manifest = _state_v2(alias)
    hub = AnnotationsHub(repo_id="org/private-annotations")

    result = publish_annotation_state(hub, state, manifest=manifest)

    assert result["checkpoint_revision"] == hub.repo_info().sha


def test_publish_still_refuses_a_declared_symlink_state_root(tmp_path: Path) -> None:
    actual, manifest = _state_v2(tmp_path)
    linked = tmp_path / "linked-state"
    linked.symlink_to(actual, target_is_directory=True)
    hub = AnnotationsHub(repo_id="org/private-annotations")

    with pytest.raises(PublicDestinationError, match="state_dir.*symlink"):
        publish_annotation_state(hub, linked, manifest=manifest)

    assert hub.info_revision_log == []


def test_resume_accepts_fresh_destination_under_symlinked_ancestor(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    publish_annotation_state(hub, state, manifest=manifest)
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    resume_annotation_state(hub, curation_id=_CID, destination=alias / "fresh")

    assert (actual / "fresh" / "queue.json").is_file()


@pytest.mark.parametrize("bad_name", ["../queue.json", "nested\\queue.json", "./queue.json", "queue.json/"])
def test_resume_path_guard_rejects_malformed_pointer_names(bad_name: str, tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    published = publish_annotation_state(hub, state, manifest=manifest)
    tree = hub.revision_files(published["checkpoint_revision"])
    pointer = json.loads(tree[_STABLE_POINTER])
    pointer["files"][0]["path"] = bad_name
    hub.seed_remote_files({_STABLE_POINTER: _canonical_bytes(pointer)})
    destination = tmp_path / "fresh"

    with pytest.raises(ValueError, match="path"):
        resume_annotation_state(hub, curation_id=_CID, destination=destination)
    assert not destination.exists()


def test_resume_failure_on_last_download_leaves_no_partial_install(tmp_path: Path) -> None:
    class LastDownloadFails(AnnotationsHub):
        fail_path: str = ""

        def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
            if path_in_repo == self.fail_path:
                raise HubDownloadError("last download failed")
            return super().download_file(path_in_repo, revision)

    hub = LastDownloadFails(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    published = publish_annotation_state(hub, state, manifest=manifest)
    pointer = json.loads(hub.revision_files(published["checkpoint_revision"])[_STABLE_POINTER])
    hub.fail_path = f"{pointer['batch_prefix']}{pointer['files'][-1]['path']}"
    destination = tmp_path / "fresh"

    with pytest.raises(HydrationError, match="last download failed"):
        resume_annotation_state(hub, curation_id=_CID, destination=destination)
    assert not destination.exists()
    assert list(tmp_path.glob(".fresh.*")) == []


def test_resume_parent_fsync_failure_removes_owned_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    publish_annotation_state(hub, state, manifest=manifest)
    destination = tmp_path / "fresh"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    real_fsync = os.fsync

    def fail_parent_after_rename(fd: int) -> None:
        stat = os.fstat(fd)
        if destination.exists() and (stat.st_dev, stat.st_ino) == parent_identity:
            raise OSError("parent fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_parent_after_rename)
    with pytest.raises(HydrationError, match="parent fsync failed"):
        resume_annotation_state(hub, curation_id=_CID, destination=destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".fresh.*")) == []


@pytest.mark.parametrize("name", ["queue.json", "observations.jsonl", "preview-ledger.json", "index.db"])
def test_publish_secret_in_every_state_payload_commits_nothing(name: str, tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    (state / name).write_bytes(b"hf_abc123secret")

    with pytest.raises(PublicDestinationError, match="credential-shaped"):
        publish_annotation_state(hub, state, manifest=manifest)
    assert hub.commit_order == []


def test_publish_secret_in_manifest_identity_commits_nothing(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    manifest["curation_id"] = "cur-hf_abc123secret"

    with pytest.raises(PublicDestinationError, match="credential-shaped"):
        publish_annotation_state(hub, state, manifest=manifest)
    assert hub.commit_order == []


def test_observation_only_concurrent_update_unions_remote_then_local_rows(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    baseline = publish_annotation_state(hub, state, manifest=manifest)
    remote = _published_payloads(hub, baseline["checkpoint_revision"])
    concurrent = _observation(
        record_id="remote",
        observed_at="2026-01-01T00:01:00Z",
        disposition="rejected",
        rationale="remote",
    )
    local = _observation(
        record_id="local",
        observed_at="2026-01-01T00:02:00Z",
        disposition="accepted",
        rationale="local",
    )
    remote["observations.jsonl"] += _canonical_bytes(concurrent)
    _, rival = _checkpoint_files(remote, manifest=manifest)
    hub.queue_concurrent_commit("batch", rival)
    with (state / "observations.jsonl").open("ab") as handle:
        handle.write(_canonical_bytes(local))

    result = publish_annotation_state(hub, state, manifest=manifest)
    rows = [
        json.loads(line)
        for line in _published_payloads(hub, result["checkpoint_revision"])[
            "observations.jsonl"
        ].splitlines()
    ]
    assert [row["record_id"] for row in rows] == ["r1", "remote", "local"]


def test_optional_index_absence_is_a_three_way_state(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    baseline = publish_annotation_state(hub, state, manifest=manifest)
    remote = _published_payloads(hub, baseline["checkpoint_revision"])
    remote["index.db"] = b"SQLite format 3\x00remote"
    _, rival = _checkpoint_files(remote, manifest=manifest)
    hub.queue_concurrent_commit("batch", rival)
    (state / "index.db").unlink()
    with (state / "observations.jsonl").open("ab") as handle:
        handle.write(
            _canonical_bytes(
                _observation(
                    record_id="force",
                    observed_at="2026-01-01T00:03:00Z",
                    disposition="accepted",
                    rationale="force CAS",
                )
            )
        )

    with pytest.raises(HydrationError, match="concurrent state conflict for index.db"):
        publish_annotation_state(hub, state, manifest=manifest)


def test_optional_index_local_removal_publishes_absence(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    publish_annotation_state(hub, state, manifest=manifest)
    (state / "index.db").unlink()

    result = publish_annotation_state(hub, state, manifest=manifest)

    assert "index.db" not in _published_payloads(hub, result["checkpoint_revision"])


def test_resume_expected_snapshot_mismatch_leaves_destination_absent(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    published = publish_annotation_state(hub, state, manifest=manifest)
    destination = tmp_path / "fresh"

    with pytest.raises(HydrationError, match="expected snapshot"):
        resume_annotation_state(
            hub,
            curation_id=_CID,
            expected_snapshot_id="f" * 64,
            revision=published["checkpoint_revision"],
            destination=destination,
        )
    assert not destination.exists()


@pytest.mark.parametrize("corruption", ["json", "digest", "duplicate"])
def test_resume_refuses_corrupt_checkpoint_without_partial_install(
    corruption: str, tmp_path: Path
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    published = publish_annotation_state(hub, state, manifest=manifest)
    tree = hub.revision_files(published["checkpoint_revision"])
    pointer = json.loads(tree[_STABLE_POINTER])
    if corruption == "json":
        pointer_bytes = b"not-json\n"
    elif corruption == "digest":
        pointer["files"][0]["sha256"] = "0" * 64
        pointer_bytes = _canonical_bytes(pointer)
    else:
        pointer["files"].append(dict(pointer["files"][0]))
        pointer_bytes = _canonical_bytes(pointer)
    bad_revision = hub.seed_remote_files({_STABLE_POINTER: pointer_bytes})
    destination = tmp_path / "fresh"

    with pytest.raises((ValueError, HydrationError)):
        resume_annotation_state(
            hub,
            curation_id=_CID,
            revision=bad_revision,
            destination=destination,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".fresh.*")) == []


def test_publish_refuses_secret_from_concurrent_remote_observation(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    state, manifest = _state_v2(tmp_path)
    baseline = publish_annotation_state(hub, state, manifest=manifest)
    remote = _published_payloads(hub, baseline["checkpoint_revision"])
    remote["observations.jsonl"] += _canonical_bytes(
        _observation(
            record_id="remote",
            observed_at="2026-01-01T00:01:00Z",
            disposition="rejected",
            rationale="hf_abc123secret",
        )
    )
    _, rival = _checkpoint_files(remote, manifest=manifest)
    hub.queue_concurrent_commit("batch", rival)
    with (state / "observations.jsonl").open("ab") as handle:
        handle.write(
            _canonical_bytes(
                _observation(
                    record_id="local",
                    observed_at="2026-01-01T00:02:00Z",
                    disposition="accepted",
                    rationale="force CAS",
                )
            )
        )
    before = len(hub.commit_order)

    with pytest.raises(PublicDestinationError, match="credential-shaped"):
        publish_annotation_state(hub, state, manifest=manifest)
    assert len(hub.commit_order) == before + 1


def test_publish_and_resume_refuse_public_repository(tmp_path: Path) -> None:
    state, manifest = _state_v2(tmp_path)
    public = AnnotationsHub(repo_id="org/public-annotations", private=False)

    with pytest.raises(PublicDestinationError, match="public Hub"):
        publish_annotation_state(public, state, manifest=manifest)
    with pytest.raises(PublicDestinationError, match="public Hub"):
        resume_annotation_state(
            public,
            curation_id=_CID,
            destination=tmp_path / "fresh",
        )


# P14 complete final publication ----------------------------------------------------


def _final_bundle(tmp_path: Path) -> tuple[Path, str]:
    from daydream.archive.hydrate_rules import derive_curation_id

    source = "a" * 40
    binding: dict[str, Any] = {
        "schema_version": "2",
        "policy_digest": "1" * 64,
        "policy_version": "production-v1",
        "allow_copyleft": ["owner/repo"],
        "exclusions_digest": "2" * 64,
        "resolved_decisions_digest": "3" * 64,
        "distribution_digest": "4" * 64,
    }
    curation_id = derive_curation_id(
        source,
        binding["policy_digest"],
        binding["policy_version"],
        frozenset(binding["allow_copyleft"]),
        binding["exclusions_digest"],
        binding["resolved_decisions_digest"],
        binding["distribution_digest"],
    )
    root = tmp_path / "final"
    root.mkdir()
    snapshot_id = "b" * 64
    labeler_version = "984-adjudicate-r1"
    rubric_version = "980-rubric-r2"
    classifier_version = "980-classifier-r1"
    payloads: dict[str, object] = {
        "annotations.jsonl": {"record_id": "r1"},
        "sessions.jsonl": {"session_id": "s1"},
        "label-observations.jsonl": {"record_id": "r1", "observed_at": "2026-01-01T00:00:00Z"},
        "coverage-report.json": {
            "outcome_coverage": {"adjudicated": 1, "total": 1},
            "silver_task_only_count": 0,
            "class_balance": {"accepted": 1, "rejected": 0},
            "unresolved": 0,
            "inter_rater": {"items": 0, "agreeing": 0},
            "strata": {"python/pr_review": 1},
            "evidence_after_as_of": [],
            "admission_gate": {
                "outcome_bearing_total": 1,
                "total": 1,
                "passes_80pct": True,
                "class_balance_ok": False,
                "gate_version": 1,
            },
        },
        "lineage.json": {
            "curation_id": curation_id,
            "sanitized_hub_commit": source,
            "snapshot_id": snapshot_id,
            "labeler_version": labeler_version,
            "rubric_version": rubric_version,
            "classifier_version": classifier_version,
            "schema_version": "annotation-snapshot/1055-snapshot-r1",
            "batch_fileset_digest": "5" * 64,
            "as_of": "",
        },
        "preview-manifest.json": {
            "curation_id": curation_id,
            "source_hub_commit": source,
            "sanitized_hub_commit": source,
            "snapshot_id": snapshot_id,
            "labeler_version": labeler_version,
            "rubric_version": rubric_version,
            "classifier_version": classifier_version,
            "as_of": None,
        },
        "policy-binding.json": binding,
    }
    for name, value in payloads.items():
        encoded = (
            (json.dumps(value, sort_keys=True) + "\n").encode()
            if name == "policy-binding.json"
            else _canonical_bytes(value)
        )
        (root / name).write_bytes(encoded)
    return root, curation_id


def _seed_final_envelope(
    hub: AnnotationsHub, bundle: Path
) -> tuple[str, str, str]:
    from daydream.training.adjudication.final_bundle import FINAL_IDENTITY_FILES, final_snapshot_id

    final_id, digests = final_snapshot_id(bundle)
    semantic = {name: (bundle / name).read_bytes() for name in FINAL_IDENTITY_FILES}
    preview = json.loads(semantic["preview-manifest.json"])
    curation_id = preview["curation_id"]
    prefix = f"annotations/{curation_id}/{final_id}/final/"
    publication = _canonical_bytes(
        {
            "schema_version": "annotation-publication/v1",
            "curation_id": curation_id,
            "source_snapshot_id": preview["snapshot_id"],
            "final_snapshot_id": final_id,
            "files": digests,
        }
    )
    data = {**semantic, "publication-manifest.json": publication}
    data["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n"
        for name, value in sorted(data.items())
    ).encode()
    data_oid = hub.seed_remote_files({f"{prefix}{name}": value for name, value in data.items()})
    success = _canonical_bytes(
        {
            "schema_version": "annotation-success/v1",
            "final_snapshot_id": final_id,
            "data_commit_oid": data_oid,
        }
    )
    success_oid = hub.seed_remote_files({f"{prefix}_SUCCESS": success})
    return curation_id, final_id, success_oid


@pytest.mark.parametrize(
    "case",
    ["missing-field", "bool-count", "contradictory-gate", "cross-count"],
)
def test_final_publish_rejects_nonproducer_coverage_report(case: str, tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    path = bundle / "coverage-report.json"
    report = json.loads(path.read_text())
    if case == "missing-field":
        report.pop("class_balance")
    elif case == "bool-count":
        report["outcome_coverage"]["total"] = True
    elif case == "contradictory-gate":
        report["outcome_coverage"] = {"adjudicated": 0, "total": 100}
        report["class_balance"] = {"accepted": 0, "rejected": 100}
        report["admission_gate"]["outcome_bearing_total"] = 0
        report["admission_gate"]["total"] = 100
    else:
        report["admission_gate"]["total"] = 2
    path.write_bytes(_canonical_bytes(report))

    with pytest.raises((ValueError, PublicDestinationError), match="coverage-report"):
        publish_final_annotation_bundle(hub, bundle)

    assert hub.commit_order == []


@pytest.mark.parametrize(
    "case",
    ["missing-pin", "schema", "digest", "shared-pin", "as-of"],
)
def test_final_publish_rejects_nonproducer_lineage(case: str, tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    path = bundle / "lineage.json"
    lineage = json.loads(path.read_text())
    if case == "missing-pin":
        lineage.pop("snapshot_id")
    elif case == "schema":
        lineage["schema_version"] = "annotation-snapshot/future"
    elif case == "digest":
        lineage["batch_fileset_digest"] = "not-a-digest"
    elif case == "shared-pin":
        lineage["classifier_version"] = "different"
    else:
        lineage["as_of"] = "None"
    path.write_bytes(_canonical_bytes(lineage))

    with pytest.raises(ValueError, match="lineage.json"):
        publish_final_annotation_bundle(hub, bundle)

    assert hub.commit_order == []


def test_final_download_revalidates_lineage_inside_valid_hash_envelope(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    lineage_path = bundle / "lineage.json"
    lineage = json.loads(lineage_path.read_text())
    lineage["schema_version"] = "annotation-snapshot/future"
    lineage_path.write_bytes(_canonical_bytes(lineage))
    curation_id, final_id, revision = _seed_final_envelope(hub, bundle)

    with pytest.raises(ValueError, match="lineage.json"):
        download_final_annotation_bundle(
            hub,
            curation_id=curation_id,
            snapshot_id=final_id,
            revision=revision,
            destination=tmp_path / "download",
        )

    assert not (tmp_path / "download").exists()


def test_final_publish_returns_actual_success_commit_is_last_and_idempotent(
    tmp_path: Path,
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)

    result = publish_final_annotation_bundle(hub, bundle)

    assert result["hub_commit_sha"] == hub.commit_order[-1]["sha"]
    assert result["data_commit_sha"] == hub.commit_order[-2]["sha"]
    assert len(hub.atomic_attempt_log[-2]["contains"]) == 9
    assert hub.commit_order[-1]["contains"] == [f"{result['prefix']}_SUCCESS"]
    assert result["prefix"] == (
        f"annotations/{curation_id}/{result['final_snapshot_id']}/final/"
    )
    marker = json.loads(
        hub.revision_files(result["hub_commit_sha"])[f"{result['prefix']}_SUCCESS"]
    )
    assert marker == {
        "schema_version": "annotation-success/v1",
        "final_snapshot_id": result["final_snapshot_id"],
        "data_commit_oid": result["data_commit_sha"],
    }
    commits = len(hub.commit_order)
    downloaded_before = len(hub.downloaded_revision_log)
    assert publish_final_annotation_bundle(hub, bundle) == result
    assert len(hub.commit_order) == commits
    assert all(
        revision is not None
        for _, revision in hub.downloaded_revision_log[downloaded_before:]
    )


def test_final_publish_hashes_the_same_bytes_it_uploads_during_local_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    annotations = (bundle / "annotations.jsonl").resolve()
    original = annotations.read_bytes()
    replacement = original + b"\n"
    real_read_bytes = Path.read_bytes
    replaced = False

    def read_then_replace(path: Path) -> bytes:
        nonlocal replaced
        data = real_read_bytes(path)
        if path == annotations and not replaced:
            annotations.write_bytes(replacement)
            replaced = True
        return data

    # Keep real filesystem reads and writes, but force a competing edit at the
    # boundary where the old publisher separated hashing from payload capture.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_then_replace)
        published = publish_final_annotation_bundle(hub, bundle)

    assert replaced
    assert annotations.read_bytes() == replacement
    prefix = published["prefix"]
    remote = hub.revision_files(published["hub_commit_sha"])
    publication = json.loads(remote[f"{prefix}publication-manifest.json"])
    for name, digest in publication["files"].items():
        assert hashlib.sha256(remote[f"{prefix}{name}"]).hexdigest() == digest
    assert remote[f"{prefix}annotations.jsonl"] == original

    destination = tmp_path / "verified download"
    download_final_annotation_bundle(
        hub,
        curation_id=curation_id,
        snapshot_id=published["final_snapshot_id"],
        revision=published["hub_commit_sha"],
        destination=destination,
    )
    assert (destination / "annotations.jsonl").read_bytes() == original


def test_final_publish_accepts_real_bundle_root_under_symlinked_ancestor(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    bundle, _curation_id = _final_bundle(alias)
    hub = AnnotationsHub(repo_id="org/private-annotations")

    result = publish_final_annotation_bundle(hub, bundle)

    assert result["hub_commit_sha"] == hub.repo_info().sha


def test_final_publish_still_refuses_a_declared_symlink_bundle_root(tmp_path: Path) -> None:
    bundle, _curation_id = _final_bundle(tmp_path)
    linked = tmp_path / "linked-final"
    linked.symlink_to(bundle, target_is_directory=True)
    hub = AnnotationsHub(repo_id="org/private-annotations")

    with pytest.raises(ValueError, match="final bundle must be a real directory"):
        publish_final_annotation_bundle(hub, linked)

    assert hub.info_revision_log == []


@pytest.mark.parametrize("stage", ["data", "success"])
def test_final_publish_retries_typed_compare_and_swap_conflicts(
    stage: str, tmp_path: Path
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    hub.queue_concurrent_commit(stage, {f"unrelated/{stage}.txt": b"rival"})

    result = publish_final_annotation_bundle(hub, bundle)

    assert result["hub_commit_sha"] == hub.commit_order[-1]["sha"]
    assert [entry["stage"] for entry in hub.atomic_attempt_log].count(stage) >= 2


def test_final_publish_refuses_populated_same_prefix_collision(tmp_path: Path) -> None:
    from daydream.training.adjudication.final_bundle import final_snapshot_id

    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    final_id, _digests = final_snapshot_id(bundle)
    prefix = f"annotations/{curation_id}/{final_id}/final/"
    hub.seed_remote_files({f"{prefix}annotations.jsonl": b"different\n"})
    before = len(hub.commit_order)

    with pytest.raises(HydrationError, match="collision"):
        publish_final_annotation_bundle(hub, bundle)
    assert len(hub.commit_order) == before


def test_final_publish_accepts_valid_rival_success_bound_to_distinct_data_commit(
    tmp_path: Path,
) -> None:
    source_hub = AnnotationsHub(repo_id="org/source-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    source_result = publish_final_annotation_bundle(source_hub, bundle)
    data_tree = source_hub.revision_files(source_result["data_commit_sha"])
    data_mapping = {
        path: value
        for path, value in data_tree.items()
        if path.startswith(source_result["prefix"])
    }

    hub = AnnotationsHub(repo_id="org/private-annotations")
    first_data = hub.seed_remote_files(data_mapping, "first identical data")
    rival_data = hub.seed_remote_files(data_mapping, "rival identical data")
    assert rival_data != first_data
    marker = _canonical_bytes(
        {
            "schema_version": "annotation-success/v1",
            "final_snapshot_id": source_result["final_snapshot_id"],
            "data_commit_oid": rival_data,
        }
    )
    hub.queue_concurrent_commit(
        "success",
        {f"{source_result['prefix']}_SUCCESS": marker},
    )
    before = len(hub.commit_order)

    result = publish_final_annotation_bundle(hub, bundle)

    assert result["data_commit_sha"] == rival_data
    assert result["hub_commit_sha"] == hub.commit_order[-1]["sha"]
    assert len(hub.commit_order) == before + 1


@pytest.mark.parametrize(
    "marker",
    [
        b"not-json\n",
        _canonical_bytes(
            {
                "schema_version": "annotation-success/v1",
                "final_snapshot_id": "0" * 64,
                "data_commit_oid": "a" * 40,
            }
        ),
        _canonical_bytes(
            {
                "schema_version": "annotation-success/v1",
                "final_snapshot_id": "PLACEHOLDER",
                "data_commit_oid": "not-an-oid",
            }
        ),
    ],
)
def test_final_publish_refuses_malformed_or_mismatched_success_marker(
    marker: bytes, tmp_path: Path
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    first = publish_final_annotation_bundle(hub, bundle)
    if b"PLACEHOLDER" in marker:
        marker = marker.replace(b"PLACEHOLDER", first["final_snapshot_id"].encode())
    hub.seed_remote_files({f"{first['prefix']}_SUCCESS": marker})
    before = len(hub.commit_order)

    with pytest.raises((ValueError, HydrationError)):
        publish_final_annotation_bundle(hub, bundle)
    assert len(hub.commit_order) == before


def test_final_publish_refuses_success_marker_with_unknown_valid_data_oid(
    tmp_path: Path,
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    result = publish_final_annotation_bundle(hub, bundle)
    marker = _canonical_bytes(
        {
            "schema_version": "annotation-success/v1",
            "final_snapshot_id": result["final_snapshot_id"],
            "data_commit_oid": "f" * 40,
        }
    )
    hub.seed_remote_files({f"{result['prefix']}_SUCCESS": marker})

    with pytest.raises(HydrationError, match="unknown revision"):
        publish_final_annotation_bundle(hub, bundle)


@pytest.mark.parametrize(
    "name",
    [
        "annotations.jsonl",
        "sessions.jsonl",
        "label-observations.jsonl",
        "coverage-report.json",
        "lineage.json",
        "preview-manifest.json",
        "policy-binding.json",
    ],
)
def test_final_publish_secret_in_every_semantic_file_commits_nothing(
    name: str, tmp_path: Path
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    (bundle / name).write_bytes((bundle / name).read_bytes() + b"hf_abc123secret")

    with pytest.raises(PublicDestinationError, match="credential-shaped"):
        publish_final_annotation_bundle(hub, bundle)
    assert hub.commit_order == []


def test_final_publish_rejects_symlinked_semantic_input_before_hub_access(
    tmp_path: Path,
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("hf_abc123secret")
    (bundle / "annotations.jsonl").unlink()
    (bundle / "annotations.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="regular file"):
        publish_final_annotation_bundle(hub, bundle)
    assert hub.info_revision_log == []


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_final_download_requires_fresh_destination(kind: str, tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    result = publish_final_annotation_bundle(hub, bundle)
    destination = tmp_path / "download"
    if kind == "file":
        destination.write_text("existing")
    elif kind == "directory":
        destination.mkdir()
    else:
        target = tmp_path / "target"
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    hub.info_revision_log.clear()

    with pytest.raises(ValueError, match="must not exist"):
        download_final_annotation_bundle(
            hub,
            curation_id=curation_id,
            snapshot_id=result["final_snapshot_id"],
            revision=result["hub_commit_sha"],
            destination=destination,
        )
    assert hub.info_revision_log == []


def test_final_download_is_pinned_and_installs_one_complete_fresh_tree(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    published = publish_final_annotation_bundle(hub, bundle)
    hub.info_revision_log.clear()
    hub.downloaded_revision_log.clear()
    destination = tmp_path / "download"

    result = download_final_annotation_bundle(
        hub,
        curation_id=curation_id,
        snapshot_id=published["final_snapshot_id"],
        revision=published["hub_commit_sha"],
        destination=destination,
    )

    assert result["hub_commit_sha"] == published["hub_commit_sha"]
    assert result["data_commit_sha"] == published["data_commit_sha"]
    assert sorted(path.name for path in destination.iterdir()) == published["files"]
    assert hub.info_revision_log[0] == published["hub_commit_sha"]
    assert all(revision is not None for _, revision in hub.downloaded_revision_log)
    assert list(tmp_path.glob(".download.*")) == []


def test_final_download_accepts_fresh_destination_under_symlinked_ancestor(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    published = publish_final_annotation_bundle(hub, bundle)
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    download_final_annotation_bundle(
        hub,
        curation_id=curation_id,
        snapshot_id=published["final_snapshot_id"],
        revision=published["hub_commit_sha"],
        destination=alias / "download",
    )

    assert (actual / "download" / "_SUCCESS").is_file()


def test_final_download_parent_fsync_failure_removes_owned_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    published = publish_final_annotation_bundle(hub, bundle)
    destination = tmp_path / "download"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    real_fsync = os.fsync

    def fail_parent_after_rename(fd: int) -> None:
        stat = os.fstat(fd)
        if destination.exists() and (stat.st_dev, stat.st_ino) == parent_identity:
            raise OSError("parent fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_parent_after_rename)
    with pytest.raises(HydrationError, match="parent fsync failed"):
        download_final_annotation_bundle(
            hub,
            curation_id=curation_id,
            snapshot_id=published["final_snapshot_id"],
            revision=published["hub_commit_sha"],
            destination=destination,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".download.*")) == []


def _record_open_directory_fds(monkeypatch: pytest.MonkeyPatch) -> set[int]:
    opened: set[int] = set()
    real_open = os.open

    def record_open(*args: Any, **kwargs: Any) -> int:
        fd = real_open(*args, **kwargs)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            opened.add(fd)
        return fd

    monkeypatch.setattr(os, "open", record_open)
    return opened


def _open_fd_identity(fd: int) -> tuple[int, int] | None:
    try:
        info = os.fstat(fd)
    except OSError:
        return None
    return info.st_dev, info.st_ino


@pytest.mark.parametrize("operation", ["download", "resume"])
def test_final_download_cleanup_preserves_concurrent_destination_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    if operation == "download":
        bundle, curation_id = _final_bundle(tmp_path)
        published = publish_final_annotation_bundle(hub, bundle)
    else:
        state, manifest = _state_v2(tmp_path)
        publish_annotation_state(hub, state, manifest=manifest)
    destination = tmp_path / "download"
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    real_fsync = os.fsync
    directory_fds = _record_open_directory_fds(monkeypatch)
    installed_inode_was_pinned: list[bool] = []

    def replace_then_fail(fd: int) -> None:
        stat = os.fstat(fd)
        if destination.exists() and (stat.st_dev, stat.st_ino) == parent_identity:
            installed = destination.stat()
            installed_identity = (installed.st_dev, installed.st_ino)
            installed_inode_was_pinned.append(
                any(_open_fd_identity(open_fd) == installed_identity for open_fd in directory_fds)
            )
            shutil.rmtree(destination)
            destination.mkdir()
            (destination / "concurrent").write_text("keep")
            raise OSError("parent fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", replace_then_fail)
    with pytest.raises(HydrationError, match="parent fsync failed"):
        if operation == "download":
            download_final_annotation_bundle(
                hub,
                curation_id=curation_id,
                snapshot_id=published["final_snapshot_id"],
                revision=published["hub_commit_sha"],
                destination=destination,
            )
        else:
            resume_annotation_state(hub, curation_id=_CID, destination=destination)

    assert (destination / "concurrent").read_text() == "keep"
    assert installed_inode_was_pinned == [True]
    assert list(tmp_path.glob(".download.*")) == []
    assert directory_fds
    assert all(_open_fd_identity(fd) is None for fd in directory_fds)


@pytest.mark.parametrize("operation", ["download", "resume"])
def test_successful_installation_closes_directory_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    if operation == "download":
        bundle, curation_id = _final_bundle(tmp_path)
        published = publish_final_annotation_bundle(hub, bundle)
    else:
        state, manifest = _state_v2(tmp_path)
        publish_annotation_state(hub, state, manifest=manifest)
    destination = tmp_path / "installed"
    directory_fds = _record_open_directory_fds(monkeypatch)

    if operation == "download":
        download_final_annotation_bundle(
            hub,
            curation_id=curation_id,
            snapshot_id=published["final_snapshot_id"],
            revision=published["hub_commit_sha"],
            destination=destination,
        )
    else:
        resume_annotation_state(hub, curation_id=_CID, destination=destination)

    assert destination.is_dir()
    assert list(tmp_path.glob(".installed.*")) == []
    assert directory_fds
    assert all(_open_fd_identity(fd) is None for fd in directory_fds)


@pytest.mark.parametrize("after_stage", ["data", "success"])
def test_final_publish_verification_failure_never_returns_success(
    after_stage: str, tmp_path: Path
) -> None:
    class CorruptingHub(AnnotationsHub):
        def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
            data = super().download_file(path_in_repo, revision)
            if (
                self.atomic_attempt_log
                and self.atomic_attempt_log[-1]["stage"] == after_stage
                and path_in_repo.endswith("/annotations.jsonl")
            ):
                return data + b"tampered"
            return data

    hub = CorruptingHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)

    with pytest.raises(HydrationError, match="collision"):
        publish_final_annotation_bundle(hub, bundle)
    stages = [entry["stage"] for entry in hub.atomic_attempt_log]
    if after_stage == "data":
        assert stages == ["data"]
        assert not any(path.endswith("/_SUCCESS") for path in hub.files)
    else:
        assert stages == ["data", "success"]


def test_final_publish_rejects_changed_data_behind_success_marker(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    result = publish_final_annotation_bundle(hub, bundle)
    hub.seed_remote_files(
        {f"{result['prefix']}annotations.jsonl": b"changed-behind-marker\n"}
    )
    before = len(hub.commit_order)

    with pytest.raises(HydrationError, match="collision"):
        publish_final_annotation_bundle(hub, bundle)
    assert len(hub.commit_order) == before


def test_final_publish_rejects_missing_data_behind_success_marker(tmp_path: Path) -> None:
    class MissingListingHub(AnnotationsHub):
        hide = False

        def list_repo_files(self, revision: str | None = None) -> list[str]:
            paths = super().list_repo_files(revision)
            if self.hide:
                return [path for path in paths if not path.endswith("/annotations.jsonl")]
            return paths

    hub = MissingListingHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    publish_final_annotation_bundle(hub, bundle)
    hub.hide = True
    before = len(hub.commit_order)

    with pytest.raises(HydrationError, match="collision"):
        publish_final_annotation_bundle(hub, bundle)
    assert len(hub.commit_order) == before


@pytest.mark.parametrize("bad_name", ["../escape", "nested\\escape", "./annotations.jsonl"])
def test_final_publish_rejects_non_normalized_remote_prefix_paths(
    bad_name: str, tmp_path: Path
) -> None:
    from daydream.training.adjudication.final_bundle import final_snapshot_id

    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    final_id, _digests = final_snapshot_id(bundle)
    prefix = f"annotations/{curation_id}/{final_id}/final/"
    hub.seed_remote_files({f"{prefix}{bad_name}": b"foreign"})

    with pytest.raises(ValueError, match="path"):
        publish_final_annotation_bundle(hub, bundle)


def test_final_publish_secret_shaped_identity_commits_nothing(tmp_path: Path) -> None:
    hub = AnnotationsHub(repo_id="org/private-annotations")
    bundle, _curation_id = _final_bundle(tmp_path)
    preview_path = bundle / "preview-manifest.json"
    preview = json.loads(preview_path.read_text())
    preview["curation_id"] = "cur-hf_abc123secret"
    preview_path.write_bytes(_canonical_bytes(preview))

    with pytest.raises(PublicDestinationError, match="credential-shaped"):
        publish_final_annotation_bundle(hub, bundle)
    assert hub.commit_order == []


def test_final_download_last_remote_read_failure_leaves_no_stage(tmp_path: Path) -> None:
    class LastDownloadFails(AnnotationsHub):
        fail = False

        def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
            if self.fail and path_in_repo.endswith("/sessions.jsonl"):
                raise HubDownloadError("last final download failed")
            return super().download_file(path_in_repo, revision)

    hub = LastDownloadFails(repo_id="org/private-annotations")
    bundle, curation_id = _final_bundle(tmp_path)
    result = publish_final_annotation_bundle(hub, bundle)
    hub.fail = True
    destination = tmp_path / "download"

    with pytest.raises(HydrationError, match="last final download failed"):
        download_final_annotation_bundle(
            hub,
            curation_id=curation_id,
            snapshot_id=result["final_snapshot_id"],
            revision=result["hub_commit_sha"],
            destination=destination,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".download.*")) == []


def test_final_publish_and_download_refuse_public_repository(tmp_path: Path) -> None:
    bundle, curation_id = _final_bundle(tmp_path)
    private = AnnotationsHub(repo_id="org/private-annotations")
    published = publish_final_annotation_bundle(private, bundle)
    public = AnnotationsHub(
        repo_id="org/public-annotations",
        private=False,
        files=private.revision_files(published["hub_commit_sha"]),
    )

    with pytest.raises(PublicDestinationError, match="public Hub"):
        publish_final_annotation_bundle(public, bundle)
    with pytest.raises(PublicDestinationError, match="public Hub"):
        download_final_annotation_bundle(
            public,
            curation_id=curation_id,
            snapshot_id=published["final_snapshot_id"],
            revision=public.repo_info().sha,
            destination=tmp_path / "download",
        )
