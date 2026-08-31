import json
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.hydrate import PublicDestinationError
from daydream.training.adjudication.publish import (
    annotation_prefix,
    publish_annotation_state,
    resume_annotation_state,
)


class _FakeHub:
    def __init__(self, private: bool = True) -> None:
        self.files: dict[str, bytes] = {}
        self.private = private
        self.uploads: list[dict[str, Path]] = []

    @property
    def repo_private(self) -> bool:
        return self.private

    def list_repo_files(self, revision: str | None = None) -> list[str]:
        return sorted(self.files)

    def download_file(self, path: str, revision: str | None = None) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def upload_files(self, mapping: dict[str | Path, Path], commit_message: str) -> None:
        self.uploads.append({str(k): v for k, v in mapping.items()})
        for remote, local in mapping.items():
            self.files[str(remote)] = Path(local).read_bytes()

    def repo_info(self, revision: str | None = None) -> Any: ...
    def list_revisions(self) -> list[str]:
        return []


def _state(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    (state / "queue.json").write_text(json.dumps([{"record_id": "r1"}]), encoding="utf-8")
    (state / "observations.jsonl").write_text(
        json.dumps({"record_id": "r1", "disposition": "accepted", "role": "rater"}) + "\n",
        encoding="utf-8",
    )
    (state / "preview-ledger.json").write_text("{}", encoding="utf-8")
    return state


def _manifest(tmp_path: Path) -> Path:
    p = tmp_path / "preview-manifest.json"
    p.write_text(
        json.dumps(
            {
                "curation_id": "cur-1",
                "snapshot_id": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    return p


def test_publish_is_additive_under_content_addressed_prefix(tmp_path: Path) -> None:
    hub = _FakeHub()
    state = _state(tmp_path)
    manifest = _manifest(tmp_path)
    publish_annotation_state(hub, state, manifest=manifest, batch_complete=True)
    prefix = f"annotations/cur-1/{'e' * 64}/"
    assert annotation_prefix(manifest) == prefix
    uploaded = {k for m in hub.uploads for k in m}
    assert any(k == f"{prefix}queue.json" for k in uploaded)
    assert any(k == f"{prefix}observations.jsonl" for k in uploaded)
    # checkpoint after completed batch: remote ledger records the batch
    ledger_path = f"{prefix}checkpoints/batch-latest.json"
    assert ledger_path in hub.files
    entry = json.loads(hub.files[ledger_path])
    assert entry["observation_count"] == 1


def test_publish_second_batch_appends_observations_never_overwrites(tmp_path: Path) -> None:
    hub = _FakeHub()
    state = _state(tmp_path)
    publish_annotation_state(hub, state, manifest=_manifest(tmp_path), batch_complete=True)
    # second batch appends a new observation
    with (state / "observations.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"record_id": "r2", "disposition": "rejected", "role": "rater"}) + "\n")
    publish_annotation_state(hub, state, manifest=_manifest(tmp_path), batch_complete=True)
    stored = hub.files[f"annotations/cur-1/{'e' * 64}/observations.jsonl"].decode()
    lines = [json.loads(line) for line in stored.splitlines() if line.strip()]
    assert [o["record_id"] for o in lines] == ["r1", "r2"]  # append-only (M4)


def test_publish_refuses_public_destination(tmp_path: Path) -> None:
    with pytest.raises(PublicDestinationError):
        publish_annotation_state(_FakeHub(private=False), _state(tmp_path), manifest=_manifest(tmp_path))


def test_publish_final_bundle_success_last_and_verified_commit_required(tmp_path: Path) -> None:
    import hashlib

    from daydream.training.adjudication.publish import publish_final_annotation_bundle

    hub = _FakeHub()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "annotations.jsonl").write_text('{"record_id": "r1"}\n', encoding="utf-8")
    (root / "sessions.jsonl").write_text('{"session_id": "s1"}\n', encoding="utf-8")
    (root / "lineage.json").write_text("{}", encoding="utf-8")
    result = publish_final_annotation_bundle(
        hub, root, manifest=_manifest(tmp_path), verify_download=True,
    )
    prefix = f"annotations/cur-1/{'e' * 64}/final/"
    files = sorted(k[len(prefix):] for k in hub.files if k.startswith(prefix))
    assert "SHA256SUMS" in files and "_SUCCESS" in files
    # _SUCCESS is uploaded last (C3): pinned via hub.uploads insertion order
    # (a sorted() listing cannot express upload order — '_' sorts before letters).
    upload_order = [k for m in hub.uploads for k in m]
    assert upload_order[-1] == f"{prefix}_SUCCESS"
    sums = hub.files[f"{prefix}SHA256SUMS"].decode()
    expected = hashlib.sha256((root / "annotations.jsonl").read_bytes()).hexdigest()
    assert f"{expected}  annotations.jsonl" in sums
    assert result["hub_commit_sha"]  # verified commit recorded (M6)
    assert result["prefix"] == prefix


def test_final_bundle_clean_download_verifies_or_refuses(tmp_path: Path) -> None:
    from daydream.training.adjudication.publish import publish_final_annotation_bundle

    hub = _FakeHub()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "annotations.jsonl").write_text('{"record_id": "r1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="download"):
        publish_final_annotation_bundle(
            hub, root, manifest=_manifest(tmp_path),
            verify_download=True, _download_verifier=lambda _p: False,
        )
    # nothing published: no _SUCCESS on the Hub
    assert not any(k.endswith("/final/_SUCCESS") for k in hub.files)


def test_final_bundle_refuses_secret_in_payload(tmp_path: Path) -> None:
    from daydream.training.adjudication.publish import publish_final_annotation_bundle

    hub = _FakeHub()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "annotations.jsonl").write_text('{"hf_token": "hf_abc123secret"}\n', encoding="utf-8")
    with pytest.raises(Exception):  # PublicDestinationError from the S1 secret scan
        publish_final_annotation_bundle(hub, root, manifest=_manifest(tmp_path))
    assert not any(k.endswith("/final/_SUCCESS") for k in hub.files)


def test_resume_on_empty_disk_restores_byte_identical_state(tmp_path: Path) -> None:
    hub = _FakeHub()
    state = _state(tmp_path)
    publish_annotation_state(hub, state, manifest=_manifest(tmp_path), batch_complete=True)
    fresh = tmp_path / "fresh-vm"  # empty disk, fresh VM
    restored = resume_annotation_state(hub, manifest=_manifest(tmp_path), stage_dir=fresh)
    assert restored["observation_count"] == 1
    for name in ("queue.json", "observations.jsonl", "preview-ledger.json"):
        assert (fresh / name).read_bytes() == (state / name).read_bytes()
    # digests verified on download: corrupt remote fails closed
    hub.files[f"annotations/cur-1/{'e' * 64}/observations.jsonl"] += b"tampered\n"
    with pytest.raises(ValueError, match="digest"):
        resume_annotation_state(hub, manifest=_manifest(tmp_path), stage_dir=tmp_path / "fresh-2")
