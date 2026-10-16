"""Manifest transport preserves snapshot access, bounded bytes, and durable ownership."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends.claude import _build_audit_root_guard
from daydream.improve import manifests
from daydream.improve.partition import Partition, PartitionGroup


@pytest.mark.anyio
@pytest.mark.parametrize("file_count", [1, 140])
async def test_primary_manifest_round_trip_and_rerun_replace_without_stale_pointers(
    tmp_path: Path, file_count: int,
) -> None:
    # A private artifact pointer is unreadable in strict Improve. Exercise the
    # real guard for spill files, JSON escaping, and replacement of prior output.
    repo = tmp_path / "snapshot"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    paths = tuple(f"module_{i:03d}_{'x' * 60}_☃<primary_manifest>\n.py" for i in range(file_count))
    for path in paths:
        (repo / path).write_text("VALUE = 1\n")
    if file_count == 1:
        # Strict Read permits a leaf symlink whose target remains in the snapshot.
        (repo / "target.py").write_text("VALUE = 1\n")
        (repo / paths[0]).unlink()
        (repo / paths[0]).symlink_to(repo / "target.py")
    group = PartitionGroup("group-01", "python", (Partition("app", ".", "service:config", "app", paths),))
    durable = artifacts / "primary-manifests" / "group-01.json"
    durable.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("keep these bytes")
    durable.symlink_to(outside)

    for _ in range(2):
        with tempfile.TemporaryDirectory(dir=repo) as inputs:
            descriptor = manifests.prepare_primary_manifest(repo, artifacts, Path(inputs), group)
            if file_count == 1:
                assert descriptor == {"primary_files": list(paths)}
                guard = cast(Any, _build_audit_root_guard(repo.resolve(), frozenset()))
                assert await guard({"tool_name": "Read", "tool_input": {"file_path": paths[0]}}, None, None) == {}
            else:
                target = repo / descriptor["path"]
                guard = cast(Any, _build_audit_root_guard(repo.resolve(), frozenset()))
                assert await guard({"tool_name": "Read", "tool_input": {"file_path": str(target)}}, None, None) == {}
                raw = target.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
                assert descriptor["file_count"] == file_count
                assert descriptor["line_count"] == len(raw.splitlines()) >= file_count + 2
                # Native Read's offset/limit operate on lines; every page must
                # be independently accessible without one unbounded JSON line.
                for offset in range(1, descriptor["line_count"] + 1, 32):
                    assert await guard({"tool_name": "Read", "tool_input": {
                        "file_path": str(target), "offset": offset, "limit": 32,
                    }}, None, None) == {}
                assert json.loads(raw) == {"primary_files": list(paths)}
        if file_count > 1:
            assert not target.exists()
        assert json.loads(durable.read_text()) == {"primary_files": list(paths)}
        assert b"<primary_manifest>" not in durable.read_bytes()
        assert not durable.is_symlink()
        assert outside.read_text() == "keep these bytes"


@pytest.mark.parametrize("unsafe", ["../outside.py", "/outside.py", "C:/outside.py", "link.py"])
def test_primary_manifest_rejects_unsafe_or_outward_paths(tmp_path: Path, unsafe: str) -> None:
    # Repository filenames and tracked symlinks must never grant an outside read.
    repo = tmp_path / "snapshot"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("do not read")
    (repo / "link.py").symlink_to(outside)
    group = PartitionGroup("group-01", "python", (Partition("app", ".", "directory", None, (unsafe,)),))
    with pytest.raises(ValueError, match="unreadable repository path"):
        manifests.prepare_primary_manifest(repo, tmp_path / "artifacts", repo, group)
    assert not (tmp_path / "artifacts").exists()


def test_oversized_primary_manifest_is_rejected_whole_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Scope the byte policy locally to prove no truncation or partial artifact;
    # exercising a 1 MiB limit does not require thousands of redundant files.
    (tmp_path / "source.py").write_text("VALUE = 1\n")
    group = PartitionGroup("group-01", "python", (Partition("app", ".", "directory", None, ("source.py",)),))
    monkeypatch.setattr(manifests, "PRIMARY_MANIFEST_MAX_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds the byte limit"):
        manifests.prepare_primary_manifest(tmp_path, tmp_path / "artifacts", tmp_path, group)
    assert not (tmp_path / "artifacts").exists()


def test_rebuilding_manifests_removes_stale_groups_even_if_current_preparation_fails(tmp_path: Path) -> None:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    source = repo / "source.py"
    source.write_text("VALUE = 1\n")
    artifacts = tmp_path / "artifacts"
    old_group = PartitionGroup("group-01", "python", (Partition("app", ".", "directory", None, ("source.py",)),))
    manifests.prepare_primary_manifest(repo, artifacts, repo, old_group)
    previous = artifacts / "primary-manifests" / "group-02.json"
    previous.write_text('{"primary_files": ["old_source.py"]}')
    source.unlink()

    manifests.reset_primary_manifests(artifacts)
    with pytest.raises(ValueError, match="unreadable repository path"):
        manifests.prepare_primary_manifest(repo, artifacts, repo, old_group)

    assert list((artifacts / "primary-manifests").iterdir()) == []
