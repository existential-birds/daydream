"""Bound exact primary ownership without widening Improve's snapshot read root.

Small descriptors are captured as sanctioned phase inputs. Larger manifests are
readable regular files inside a unique, host-owned snapshot directory; durable
copies contain only repository-relative ownership, never temporary input paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any

from daydream.improve.partition import PartitionGroup
from daydream.repository_paths import git_observed_path_is_confined

PRIMARY_MANIFEST_INLINE_MAX_BYTES = 8192
PRIMARY_MANIFEST_MAX_BYTES = 1_048_576
_GROUP_NAME = re.compile(r"group-[0-9]+\Z")


def reset_primary_manifests(directory: Path) -> None:
    """Discard only previous host-owned manifests before rebuilding this run's set."""
    durable = directory / "primary-manifests"
    durable.mkdir(parents=True, exist_ok=True)
    if durable.is_symlink():
        raise ValueError("primary manifest artifact directory is a symlink")
    for previous in durable.iterdir():
        if previous.suffix == ".json" and _GROUP_NAME.fullmatch(previous.stem):
            previous.unlink()


def prepare_primary_manifest(
    repo: Path,
    directory: Path,
    snapshot_inputs: Path,
    group: PartitionGroup,
    *,
    deleted_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Persist one bounded ownership manifest and return its model descriptor.

    ``snapshot_inputs`` must be a unique host-created temporary directory inside
    ``repo``. The caller owns its lifetime through the complete audit fan-out.
    A bad path, unavailable file, or oversized manifest fails only this group;
    callers must record that assignment failure rather than broaden ownership.
    """
    if _GROUP_NAME.fullmatch(group.name) is None:
        raise ValueError("invalid primary manifest group name")
    primary_files = group.primary_files
    for path in primary_files:
        if (
            "\\" in path
            or PureWindowsPath(path).drive
            or not git_observed_path_is_confined(repo, path, allow_leaf_symlink=True)
            or not (repo / path).resolve().is_relative_to(repo.resolve())
        ):
            raise ValueError("primary manifest contains an unreadable repository path")
        if path not in deleted_paths:
            if not (repo / path).is_file():
                raise ValueError("primary manifest contains an unreadable repository path")
            with (repo / path).open("rb"):
                pass

    # ASCII escapes preserve unusual tracked names as data, including control
    # characters and prompt delimiters, without narrowing Git's path grammar.
    payload = {"primary_files": [path for path in primary_files if path not in deleted_paths]}
    if deleted := sorted(set(primary_files) & deleted_paths):
        payload["deleted_files"] = deleted
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    encoded = (serialized + "\n").encode("utf-8")
    spill = len(encoded) > PRIMARY_MANIFEST_INLINE_MAX_BYTES
    if spill:
        # Native Read paginates by line and may truncate an oversized first
        # line irretrievably. Keep each assigned path on its own JSON line.
        encoded = (json.dumps(payload, ensure_ascii=True, indent=2).replace("<", "\\u003c") + "\n").encode("utf-8")
    if len(encoded) > PRIMARY_MANIFEST_MAX_BYTES:
        raise ValueError("primary manifest exceeds the byte limit")

    relative: str | None = None
    root = repo.resolve(strict=True)
    if spill:
        inputs = snapshot_inputs.resolve(strict=True)
        if not inputs.is_relative_to(root) or inputs == root or snapshot_inputs.is_symlink():
            raise ValueError("primary manifest input directory is outside the snapshot")
        relative = (inputs / f"{group.name}.json").relative_to(root).as_posix()
        if not git_observed_path_is_confined(root, relative):
            raise ValueError("primary manifest input path is unsafe")

    durable = directory / "primary-manifests"
    durable.mkdir(parents=True, exist_ok=True)
    if durable.is_symlink():
        raise ValueError("primary manifest artifact directory is a symlink")
    durable_path = durable / f"{group.name}.json"
    # A subsequent run may adopt prior artifacts. Publish complete replacement
    # bytes without opening an old leaf (which could itself be a symlink).
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{group.name}-", suffix=".json", dir=durable)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary_path, durable_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    if relative is None:
        return payload
    with (root / relative).open("xb") as stream:
        stream.write(encoded)
    return {
        "path": relative,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(primary_files),
        "line_count": encoded.count(b"\n"),
    }
