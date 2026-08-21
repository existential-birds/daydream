"""Strict, mode-safe filesystem primitives for the private benchmark workspace.

This module owns the on-disk safety layer for ``daydream benchmark``: strict
YAML/JSON loaders that reject duplicate keys and unsafe tags, mode-``0600``
atomic writes, mode-``0700`` private directory creation, sha256 checksums,
and (in later tasks) the workspace lock and transaction journal.

Every parse failure raises :class:`WorkspaceCorrupt` naming the offending file
— a corrupt file is an error, never silently defaulted.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml


class WorkspaceError(Exception):
    """Base error for the private benchmark workspace subsystem."""


class WorkspaceCorrupt(WorkspaceError):
    """A workspace file violates a schema/checksum/path/orphan invariant."""


class LockContentionError(WorkspaceError):
    """Another process holds the workspace lock (explicit non-blocking probe)."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A :class:`yaml.SafeLoader` that rejects documents with duplicate keys.

    Duplicate mapping keys in a manifest or case file are almost always a
    mistake (or a smuggling attempt) — they are treated as corruption.
    """


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise WorkspaceCorrupt(
                f"duplicate key {key!r} in YAML mapping at line {key_node.start_mark.line}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml_strict(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dict, rejecting duplicates, non-dict roots, and unsafe tags.

    Raises:
        WorkspaceCorrupt: if the file is not valid YAML, is not a mapping, is
            empty, or contains duplicate keys / unsafe constructor tags.
    """
    try:
        data = yaml.load(Path(path).read_bytes(), Loader=_UniqueKeyLoader)
    except WorkspaceCorrupt:
        raise
    except yaml.YAMLError as exc:
        raise WorkspaceCorrupt(f"{path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise WorkspaceCorrupt(f"{path}: unreadable: {exc}") from exc
    if data is None:
        raise WorkspaceCorrupt(f"{path}: empty YAML document")
    if not isinstance(data, dict):
        raise WorkspaceCorrupt(f"{path}: YAML root is not a mapping")
    return data


def load_json_strict(path: Path) -> dict[str, Any]:
    """Load a JSON file as a strict dict, rejecting parse errors / non-dict roots."""
    try:
        data = json.loads(Path(path).read_bytes())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise WorkspaceCorrupt(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkspaceCorrupt(f"{path}: JSON root is not an object")
    return data


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    """Atomically write ``content`` to ``path`` via same-dir temp + ``os.replace``.

    Mirrors :func:`daydream.json_utils.atomic_write_json` (temp + replace +
    fsync) but enforces a strict file mode on the final path and creates parent
    directories as private ``0700`` chains.
    """
    ensure_private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        os.chmod(path, mode)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, data: Any, *, mode: int = 0o600) -> None:
    """Atomically write ``data`` as JSON to ``path`` with a strict mode."""
    payload = json.dumps(data, indent=2).encode("utf-8")
    _atomic_write(path, payload, mode=mode)


def atomic_write_yaml(path: Path, data: Any, *, mode: int = 0o600) -> None:
    """Atomically write ``data`` as YAML to ``path`` with a strict mode."""
    payload = yaml.safe_dump(data, sort_keys=False).encode("utf-8")
    _atomic_write(path, payload, mode=mode)


def ensure_private_dir(path: Path, mode: int = 0o700) -> None:
    """Create ``path`` (and any missing parents) with private ``0700`` modes."""
    missing: list[Path] = []
    cursor: Path | None = path
    while cursor is not None and not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        os.chmod(created, mode)
    if path.exists():
        os.chmod(path, mode)


def sha256_file(path: Path) -> str:
    """Return the lowercase 64-hex sha256 digest of ``path``'s bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()
