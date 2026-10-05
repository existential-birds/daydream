"""Strict no-follow filesystem inspection and immutable artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

from daydream.artifacts.models import _SCHEMA_VERSION, ArtifactManifestEntry, ArtifactVisibilityError
from daydream.json_utils import _fsync_directory, atomic_write_bytes


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _projection_path(path: Path) -> Path:
    """Admit one absolute lexical path without resolving absent components."""
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or "\0" in os.fspath(path)
        or _absolute_lexical(path) != path
    ):
        raise ArtifactVisibilityError("artifact projection path is unsafe")
    return path


def _validate_projection_ancestry(
    path: Path,
    *,
    expected_kind: Literal["file", "directory", "either"],
) -> None:
    """Validate existing projection components without requiring the leaf."""
    for index, component in enumerate((path, *path.parents)):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ArtifactVisibilityError("artifact projection path could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVisibilityError("artifact projection ancestry contains a symlink")
        if index == 0:
            valid_leaf = (
                expected_kind == "directory"
                and stat.S_ISDIR(metadata.st_mode)
                or expected_kind == "file"
                and stat.S_ISREG(metadata.st_mode)
                or expected_kind == "either"
                and (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
            )
            if not valid_leaf:
                raise ArtifactVisibilityError("artifact projection has the wrong filesystem type")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError("artifact projection ancestry is not a directory")


def _declared_directory_metadata(path: Path, *, label: str) -> tuple[Path, os.stat_result]:
    """Resolve a declared directory, returning it with its own no-follow metadata."""
    declared = _absolute_lexical(path)
    declared_metadata: os.stat_result | None = None
    for index, component in enumerate((declared, *declared.parents)):
        try:
            metadata = component.lstat()
        except OSError as exc:
            raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVisibilityError(f"{label} ancestry must not contain a symlink")
        if index == 0:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVisibilityError(f"{label} must be a real directory, not a symlink")
            declared_metadata = metadata
    assert declared_metadata is not None
    try:
        return declared.resolve(strict=True), declared_metadata
    except OSError as exc:
        raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc


def _declared_directory(path: Path, *, label: str) -> Path:
    return _declared_directory_metadata(path, label=label)[0]


def _open_directory_descriptor(path: Path, *, label: str) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ArtifactVisibilityError(f"{label} is not an accessible directory")
    return descriptor


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_private_root_declaration(path: Path, *, label: str) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or _absolute_lexical(path) != path:
        raise ArtifactVisibilityError(f"{label} must be an absolute lexical path")
    for component in (path, *path.parents):
        if not component.exists() and not component.is_symlink():
            continue
        metadata = component.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVisibilityError(f"{label} ancestry contains a symlink")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError(f"{label} ancestry is not a directory")


def validate_private_directory(path: Path, *, label: str, allow_absent: bool = False) -> None:
    """Refuse anything but a real, mode-0700 directory at ``path``.

    ``allow_absent`` accepts a path that does not exist yet, which discovery roots need.
    """
    if allow_absent and not path.exists() and not path.is_symlink():
        return
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactVisibilityError(f"{label} must be a real directory")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ArtifactVisibilityError(f"{label} must have mode 0700")


def _create_private_directory(path: Path, *, exist_ok: bool = False) -> None:
    """Create ``path`` and missing parents as private 0700 directories, durably.

    Every created directory is chmod'ed to ``0o700`` immediately after its
    mkdir (umask-immune), the parent entry of every creation is fsync'ed, and
    the final directory is validated as a real, private directory — a symlink
    anywhere in the ancestry fails closed instead of being chmod'ed through.
    """
    missing: list[Path] = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    for existing in (cursor, *cursor.parents):
        if existing == path:
            # The leaf's own kind and mode are the storage root's, not its
            # ancestry's; validate_private_directory below reports it as such.
            continue
        try:
            metadata = existing.lstat()
        except OSError as exc:
            raise ArtifactVisibilityError("artifact runtime ancestry is not accessible") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVisibilityError("artifact runtime ancestry contains a symlink")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=exist_ok)
        directory_fd = _open_directory_descriptor(directory, label="private storage root")
        try:
            os.fchmod(directory_fd, 0o700)
        finally:
            os.close(directory_fd)
        _fsync_directory(directory.parent)
    validate_private_directory(path, label="private storage root")


def _atomic_json(path: Path, payload: object) -> None:
    """Atomically persist one canonical-JSON manifest document.

    Compact sorted separators (byte-exact for all persisted manifests) with a
    trailing newline, file fsync before the rename, and a parent-directory
    fsync after it.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    atomic_write_bytes(path, encoded, fsync=True, dir_fsync=True, mode=0o600)


def _atomic_bytes(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    """Atomically persist raw bytes at a strict mode (private artifact files)."""
    atomic_write_bytes(path, content, fsync=True, dir_fsync=True, mode=mode)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVisibilityError("artifact metadata is not a regular file")
        value = json.loads(_read_regular(path, metadata))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactVisibilityError("artifact metadata is malformed") from exc
    if not isinstance(value, dict):
        raise ArtifactVisibilityError("artifact metadata is malformed")
    return cast(dict[str, Any], value)


def _load_ledger(
    path: Path,
    *,
    items_key: str,
    message: str,
    identity: Sequence[tuple[str, object]] = (),
    required: bool = False,
) -> list[dict[str, object]]:
    """Read one versioned ledger envelope, returning its still-unvalidated items.

    An absent ledger reads as empty unless ``required``; per-entry validation
    stays with the caller that knows the entry shape.
    """
    if not required and not path.exists() and not path.is_symlink():
        return []
    payload = _load_json(path)
    if (
        set(payload) != {"schema_version", items_key, *(key for key, _ in identity)}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != _SCHEMA_VERSION
        or any(type(payload[key]) is not type(value) or payload[key] != value for key, value in identity)
        or not isinstance(payload.get(items_key), list)
    ):
        raise ArtifactVisibilityError(message)
    return cast(list[dict[str, object]], payload[items_key])


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_relative_name(name: str) -> None:
    raw_parts = name.split("/")
    if (
        not name
        or "\0" in name
        or "\\" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in raw_parts)
    ):
        raise ArtifactVisibilityError("artifact manifest contains an unsafe path")


def _read_regular(path: Path, metadata: os.stat_result) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ArtifactVisibilityError("artifact file changed during inspection")
            return stream.read()
    except OSError as exc:
        raise ArtifactVisibilityError("artifact file could not be read safely") from exc


def _walk(
    root: Path,
    path: Path,
    entries: list[ArtifactManifestEntry],
    inodes: set[tuple[int, int]],
    *,
    digest: bool,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArtifactVisibilityError("artifact entry could not be inspected") from exc
    relative = path.relative_to(root).as_posix()
    _validate_relative_name(relative)
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
    if stat.S_ISDIR(metadata.st_mode):
        entries.append(ArtifactManifestEntry(relative, "directory", 0, mode, None))
        try:
            children = sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
        except OSError as exc:
            raise ArtifactVisibilityError("artifact directory could not be inspected") from exc
        normalized: set[str] = set()
        for child in children:
            key = unicodedata.normalize("NFC", child.name)
            if key in normalized:
                raise ArtifactVisibilityError("artifact tree contains duplicate normalized names")
            normalized.add(key)
            _walk(root, child, entries, inodes, digest=digest)
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
    inode = (metadata.st_dev, metadata.st_ino)
    if inode in inodes:
        raise ArtifactVisibilityError("artifact tree contains duplicate filesystem aliases")
    inodes.add(inode)
    if not digest:
        entries.append(ArtifactManifestEntry(relative, "file", metadata.st_size, mode, None))
        return
    content = _read_regular(path, metadata)
    entries.append(ArtifactManifestEntry(relative, "file", len(content), mode, hashlib.sha256(content).hexdigest()))


def manifest_tree(
    root: Path,
    names: Sequence[str] | None = None,
    *,
    digest: bool = True,
) -> tuple[ArtifactManifestEntry, ...]:
    """Enumerate a tree, hashing every file unless ``digest`` is disabled.

    Reject symlinks, special files, normalized-name collisions, and duplicate inodes.
    A digest-free listing carries no ``sha256`` and is therefore only valid for
    enumerating a tree in order to delete it, never for attestation or storage.
    """
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise ArtifactVisibilityError("artifact manifest root is inaccessible") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ArtifactVisibilityError("artifact manifest root must be a real directory")
    entries: list[ArtifactManifestEntry] = []
    inodes: set[tuple[int, int]] = set()
    selected = sorted(names if names is not None else (child.name for child in root.iterdir()), key=os.fsencode)
    normalized: set[str] = set()
    for name in selected:
        _validate_relative_name(name)
        key = unicodedata.normalize("NFC", name)
        if key in normalized:
            raise ArtifactVisibilityError("artifact tree contains duplicate normalized names")
        normalized.add(key)
        path = root / name
        if path.exists() or path.is_symlink():
            _walk(root, path, entries, inodes, digest=digest)
    return tuple(entries)


def _manifest_payload(entries: tuple[ArtifactManifestEntry, ...]) -> dict[str, object]:
    return {"schema_version": _SCHEMA_VERSION, "entries": [asdict(entry) for entry in entries]}


def _parse_manifest(path: Path) -> tuple[ArtifactManifestEntry, ...]:
    raw_entries = _load_ledger(
        path,
        items_key="entries",
        message="artifact manifest schema is unsupported",
        required=True,
    )
    result: list[ArtifactManifestEntry] = []
    seen: set[str] = set()
    normalized: set[str] = set()
    for raw in raw_entries:
        entry = _manifest_entry_from_payload(raw, message="artifact manifest is malformed")
        if entry.path in seen:
            raise ArtifactVisibilityError("artifact manifest contains duplicate paths")
        normalized_path = unicodedata.normalize("NFC", entry.path)
        if normalized_path in normalized:
            raise ArtifactVisibilityError("artifact manifest contains duplicate normalized paths")
        seen.add(entry.path)
        normalized.add(normalized_path)
        result.append(entry)
    if [entry.path for entry in result] != sorted((entry.path for entry in result), key=os.fsencode):
        raise ArtifactVisibilityError("artifact manifest entries are not sorted")
    return tuple(result)


def _entries_belong_to(relative: str, entries: tuple[ArtifactManifestEntry, ...]) -> bool:
    prefix = f"{relative}/"
    return all(entry.path == relative or entry.path.startswith(prefix) for entry in entries)


def _entry_at(
    entries: Sequence[ArtifactManifestEntry],
    relative: str,
    *,
    kind: str | None = None,
) -> ArtifactManifestEntry | None:
    """Return the manifest entry at ``relative``, optionally of one kind only."""
    return next(
        (entry for entry in entries if entry.path == relative and (kind is None or entry.kind == kind)),
        None,
    )


def _copy_tree(source: Path, destination: Path, entries: tuple[ArtifactManifestEntry, ...]) -> None:
    destination.mkdir(parents=True, mode=0o700)
    for entry in entries:
        target = destination / entry.path
        if entry.kind == "directory":
            target.mkdir()
            continue
        _copy_regular_entry(
            source / entry.path,
            target,
            entry,
            preserve_times=True,
            changed_message="artifact file changed during copy",
        )
    for entry in reversed(entries):
        if entry.kind == "directory":
            directory = destination / entry.path
            os.chmod(directory, entry.mode)
            _fsync_directory(directory)
    _fsync_directory(destination)
    if manifest_tree(destination) != entries:
        raise ArtifactVisibilityError("artifact copy verification failed")


def _manifest_entry_from_payload(
    value: object,
    *,
    message: str = "artifact transfer intent is malformed",
) -> ArtifactManifestEntry:
    """Validate one persisted manifest entry payload into its dataclass."""
    if not isinstance(value, dict) or set(value) != {"path", "kind", "size", "mode", "sha256"}:
        raise ArtifactVisibilityError(message)
    relative = value["path"]
    kind = value["kind"]
    size = value["size"]
    mode = value["mode"]
    digest = value["sha256"]
    if (
        not isinstance(relative, str)
        or kind not in ("directory", "file")
        or type(size) is not int
        or type(mode) is not int
        or size < 0
        or not 0 <= mode <= 0o7777
        or (kind == "directory" and (size != 0 or digest is not None))
        or (kind == "file" and not _is_sha256(digest))
    ):
        raise ArtifactVisibilityError(message)
    _validate_relative_name(relative)
    return ArtifactManifestEntry(
        relative,
        cast(Literal["directory", "file"], kind),
        size,
        mode,
        cast(str | None, digest),
    )


def _deepest_first_directories(entries: Sequence[ArtifactManifestEntry]) -> list[ArtifactManifestEntry]:
    """Return the directory entries ordered so children always precede parents."""
    return sorted(
        (entry for entry in entries if entry.kind == "directory"),
        key=lambda item: item.path.count("/"),
        reverse=True,
    )


def _remove_owned_tree(path: Path, owner: Path) -> None:
    if path.parent != owner or path.is_symlink():
        raise ArtifactVisibilityError("refusing unsafe transaction cleanup")
    if not path.exists():
        return
    entries = manifest_tree(path, digest=False)
    for entry in entries:
        if entry.kind == "file":
            (path / entry.path).unlink()
    for entry in _deepest_first_directories(entries):
        (path / entry.path).rmdir()
    path.rmdir()
    _fsync_directory(owner)


def _validate_owner(path: Path, expected: dict[str, object]) -> None:
    owner = _load_json(path)
    if (
        set(owner) != {"schema_version", "workspace_key", "source", "git_common_dir"}
        or type(owner["schema_version"]) is not int
        or owner["schema_version"] != _SCHEMA_VERSION
        or not isinstance(owner["workspace_key"], str)
        or not isinstance(owner["source"], str)
        or not isinstance(owner["git_common_dir"], str)
        or owner != expected
    ):
        raise ArtifactVisibilityError("artifact owner metadata does not match the source")


def _manifest_is_subset(
    actual: tuple[ArtifactManifestEntry, ...],
    expected: tuple[ArtifactManifestEntry, ...],
) -> bool:
    expected_by_path = {entry.path: entry for entry in expected}
    return all(expected_by_path.get(entry.path) == entry for entry in actual)


def _copy_regular_entry(
    source: Path,
    target: Path,
    entry: ArtifactManifestEntry,
    *,
    changed_message: str = "artifact file changed during destination staging",
    preserve_times: bool = False,
) -> None:
    """Attest bytes before exclusively creating and fsyncing a manifested file."""
    metadata = source.lstat()
    content = _read_regular(source, metadata)
    if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
        raise ArtifactVisibilityError(changed_message)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, entry.mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fchmod(stream.fileno(), entry.mode)
        if preserve_times:
            os.utime(stream.fileno(), ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        os.fsync(stream.fileno())
