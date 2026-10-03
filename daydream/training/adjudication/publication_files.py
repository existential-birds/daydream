"""Validated publication bytes and owned-directory atomic installation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import AbstractSet, Any, Mapping

from daydream.archive.hydrate import HydrationError, PublicDestinationError
from daydream.json_utils import canonical_json
from daydream.trajectory import redact_text

# Refuse credential-shaped payloads rather than silently scrubbing publication.
_SECRET_SHAPES = re.compile(r"(?:hf_[0-9A-Za-z]{8,}|github_pat_[0-9A-Za-z_]{8,}|ghp_[0-9A-Za-z]{8,})")
_BINARY_STATE_FILES = frozenset({"index.db"})


def _scan_for_secrets(name: str, data: bytes) -> None:
    """Fail-closed secret scan (S1): refuse to upload credential-shaped payloads."""
    try:
        if name in _BINARY_STATE_FILES:
            # latin-1 maps every byte to a code point losslessly; the ASCII
            # secret shapes the regex targets are still detectable in SQLite pages.
            text = data.decode("latin-1")
        else:
            text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicDestinationError(f"{name}: payload is not valid UTF-8: {exc}") from exc
    hit = _SECRET_SHAPES.search(text)
    if hit is not None:
        raise PublicDestinationError(f"{name}: refusing to publish: credential-shaped value detected in payload")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty identifier")
    if "/" in value or "\\" in value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(f"{name} contains an invalid path character")
    _scan_for_secrets(name, value.encode("utf-8"))
    return value


def _validate_oid(value: Any, *, what: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise HydrationError(f"{what} is not a lowercase 40-hex commit OID")
    return value


def _validate_remote_path(path: Any, *, allowed: AbstractSet[str], what: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path or path.endswith("/"):
        raise ValueError(f"{what}: invalid remote path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or str(pure) != path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"{what}: invalid remote path")
    if path not in allowed:
        raise ValueError(f"{what}: remote path {path!r} is outside the exact allowlist")
    _scan_for_secrets(what, path.encode("utf-8"))
    return path


def _read_regular_file(path: Path, *, label: str) -> bytes:
    # Operator paths may legitimately live below platform aliases such as
    # macOS /tmp -> /private/tmp.  The declared leaf remains non-following:
    # state/bundle roots are checked by their callers and file leaves here.
    if path.is_symlink():
        raise PublicDestinationError(f"{label}: refusing symlinked input")
    if not path.is_file():
        raise FileNotFoundError(f"{label}: required regular file is missing")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise HydrationError(redact_text(f"{label}: cannot read input: {exc}")) from None


def _parse_json_object(name: str, data: bytes) -> dict[str, Any]:
    _scan_for_secrets(name, data)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return value


def _fresh_destination(path: Path, *, label: str) -> tuple[Path, Path]:
    requested = Path(path)
    if requested.exists() or requested.is_symlink():
        raise ValueError(f"{label} destination {requested} must not exist")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} destination parent must be an existing directory: {exc}") from None
    if not parent.is_dir():
        raise ValueError(f"{label} destination parent must be an existing directory")
    anchored = parent / requested.name
    if anchored.exists() or anchored.is_symlink():
        raise ValueError(f"{label} destination {requested} must not exist")
    return requested, anchored


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise HydrationError(f"installation staging path {path} is not a directory")
    return info.st_dev, info.st_ino


def _remove_owned_tree(path: Path, identity: tuple[int, int]) -> None:
    """Best-effort collision avoidance while the caller holds the owned inode.

    This is not a sandbox against a hostile same-parent writer: the pathname
    can still change between the identity check and recursive removal.
    """
    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == identity:
        shutil.rmtree(path, ignore_errors=True)


def _install_staging(
    destination: Path,
    files: Mapping[str, bytes],
    *,
    error_label: str,
    propagate: tuple[type[BaseException], ...] = (),
) -> None:
    """Atomically install ``files`` into a freshly staged ``destination`` tree."""
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    stage_fd: int | None = None
    owned_identity: tuple[int, int] | None = None
    installed = False
    try:
        stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stage_info = os.fstat(stage_fd)
        if not stat.S_ISDIR(stage_info.st_mode):
            raise HydrationError("installation staging descriptor is not a directory")
        owned_identity = (stage_info.st_dev, stage_info.st_ino)
        if _directory_identity(stage) != owned_identity:
            raise HydrationError("installation staging directory changed")
        for name, data in sorted(files.items()):
            target = stage / name
            with target.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        os.fsync(stage_fd)
        if _directory_identity(stage) != owned_identity:
            raise HydrationError("installation staging directory changed")
        os.replace(stage, destination)
        installed = True
        if _directory_identity(destination) != owned_identity:
            raise HydrationError("installation destination changed")
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception as exc:
        if owned_identity is not None:
            _remove_owned_tree(destination if installed else stage, owned_identity)
        if propagate and isinstance(exc, propagate):
            raise
        raise HydrationError(redact_text(f"{error_label}: {exc}")) from None
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
