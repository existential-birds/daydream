"""One accepted local or committed source, independent of the user's checkout.

The before/after Git objects, native regular files, target inventory and lookup
identity belong to this owner. Native symlinks and submodules are deliberately
not materialized. Capturing is a bounded stability check, not a filesystem-wide
transaction. The caller owns the destination and its cleanup.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import zlib
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from uuid import uuid4

from daydream.git_ops import immutable, process, queries, snapshot
from daydream.git_ops.immutable import _fingerprint
from daydream.git_ops.models import GitError, PathAbsentError, SnapshotPreparationError
from daydream.git_ops.source import FrozenSourceRejected, _validate_commit

_ATTEMPTS = 3
_RERUN = "source changed during capture; stop concurrent writers and rerun the review"
_Entry = tuple[str, str, str]


class _UnstableSource(GitError):
    pass


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise GitError("captured source directory was replaced or is unavailable")
    return info.st_dev, info.st_ino


def _storage_matches(root: Path, identities: dict[str, tuple[int, int]]) -> bool:
    try:
        return all(_directory_identity(root / p) == identity for p, identity in identities.items())
    except (OSError, GitError):
        return False


def _runtime_artifact(path: str) -> bool:
    return path == ".daydream" or queries._is_untracked_runtime_artifact(path)


def _selector(path: str) -> None:
    relative = PurePosixPath(path)
    if (not path or not relative.parts or relative.is_absolute() or path != relative.as_posix()
            or ".." in relative.parts or ".git" in {p.lower() for p in relative.parts} or "\\" in path or "\0" in path):
        raise FrozenSourceRejected("invalid captured source selector")


@dataclass(frozen=True)
class _NativeEntry:
    mode: str
    oid: str
    size: int
    digest: str
    fingerprint: tuple[int, ...]
    metadata: bytes = b""
    spool_fingerprint: tuple[int, ...] | None = field(default=None, compare=False)


@contextmanager
def _regular_leaf(parent: int, name: str, before: os.stat_result, *, private: bool = False) -> Iterator[BinaryIO]:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError as exc:
        if private:
            raise immutable._StorageConflict("captured private spool is unavailable") from exc
        raise
    with os.fdopen(fd, "rb") as stream:
        failure = immutable._StorageConflict if private else _UnstableSource
        message = "captured private spool was replaced or changed" if private else _RERUN

        def validate_leaf() -> None:
            try:
                if (_fingerprint(os.fstat(stream.fileno())) != _fingerprint(before)
                        or _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False)) != _fingerprint(before)):
                    raise failure(message)
            except OSError as exc:
                if private:
                    raise immutable._StorageConflict("captured private spool is unavailable") from exc
                raise

        validate_leaf()
        try:
            yield stream
        except immutable._StorageConflict:
            raise
        except BaseException as error:
            if private:
                try:
                    validate_leaf()
                except immutable._StorageConflict as conflict:
                    raise conflict from error
            raise
        else:
            validate_leaf()


def _native_entry(root: Path, path: str, gitlink: str | None = None, *, algorithm: str = "sha1",
                  spool: Path | None = None, compare: Path | None = None,
                  compare_fingerprint: tuple[int, ...] | None = None,
                  consume: Callable[[bytes], object] | None = None, private: bool = True) -> _NativeEntry | None:
    """Hash confined regular leaves in fixed chunks; only tiny metadata retains bytes."""
    _selector(path)
    failure = immutable._StorageConflict if private else _UnstableSource
    message = "captured private spool was replaced or changed" if private else _RERUN
    found = False
    try:
        with immutable._parent(root, path, private=private) as (parent, name), ExitStack() as stack:
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            found = True
            if private and not stat.S_ISREG(before.st_mode):
                raise immutable._StorageConflict("captured private spool was replaced or changed")
            body = b""
            if stat.S_ISLNK(before.st_mode):
                mode, body = "120000", os.fsencode(os.readlink(name, dir_fd=parent))
            elif stat.S_ISDIR(before.st_mode):
                if gitlink is None:
                    return None
                queries._require_git_path_confined(root, path)
                nested = root / path
                try:
                    (nested / ".git").lstat()
                except FileNotFoundError:
                    pass
                else:
                    git_dir, common = immutable._source_git_dir(nested)
                    head = immutable._source_head(git_dir, common)
                    if len(head) != len(gitlink) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head):
                        raise GitError("captured gitlink HEAD metadata is malformed")
                    gitlink = head
                mode, body = "160000", gitlink.encode()
            elif stat.S_ISREG(before.st_mode):
                if before.st_size > immutable._MAX_SOURCE_BYTES:
                    if private:
                        raise immutable._StorageConflict("captured private spool exceeds the accepted size")
                    raise GitError("captured source file exceeds the supported 256 MiB capture limit")
                mode = "100755" if before.st_mode & stat.S_IXUSR else "100644"
                stream = stack.enter_context(_regular_leaf(parent, name, before, private=private))
                output = stack.enter_context(immutable._exclusive_file(spool, path)) if spool else None
                other = None
                spool_fingerprint = None
                if compare is not None:
                    try:
                        other_parent, other_name = stack.enter_context(immutable._parent(compare, path))
                        info = os.stat(other_name, dir_fd=other_parent, follow_symlinks=False)
                    except OSError as exc:
                        raise immutable._StorageConflict("captured private spool is unavailable") from exc
                    if _fingerprint(info) != compare_fingerprint:
                        raise immutable._StorageConflict("captured private spool was replaced or changed")
                    if not stat.S_ISREG(info.st_mode) or info.st_size != before.st_size:
                        raise failure(message)
                    spool_fingerprint = _fingerprint(info)
                    other = stack.enter_context(_regular_leaf(other_parent, other_name, info, private=True))
                digest = hashlib.sha256()
                oid = hashlib.new(algorithm, b"blob " + str(before.st_size).encode() + b"\0")
                remaining = before.st_size
                while remaining:
                    chunk = stream.read(min(immutable._CHUNK_BYTES, remaining))
                    if not chunk:
                        raise failure(message)
                    remaining -= len(chunk)
                    digest.update(chunk)
                    oid.update(chunk)
                    if output is not None:
                        output.write(chunk)
                    if other is not None and other.read(len(chunk)) != chunk:
                        if _fingerprint(os.fstat(other.fileno())) != compare_fingerprint:
                            raise immutable._StorageConflict("captured private spool was replaced or changed")
                        raise failure(message)
                    if consume is not None:
                        consume(chunk)
                if stream.read(1) or (other is not None and other.read(1)):
                    raise failure(message)
                if output is not None:
                    os.fchmod(output.fileno(), 0o755 if mode == "100755" else 0o644)
                    output.flush()
                    spool_fingerprint = _fingerprint(os.fstat(output.fileno()))
                return _NativeEntry(mode, oid.hexdigest(), before.st_size, digest.hexdigest(), _fingerprint(before),
                                    spool_fingerprint=spool_fingerprint)
            else:
                raise FrozenSourceRejected("captured source entry is not a regular file, symlink or gitlink")
            after = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if _fingerprint(before) != _fingerprint(after):
                raise failure(message)
            oid_text = body.decode() if mode == "160000" else _blob_digest(body, algorithm)
            return _NativeEntry(mode, oid_text, len(body), hashlib.sha256(body).hexdigest(), _fingerprint(after), body)
    except immutable._SourceChanged as exc:
        raise _UnstableSource(_RERUN) from exc
    except (FileNotFoundError, NotADirectoryError) as exc:
        if private:
            raise immutable._StorageConflict("captured private spool is unavailable") from exc
        if found:
            raise _UnstableSource(_RERUN) from exc
        return None


def _population(source: Path, head_entries: list[_Entry]) -> tuple[list[str], dict[str, str]]:
    indexed = process._run_git(
        source, ["ls-files", "--stage", "-t", "-z"], capture_bytes=True, retries=0,
        error_context="local captured source inventory unavailable",
    ).stdout
    tracked: set[str] = set()
    gitlinks = {name: oid for mode, oid, name in head_entries if mode == "160000"}
    for entry in indexed.split(b"\0"):
        if not entry:
            continue
        metadata, _, name_bytes = entry.partition(b"\t")
        tag, mode, oid, _ = metadata.split()
        name = os.fsdecode(name_bytes)
        tracked.add(name)
        if mode == b"160000":
            gitlinks[name] = oid.decode()
        if tag == b"S" and _native_entry(source, name, gitlinks.get(name), private=False) is None:
            raise SnapshotPreparationError(
                "local capture has omitted skip-worktree files; materialize those working contents and rerun the review"
            )
    paths = tracked | set(queries._list_untracked_z(source))
    selected = sorted((p for p in paths if not _runtime_artifact(p)), key=os.fsencode)
    if len(selected) > immutable._MAX_FILES:
        raise GitError("captured source file count exceeds the supported size")
    for path in selected:
        _selector(path)
    return selected, gitlinks


def _copy_objects(common: Path, target: Path) -> None:
    before = {relative: _fingerprint(path.lstat()) for path, relative in immutable._object_files(common)}
    try:
        immutable._copy_objects(common, target)
    except immutable._SourceChanged as exc:
        raise _UnstableSource(_RERUN) from exc
    after = {relative: _fingerprint(path.lstat()) for path, relative in immutable._object_files(common)}
    if before != after:
        raise _UnstableSource(_RERUN)


def _verify_object(root: Path, oid: str, raw_size: int, algorithm: str) -> None:
    """Verify an existing loose object through pinned descriptors and bounded inflation."""
    relative = f".git/objects/{oid[:2]}/{oid[2:]}"
    try:
        with immutable._parent(root, relative) as (parent, name):
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise immutable._StorageConflict("captured source object conflicts with existing bytes")
            with _regular_leaf(parent, name, info, private=True) as stream:
                decoder = zlib.decompressobj()
                digest = hashlib.new(algorithm)
                size = 0
                while chunk := stream.read(immutable._CHUNK_BYTES):
                    while True:
                        raw = decoder.decompress(chunk, immutable._CHUNK_BYTES)
                        size += len(raw)
                        if size > raw_size or decoder.unused_data:
                            raise immutable._StorageConflict("captured source object conflicts with existing bytes")
                        digest.update(raw)
                        chunk = decoder.unconsumed_tail
                        if not chunk and len(raw) < immutable._CHUNK_BYTES:
                            break
                if not decoder.eof or size != raw_size or digest.hexdigest() != oid:
                    raise immutable._StorageConflict("captured source object conflicts with existing bytes")
    except (zlib.error, OSError) as exc:
        raise immutable._StorageConflict("captured source object conflicts with existing bytes") from exc


def _object(root: Path, kind: str, body: bytes, algorithm: str) -> str:
    raw = kind.encode() + b" " + str(len(body)).encode() + b"\0" + body
    oid = hashlib.new(algorithm, raw).hexdigest()
    relative = f".git/objects/{oid[:2]}/{oid[2:]}"
    try:
        with immutable._exclusive_file(root, relative) as output:
            output.write(zlib.compress(raw))
    except immutable._PrivateFileExists:
        _verify_object(root, oid, len(raw), algorithm)
    return oid


def _tree(root: Path, entries: list[_Entry], algorithm: str) -> str:
    children: dict[str, list[_Entry]] = {}
    leaves: list[tuple[bytes, bytes]] = []
    for mode, oid, name in entries:
        first, separator, rest = name.partition("/")
        if separator:
            children.setdefault(first, []).append((mode, oid, rest))
        else:
            leaves.append((os.fsencode(name), mode.encode() + b" " + os.fsencode(name) + b"\0" + bytes.fromhex(oid)))
    for name, descendants in children.items():
        oid = _tree(root, descendants, algorithm)
        leaves.append((os.fsencode(name) + b"/", b"40000 " + os.fsencode(name) + b"\0" + bytes.fromhex(oid)))
    return _object(root, "tree", b"".join(raw for _, raw in sorted(leaves)), algorithm)


def _blob_digest(body: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


class _BlobReader:
    """One raw reader per operation, with independent request and payload bounds."""

    def __init__(self, transport: process._GitStream, algorithm: str, *, total: int, count: int,
                 limit: int = immutable._MAX_SOURCE_BYTES) -> None:
        self.transport, self.algorithm = transport, algorithm
        self.remaining, self.requests, self.limit = total, count, limit

    def read(self, oid: str, consume: Callable[[bytes], object]) -> None:
        if self.requests <= 0 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
            raise GitError("captured source blob request exceeds the supported bounds")
        self.requests -= 1
        self.transport.send(oid.encode() + b"\n")
        fields = self.transport.line().split()
        if (len(fields) != 3 or fields[:2] != [oid.encode(), b"blob"] or not fields[2].isdigit()
                or len(fields[2]) > 12):
            raise GitError("captured source blob metadata is malformed or missing")
        size = int(fields[2])
        if size > self.limit:
            raise FrozenSourceRejected("captured source exceeds the bounded retrieval limit")
        if size > self.remaining:
            raise GitError("captured source blob population exceeds the supported size")
        self.remaining -= size
        digest = hashlib.new(self.algorithm, b"blob " + str(size).encode() + b"\0")
        remaining = size
        while remaining:
            chunk = self.transport.read(min(immutable._CHUNK_BYTES, remaining))
            if not chunk:
                raise GitError("captured source blob is truncated")
            remaining -= len(chunk)
            digest.update(chunk)
            consume(chunk)
        if self.transport.read(1) != b"\n" or digest.hexdigest() != oid:
            raise GitError("captured source blob integrity failed")


@contextmanager
def _blob_reader(root: Path, algorithm: str, *, total: int, count: int,
                 limit: int = immutable._MAX_SOURCE_BYTES) -> Iterator[_BlobReader]:
    with process._stream_git(root, ["cat-file", "--batch"]) as transport:
        yield _BlobReader(transport, algorithm, total=total, count=count, limit=limit)


def _blob(root: Path, oid: str, algorithm: str) -> bytes:
    body = bytearray()
    with _blob_reader(root, algorithm, total=immutable._MAX_BLOB_BYTES, count=1,
                      limit=immutable._MAX_BLOB_BYTES) as reader:
        reader.read(oid, body.extend)
    return bytes(body)


def _materialize(root: Path, entries: list[_Entry], algorithm: str) -> None:
    regular = [entry for entry in entries if entry[0] in {"100644", "100755"}]
    with _blob_reader(root, algorithm, total=immutable._MAX_SOURCE_BYTES, count=len(regular)) as reader:
        for mode, oid, name in regular:
            with immutable._exclusive_file(root, name) as stream:
                reader.read(oid, stream.write)
                os.fchmod(stream.fileno(), 0o755 if mode == "100755" else 0o644)


def _spooled_blob(root: Path, path: str, accepted: _NativeEntry, algorithm: str) -> None:
    """Compress the sole native spool into an exclusively created loose object."""
    relative = f".git/objects/{accepted.oid[:2]}/{accepted.oid[2:]}"
    header = b"blob " + str(accepted.size).encode() + b"\0"
    if os.path.lexists(root / relative):
        _verify_object(root, accepted.oid, len(header) + accepted.size, algorithm)
        return
    temporary = relative + "." + uuid4().hex
    with immutable._exclusive_file(root, temporary) as output:
        compressor = zlib.compressobj()
        output.write(compressor.compress(header))

        def compress(chunk: bytes) -> None:
            output.write(compressor.compress(chunk))

        try:
            current = _native_entry(root, path, algorithm=algorithm, consume=compress)
        except _UnstableSource as exc:
            raise immutable._StorageConflict("captured private spool was replaced or changed") from exc
        if current is not None and current.fingerprint != accepted.spool_fingerprint:
            raise immutable._StorageConflict("captured private spool was replaced or changed")
        if (current is None or current.mode != accepted.mode or current.oid != accepted.oid
                or current.digest != accepted.digest or current.size != accepted.size):
            raise immutable._StorageConflict("captured private spool integrity failed")
        output.write(compressor.flush())
        output.flush()
        owned = os.fstat(output.fileno())
        with immutable._parent(root, relative) as (parent, name):
            # link is an atomic no-replace publication. This link is private;
            # no original source inode is borrowed. Competing bytes survive.
            try:
                os.link(Path(temporary).name, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            except FileExistsError:
                _verify_object(root, accepted.oid, len(header) + accepted.size, algorithm)
    with immutable._parent(root, temporary) as (parent, name):
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (info.st_dev, info.st_ino) != (owned.st_dev, owned.st_ino):
            raise immutable._StorageConflict("captured private object temporary was replaced")
        os.unlink(name, dir_fd=parent)


@dataclass(frozen=True)
class CapturedSource:
    """Accepted source identity and bounded before/after reads without live fallback."""

    root: Path
    base_sha: str
    head_sha: str
    diff: str
    files: tuple[str, ...]
    source_id: str | None
    _after_revision: str = field(repr=False)
    _before: tuple[_Entry, ...] = field(repr=False)
    _after: tuple[_Entry, ...] = field(repr=False)
    _directories: tuple[tuple[int, int], ...] = field(repr=False)
    _metadata: tuple[bytes, ...] = field(repr=False)

    def revision_for(self, side: str) -> str:
        if side not in {"old", "new"}:
            raise FrozenSourceRejected("invalid captured source side")
        return self.base_sha if side == "old" else self.source_id or self.head_sha

    def _check_storage(self) -> None:
        try:
            current = tuple(_directory_identity(self.root / p) for p in (".", ".git", ".git/objects"))
        except OSError as exc:
            raise GitError("captured source storage is unavailable") from exc
        if current != self._directories:
            raise GitError("captured source storage identity changed")

    def remove(self) -> None:
        """Remove only the independently pinned storage, preserving replacements."""
        self._check_storage()
        shutil.rmtree(self.root)

    def _check_metadata(self) -> None:
        current = tuple(immutable._regular(self.root / ".git" / name, limit=16 * 1024 * 1024)
                        for name in ("HEAD", "config", "index"))
        if current[:2] != self._metadata[:2]:
            raise GitError("captured source Git metadata integrity failed")
        if current[2] == self._metadata[2]:
            return
        # Native Git inspection may refresh stat/cache fields. Its index must
        # retain the accepted paths, blobs, modes, stages and visibility flags.
        expected = b"".join(f"H {mode} {oid} 0\t{name}\0".encode()
                            for mode, oid, name in sorted(self._after, key=lambda entry: os.fsencode(entry[2])))
        with immutable.inert_git_context(self.root):
            index = process._run_git(self.root, ["ls-files", "-t", "-v", "--stage", "-z"], capture_bytes=True,
                max_output_bytes=16 * 1024 * 1024, error_context="captured source index unavailable").stdout
            if index != expected:
                raise GitError("captured source index integrity failed")
            # Intent-to-add on an accepted empty file has identical ls-files
            # rows, but hides that file from this independently checked view.
            process._run_git(self.root, ["diff-index", "--cached", "--quiet", "--ita-invisible-in-index",
                "--ignore-submodules=none", "--no-ext-diff", "--no-textconv", self._after_revision, "--"],
                max_output_bytes=4096, error_context="captured source index integrity failed")

    def read(self, side: str, path: str) -> tuple[str, bytes]:
        self.revision_for(side)
        _selector(path)
        self._check_storage()
        self._check_metadata()
        entries = self._before if side == "old" else self._after
        selected = next(((mode, oid) for mode, oid, name in entries if name == path), None)
        if selected is None:
            if any(name.startswith(path + "/") for _, _, name in entries):
                raise FrozenSourceRejected("captured source is not a regular blob")
            raise PathAbsentError("path is absent from the accepted captured source")
        mode, oid = selected
        if mode not in {"100644", "100755"}:
            raise FrozenSourceRejected("captured source is not a regular blob")
        with immutable.inert_git_context(self.root):
            body = _blob(self.root, oid, "sha256" if len(self.head_sha) == 64 else "sha1")
        if side == "new":
            try:
                native = _native_entry(self.root, path, algorithm="sha256" if len(self.head_sha) == 64 else "sha1")
            except OSError as exc:
                raise GitError("captured native source is unavailable") from exc
            if native is None or (native.mode, native.oid) != (mode, oid):
                raise immutable._StorageConflict("captured native source integrity failed")
        return oid, body

    def validate(self) -> None:
        """Reject changed native bytes, population, directories or Git evidence."""
        self._check_storage()
        self._check_metadata()
        for _ in immutable._object_files(self.root / ".git"):
            pass
        with immutable.inert_git_context(self.root):
            if tuple(immutable._inventory(
                    self.root, self.base_sha, max_blob_bytes=immutable._MAX_SOURCE_BYTES)) != self._before:
                raise GitError("captured before-source inventory integrity failed")
            if tuple(immutable._inventory(
                    self.root, self._after_revision, max_blob_bytes=immutable._MAX_SOURCE_BYTES)) != self._after:
                raise GitError("captured after-source inventory integrity failed")
            if immutable._git(self.root, "rev-parse", "HEAD") != self._after_revision:
                raise GitError("captured native source revision changed")
            algorithm = "sha256" if len(self.head_sha) == 64 else "sha1"
            oids = list(dict.fromkeys(oid for mode, oid, _ in (*self._before, *self._after) if mode != "160000"))
            with _blob_reader(self.root, algorithm, total=2 * immutable._MAX_SOURCE_BYTES,
                              count=len(oids)) as reader:
                for oid in oids:
                    reader.read(oid, lambda chunk: None)
        expected = {name for mode, _, name in self._after if mode in {"100644", "100755"}}
        for mode, oid, name in self._after:
            if name not in expected:
                continue
            native = _native_entry(self.root, name, algorithm=algorithm)
            if native is None or native.mode != mode or native.oid != oid:
                raise immutable._StorageConflict("captured native source integrity failed")
        actual: set[str] = set()
        for directory, dirs, names in os.walk(self.root, followlinks=False):
            parent = Path(directory)
            if parent == self.root:
                dirs[:] = [name for name in dirs if name != ".git"]
            for name in [*dirs, *names]:
                leaf = parent / name
                relative = leaf.relative_to(self.root).as_posix()
                if _runtime_artifact(relative):
                    if name in dirs:
                        dirs.remove(name)
                    continue
                if name in names or leaf.is_symlink():
                    actual.add(relative)
        unexpected = actual - expected
        if unexpected:
            # Reviewer checks may leave output capture would exclude. Only accepted
            # .gitignore rules apply, never info/exclude in this writable Git directory.
            with immutable.inert_git_context(self.root):
                unexpected -= set(queries._decode_nul_paths(process._run_git(self.root,
                    ["ls-files", "--others", "--ignored", "--exclude-per-directory=.gitignore", "-z"],
                    timeout=30, capture_bytes=True, max_output_bytes=immutable._MAX_SOURCE_BYTES,
                    error_context="captured source ignored inventory unavailable").stdout))
        if changed := unexpected | (expected - actual):
            raise GitError(f"captured native source population changed at {min(changed)!r}")


def capture_source(
    source: Path, destination: Path, *, base_sha: str, head_sha: str, local: bool,
    exclude: list[str] | None = None,
) -> CapturedSource:
    """Capture pinned committed bytes or a stable net base-to-working-tree change.

    Only the new private destination is written. Local retries compare accepted
    population, file identity and bytes around actual reading/materialization.
    """
    violations = snapshot._snapshot_git_environment_violations()
    if violations:
        raise SnapshotPreparationError("source capture refuses inherited Git routing/configuration overrides")
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.resolve() != destination or destination.exists() or destination.is_symlink():
        raise SnapshotPreparationError("captured source destination must be absent without symlink parents")
    snapshot._require_disjoint_snapshot_paths(source, destination)
    git_dir, common = immutable._source_git_dir(source)
    source_identity = _directory_identity(source)
    scope = sorted(set(p.rstrip("/") for p in exclude or []))
    for attempt in range(_ATTEMPTS):
        destination.mkdir(mode=0o700, parents=True)
        owned_directories = {".": _directory_identity(destination)}
        try:
            with immutable.inert_git_context(destination):
                algorithm = "sha256" if len(head_sha) == 64 else "sha1"
                immutable._git(destination, "init", "--template=", "--object-format=" + algorithm, ".")
                owned_directories.update((p, _directory_identity(destination / p)) for p in (".git", ".git/objects"))
                _copy_objects(common, destination / ".git/objects")
                _validate_commit(destination, base_sha)
                _validate_commit(destination, head_sha)
                before = immutable._inventory(destination, base_sha, max_blob_bytes=immutable._MAX_SOURCE_BYTES)
                committed_after = immutable._inventory(destination, head_sha,
                                                       max_blob_bytes=immutable._MAX_SOURCE_BYTES)
                if any(_runtime_artifact(name) for _, _, name in (*before, *committed_after)):
                    raise GitError("captured source contains tracked generated Daydream artifacts")
                population: list[str] = []
                observed: dict[str, _NativeEntry | None] = {}
                established = {oid for mode, oid, _ in (*before, *committed_after) if mode != "160000"}
                if local:
                    if immutable._source_head(git_dir, common) != head_sha:
                        raise _UnstableSource(_RERUN)
                    population, gitlinks = _population(source, committed_after)
                    after = []
                    total = 0
                    for path in population:
                        native = observed[path] = _native_entry(source, path, gitlinks.get(path),
                                                              algorithm=algorithm, spool=destination, private=False)
                        if native is None:
                            continue
                        total += native.size
                        if total > immutable._MAX_SOURCE_BYTES:
                            raise GitError("captured source population exceeds the supported 256 MiB limit")
                        if native.mode in {"100644", "100755"}:
                            if native.oid not in established:
                                _spooled_blob(destination, path, native, algorithm)
                        elif native.mode == "120000":
                            _object(destination, "blob", native.metadata, algorithm)
                        established.add(native.oid)
                        after.append((native.mode, native.oid, path))
                    tree = _tree(destination, after, algorithm)
                    commit = (f"tree {tree}\nparent {head_sha}\nauthor Daydream <source@daydream.invalid> 0 +0000\n"
                              "committer Daydream <source@daydream.invalid> 0 +0000\n\nCaptured local source\n")
                    after_revision = _object(destination, "commit", commit.encode(), algorithm)
                    after = immutable._inventory(destination, after_revision,
                                                 max_blob_bytes=immutable._MAX_SOURCE_BYTES)
                else:
                    after = committed_after
                    after_revision = head_sha
                if not local:
                    _materialize(destination, after, algorithm)
                if local:
                    final_population, final_gitlinks = _population(source, committed_after)
                    if final_population != population or _directory_identity(source) != source_identity:
                        raise _UnstableSource(_RERUN)
                    for path in population:
                        accepted = observed[path]
                        spool_fingerprint = accepted.spool_fingerprint if accepted else None
                        final = _native_entry(source, path, final_gitlinks.get(path), algorithm=algorithm,
                            compare=destination if spool_fingerprint is not None else None,
                            compare_fingerprint=spool_fingerprint, private=False)
                        if final != accepted:
                            raise _UnstableSource(_RERUN)
                    if (_population(source, committed_after)[0] != population
                            or immutable._source_head(git_dir, common) != head_sha
                            or _directory_identity(source) != source_identity):
                        raise _UnstableSource(_RERUN)
                (destination / ".git/HEAD").write_text(after_revision + "\n")
                immutable._git(destination, "read-tree", after_revision)
                paths = ["--", ".", ":(exclude).daydream", ":(exclude).review-output.md",
                         *(f":(exclude){path}" for path in scope)]
                diff = process._run_git(destination, ["--attr-source=" + after_revision, "diff", "--no-ext-diff",
                    "--no-textconv", "--no-color", "--find-renames", "--src-prefix=a/", "--dst-prefix=b/",
                    base_sha, after_revision, *paths], timeout=30, error_context="captured source diff unavailable",
                    max_output_bytes=immutable._MAX_SOURCE_BYTES).stdout
                files = queries._decode_nul_paths(process._run_git(destination,
                    ["diff", "--find-renames", "--name-only", "-z", base_sha, after_revision, *paths],
                    capture_bytes=True, timeout=30).stdout)
                identity = {"version": 1, "base": base_sha, "before": before, "after": after,
                            "population": population, "exclude": scope}
                source_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() if local else None
                captured = CapturedSource(destination, base_sha, head_sha, diff, tuple(files), source_id,
                    after_revision,
                    tuple(before), tuple(after), tuple(_directory_identity(destination / p)
                        for p in (".", ".git", ".git/objects")),
                    tuple(immutable._regular(destination / ".git" / name, limit=16 * 1024 * 1024)
                        for name in ("HEAD", "config", "index")))
                captured.validate()
                return captured
        except immutable._StorageConflict as conflict:
            conflict.reraise()
        except (_UnstableSource, OSError) as exc:
            if not _storage_matches(destination, owned_directories):
                raise GitError("captured source storage was replaced; preserving competing bytes") from exc
            shutil.rmtree(destination)
            if attempt + 1 == _ATTEMPTS:
                raise SnapshotPreparationError(_RERUN) from exc
        except BaseException:
            if not _storage_matches(destination, owned_directories):
                raise
            shutil.rmtree(destination)
            raise
    raise AssertionError("capture retry loop exhausted")
