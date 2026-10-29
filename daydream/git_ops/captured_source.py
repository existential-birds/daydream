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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from daydream.git_ops import immutable, process, queries, snapshot
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


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _runtime_artifact(path: str) -> bool:
    return path == ".daydream" or queries._is_untracked_runtime_artifact(path)


def _selector(path: str) -> None:
    relative = PurePosixPath(path)
    if (not path or not relative.parts or relative.is_absolute() or path != relative.as_posix()
            or ".." in relative.parts or ".git" in {p.lower() for p in relative.parts} or "\\" in path or "\0" in path):
        raise FrozenSourceRejected("invalid captured source selector")


def _native_entry(root: Path, path: str, gitlink: str | None = None) -> tuple[str, bytes, tuple[int, ...]] | None:
    """Read a confined leaf through no-follow directory descriptors."""
    _selector(path)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = path.split("/")
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        name = parts[-1]
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            mode, body = "120000", os.fsencode(os.readlink(name, dir_fd=descriptor))
        elif stat.S_ISDIR(before.st_mode):
            if gitlink is None:
                return None
            queries._require_git_path_confined(root, path)
            nested = root / path
            try:
                (nested / ".git").lstat()
            except FileNotFoundError:
                pass  # Uninitialized gitlinks retain their recorded commit.
            else:
                git_dir, common = immutable._source_git_dir(nested)
                head = immutable._source_head(git_dir, common)
                if len(head) != len(gitlink) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head):
                    raise GitError("captured gitlink HEAD metadata is malformed")
                gitlink = head
            mode, body = "160000", gitlink.encode()
        elif stat.S_ISREG(before.st_mode):
            if before.st_size > immutable._MAX_SOURCE_BYTES:
                raise GitError("captured source file exceeds the supported 256 MiB capture limit")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if _fingerprint(opened) != _fingerprint(before):
                    raise _UnstableSource(_RERUN)
                body = stream.read(immutable._MAX_SOURCE_BYTES + 1)
                if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(before):
                    raise _UnstableSource(_RERUN)
            if len(body) != before.st_size:
                raise _UnstableSource(_RERUN)
            mode = "100755" if before.st_mode & stat.S_IXUSR else "100644"
        else:
            raise FrozenSourceRejected("captured source entry is not a regular file, symlink or gitlink")
        after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if _fingerprint(before) != _fingerprint(after):
            raise _UnstableSource(_RERUN)
        return mode, body, _fingerprint(after)
    except (FileNotFoundError, NotADirectoryError):
        return None
    finally:
        os.close(descriptor)


def _population(source: Path, head_entries: list[_Entry]) -> tuple[list[str], dict[str, str]]:
    indexed = process._run_git(
        source, ["ls-files", "--stage", "-t", "-z"], capture_bytes=True, retries=0,
        error_context="local captured source inventory unavailable",
    ).stdout
    tracked = {name for _, _, name in head_entries}
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
        if tag == b"S" and _native_entry(source, name, gitlinks.get(name)) is None:
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
    immutable._copy_objects(common, target)
    after = {relative: _fingerprint(path.lstat()) for path, relative in immutable._object_files(common)}
    if before != after:
        raise _UnstableSource(_RERUN)


def _object(root: Path, kind: str, body: bytes, algorithm: str) -> str:
    raw = kind.encode() + b" " + str(len(body)).encode() + b"\0" + body
    oid = hashlib.new(algorithm, raw).hexdigest()
    path = root / ".git" / "objects" / oid[:2] / oid[2:]
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(zlib.compress(raw))
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


def _blobs(root: Path, oids: list[str], algorithm: str) -> Iterator[tuple[str, bytes]]:
    """Read bounded raw blob batches without filters, text conversion or checkout."""
    for offset in range(0, len(oids), 16):
        batch = oids[offset:offset + 16]
        metadata = process._run_git(root, ["cat-file", "--batch-check"], capture_bytes=True,
            input_text="\n".join(batch) + "\n", timeout=30,
            error_context="captured source blob metadata unavailable").stdout.splitlines()
        if len(metadata) != len(batch):
            raise GitError("captured source blob metadata is incomplete")
        sizes = []
        for oid, row in zip(batch, metadata, strict=True):
            fields = row.split()
            if (len(fields) != 3 or fields[:2] != [oid.encode(), b"blob"] or not fields[2].isdigit()
                    or int(fields[2]) > immutable._MAX_SOURCE_BYTES):
                raise GitError("captured source blob metadata exceeds the supported bounds")
            sizes.append(int(fields[2]))
        # `show` concatenates raw blobs in argument order, including duplicates.
        # Metadata input stays small; body output uses the subprocess byte bound.
        raw = process._run_git(root, ["show", "--no-ext-diff", "--no-textconv", "--no-color", *batch, "--"],
            capture_bytes=True, timeout=30, max_output_bytes=sum(sizes),
            error_context="captured source blobs unavailable").stdout
        cursor = 0
        for oid, size in zip(batch, sizes, strict=True):
            body = raw[cursor:cursor + size]
            cursor += size
            if len(body) != size or _blob_digest(body, algorithm) != oid:
                raise GitError("captured source blob integrity failed")
            yield oid, body
        if cursor != len(raw):
            raise GitError("captured source blob population integrity failed")


def _blob(root: Path, oid: str, algorithm: str) -> bytes:
    size = process._run_git(root, ['cat-file', '-s', oid], timeout=30,
                            error_context='captured source blob size unavailable').stdout.strip()
    if not size.isdecimal():
        raise GitError('captured source blob metadata is malformed')
    if int(size) > immutable._MAX_BLOB_BYTES:
        raise FrozenSourceRejected('captured source exceeds the bounded retrieval limit')
    body = process._run_git(
        root, ["cat-file", "blob", oid], capture_bytes=True, timeout=30,
        error_context="captured source blob unavailable", max_output_bytes=immutable._MAX_BLOB_BYTES + 4096,
    ).stdout
    if len(body) > immutable._MAX_BLOB_BYTES or _blob_digest(body, algorithm) != oid:
        raise GitError("captured source blob integrity failed")
    return body


def _materialize(root: Path, entries: list[_Entry], algorithm: str) -> None:
    regular = [entry for entry in entries if entry[0] in {"100644", "100755"}]
    bodies = _blobs(root, [oid for _, oid, _ in regular], algorithm)
    for (mode, _, name), (_, body) in zip(regular, bodies, strict=True):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        target.chmod(0o755 if mode == "100755" else 0o644)


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
                native = _native_entry(self.root, path)
            except OSError as exc:
                raise GitError("captured native source is unavailable") from exc
            if native is None or native[:2] != (mode, body):
                raise GitError("captured native source integrity failed")
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
            for _ in _blobs(self.root, oids, algorithm):
                pass
        expected = {name for mode, _, name in self._after if mode in {"100644", "100755"}}
        for mode, oid, name in self._after:
            if name not in expected:
                continue
            native = _native_entry(self.root, name)
            if native is None or native[0] != mode or _blob_digest(native[1], algorithm) != oid:
                raise GitError("captured native source integrity failed")
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
                if leaf.is_symlink():
                    raise GitError("captured native source contains an unexpected symlink")
                if name in names:
                    actual.add(relative)
        if actual != expected:
            raise GitError("captured native source population changed")


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
        try:
            with immutable.inert_git_context(destination):
                algorithm = "sha256" if len(head_sha) == 64 else "sha1"
                immutable._git(destination, "init", "--template=", "--object-format=" + algorithm, ".")
                _copy_objects(common, destination / ".git/objects")
                _validate_commit(destination, base_sha)
                _validate_commit(destination, head_sha)
                before = immutable._inventory(destination, base_sha, max_blob_bytes=immutable._MAX_SOURCE_BYTES)
                committed_after = immutable._inventory(destination, head_sha,
                                                       max_blob_bytes=immutable._MAX_SOURCE_BYTES)
                if any(_runtime_artifact(name) for _, _, name in (*before, *committed_after)):
                    raise GitError("captured source contains tracked generated Daydream artifacts")
                population: list[str] = []
                observed: dict[str, tuple[str, bytes, tuple[int, ...]] | None] = {}
                if local:
                    if immutable._source_head(git_dir, common) != head_sha:
                        raise _UnstableSource(_RERUN)
                    population, gitlinks = _population(source, committed_after)
                    after = []
                    total = 0
                    for path in population:
                        native = observed[path] = _native_entry(source, path, gitlinks.get(path))
                        if native is None:
                            continue
                        mode, body, _ = native
                        total += len(body)
                        if total > immutable._MAX_SOURCE_BYTES:
                            raise GitError("captured source population exceeds the supported 256 MiB limit")
                        oid = body.decode() if mode == "160000" else _object(destination, "blob", body, algorithm)
                        after.append((mode, oid, path))
                    tree = _tree(destination, after, algorithm)
                    commit = (f"tree {tree}\nparent {head_sha}\nauthor Daydream <source@daydream.invalid> 0 +0000\n"
                              "committer Daydream <source@daydream.invalid> 0 +0000\n\nCaptured local source\n")
                    after_revision = _object(destination, "commit", commit.encode(), algorithm)
                    after = immutable._inventory(destination, after_revision,
                                                 max_blob_bytes=immutable._MAX_SOURCE_BYTES)
                else:
                    after = committed_after
                    after_revision = head_sha
                _materialize(destination, after, algorithm)
                if local:
                    final_population, final_gitlinks = _population(source, committed_after)
                    if final_population != population or _directory_identity(source) != source_identity:
                        raise _UnstableSource(_RERUN)
                    for path in population:
                        if _native_entry(source, path, final_gitlinks.get(path)) != observed[path]:
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
        except (_UnstableSource, OSError) as exc:
            shutil.rmtree(destination)
            if attempt + 1 == _ATTEMPTS:
                raise SnapshotPreparationError(_RERUN) from exc
        except BaseException:
            shutil.rmtree(destination)
            raise
    raise AssertionError("capture retry loop exhausted")
