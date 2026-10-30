"""Inert object-only preparation for the trusted immutable embedding contract.

No command opens the supplied repository: only regular object bytes and HEAD are
read. A fresh empty-template repository owns all subsequent Git reads. Checkout
is implemented from blobs, so target hooks, filters and configuration never run.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import BinaryIO

from daydream.git_ops import process
from daydream.git_ops.models import GitError
from daydream.immutable_review import ImmutableReviewInput
from daydream.repository_paths import git_observed_path_is_confined

_CONTEXT: ContextVar[tuple[Path, dict[str, str]] | None] = ContextVar("immutable_git_context", default=None)
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_BLOB_BYTES = 8 * 1024 * 1024
_MAX_OBJECT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_FILES = 100_000


@contextmanager
def inert_git_context(repo: Path, environment: Mapping[str, str] | None = None) -> Iterator[None]:
    """Scope sanitized Git reads to one fresh host-owned repository subtree."""
    env = {
        "PATH": os.defpath if environment is None else environment.get("PATH", os.defpath),
        "HOME": str(repo), "XDG_CONFIG_HOME": str(repo / ".inert-config"),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C",
        "GIT_ATTR_NOSYSTEM": "1",
        # Git's checked allocations include a trailing NUL for object contents.
        "GIT_ALLOC_LIMIT": str(_MAX_OBJECT_BYTES + 1),
    }
    token = _CONTEXT.set((repo.resolve(), env))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def git_environment(repo: Path) -> dict[str, str] | None:
    """Return the scoped environment only for the independently prepared source."""
    current = _CONTEXT.get()
    if current is not None and repo.resolve().is_relative_to(current[0]):
        return dict(current[1])
    return None


def redaction_environment() -> Mapping[str, str]:
    """Inert diagnostics never consult unrelated GitHub credential variables."""
    current = _CONTEXT.get()
    return os.environ if current is None else current[1]


def immutable_review_diff(repo: Path, base_sha: str, head_sha: str) -> str:
    """Decode the review diff using captured-head attributes and inert configuration."""
    args = _diff_args(base_sha, head_sha)
    with _immutable_diff_source(repo, base_sha, head_sha) as view:
        diff = process._run_git(
            view, ["--attr-source=" + head_sha, "diff", *args], timeout=30,
            error_context="immutable review diff unavailable",
            max_output_bytes=_MAX_SOURCE_BYTES,
        ).stdout
    if len(diff.encode()) > _MAX_SOURCE_BYTES:
        raise GitError("immutable review diff exceeds the supported size")
    return diff


def _diff_args(base_sha: str, head_sha: str) -> list[str]:
    if not all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha) for sha in (base_sha, head_sha)):
        raise GitError("immutable diff requires full commit IDs")
    return ["--no-ext-diff", "--no-textconv", "--no-color", "--src-prefix=a/", "--dst-prefix=b/",
            base_sha, head_sha, "--"]


@contextmanager
def _immutable_diff_source(repo: Path, base_sha: str, head_sha: str) -> Iterator[Path]:
    """Borrow object bytes in fresh storage, excluding source configuration."""
    marker = repo / ".git"
    common = _source_git_dir(repo)[1] if marker.exists() or marker.is_symlink() else repo
    objects = _object_store(common).resolve()
    for _ in _object_files(common):
        pass
    if "\n" in str(objects):
        raise GitError("immutable object directory path is unsupported")
    current = _CONTEXT.get()
    with tempfile.TemporaryDirectory(prefix="daydream-immutable-diff-") as directory:
        view = Path(directory).resolve()
        with inert_git_context(view, None if current is None else current[1]):
            _git(view, "init", "--template=", "--object-format=" +
                 ("sha256" if len(head_sha) == 64 else "sha1"), ".")
            (view / ".git/objects/info/alternates").write_text(str(objects) + "\n")
            _validate_object_population(view)
            # Both sides can expand into the patch, including deleted blobs.
            _inventory(view, base_sha)
            _inventory(view, head_sha)
            # Callers select head-tree attributes explicitly: a two-commit diff
            # does not use index attributes when this repository has no HEAD.
            yield view


@contextmanager
def _regular_file(path: Path, *, limit: int) -> Iterator[tuple[BinaryIO, int]]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise GitError("immutable source evidence must be a bounded regular file")
            yield stream, info.st_size
    except OSError as exc:
        raise GitError("immutable source evidence is unavailable") from exc


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class _SourceChanged(GitError):
    """A confined source changed during copying; callers may retry a fresh capture."""


class _StorageConflict(GitError):
    """Private storage contains competing bytes that cleanup must preserve."""


class _PrivateFileExists(_StorageConflict):
    """An existing leaf may be verified only by the object publisher."""


# Payload buffers are independent of the admitted leaf/store size.
_CHUNK_BYTES = 1024 * 1024


@contextmanager
def _parent(root: Path, relative: str, *, create: bool = False,
            private: bool = True) -> Iterator[tuple[int, str]]:
    """Pin every parent and reject symlink/replacement races before returning."""
    descriptors = [os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)]
    bindings: list[tuple[int, str, int]] = []
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            parent = descriptors[-1]
            if create:
                try:
                    os.mkdir(part, dir_fd=parent)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            descriptors.append(child)
            bindings.append((parent, part, child))
        yield descriptors[-1], parts[-1]
        failure = _StorageConflict if private else _SourceChanged
        for parent, part, child in bindings:
            info = os.stat(part, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(child)
            if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
                raise failure("immutable source parent changed during reading")
        info, opened = root.lstat(), os.fstat(descriptors[0])
        if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
            raise failure("immutable source root changed during reading")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextmanager
def _exclusive_file(root: Path, relative: str) -> Iterator[BinaryIO]:
    """Create private bytes without replacing a competing leaf; preserve replacements."""
    with _parent(root, relative, create=True) as (parent, name):
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        except FileExistsError as exc:
            raise _PrivateFileExists("private source creation conflicts with existing bytes") from exc
        with os.fdopen(fd, "wb") as stream:
            owned = os.fstat(stream.fileno())
            try:
                yield stream
                stream.flush()
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
                    raise _StorageConflict("private source file was replaced")
            except BaseException:
                try:
                    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                        os.unlink(name, dir_fd=parent)
                except FileNotFoundError:
                    pass
                raise


def _regular(path: Path, *, limit: int) -> bytes:
    with _regular_file(path, limit=limit) as (stream, size):
        content = stream.read(size + 1)
    if len(content) > limit:
        raise GitError("immutable source evidence exceeds the supported size")
    return content


def _source_git_dir(source: Path) -> tuple[Path, Path]:
    marker = source / ".git"
    if marker.is_symlink():
        raise GitError("immutable source Git directory cannot be a symlink")
    if marker.is_dir():
        git_dir = marker
    else:
        pointer = _regular(marker, limit=4096).decode().strip()
        if not pointer.startswith("gitdir: "):
            raise GitError("immutable source Git pointer is malformed")
        git_dir = (source / pointer.removeprefix("gitdir: ")).resolve(strict=True)
    common_file = git_dir / "commondir"
    common = (
        (git_dir / _regular(common_file, limit=4096).decode().strip()).resolve(strict=True)
        if common_file.exists() else git_dir
    )
    if not common.is_dir() or common.is_symlink():
        raise GitError("immutable source object directory is unavailable")
    return git_dir, common


def _source_head(git_dir: Path, common: Path) -> str:
    # Git trims trailing ASCII whitespace without changing Unicode ref bytes.
    value = _regular(git_dir / "HEAD", limit=4096).decode().rstrip(" \t\r\n\v\f")
    if value.startswith("ref: "):
        ref = value.removeprefix("ref: ")
        if (not ref.startswith("refs/") or re.search(r"[\x00-\x20\x7f\\]", ref)
                or not git_observed_path_is_confined(common, ref)):
            raise GitError("immutable source HEAD reference is malformed")
        loose = common / ref
        if loose.exists():
            value = _regular(loose, limit=4096).decode().strip()
        else:
            matches = [oid for oid, _, packed_ref in (
                line.partition(" ") for line in _regular(common / "packed-refs", limit=8 * 1024 * 1024)
                .decode().split("\n")) if packed_ref == ref]
            if len(matches) != 1:
                raise GitError("immutable source HEAD reference is unavailable")
            value = matches[0]
    return value


def _object_store(common: Path) -> Path:
    objects = common / "objects"
    if not objects.is_dir() or objects.is_symlink():
        raise GitError("immutable source objects are unavailable")
    alternates = objects / "info" / "alternates"
    if alternates.exists() and _regular(alternates, limit=4096).strip():
        raise GitError("immutable source object alternates are unsupported")
    return objects


def _object_files(common: Path) -> Iterator[tuple[Path, Path]]:
    """Bound the physical store, including metadata Git may read via alternates."""
    objects = _object_store(common)
    total = count = 0
    for root, dirs, files in os.walk(objects, followlinks=False):
        parent = Path(root)
        for name in dirs:
            if (parent / name).is_symlink():
                raise GitError("immutable source object directories cannot be symlinks")
        for name in files:
            path = parent / name
            relative = path.relative_to(objects)
            with _regular_file(path, limit=_MAX_OBJECT_BYTES) as (_, size):
                total += size
            count += 1
            if total > _MAX_OBJECT_BYTES or count > _MAX_FILES:
                raise GitError("immutable source object population exceeds the supported size")
            if relative.parts[0] == "info":
                continue
            if relative.parts[:2] == ("pack", "multi-pack-index.d"):
                continue
            if len(relative.parts) == 2 and relative.parts[0] == "pack" and (
                re.fullmatch(r"pack-[0-9a-f]+\.(bitmap|keep|mtimes|promisor)", name)
                or name == "multi-pack-index" or re.fullmatch(r"multi-pack-index-[0-9a-f]+\.bitmap", name)
            ):
                continue
            if not (
                len(relative.parts) == 2
                and (re.fullmatch(r"[0-9a-f]{2}", relative.parts[0])
                     and re.fullmatch(r"[0-9a-f]{38}|[0-9a-f]{62}", name)
                     or relative.parts[0] == "pack" and re.fullmatch(r"pack-[0-9a-f]+\.(pack|idx|rev)", name))
            ):
                raise GitError("immutable source object layout is unsupported")
            yield path, relative


def _copy_objects(common: Path, target: Path) -> None:
    total = 0
    for path, relative in _object_files(common):
        with _parent(common / "objects", relative.as_posix(), private=False) as (parent, name):
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as source, _exclusive_file(target, relative.as_posix()) as destination:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or _fingerprint(opened) != _fingerprint(before):
                    raise _SourceChanged("immutable source object changed during copying")
                total += before.st_size
                if total > _MAX_OBJECT_BYTES:
                    raise GitError("immutable source object population exceeds the supported size")
                remaining = before.st_size
                while remaining:
                    chunk = source.read(min(_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise _SourceChanged("immutable source object was truncated")
                    destination.write(chunk)
                    remaining -= len(chunk)
                if source.read(1) or _fingerprint(os.fstat(source.fileno())) != _fingerprint(before):
                    raise _SourceChanged("immutable source object changed during copying")
                after = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if _fingerprint(after) != _fingerprint(before):
                    raise _SourceChanged("immutable source object changed during copying")


def _git(repo: Path, *args: str, retries: int = 0) -> str:
    """Run once unless a read-only caller explicitly opts into timeout retries."""
    return process._run_git(
        repo, list(args), timeout=30, retries=retries, error_context="immutable source validation failed",
        max_output_bytes=_MAX_FILES * 96 + 4096,
    ).stdout.strip()


def _validate_object_population(repo: Path) -> None:
    """Inspect reconstructed sizes without asking Git to inflate object bodies."""
    objects = _git(repo, "cat-file", "--batch-all-objects",
                   "--batch-check=%(objectname) %(objecttype) %(objectsize)")
    total = count = 0
    for line in objects.splitlines():
        fields = line.split(" ")
        if (len(fields) != 3 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", fields[0])
                or fields[1] not in {"blob", "tree", "commit", "tag"} or not fields[2].isdecimal()):
            raise GitError("immutable source object metadata is unsupported")
        total += int(fields[2])
        count += 1
        if total > _MAX_OBJECT_BYTES or count > _MAX_FILES:
            raise GitError("immutable source reconstructed object population exceeds the supported size")


def validate_synthetic_history(
    repo: Path, base: str, head: str, *, retries: int = process._GIT_TIMEOUT_RETRIES,
) -> None:
    """Require a root base, its single-child head, and no other commit objects."""
    if (_git(repo, "rev-list", "--parents", "-n", "1", base, retries=retries) != base
            or _git(repo, "rev-list", "--parents", "-n", "1", head, retries=retries) != head + " " + base):
        raise GitError("immutable synthetic projection requires exactly two commits")
    objects = _git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)",
                   retries=retries)
    if {line.split()[0] for line in objects.splitlines() if line.endswith(" commit")} != {base, head}:
        raise GitError("immutable synthetic bundle contains extra commit history")


def _inventory(repo: Path, head: str, *, max_blob_bytes: int | None = None) -> list[tuple[str, str, str]]:
    raw = process._run_git(repo, ["ls-tree", "-r", "-z", "-l", head], capture_bytes=True,
                           timeout=30, error_context="immutable source inventory unavailable",
                           max_output_bytes=_MAX_SOURCE_BYTES).stdout
    if raw.count(b"\0") > _MAX_FILES:
        raise GitError("immutable source file count exceeds the supported size")
    result = []
    total = 0
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        identity, _, raw_name = entry.partition(b"\t")
        mode, kind, oid, raw_size = identity.decode().split()
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitError("immutable source requires UTF-8 paths") from exc
        if (not name or name.startswith("/") or ".." in Path(name).parts or "\\" in name
                or ".git" in {p.lower() for p in Path(name).parts}):
            raise GitError("immutable source contains an unsafe path")
        if (mode, kind) not in {("100644", "blob"), ("100755", "blob"), ("120000", "blob"), ("160000", "commit")}:
            raise GitError("immutable source entry metadata is unsupported")
        if kind == "blob":
            size = int(raw_size)
            total += size
            if size > (_MAX_BLOB_BYTES if max_blob_bytes is None else max_blob_bytes) or total > _MAX_SOURCE_BYTES:
                raise GitError("immutable source files exceed the supported size")
        result.append((mode, oid, name))
    return result


def _materialize(repo: Path, source: Path, head: str) -> None:
    for mode, oid, name in _inventory(repo, head):
        if mode == "160000":
            continue
        body = process._run_git(repo, ["cat-file", "blob", oid], capture_bytes=True, timeout=30,
                                error_context="immutable source blob unavailable",
                                max_output_bytes=_MAX_BLOB_BYTES + 4096).stdout
        if len(body) > _MAX_BLOB_BYTES:
            raise GitError("immutable source files exceed the supported size")
        original = source / name
        try:
            info = original.lstat()
            confined = not any(parent.is_symlink() for parent in original.parents if parent.is_relative_to(source))
            if mode == "120000":
                valid = confined and stat.S_ISLNK(info.st_mode) and os.fsencode(os.readlink(original)) == body
            else:
                valid = (confined and stat.S_ISREG(info.st_mode) and _regular(original, limit=_MAX_BLOB_BYTES) == body
                    and bool(info.st_mode & stat.S_IXUSR) == (mode == "100755"))
        except OSError as exc:
            raise GitError("immutable source checkout is unavailable") from exc
        if not valid:
            raise GitError("immutable source checkout differs from captured raw bytes or modes")
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            destination.symlink_to(os.fsdecode(body))
        else:
            destination.write_bytes(body)
            destination.chmod(0o755 if mode == "100755" else 0o644)


@contextmanager
def prepare_immutable_source(
    source: Path, identity: ImmutableReviewInput, destination: Path, *,
    bundle: Path | None = None, environment: Mapping[str, str] | None = None,
) -> Iterator[Path]:
    """Validate supplied identities and checkout bytes, then yield an inert source."""
    value = identity.to_dict()
    local = value["materialization"]
    git_dir, common = _source_git_dir(source)
    if _source_head(git_dir, common) != local["head_sha"]:
        raise GitError("immutable source checkout HEAD differs from requested local head")
    if (local["bundle_sha256"] is None) != (bundle is None):
        raise GitError("immutable frozen bundle evidence is missing or unexpected")
    bundle_bytes = None if bundle is None else _regular(bundle, limit=_MAX_OBJECT_BYTES)
    if bundle_bytes is not None and hashlib.sha256(bundle_bytes).hexdigest() != local["bundle_sha256"]:
        raise GitError("immutable frozen bundle digest mismatch")
    destination.mkdir(mode=0o700, parents=True)
    try:
        with inert_git_context(destination, environment):
            _git(destination, "init", "--template=", "--object-format=" +
                 ("sha256" if len(local["head_sha"]) == 64 else "sha1"), ".")
            if bundle_bytes is None:
                _copy_objects(common, destination / ".git" / "objects")
            else:
                frozen = destination / ".git" / "input.bundle"
                frozen.write_bytes(bundle_bytes)
                heads = _git(destination, "bundle", "list-heads", str(frozen)).splitlines()
                if local["kind"] == "synthetic" and set(heads) != {
                    local["comparison_base_sha"] + " refs/heads/base", local["head_sha"] + " refs/heads/head",
                }:
                    raise GitError("immutable synthetic bundle must advertise exactly base and head")
                _git(destination, "bundle", "unbundle", str(frozen))
                frozen.unlink()
                for _ in _object_files(destination / ".git"):
                    pass
            _validate_object_population(destination)
            _git(destination, "fsck", "--full", "--strict", "--no-reflogs", "--no-dangling")
            for key in ("head_sha", "requested_base_sha", "comparison_base_sha"):
                if _git(destination, "rev-parse", "--verify", local[key] + "^{commit}") != local[key]:
                    raise GitError("immutable local commit is unavailable")
            comparison = _git(destination, "merge-base", local["requested_base_sha"], local["head_sha"])
            if comparison != local["comparison_base_sha"]:
                raise GitError("immutable comparison does not match the local merge base")
            for commit, tree in (("head_sha", "head_tree_sha"), ("comparison_base_sha", "comparison_tree_sha")):
                if _git(destination, "rev-parse", local[commit] + "^{tree}") != local[tree]:
                    raise GitError("immutable source tree identity mismatch")
            if local["kind"] == "synthetic":
                validate_synthetic_history(destination, local["comparison_base_sha"], local["head_sha"])
            diff = immutable_review_diff(destination, local["comparison_base_sha"], local["head_sha"])
            if hashlib.sha256(diff.encode()).hexdigest() != local["review_diff_key"]:
                raise GitError("immutable review diff identity mismatch")
            _materialize(destination, source, local["head_sha"])
            (destination / ".git" / "HEAD").write_text(local["head_sha"] + "\n")
            _git(destination, "read-tree", local["head_sha"])
            yield destination
    finally:
        shutil.rmtree(destination)
