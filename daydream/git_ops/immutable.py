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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from daydream.git_ops import process
from daydream.git_ops.models import GitError
from daydream.immutable_review import ImmutableReviewInput

_CONTEXT: ContextVar[tuple[Path, dict[str, str]] | None] = ContextVar("immutable_git_context", default=None)
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
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


def canonical_diff_sha256(repo: Path, base_sha: str, head_sha: str) -> str:
    """Hash the binary-safe compiler diff, separately from the decoded review key."""
    raw = process._run_git(
        repo, ["-c", "core.abbrev=40", "diff", "--binary", "--no-ext-diff", "--no-textconv", "--no-color",
               "--src-prefix=a/", "--dst-prefix=b/", base_sha, head_sha, "--"],
        capture_bytes=True, timeout=30, error_context="immutable canonical diff unavailable",
    ).stdout
    if len(raw) > _MAX_SOURCE_BYTES:
        raise GitError("immutable canonical diff exceeds the supported size")
    return hashlib.sha256(raw).hexdigest()


def _regular(path: Path, *, limit: int) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise GitError("immutable source evidence must be a bounded regular file")
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise GitError("immutable source evidence exceeds the supported size")
        return content
    except OSError as exc:
        raise GitError("immutable source evidence is unavailable") from exc


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
    value = _regular(git_dir / "HEAD", limit=4096).decode().strip()
    if value.startswith("ref: "):
        ref = value.removeprefix("ref: ")
        if not re.fullmatch(r"refs/[A-Za-z0-9_./-]+", ref) or ".." in ref:
            raise GitError("immutable source HEAD reference is malformed")
        loose = common / ref
        if loose.exists():
            value = _regular(loose, limit=4096).decode().strip()
        else:
            matches = [line.split()[0] for line in _regular(common / "packed-refs", limit=8 * 1024 * 1024)
                       .decode().splitlines() if line.endswith(" " + ref)]
            if len(matches) != 1:
                raise GitError("immutable source HEAD reference is unavailable")
            value = matches[0]
    return value


def _copy_objects(common: Path, target: Path) -> None:
    objects = common / "objects"
    if not objects.is_dir() or objects.is_symlink():
        raise GitError("immutable source objects are unavailable")
    alternates = objects / "info" / "alternates"
    if alternates.exists() and _regular(alternates, limit=4096).strip():
        raise GitError("immutable source object alternates are unsupported")
    total = count = 0
    for root, dirs, files in os.walk(objects, followlinks=False):
        parent = Path(root)
        for name in dirs:
            if (parent / name).is_symlink():
                raise GitError("immutable source object directories cannot be symlinks")
        for name in files:
            path = parent / name
            relative = path.relative_to(objects)
            if relative.parts[0] == "info":
                continue
            if not (
                len(relative.parts) == 2
                and (re.fullmatch(r"[0-9a-f]{2}", relative.parts[0])
                     and re.fullmatch(r"[0-9a-f]{38}|[0-9a-f]{62}", name)
                     or relative.parts[0] == "pack" and re.fullmatch(r"pack-[0-9a-f]+\.(pack|idx|rev)", name))
            ):
                raise GitError("immutable source object layout is unsupported")
            body = _regular(path, limit=_MAX_OBJECT_BYTES)
            total += len(body)
            count += 1
            if total > _MAX_OBJECT_BYTES or count > _MAX_FILES:
                raise GitError("immutable source object population exceeds the supported size")
            destination = target / relative
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(body)


def _git(repo: Path, *args: str) -> str:
    return process._run_git(
        repo, list(args), timeout=30, error_context="immutable source validation failed",
    ).stdout.strip()


def _inventory(repo: Path, head: str) -> list[tuple[str, str, str]]:
    raw = process._run_git(repo, ["ls-tree", "-r", "-z", head], capture_bytes=True,
                           timeout=30, error_context="immutable source inventory unavailable").stdout
    result = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        identity, _, raw_name = entry.partition(b"\t")
        mode, kind, oid = identity.decode().split()
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitError("immutable source requires UTF-8 paths") from exc
        if (not name or name.startswith("/") or ".." in Path(name).parts or "\\" in name
                or ".git" in {p.lower() for p in Path(name).parts}):
            raise GitError("immutable source contains an unsafe path")
        if (mode, kind) not in {("100644", "blob"), ("100755", "blob"), ("120000", "blob"), ("160000", "commit")}:
            raise GitError("immutable source entry metadata is unsupported")
        result.append((mode, oid, name))
    if len(result) > _MAX_FILES:
        raise GitError("immutable source file count exceeds the supported size")
    return result


def _materialize(repo: Path, source: Path, head: str) -> None:
    total = 0
    for mode, oid, name in _inventory(repo, head):
        if mode == "160000":
            continue
        size = int(_git(repo, "cat-file", "-s", oid))
        total += size
        if size > 8 * 1024 * 1024 or total > _MAX_SOURCE_BYTES:
            raise GitError("immutable source files exceed the supported size")
        body = process._run_git(repo, ["cat-file", "blob", oid], capture_bytes=True, timeout=30,
                                error_context="immutable source blob unavailable").stdout
        original = source / name
        try:
            info = original.lstat()
            confined = not any(parent.is_symlink() for parent in original.parents if parent.is_relative_to(source))
            if mode == "120000":
                valid = confined and stat.S_ISLNK(info.st_mode) and os.fsencode(os.readlink(original)) == body
            else:
                valid = (confined and stat.S_ISREG(info.st_mode) and _regular(original, limit=8 * 1024 * 1024) == body
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
                base, head = local["comparison_base_sha"], local["head_sha"]
                if (_git(destination, "rev-list", "--parents", "-n", "1", base) != base
                        or _git(destination, "rev-list", "--parents", "-n", "1", head) != head + " " + base
                        or _git(destination, "rev-list", "--count", head) != "2"):
                    raise GitError("immutable synthetic projection requires exactly two commits")
                objects = _git(
                    destination, "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)",
                )
                if {line.split()[0] for line in objects.splitlines() if line.endswith(" commit")} != {base, head}:
                    raise GitError("immutable synthetic bundle contains extra commit history")
            canonical_digest = canonical_diff_sha256(destination, local["comparison_base_sha"], local["head_sha"])
            if canonical_digest != local["canonical_diff_sha256"]:
                raise GitError("immutable canonical diff digest mismatch")
            from daydream.git_ops.source import committed_diff

            diff = committed_diff(destination, local["comparison_base_sha"], local["head_sha"])
            if hashlib.sha256(diff.encode()).hexdigest() != local["review_diff_key"]:
                raise GitError("immutable review diff identity mismatch")
            _materialize(destination, source, local["head_sha"])
            (destination / ".git" / "HEAD").write_text(local["head_sha"] + "\n")
            _git(destination, "read-tree", local["head_sha"])
            yield destination
    finally:
        shutil.rmtree(destination)
