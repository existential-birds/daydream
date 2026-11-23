"""Real-process/filesystem tests for production artifact visibility.

The independent manifest oracle reads filesystem bytes and modes directly, so
round-trip assertions do not derive their expected results from production code.
Child processes exercise session locks and crash recovery at durable transitions.
"""

from __future__ import annotations

import atexit
import errno
import functools
import hashlib
import inspect
import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Literal, cast

import anyio
import pytest

from daydream import artifact_visibility
from daydream.artifact_visibility import (
    ArtifactTreeSnapshot,
    ArtifactVisibilityError,
    OutputLabel,
    PrivateRootLocations,
    PrivateWorkspaceOwner,
    artifact_dir_for,
    artifact_session_active,
    derive_workspace_identity,
    open_artifact_session as _open_artifact_session,
    operational_worktree_root,
    private_root_locations,
    resolve_private_workspace_owner,
    review_output_path_for,
    validate_private_workspace_owner,
)
from daydream.artifacts import (
    external as artifact_external,
    filesystem as artifact_filesystem,
    ownership as artifact_ownership,
    publication as artifact_publication,
    transactions as artifact_transactions,
    transfer as artifact_transfer,
)
from daydream.artifacts.filesystem import _atomic_json
from daydream.config import DEFAULT_RETAINED_RUNS
from daydream.json_utils import _fsync_directory, _fsync_file
from daydream.run_artifacts import _RunWriteCapture
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    SIBLINGS_DIRNAME,
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    TrajectoryDocumentSnapshot,
)
from daydream.workspace import WorkContext
from tests.harness.git_helpers import git as _git
from tests.harness.trajectory import evidence_document, make_recorder

_CUTOFF = "2026-09-06T12:00:00Z"
_COMPLETE = artifact_visibility.ArtifactDisposition.COMPLETE


@dataclass(frozen=True)
class _Entry:
    path: str
    kind: Literal["directory", "file"]
    mode: int
    size: int
    sha256: str | None


@functools.cache
def _template_repo() -> Path:
    """Build the one-commit repository once per worker; each test gets an independent copy."""
    repo = Path(tempfile.mkdtemp(prefix="daydream-visibility-template-")) / "repo"
    atexit.register(shutil.rmtree, repo.parent, True)
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    # Background Git maintenance must not race the full source manifest assertions.
    _git(repo, "config", "maintenance.auto", "false")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Storage Spike")
    (repo / "source.txt").write_text("source canary\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def _init_repo(repo: Path) -> None:
    repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_template_repo(), repo, symlinks=True)


def _git_common_dir(repo: Path) -> Path:
    raw = Path(_git(repo, "rev-parse", "--git-common-dir"))
    return (raw if raw.is_absolute() else repo / raw).resolve(strict=True)


def _workspace_key(source: Path) -> str:
    payload = (
        b"daydream-artifacts-v1\0"
        + os.fsencode(source.resolve(strict=True))
        + b"\0"
        + os.fsencode(_git_common_dir(source))
    )
    return hashlib.sha256(payload).hexdigest()


def _owner(source: Path, *, locations: PrivateRootLocations | None = None) -> PrivateWorkspaceOwner:
    selected = private_root_locations() if locations is None else locations
    return resolve_private_workspace_owner(source, locations=selected)


@asynccontextmanager
async def open_artifact_session(work: WorkContext, *, session_id: str, owner: PrivateWorkspaceOwner | None = None,
    retained_runs: int = DEFAULT_RETAINED_RUNS,
) -> AsyncIterator[Any]:
    selected = _owner(work.source) if owner is None else owner
    async with _open_artifact_session(work, session_id=session_id, owner=selected,
                                      retained_runs=retained_runs) as session:
        yield session


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _read_regular_nofollow(path: Path, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino,):
            raise RuntimeError("artifact entry changed during no-follow read")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _walk_entry(root: Path, path: Path, entries: list[_Entry]) -> None:
    info = path.lstat()
    relative = path.relative_to(root).as_posix()
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError("manifest oracle refuses symlink entries")
    if stat.S_ISDIR(info.st_mode):
        entries.append(_Entry(relative, "directory", mode, 0, None))
        children = sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
        for child in children:
            _walk_entry(root, child, entries)
        return
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("manifest oracle accepts only directories and regular files")
    content = _read_regular_nofollow(path, info)
    entries.append(_Entry(relative, "file", mode, len(content), hashlib.sha256(content).hexdigest()))


def _manifest(root: Path) -> tuple[_Entry, ...]:
    entries: list[_Entry] = []
    for name in (".daydream", ".review-output.md"):
        path = root / name
        if path.exists() or path.is_symlink():
            _walk_entry(root, path, entries)
    return tuple(entries)


def _shared(entries: tuple[_Entry, ...]) -> tuple[_Entry, ...]:
    """The shared set a live tree receives: every public entry except earlier run evidence."""
    return tuple(entry for entry in entries if not entry.path.startswith(".daydream/runs/"))


def _park(marker: Path, state: str) -> None:
    """Publish the kill-point marker durably, then wait to be SIGKILLed."""
    marker.write_text(state, encoding="ascii")
    _fsync_file(marker)
    _fsync_directory(marker.parent)
    while True:
        signal.pause()


def _wait_for_marker(process: subprocess.Popen[str], marker: Path) -> None:
    # A loaded host can need tens of seconds to import the child; only a hang should fail here.
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if marker.exists():
            return
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise AssertionError(f"storage helper exited before marker: {stderr}")
        time.sleep(0.01)
    process.kill()
    _, stderr = process.communicate(timeout=5)
    raise AssertionError(f"storage helper did not reach marker: {stderr}")


def _spawn_helper(*args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603 - this module is the fixed test helper
        [sys.executable, str(Path(__file__).resolve()), *args], cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _production_external_child(source: Path, repo: Path, runtime: Path, checkpoint: str, purpose: str, marker: Path,
) -> None:
    locations = private_root_locations(base=runtime.parent)
    owner = resolve_private_workspace_owner(source, locations=locations)

    def stop_at_external(state: str, observed_purpose: str, _path: Path) -> None:
        if (state, observed_purpose) == (checkpoint, purpose):
            _park(marker, f"{state}:{observed_purpose}")

    artifact_external._external_entry_observer = stop_at_external

    async def run() -> None:
        requested = source.parent / "external-crash-output" / RUN_DOCUMENT_NAME
        _external_target(requested.parent, prior=False)
        async with open_artifact_session(_work(source, repo=repo), session_id="external-crash", owner=owner) as session:
            session.register_trajectory_output(requested)
            raise AssertionError(f"external checkpoint was not observed: {checkpoint}:{purpose}")

    anyio.run(run)


def _cross_device_park(move_out: Callable[..., None]) -> Callable[..., None]:
    """Make the private park fail with EXDEV, as on a checkout another filesystem holds."""

    def cross_device(source: artifact_ownership.SourceOwnership, name: str, destination: Path) -> None:
        if destination.parent.name == "parked":
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))
        move_out(source, name, destination)

    return cross_device


def _production_transaction_child(source: Path, repo: Path, runtime: Path, transition: str, location: str,
                                  marker: Path) -> None:
    locations = private_root_locations(base=runtime.parent)
    owner = resolve_private_workspace_owner(source, locations=locations)
    if location == "beside":
        artifact_publication._move_out = _cross_device_park(artifact_publication._move_out)

    def stop_at_transition(state: str) -> None:
        if state == transition:
            _park(marker, state)

    artifact_transactions._transition_observer = stop_at_transition

    async def run() -> None:
        async with open_artifact_session(_work(source, repo=repo), session_id="production-crash", owner=owner):
            raise AssertionError(f"transition was not observed: {transition}")

    anyio.run(run)


_EARLIER_EVIDENCE = b"\x00\xff\xfe\x80payload\n"


def _copies(content: bytes, *roots: Path) -> int:
    """Regular files under ``roots`` holding ``content``; earlier evidence is moved, never copied."""
    return sum(path.is_file() and not path.is_symlink() and path.read_bytes() == content
               for root in roots for path in root.rglob("*"))


def _seed_public_artifacts(source: Path) -> tuple[_Entry, ...]:
    deep = source / ".daydream" / "deep"
    runs = source / ".daydream" / "runs"
    empty = deep / "empty-directory"
    deep.mkdir(parents=True)
    runs.mkdir()
    empty.mkdir()
    (deep / "prior.md").write_bytes(b"prior reasoning\n")
    (runs / "opaque.bin").write_bytes(_EARLIER_EVIDENCE)
    (source / ".review-output.md").write_bytes(b"review output\n")
    os.chmod(source / ".daydream", 0o750)
    os.chmod(deep, 0o710)
    os.chmod(runs, 0o700)
    os.chmod(empty, 0o711)
    os.chmod(deep / "prior.md", 0o640)
    os.chmod(runs / "opaque.bin", 0o600)
    os.chmod(source / ".review-output.md", 0o644)
    return _manifest(source)


def _work(source: Path, *, repo: Path | None = None, run_id: str = "work") -> WorkContext:
    return WorkContext(repo=source if repo is None else repo, source=source, base_branch="main",
        base_sha=_git(source, "rev-parse", "HEAD"), head_branch="main" if repo is None else None,
        head_sha=_git(source, "rev-parse", "HEAD"), is_ephemeral=repo is not None, run_id=run_id,
    )


def _worktrees(source: Path, *names: str) -> tuple[Path, ...]:
    """Detached ephemeral worktrees of ``source``, created as siblings of it."""
    repos = tuple(source.parent / name for name in names)
    for repo in repos:
        _git(source, "worktree", "add", "--detach", str(repo), "HEAD")
    return repos


def _external_target(parent: Path, *, prior: bool = True) -> tuple[Path, Path]:
    """A fresh external ``trajectory.json``/``.partial`` pair under ``parent``."""
    parent.mkdir(parents=True)
    requested = parent / RUN_DOCUMENT_NAME
    partial = requested.with_suffix(requested.suffix + ".partial")
    if prior:
        requested.write_bytes(b"prior full")
        partial.write_bytes(b"prior partial")
    return requested, partial


def _payload(session_id: str, trajectory_id: str | None = None, **extra: Any) -> bytes:
    return evidence_document(session_id if trajectory_id is None else trajectory_id, session_id=session_id,
                             extra=extra or None)


def _reference(path: Path, trajectory_id: str, payload: bytes) -> TrajectoryDocumentReference:
    """Write one evidence document where the recorder would and reference its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return TrajectoryDocumentReference.written(trajectory_id, path, path, payload)


def _snapshot(session_id: str, documents: tuple[TrajectoryDocumentReference, ...], *, status: str = "complete",
    root_trajectory_id: str | None = None,
) -> RunWriteSnapshot:
    return RunWriteSnapshot(status=cast(Any, status), cutoff_at=_CUTOFF,
        root_trajectory_id=session_id if root_trajectory_id is None else root_trajectory_id, documents=documents,
    )


def _freeze_run(session: Any, session_id: str, *, payload: bytes | None = None) -> Any:
    """Freeze one root evidence document written into the session's own run dir."""
    document = _reference(session.daydream_dir / "runs" / session_id / RUN_DOCUMENT_NAME, session_id,
                          _payload(session_id) if payload is None else payload)
    return session.freeze(_snapshot(session_id, (document,)))


def _publish(session: Any, frozen: Any, disposition: Any = None) -> None:
    session.finalize_frozen(frozen, disposition=_COMPLETE if disposition is None else disposition)


async def _publish_custom_public_trajectory(source: Path, *, session_id: str, status: str = "complete",
) -> tuple[Path, Path, bytes]:
    requested = source / ".daydream" / "custom" / "root.json"
    selected = requested if status == "complete" else requested.with_suffix(".json.partial")
    payload = _payload(session_id)
    async with open_artifact_session(_work(source), session_id=session_id) as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        assert not session.daydream_dir.exists()
        route = session.register_trajectory_output(requested)
        selected_route = route.full if status == "complete" else route.partial
        assert selected_route.write_path is not None
        root = _reference(route.run_dir / RUN_DOCUMENT_NAME, session_id, payload)
        session.write_trajectory_document(route, TrajectoryDocumentSnapshot(session_id, root.path, payload),
                                          cast(Any, status))
        assert not (source / ".daydream").exists()
        frozen = session.freeze(_snapshot(session_id, (root,), status=status))
        state_root = session.layout.state_root
        _publish(session, frozen)
    return state_root, selected, payload


@contextmanager
def _child_at_marker(marker: Path, *args: str) -> Iterator[subprocess.Popen[str]]:
    """Run a helper child until it publishes ``marker``, then SIGKILL it.

    The kill happens when the ``with`` body returns, so a body may inspect
    on-disk state (or contend for the lock) while the child is still parked.
    """
    process = _spawn_helper(*args)
    try:
        _wait_for_marker(process, marker)
        yield process
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


class _ReplaceAtTransfer:
    """Replace ``target`` with unique bytes the first time the host transfers it."""

    def __init__(self, target: Path, replacement: bytes) -> None:
        self.target = target
        self.replacement = replacement
        self.observed = False

    def __call__(self, path: Path, _stage: Path) -> None:
        if path != self.target or self.observed:
            return
        self.observed = True
        path.unlink()
        path.write_bytes(self.replacement)


def _no_reversal() -> Callable[[], Any]:
    """A ``_name_exchange_factory`` that fails if any further exchange is attempted."""

    def factory() -> Any:
        raise AssertionError("an additional name exchange (reversal) was attempted")

    return factory


@contextmanager
def _case_note(label: str) -> Iterator[None]:
    """Name the table row in any failure raised by a merged loop."""
    try:
        yield
    except BaseException as error:
        error.add_note(label)
        raise


async def _run_case(
    case: Callable[..., Any], *, root: Path, monkeypatch: pytest.MonkeyPatch, **extra: Any,
) -> None:
    """Run one merged case with its own root, and a fresh source repository when it asks for one."""
    root.mkdir(parents=True)
    available: dict[str, Any] = {"tmp_path": root, "monkeypatch": monkeypatch, **extra}
    parameters = inspect.signature(case).parameters
    if "source" in parameters:
        available["source"] = root / "source"
        _init_repo(available["source"])
    try:
        result = case(**{name: available[name] for name in parameters})
        if inspect.isawaitable(result):
            await result
    except BaseException as error:
        error.add_note(f"merged case: {case.__name__} {extra!r}")
        raise


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A real one-commit Git repository at ``tmp_path/source``."""
    repo = tmp_path / "source"
    _init_repo(repo)
    return repo


async def test_private_workspace_owner_rejects_unsafe_roots_peers_modes_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner resolution and validation refuse unsafe roots, foreign peers, unsafe modes and tampered paths."""
    cases: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
        (_resolve_private_workspace_owner_rejects_unsafe_direct_root_pairs_case, {}),
        (_private_workspace_owner_conflicting_peer_prevents_missing_peer_creation_case, {}),
        (_artifact_session_revalidates_supplied_owner_before_mutation_case, {}),
        (_validate_private_workspace_owner_rejects_direct_dataclass_path_tampering_case, {}),
        (_derive_workspace_identity_rejects_model_repo_inside_artifact_runtime_case, {}),
        (_validate_private_workspace_owner_rejects_unsafe_private_modes_case, {}),
        (_artifact_session_rejects_symlinked_source_and_runtime_ancestry_case, {}),
        (_artifact_session_rejects_owner_mismatch_and_runtime_overlap_case, {"peer": "operational"}),
    )
    for index, (case, extra) in enumerate(cases):
        with monkeypatch.context() as patch:
            await _run_case(case, root=tmp_path / f"case-{index}", monkeypatch=patch, **extra)


def _resolve_private_workspace_owner_rejects_unsafe_direct_root_pairs_case(tmp_path: Path, source: Path) -> None:
    for locations in (
        PrivateRootLocations(artifact_runtime=Path("relative-runtime"), operational_workspaces=tmp_path / "operations"),
        PrivateRootLocations(artifact_runtime=tmp_path / "private", operational_workspaces=tmp_path / "private"),
    ):
        with pytest.raises(ArtifactVisibilityError, match="absolute lexical|overlap"):
            resolve_private_workspace_owner(source, locations=locations)


def _private_workspace_owner_conflicting_peer_prevents_missing_peer_creation_case(tmp_path: Path, source: Path,
) -> None:
    locations = private_root_locations(base=tmp_path / "private")
    key = _workspace_key(source)
    conflicting_state = locations.operational_workspaces / key
    conflicting_state.mkdir(parents=True, mode=0o700)
    os.chmod(locations.operational_workspaces, 0o700)
    os.chmod(conflicting_state, 0o700)
    _atomic_json(conflicting_state / "owner.json",
        {"schema_version": 1, "workspace_key": key, "source": "different-source",
            "git_common_dir": str(_git_common_dir(source)),
        },
    )

    with pytest.raises(ArtifactVisibilityError, match="owner metadata"):
        resolve_private_workspace_owner(source, locations=locations)

    assert not (locations.artifact_runtime / key).exists()


async def _artifact_session_revalidates_supplied_owner_before_mutation_case(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _init_repo(first)
    _init_repo(second)
    locations = private_root_locations(base=tmp_path / "private")
    owner = resolve_private_workspace_owner(first, locations=locations)

    with pytest.raises(ArtifactVisibilityError, match="identity mismatch"):
        async with _open_artifact_session(_work(second), session_id="wrong-owner", owner=owner):
            pass

    assert not (second / ".daydream").exists()
    assert not (second / ".review-output.md").exists()
    assert not (owner.artifact_state_root / "runs").exists()


def _validate_private_workspace_owner_rejects_direct_dataclass_path_tampering_case(tmp_path: Path, source: Path,
) -> None:
    owner = _owner(source, locations=private_root_locations(base=tmp_path / "private"))
    tampered = PrivateWorkspaceOwner(
        source=owner.source, git_common_dir=owner.git_common_dir, workspace_key=owner.workspace_key,
        artifact_state_root=owner.artifact_state_root.parent / "different-key",
        operational_state_root=owner.operational_state_root,
    )

    with pytest.raises(ArtifactVisibilityError, match="path mismatch|accessible|real directory"):
        validate_private_workspace_owner(tampered, source=source)


async def test_artifact_session_preserves_resume_mtimes_across_publication_and_reopen(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One owned workspace rolls back, publishes several output kinds, then resumes elsewhere.

    This sequence pins owner metadata, byte/mode preservation and projection together; independent
    setup fragments cannot expose a route or shared-set regression across those transitions. Earlier
    evidence is carried by rename, so it keeps its inode, and no canonical copy is kept.
    """

    def fail_default_lookup() -> Path:
        raise AssertionError("explicit private base consulted the default provider")

    environment = dict(os.environ)
    with monkeypatch.context() as patch:
        patch.setattr(artifact_ownership, "_default_private_base", fail_default_lookup)
        locations = private_root_locations(base=tmp_path / "private")
    assert locations == PrivateRootLocations(
        artifact_runtime=tmp_path / "private/runtime", operational_workspaces=tmp_path / "private/workspaces"
    )
    assert dict(os.environ) == environment
    assert not locations.artifact_runtime.exists() and not locations.operational_workspaces.exists()
    worktrees_before = _git(source, "worktree", "list", "--porcelain")
    owner = resolve_private_workspace_owner(source, locations=locations)
    operational = operational_worktree_root(owner)
    assert owner.artifact_state_root == locations.artifact_runtime / owner.workspace_key
    assert owner.operational_state_root == locations.operational_workspaces / owner.workspace_key
    assert operational == owner.operational_state_root / "operational"
    assert _git(source, "worktree", "list", "--porcelain") == worktrees_before
    for directory in (locations.artifact_runtime, locations.operational_workspaces, owner.artifact_state_root,
        owner.operational_state_root, operational,
    ):
        assert stat.S_IMODE(directory.lstat().st_mode) == 0o700
    artifact_owner, operational_owner = (
        owner.artifact_state_root / "owner.json",
        owner.operational_state_root / "owner.json",
    )
    assert artifact_owner.read_bytes() == operational_owner.read_bytes()
    assert stat.S_IMODE(artifact_owner.lstat().st_mode) == stat.S_IMODE(operational_owner.lstat().st_mode) == 0o600
    assert _load_json(artifact_owner) == {
        "git_common_dir": str(_git_common_dir(source)), "schema_version": 1, "source": str(source.resolve(strict=True)),
        "workspace_key": _workspace_key(source),
    }
    validate_private_workspace_owner(owner, source=source, repo=source)

    (source / ".daydream").mkdir(mode=0o750)
    os.chmod(source / ".daydream", 0o750)
    directory_only = (_Entry(".daydream", "directory", 0o750, 0, None),)
    async with open_artifact_session(_work(source), session_id="restore-directory-only", owner=owner) as empty:
        assert _manifest(empty.layout.live_root) == directory_only
        assert not (source / ".daydream").exists()
    assert _manifest(source) == directory_only

    _seed_public_artifacts(source)
    deep = source / ".daydream/deep"
    intent, alternatives, key = deep / "intent.md", deep / "alternatives.json", deep / "diff-key"
    intent.write_text("intent\n")
    alternatives.write_text("[]\n")
    current_diff_sha = "a" * 64
    key.write_text(current_diff_sha, encoding="ascii")
    key_mtime_ns = 1_700_000_000_000_000_000
    prerequisite_mtime_ns = key_mtime_ns + 10_000_000_000
    os.utime(key, ns=(key_mtime_ns, key_mtime_ns))
    for prerequisite in (intent, alternatives):
        os.utime(prerequisite, ns=(prerequisite_mtime_ns, prerequisite_mtime_ns))
    (source / ".daydream/improve").mkdir()
    (source / ".daydream/recommended.patch").write_bytes(b"diff bytes\n")
    (source / ".daydream/.DS_Store").write_bytes(b"diff bytes\n")
    baseline = _manifest(source)
    findings = source / "output with spaces.json"
    findings.write_bytes(b"prior explicit bytes")
    assert artifact_dir_for(source, allow_standalone=True) == source / ".daydream"
    assert review_output_path_for(source, allow_standalone=True) == source / ".review-output.md"
    async with open_artifact_session(_work(source), session_id="restore-explicit", owner=owner) as rollback:
        assert _manifest(rollback.layout.live_root) == _shared(baseline)
        assert not (source / ".daydream").exists() and not (source / ".review-output.md").exists()
        assert artifact_dir_for(source, session=rollback) == rollback.daydream_dir
        assert review_output_path_for(source, session=rollback) == rollback.review_output
        routed = rollback.register_destination(findings, label=OutputLabel.FINDINGS_OUTPUT)
        assert not findings.exists() and routed.delivery is artifact_visibility.DestinationDelivery.DEFERRED
        assert routed.write_path is not None
        routed.write_path.parent.mkdir(parents=True)
        routed.write_path.write_bytes(b"discarded failed-run bytes")
    assert _manifest(source) == baseline and findings.read_bytes() == b"prior explicit bytes"

    first, second = _worktrees(source, "ephemeral-one", "ephemeral-two")
    dump = source / "dump output"
    dump.mkdir()
    (dump / "unrelated.bin").write_bytes(b"preserve me")
    (dump / "replace.txt").write_bytes(b"old")
    unused_dump = tmp_path / "never-existed/out"
    # An earlier build's review cache is no artifact of this one: the publishing run deletes it.
    stale_cache = source / ".daydream/review-cache/entries/key.json"
    stale_cache.parent.mkdir(parents=True)
    stale_cache.write_bytes(b"{}")
    session_id = "mtime-first"
    payload = _payload(session_id)
    earlier_evidence = source / ".daydream/runs/opaque.bin"
    earlier_inode = earlier_evidence.lstat().st_ino
    async with open_artifact_session(_work(source, repo=first), session_id=session_id, owner=owner) as session:
        assert session.layout.workspace_key == owner.workspace_key
        assert session.layout.state_root == owner.artifact_state_root
        live_deep = session.daydream_dir / "deep"
        assert (live_deep / "diff-key").stat().st_mtime_ns == key_mtime_ns
        assert (live_deep / "alternatives.json").stat().st_mtime_ns == prerequisite_mtime_ns
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        session.register_destination(source / ".review-output.md", label=OutputLabel.PUBLIC_REVIEW_OUTPUT)
        assert session.durable_path_for(live_deep / "summary.md", repo=first) == source / ".daydream/deep/summary.md"
        assert session.live_path_for(source / ".daydream/deep/summary.md", repo=first) == live_deep / "summary.md"
        assert session.durable_path_for(session.review_output, repo=first) == source / ".review-output.md"
        assert session.live_path_for(source / ".review-output.md", repo=first) == session.review_output
        routed = session.register_destination(findings, label=OutputLabel.FINDINGS_OUTPUT)
        assert not findings.exists() and routed.write_path is not None
        assert routed.requested == findings and routed.write_path.is_relative_to(session.layout.live_root)
        with pytest.raises(ArtifactVisibilityError, match="destination collision"):
            session.register_destination(source / "." / findings.name, label=OutputLabel.FINDINGS_OUTPUT)
        with pytest.raises(ArtifactVisibilityError, match="destination overlap"):
            session.register_destination(findings / "child", label=OutputLabel.DUMP_DIRECTORY)
        assert session.durable_path_for(routed.write_path, repo=first) == findings
        assert session.live_path_for(findings, repo=first) == routed.write_path
        routed.write_path.parent.mkdir(parents=True)
        routed.write_path.write_bytes(b"published explicit bytes")
        session.register_destination(unused_dump, label=OutputLabel.DUMP_DIRECTORY)
        dump_route = session.register_destination(dump, label=OutputLabel.DUMP_DIRECTORY)
        assert dump_route.write_path is None and not any(path.is_file() for path in dump.rglob("*"))
        document = _reference(session.daydream_dir / "runs" / session_id / RUN_DOCUMENT_NAME, session_id, payload)
        # A directory beside a sibling whose name extends it (`evidence/`, `evidence.otlp.jsonl`) stays publishable.
        (child := session.daydream_dir / "runs" / session_id / "evidence/child.json").parent.mkdir()
        child.write_bytes(b"child")
        frozen = session.freeze(_snapshot(session_id, (document,)))
        assert (frozen.root / ".daydream/runs" / session_id / RUN_DOCUMENT_NAME).read_bytes() == payload
        with pytest.raises(ArtifactVisibilityError, match="frozen"):
            artifact_dir_for(first, session=session)
        late = session.finalization_merge_path(dump_route, snapshot=frozen)
        (late / "replace.txt").write_bytes(b"new")
        (late / "added.txt").write_bytes(b"added")
        (late / "name").mkdir()
        (late / "name/child").write_bytes(b"child")
        (late / "name.ext").write_bytes(b"ext")
        _publish(session, frozen)
    assert findings.read_bytes() == b"published explicit bytes"
    assert (dump / "unrelated.bin").read_bytes() == b"preserve me"
    assert (dump / "replace.txt").read_bytes() == b"new" and (dump / "added.txt").read_bytes() == b"added"
    assert (dump / "name/child").read_bytes() == b"child" and (dump / "name.ext").read_bytes() == b"ext"
    assert not unused_dump.exists() and not unused_dump.parent.exists()
    assert (source / ".daydream/deep/prior.md").read_bytes() == b"prior reasoning\n"
    assert not (source / ".daydream/review-cache").exists()
    assert not any((owner.artifact_state_root / "transactions").iterdir())
    assert (source / ".daydream/runs" / session_id / RUN_DOCUMENT_NAME).read_bytes() == payload
    assert (source / ".daydream/runs" / session_id / "evidence/child.json").read_bytes() == b"child"
    assert earlier_evidence.lstat().st_ino == earlier_inode
    assert not (owner.artifact_state_root / "canonical").exists()
    public_deep = source / ".daydream/deep"
    assert (public_deep / "diff-key").stat().st_mtime_ns == key_mtime_ns
    assert (public_deep / "alternatives.json").stat().st_mtime_ns == prerequisite_mtime_ns
    contradictory_mtime_ns = key_mtime_ns - 10_000_000_000
    os.utime(public_deep / "alternatives.json", ns=(contradictory_mtime_ns, contradictory_mtime_ns))
    published = _manifest(source)
    async with open_artifact_session(_work(source, repo=second), session_id="mtime-second", owner=owner) as resumed:
        assert resumed.layout.workspace_key == owner.workspace_key
        assert resumed.layout.state_root == owner.artifact_state_root
        assert _manifest(resumed.layout.live_root) == _shared(published)
        live_deep = resumed.daydream_dir / "deep"
        # The live copy keeps the current public times, including a change made between runs.
        assert (live_deep / "diff-key").stat().st_mtime_ns == key_mtime_ns
        assert (live_deep / "alternatives.json").stat().st_mtime_ns == contradictory_mtime_ns
    assert _manifest(source) == published
    assert artifact_dir_for(source, allow_standalone=True) == source / ".daydream"


async def test_closing_session_bounds_finished_runs_and_drops_finalized_live_copies(
    tmp_path: Path, source: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Retention runs under the closing session's lock and never deletes evidence recovery still holds.

    An unfrozen run keeps its only copy until the bound prunes it; a finalized run keeps only its
    frozen copy; a quarantined run, a symlink and an unsafe tree beyond the bound all stay in place.
    The checkout's public runs follow the same rule, ordered by directory time.
    """
    owner = _owner(source)
    runs = owner.artifact_state_root / "runs"

    async def finish(session_id: str, *, freeze: bool, retained_runs: int,
                     during: Callable[[], None] | None = None) -> None:
        async with open_artifact_session(
            _work(source), session_id=session_id, owner=owner, retained_runs=retained_runs,
        ) as session:
            if during is not None:
                during()
            if freeze:
                _publish(session, _freeze_run(session, session_id))

    await finish("unfrozen", freeze=False, retained_runs=1)
    await finish("published", freeze=True, retained_runs=3)
    assert sorted(path.name for path in (runs / "unfrozen").iterdir()) == ["live"]
    assert sorted(path.name for path in (runs / "published").iterdir()) == ["frozen", "frozen-manifest.json"]
    os.utime(runs / "unfrozen", ns=(1, 1))
    os.utime(runs / "published/frozen-manifest.json", ns=(3, 3))

    held = runs / "held"
    for name in ("live", "frozen"):
        (held / name).mkdir(parents=True)
        (held / name / "evidence").write_bytes(b"retained for the operator")
    (held / "frozen-manifest.json").write_bytes(b"{}")
    (owner.artifact_state_root / "unreconciled/detach-held-0123456789abcdef").mkdir(parents=True)
    unsafe = runs / "unsafe"
    unsafe.mkdir()
    os.mkfifo(unsafe / "fifo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"not runtime state")
    (runs / "link").symlink_to(outside, target_is_directory=True)
    for run in (held, unsafe):
        os.utime(run, ns=(0, 0))
    public = source / ".daydream/runs"
    for name, mtime in (("older", 1), ("published", 2), ("newer", 5), ("held", 0)):
        (public / name).mkdir(exist_ok=True)
        (public / name / "evidence").write_bytes(name.encode())
        os.utime(public / name, ns=(mtime, mtime))

    def plant_unsafe_public() -> None:
        # Session open refuses it in the checkout, so it appears while the tree is parked.
        parked, = owner.artifact_state_root.glob("transactions/detach-latest-*/parked/.daydream/runs")
        (parked / "public-unsafe").mkdir()
        os.mkfifo(parked / "public-unsafe" / "fifo")
        os.utime(parked / "public-unsafe", ns=(0, 0))

    capsys.readouterr()

    await finish("latest", freeze=True, retained_runs=2, during=plant_unsafe_public)

    assert sorted(path.name for path in runs.iterdir()) == ["held", "latest", "link", "published", "unsafe"]
    assert sorted(path.name for path in (runs / "latest").iterdir()) == ["frozen", "frozen-manifest.json"]
    assert sorted(path.name for path in held.iterdir()) == ["frozen", "frozen-manifest.json", "live"]
    assert (unsafe / "fifo").exists() and (outside / "keep").read_bytes() == b"not runtime state"
    assert " ".join(capsys.readouterr().out.replace("│", " ").split()).count("could not be cleaned") == 2
    assert sorted(path.name for path in public.iterdir()) == ["held", "latest", "newer", "public-unsafe"]
    assert (public / "latest" / RUN_DOCUMENT_NAME).is_file() and (public / "newer/evidence").read_bytes() == b"newer"
    assert (public / "public-unsafe/fifo").exists()

    # A rolled-back run publishes nothing, so it takes no slot: the newest earlier public run stays.
    shutil.rmtree(public / "public-unsafe")
    await finish("rolled-back", freeze=False, retained_runs=1)
    assert sorted(path.name for path in public.iterdir()) == ["held", "latest"]

    # A .daydream link that appears while nothing is parked is never followed: its target keeps every run.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (source / ".daydream").rename(elsewhere / ".daydream")
    (elsewhere / ".daydream/runs/old").mkdir()
    os.utime(elsewhere / ".daydream/runs/old", ns=(0, 0))
    capsys.readouterr()
    await finish("linked", freeze=False, retained_runs=1,
                 during=lambda: (source / ".daydream").symlink_to(elsewhere / ".daydream", target_is_directory=True))
    assert sorted(path.name for path in (elsewhere / ".daydream/runs").iterdir()) == ["held", "latest", "old"]
    assert "could not be cleaned" in " ".join(capsys.readouterr().out.replace("│", " ").split())


async def test_bound_routing_propagates_to_tasks_and_rejects_wrong_or_aliased_repo(tmp_path: Path, source: Path,
) -> None:
    work = _work(source)
    observed: list[Path] = []
    alias = tmp_path / "repo-alias"
    alias.symlink_to(source, target_is_directory=True)

    assert artifact_session_active() is False
    with pytest.raises(ArtifactVisibilityError, match="explicit artifact session"):
        artifact_dir_for(work.repo)
    with pytest.raises(ArtifactVisibilityError, match="explicit artifact session"):
        review_output_path_for(work.repo)
    async with open_artifact_session(work, session_id="routing") as session:
        assert artifact_session_active() is True
        with pytest.raises(ArtifactVisibilityError, match="explicit artifact session"):
            artifact_dir_for(work.repo)
        with pytest.raises(ArtifactVisibilityError, match="explicit artifact session"):
            review_output_path_for(work.repo)
        async def child() -> None:
            observed.append(artifact_dir_for(work.repo, allow_standalone=True))

        async with anyio.create_task_group() as group:
            group.start_soon(child)
        assert observed == [session.daydream_dir]
        assert artifact_dir_for(work.repo, allow_standalone=True) == session.daydream_dir
        assert review_output_path_for(work.repo, allow_standalone=True) == session.review_output
        with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
            artifact_dir_for(tmp_path / "wrong-repo", allow_standalone=True)
        with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
            artifact_dir_for(alias, allow_standalone=True)
        assert not (source / ".daydream").exists()
        assert not (source / ".review-output.md").exists()

    assert artifact_session_active() is False
    assert artifact_dir_for(work.repo, allow_standalone=True) == source / ".daydream"
    with pytest.raises(ArtifactVisibilityError, match="explicit artifact session"):
        artifact_dir_for(work.repo)


async def test_artifact_session_open_rejects_unsafe_public_tree_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session open refuses tracked collisions, wrong anchor types, racing manifests, unsupported roots, non-regular
    nodes and aliases without mutating them.
    """
    cases: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
        (_artifact_session_rejects_tracked_public_collision_untouched_case, {"tracked": ".review-output.md"}),
        (_artifact_session_rejects_registered_public_anchor_with_wrong_type_case,
         {"anchor": "partial-fixes", "make_wrong_type": lambda path: path.write_bytes(b"not a directory")}),
        (_public_validation_classifies_the_exact_manifested_tree_case,
         {"mutation": "static-kind", "message": "public artifact anchor has the wrong filesystem type"}),
        (_artifact_session_rejects_empty_unsupported_operational_root_case, {"namespace": "worktrees"}),
        (_artifact_session_rejects_nonregular_public_nodes_without_following_case, {}),
        (_artifact_session_rejects_duplicate_filesystem_aliases_case, {}),
    )
    for index, (case, extra) in enumerate(cases):
        with monkeypatch.context() as patch:
            await _run_case(case, root=tmp_path / f"case-{index}", monkeypatch=patch, **extra)


async def _artifact_session_rejects_tracked_public_collision_untouched_case(source: Path, tracked: str) -> None:
    target = source / tracked
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"tracked collision")
    _git(source, "add", tracked)
    _git(source, "commit", "-m", "tracked collision")

    with pytest.raises(ArtifactVisibilityError, match="tracked artifact collision"):
        async with open_artifact_session(_work(source), session_id="tracked"):
            pass

    assert target.read_bytes() == b"tracked collision"


async def _artifact_session_rejects_registered_public_anchor_with_wrong_type_case(
    source: Path, anchor: str, make_wrong_type: Callable[[Path], object],
) -> None:
    target = source / ".daydream" / anchor
    target.parent.mkdir()
    make_wrong_type(target)

    with pytest.raises(ArtifactVisibilityError, match="public artifact anchor has the wrong filesystem type"):
        async with open_artifact_session(_work(source), session_id=f"wrong-{anchor}"):
            pass

    assert target.exists()


async def _public_validation_classifies_the_exact_manifested_tree_case(
    source: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, message: str,
) -> None:
    daydream = source / ".daydream"
    anchor = daydream / "improve"
    anchor.mkdir(parents=True)
    original_manifest = artifact_filesystem.manifest_tree
    mutated = False

    def mutate_before_manifest(root: Path, names: Any = None, **kwargs: Any) -> Any:
        nonlocal mutated
        if root == source and names == (".daydream", ".review-output.md") and not mutated:
            mutated = True
            anchor.rmdir()
            anchor.write_bytes(b"wrong kind after classification")
        return original_manifest(root, names, **kwargs)

    monkeypatch.setattr(artifact_filesystem, "manifest_tree", mutate_before_manifest)
    with pytest.raises(ArtifactVisibilityError, match=message):
        async with open_artifact_session(_work(source), session_id=f"manifest-race-{mutation}"):
            pass

    assert mutated is True


async def _artifact_session_rejects_empty_unsupported_operational_root_case(
    tmp_path: Path, namespace: str,
) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    operational = source / ".daydream" / namespace
    operational.mkdir(parents=True)
    before = artifact_filesystem.manifest_tree(source)
    with pytest.raises(ArtifactVisibilityError, match="unsupported operational workspace"):
        async with open_artifact_session(_work(source), session_id="empty-root"):
            pass
    assert artifact_filesystem.manifest_tree(source) == before
    assert operational.is_dir() and not list(operational.iterdir())


async def _artifact_session_rejects_nonregular_public_nodes_without_following_case(
    tmp_path: Path, source: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary").write_bytes(b"outside")
    public = source / ".daydream"
    for node_kind in ("root-symlink", "directory-symlink", "leaf-symlink", "fifo"):
        if node_kind == "root-symlink":
            public.symlink_to(outside, target_is_directory=True)
        elif node_kind == "directory-symlink":
            public.mkdir()
            (public / "runs").symlink_to(outside, target_is_directory=True)
        else:
            (public / "runs").mkdir(parents=True)
            leaf = public / "runs" / "node"
            if node_kind == "leaf-symlink":
                leaf.symlink_to(outside / "canary")
            else:
                os.mkfifo(leaf)
        with pytest.raises(ArtifactVisibilityError, match="regular files and directories|wrong filesystem type"):
            async with open_artifact_session(_work(source), session_id=node_kind):
                pass
        assert (outside / "canary").read_bytes() == b"outside"
        if public.is_symlink():
            public.unlink()
        else:
            shutil.rmtree(public)


async def _artifact_session_rejects_owner_mismatch_and_runtime_overlap_case(tmp_path: Path, source: Path, peer: str,
) -> None:
    work = _work(source)
    locations = private_root_locations(base=tmp_path / "private-explicit")
    owner_identity = _owner(source, locations=locations)
    async with open_artifact_session(work, session_id="owner", owner=owner_identity) as session:
        state_root = session.layout.state_root
    owner_path = (owner_identity.artifact_state_root
        if peer == "artifact"
        else owner_identity.operational_state_root
    ) / "owner.json"
    owner = _load_json(owner_path)
    owner["source"] = "different-source"
    _atomic_json(owner_path, owner)
    with pytest.raises(ArtifactVisibilityError, match="owner metadata"):
        async with open_artifact_session(work, session_id="owner-two", owner=owner_identity):
            pass
    assert state_root.exists()

    overlapping = PrivateRootLocations(
        artifact_runtime=source / "nested-runtime", operational_workspaces=tmp_path / "operations",
    )
    with pytest.raises(ArtifactVisibilityError, match="overlap"):
        resolve_private_workspace_owner(source, locations=overlapping)


def _derive_workspace_identity_rejects_model_repo_inside_artifact_runtime_case(tmp_path: Path, source: Path) -> None:
    locations = private_root_locations(base=tmp_path / "private")
    owner = resolve_private_workspace_owner(source, locations=locations)
    model_repo = locations.artifact_runtime / "model-worktree"
    _git(source, "worktree", "add", "--detach", str(model_repo), "HEAD")

    with pytest.raises(ArtifactVisibilityError, match="runtime and repository overlap"):
        derive_workspace_identity(_work(source, repo=model_repo), owner=owner)


def _validate_private_workspace_owner_rejects_unsafe_private_modes_case(source: Path) -> None:
    owner = _owner(source)
    for changed, unsafe, safe in ((owner.artifact_state_root, 0o755, 0o700),
                                  (owner.operational_state_root / "owner.json", 0o644, 0o600)):
        changed.chmod(unsafe)
        with pytest.raises(ArtifactVisibilityError, match="0700|0600"):
            validate_private_workspace_owner(owner, source=source)
        changed.chmod(safe)


async def test_artifact_session_freeze_rejects_invalid_run_snapshot(source: Path) -> None:
    work = _work(source)
    session_id = "snapshot-session"
    async with open_artifact_session(work, session_id=session_id) as session:
        path = session.daydream_dir / "runs" / session_id / RUN_DOCUMENT_NAME
        document = TrajectoryDocumentReference.written(session_id, path, path, _payload(session_id))
        valid = _snapshot(session_id, (document,), root_trajectory_id=session_id)
        child_path = path.parent / SIBLINGS_DIRNAME / "child.otlp.jsonl"
        child = TrajectoryDocumentReference.written("child", child_path, child_path, _payload(session_id, "child"))
        for snapshot in (
            _snapshot(session_id, (document, document), root_trajectory_id=session_id),
            replace(valid, documents=()),
            # Reference metadata is rejected without reading: a missing root and a malformed digest.
            replace(valid, documents=(child,)),
            replace(valid, documents=(document, replace(child, sha256="0" * 63))),
            replace(valid, status=cast(Any, "unfinished")),
        ):
            with pytest.raises(ValueError):
                snapshot.validate(session_id)
            capture = _RunWriteCapture(session_id)
            capture.retain(make_recorder(source, session_id=session_id), snapshot)
            assert capture.validation_error is not None
            assert capture.final is None and capture.partial is None
            with pytest.raises(ArtifactVisibilityError, match="snapshot"):
                session.freeze(snapshot)


async def test_artifact_session_recovers_real_process_death_at_durable_transition(
    tmp_path: Path, artifact_runtime_root: Path,
) -> None:
    """A session owner SIGKILLed while its tree is parked, privately or beside the checkout, is recovered next open."""
    runtime_canary = artifact_runtime_root.parent / "cleanup-runtime-canary"
    runtime_canary.parent.mkdir(parents=True, exist_ok=True)
    runtime_canary.write_bytes(b"runtime sibling")
    for location in ("private", "beside"):
        source = tmp_path / location / "source"
        _init_repo(source)
        expected = _seed_public_artifacts(source)
        requested = source / "explicit-output.txt"
        requested.write_bytes(b"prior explicit bytes")
        (second_repo,) = _worktrees(source, "ephemeral-two")
        source_canary = source.parent / "cleanup-source-canary"
        source_canary.write_bytes(b"source sibling")
        state_root = artifact_runtime_root / _workspace_key(source)
        marker = tmp_path / f"production-{location}"
        # Only an in-place run parks at open.
        with _case_note(location), _child_at_marker(
            marker, "production-transaction", str(source), str(source), str(artifact_runtime_root),
            "PUBLIC_PARKED", location, str(marker),
        ):
            assert not (source / ".daydream").exists() and not (source / ".review-output.md").exists()
            parked, = (state_root.glob("transactions/detach-production-crash-*/parked") if location == "private"
                       else source.parent.glob(f".{source.name}.daydream-park-detach-production-crash-*"))
            assert _manifest(parked) == expected
            # The parked owner still holds the workspace lock; a live contender must not interleave.
            with pytest.raises(ArtifactVisibilityError, match="locked by another process"):
                async with open_artifact_session(_work(source, repo=second_repo), session_id="contender"):
                    pass

        async with open_artifact_session(_work(source, repo=second_repo), session_id="recovery-parked",
        ) as recovered:
            assert (recovered.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
            assert requested.read_bytes() == b"prior explicit bytes"
            recovered.register_destination(requested, label=OutputLabel.FINDINGS_OUTPUT)
            assert not requested.exists()

        assert requested.read_bytes() == b"prior explicit bytes"
        # Bytes, modes and empty directories remain exact.
        assert _manifest(source) == expected
        assert not any((state_root / "transactions").iterdir())
        assert not list(source.parent.glob(f".{source.name}.daydream-*"))
        cleanup = state_root / "cleanup"
        assert not cleanup.exists() or not any(cleanup.iterdir())
        assert source_canary.read_bytes() == b"source sibling"
    assert runtime_canary.read_bytes() == b"runtime sibling"


async def test_artifact_session_rejects_existing_session_and_malformed_transaction(
    source: Path, artifact_runtime_root: Path,
) -> None:
    async with open_artifact_session(_work(source), session_id="fixed-session"):
        pass
    with pytest.raises(ArtifactVisibilityError, match="session id already exists"):
        async with open_artifact_session(_work(source), session_id="fixed-session"):
            pass
    # Public evidence whose private run directory retention dropped still claims its id.
    earlier = source / ".daydream" / "runs" / "public-only" / RUN_DOCUMENT_NAME
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(b"earlier evidence")
    with pytest.raises(ArtifactVisibilityError, match="session id already exists"):
        async with open_artifact_session(_work(source), session_id="public-only"):
            pass
    assert earlier.read_bytes() == b"earlier evidence"

    state_root = artifact_runtime_root / _workspace_key(source)
    residue = state_root / "transactions" / "malformed"
    residue.mkdir()
    (residue / "journal.json").write_bytes(b"not-json")
    with pytest.raises(ArtifactVisibilityError, match="metadata is malformed"):
        async with open_artifact_session(_work(source), session_id="after-malformed"):
            pass
    # A parseable journal whose state is not a string is malformed too, never an unhandled error.
    _atomic_json(residue / "journal.json",
                 {"schema_version": 1, "transaction_id": "malformed", "session_id": "prior", "state": ["DETACHED"]})
    with pytest.raises(ArtifactVisibilityError, match="journal state is malformed"):
        async with open_artifact_session(_work(source), session_id="after-unhashable"):
            pass
    shutil.rmtree(residue)

    # Cleanup residue that is neither a ticket nor a retired transaction is never adopted or deleted.
    stray = state_root / "cleanup" / "operator-note.txt"
    stray.parent.mkdir(mode=0o700, exist_ok=True)
    stray.write_bytes(b"unrelated")
    with pytest.raises(ArtifactVisibilityError, match="cleanup residue is unsafe"):
        async with open_artifact_session(_work(source), session_id="after-stray-cleanup"):
            pass
    assert stray.read_bytes() == b"unrelated"


async def test_artifact_session_rejects_closed_recovery_manifest_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead session's pending publication is refused untouched when malformed, and unparked when unstarted."""
    # Each case retains private evidence in its workspace, so every case owns a fresh repository.
    for index, manifest_problem in enumerate(
        ("duplicate-path", "normalized-duplicate", "embedded-dot", "bool-schema-version", "missing-proof",
         "bool-device-proof", "unstarted"),
    ):
        await _run_case(_artifact_session_rejects_closed_recovery_manifest_untouched_case,
                        root=tmp_path / f"case-{index}", monkeypatch=monkeypatch, manifest_problem=manifest_problem)


async def _artifact_session_rejects_closed_recovery_manifest_untouched_case(source: Path, manifest_problem: str,
) -> None:
    expected = _seed_public_artifacts(source)
    state_root = _owner(source).artifact_state_root
    # A dead session parked its tree and staged a publication whose manifest or source proof is malformed,
    # or that died before its first journal entry.
    transactions = state_root / "transactions"
    transactions.mkdir(mode=0o700)
    session, publish = transactions / "detach-prior-0123456789abcdef", transactions / "publish-prior-0123456789abcdef"
    (session / "parked").mkdir(parents=True)
    publish.mkdir()
    for name in (".daydream", ".review-output.md"):
        (source / name).rename(session / "parked" / name)
    metadata = source.lstat()
    for transaction, kind in ((session, "detach"), (publish, "publish")):
        _atomic_json(transaction / "transaction-owner.json", {
            "schema_version": 1, "workspace_key": state_root.name, "transaction_id": transaction.name,
            "session_id": "prior", "kind": kind,
        })
        # A valid journal cannot authorize recovery with an absent or malformed source proof.
        if manifest_problem != "missing-proof":
            _atomic_json(transaction / "source-identity.json", {
                "schema_version": 1, "source": str(source), "inode": metadata.st_ino,
                "device": True if manifest_problem == "bool-device-proof" else metadata.st_dev,
            })
    if manifest_problem == "unstarted":
        async with open_artifact_session(_work(source), session_id="after-unstarted"):
            pass
        assert _manifest(source) == expected and not any(transactions.iterdir())
        return
    raw_path = {"duplicate-path": ".daydream", "normalized-duplicate": ".daydream/e\u0301.json",
        "embedded-dot": ".daydream/./record.json", "bool-schema-version": ".daydream/record.json",
        "missing-proof": ".daydream", "bool-device-proof": ".daydream",
    }[manifest_problem]
    entry = {"path": raw_path, "kind": "directory", "size": 0, "mode": 0o700, "sha256": None}
    if manifest_problem == "duplicate-path":
        entries = [entry, entry]
    elif manifest_problem == "normalized-duplicate":
        # Distinct, individually valid raw paths in byte order; only the NFC uniqueness guard refuses them.
        entries = [entry, {**entry, "path": ".daydream/\u00e9.json"}]
    else:
        entries = [entry]
    schema_version = True if manifest_problem == "bool-schema-version" else 1
    _atomic_json(publish / "publish-manifest.json", {"schema_version": schema_version, "entries": entries})
    _atomic_json(publish / "journal.json",
        {"schema_version": 1, "transaction_id": publish.name, "session_id": "prior", "state": "PUBLIC_STAGED"},
    )

    expected_message = "duplicate normalized paths" if manifest_problem == "normalized-duplicate" else "manifest"
    with pytest.raises(ArtifactVisibilityError, match=expected_message):
        async with open_artifact_session(_work(source), session_id="blocked"):
            pass
    assert _manifest(session / "parked") == expected
    assert not (source / ".daydream").exists() and not (source / ".review-output.md").exists()


async def test_bound_routing_rejects_same_spelling_repository_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = tmp_path / "container"
    source = container / "source"
    _init_repo(source)
    work = _work(source)

    async with open_artifact_session(work, session_id="repo-replacement") as session:
        original_lstat = Path.lstat
        denied = PermissionError("source inspection denied")

        def deny_source(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
            if path == source:
                raise denied
            return original_lstat(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "lstat", deny_source)
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError) as unavailable:
                    resolver(source, session=session)
                assert type(unavailable.value).__name__ == "ArtifactSourceUnavailableError"
                assert unavailable.value.__cause__ is denied
        moved = tmp_path / "held-source"
        source.rename(moved)
        try:
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError) as missing:
                    resolver(source, session=session)
                assert type(missing.value).__name__ == "ArtifactSourceUnavailableError"
                assert isinstance(missing.value.__cause__, FileNotFoundError)
                with pytest.raises(ArtifactVisibilityError, match="active artifact session") as foreign:
                    resolver(tmp_path / "foreign-missing", session=session)
                assert type(foreign.value) is ArtifactVisibilityError
        finally:
            moved.rename(source)
        source.rename(moved)
        source.write_bytes(b"replacement file")
        try:
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
                    resolver(source, session=session)
        finally:
            source.unlink()
            moved.rename(source)

        moved = tmp_path / "held-source-git"
        source.rename(moved)
        _init_repo(source)
        try:
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
                    resolver(source, session=session)
        finally:
            source.rename(tmp_path / "unrelated-repository")
            moved.rename(source)

        moved_container = tmp_path / "held-container"
        container.rename(moved_container)
        container.symlink_to(moved_container, target_is_directory=True)
        try:
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
                    resolver(source, session=session)
            moved_source = moved_container / "source"
            missing_source = moved_container / "missing-source"
            moved_source.rename(missing_source)
            try:
                with pytest.raises(ArtifactVisibilityError, match="active artifact session") as unsafe:
                    artifact_dir_for(source, session=session)
                assert type(unsafe.value) is ArtifactVisibilityError
            finally:
                missing_source.rename(moved_source)
        finally:
            container.unlink()
            moved_container.rename(container)

        assert not session.layout.public_daydream_dir.exists()
        assert not session.layout.public_review_output.exists()


async def test_publication_preparation_cleanup_preserves_primary_and_retains_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each case retains private evidence in its workspace, so every case owns a fresh repository.
    cases = [(primary, cleanup) for primary in ("error", "cancel") for cleanup in ("source-stage", "transaction")]
    for index, (primary_kind, cleanup_failure) in enumerate(cases):
        with monkeypatch.context() as patch:
            await _run_case(_publication_preparation_cleanup_preserves_primary_and_retains_evidence_case,
                            root=tmp_path / f"case-{index}", monkeypatch=patch, primary_kind=primary_kind,
                            cleanup_failure=cleanup_failure)


async def _publication_preparation_cleanup_preserves_primary_and_retains_evidence_case(
    source: Path, monkeypatch: pytest.MonkeyPatch, primary_kind: str, cleanup_failure: str,
) -> None:
    """A failed cleanup syscall cannot replace publication's pending error or cancellation.

    Existing preparation failures exercise successful cleanup and therefore miss
    both a retained source stage and a failed private transaction retirement.
    """
    expected = _seed_public_artifacts(source)
    owner = _owner(source)
    primary: BaseException = (
        RuntimeError("publication preparation failed") if primary_kind == "error" else anyio.get_cancelled_exc_class()()
    )
    real_open, real_unlink, real_replace = os.open, Path.unlink, os.replace
    stage: Path | None = None
    cleanup_failed = False

    def fail_public_copy(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal stage
        target = Path(path)
        if target.name == ".review-output.md" and target.parent.name == "public":
            stage, = source.parent.glob(f".{source.name}.daydream-publish-*")
            raise primary
        return real_open(path, flags, *args, **kwargs)

    def fail_stage_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal cleanup_failed
        if cleanup_failure == "source-stage" and stage is not None and path == stage / "stage-owner.json":
            cleanup_failed = True
            raise PermissionError("secondary private cleanup detail")
        real_unlink(path, *args, **kwargs)

    def fail_transaction_retirement(original: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal cleanup_failed
        path = Path(original)
        if (cleanup_failure == "transaction" and path.parent == owner.artifact_state_root / "transactions"
                and path.name.startswith("publish-")):
            cleanup_failed = True
            raise PermissionError("secondary private cleanup detail")
        real_replace(original, destination, *args, **kwargs)

    with pytest.raises(BaseException) as caught:
        async with open_artifact_session(_work(source), session_id="preparation-cleanup", owner=owner) as session:
            frozen = _freeze_run(session, "preparation-cleanup")
            frozen_manifest = _manifest(frozen.root)
            with monkeypatch.context() as patch:
                patch.setattr(os, "open", fail_public_copy)
                patch.setattr(Path, "unlink", fail_stage_unlink)
                patch.setattr(os, "replace", fail_transaction_retirement)
                _publish(session, frozen)
    assert caught.value is primary and cleanup_failed
    assert any("publication staging cleanup retained evidence (PermissionError)" == note for note in primary.__notes__)
    assert all("secondary private cleanup detail" not in note for note in primary.__notes__)
    assert _manifest(source) == expected
    assert _manifest(frozen.root) == frozen_manifest
    transaction, = (owner.artifact_state_root / "transactions").glob("publish-*")
    assert (transaction / "source-identity.json").is_file()
    assert (transaction / "publish-manifest.json").is_file()
    assert stage is not None
    if cleanup_failure == "source-stage":
        assert (stage / "stage-owner.json").is_file()
    else:
        assert not stage.exists()
        assert (owner.artifact_state_root / "cleanup" / f"{transaction.name}.json").is_file()
    # The next session finishes the retained retirement before it may proceed.
    monkeypatch.undo()
    async with open_artifact_session(_work(source), session_id="after-preparation-cleanup", owner=owner):
        assert not any((owner.artifact_state_root / "transactions").glob("publish-*"))
        cleanup = owner.artifact_state_root / "cleanup"
        assert not cleanup.exists() or not any(cleanup.iterdir())


async def test_code_workspace_missing_git_query_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each case retains private evidence in its workspace, so every case owns a fresh repository.
    for index, loss in enumerate(("marker", "source-at-query", "corrupt-marker")):
        with monkeypatch.context() as patch:
            await _run_case(_code_workspace_missing_git_query_input_case, root=tmp_path / f"case-{index}",
                            monkeypatch=patch, loss=loss)


async def _code_workspace_missing_git_query_input_case(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch, loss: str,
) -> None:
    """Required Git inputs lost around routing queries retain unavailable causes.

    Held-directory checks miss an absent worktree marker or loss at subprocess
    dispatch; malformed marker contents must still preserve genuine Git errors.
    """
    from daydream.git_ops import GitError

    (model_repo,) = _worktrees(source, "model-repo")
    work = _work(source, repo=model_repo)
    marker = model_repo / ".git"
    marker_bytes = marker.read_bytes()
    original_run = subprocess.run
    dispatched = False

    def lose_at_query(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        nonlocal dispatched
        if not dispatched and args == ["git", "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"]:
            dispatched = True
            source.rename(tmp_path / "held-source")
        return original_run(args, **kwargs)

    async with open_artifact_session(work, session_id="query-input-loss") as session:
        try:
            if loss == "marker":
                marker.unlink()
            elif loss == "corrupt-marker":
                marker.write_bytes(b"invalid Git marker\n")
            else:
                monkeypatch.setattr("daydream.git_ops.process.subprocess.run", lose_at_query)
            expected = GitError if loss == "corrupt-marker" else artifact_visibility.ArtifactSourceUnavailableError
            with pytest.raises(expected) as failure:
                session.validate_code_workspace()
            if loss != "corrupt-marker":
                assert isinstance(failure.value.__cause__, FileNotFoundError)
            assert dispatched is (loss == "source-at-query")
        finally:
            monkeypatch.setattr("daydream.git_ops.process.subprocess.run", original_run)
            if loss == "source-at-query":
                (tmp_path / "held-source").rename(source)
            marker.write_bytes(marker_bytes)
        session.validate_code_workspace()


async def test_publication_rechecks_source_between_public_and_deferred_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source loss between real installs cannot redirect the next artifact write.

    Session-entry checks miss a replacement after the public directory install,
    or before a source-internal deferred export.
    """
    for index, loss_after in enumerate(("public-directory", "deferred-syscall")):
        with monkeypatch.context() as patch:
            await _run_case(_publication_rechecks_source_case, root=tmp_path / f"case-{index}", monkeypatch=patch,
                            loss_after=loss_after)


async def _publication_rechecks_source_case(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch, loss_after: str,
) -> None:
    expected = _seed_public_artifacts(source)
    owner = _owner(source)
    primary = RuntimeError("retain publication body failure")
    replacement = source
    lost = False
    original_replace, original_link = os.replace, os.link
    publication_error: ArtifactVisibilityError | None = None

    def lose_source(target: Path) -> None:
        nonlocal lost
        trigger = source / ("findings.json" if loss_after == "deferred-syscall" else ".daydream")
        if lost or target != trigger:
            return
        lost = True
        source.rename(tmp_path / "held-source")
        _init_repo(replacement)

    def replace_then_lose(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        target = Path(dst)
        if "dst_dir_fd" in kwargs:
            target = source / target
        original_replace(src, dst, *args, **kwargs)
        if loss_after == "public-directory":
            lose_source(target)

    def link_then_lose(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        if loss_after == "deferred-syscall":
            target = Path(dst)
            if "dst_dir_fd" in kwargs:
                target = source / target
            lose_source(target)
        original_link(src, dst, *args, **kwargs)

    with pytest.raises(RuntimeError) as caught:
        async with open_artifact_session(_work(source), session_id="loss-between-installs", owner=owner) as session:
            output = session.register_destination(source / "findings.json", label=OutputLabel.FINDINGS_OUTPUT)
            assert output.write_path is not None
            output.write_path.parent.mkdir(parents=True)
            output.write_path.write_bytes(b"frozen findings\n")
            frozen = _freeze_run(session, "loss-between-installs")
            monkeypatch.setattr(os, "replace", replace_then_lose)
            monkeypatch.setattr(os, "link", link_then_lose)
            try:
                _publish(session, frozen)
            except ArtifactVisibilityError as exc:
                publication_error = exc
            raise primary
    assert caught.value is primary and lost
    assert _manifest(replacement) == ()
    assert not (replacement / "findings.json").exists()
    assert publication_error is not None
    assert _git(replacement, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert expected and _copies(_EARLIER_EVIDENCE, tmp_path, owner.artifact_state_root) == 1
    assert (frozen.root / ".explicit" / "0000" / "findings.json").read_bytes() == b"frozen findings\n"
    assert len(tuple((owner.artifact_state_root / "transactions").glob("*/journal.json"))) == 2


async def test_destination_collisions_are_rejected_without_leaving_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registered destinations cannot collide with private, model, Git or other registered roots, and failures leave no
    parent.
    """
    cases: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
        (_destination_collision_matrix_rejects_all_private_model_and_git_roots_case, {}),
        (_artifact_session_destination_collision_matrix_case, {}),
        (_absent_nested_explicit_destination_leaves_no_parent_on_failure_case, {}),
    )
    for index, (case, extra) in enumerate(cases):
        with monkeypatch.context() as patch:
            await _run_case(case, root=tmp_path / f"case-{index}", monkeypatch=patch, **extra)


async def _destination_collision_matrix_rejects_all_private_model_and_git_roots_case(tmp_path: Path, source: Path,
) -> None:
    (model_repo,) = _worktrees(source, "model-worktree")
    owner = _owner(source)

    async with open_artifact_session(_work(source, repo=model_repo), session_id="protected-destinations", owner=owner,
    ) as session:
        other_key = owner.artifact_state_root.parent / ("f" * 64)
        protected = (other_key / "output.json", owner.artifact_state_root.parent / "output.json",
            owner.operational_state_root / "output.json", owner.operational_state_root.parent / "output.json",
            model_repo / "output.json", source / ".git" / "config", model_repo / ".git",
            session.layout.repo_git_dir / "output.json", session.layout.git_common_dir / "output.json",
        )
        for index, requested in enumerate(protected):
            with pytest.raises(ArtifactVisibilityError, match="overlap|Git|model"):
                session.register_destination(requested, label=OutputLabel.FINDINGS_OUTPUT)
            assert not (session.layout.live_root / ".explicit" / f"{index:04d}").exists()

    async with open_artifact_session(_work(source), session_id="in-source-control") as session:
        route = session.register_destination(source / "safe-untracked.json", label=OutputLabel.FINDINGS_OUTPUT)
        assert route.write_path is not None
        assert route.write_path.is_relative_to(session.layout.live_root)


async def _artifact_session_rejects_duplicate_filesystem_aliases_case(source: Path) -> None:
    runs = source / ".daydream" / "runs"
    for alias_kind, first_name, second_name in (("hardlink", "original", "alias"),
                                                ("normalized-name", "e\u0301", "\u00e9")):
        runs.mkdir(parents=True)
        first, second = runs / first_name, runs / second_name
        first.write_bytes(b"content")
        if alias_kind == "hardlink":
            os.link(first, second)
        else:
            second.write_bytes(b"other")
        # A filesystem that canonicalizes both Unicode spellings has no duplicate to reject.
        if len({child.name for child in runs.iterdir()}) == 2:
            with pytest.raises(ArtifactVisibilityError, match="duplicate"):
                async with open_artifact_session(_work(source), session_id=alias_kind):
                    pass
            assert first.read_bytes() == b"content"
        shutil.rmtree(source / ".daydream")


async def _artifact_session_rejects_symlinked_source_and_runtime_ancestry_case(tmp_path: Path, source: Path) -> None:
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ArtifactVisibilityError, match="symlink"):
        async with open_artifact_session(_work(source, repo=alias), session_id="source-alias"):
            pass

    actual_runtime = tmp_path / "actual-runtime"
    actual_runtime.mkdir()
    runtime_alias = tmp_path / "runtime-alias"
    runtime_alias.symlink_to(actual_runtime, target_is_directory=True)
    locations = PrivateRootLocations(artifact_runtime=runtime_alias, operational_workspaces=tmp_path / "operations")
    with pytest.raises(ArtifactVisibilityError, match="runtime ancestry contains a symlink"):
        resolve_private_workspace_owner(source, locations=locations)


async def test_artifact_session_detects_source_mutation_without_deleting_unique_bytes(
    tmp_path: Path, source: Path,
) -> None:
    """A tree that appears in the checkout while the prior one is parked, or a parked tree changed during the run,
    refuses publication; both versions survive and the refusal names both paths.
    """
    expected = _seed_public_artifacts(source)
    owner = _owner(source)
    concurrent = source / ".daydream" / "concurrent.bin"
    with pytest.raises(ArtifactVisibilityError, match="both are kept") as caught:
        async with open_artifact_session(_work(source), session_id="competing", owner=owner) as session:
            frozen = _freeze_run(session, "competing")
            concurrent.parent.mkdir()
            concurrent.write_bytes(b"unique concurrent bytes")
            with pytest.raises(ArtifactVisibilityError, match="appeared in the checkout"):
                _publish(session, frozen)
    parked, = owner.artifact_state_root.glob("transactions/detach-competing-*/parked")
    assert f"{source / '.daydream'} (kept in the checkout) and {parked / '.daydream'}" in str(caught.value)
    assert concurrent.read_bytes() == b"unique concurrent bytes"
    assert _manifest(parked) == tuple(entry for entry in expected if entry.path != ".review-output.md")
    assert (frozen.root / ".daydream" / "runs" / "competing" / RUN_DOCUMENT_NAME).is_file()
    # Every open refuses the same way until the competing tree is moved away; then the prior tree returns exactly.
    with pytest.raises(ArtifactVisibilityError, match="both are kept"):
        async with open_artifact_session(_work(source), session_id="still-competing", owner=owner):
            pass
    shutil.rmtree(concurrent.parent)
    async with open_artifact_session(_work(source), session_id="after-competing", owner=owner):
        pass
    assert _manifest(source) == expected

    # A shared entry changed during the run refuses publication, both versions survive: the user's change in the
    # checkout (parked for an in-place run, still visible otherwise) and the run's own outputs in its frozen copy.
    for in_place, mutation in ((True, "edit"), (False, "edit"), (False, "delete")):
        changed_source = tmp_path / f"changed-source-{in_place}-{mutation}"
        _init_repo(changed_source)
        _seed_public_artifacts(changed_source)
        changed_owner = _owner(changed_source)
        work = (_work(changed_source) if in_place
                else _work(changed_source, repo=_worktrees(changed_source, f"wt-{mutation}")[0]))
        with _case_note(f"in_place={in_place} mutation={mutation}"):
            async with open_artifact_session(work, session_id="changed", owner=changed_owner) as session:
                frozen = _freeze_run(session, "changed")
                if in_place:
                    edited, = changed_owner.artifact_state_root.glob("transactions/detach-changed-*/parked")
                else:
                    edited = changed_source
                    assert (changed_source / ".daydream" / "deep" / "prior.md").is_file()
                if mutation == "edit":
                    (edited / ".daydream" / "deep" / "prior.md").write_bytes(b"edited during the run\n")
                else:
                    shutil.rmtree(edited / ".daydream")
                    (edited / ".review-output.md").unlink()
                with pytest.raises(ArtifactVisibilityError, match="changed during the run"):
                    _publish(session, frozen)
            if mutation == "edit":
                assert (changed_source / ".daydream/deep/prior.md").read_bytes() == b"edited during the run\n"
                assert not (changed_source / ".daydream" / "runs" / "changed").exists()
            else:
                assert _manifest(changed_source) == ()
            assert (frozen.root / ".daydream" / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"


async def test_publication_preparation_failure_restores_source_without_orphan_state(
    source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _seed_public_artifacts(source)
    session_id = "publish-preparation"
    owner = _owner(source)
    collision = source.parent / f".{source.name}.daydream-publish-publish-{session_id}-fixed-token"
    collision.mkdir()
    canary = collision / "external-canary"
    canary.write_bytes(b"must survive")

    async with open_artifact_session(_work(source), session_id=session_id, owner=owner) as session:
        frozen = _freeze_run(session, session_id)
        monkeypatch.setattr(secrets, "token_hex", lambda _size: "fixed-token")
        with pytest.raises(ArtifactVisibilityError, match="source-stage collision"):
            _publish(session, frozen)

    assert _manifest(source) == expected
    assert canary.read_bytes() == b"must survive"
    assert not any((owner.artifact_state_root / "transactions").iterdir())

    canary.unlink()
    collision.rmdir()
    async with open_artifact_session(_work(source), session_id="after-preparation-failure", owner=owner) as recovered:
        assert (recovered.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"


async def test_recoverable_detach_failure_restores_before_unlock_and_preserves_primary(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure once the tree is parked unparks it byte-exact before unlock; an interrupted unpark resumes."""
    expected = _seed_public_artifacts(source)
    primary: BaseException = anyio.get_cancelled_exc_class()()

    def stop_parked(state: str) -> None:
        if state == "PUBLIC_PARKED":
            raise primary

    monkeypatch.setattr(artifact_transactions, "_transition_observer", stop_parked)
    with pytest.raises(BaseException) as caught:
        async with open_artifact_session(_work(source), session_id="parked-primary"):
            pass
    assert caught.value is primary
    assert _manifest(source) == expected
    monkeypatch.setattr(artifact_transactions, "_transition_observer", None)
    async with open_artifact_session(_work(source), session_id="after-parked-primary"):
        assert not (source / ".daydream").exists()

    # An unpark interrupted after linking .review-output.md back, before dropping its parked name, resumes next open.
    link_source = tmp_path / "link-source"
    _init_repo(link_source)
    link_expected = _seed_public_artifacts(link_source)
    unlink = Path.unlink
    interrupted: list[Path] = []

    def stop_after_link(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == ".review-output.md" and path.parent.name == "parked" and not interrupted:
            interrupted.append(path)
            raise OSError("interrupted after link")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", stop_after_link)
    with pytest.raises(ArtifactVisibilityError, match="atomic install"):
        async with open_artifact_session(_work(link_source), session_id="link-primary"):
            pass
    assert interrupted and interrupted[0].samefile(link_source / ".review-output.md")
    monkeypatch.setattr(Path, "unlink", unlink)
    async with open_artifact_session(_work(link_source), session_id="after-link-primary") as reopened:
        assert (reopened.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
    assert _manifest(link_source) == link_expected


async def test_recoverable_publication_failure_reconciles_before_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each case retains private evidence in its workspace, so every case owns a fresh repository.
    cases = [(transition, exception_kind, loss)
             for transition, exception_kind in (("PUBLIC_STAGED", "keyboard"), ("PUBLIC_CARRIED", "cancel"),
                                                ("PUBLIC_INSTALLED", "keyboard"), ("PUBLIC_VERIFIED", "cancel"))
             for loss in (False, True)]
    # A failure between installs, after .daydream was renamed in, rolls back from existence checks alone.
    cases += [("install:.review-output.md", "keyboard", False), ("install:findings.md", "cancel", False)]
    rows: list[tuple[str, str, bool, str, str | None]] = [(*case, "private", None) for case in cases]
    # Recovery itself fails once and is rerun: after rolling forward before its session settles, and, with the
    # tree parked beside the checkout, after rolling back once its stage is gone.
    rows += [("PUBLIC_INSTALLED", "keyboard", False, "private", "discard"),
             ("PUBLIC_CARRIED", "cancel", False, "beside", "retire")]
    for index, (transition, exception_kind, source_loss, park, recovery_fault) in enumerate(rows):
        with monkeypatch.context() as patch:
            await _run_case(_recoverable_publication_failure_reconciles_before_unlock_case,
                            root=tmp_path / f"case-{index}", monkeypatch=patch, transition=transition,
                            exception_kind=exception_kind, source_loss=source_loss, park=park,
                            recovery_fault=recovery_fault)


def _fail_once(monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    """Fail recovery once: at discarding the replaced shared set, or at retiring the publication."""
    owner: Any = artifact_publication if step == "discard" else artifact_transactions
    name = "_discard_parked" if step == "discard" else "_retire_transaction"
    original = getattr(owner, name)
    failed: list[str] = []

    def fail(*args: Any, **kwargs: Any) -> None:
        transaction = args[1] if step != "discard" else None
        if not failed and (transaction is None or transaction.name.startswith("publish-")):
            failed.append(step)
            raise ArtifactVisibilityError(f"recovery fault at {step}")
        original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail)


async def _recoverable_publication_failure_reconciles_before_unlock_case(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch,
    transition: str, exception_kind: str, source_loss: bool, park: str, recovery_fault: str | None,
) -> None:
    """Recovery preserves its primary failure and cannot replay into a new checkout.

    Existing durable-transition recovery kept the original source alive, missing
    a replacement between installation and recovery with identical path spelling.
    """
    expected = _seed_public_artifacts(source)
    owner = _owner(source)
    session_id = f"recover-{transition.lower()}"
    payload = _payload(session_id)
    primary: BaseException = (anyio.get_cancelled_exc_class()()
        if exception_kind == "cancel"
        else KeyboardInterrupt("publication primary")
    )

    def fail_at_transition(state: str) -> None:
        if state == transition:
            if source_loss:
                source.rename(tmp_path / "held-source")
                _init_repo(source)
            if recovery_fault is not None:
                _fail_once(monkeypatch, recovery_fault)
            raise primary

    if park == "beside":
        monkeypatch.setattr(artifact_publication, "_move_out", _cross_device_park(artifact_publication._move_out))

    caught: BaseException | None = None
    try:
        async with open_artifact_session(_work(source), session_id=session_id) as session:
            explicit = session.register_destination(source / "explicit" / "findings.md",
                                                    label=OutputLabel.FINDINGS_OUTPUT)
            explicit.write_path.parent.mkdir(parents=True)
            explicit.write_path.write_bytes(b"explicit findings")
            frozen = _freeze_run(session, session_id, payload=payload)
            monkeypatch.setattr(artifact_transactions, "_transition_observer", fail_at_transition)
            install = artifact_publication._install_staged_path
            failed: list[Path] = []

            def fail_installing(staged: Path, target: Path, **kwargs: Any) -> None:
                if transition == f"install:{target.name}" and not failed:
                    failed.append(target)
                    raise primary
                install(staged, target, **kwargs)

            monkeypatch.setattr(artifact_publication, "_install_staged_path", fail_installing)
            _publish(session, frozen)
    except BaseException as exc:
        caught = exc
    assert caught is primary
    if recovery_fault is not None:
        assert any("artifact recovery retained" in note for note in primary.__notes__)
        assert any((owner.artifact_state_root / "transactions").iterdir())
    if source_loss:
        assert _manifest(source) == ()
        assert (source / "source.txt").read_bytes() == b"source canary\n"
        assert _git(source, "status", "--porcelain=v1", "--untracked-files=all") == ""
        assert _copies(_EARLIER_EVIDENCE, tmp_path, owner.artifact_state_root) == 1
        assert (frozen.root / ".daydream" / "runs" / session_id / RUN_DOCUMENT_NAME).read_bytes() == payload
        journals = tuple((owner.artifact_state_root / "transactions").glob("*/journal.json"))
        # A failure while staging discards its unstarted publication; the session journal remains.
        assert len(journals) == (1 if transition == "PUBLIC_STAGED" else 2)
        assert any("artifact recovery retained" in note for note in primary.__notes__)
        monkeypatch.setattr(artifact_transactions, "_transition_observer", None)
        with pytest.raises(ArtifactVisibilityError, match="identity"):
            async with open_artifact_session(_work(source), session_id=f"after-{session_id}", owner=owner):
                pass
        assert all(journal.is_file() for journal in journals)
        assert _manifest(source) == ()
        assert _git(source, "status", "--porcelain=v1", "--untracked-files=all") == ""
        return
    if transition in {"PUBLIC_INSTALLED", "PUBLIC_VERIFIED"}:
        # Recovery rolls an installed publication forward, explicit destinations included.
        assert (source / ".daydream" / "runs" / session_id / RUN_DOCUMENT_NAME).read_bytes() == payload
        assert (source / "explicit" / "findings.md").read_bytes() == b"explicit findings"
        assert _shared(_manifest(source)) == _shared(expected)
        assert _copies(_EARLIER_EVIDENCE, tmp_path, owner.artifact_state_root) == 1
    else:
        assert _manifest(source) == expected
        assert not (source / "explicit").exists()  # Rolled back, including the parent publication created.
    monkeypatch.setattr(artifact_transactions, "_transition_observer", None)
    async with open_artifact_session(_work(source), session_id=f"after-{session_id}"):
        assert not (source / ".daydream").exists()


async def _artifact_session_destination_collision_matrix_case(tmp_path: Path, source: Path) -> None:
    tracked = source / "tracked-dir" / "file.txt"
    tracked.parent.mkdir()
    tracked.write_text("tracked", encoding="utf-8")
    _git(source, "add", "tracked-dir/file.txt")
    _git(source, "commit", "-m", "tracked destination")
    parent_file = tmp_path / "parent-file"
    parent_file.write_bytes(b"file")
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(tmp_path, target_is_directory=True)
    wrong_type = tmp_path / "wrong-type"
    wrong_type.mkdir()

    async with open_artifact_session(_work(source), session_id="destination-matrix") as session:
        public = session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        assert public.write_path == session.daydream_dir
        findings, dump = OutputLabel.FINDINGS_OUTPUT, OutputLabel.DUMP_DIRECTORY
        rejected: tuple[tuple[Path, OutputLabel, str], ...] = (
            (source / "tracked-dir", dump, "tracked artifact destination"),
            (source / "tracked-dir" / "file.txt" / "child", findings, "tracked artifact destination"),
            (parent_file / "child.json", findings, "ancestry is not a directory"),
            (symlink_parent / "child.json", findings, "ancestry contains a symlink"),
            (wrong_type, findings, "wrong filesystem type"),
            (session.layout.state_root / "leak.json", findings, "overlaps private"),
            (source, dump, "replace the source root"),
            (source / ".daydream" / "nested.json", findings, "public compatibility root"),
            (source / "not-public", OutputLabel.PUBLIC_REVIEW_OUTPUT, "invalid destination"),
        )
        for requested, label, message in rejected:
            with pytest.raises(ArtifactVisibilityError, match=message):
                session.register_destination(requested, label=label)


async def _absent_nested_explicit_destination_leaves_no_parent_on_failure_case(source: Path) -> None:
    requested = source / "new parent" / "nested" / "findings.json"
    async with open_artifact_session(_work(source), session_id="absent-explicit") as session:
        routed = session.register_destination(requested, label=OutputLabel.FINDINGS_OUTPUT)
        routed.write_path.parent.mkdir(parents=True)
        routed.write_path.write_bytes(b"failed run")
        assert not requested.exists()
    assert not requested.exists()
    assert not (source / "new parent").exists()


async def test_artifact_session_rejects_forged_snapshot_object(tmp_path: Path, source: Path) -> None:
    session_id = "forged-snapshot"
    async with open_artifact_session(_work(source), session_id=session_id) as session:
        snapshot = _freeze_run(session, session_id)
        forged = ArtifactTreeSnapshot(
            session_id=snapshot.session_id, workspace_key=snapshot.workspace_key, root=tmp_path,
            manifest=snapshot.manifest, destinations=snapshot.destinations,
        )
        with pytest.raises(ArtifactVisibilityError, match="snapshot identity mismatch"):
            _publish(session, forged)


async def test_public_subtree_trajectory_uses_whole_daydream_transaction(
    source: Path,
) -> None:
    disposition = artifact_visibility.ArtifactDisposition.ROLLBACK
    status = "complete"
    _seed_public_artifacts(source)
    custom_relative = Path("deep") / "custom outputs" / "nested root.json"
    requested = source / ".daydream" / custom_relative
    requested.parent.mkdir()
    requested.write_bytes(b"prior full")
    partial_requested = requested.with_suffix(requested.suffix + ".partial")
    partial_requested.write_bytes(b"prior partial")
    unrelated = requested.parent / "unrelated.bin"
    unrelated.write_bytes(b"unrelated")
    baseline = _manifest(source)
    session_id = f"custom-{disposition.value}"
    root_bytes = _payload(session_id)
    child_id = f"{session_id}:child"
    child_bytes = _payload(session_id, child_id)

    async with open_artifact_session(_work(source), session_id=session_id) as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        session.register_destination(
            source / ".review-output.md", label=OutputLabel.PUBLIC_REVIEW_OUTPUT,
        )
        assert not (source / ".daydream").exists()

        route = session.register_trajectory_output(requested)
        private_full = session.daydream_dir / custom_relative
        private_partial = private_full.with_suffix(private_full.suffix + ".partial")
        assert route.full.requested == requested
        assert route.partial.requested == partial_requested
        assert route.full.write_path == route.full.frozen_path == private_full
        assert route.partial.write_path == route.partial.frozen_path == private_partial
        assert (session.durable_path_for(private_full, repo=source),
                session.durable_path_for(private_partial, repo=source)) == (requested, partial_requested)
        assert (session.live_path_for(requested, repo=source),
                session.live_path_for(partial_requested, repo=source)) == (private_full, private_partial)
        assert route.run_dir == session.daydream_dir / "runs" / session_id
        assert private_full.read_bytes() == b"prior full"
        assert private_partial.read_bytes() == b"prior partial"
        assert (session.daydream_dir / custom_relative.parent / "unrelated.bin").read_bytes() == b"unrelated"
        assert (session.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"

        selected = route.full
        assert route.explicit
        root_document = _reference(route.run_dir / RUN_DOCUMENT_NAME, session_id, root_bytes)
        session.write_trajectory_document(
            route, TrajectoryDocumentSnapshot(session_id, root_document.path, root_bytes), cast(Any, status))
        assert cast(Path, selected.write_path).read_bytes() == root_bytes
        child_path = route.run_dir / SIBLINGS_DIRNAME / "child.otlp.jsonl"
        child_document = _reference(child_path, child_id, child_bytes)
        # Only the root document has an explicit output; children stay in the run's private directory.
        with pytest.raises(ArtifactVisibilityError, match="only the root"):
            session.write_trajectory_document(
                route, TrajectoryDocumentSnapshot(child_id, child_path, child_bytes), cast(Any, status))
        assert not (source / ".daydream").exists()

        frozen = session.freeze(_snapshot(session_id, (root_document, child_document), status=status))
        _publish(session, frozen, disposition)

    published_full = requested.read_bytes()
    published_partial = partial_requested.read_bytes()
    if disposition is artifact_visibility.ArtifactDisposition.COMPLETE:
        assert (published_full, published_partial) == (root_bytes, b"prior partial")
    elif disposition is artifact_visibility.ArtifactDisposition.PARTIAL_EVIDENCE:
        assert (published_full, published_partial) == (b"prior full", root_bytes)
    else:
        assert _manifest(source) == baseline
        assert (published_full, published_partial) == (b"prior full", b"prior partial")
    assert unrelated.read_bytes() == b"unrelated"
    published_child = source / ".daydream" / "runs" / session_id / SIBLINGS_DIRNAME / "child.otlp.jsonl"
    assert published_child.exists() is (disposition is not artifact_visibility.ArtifactDisposition.ROLLBACK)

    async with open_artifact_session(_work(source), session_id=f"reopen-{disposition.value}") as reopened:
        imported_full = reopened.daydream_dir / custom_relative
        imported_partial = imported_full.with_suffix(imported_full.suffix + ".partial")
        assert imported_full.read_bytes() == published_full
        assert imported_partial.read_bytes() == published_partial
        assert (reopened.daydream_dir / custom_relative.parent / "unrelated.bin").read_bytes() == b"unrelated"


def _restore_stage_leaks(source: Path) -> list[str]:
    """Leftover ``.<source>.daydream-restore-*`` stage names beside ``source``."""
    prefix = f".{source.name}.daydream-restore-"
    return sorted(child.name for child in source.parent.iterdir() if child.name.startswith(prefix))


async def test_session_open_refuses_earlier_format_transactions_and_retires_stale_canonical(source: Path) -> None:
    """A pending canonical-format transaction is preserved and refused with recovery steps.

    Once it is resolved, the next open deletes the stale canonical copy and proceeds.
    """
    baseline = _seed_public_artifacts(source)
    owner = _owner(source)
    state_root = owner.artifact_state_root
    canonical = state_root / "canonical"
    (canonical / ".daydream").mkdir(parents=True)
    (canonical / ".daydream" / "prior.md").write_bytes(b"canonical bytes")
    _atomic_json(state_root / "canonical-manifest.json", {"schema_version": 1, "entries": []})
    for kind, state in (("detach", "DETACHED"), ("publish", "PUBLISH_BACKED_UP")):
        pending = state_root / "transactions" / f"{kind}-earlier-0123456789abcdef"
        pending.parent.mkdir(mode=0o700, exist_ok=True)
        pending.mkdir()
        _atomic_json(pending / "journal.json",
                     {"schema_version": 1, "transaction_id": pending.name, "session_id": "earlier", "state": state})
        with _case_note(state), pytest.raises(ArtifactVisibilityError, match="earlier Daydream") as refused:
            async with open_artifact_session(_work(source), session_id=f"refused-{kind}", owner=owner):
                pass
        assert pending.name in str(refused.value) and str(canonical) in str(refused.value)
        assert "Run that earlier release once" in str(refused.value)
        assert _manifest(source) == baseline and (pending / "journal.json").is_file()
        assert (canonical / ".daydream" / "prior.md").read_bytes() == b"canonical bytes"
        shutil.rmtree(pending)

    async with open_artifact_session(_work(source), session_id="after-earlier", owner=owner):
        pass
    assert not canonical.exists() and not (state_root / "canonical-manifest.json").exists()
    assert _manifest(source) == baseline


async def test_failed_destination_restore_preserves_the_transaction_and_reopens(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Move failed restores to unreconciled storage, retaining every baseline byte.

    Report the retained location and allow a fresh session without replaying the failure.
    """
    baseline = _seed_public_artifacts(source)
    owner = _owner(source)
    state_root = owner.artifact_state_root
    transactions = state_root / "transactions"
    findings = tmp_path / "external" / "findings.json"
    findings.parent.mkdir()
    findings.write_bytes(b"prior findings")

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise ArtifactVisibilityError("synthetic destination restore failure")

    with monkeypatch.context() as patched:
        patched.setattr(artifact_publication, "_restore_destination_records", refuse)
        with pytest.raises(ArtifactVisibilityError, match="synthetic destination restore failure") as caught:
            async with open_artifact_session(_work(source), session_id="unrestorable", owner=owner) as session:
                session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
                session.register_destination(source / ".review-output.md", label=OutputLabel.PUBLIC_REVIEW_OUTPUT)
                session.register_destination(findings, label=OutputLabel.FINDINGS_OUTPUT)
                frozen = _freeze_run(session, "unrestorable")
                _publish(session, frozen, artifact_visibility.ArtifactDisposition.ROLLBACK)

    # The operator is told which transaction failed, in which workspace, and
    # where its baselines were kept.
    preserved = state_root / "unreconciled"
    message = str(caught.value)
    retained = [child.name for child in preserved.iterdir()]
    assert len(retained) == 1
    assert retained[0] in message
    assert str(state_root) in message
    assert str(preserved / retained[0]) in message

    assert _manifest(source) == baseline
    assert (preserved / retained[0] / "destination-0000-baseline" / "findings.json").read_bytes() == (b"prior findings"
    )
    assert not any(transactions.iterdir())
    assert _restore_stage_leaks(source) == []

    async with open_artifact_session(_work(source), session_id="after-unrestorable", owner=owner) as reopened:
        assert (reopened.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
    assert not any(transactions.iterdir())


async def test_custom_public_subtree_requires_exact_owner_and_unknown_entries_are_carried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom public subtrees need their exact public owner; unknown top-level entries are carried unchanged."""
    cases: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
        (_public_subtree_trajectory_requires_exact_public_owner_case, {}),
        (_unknown_public_entries_are_carried_case, {}),
    )
    for index, (case, extra) in enumerate(cases):
        with monkeypatch.context() as patch:
            await _run_case(case, root=tmp_path / f"case-{index}", monkeypatch=patch, **extra)


async def _public_subtree_trajectory_requires_exact_public_owner_case(source: Path) -> None:
    requested = source / ".daydream" / "custom.json"

    async with open_artifact_session(_work(source), session_id="missing-owner") as session:
        with pytest.raises(ArtifactVisibilityError, match="overlaps a public compatibility root"):
            session.register_trajectory_output(requested)
        # An explicit in-source path outside the public roots is deferred to publication, never written live.
        explicit = source / "explicit" / RUN_DOCUMENT_NAME
        route = session.register_trajectory_output(explicit)
        assert route.full.write_path.is_relative_to(session.layout.live_root) and not explicit.parent.exists()


async def _unknown_public_entries_are_carried_case(source: Path) -> None:
    _state_root, selected, payload = await _publish_custom_public_trajectory(source, session_id="custom-carried")
    sibling = source / ".daydream" / "not-session-owned" / "opaque.bin"
    sibling.parent.mkdir()
    sibling.write_bytes(b"unowned bytes")
    expected = _manifest(source)

    async with open_artifact_session(_work(source), session_id="carry-unknown") as session:
        assert (session.daydream_dir / "not-session-owned" / "opaque.bin").read_bytes() == b"unowned bytes"
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        _publish(session, _freeze_run(session, "carry-unknown"))

    assert _shared(_manifest(source)) == _shared(expected)
    assert selected.read_bytes() == payload


async def test_public_subtree_trajectory_rejects_unsafe_private_counterpart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each case retains private evidence in its workspace, so every case owns a fresh repository.
    for index, (location, kind) in enumerate((("ancestor", "file"), ("leaf", "symlink"), ("leaf", "directory"))):
        await _run_case(_public_subtree_trajectory_rejects_unsafe_private_counterpart_case,
                        root=tmp_path / f"case-{index}", monkeypatch=monkeypatch, location=location, kind=kind)


async def _public_subtree_trajectory_rejects_unsafe_private_counterpart_case(
    tmp_path: Path, source: Path, location: str, kind: str,
) -> None:
    baseline = _seed_public_artifacts(source)
    requested = source / ".daydream" / "nested" / "custom.json"
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "canary.bin").write_bytes(b"unrelated bytes")
    outside_before = artifact_filesystem.manifest_tree(outside)

    async with open_artifact_session(_work(source), session_id=f"{location}-{kind}") as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        target = {"root": session.daydream_dir, "ancestor": session.daydream_dir / "nested",
            "leaf": session.daydream_dir / "nested" / "custom.json"}[location]
        saved_root = session.layout.live_root / "saved-daydream"
        if location == "root":
            target.rename(saved_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "directory":
                target.mkdir()
            elif kind == "file":
                target.write_bytes(b"not a directory")
            else:
                destination = outside if location == "root" else session.daydream_dir / "deep"
                if location == "leaf":
                    destination /= "prior.md"
                target.symlink_to(destination, target_is_directory=location != "leaf")
            with pytest.raises(ArtifactVisibilityError, match="private trajectory"):
                session.register_trajectory_output(requested)
            assert session._trajectory_route is None
            assert artifact_filesystem.manifest_tree(outside) == outside_before
        finally:
            if location == "root":
                target.unlink()
                saved_root.rename(target)

    assert _manifest(source) == baseline
    assert artifact_filesystem.manifest_tree(outside) == outside_before


async def test_live_external_destinations_are_probed_and_isolated_before_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External destinations are probed live and kept disjoint from model working directories before any producer runs.

    """
    cases: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
        (_independent_model_cwd_rejects_live_external_destination_overlap_case, {}),
        (_live_external_capability_probe_covers_exchange_link_and_missing_parent_case, {}),
        (_live_external_capability_failure_rejects_route_before_producer_case,
         {"operation": "_external_link", "error": PermissionError("test link refusal"), "message": "link probe"}),
    )
    for index, (case, extra) in enumerate(cases):
        with monkeypatch.context() as patch:
            await _run_case(case, root=tmp_path / f"case-{index}", monkeypatch=patch, **extra)


async def _independent_model_cwd_rejects_live_external_destination_overlap_case(tmp_path: Path, source: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    requested = external / "nested" / RUN_DOCUMENT_NAME
    nested = requested.parent
    nested.mkdir()
    disjoint = tmp_path / "independent-audit"
    disjoint.mkdir()

    async with open_artifact_session(_work(source), session_id="independent-cwd") as session:
        session.register_trajectory_output(requested)
        with pytest.raises(ArtifactVisibilityError, match="live external"):
            artifact_visibility.assert_model_cwd_clean(external)
        with pytest.raises(ArtifactVisibilityError, match="live external"):
            artifact_visibility.assert_model_cwd_clean(nested)
        artifact_visibility.assert_model_cwd_clean(disjoint)


async def _live_external_capability_probe_covers_exchange_link_and_missing_parent_case(tmp_path: Path, source: Path,
) -> None:
    requested = tmp_path / "missing" / "nested" / RUN_DOCUMENT_NAME
    canary = tmp_path / "probe-canary"
    canary.write_bytes(b"unrelated")

    async with open_artifact_session(_work(source), session_id="probe-success") as session:
        route = session.register_trajectory_output(requested)
        assert route.full.delivery is artifact_visibility.DestinationDelivery.LIVE_EXTERNAL
        records = cast(
            list[dict[str, object]], _load_json(session._detach_transaction / "external-entries.json")["entries"],
        )
        purposes = {cast(str, record["purpose"]) for record in records}
        assert {"probe_exchange_a", "probe_exchange_b", "probe_link_target", "missing_parent"} <= purposes
        assert all(record["lifecycle"] == "retired"
            for record in records
            if cast(str, record["purpose"]).startswith("probe_")
        )
        assert not any(path.name.startswith(".daydream-probe-") for path in requested.parent.iterdir())

    assert not requested.parent.exists()
    assert canary.read_bytes() == b"unrelated"


async def _live_external_capability_failure_rejects_route_before_producer_case(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch, operation: str, error: Exception, message: str,
) -> None:
    requested, _ = _external_target(tmp_path / "external", prior=False)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(artifact_external, operation, refuse)
    async with open_artifact_session(_work(source), session_id="probe-refused") as session:
        with pytest.raises(ArtifactVisibilityError, match=message):
            session.register_trajectory_output(requested)
    assert not requested.exists()
    assert not any(path.name.startswith(".daydream-probe-") for path in requested.parent.iterdir())


async def test_absent_first_publication_refuses_concurrent_target_without_exchange(tmp_path: Path, source: Path,
) -> None:
    requested, _ = _external_target(tmp_path / "external", prior=False)
    unique = b"concurrent target"
    payload = _payload("absent-race")

    with pytest.raises(ArtifactVisibilityError, match="conflict"):
        async with open_artifact_session(_work(source), session_id="absent-race") as session:
            route = session.register_trajectory_output(requested)
            requested.write_bytes(unique)
            with pytest.raises(ArtifactVisibilityError, match="no-clobber"):
                session.write_trajectory_document(
                    route, TrajectoryDocumentSnapshot("absent-race", requested, payload), "complete",
                )
    assert requested.read_bytes() == unique
    stages = list(requested.parent.glob(".daydream-output-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == payload


class _FailAfterExchange:
    def __init__(self, *, impossible: bool = False) -> None:
        self.real = artifact_external._AtomicNameExchange()
        self.calls = 0
        self.impossible = impossible

    def call(self, parent_fd: int, staged_name: str, target_name: str) -> Any:
        self.calls += 1
        if self.calls <= 2:
            return self.real.call(parent_fd, staged_name, target_name)
        if self.impossible:
            return artifact_external._NameExchangeResult(7, None)
        result = self.real.call(parent_fd, staged_name, target_name)
        assert result.result == 0
        return artifact_external._NameExchangeResult(-1, 5)


class _ReplaceBeforeSuccessfulExchange:
    def __init__(self, requested: Path, unexpected: bytes, events: list[str]) -> None:
        self.real = artifact_external._AtomicNameExchange()
        self.requested = requested
        self.unexpected = unexpected
        self.events = events
        self.calls = 0

    def call(self, parent_fd: int, staged_name: str, target_name: str) -> Any:
        self.calls += 1
        if self.calls <= 2:
            return self.real.call(parent_fd, staged_name, target_name)
        if self.calls == 3:
            self.requested.write_bytes(self.unexpected)
        else:
            assert "REVERSAL_ATTEMPTED" in self.events
        return self.real.call(parent_fd, staged_name, target_name)


class _ReplaceWithDirectoryBeforeSuccessfulExchange:
    def __init__(self, requested: Path) -> None:
        self.real = artifact_external._AtomicNameExchange()
        self.requested = requested
        self.calls = 0

    def call(self, parent_fd: int, staged_name: str, target_name: str) -> Any:
        self.calls += 1
        if self.calls == 3:
            self.requested.unlink()
            self.requested.mkdir()
        elif self.calls > 3:
            raise AssertionError("nonregular displaced target must not be reversed")
        return self.real.call(parent_fd, staged_name, target_name)


async def test_external_exchange_outcomes_never_report_false_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonzero, impossible and displaced exchange outcomes fail closed and preserve every occupant."""
    cases: tuple[tuple[Callable[..., Any], dict[str, Any]], ...] = (
        (_exchange_nonzero_never_reports_success_or_falls_back_case, {"impossible": False}),
        (_exchange_nonzero_never_reports_success_or_falls_back_case, {"impossible": True}),
        (_zero_exchange_with_unexpected_displaced_target_reverses_once_and_fails_case, {}),
        (_zero_exchange_with_displaced_directory_records_conflict_without_reversal_case, {}),
    )
    for index, (case, extra) in enumerate(cases):
        with monkeypatch.context() as patch:
            await _run_case(case, root=tmp_path / f"case-{index}", monkeypatch=patch, **extra)


async def _exchange_nonzero_never_reports_success_or_falls_back_case(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch, impossible: bool,
) -> None:
    # Each case retains a closed conflict in its workspace, so rows need independent sources.
    requested, _ = _external_target(tmp_path / f"external-{impossible}")
    exchange = _FailAfterExchange(impossible=impossible)
    monkeypatch.setattr(artifact_external, "_name_exchange_factory", lambda: exchange)
    payload = _payload(f"exchange-{impossible}")

    expected_message = "failed after mutation" if not impossible else "atomic exchange"
    with pytest.raises(ArtifactVisibilityError, match=expected_message):
        async with open_artifact_session(_work(source), session_id=f"exchange-{impossible}") as session:
            route = session.register_trajectory_output(requested)
            session.write_trajectory_document(
                route, TrajectoryDocumentSnapshot(f"exchange-{impossible}", requested, payload), "complete",
            )
    assert requested.read_bytes() == b"prior full"
    assert exchange.calls == (4 if not impossible else 3)


async def _zero_exchange_with_unexpected_displaced_target_reverses_once_and_fails_case(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested, _ = _external_target(tmp_path / "external")
    unexpected = b"unexpected displaced target"
    events: list[str] = []
    exchange = _ReplaceBeforeSuccessfulExchange(requested, unexpected, events)
    monkeypatch.setattr(artifact_external, "_name_exchange_factory", lambda: exchange)
    monkeypatch.setattr(artifact_external, "_external_entry_observer",
        lambda state, purpose, _path: events.append(state)
        if purpose == "publication_stage"
        else None,
    )
    payload = _payload("unexpected-zero")

    with pytest.raises(ArtifactVisibilityError, match="failed after mutation"):
        async with open_artifact_session(_work(source), session_id="unexpected-zero") as session:
            route = session.register_trajectory_output(requested)
            session.write_trajectory_document(
                route, TrajectoryDocumentSnapshot("unexpected-zero", requested, payload), "complete",
            )

    assert exchange.calls == 4
    assert "REVERSAL_ATTEMPTED" in events
    assert events.index("REVERSAL_ATTEMPTED") < events.index("REVERSAL_CALLED")
    assert requested.read_bytes() == unexpected
    retained = [path.read_bytes() for path in requested.parent.glob(".daydream-output-*")]
    assert payload in retained
    monkeypatch.setattr(artifact_external, "_name_exchange_factory", _no_reversal())
    with pytest.raises(ArtifactVisibilityError, match="conflict"):
        async with open_artifact_session(_work(source), session_id="unexpected-zero-reopen"):
            pass
    assert requested.read_bytes() == unexpected
    assert payload in [path.read_bytes() for path in requested.parent.glob(".daydream-output-*")]


async def _zero_exchange_with_displaced_directory_records_conflict_without_reversal_case(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested, _ = _external_target(tmp_path / "external")
    exchange = _ReplaceWithDirectoryBeforeSuccessfulExchange(requested)
    monkeypatch.setattr(artifact_external, "_name_exchange_factory", lambda: exchange)
    payload = _payload("nonregular-zero")

    with pytest.raises(ArtifactVisibilityError, match="entry|ambiguous|conflict"):
        async with open_artifact_session(_work(source), session_id="nonregular-zero") as session:
            route = session.register_trajectory_output(requested)
            session.write_trajectory_document(
                route, TrajectoryDocumentSnapshot("nonregular-zero", requested, payload), "complete",
            )

    assert exchange.calls == 3
    assert requested.is_file()
    assert requested.read_bytes() == payload
    staged = list(requested.parent.glob(".daydream-output-*"))
    assert len(staged) == 1
    assert staged[0].is_dir()
    owner = _owner(source)
    transactions = list((owner.artifact_state_root / "transactions").iterdir())
    assert len(transactions) == 1
    entries = cast(list[dict[str, object]], _load_json(transactions[0] / "external-entries.json")["entries"])
    publication = next(entry for entry in entries if entry["purpose"] == "publication_stage")
    assert publication["lifecycle"] == "conflict"
    assert publication["failure_reason"] == "identity_changed"
    conflicts = cast(list[dict[str, object]], _load_json(transactions[0] / "conflicts.json")["conflicts"])
    assert conflicts[-1]["observed_kind"] == "directory"
    monkeypatch.setattr(artifact_external, "_name_exchange_factory", _no_reversal())
    with pytest.raises(ArtifactVisibilityError, match="conflict"):
        async with open_artifact_session(_work(source), session_id="nonregular-reopen"):
            pass
    assert requested.read_bytes() == payload
    assert staged[0].is_dir()


def test_external_fifo_observation_is_nonblocking(tmp_path: Path) -> None:
    """A writer-less FIFO is classified as an issue; a blocking open would never return."""
    parent = tmp_path / "external"
    parent.mkdir()
    fifo = parent / "unexpected.fifo"
    os.mkfifo(fifo)
    observed: list[object] = []

    def observe() -> None:
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            observed.append(artifact_external._external_identity(parent_fd, fifo.name))
        finally:
            os.close(parent_fd)

    # A daemon thread keeps a regression from hanging the worker; any finite window proves non-blocking.
    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    observer.join(timeout=10)
    assert not observer.is_alive(), "external observation blocked on a writer-less FIFO"
    assert [type(observation).__name__ for observation in observed] == ["_ExternalEntryIssue"]
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


async def test_external_probe_process_death_after_primary_exchange_reconciles(
    tmp_path: Path, source: Path, artifact_runtime_root: Path,
) -> None:
    """A SIGKILL after the probe exchange's primary call is reconciled from the ledgered probe pair."""
    first_repo, second_repo = _worktrees(source, "external-ephemeral-one", "external-ephemeral-two")
    marker = tmp_path / "external-probe-primary"
    canary = source.parent / "external-crash-output" / "canary"
    with _child_at_marker(
        marker, "production-external", str(source), str(first_repo), str(artifact_runtime_root), "PRIMARY_CALLED",
        "probe_exchange_a", str(marker),
    ):
        canary.write_bytes(b"unrelated external canary")
    owner = _owner(source)
    async with open_artifact_session(_work(source, repo=second_repo), session_id="recover-probe", owner=owner):
        pass
    assert not any((owner.artifact_state_root / "transactions").glob("*/conflicts.json"))
    assert sorted(path.name for path in canary.parent.iterdir()) == ["canary"]
    assert canary.read_bytes() == b"unrelated external canary"


async def test_external_entry_interruption_reconciles_before_unlock(tmp_path: Path) -> None:
    """An interrupt at each external-entry checkpoint is reconciled before unlock; the target is never torn.

    An unattested probe entry cannot be proven to be ours, so it is retained as a conflict that blocks reopening.
    """
    for index, (checkpoint, purpose, conflict) in enumerate((
        ("CREATION_INTENT", "publication_stage", False), ("OPERATION_PREPARED", "publication_stage", False),
        ("LINK_CREATED", "publication_link_target", False), ("ATTESTED", "publication_link_target", False),
        ("ATTESTED", "missing_parent", False), ("ENTRY_CREATED", "probe_exchange_a", True),
    )):
        source = tmp_path / f"case-{index}" / "source"
        _init_repo(source)
        requested = source.parent / "external" / RUN_DOCUMENT_NAME
        if purpose != "missing_parent":
            _external_target(requested.parent, prior=purpose == "publication_stage")
        primary = KeyboardInterrupt(f"{checkpoint}:{purpose}")

        def interrupt(state: str, observed: str, _path: Path, *, expected: tuple[str, str] = (checkpoint, purpose),
                      primary: BaseException = primary) -> None:
            if (state, observed) == expected:
                raise primary

        with _case_note(f"external checkpoint {checkpoint}:{purpose}"):
            payload = _payload("external-interrupted")
            with pytest.raises(KeyboardInterrupt) as caught:
                original = artifact_external._external_entry_observer
                artifact_external._external_entry_observer = interrupt
                try:
                    async with open_artifact_session(_work(source), session_id="external-interrupted") as session:
                        route = session.register_trajectory_output(requested)
                        session.write_trajectory_document(
                            route, TrajectoryDocumentSnapshot("external-interrupted", requested, payload), "complete",
                        )
                finally:
                    artifact_external._external_entry_observer = original
            assert caught.value is primary
            if requested.exists():
                assert requested.read_bytes() in (b"prior full", payload)
            if conflict:
                with pytest.raises(ArtifactVisibilityError, match="conflict|unattested"):
                    async with open_artifact_session(_work(source), session_id="after-interruption"):
                        pass
                assert any((_owner(source).artifact_state_root / "transactions").iterdir())
            else:
                async with open_artifact_session(_work(source), session_id="after-interruption"):
                    pass


async def test_external_trajectory_write_freeze_and_dispositions_use_immutable_bytes(tmp_path: Path, source: Path,
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested, partial_path = _external_target(tmp_path / "external")
    session_id = "external-freeze"
    partial_bytes = _payload(session_id, session_id, **{"extra": {"partial": True}})
    full_bytes = _payload(session_id, session_id, **{"extra": {"partial": False}})

    with pytest.raises(ArtifactVisibilityError, match="changed|conflict"):
        async with open_artifact_session(_work(source), session_id=session_id) as session:
            route = session.register_trajectory_output(requested)
            assert route.run_dir == session.daydream_dir / "runs" / session_id
            assert (route.full.requested, route.full.write_path) == (requested, requested)
            assert (route.partial.requested, route.partial.write_path) == (partial_path, partial_path)
            assert (route.full.frozen_path, route.partial.frozen_path) == (
                route.run_dir / RUN_DOCUMENT_NAME, route.run_dir / f"{RUN_DOCUMENT_NAME}.partial")
            assert route.full.delivery is artifact_visibility.DestinationDelivery.LIVE_EXTERNAL
            assert route.partial.delivery is route.full.delivery
            registry = _load_json(session._detach_transaction / "destinations.json")
            assert len(cast(list[object], registry["destinations"])) == 2
            with pytest.raises(ArtifactVisibilityError, match="paired trajectory"):
                session.register_destination(requested, label=OutputLabel.EXPLICIT_TRAJECTORY)
            with pytest.raises(ArtifactVisibilityError, match="paired trajectory"):
                session.register_destination(requested, label=OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL)
            partial_document = TrajectoryDocumentSnapshot(session_id, partial_path, partial_bytes)
            session.write_trajectory_document(route, partial_document, "partial")
            assert partial_path.read_bytes() == partial_bytes
            assert route.partial.frozen_path is not None
            assert route.partial.frozen_path.read_bytes() == partial_bytes
            # The complete output's private copy is the root evidence document itself.
            full_document = _reference(route.run_dir / RUN_DOCUMENT_NAME, session_id, full_bytes)
            session.write_trajectory_document(
                route, TrajectoryDocumentSnapshot(session_id, full_document.path, full_bytes), "complete")
            assert requested.read_bytes() == full_bytes
            assert route.full.frozen_path == full_document.path
            # A concurrent reader only ever sees one complete document, never a missing or torn target.
            payloads = tuple(_payload(session_id, session_id, turn=turn) for turn in range(5)) + (full_bytes,)
            observed: list[bytes] = []
            missing: list[bool] = []
            stop = threading.Event()

            def read_until_stopped() -> None:
                while not stop.is_set():
                    try:
                        observed.append(requested.read_bytes())
                    except FileNotFoundError:
                        missing.append(True)

            reader = threading.Thread(target=read_until_stopped)
            reader.start()
            try:
                for payload in payloads:
                    session.write_trajectory_document(
                        route, TrajectoryDocumentSnapshot(session_id, requested, payload), "complete")
            finally:
                stop.set()
                reader.join(timeout=5)
            assert not reader.is_alive() and not missing and observed
            assert set(observed) <= set(payloads)

            # Completed children stay on disk as prefix references; freeze checks them in place.
            children = route.run_dir / SIBLINGS_DIRNAME
            references = {
                name: _reference(children / f"{name}.otlp.jsonl", name,
                                 _payload("other-session" if name == "identity" else session_id, name))
                for name in ("deleted", "digest", "identity", "kept", "late", "truncated", "grown")
            }
            (children / "deleted.otlp.jsonl").unlink()
            (children / "digest.otlp.jsonl").write_bytes((children / "digest.otlp.jsonl").read_bytes().upper())
            (children / "truncated.otlp.jsonl").write_bytes((children / "truncated.otlp.jsonl").read_bytes()[:-1])
            # A document that kept growing after its snapshot still holds the referenced prefix.
            with (children / "grown.otlp.jsonl").open("ab") as grown:
                grown.write(b"later record\n")
            # A child whose bytes are intact but whose identity is wrong is a writer fault, not a drop.
            with pytest.raises(ArtifactVisibilityError, match="snapshot document identity"):
                session.freeze(_snapshot(session_id, (full_document, references.pop("identity"))))
            capsys.readouterr()
            before = {path.name: path.read_bytes() for path in children.iterdir()}
            manifest_tree = artifact_filesystem.manifest_tree

            late_bytes = (children / "late.otlp.jsonl").read_bytes().upper()

            def edit_late_child(root: Path, *args: Any) -> Any:
                if root == session.layout.live_root:  # An edit after the in-place check, before the copy.
                    (children / "late.otlp.jsonl").write_bytes(late_bytes)
                return manifest_tree(root, *args)

            monkeypatch.setattr(artifact_filesystem, "manifest_tree", edit_late_child)
            requested.unlink()
            partial_path.write_bytes(b"mutable path is not evidence")
            snapshot = _snapshot(session_id, (full_document, *references.values()))
            frozen = session.freeze(snapshot)
            monkeypatch.undo()
            frozen_run = frozen.root / ".daydream" / "runs" / session_id
            assert (frozen_run / RUN_DOCUMENT_NAME).read_bytes() == full_bytes
            # Mismatched children are dropped with a warning, left untouched, and the kept one is rebased.
            assert capsys.readouterr().out.count("TrajectoryDocumentIntegrityError") == 4
            after = {path.name: path.read_bytes() for path in children.iterdir()}
            assert after == {**before, "late.otlp.jsonl": late_bytes}
            frozen_children = frozen_run / SIBLINGS_DIRNAME
            kept = references["kept"].frozen_at(frozen_children / "kept.otlp.jsonl")
            grown_kept = references["grown"].frozen_at(frozen_children / "grown.otlp.jsonl")
            assert frozen.trajectories == replace(snapshot, documents=(
                full_document.frozen_at(frozen_run / RUN_DOCUMENT_NAME), kept, grown_kept))
            assert kept.read_bytes() == (children / "kept.otlp.jsonl").read_bytes()
            assert grown_kept.read_bytes() + b"later record\n" == (children / "grown.otlp.jsonl").read_bytes()
            _publish(session, frozen)

    assert partial_path.read_bytes() == b"mutable path is not evidence"
    assert requested.read_bytes() == b"prior full"


async def test_findings_delivery_and_finalization_merge_are_closed_host_boundaries(tmp_path: Path, source: Path,
) -> None:
    findings = tmp_path / "external" / "findings.json"
    findings.parent.mkdir()
    findings.write_bytes(b"prior findings")
    dump = tmp_path / "external" / "dump"
    dump.mkdir()
    (dump / "unrelated.bin").write_bytes(b"unrelated")
    session_id = "late-host"
    payload = _payload(session_id)

    async with open_artifact_session(_work(source), session_id=session_id) as session:
        findings_route = session.register_destination(findings, label=OutputLabel.FINDINGS_OUTPUT)
        dump_route = session.register_destination(dump, label=OutputLabel.DUMP_DIRECTORY)
        assert findings_route.delivery is artifact_visibility.DestinationDelivery.DEFERRED
        assert findings_route.write_path is not None
        assert findings_route.write_path.is_relative_to(session.layout.live_root)
        assert findings.read_bytes() == b"prior findings"
        findings_route.write_path.parent.mkdir(parents=True)
        findings_route.write_path.write_bytes(b"new findings")
        assert dump_route.delivery is artifact_visibility.DestinationDelivery.FINALIZATION_MERGE
        assert dump_route.write_path is None
        assert dump_route.frozen_path is None
        with pytest.raises(ArtifactVisibilityError, match="frozen"):
            session.finalization_merge_path(dump_route, snapshot=cast(Any, object()))

        frozen = _freeze_run(session, session_id, payload=payload)
        before_manifest = frozen.manifest
        late = session.finalization_merge_path(dump_route, snapshot=frozen)
        (late / "nested").mkdir(parents=True)
        (late / "nested" / "bundle.json").write_bytes(b"bundle")
        assert frozen.manifest == before_manifest
        _publish(session, frozen)

    assert findings.read_bytes() == b"new findings"
    assert (dump / "unrelated.bin").read_bytes() == b"unrelated"
    assert (dump / "nested" / "bundle.json").read_bytes() == b"bundle"


async def test_transfer_conflict_never_replaces_a_second_concurrent_occupant(
    source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-source destination replaced while it transfers out keeps both occupants and blocks reopening.

    Every transfer intent is durable before the first move (#1162).
    """
    dump = source / "dump"
    dump.mkdir()
    (dump / "other.bin").write_bytes(b"other dump bytes")
    target = dump / "bundle.json"
    target.write_bytes(b"prior bundle")
    first = b"first concurrent bundle bytes"
    second = b"second concurrent bundle bytes"
    intents: list[dict[str, Any]] = []

    def install_first(path: Path, moved: Path) -> None:
        if not intents:
            stage = next(candidate for candidate in moved.parents if (candidate / "stage-owner.json").is_file())
            intents.append(_load_json(stage / "intents.json"))
        if path == target:
            path.unlink()
            path.write_bytes(first)

    real_os = os

    class SecondOccupantAtRestore:
        injected = False

        def __getattr__(self, name: str) -> Any:
            return getattr(real_os, name)

        def link(self, source_path: Any, destination_path: Any, **kwargs: Any) -> None:
            if Path(destination_path) == Path(target.name) and kwargs.get("dst_dir_fd") is not None:
                target.write_bytes(second)
                self.injected = True
            real_os.link(source_path, destination_path, **kwargs)

    raced_os = SecondOccupantAtRestore()
    monkeypatch.setattr(artifact_transfer, "_transfer_observer", install_first)
    monkeypatch.setattr(artifact_transfer, "os", raced_os)
    with pytest.raises(ArtifactVisibilityError, match="changed|conflict"):
        async with open_artifact_session(_work(source), session_id="double-current-transfer") as session:
            session.register_destination(dump, label=OutputLabel.DUMP_DIRECTORY)

    assert raced_os.injected is True
    assert target.read_bytes() == second
    retained = [path.read_bytes() for path in source.parent.glob(".daydream-transfer-*/entries/dump/bundle.json")]
    assert first in retained
    (recorded,) = intents
    assert sorted(cast(str, intent["relative"]) for intent in recorded["intents"]) == [
        "dump/bundle.json", "dump/other.bin"]
    with pytest.raises(ArtifactVisibilityError, match="conflict"):
        async with open_artifact_session(_work(source), session_id="double-current-reopen"):
            pass
    assert target.read_bytes() == second
    assert first in [path.read_bytes() for path in source.parent.glob(".daydream-transfer-*/entries/dump/bundle.json")]


async def test_current_entry_replacement_is_preserved_during_publication(
    source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = OutputLabel.DUMP_DIRECTORY
    dump = label is OutputLabel.DUMP_DIRECTORY
    session_id = "current-dump" if dump else "current-explicit"
    requested = source / ("dump" if dump else "findings.json")
    prior = b"old bundle" if dump else b"prior findings"
    if dump:
        requested.mkdir()
    target = requested / "bundle.json" if dump else requested
    target.write_bytes(prior)
    replacement = b"unique concurrent publication bytes"
    replacer = _ReplaceAtTransfer(target, replacement)

    with pytest.raises(ArtifactVisibilityError, match="changed|conflict"):
        async with open_artifact_session(_work(source), session_id=session_id) as session:
            route = session.register_destination(requested, label=label)
            if not dump:
                assert route.write_path is not None
                route.write_path.parent.mkdir(parents=True)
                route.write_path.write_bytes(b"generated findings")
            frozen = _freeze_run(session, session_id)
            if dump:
                late = session.finalization_merge_path(route, snapshot=frozen)
                (late / "bundle.json").write_bytes(b"new bundle")
            target.write_bytes(prior)
            monkeypatch.setattr(artifact_transfer, "_transfer_observer", replacer, raising=False)
            _publish(session, frozen)

    assert replacer.observed is True
    assert target.read_bytes() == replacement


# ``<action>`` to child entrypoint, with the positions of its non-Path arguments.
_HELPERS: dict[str, tuple[Callable[..., None], tuple[int, ...]]] = {
    "production-transaction": (_production_transaction_child, (3, 4)),
    "production-external": (_production_external_child, (3, 4)),
}


async def test_artifact_session_projection_rejects_unowned_or_unsafe_paths(tmp_path: Path,) -> None:
    source = tmp_path / "source"
    other = tmp_path / "other"
    _init_repo(source)
    _init_repo(other)
    work = _work(source)

    async with open_artifact_session(work, session_id="prior") as prior:
        prior.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        prior_live = prior.daydream_dir / "deep" / "prior.md"

    async with open_artifact_session(work, session_id="current") as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        findings = session.register_destination(source / "findings.json", label=OutputLabel.FINDINGS_OUTPUT)
        dump = session.register_destination(tmp_path / "dump", label=OutputLabel.DUMP_DIRECTORY)

        with pytest.raises(ArtifactVisibilityError, match="requested repo"):
            session.durable_path_for(session.daydream_dir / "deep" / "current.md", repo=other)
        with pytest.raises(ArtifactVisibilityError, match="registered artifact destination"):
            session.durable_path_for(prior_live, repo=work.repo)
        with pytest.raises(ArtifactVisibilityError, match="registered artifact destination"):
            session.live_path_for(source / "unregistered.json", repo=work.repo)
        with pytest.raises(ArtifactVisibilityError, match="writable path"):
            session.live_path_for(dump.requested, repo=work.repo)

        outside = tmp_path / "outside"
        outside.mkdir()
        session.daydream_dir.mkdir()
        linked = session.daydream_dir / "linked"
        linked.symlink_to(outside, target_is_directory=True)
        try:
            with pytest.raises(ArtifactVisibilityError, match="symlink"):
                session.durable_path_for(linked / "escape.md", repo=work.repo)
        finally:
            linked.unlink()

        assert findings.write_path is not None
        findings.write_path.mkdir(parents=True)
        try:
            with pytest.raises(ArtifactVisibilityError, match="wrong filesystem type"):
                session.durable_path_for(findings.write_path, repo=work.repo)
        finally:
            findings.write_path.rmdir()

    with pytest.raises(ArtifactVisibilityError, match="frozen and no longer writable"):
        session.live_path_for(source / ".daydream", repo=work.repo)


def _main() -> None:
    action, *arguments = sys.argv[1:]
    if action not in _HELPERS:
        raise SystemExit(f"unknown artifact visibility helper action: {action}")
    handler, strings = _HELPERS[action]
    handler(*(value if index in strings else Path(value) for index, value in enumerate(arguments)))


def test_private_workspace_creation_preserves_symlink_target_permissions_during_leaf_replacement(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    locations = PrivateRootLocations(tmp_path / "runtime", tmp_path / "operations")
    original_mkdir = Path.mkdir

    def replace_created_directory(path: Path, *args: Any, **kwargs: Any) -> None:
        original_mkdir(path, *args, **kwargs)
        if path == locations.artifact_runtime:
            path.rmdir()
            path.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(Path, "mkdir", replace_created_directory)
    with pytest.raises(ArtifactVisibilityError, match="private storage root is not an accessible directory"):
        resolve_private_workspace_owner(source, locations=locations)

    assert stat.S_IMODE(outside.stat().st_mode) == 0o755
    assert list(outside.iterdir()) == []


if __name__ == "__main__":
    _main()
