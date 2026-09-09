"""Real-process/filesystem spike for the artifact visibility storage design.

This module deliberately contains a small executable transaction model.  Task 0
uses it to prove the OS and filesystem assumptions before production code adopts
the protocol in Task 1; it is not a substitute implementation of that API.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
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
    bind_artifact_session,
    derive_workspace_identity,
    operational_worktree_root,
    private_root_locations,
    resolve_private_workspace_owner,
    review_output_path_for,
    validate_private_workspace_owner,
)
from daydream.artifact_visibility import (
    _manifest as _pv_manifest,
)
from daydream.artifact_visibility import (
    open_artifact_session as _open_artifact_session,
)
from daydream.trajectory import RunWriteSnapshot, TrajectoryDocumentSnapshot
from daydream.workspace import WorkContext

_TRANSITIONS = (
    "DETACH_STAGED",
    "DETACH_CANONICAL",
    "DETACH_REMOVING",
    "DETACHED",
    "PUBLISH_STAGED",
    "PUBLISH_BACKED_UP",
    "PUBLISH_INSTALLED",
    "PUBLISH_VERIFIED",
)

_CLEANUP_TRANSITIONS = (
    "STAGES_REMOVED",
    "TICKET_FSYNCED",
    "TRANSACTION_RENAMED",
    "JOURNAL_REMOVED",
    "CLEANUP_DIRECTORY_REMOVED",
)

_CUTOFF = "2026-09-06T12:00:00Z"
_COMPLETE = artifact_visibility.ArtifactDisposition.COMPLETE


@dataclass(frozen=True)
class _Entry:
    path: str
    kind: Literal["directory", "file"]
    mode: int
    size: int
    sha256: str | None


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed test-only Git argv
        ["git", *args],  # noqa: S607 - Git is the tested external boundary
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Storage Spike")
    (repo / "source.txt").write_text("source canary\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "initial")


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
async def open_artifact_session(
    work: WorkContext,
    *,
    session_id: str,
    owner: PrivateWorkspaceOwner | None = None,
) -> AsyncIterator[Any]:
    selected = _owner(work.source) if owner is None else owner
    async with _open_artifact_session(work, session_id=session_id, owner=selected) as session:
        yield session


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _read_regular_nofollow(path: Path, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
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
        raise RuntimeError("artifact spike refuses symlink entries")
    if stat.S_ISDIR(info.st_mode):
        entries.append(_Entry(relative, "directory", mode, 0, None))
        children = sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
        for child in children:
            _walk_entry(root, child, entries)
        return
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("artifact spike accepts only directories and regular files")
    content = _read_regular_nofollow(path, info)
    entries.append(_Entry(relative, "file", mode, len(content), hashlib.sha256(content).hexdigest()))


def _manifest(root: Path) -> tuple[_Entry, ...]:
    entries: list[_Entry] = []
    for name in (".daydream", ".review-output.md"):
        _walk_entry(root, root / name, entries)
    return tuple(entries)


def _manifest_payload(entries: tuple[_Entry, ...]) -> dict[str, object]:
    return {"schema_version": 1, "entries": [asdict(entry) for entry in entries]}


def _load_manifest(path: Path) -> tuple[_Entry, ...]:
    payload = _load_json(path)
    assert payload.get("schema_version") == 1
    raw_entries = payload.get("entries")
    assert isinstance(raw_entries, list)
    entries: list[_Entry] = []
    for raw in raw_entries:
        assert isinstance(raw, dict)
        kind = raw.get("kind")
        assert kind in ("directory", "file")
        entries.append(
            _Entry(
                path=cast(str, raw["path"]),
                kind=cast(Literal["directory", "file"], kind),
                mode=cast(int, raw["mode"]),
                size=cast(int, raw["size"]),
                sha256=cast(str | None, raw["sha256"]),
            )
        )
    return tuple(entries)


def _copy_public(source: Path, destination: Path) -> tuple[_Entry, ...]:
    entries = _manifest(source)
    destination.mkdir(parents=True, mode=0o700)
    for entry in entries:
        target = destination / entry.path
        if entry.kind == "directory":
            target.mkdir()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source / entry.path
        content = _read_regular_nofollow(original, original.lstat())
        with target.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, entry.mode)
    for entry in reversed(entries):
        if entry.kind == "directory":
            directory = destination / entry.path
            os.chmod(directory, entry.mode)
            _fsync_dir(directory)
    _fsync_dir(destination)
    assert _manifest(destination) == entries
    return entries


def _remove_manifested_public(source: Path, entries: tuple[_Entry, ...]) -> None:
    if _manifest(source) != entries:
        raise RuntimeError("public artifact bytes changed before removal")
    for entry in entries:
        if entry.kind == "file":
            target = source / entry.path
            target.unlink()
            _fsync_dir(target.parent)
    directories = sorted(
        (entry for entry in entries if entry.kind == "directory"),
        key=lambda entry: entry.path.count("/"),
        reverse=True,
    )
    for entry in directories:
        target = source / entry.path
        target.rmdir()
        _fsync_dir(target.parent)


def _install_stage(source: Path, stage: Path) -> None:
    assert not (source / ".daydream").exists()
    assert not (source / ".review-output.md").exists()
    os.replace(stage / ".daydream", source / ".daydream")
    _fsync_dir(source)
    os.replace(stage / ".review-output.md", source / ".review-output.md")
    _fsync_file(source / ".review-output.md")
    _fsync_dir(source)


def _state_root(runtime: Path, source: Path) -> Path:
    root = runtime / _workspace_key(source)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    owner = {
        "schema_version": 1,
        "source": str(source.resolve(strict=True)),
        "git_common_dir": str(_git_common_dir(source)),
    }
    owner_path = root / "owner.json"
    if owner_path.exists():
        assert _load_json(owner_path) == owner
    else:
        _atomic_json(owner_path, owner)
    return root


def _park(marker: Path, state: str) -> None:
    """Publish the kill-point marker durably, then wait to be SIGKILLed."""
    marker.write_text(state, encoding="ascii")
    _fsync_file(marker)
    _fsync_dir(marker.parent)
    while True:
        signal.pause()


def _transition(journal: Path, state: str, kill_at: str, marker: Path) -> None:
    _atomic_json(journal, {"schema_version": 1, "state": state})
    if state == kill_at:
        _park(marker, state)


def _check_ephemeral_repo(source: Path, repo: Path) -> None:
    assert _git(repo, "rev-parse", "--is-inside-work-tree") == "true"
    assert _git_common_dir(repo) == _git_common_dir(source)


def _transaction_child(source: Path, repo: Path, runtime: Path, kill_at: str, marker: Path) -> None:
    _check_ephemeral_repo(source, repo)
    state = _state_root(runtime, source)
    transaction = state / "transactions" / "spike"
    transaction.mkdir(parents=True)
    journal = transaction / "journal.json"
    stage = transaction / "detach-stage"
    entries = _copy_public(source, stage)
    manifest_path = transaction / "manifest.json"
    _atomic_json(manifest_path, _manifest_payload(entries))
    _transition(journal, "DETACH_STAGED", kill_at, marker)

    canonical = state / "canonical"
    os.replace(stage, canonical)
    _fsync_dir(state)
    _atomic_json(state / "canonical-manifest.json", _manifest_payload(entries))
    _transition(journal, "DETACH_CANONICAL", kill_at, marker)
    _transition(journal, "DETACH_REMOVING", kill_at, marker)
    _remove_manifested_public(source, entries)
    _transition(journal, "DETACHED", kill_at, marker)

    publish_stage = transaction / "publish-stage"
    assert _copy_public(canonical, publish_stage) == entries
    _transition(journal, "PUBLISH_STAGED", kill_at, marker)
    backup = transaction / "public-backup"
    backup.mkdir()
    _fsync_dir(transaction)
    _transition(journal, "PUBLISH_BACKED_UP", kill_at, marker)
    _install_stage(source, publish_stage)
    _transition(journal, "PUBLISH_INSTALLED", kill_at, marker)
    assert _manifest(source) == entries
    _transition(journal, "PUBLISH_VERIFIED", kill_at, marker)


def _recover_child(source: Path, repo: Path, runtime: Path) -> None:
    _check_ephemeral_repo(source, repo)
    state = _state_root(runtime, source)
    transaction = state / "transactions" / "spike"
    journal = transaction / "journal.json"
    payload = _load_json(journal)
    transition = cast(str, payload["state"])
    manifest_path = transaction / "manifest.json"
    entries = _load_manifest(manifest_path)
    canonical = state / "canonical"

    if transition == "DETACH_STAGED":
        stage = transaction / "detach-stage"
        assert _manifest(stage) == entries
        os.replace(stage, canonical)
        _fsync_dir(state)
        _atomic_json(state / "canonical-manifest.json", _manifest_payload(entries))
        transition = "DETACH_CANONICAL"
        _atomic_json(journal, {"schema_version": 1, "state": transition})

    assert _load_manifest(state / "canonical-manifest.json") == entries
    assert _manifest(canonical) == entries
    if transition in ("DETACH_CANONICAL", "DETACH_REMOVING"):
        _remove_manifested_public(source, entries)
        transition = "DETACHED"
        _atomic_json(journal, {"schema_version": 1, "state": transition})

    public_exists = (source / ".daydream").exists() or (source / ".review-output.md").exists()
    if transition in ("PUBLISH_INSTALLED", "PUBLISH_VERIFIED"):
        assert _manifest(source) == entries
    else:
        assert not public_exists
        publish_stage = transaction / "recovery-publish-stage"
        assert _copy_public(canonical, publish_stage) == entries
        _install_stage(source, publish_stage)
    assert _manifest(source) == entries
    _atomic_json(journal, {"schema_version": 1, "state": "PUBLISH_VERIFIED"})


def _lock_child(lock_path: Path, marker: Path) -> None:
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        marker.write_text("locked", encoding="ascii")
        _fsync_file(marker)
        while True:
            signal.pause()
    finally:
        os.close(fd)


def _wait_for_marker(process: subprocess.Popen[str], marker: Path) -> None:
    deadline = time.monotonic() + 10.0
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
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _production_transaction_child(source: Path, repo: Path, runtime: Path, transition: str, marker: Path) -> None:
    locations = private_root_locations(base=runtime.parent)
    owner = resolve_private_workspace_owner(source, locations=locations)

    def stop_at_transition(state: str) -> None:
        if state == transition:
            _park(marker, state)

    def stop_during_cleanup(state: str, transaction_id: str) -> None:
        if not transaction_id.startswith("publish-"):
            return
        stop_at_transition(state)

    artifact_visibility._transition_observer = stop_at_transition
    artifact_visibility._cleanup_observer = stop_during_cleanup

    async def run() -> None:
        session_id = "production-crash"
        async with open_artifact_session(_work(source, repo=repo), session_id=session_id, owner=owner) as session:
            explicit = session.register_destination(source / "explicit-output.txt", label=OutputLabel.FINDINGS_OUTPUT)
            if transition == "EXPLICIT_REGISTERED":
                _park(marker, transition)
            if not (transition.startswith("PUBLISH_") or transition in _CLEANUP_TRANSITIONS):
                raise AssertionError(f"transition was not observed: {transition}")
            explicit.write_path.parent.mkdir(parents=True)
            explicit.write_path.write_bytes(b"published explicit bytes")
            _publish(session, _freeze_run(session, session_id))
            raise AssertionError(f"transition was not observed: {transition}")

    anyio.run(run)


def _production_lock_child(source: Path, runtime: Path, marker: Path) -> None:
    locations = private_root_locations(base=runtime.parent)
    owner = resolve_private_workspace_owner(source, locations=locations)

    async def run() -> None:
        async with open_artifact_session(_work(source), session_id="lock-holder", owner=owner):
            _park(marker, "locked")

    anyio.run(run)


def _production_external_child(
    source: Path,
    repo: Path,
    runtime: Path,
    checkpoint: str,
    purpose: str,
    marker: Path,
) -> None:
    locations = private_root_locations(base=runtime.parent)
    owner = resolve_private_workspace_owner(source, locations=locations)

    observed_checkpoint = checkpoint.removeprefix("UNEXPECTED_")

    def stop_at_external(state: str, observed_purpose: str, _path: Path) -> None:
        if state == observed_checkpoint and observed_purpose == purpose:
            _park(marker, f"{state}:{observed_purpose}")

    artifact_visibility._external_entry_observer = stop_at_external

    async def run() -> None:
        session_id = "external-crash"
        requested = source.parent / "external-crash-output" / "trajectory.json"
        if purpose != "missing_parent":
            _external_target(requested.parent, prior=purpose == "publication_stage")
        async with open_artifact_session(_work(source, repo=repo), session_id=session_id, owner=owner) as session:
            route = session.register_trajectory_output(requested)
            if checkpoint.startswith("UNEXPECTED_"):
                real = artifact_visibility._AtomicNameExchange()

                class ReplaceBeforeSuccessfulExchange:
                    def __init__(self) -> None:
                        self.calls = 0

                    def call(self, parent_fd: int, staged_name: str, target_name: str) -> Any:
                        self.calls += 1
                        if self.calls == 1:
                            requested.write_bytes(b"unexpected displaced external bytes")
                        return real.call(parent_fd, staged_name, target_name)

                session._name_exchange = ReplaceBeforeSuccessfulExchange()
            elif checkpoint in ("REVERSAL_ATTEMPTED", "REVERSAL_CALLED"):
                real = artifact_visibility._AtomicNameExchange()

                class FailAfterMutation:
                    def call(self, parent_fd: int, staged_name: str, target_name: str) -> Any:
                        result = real.call(parent_fd, staged_name, target_name)
                        assert result.result == 0
                        return artifact_visibility._NameExchangeResult(-1, 5)

                session._name_exchange = FailAfterMutation()
            session.write_trajectory_document(
                route,
                TrajectoryDocumentSnapshot(session_id, requested, _payload(session_id)),
                "complete",
            )
            raise AssertionError(f"external checkpoint was not observed: {checkpoint}:{purpose}")

    anyio.run(run)


def _external_fifo_observation_child(fifo: Path, entered: Path, completed: Path) -> None:
    parent_fd = os.open(fifo.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        entered.write_text("entered", encoding="ascii")
        _fsync_file(entered)
        _fsync_dir(entered.parent)
        observation = artifact_visibility._external_identity(parent_fd, fifo.name)
        completed.write_text(type(observation).__name__, encoding="ascii")
        _fsync_file(completed)
        _fsync_dir(completed.parent)
    finally:
        os.close(parent_fd)


def _seed_public_artifacts(source: Path) -> tuple[_Entry, ...]:
    deep = source / ".daydream" / "deep"
    runs = source / ".daydream" / "runs"
    empty = source / ".daydream" / "empty-directory"
    deep.mkdir(parents=True)
    runs.mkdir()
    empty.mkdir()
    (deep / "prior.md").write_bytes(b"prior reasoning\n")
    (runs / "opaque.bin").write_bytes(b"\x00\xff\xfe\x80payload\n")
    (source / ".review-output.md").write_bytes(b"review output\n")
    os.chmod(source / ".daydream", 0o750)
    os.chmod(deep, 0o710)
    os.chmod(runs, 0o700)
    os.chmod(empty, 0o711)
    os.chmod(deep / "prior.md", 0o640)
    os.chmod(runs / "opaque.bin", 0o600)
    os.chmod(source / ".review-output.md", 0o644)
    return _manifest(source)


def _manifest_identity(entries: Any) -> tuple[tuple[str, str, int, int, Any], ...]:
    """Comparable identity of manifest entries regardless of dataclass type."""
    return tuple(
        (entry.path, entry.kind, entry.size, entry.mode, entry.sha256) for entry in entries
    )


def _projection_matches(root: Path, entries: tuple[_Entry, ...]) -> bool:
    try:
        return _manifest(root) == entries
    except FileNotFoundError:
        return False


def _work(source: Path, *, repo: Path | None = None, run_id: str = "work") -> WorkContext:
    return WorkContext(
        repo=source if repo is None else repo,
        source=source,
        base_branch="main",
        base_sha=_git(source, "rev-parse", "HEAD"),
        head_branch="main" if repo is None else None,
        head_sha=_git(source, "rev-parse", "HEAD"),
        is_ephemeral=repo is not None,
        run_id=run_id,
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
    requested = parent / "trajectory.json"
    partial = requested.with_suffix(requested.suffix + ".partial")
    if prior:
        requested.write_bytes(b"prior full")
        partial.write_bytes(b"prior partial")
    return requested, partial


def _payload(session_id: str, trajectory_id: str | None = None, **extra: Any) -> bytes:
    return json.dumps(
        {
            "session_id": session_id,
            "trajectory_id": session_id if trajectory_id is None else trajectory_id,
            **extra,
        },
        sort_keys=True,
    ).encode()


def _snapshot(
    session_id: str,
    documents: tuple[TrajectoryDocumentSnapshot, ...],
    *,
    status: str = "complete",
    root_trajectory_id: str | None = None,
) -> RunWriteSnapshot:
    return RunWriteSnapshot(
        status=cast(Any, status),
        cutoff_at=_CUTOFF,
        root_trajectory_id=session_id if root_trajectory_id is None else root_trajectory_id,
        documents=documents,
    )


def _freeze_run(session: Any, session_id: str, *, payload: bytes | None = None) -> Any:
    """Freeze one root trajectory document written into the session's own run dir."""
    document = TrajectoryDocumentSnapshot(
        session_id,
        session.daydream_dir / "runs" / session_id / "trajectory.json",
        _payload(session_id) if payload is None else payload,
    )
    return session.freeze(_snapshot(session_id, (document,)))


def _publish(session: Any, frozen: Any, disposition: Any = None) -> None:
    session.finalize_frozen(frozen, disposition=_COMPLETE if disposition is None else disposition)


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


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A real one-commit Git repository at ``tmp_path/source``."""
    repo = tmp_path / "source"
    _init_repo(repo)
    return repo


def test_lock_process_rejects_second_process_and_releases_on_death(tmp_path: Path) -> None:
    lock_path = tmp_path / "workspace.lock"
    marker = tmp_path / "lock-ready"
    with _child_at_marker(marker, "lock", str(lock_path), str(marker)):
        contender = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)

    released = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(released, fcntl.LOCK_UN)
    finally:
        os.close(released)


def test_recovery_spike_rejects_symlink_without_following(tmp_path: Path, source: Path) -> None:
    _seed_public_artifacts(source)
    outside = tmp_path / "outside-secret"
    outside.write_bytes(b"must remain outside")
    (source / ".daydream" / "deep" / "outward-link").symlink_to(outside)

    with pytest.raises(RuntimeError, match="refuses symlink"):
        _copy_public(source, tmp_path / "stage")

    assert outside.read_bytes() == b"must remain outside"
    assert not (tmp_path / "stage" / ".daydream" / "deep" / "outward-link").exists()


@pytest.mark.parametrize("transition", _TRANSITIONS)
def test_recovery_spike_survives_real_process_death_at_each_journal_transition(
    tmp_path: Path,
    source: Path,
    transition: str,
) -> None:
    expected = _seed_public_artifacts(source)
    runtime = tmp_path / "operator-runtime"
    first_repo, second_repo = _worktrees(source, "ephemeral-one", "ephemeral-two")
    common_dir = _git_common_dir(source)
    assert _git_common_dir(first_repo) == common_dir
    assert _git_common_dir(second_repo) == common_dir
    marker = tmp_path / f"reached-{transition}"

    with _child_at_marker(marker, "transaction", str(source), str(first_repo), str(runtime), transition, str(marker)):
        pass

    state = runtime / _workspace_key(source)
    assert _projection_matches(source, expected) or _projection_matches(state / "canonical", expected)
    public_expected = transition in {
        "DETACH_STAGED",
        "DETACH_CANONICAL",
        "DETACH_REMOVING",
        "PUBLISH_INSTALLED",
        "PUBLISH_VERIFIED",
    }
    assert _projection_matches(source, expected) is public_expected

    recovered = subprocess.run(  # noqa: S603 - fixed test helper and paths
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "recover",
            str(source),
            str(second_repo),
            str(runtime),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert _manifest(source) == expected
    assert _manifest(state / "canonical") == expected
    assert (source / ".daydream" / "runs" / "opaque.bin").read_bytes() == b"\x00\xff\xfe\x80payload\n"
    assert (source / ".daydream" / "empty-directory").is_dir()
    assert not any((source / ".daydream" / "empty-directory").iterdir())
    assert stat.S_IMODE((source / ".daydream" / "deep" / "prior.md").stat().st_mode) == 0o640
    assert stat.S_IMODE((source / ".daydream" / "runs" / "opaque.bin").stat().st_mode) == 0o600
    assert stat.S_IMODE((source / ".daydream" / "empty-directory").stat().st_mode) == 0o711
    assert stat.S_IMODE((source / ".review-output.md").stat().st_mode) == 0o644
    assert _load_json(state / "owner.json") == {
        "schema_version": 1,
        "source": str(source.resolve(strict=True)),
        "git_common_dir": str(_git_common_dir(source)),
    }


def test_private_root_locations_explicit_base_bypasses_default_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "explicit-private"

    def fail_default_lookup() -> Path:
        raise AssertionError("explicit private base consulted the default provider")

    monkeypatch.setattr(artifact_visibility, "_default_private_base", fail_default_lookup)
    environment = dict(os.environ)

    locations = private_root_locations(base=base)

    assert locations == PrivateRootLocations(
        artifact_runtime=base / "runtime",
        operational_workspaces=base / "workspaces",
    )
    assert dict(os.environ) == environment
    assert not locations.artifact_runtime.exists()
    assert not locations.operational_workspaces.exists()


@pytest.mark.parametrize("kind", ["relative", "equal", "nested", "symlink"])
def test_resolve_private_workspace_owner_rejects_unsafe_direct_root_pairs(
    tmp_path: Path,
    source: Path,
    kind: str,
) -> None:
    if kind == "relative":
        locations = PrivateRootLocations(
            artifact_runtime=Path("relative-runtime"),
            operational_workspaces=tmp_path / "operations",
        )
    elif kind == "equal":
        locations = PrivateRootLocations(
            artifact_runtime=tmp_path / "private",
            operational_workspaces=tmp_path / "private",
        )
    elif kind == "nested":
        locations = PrivateRootLocations(
            artifact_runtime=tmp_path / "private",
            operational_workspaces=tmp_path / "private" / "operations",
        )
    else:
        actual = tmp_path / "actual"
        actual.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        locations = PrivateRootLocations(artifact_runtime=alias, operational_workspaces=tmp_path / "operations")

    with pytest.raises(ArtifactVisibilityError, match="absolute lexical|overlap|symlink"):
        resolve_private_workspace_owner(source, locations=locations)


def test_private_workspace_owner_creates_byte_identical_peers_and_operational_root(
    tmp_path: Path,
    source: Path,
) -> None:
    locations = private_root_locations(base=tmp_path / "private")
    before_worktrees = _git(source, "worktree", "list", "--porcelain")

    owner = resolve_private_workspace_owner(source, locations=locations)
    operational = operational_worktree_root(owner)

    assert owner.artifact_state_root == locations.artifact_runtime / owner.workspace_key
    assert owner.operational_state_root == locations.operational_workspaces / owner.workspace_key
    assert operational == owner.operational_state_root / "operational"
    assert _git(source, "worktree", "list", "--porcelain") == before_worktrees
    for directory in (
        locations.artifact_runtime,
        locations.operational_workspaces,
        owner.artifact_state_root,
        owner.operational_state_root,
        operational,
    ):
        assert stat.S_IMODE(directory.lstat().st_mode) == 0o700
    artifact_owner = owner.artifact_state_root / "owner.json"
    operational_owner = owner.operational_state_root / "owner.json"
    assert artifact_owner.read_bytes() == operational_owner.read_bytes()
    assert stat.S_IMODE(artifact_owner.lstat().st_mode) == 0o600
    assert stat.S_IMODE(operational_owner.lstat().st_mode) == 0o600
    assert _load_json(artifact_owner) == {
        "git_common_dir": str(_git_common_dir(source)),
        "schema_version": 1,
        "source": str(source.resolve(strict=True)),
        "workspace_key": _workspace_key(source),
    }
    validate_private_workspace_owner(owner, source=source, repo=source)


def test_private_workspace_owner_conflicting_peer_prevents_missing_peer_creation(
    tmp_path: Path,
    source: Path,
) -> None:
    locations = private_root_locations(base=tmp_path / "private")
    key = _workspace_key(source)
    conflicting_state = locations.operational_workspaces / key
    conflicting_state.mkdir(parents=True, mode=0o700)
    os.chmod(locations.operational_workspaces, 0o700)
    os.chmod(conflicting_state, 0o700)
    _atomic_json(
        conflicting_state / "owner.json",
        {
            "schema_version": 1,
            "workspace_key": key,
            "source": "different-source",
            "git_common_dir": str(_git_common_dir(source)),
        },
    )

    with pytest.raises(ArtifactVisibilityError, match="owner metadata"):
        resolve_private_workspace_owner(source, locations=locations)

    assert not (locations.artifact_runtime / key).exists()


async def test_artifact_session_revalidates_supplied_owner_before_mutation(tmp_path: Path) -> None:
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


def test_validate_private_workspace_owner_rejects_direct_dataclass_path_tampering(
    tmp_path: Path,
    source: Path,
) -> None:
    owner = _owner(source, locations=private_root_locations(base=tmp_path / "private"))
    tampered = PrivateWorkspaceOwner(
        source=owner.source,
        git_common_dir=owner.git_common_dir,
        workspace_key=owner.workspace_key,
        artifact_state_root=owner.artifact_state_root.parent / "different-key",
        operational_state_root=owner.operational_state_root,
    )

    with pytest.raises(ArtifactVisibilityError, match="path mismatch|accessible|real directory"):
        validate_private_workspace_owner(tampered, source=source)


def test_derive_workspace_identity_rejects_repo_with_different_git_common_dir(tmp_path: Path, source: Path) -> None:
    unrelated = tmp_path / "unrelated"
    _init_repo(unrelated)
    owner = _owner(source)

    with pytest.raises(ArtifactVisibilityError, match="share the source Git identity"):
        derive_workspace_identity(_work(source, repo=unrelated), owner=owner)


async def test_artifact_session_detaches_routes_and_restores_public_bytes(
    source: Path,
    artifact_runtime_root: Path,
) -> None:
    expected = _seed_public_artifacts(source)
    work = _work(source)

    assert artifact_dir_for(work.repo) == source / ".daydream"
    assert review_output_path_for(work.repo) == source / ".review-output.md"
    async with open_artifact_session(work, session_id="session-one") as session:
        assert session.layout.state_root.parent == artifact_runtime_root
        assert not (source / ".daydream").exists()
        assert not (source / ".review-output.md").exists()
        assert _manifest(session.layout.live_root) == expected
        assert artifact_dir_for(work.repo) == session.daydream_dir
        assert review_output_path_for(work.repo) == session.review_output
        assert artifact_dir_for(work.repo).is_relative_to(session.layout.state_root)

    assert _manifest(source) == expected
    assert artifact_dir_for(work.repo) == source / ".daydream"


async def test_artifact_session_preserves_resume_mtimes_across_publication_and_reopen(source: Path) -> None:
    from daydream.deep.artifacts import check_deep_artifacts

    deep = source / ".daydream" / "deep"
    deep.mkdir(parents=True)
    intent = deep / "intent.md"
    alternatives = deep / "alternatives.json"
    key = deep / "diff-key"
    intent.write_text("intent\n", encoding="utf-8")
    alternatives.write_text("[]\n", encoding="utf-8")
    current_diff_sha = "a" * 64
    key.write_text(current_diff_sha, encoding="ascii")

    key_mtime_ns = 1_700_000_000_000_000_000
    prerequisite_mtime_ns = key_mtime_ns + 10_000_000_000
    os.utime(key, ns=(key_mtime_ns, key_mtime_ns))
    for prerequisite in (intent, alternatives):
        os.utime(prerequisite, ns=(prerequisite_mtime_ns, prerequisite_mtime_ns))

    session_id = "mtime-first"
    canonical: Path
    async with open_artifact_session(_work(source), session_id=session_id) as session:
        live_deep = session.daydream_dir / "deep"
        assert (live_deep / "diff-key").stat().st_mtime_ns == key_mtime_ns
        assert (live_deep / "alternatives.json").stat().st_mtime_ns == prerequisite_mtime_ns
        check_deep_artifacts("per-stack", live_deep, current_diff_sha=current_diff_sha)

        canonical = session.layout.state_root / "canonical"
        _publish(session, _freeze_run(session, session_id))

    for copied_deep in (source / ".daydream" / "deep", canonical / ".daydream" / "deep"):
        assert (copied_deep / "diff-key").stat().st_mtime_ns == key_mtime_ns
        assert (copied_deep / "alternatives.json").stat().st_mtime_ns == prerequisite_mtime_ns

    public_deep = source / ".daydream" / "deep"
    contradictory_mtime_ns = key_mtime_ns - 10_000_000_000
    os.utime(public_deep / "alternatives.json", ns=(contradictory_mtime_ns, contradictory_mtime_ns))

    async with open_artifact_session(_work(source), session_id="mtime-second") as session:
        live_deep = session.daydream_dir / "deep"
        assert (live_deep / "diff-key").stat().st_mtime_ns == key_mtime_ns
        assert (live_deep / "alternatives.json").stat().st_mtime_ns == prerequisite_mtime_ns
        check_deep_artifacts("per-stack", live_deep, current_diff_sha=current_diff_sha)


async def test_artifact_session_uses_stable_source_key_across_ephemeral_repos(tmp_path: Path, source: Path) -> None:
    expected = _seed_public_artifacts(source)
    first, second = _worktrees(source, "ephemeral-one", "ephemeral-two")

    async with open_artifact_session(_work(source, repo=first), session_id="first") as session:
        key = session.layout.workspace_key
        state_root = session.layout.state_root
    async with open_artifact_session(_work(source, repo=second), session_id="second") as session:
        assert session.layout.workspace_key == key
        assert session.layout.state_root == state_root
        assert _manifest(session.layout.live_root) == expected

    assert _manifest(source) == expected


async def test_bound_routing_propagates_to_tasks_and_rejects_wrong_or_aliased_repo(
    tmp_path: Path,
    source: Path,
) -> None:
    work = _work(source)
    observed: list[Path] = []
    alias = tmp_path / "repo-alias"
    alias.symlink_to(source, target_is_directory=True)

    assert artifact_session_active() is False
    async with open_artifact_session(work, session_id="routing") as session:
        assert artifact_session_active() is True
        async def child() -> None:
            observed.append(artifact_dir_for(work.repo))

        async with anyio.create_task_group() as group:
            group.start_soon(child)
        assert observed == [session.daydream_dir]
        with bind_artifact_session(session):
            assert artifact_dir_for(work.repo) == session.daydream_dir
        assert artifact_dir_for(work.repo) == session.daydream_dir
        with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
            artifact_dir_for(tmp_path / "wrong-repo")
        with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
            artifact_dir_for(alias)
        assert not (source / ".daydream").exists()
        assert not (source / ".review-output.md").exists()

    assert artifact_session_active() is False
    assert artifact_dir_for(work.repo) == source / ".daydream"


@pytest.mark.parametrize("exit_kind", ["error", "cancel"])
async def test_artifact_session_resets_binding_after_error_or_cancellation(source: Path, exit_kind: str) -> None:
    work = _work(source)

    if exit_kind == "error":
        with pytest.raises(RuntimeError, match="primary"):
            async with open_artifact_session(work, session_id="error"):
                raise RuntimeError("primary")
    else:
        with anyio.CancelScope() as scope:
            async with open_artifact_session(work, session_id="cancel"):
                scope.cancel()
                await anyio.sleep(0)

    assert artifact_dir_for(work.repo) == source / ".daydream"


async def test_artifact_session_open_and_recovery_run_off_async_owner_thread(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocking Git/copy/fsync lifecycle work is awaited outside the event loop."""
    owner_thread = threading.get_ident()
    open_threads: list[int] = []
    restore_threads: list[int] = []
    original_open = artifact_visibility._open_layout
    original_restore = artifact_visibility.ArtifactSession._restore_prior

    def observed_open(*args: Any, **kwargs: Any) -> Any:
        open_threads.append(threading.get_ident())
        return original_open(*args, **kwargs)

    def observed_restore(session: Any) -> None:
        restore_threads.append(threading.get_ident())
        original_restore(session)

    monkeypatch.setattr(artifact_visibility, "_open_layout", observed_open)
    monkeypatch.setattr(artifact_visibility.ArtifactSession, "_restore_prior", observed_restore)

    async with open_artifact_session(_work(source), session_id="threaded-lifecycle"):
        assert artifact_dir_for(source) != source / ".daydream"

    assert open_threads and all(thread != owner_thread for thread in open_threads)
    assert restore_threads and all(thread != owner_thread for thread in restore_threads)
    assert artifact_dir_for(source) == source / ".daydream"


@pytest.mark.parametrize("tracked", [".daydream/tracked.txt", ".review-output.md"])
async def test_artifact_session_rejects_tracked_public_collision_untouched(source: Path, tracked: str) -> None:
    target = source / tracked
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"tracked collision")
    _git(source, "add", tracked)
    _git(source, "commit", "-m", "tracked collision")

    with pytest.raises(ArtifactVisibilityError, match="tracked artifact collision"):
        async with open_artifact_session(_work(source), session_id="tracked"):
            pass

    assert target.read_bytes() == b"tracked collision"


async def test_artifact_session_rejects_unknown_legacy_tree_but_preserves_extensions(source: Path) -> None:
    unknown = source / ".daydream" / "extension-only" / "opaque.bin"
    unknown.parent.mkdir(parents=True)
    unknown.write_bytes(b"extension")
    with pytest.raises(ArtifactVisibilityError, match="recognized artifact anchor"):
        async with open_artifact_session(_work(source), session_id="unknown"):
            pass
    assert unknown.read_bytes() == b"extension"

    (source / ".daydream" / "runs").mkdir()
    async with open_artifact_session(_work(source), session_id="anchored") as session:
        assert (session.daydream_dir / "extension-only" / "opaque.bin").read_bytes() == b"extension"
    assert unknown.read_bytes() == b"extension"


async def test_artifact_session_rebaselines_public_deletion_at_open(
    tmp_path: Path,
) -> None:
    """F2 brick: a benign between-run public deletion is adopted, not fatal.

    A first run publishes artifacts (canonical seeded). The operator then
    deletes ``.review-output.md`` between runs. The next session open must
    adopt the observed public state into the canonical recovery copy (with a
    warning) instead of failing with "public artifacts do not match canonical
    recovery state".
    """
    source = tmp_path / "source"
    _init_repo(source)
    seeded = _seed_public_artifacts(source)
    async with open_artifact_session(_work(source), session_id="baseline"):
        # No publish: canonical is seeded from the observed public tree and
        # the close restores it. (A mid-session write into the public tree
        # would be a mid-run mutation, which stays fail-closed by design.)
        pass
    assert seeded
    # First session restored the seeded public tree on close.
    assert (source / ".review-output.md").read_bytes() == b"review output\n"

    # Benign between-run operator cleanup: delete the public review output.
    (source / ".review-output.md").unlink()
    assert (source / ".review-output.md").exists() is False

    async with open_artifact_session(_work(source), session_id="rebaseline") as session:
        state_root = session.layout.state_root
        canonical = state_root / "canonical"
        entries = _load_manifest(state_root / "canonical-manifest.json")
        assert _manifest_identity(_pv_manifest(canonical)) == _manifest_identity(entries)
        assert all(entry.path != ".review-output.md" for entry in entries)
        assert (canonical / ".daydream" / "deep" / "prior.md").read_bytes() == (
            b"prior reasoning\n"
        )
        # The live private tree still carries the adopted baseline.
        assert (session.daydream_dir / "deep" / "prior.md").read_bytes() == (
            b"prior reasoning\n"
        )
    # On close the adopted baseline is restored to the public tree.
    restored = _load_manifest(state_root / "canonical-manifest.json")
    assert _manifest_identity(_pv_manifest(source, (".daydream", ".review-output.md"))) == (
        _manifest_identity(restored)
    )
    assert (source / ".review-output.md").exists() is False
    assert seeded[0].path  # seeded manifest shape unchanged, sanity


async def test_artifact_session_rebaselines_git_clean_between_runs(
    tmp_path: Path,
) -> None:
    """``git clean -fdx`` style full public wipe is adopted as the new baseline."""
    source = tmp_path / "source"
    _init_repo(source)
    _seed_public_artifacts(source)
    async with open_artifact_session(_work(source), session_id="baseline") as session:
        state_root = session.layout.state_root

    # Simulate git clean -fdx: the whole public artifact tree disappears.
    shutil.rmtree(source / ".daydream")
    (source / ".review-output.md").unlink()

    async with open_artifact_session(_work(source), session_id="after-clean") as session:
        entries = _load_manifest(state_root / "canonical-manifest.json")
        assert entries == ()
        assert _pv_manifest(state_root / "canonical") == ()
        assert not (source / ".daydream").exists()
        assert session.layout.daydream_dir == session.layout.live_root / ".daydream"


async def test_artifact_session_rebaselines_stray_public_addition(
    tmp_path: Path,
) -> None:
    """A stray external write into the public tree is adopted at session open."""
    source = tmp_path / "source"
    _init_repo(source)
    _seed_public_artifacts(source)
    async with open_artifact_session(_work(source), session_id="baseline") as session:
        state_root = session.layout.state_root

    stray = source / ".daydream" / "operator-notes.txt"
    stray.write_bytes(b"stray external bytes\n")

    async with open_artifact_session(_work(source), session_id="after-stray") as session:
        entries = _load_manifest(state_root / "canonical-manifest.json")
        assert _manifest_identity(_pv_manifest(state_root / "canonical")) == _manifest_identity(entries)
        stray_entry = next(entry for entry in entries if entry.path.endswith("operator-notes.txt"))
        assert stray_entry.kind == "file"
        assert (session.daydream_dir / "operator-notes.txt").read_bytes() == (
            b"stray external bytes\n"
        )
    # On close the adopted baseline (with the stray file) is restored publicly.
    restored = _load_manifest(state_root / "canonical-manifest.json")
    assert _manifest_identity(_pv_manifest(source, (".daydream", ".review-output.md"))) == (
        _manifest_identity(restored)
    )
    assert (source / ".daydream" / "operator-notes.txt").read_bytes() == b"stray external bytes\n"


async def test_artifact_session_open_still_fails_closed_on_nonregular_public_node(
    tmp_path: Path,
) -> None:
    """Re-baselining adopts benign drift only; unsafe nodes still fail closed."""
    source = tmp_path / "source"
    _init_repo(source)
    _seed_public_artifacts(source)
    async with open_artifact_session(_work(source), session_id="baseline"):
        pass
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary").write_bytes(b"outside")
    shutil.rmtree(source / ".daydream" / "runs")
    (source / ".daydream" / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactVisibilityError, match="regular files and directories"):
        async with open_artifact_session(_work(source), session_id="unsafe"):
            pass
    assert (outside / "canary").read_bytes() == b"outside"


async def test_artifact_session_succeeds_with_only_empty_operational_root(
    tmp_path: Path,
) -> None:
    """F1 cross-check: an EMPTY operational root is not an anchor blocker.

    A wedged checkout can hold ``.daydream/worktrees`` (empty residue) with no
    other anchor; session open must succeed rather than raising "legacy
    .daydream tree has no recognized artifact anchor".
    """
    source = tmp_path / "source"
    _init_repo(source)
    (source / ".daydream" / "worktrees").mkdir(parents=True)

    async with open_artifact_session(_work(source), session_id="empty-root") as session:
        assert (session.daydream_dir / "worktrees").is_dir()

    # On close the empty root is restored alongside the rest of the baseline.
    assert (source / ".daydream" / "worktrees").is_dir()


async def test_artifact_session_still_fails_closed_on_mid_run_public_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-baseline is session-open-only; mid-run divergence still fails closed.

    A concurrent write into the public tree during the detach window must
    keep raising through the existing conflict machinery (previously pinned
    by the strict-equality path; now pinned by the same observer seam).
    """
    from daydream import artifact_visibility as av

    source = tmp_path / "source"
    _init_repo(source)
    _seed_public_artifacts(source)

    def mutate_before_removal(state: str) -> None:
        if state == "DETACH_REMOVING":
            (source / ".daydream" / "concurrent.bin").write_bytes(b"unique concurrent bytes")

    monkeypatch.setattr(av, "_transition_observer", mutate_before_removal)
    with pytest.raises(ArtifactVisibilityError, match="changed during detach"):
        async with open_artifact_session(_work(source), session_id="mid-run-mutation"):
            pass
    assert (source / ".daydream" / "concurrent.bin").read_bytes() == b"unique concurrent bytes"
    assert (source / ".daydream" / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"


@pytest.mark.parametrize(
    "node_kind",
    ["root-symlink", "directory-symlink", "leaf-symlink", "fifo", "socket"],
)
async def test_artifact_session_rejects_nonregular_public_nodes_without_following(
    tmp_path: Path,
    source: Path,
    node_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary").write_bytes(b"outside")
    listener: socket.socket | None = None
    if node_kind == "root-symlink":
        (source / ".daydream").symlink_to(outside, target_is_directory=True)
    elif node_kind == "directory-symlink":
        (source / ".daydream").mkdir()
        (source / ".daydream" / "runs").symlink_to(outside, target_is_directory=True)
    else:
        (source / ".daydream" / "runs").mkdir(parents=True)
        leaf = source / ".daydream" / "runs" / "node"
        if node_kind == "leaf-symlink":
            leaf.symlink_to(outside / "canary")
        elif node_kind == "fifo":
            os.mkfifo(leaf)
        else:
            monkeypatch.chdir(source)
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(".daydream/runs/node")

    try:
        with pytest.raises(ArtifactVisibilityError, match="regular files and directories"):
            async with open_artifact_session(_work(source), session_id=node_kind):
                pass
        assert (outside / "canary").read_bytes() == b"outside"
    finally:
        if node_kind == "socket":
            assert listener is not None
            listener.close()


@pytest.mark.parametrize("peer", ["artifact", "operational"])
async def test_artifact_session_rejects_owner_mismatch_and_runtime_overlap(
    tmp_path: Path,
    source: Path,
    peer: str,
) -> None:
    work = _work(source)
    locations = private_root_locations(base=tmp_path / "private-explicit")
    owner_identity = _owner(source, locations=locations)
    async with open_artifact_session(work, session_id="owner", owner=owner_identity) as session:
        state_root = session.layout.state_root
    owner_path = (
        owner_identity.artifact_state_root
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
        artifact_runtime=source / "nested-runtime",
        operational_workspaces=tmp_path / "operations",
    )
    with pytest.raises(ArtifactVisibilityError, match="overlap"):
        resolve_private_workspace_owner(source, locations=overlapping)


def test_private_workspace_owner_completes_one_missing_peer_after_interrupted_creation(
    tmp_path: Path,
    source: Path,
) -> None:
    locations = private_root_locations(base=tmp_path / "private")
    key = _workspace_key(source)
    artifact_state = locations.artifact_runtime / key
    artifact_state.mkdir(parents=True, mode=0o700)
    os.chmod(locations.artifact_runtime, 0o700)
    os.chmod(artifact_state, 0o700)
    expected = {
        "schema_version": 1,
        "workspace_key": key,
        "source": str(source.resolve(strict=True)),
        "git_common_dir": str(_git_common_dir(source)),
    }
    _atomic_json(artifact_state / "owner.json", expected)

    owner = resolve_private_workspace_owner(source, locations=locations)

    assert owner.artifact_state_root == artifact_state
    assert (owner.operational_state_root / "owner.json").read_bytes() == (artifact_state / "owner.json").read_bytes()


def test_derive_workspace_identity_rejects_model_repo_inside_artifact_runtime(tmp_path: Path, source: Path) -> None:
    locations = private_root_locations(base=tmp_path / "private")
    owner = resolve_private_workspace_owner(source, locations=locations)
    model_repo = locations.artifact_runtime / "model-worktree"
    _git(source, "worktree", "add", "--detach", str(model_repo), "HEAD")

    with pytest.raises(ArtifactVisibilityError, match="runtime and repository overlap"):
        derive_workspace_identity(_work(source, repo=model_repo), owner=owner)


def test_private_workspace_owner_rejects_root_overlapping_external_git_common_dir(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    source = tmp_path / "linked-source"
    _init_repo(primary)
    _git(primary, "worktree", "add", "--detach", str(source), "HEAD")
    common_dir = _git_common_dir(source)
    locations = PrivateRootLocations(
        artifact_runtime=common_dir / "private-runtime",
        operational_workspaces=tmp_path / "operations",
    )

    with pytest.raises(ArtifactVisibilityError, match="overlaps source Git ownership"):
        resolve_private_workspace_owner(source, locations=locations)

    assert not locations.artifact_runtime.exists()


@pytest.mark.parametrize("target", ["artifact-root", "operational-owner"])
def test_validate_private_workspace_owner_rejects_unsafe_private_modes(source: Path, target: str) -> None:
    owner = _owner(source)
    changed = (
        owner.artifact_state_root
        if target == "artifact-root"
        else owner.operational_state_root / "owner.json"
    )
    changed.chmod(0o755 if target == "artifact-root" else 0o644)

    with pytest.raises(ArtifactVisibilityError, match="0700|0600"):
        validate_private_workspace_owner(owner, source=source)


async def test_artifact_session_registers_destinations_and_rejects_aliases(tmp_path: Path, source: Path) -> None:
    work = _work(source)
    async with open_artifact_session(work, session_id="destinations") as session:
        internal = session.register_destination(source / "findings.json", label=OutputLabel.FINDINGS_OUTPUT)
        assert internal.requested == source / "findings.json"
        assert internal.write_path is not None
        assert internal.write_path.is_relative_to(session.layout.live_root)
        assert internal.delivery is artifact_visibility.DestinationDelivery.DEFERRED
        external = session.register_trajectory_output(tmp_path / "external" / "trajectory.json").full
        assert external.write_path == external.requested
        assert external.delivery is artifact_visibility.DestinationDelivery.LIVE_EXTERNAL
        with pytest.raises(ArtifactVisibilityError, match="destination collision"):
            session.register_destination(source / "." / "findings.json", label=OutputLabel.FINDINGS_OUTPUT)
        with pytest.raises(ArtifactVisibilityError, match="destination overlap"):
            session.register_destination(source / "findings.json" / "child", label=OutputLabel.DUMP_DIRECTORY)


async def test_artifact_session_freezes_snapshot_bytes_not_document_path_and_publishes(
    tmp_path: Path,
    source: Path,
) -> None:
    work = _work(source)
    session_id = "snapshot-session"
    payload = _payload(session_id)

    async with open_artifact_session(work, session_id=session_id) as session:
        stale = tmp_path / "stale-trajectory.json"
        stale.write_bytes(b"wrong bytes")
        frozen = _freeze_run(session, session_id, payload=payload)
        assert (frozen.root / ".daydream" / "runs" / session_id / "trajectory.json").read_bytes() == payload
        assert stale.read_bytes() == b"wrong bytes"
        with pytest.raises(ArtifactVisibilityError, match="frozen"):
            artifact_dir_for(work.repo)
        _publish(session, frozen)

    assert (source / ".daydream" / "runs" / session_id / "trajectory.json").read_bytes() == payload


@pytest.mark.parametrize("problem", ["wrong-root", "wrong-session", "duplicate-path"])
async def test_artifact_session_freeze_rejects_invalid_run_snapshot(source: Path, problem: str) -> None:
    work = _work(source)
    session_id = "snapshot-session"
    trajectory_id = "other" if problem == "wrong-session" else session_id
    async with open_artifact_session(work, session_id=session_id) as session:
        path = session.daydream_dir / "runs" / session_id / "trajectory.json"
        document = TrajectoryDocumentSnapshot(trajectory_id, path, _payload(trajectory_id))
        snapshot = _snapshot(
            session_id,
            (document, document) if problem == "duplicate-path" else (document,),
            root_trajectory_id="other" if problem == "wrong-root" else session_id,
        )
        with pytest.raises(ArtifactVisibilityError, match="snapshot"):
            session.freeze(snapshot)


@pytest.mark.parametrize("transition", _TRANSITIONS)
async def test_artifact_session_recovers_real_process_death_at_every_transition(
    tmp_path: Path,
    source: Path,
    artifact_runtime_root: Path,
    transition: str,
) -> None:
    _seed_public_artifacts(source)
    (source / "explicit-output.txt").write_bytes(b"prior explicit bytes")
    first_repo, second_repo = _worktrees(source, "ephemeral-one", "ephemeral-two")
    marker = tmp_path / f"production-{transition}"
    with _child_at_marker(
        marker,
        "production-transaction",
        str(source),
        str(first_repo),
        str(artifact_runtime_root),
        transition,
        str(marker),
    ):
        pass

    has_published_run = transition in {"PUBLISH_INSTALLED", "PUBLISH_VERIFIED"}
    async with open_artifact_session(
        _work(source, repo=second_repo),
        session_id=f"recovery-{transition.lower()}",
    ) as recovered:
        prior = recovered.daydream_dir / "deep" / "prior.md"
        published = recovered.daydream_dir / "runs" / "production-crash" / "trajectory.json"
        assert prior.read_bytes() == b"prior reasoning\n"
        assert published.exists() is has_published_run
        recovered.register_destination(source / "explicit-output.txt", label=OutputLabel.FINDINGS_OUTPUT)
        assert not (source / "explicit-output.txt").exists()

    assert (source / ".daydream" / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
    assert (source / ".daydream" / "runs" / "production-crash" / "trajectory.json").exists() is has_published_run
    assert (source / "explicit-output.txt").read_bytes() == (
        b"published explicit bytes" if has_published_run else b"prior explicit bytes"
    )
    state_root = artifact_runtime_root / _workspace_key(source)
    assert not any((state_root / "transactions").iterdir())


@pytest.mark.parametrize("transition", _CLEANUP_TRANSITIONS)
async def test_artifact_session_recovers_real_process_death_during_cleanup(
    tmp_path: Path,
    source: Path,
    artifact_runtime_root: Path,
    transition: str,
) -> None:
    _seed_public_artifacts(source)
    (source / "explicit-output.txt").write_bytes(b"prior explicit bytes")
    first_repo, second_repo = _worktrees(source, "cleanup-ephemeral-one", "cleanup-ephemeral-two")
    source_canary = source.parent / "cleanup-source-canary"
    source_canary.write_bytes(b"source sibling")
    runtime_canary = artifact_runtime_root.parent / "cleanup-runtime-canary"
    runtime_canary.parent.mkdir(parents=True, exist_ok=True)
    runtime_canary.write_bytes(b"runtime sibling")
    marker = tmp_path / f"production-cleanup-{transition}"
    with _child_at_marker(
        marker,
        "production-transaction",
        str(source),
        str(first_repo),
        str(artifact_runtime_root),
        transition,
        str(marker),
    ):
        pass

    async with open_artifact_session(
        _work(source, repo=second_repo),
        session_id=f"cleanup-recovery-{transition.lower()}",
    ) as recovered:
        assert (recovered.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"

    state_root = artifact_runtime_root / _workspace_key(source)
    assert not any((state_root / "transactions").iterdir())
    cleanup = state_root / "cleanup"
    assert not cleanup.exists() or not any(cleanup.iterdir())
    assert source_canary.read_bytes() == b"source sibling"
    assert runtime_canary.read_bytes() == b"runtime sibling"


async def test_artifact_session_lock_rejects_live_process_and_recovers_after_death(
    tmp_path: Path,
    source: Path,
    artifact_runtime_root: Path,
) -> None:
    _seed_public_artifacts(source)
    marker = tmp_path / "production-lock"
    with _child_at_marker(marker, "production-lock", str(source), str(artifact_runtime_root), str(marker)):
        with pytest.raises(ArtifactVisibilityError, match="locked by another process"):
            async with open_artifact_session(_work(source), session_id="contender"):
                pass

    async with open_artifact_session(_work(source), session_id="after-death") as session:
        assert (session.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"


async def test_artifact_session_recovers_staged_detach_with_existing_canonical(
    tmp_path: Path,
    source: Path,
    artifact_runtime_root: Path,
) -> None:
    _seed_public_artifacts(source)
    async with open_artifact_session(_work(source), session_id="prime-canonical"):
        pass
    marker = tmp_path / "staged-with-canonical"
    with _child_at_marker(
        marker,
        "production-transaction",
        str(source),
        str(source),
        str(artifact_runtime_root),
        "DETACH_STAGED",
        str(marker),
    ):
        pass

    async with open_artifact_session(_work(source), session_id="recover-existing") as session:
        assert (session.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"


async def test_artifact_session_recovers_registered_explicit_baseline_after_death(
    tmp_path: Path,
    source: Path,
    artifact_runtime_root: Path,
) -> None:
    requested = source / "explicit-output.txt"
    requested.write_bytes(b"prior explicit bytes")
    marker = tmp_path / "explicit-registered"
    with _child_at_marker(
        marker,
        "production-transaction",
        str(source),
        str(source),
        str(artifact_runtime_root),
        "EXPLICIT_REGISTERED",
        str(marker),
    ):
        assert not requested.exists()

    async with open_artifact_session(_work(source), session_id="recover-explicit"):
        assert requested.read_bytes() == b"prior explicit bytes"
    assert requested.read_bytes() == b"prior explicit bytes"


async def test_artifact_session_rejects_existing_session_and_malformed_transaction(
    source: Path,
    artifact_runtime_root: Path,
) -> None:
    async with open_artifact_session(_work(source), session_id="fixed-session"):
        pass
    with pytest.raises(ArtifactVisibilityError, match="session id already exists"):
        async with open_artifact_session(_work(source), session_id="fixed-session"):
            pass

    state_root = artifact_runtime_root / _workspace_key(source)
    residue = state_root / "transactions" / "malformed"
    residue.mkdir()
    (residue / "journal.json").write_bytes(b"not-json")
    with pytest.raises(ArtifactVisibilityError, match="metadata is malformed"):
        async with open_artifact_session(_work(source), session_id="after-malformed"):
            pass


@pytest.mark.parametrize(
    "manifest_problem",
    [
        "unsafe-path",
        "duplicate-path",
        "dot",
        "leading-dot",
        "embedded-dot",
        "empty-component",
        "normalized-duplicate",
        "bool-schema-version",
    ],
)
async def test_artifact_session_rejects_closed_recovery_manifest_untouched(
    source: Path,
    manifest_problem: str,
) -> None:
    _seed_public_artifacts(source)
    state_root = _owner(source).artifact_state_root
    transaction = state_root / "transactions" / "malicious"
    stage = transaction / "detach-stage"
    stage.mkdir(parents=True)
    stage_canary = stage / "canary"
    stage_canary.write_bytes(b"stage must remain unchanged")
    raw_path = {
        "unsafe-path": "../escape",
        "duplicate-path": ".daydream",
        "dot": ".",
        "leading-dot": "./runs",
        "embedded-dot": "runs/./record.json",
        "empty-component": "runs//record.json",
        "normalized-duplicate": "runs/record.json",
        "bool-schema-version": "runs/record.json",
    }[manifest_problem]
    entry = {"path": raw_path, "kind": "directory", "size": 0, "mode": 0o700, "sha256": None}
    if manifest_problem == "duplicate-path":
        entries = [entry, entry]
    elif manifest_problem == "normalized-duplicate":
        entries = [entry, {**entry, "path": "./runs/record.json"}]
    else:
        entries = [entry]
    schema_version = True if manifest_problem == "bool-schema-version" else 1
    _atomic_json(transaction / "manifest.json", {"schema_version": schema_version, "entries": entries})
    _atomic_json(
        transaction / "journal.json",
        {
            "schema_version": 1,
            "transaction_id": "malicious",
            "session_id": "prior",
            "state": "DETACH_STAGED",
        },
    )

    with pytest.raises(ArtifactVisibilityError, match="manifest"):
        async with open_artifact_session(_work(source), session_id="blocked"):
            pass
    assert (source / ".daydream" / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
    assert stage_canary.read_bytes() == b"stage must remain unchanged"


async def test_bound_routing_rejects_same_spelling_repository_replacements(tmp_path: Path) -> None:
    container = tmp_path / "container"
    source = container / "source"
    _init_repo(source)
    work = _work(source)

    async with open_artifact_session(work, session_id="repo-replacement") as session:
        moved = tmp_path / "held-source"
        source.rename(moved)
        source.write_bytes(b"replacement file")
        try:
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
                    resolver(source)
        finally:
            source.unlink()
            moved.rename(source)

        moved = tmp_path / "held-source-git"
        source.rename(moved)
        _init_repo(source)
        try:
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
                    resolver(source)
        finally:
            source.rename(tmp_path / "unrelated-repository")
            moved.rename(source)

        moved_container = tmp_path / "held-container"
        container.rename(moved_container)
        container.symlink_to(moved_container, target_is_directory=True)
        try:
            for resolver in (artifact_dir_for, review_output_path_for):
                with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
                    resolver(source)
        finally:
            container.unlink()
            moved_container.rename(container)

        assert not session.layout.public_daydream_dir.exists()
        assert not session.layout.public_review_output.exists()


async def test_destination_collision_matrix_rejects_all_private_model_and_git_roots(
    tmp_path: Path,
    source: Path,
) -> None:
    (model_repo,) = _worktrees(source, "model-worktree")
    owner = _owner(source)

    async with open_artifact_session(
        _work(source, repo=model_repo),
        session_id="protected-destinations",
        owner=owner,
    ) as session:
        other_key = owner.artifact_state_root.parent / ("f" * 64)
        protected = (
            other_key / "output.json",
            owner.artifact_state_root.parent / "output.json",
            owner.operational_state_root / "output.json",
            owner.operational_state_root.parent / "output.json",
            model_repo / "output.json",
            source / ".git" / "config",
            model_repo / ".git",
            session.layout.repo_git_dir / "output.json",
            session.layout.git_common_dir / "output.json",
        )
        for index, requested in enumerate(protected):
            with pytest.raises(ArtifactVisibilityError, match="overlap|Git|model"):
                session.register_destination(requested, label=OutputLabel.FINDINGS_OUTPUT)
            assert not (session.layout.live_root / ".explicit" / f"{index:04d}").exists()

    async with open_artifact_session(_work(source), session_id="in-source-control") as session:
        route = session.register_destination(source / "safe-untracked.json", label=OutputLabel.FINDINGS_OUTPUT)
        assert route.write_path is not None
        assert route.write_path.is_relative_to(session.layout.live_root)


@pytest.mark.parametrize("alias_kind", ["hardlink", "normalized-name"])
async def test_artifact_session_rejects_duplicate_filesystem_aliases(source: Path, alias_kind: str) -> None:
    runs = source / ".daydream" / "runs"
    runs.mkdir(parents=True)
    first = runs / ("original" if alias_kind == "hardlink" else "e\u0301")
    second = runs / ("alias" if alias_kind == "hardlink" else "\u00e9")
    first.write_bytes(b"content")
    if alias_kind == "hardlink":
        os.link(first, second)
    else:
        second.write_bytes(b"other")
        if len({child.name for child in runs.iterdir()}) != 2:
            pytest.skip("filesystem canonicalizes the two Unicode spellings")

    with pytest.raises(ArtifactVisibilityError, match="duplicate"):
        async with open_artifact_session(_work(source), session_id=alias_kind):
            pass
    assert first.read_bytes() == b"content"


async def test_artifact_session_rejects_symlinked_source_and_runtime_ancestry(tmp_path: Path, source: Path) -> None:
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
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_public_artifacts(source)

    def mutate_before_removal(state: str) -> None:
        if state == "DETACH_REMOVING":
            (source / ".daydream" / "concurrent.bin").write_bytes(b"unique concurrent bytes")

    monkeypatch.setattr(artifact_visibility, "_transition_observer", mutate_before_removal)
    with pytest.raises(ArtifactVisibilityError, match="changed during detach"):
        async with open_artifact_session(_work(source), session_id="detach-mutation"):
            pass
    assert (source / ".daydream" / "concurrent.bin").read_bytes() == b"unique concurrent bytes"
    assert (source / ".daydream" / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"


async def test_detach_revalidates_each_file_after_an_earlier_removal(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_public_artifacts(source)
    owner = _owner(source)
    first = source / ".daydream" / "deep" / "prior.md"
    later = source / ".daydream" / "runs" / "opaque.bin"
    mutated = False

    def transfer_and_mutate_later(path: Path, _moved: Path) -> None:
        nonlocal mutated
        if path == first and not mutated:
            mutated = True
            later.write_bytes(b"unique concurrent replacement")

    monkeypatch.setattr(artifact_visibility, "_transfer_post_observer", transfer_and_mutate_later)
    with pytest.raises(ArtifactVisibilityError, match="changed|conflict"):
        async with open_artifact_session(_work(source), session_id="per-file-mutation", owner=owner):
            pass

    assert mutated is True
    assert later.read_bytes() == b"unique concurrent replacement"
    assert (owner.artifact_state_root / "canonical" / ".daydream" / "deep" / "prior.md").read_bytes() == (
        b"prior reasoning\n"
    )


async def test_artifact_session_detects_publication_insertion_without_deleting_it(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_public_artifacts(source)
    session_id = "publish-mutation"
    with pytest.raises(ArtifactVisibilityError, match="changed during recovery"):
        async with open_artifact_session(_work(source), session_id=session_id) as session:
            frozen = _freeze_run(session, session_id)

            def insert_before_install(state: str) -> None:
                if state == "PUBLISH_BACKED_UP":
                    concurrent = source / ".daydream" / "concurrent.bin"
                    concurrent.parent.mkdir()
                    concurrent.write_bytes(b"unique concurrent bytes")

            monkeypatch.setattr(artifact_visibility, "_transition_observer", insert_before_install)
            with pytest.raises(ArtifactVisibilityError, match="changed before publication"):
                _publish(session, frozen)

    assert (source / ".daydream" / "concurrent.bin").read_bytes() == b"unique concurrent bytes"
    canonical = frozen.root.parents[2] / "canonical"
    assert (canonical / ".daydream" / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"


async def test_publication_preparation_failure_restores_source_without_orphan_state(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
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


@pytest.mark.parametrize("exception_kind", ["ordinary", "cancel", "keyboard"])
async def test_recoverable_detach_failure_restores_before_unlock_and_preserves_primary(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_kind: str,
) -> None:
    expected = _seed_public_artifacts(source)
    if exception_kind == "ordinary":
        primary: BaseException = RuntimeError("ordinary primary")
    elif exception_kind == "cancel":
        primary = anyio.get_cancelled_exc_class()()
    else:
        primary = KeyboardInterrupt("keyboard primary")
    observed = False

    def fail_after_transfer(_path: Path, _moved: Path) -> None:
        nonlocal observed
        if observed:
            return
        observed = True
        raise primary

    monkeypatch.setattr(artifact_visibility, "_transfer_post_observer", fail_after_transfer)
    caught: BaseException | None = None
    try:
        async with open_artifact_session(_work(source), session_id=f"primary-{exception_kind}"):
            pass
    except BaseException as exc:
        caught = exc
    assert observed is True
    assert caught is primary
    assert _manifest(source) == expected

    monkeypatch.setattr(artifact_visibility, "_transfer_post_observer", None)
    async with open_artifact_session(_work(source), session_id=f"after-primary-{exception_kind}"):
        assert not (source / ".daydream").exists()


@pytest.mark.parametrize(
    ("transition", "exception_kind"),
    [("PUBLISH_BACKED_UP", "cancel"), ("PUBLISH_INSTALLED", "keyboard")],
)
async def test_recoverable_publication_failure_reconciles_before_unlock(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
    exception_kind: str,
) -> None:
    expected = _seed_public_artifacts(source)
    session_id = f"recover-{transition.lower()}"
    payload = _payload(session_id)
    primary: BaseException = (
        anyio.get_cancelled_exc_class()()
        if exception_kind == "cancel"
        else KeyboardInterrupt("publication primary")
    )

    def fail_at_transition(state: str) -> None:
        if state == transition:
            raise primary

    caught: BaseException | None = None
    try:
        async with open_artifact_session(_work(source), session_id=session_id) as session:
            frozen = _freeze_run(session, session_id, payload=payload)
            monkeypatch.setattr(artifact_visibility, "_transition_observer", fail_at_transition)
            _publish(session, frozen)
    except BaseException as exc:
        caught = exc
    assert caught is primary
    if transition == "PUBLISH_INSTALLED":
        assert (source / ".daydream" / "runs" / session_id / "trajectory.json").read_bytes() == payload
    else:
        assert _manifest(source) == expected
    monkeypatch.setattr(artifact_visibility, "_transition_observer", None)
    async with open_artifact_session(_work(source), session_id=f"after-{session_id}"):
        assert not (source / ".daydream").exists()


async def test_artifact_session_destination_collision_matrix(tmp_path: Path, source: Path) -> None:
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


async def test_in_source_explicit_file_is_hidden_then_restored_or_published(source: Path) -> None:
    requested = source / "output with spaces.json"
    requested.write_bytes(b"prior explicit bytes")

    async with open_artifact_session(_work(source), session_id="restore-explicit") as session:
        routed = session.register_destination(requested, label=OutputLabel.FINDINGS_OUTPUT)
        assert not requested.exists()
        assert routed.delivery is artifact_visibility.DestinationDelivery.DEFERRED
        assert routed.write_path is not None
        routed.write_path.parent.mkdir(parents=True)
        routed.write_path.write_bytes(b"discarded failed-run bytes")
    assert requested.read_bytes() == b"prior explicit bytes"

    session_id = "publish-explicit"
    async with open_artifact_session(_work(source), session_id=session_id) as session:
        routed = session.register_destination(requested, label=OutputLabel.FINDINGS_OUTPUT)
        assert not requested.exists()
        assert routed.write_path is not None
        routed.write_path.parent.mkdir(parents=True)
        routed.write_path.write_bytes(b"published explicit bytes")
        frozen = _freeze_run(session, session_id)
        _publish(session, frozen)
    assert requested.read_bytes() == b"published explicit bytes"


async def test_in_source_dump_publication_merges_and_preserves_unrelated_bytes(source: Path) -> None:
    requested = source / "dump output"
    requested.mkdir()
    (requested / "unrelated.bin").write_bytes(b"preserve me")
    (requested / "replace.txt").write_bytes(b"old")
    session_id = "publish-dump"

    async with open_artifact_session(_work(source), session_id=session_id) as session:
        routed = session.register_destination(requested, label=OutputLabel.DUMP_DIRECTORY)
        assert requested.is_dir()
        assert not any(path.is_file() for path in requested.rglob("*"))
        assert routed.write_path is None
        frozen = _freeze_run(session, session_id)
        late = session.finalization_merge_path(routed, snapshot=frozen)
        (late / "replace.txt").write_bytes(b"new")
        (late / "added.txt").write_bytes(b"added")
        _publish(session, frozen)

    assert (requested / "unrelated.bin").read_bytes() == b"preserve me"
    assert (requested / "replace.txt").read_bytes() == b"new"
    assert (requested / "added.txt").read_bytes() == b"added"


async def test_absent_nested_explicit_destination_leaves_no_parent_on_failure(source: Path) -> None:
    requested = source / "new parent" / "nested" / "findings.json"
    async with open_artifact_session(_work(source), session_id="absent-explicit") as session:
        routed = session.register_destination(requested, label=OutputLabel.FINDINGS_OUTPUT)
        routed.write_path.parent.mkdir(parents=True)
        routed.write_path.write_bytes(b"failed run")
        assert not requested.exists()
    assert not requested.exists()
    assert not (source / "new parent").exists()


async def test_nested_artifact_sessions_restore_exact_context_token(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _init_repo(first)
    _init_repo(second)
    first_work = _work(first)
    second_work = _work(second)

    async with open_artifact_session(first_work, session_id="first") as first_session:
        assert artifact_dir_for(first) == first_session.daydream_dir
        async with open_artifact_session(second_work, session_id="second") as second_session:
            assert artifact_dir_for(second) == second_session.daydream_dir
            with pytest.raises(ArtifactVisibilityError, match="active artifact session"):
                artifact_dir_for(first)
        assert artifact_dir_for(first) == first_session.daydream_dir
    assert artifact_dir_for(first) == first / ".daydream"


async def test_artifact_session_rejects_forged_snapshot_object(tmp_path: Path, source: Path) -> None:
    session_id = "forged-snapshot"
    async with open_artifact_session(_work(source), session_id=session_id) as session:
        snapshot = _freeze_run(session, session_id)
        forged = ArtifactTreeSnapshot(
            session_id=snapshot.session_id,
            workspace_key=snapshot.workspace_key,
            root=tmp_path,
            manifest=snapshot.manifest,
            destinations=snapshot.destinations,
        )
        with pytest.raises(ArtifactVisibilityError, match="snapshot identity mismatch"):
            _publish(session, forged)


async def test_trajectory_output_route_pairs_external_baselines_and_rejects_unpaired(
    tmp_path: Path,
    source: Path,
) -> None:
    requested, partial = _external_target(tmp_path / "external")

    async with open_artifact_session(_work(source), session_id="external-route") as session:
        route = session.register_trajectory_output(requested)
        assert route.run_dir == session.daydream_dir / "runs" / "external-route"
        assert route.full.requested == requested
        assert route.partial.requested == partial
        assert route.full.write_path == requested
        assert route.partial.write_path == partial
        assert route.full.frozen_path == route.run_dir / "trajectory.json"
        assert route.partial.frozen_path == route.run_dir / "trajectory.json.partial"
        assert route.full.delivery is artifact_visibility.DestinationDelivery.LIVE_EXTERNAL
        assert route.partial.delivery is artifact_visibility.DestinationDelivery.LIVE_EXTERNAL
        registry = _load_json(session._detach_transaction / "destinations.json")
        assert len(cast(list[object], registry["destinations"])) == 2
        for label in (
            OutputLabel.EXPLICIT_TRAJECTORY,
            OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL,
        ):
            with pytest.raises(ArtifactVisibilityError, match="paired trajectory"):
                session.register_destination(requested, label=label)


@pytest.mark.parametrize(
    ("disposition", "status"),
    [
        (artifact_visibility.ArtifactDisposition.COMPLETE, "complete"),
        (artifact_visibility.ArtifactDisposition.PARTIAL_EVIDENCE, "partial"),
        (artifact_visibility.ArtifactDisposition.ROLLBACK, "complete"),
    ],
)
async def test_public_subtree_trajectory_uses_whole_daydream_transaction(
    source: Path,
    disposition: artifact_visibility.ArtifactDisposition,
    status: str,
) -> None:
    _seed_public_artifacts(source)
    custom_relative = Path("custom outputs") / "nested root.json"
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
        public_owner = session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        review_owner = session.register_destination(
            source / ".review-output.md",
            label=OutputLabel.PUBLIC_REVIEW_OUTPUT,
        )
        assert not (source / ".daydream").exists()
        destinations_before = tuple(session._destinations)
        records_before = tuple(session._records())

        route = session.register_trajectory_output(requested)
        private_full = session.daydream_dir / custom_relative
        private_partial = private_full.with_suffix(private_full.suffix + ".partial")
        assert route.full.requested == requested
        assert route.partial.requested == partial_requested
        assert route.full.write_path == route.full.frozen_path == private_full
        assert route.partial.write_path == route.partial.frozen_path == private_partial
        assert route.run_dir == session.daydream_dir / "runs" / session_id
        assert tuple(session._destinations) == destinations_before == (public_owner, review_owner)
        assert tuple(session._records()) == records_before == ()
        assert not (session._detach_transaction / "destinations.json").exists()
        assert private_full.read_bytes() == b"prior full"
        assert private_partial.read_bytes() == b"prior partial"
        assert (session.daydream_dir / custom_relative.parent / "unrelated.bin").read_bytes() == b"unrelated"
        assert (session.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"

        selected = route.full if status == "complete" else route.partial
        root_document = TrajectoryDocumentSnapshot(session_id, cast(Path, selected.write_path), root_bytes)
        session.write_trajectory_document(route, root_document, cast(Any, status))
        assert cast(Path, selected.write_path).read_bytes() == root_bytes
        assert not (route.run_dir / "trajectory.json").exists()
        assert not (route.run_dir / "trajectory.json.partial").exists()
        child_path = route.run_dir / "trajectories" / "child.json"
        child_document = TrajectoryDocumentSnapshot(child_id, child_path, child_bytes)
        session.write_trajectory_document(route, child_document, cast(Any, status))
        assert child_path.read_bytes() == child_bytes
        assert not child_path.is_relative_to(private_full.parent)
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
    published_child = source / ".daydream" / "runs" / session_id / "trajectories" / "child.json"
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


def _open_rolled_back_dump(session: Any, session_id: str, dump: Path) -> Path:
    """Register an absent dump destination on ``session`` and roll the run back.

    Mirrors what a ``--dump-artifacts`` run does when strict archive
    finalization refuses: the finalization-merge stage is created and left
    empty, because ``finalize_archive_run`` raises before it copies anything
    into it, and the disposition becomes ``ROLLBACK``.
    """
    session.register_destination(session.layout.source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
    session.register_destination(
        session.layout.source / ".review-output.md", label=OutputLabel.PUBLIC_REVIEW_OUTPUT
    )
    route = session.register_destination(dump, label=OutputLabel.DUMP_DIRECTORY)
    frozen = _freeze_run(session, session_id)
    late = cast(Path, session.finalization_merge_path(route, snapshot=frozen))
    assert late.is_dir()
    assert not any(late.iterdir())
    _publish(session, frozen, artifact_visibility.ArtifactDisposition.ROLLBACK)
    return late


async def test_rollback_of_an_absent_dump_destination_reopens_the_workspace(source: Path) -> None:
    """#1171/#1172: an empty directory baseline is a restore target, not a fault.

    A ``--dump-artifacts`` directory that did not exist before the run records
    an empty baseline, so "restore to nothing" is the correct rollback target.
    Raising "dump destination projection is malformed" instead aborted
    ``_restore_prior`` before it retired anything, leaving a ``DETACHED``
    journal that every later session open replayed into the identical failure,
    plus a leaked restore stage that made the third open fail even earlier with
    "artifact source-stage collision".
    """
    baseline = _seed_public_artifacts(source)
    owner = _owner(source)
    transactions = owner.artifact_state_root / "transactions"
    dump = source.parent / "never-existed" / "out"

    async with open_artifact_session(_work(source), session_id="rollback-dump", owner=owner) as session:
        late = _open_rolled_back_dump(session, "rollback-dump", dump)

    assert not dump.exists()
    assert not dump.parent.exists()
    assert _manifest(source) == baseline
    assert not any(transactions.iterdir())
    assert not late.exists()
    assert not late.parent.exists()
    assert _restore_stage_leaks(source) == []

    # Two further opens: the first proves the replay is gone, the second that
    # no restore stage was left to collide with.
    for session_id in ("after-rollback-one", "after-rollback-two"):
        async with open_artifact_session(_work(source), session_id=session_id, owner=owner) as reopened:
            assert (reopened.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
        assert _manifest(source) == baseline
        assert not any(transactions.iterdir())
        assert _restore_stage_leaks(source) == []


async def test_publishing_an_unused_dump_destination_leaves_it_absent(source: Path) -> None:
    """The publish direction of the same projection: nothing written, nothing created.

    ``_merge_directory_from_tree`` is reached twice — once with the baseline as
    the projection (rollback) and once with the published manifest (install).
    A registered dump route whose finalization stage was never taken publishes
    an empty manifest, which used to abort the whole publication with
    "dump destination projection is malformed".
    """
    _seed_public_artifacts(source)
    owner = _owner(source)
    dump = source.parent / "never-existed" / "out"

    async with open_artifact_session(_work(source), session_id="unused-dump", owner=owner) as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        session.register_destination(source / ".review-output.md", label=OutputLabel.PUBLIC_REVIEW_OUTPUT)
        session.register_destination(dump, label=OutputLabel.DUMP_DIRECTORY)
        _publish(session, _freeze_run(session, "unused-dump"))

    assert not dump.exists()
    assert not dump.parent.exists()
    assert (source / ".daydream" / "runs" / "unused-dump" / "trajectory.json").is_file()
    assert (source / ".daydream" / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
    assert not any((owner.artifact_state_root / "transactions").iterdir())


async def test_session_open_heals_a_pre_fix_wedged_dump_transaction(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1172: a workspace already wedged by the old behaviour heals on next open.

    The wedge is built with the pre-fix code paths restored — the directory
    merge raising on an empty projection, and a failed restore that neither
    preserved the transaction nor reclaimed its stage — so the on-disk state is
    the one operators are stuck with today: a ``DETACHED`` journal plus a
    leaked ``.<source>.daydream-restore-*`` stage. Nothing about the wedge is
    repaired by hand; the next real session open has to do it.
    """
    baseline = _seed_public_artifacts(source)
    owner = _owner(source)
    transactions = owner.artifact_state_root / "transactions"
    dump = source.parent / "never-existed" / "out"
    merge = artifact_visibility._merge_directory_from_tree

    def pre_fix_merge(tree: Path, record: Any, *, desired: Any, **kwargs: Any) -> None:
        if not desired:
            raise ArtifactVisibilityError("dump destination projection is malformed")
        merge(tree, record, desired=desired, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(artifact_visibility, "_merge_directory_from_tree", pre_fix_merge)
        patched.setattr(
            artifact_visibility, "_clear_failed_restore", lambda _root, _source, _txn, error: error
        )
        patched.setattr(artifact_visibility, "_reset_source_stage", lambda *_args, **_kwargs: None)

        with pytest.raises(ArtifactVisibilityError, match="projection is malformed"):
            async with open_artifact_session(_work(source), session_id="wedge", owner=owner) as session:
                _open_rolled_back_dump(session, "wedge", dump)

        wedged = [child.name for child in transactions.iterdir()]
        assert len(wedged) == 1
        assert _load_json(transactions / wedged[0] / "journal.json")["state"] == "DETACHED"

        # The second open replays the journal, fails identically, and leaks the
        # restore stage that makes every later open collide instead.
        with pytest.raises(ArtifactVisibilityError, match="projection is malformed"):
            async with open_artifact_session(_work(source), session_id="wedged-open", owner=owner):
                pass
        assert _restore_stage_leaks(source) == [f".{source.name}.daydream-restore-{wedged[0]}"]

    async with open_artifact_session(_work(source), session_id="healed", owner=owner) as healed:
        assert (healed.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"

    assert _manifest(source) == baseline
    assert not any(transactions.iterdir())
    assert _restore_stage_leaks(source) == []
    assert not dump.exists()
    assert not (owner.artifact_state_root / "unreconciled").exists()


async def test_failed_destination_restore_preserves_the_transaction_and_reopens(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1172: a restore that cannot finish is preserved, never replayed forever.

    Recovery reruns the same records through the same code, so a destination
    restore that fails once fails identically at every later open. The
    transaction's baselines are the only surviving copy of an in-source
    destination's prior bytes, so the transaction is moved under
    ``unreconciled/`` rather than retired — the operator keeps every byte, the
    error names where they are, and a fresh session still starts.
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
        patched.setattr(artifact_visibility, "_restore_destination_records", refuse)
        with pytest.raises(ArtifactVisibilityError, match="synthetic destination restore failure") as caught:
            async with open_artifact_session(_work(source), session_id="unrestorable", owner=owner) as session:
                session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
                session.register_destination(
                    source / ".review-output.md", label=OutputLabel.PUBLIC_REVIEW_OUTPUT
                )
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

    # Nothing was discarded: the public tree is back and the baseline copy of
    # the explicit destination survives under the preserved transaction.
    assert _manifest(source) == baseline
    assert (preserved / retained[0] / "destination-0000-baseline" / "findings.json").read_bytes() == (
        b"prior findings"
    )
    assert not any(transactions.iterdir())
    assert _restore_stage_leaks(source) == []

    async with open_artifact_session(_work(source), session_id="after-unrestorable", owner=owner) as reopened:
        assert (reopened.daydream_dir / "deep" / "prior.md").read_bytes() == b"prior reasoning\n"
    assert not any(transactions.iterdir())


async def test_public_subtree_trajectory_requires_exact_public_owner(source: Path) -> None:
    requested = source / ".daydream" / "custom.json"

    async with open_artifact_session(_work(source), session_id="missing-owner") as session:
        with pytest.raises(ArtifactVisibilityError, match="overlaps a public compatibility root"):
            session.register_trajectory_output(requested)


@pytest.mark.parametrize("replacement", ["symlink", "file"])
async def test_public_subtree_trajectory_checks_private_root_itself(
    tmp_path: Path,
    source: Path,
    replacement: str,
) -> None:
    baseline = _seed_public_artifacts(source)
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "canary.bin").write_bytes(b"unrelated bytes")
    outside_before = artifact_visibility._manifest(outside)

    async with open_artifact_session(_work(source), session_id="root-replaced") as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        private_root = session.daydream_dir
        saved_root = session.layout.live_root / "saved-daydream"
        private_root.rename(saved_root)
        try:
            if replacement == "symlink":
                private_root.symlink_to(outside, target_is_directory=True)
            else:
                private_root.write_bytes(b"not a directory")
            with pytest.raises(ArtifactVisibilityError, match="private trajectory"):
                session.register_trajectory_output(source / ".daydream" / "custom" / "root.json")
            assert session._trajectory_route is None
            assert artifact_visibility._manifest(outside) == outside_before
        finally:
            private_root.unlink()
            saved_root.rename(private_root)

    assert _manifest(source) == baseline
    assert artifact_visibility._manifest(outside) == outside_before


async def test_public_subtree_trajectory_allows_missing_private_root(source: Path) -> None:
    session_id = "fresh-custom-root"
    requested = source / ".daydream" / "custom" / "root.json"
    payload = _payload(session_id)
    async with open_artifact_session(_work(source), session_id=session_id) as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        assert not session.daydream_dir.exists()
        route = session.register_trajectory_output(requested)
        assert route.full.write_path is not None
        document = TrajectoryDocumentSnapshot(session_id, route.full.write_path, payload)
        session.write_trajectory_document(route, document, "complete")
        assert not (source / ".daydream").exists()
        frozen = session.freeze(_snapshot(session_id, (document,)))
        _publish(session, frozen)
    assert requested.read_bytes() == payload


@pytest.mark.parametrize(
    "problem",
    ["directory-leaf", "file-ancestor", "symlink-leaf", "symlink-ancestor"],
)
async def test_public_subtree_trajectory_rejects_unsafe_private_counterpart(source: Path, problem: str) -> None:
    baseline = _seed_public_artifacts(source)
    requested = source / ".daydream" / "nested" / "custom.json"

    async with open_artifact_session(_work(source), session_id=problem) as session:
        session.register_destination(source / ".daydream", label=OutputLabel.PUBLIC_DAYDREAM)
        private_nested = session.daydream_dir / "nested"
        private_leaf = private_nested / "custom.json"
        if problem == "directory-leaf":
            private_leaf.mkdir(parents=True)
        elif problem == "file-ancestor":
            private_nested.write_bytes(b"not a directory")
        elif problem == "symlink-leaf":
            private_nested.mkdir()
            private_leaf.symlink_to(session.daydream_dir / "deep" / "prior.md")
        else:
            private_nested.symlink_to(session.daydream_dir / "deep", target_is_directory=True)

        with pytest.raises(ArtifactVisibilityError, match="private trajectory"):
            session.register_trajectory_output(requested)

    assert _manifest(source) == baseline


async def test_independent_model_cwd_rejects_live_external_destination_overlap(tmp_path: Path, source: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    requested = external / "nested" / "trajectory.json"
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


async def test_live_external_capability_probe_covers_exchange_link_and_missing_parent(
    tmp_path: Path,
    source: Path,
) -> None:
    requested = tmp_path / "missing" / "nested" / "trajectory.json"
    canary = tmp_path / "probe-canary"
    canary.write_bytes(b"unrelated")

    async with open_artifact_session(_work(source), session_id="probe-success") as session:
        route = session.register_trajectory_output(requested)
        assert route.full.delivery is artifact_visibility.DestinationDelivery.LIVE_EXTERNAL
        records = cast(
            list[dict[str, object]],
            _load_json(session._detach_transaction / "external-entries.json")["entries"],
        )
        purposes = {cast(str, record["purpose"]) for record in records}
        assert {"probe_exchange_a", "probe_exchange_b", "probe_link_target", "missing_parent"} <= purposes
        assert all(
            record["lifecycle"] == "retired"
            for record in records
            if cast(str, record["purpose"]).startswith("probe_")
        )
        assert not any(path.name.startswith(".daydream-probe-") for path in requested.parent.iterdir())

    assert not requested.parent.exists()
    assert canary.read_bytes() == b"unrelated"


async def test_live_external_refused_link_probe_rejects_route_before_producer(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested, _ = _external_target(tmp_path / "external", prior=False)

    def refuse_link(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("test link refusal")

    monkeypatch.setattr(artifact_visibility, "_external_link", refuse_link)
    async with open_artifact_session(_work(source), session_id="probe-link-refused") as session:
        with pytest.raises(ArtifactVisibilityError, match="link probe"):
            session.register_trajectory_output(requested)
    # Rejection is before any producing write: no trajectory file and no
    # staged probe residue may exist at the destination.
    assert not requested.exists()
    assert not any(path.name.startswith(".daydream-probe-") for path in requested.parent.iterdir())


async def test_unsupported_exchange_rejects_live_route_before_producer(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested, _ = _external_target(tmp_path / "external", prior=False)

    def unsupported() -> None:
        raise ArtifactVisibilityError("live external atomic exchange is unsupported")

    monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", unsupported)
    async with open_artifact_session(_work(source), session_id="unsupported-exchange") as session:
        with pytest.raises(ArtifactVisibilityError, match="unsupported"):
            session.register_trajectory_output(requested)
    assert not requested.exists()


async def test_atomic_exchange_visibility_is_always_one_complete_document(tmp_path: Path, source: Path) -> None:
    requested, _ = _external_target(tmp_path / "external")
    observed: list[bytes] = []
    missing: list[bool] = []
    stop = threading.Event()

    async with open_artifact_session(_work(source), session_id="atomic-visible") as session:
        route = session.register_trajectory_output(requested)
        payloads = tuple(_payload("atomic-visible", turn=turn) for turn in range(20))

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
                    route,
                    TrajectoryDocumentSnapshot("atomic-visible", requested, payload),
                    "complete",
                )
        finally:
            stop.set()
            reader.join(timeout=5)
        assert not reader.is_alive()
        assert not missing
        assert observed
        assert set(observed) <= {b"prior full", *payloads}


async def test_absent_first_publication_refuses_concurrent_target_without_exchange(
    tmp_path: Path,
    source: Path,
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
                    route,
                    TrajectoryDocumentSnapshot("absent-race", requested, payload),
                    "complete",
                )
    assert requested.read_bytes() == unique
    stages = list(requested.parent.glob(".daydream-output-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == payload


class _FailAfterExchange:
    def __init__(self, *, impossible: bool = False) -> None:
        self.real = artifact_visibility._AtomicNameExchange()
        self.calls = 0
        self.impossible = impossible

    def call(self, parent_fd: int, staged_name: str, target_name: str) -> Any:
        self.calls += 1
        if self.calls <= 2:
            return self.real.call(parent_fd, staged_name, target_name)
        if self.impossible:
            return artifact_visibility._NameExchangeResult(7, None)
        result = self.real.call(parent_fd, staged_name, target_name)
        assert result.result == 0
        return artifact_visibility._NameExchangeResult(-1, 5)


class _ReplaceBeforeSuccessfulExchange:
    def __init__(self, requested: Path, unexpected: bytes, events: list[str]) -> None:
        self.real = artifact_visibility._AtomicNameExchange()
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
        self.real = artifact_visibility._AtomicNameExchange()
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


@pytest.mark.parametrize("impossible", [False, True])
async def test_exchange_nonzero_never_reports_success_or_falls_back(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    impossible: bool,
) -> None:
    requested, _ = _external_target(tmp_path / f"external-{impossible}")
    exchange = _FailAfterExchange(impossible=impossible)
    monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", lambda: exchange)
    payload = _payload(f"exchange-{impossible}")

    expected_message = "failed after mutation" if not impossible else "atomic exchange"
    with pytest.raises(ArtifactVisibilityError, match=expected_message):
        async with open_artifact_session(_work(source), session_id=f"exchange-{impossible}") as session:
            route = session.register_trajectory_output(requested)
            session.write_trajectory_document(
                route,
                TrajectoryDocumentSnapshot(f"exchange-{impossible}", requested, payload),
                "complete",
            )
    assert requested.read_bytes() == b"prior full"
    assert exchange.calls == (4 if not impossible else 3)


async def test_zero_exchange_with_unexpected_displaced_target_reverses_once_and_fails(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested, _ = _external_target(tmp_path / "external")
    unexpected = b"unexpected displaced target"
    events: list[str] = []
    exchange = _ReplaceBeforeSuccessfulExchange(requested, unexpected, events)
    monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", lambda: exchange)
    monkeypatch.setattr(
        artifact_visibility,
        "_external_entry_observer",
        lambda state, purpose, _path: events.append(state)
        if purpose == "publication_stage"
        else None,
    )
    payload = _payload("unexpected-zero")

    with pytest.raises(ArtifactVisibilityError, match="failed after mutation"):
        async with open_artifact_session(_work(source), session_id="unexpected-zero") as session:
            route = session.register_trajectory_output(requested)
            session.write_trajectory_document(
                route,
                TrajectoryDocumentSnapshot("unexpected-zero", requested, payload),
                "complete",
            )

    assert exchange.calls == 4
    assert "REVERSAL_ATTEMPTED" in events
    assert events.index("REVERSAL_ATTEMPTED") < events.index("REVERSAL_CALLED")
    assert requested.read_bytes() == unexpected
    retained = [path.read_bytes() for path in requested.parent.glob(".daydream-output-*")]
    assert payload in retained
    monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", _no_reversal())
    with pytest.raises(ArtifactVisibilityError, match="conflict"):
        async with open_artifact_session(_work(source), session_id="unexpected-zero-reopen"):
            pass
    assert requested.read_bytes() == unexpected
    assert payload in [path.read_bytes() for path in requested.parent.glob(".daydream-output-*")]


async def test_zero_exchange_with_displaced_directory_records_conflict_without_reversal(
    tmp_path: Path,
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested, _ = _external_target(tmp_path / "external")
    exchange = _ReplaceWithDirectoryBeforeSuccessfulExchange(requested)
    monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", lambda: exchange)
    payload = _payload("nonregular-zero")

    with pytest.raises(ArtifactVisibilityError, match="entry|ambiguous|conflict"):
        async with open_artifact_session(_work(source), session_id="nonregular-zero") as session:
            route = session.register_trajectory_output(requested)
            session.write_trajectory_document(
                route,
                TrajectoryDocumentSnapshot("nonregular-zero", requested, payload),
                "complete",
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
    monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", _no_reversal())
    with pytest.raises(ArtifactVisibilityError, match="conflict"):
        async with open_artifact_session(_work(source), session_id="nonregular-reopen"):
            pass
    assert requested.read_bytes() == payload
    assert staged[0].is_dir()


def test_external_fifo_observation_is_nonblocking_and_reaps_child(tmp_path: Path) -> None:
    parent = tmp_path / "external"
    parent.mkdir()
    fifo = parent / "unexpected.fifo"
    os.mkfifo(fifo)
    entered = tmp_path / "fifo-entered"
    completed = tmp_path / "fifo-completed"
    process = _spawn_helper("external-observe-fifo", str(fifo), str(entered), str(completed))
    blocked = False
    try:
        _wait_for_marker(process, entered)
        try:
            # A blocking open on a writer-less FIFO never returns, so any finite
            # window proves non-blocking; keep it wide enough to survive load.
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            blocked = True
            process.kill()
            returncode = process.wait(timeout=5)
        assert blocked is False
        assert returncode == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.poll() is not None
    assert completed.read_text(encoding="ascii") == "_ExternalEntryIssue"
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


@pytest.mark.parametrize(
    ("checkpoint", "purpose", "conflict"),
    [
        # Publication entries: the atomic-exchange stage, its link target, and
        # the absent-parent route.
        ("CREATION_INTENT", "publication_stage", False),
        ("ENTRY_CREATED", "publication_stage", True),
        ("ATTESTED", "publication_stage", False),
        ("OPERATION_PREPARED", "publication_stage", False),
        ("PRIMARY_CALLED", "publication_stage", False),
        ("REVERSAL_ATTEMPTED", "publication_stage", True),
        ("REVERSAL_CALLED", "publication_stage", True),
        ("CLEANUP_PREPARED", "publication_stage", False),
        ("ENTRY_REMOVED", "publication_stage", False),
        ("CREATION_INTENT", "publication_link_target", False),
        ("LINK_CREATED", "publication_link_target", True),
        ("ATTESTED", "publication_link_target", False),
        ("CREATION_INTENT", "missing_parent", False),
        ("ENTRY_CREATED", "missing_parent", True),
        ("ATTESTED", "missing_parent", False),
        # Registration-time capability probes take the same reconciliation path.
        ("CREATION_INTENT", "probe_exchange_a", False),
        ("ENTRY_CREATED", "probe_exchange_a", True),
        ("ATTESTED", "probe_exchange_a", False),
        ("PRIMARY_CALLED", "probe_exchange_a", False),
        ("REVERSAL_ATTEMPTED", "probe_exchange_a", True),
        ("REVERSAL_CALLED", "probe_exchange_a", True),
        ("CREATION_INTENT", "probe_link_target", False),
        ("LINK_CREATED", "probe_link_target", True),
        ("ATTESTED", "probe_link_target", False),
    ],
)
async def test_external_entry_process_death_reconciles_attested_and_retains_unattested(
    tmp_path: Path,
    source: Path,
    artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
    purpose: str,
    conflict: bool,
) -> None:
    _seed_public_artifacts(source)
    first_repo, second_repo = _worktrees(source, "external-ephemeral-one", "external-ephemeral-two")
    marker = tmp_path / f"external-{checkpoint}-{purpose}"
    canary = source.parent / f"canary-{checkpoint}-{purpose}"
    canary.write_bytes(b"unrelated external canary")
    with _child_at_marker(
        marker,
        "production-external",
        str(source),
        str(first_repo),
        str(artifact_runtime_root),
        checkpoint,
        purpose,
        str(marker),
    ):
        pass

    owner = _owner(source)
    if checkpoint in ("REVERSAL_ATTEMPTED", "REVERSAL_CALLED"):
        monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", _no_reversal())
    reopen = open_artifact_session(
        _work(source, repo=second_repo),
        session_id=f"recover-{checkpoint.lower()}-{purpose}",
        owner=owner,
    )
    if conflict:
        with pytest.raises(ArtifactVisibilityError, match="conflict|unattested|reversal"):
            async with reopen:
                pass
        assert any((owner.artifact_state_root / "transactions").iterdir())
    else:
        async with reopen:
            pass
    assert canary.read_bytes() == b"unrelated external canary"


async def test_unexpected_displaced_target_death_after_reversal_marker_never_reverses_on_reopen(
    tmp_path: Path,
    source: Path,
    artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_public_artifacts(source)
    first_repo, second_repo = _worktrees(source, "unexpected-ephemeral-one", "unexpected-ephemeral-two")
    marker = tmp_path / "unexpected-reversal-attempted"
    with _child_at_marker(
        marker,
        "production-external",
        str(source),
        str(first_repo),
        str(artifact_runtime_root),
        "UNEXPECTED_REVERSAL_ATTEMPTED",
        "publication_stage",
        str(marker),
    ):
        pass

    requested = source.parent / "external-crash-output" / "trajectory.json"
    payload = _payload("external-crash")
    assert requested.read_bytes() == payload
    stages = list(requested.parent.glob(".daydream-output-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"unexpected displaced external bytes"
    monkeypatch.setattr(artifact_visibility, "_name_exchange_factory", _no_reversal())
    owner = _owner(source)
    with pytest.raises(ArtifactVisibilityError, match="reversal|conflict"):
        async with open_artifact_session(
            _work(source, repo=second_repo),
            session_id="unexpected-reversal-reopen",
            owner=owner,
        ):
            pass
    assert requested.read_bytes() == payload
    assert stages[0].read_bytes() == b"unexpected displaced external bytes"


async def test_external_trajectory_write_freeze_and_dispositions_use_immutable_bytes(
    tmp_path: Path,
    source: Path,
) -> None:
    requested, partial_path = _external_target(tmp_path / "external")
    session_id = "external-freeze"
    partial_bytes = _payload(session_id, session_id, **{"extra": {"partial": True}})
    full_bytes = _payload(session_id, session_id, **{"extra": {"partial": False}})

    with pytest.raises(ArtifactVisibilityError, match="changed|conflict"):
        async with open_artifact_session(_work(source), session_id=session_id) as session:
            route = session.register_trajectory_output(requested)
            partial_document = TrajectoryDocumentSnapshot(session_id, partial_path, partial_bytes)
            session.write_trajectory_document(route, partial_document, "partial")
            assert partial_path.read_bytes() == partial_bytes
            assert route.partial.frozen_path is not None
            assert route.partial.frozen_path.read_bytes() == partial_bytes
            full_document = TrajectoryDocumentSnapshot(session_id, requested, full_bytes)
            session.write_trajectory_document(route, full_document, "complete")
            assert requested.read_bytes() == full_bytes
            assert route.full.frozen_path is not None
            assert route.full.frozen_path.read_bytes() == full_bytes

            requested.unlink()
            partial_path.write_bytes(b"mutable path is not evidence")
            frozen = session.freeze(_snapshot(session_id, (full_document,)))
            assert (frozen.root / ".daydream" / "runs" / session_id / "trajectory.json").read_bytes() == full_bytes
            _publish(session, frozen)

    assert partial_path.read_bytes() == b"mutable path is not evidence"
    assert requested.read_bytes() == b"prior full"


@pytest.mark.parametrize("status", ["complete", "partial"])
async def test_child_trajectory_write_never_updates_external_root_destination(
    tmp_path: Path,
    source: Path,
    status: str,
) -> None:
    requested, partial = _external_target(tmp_path / f"external-{status}")
    session_id = f"child-{status}"
    selected_path = requested if status == "complete" else partial
    untouched_path = partial if status == "complete" else requested
    untouched_bytes = b"prior partial" if status == "complete" else b"prior full"
    root_bytes = _payload(session_id, session_id, **{"marker": "root"})
    child_id = f"{session_id}-child"
    child_bytes = _payload(session_id, child_id, **{"marker": "child"})

    async with open_artifact_session(_work(source), session_id=session_id) as session:
        route = session.register_trajectory_output(requested)
        root_document = TrajectoryDocumentSnapshot(session_id, selected_path, root_bytes)
        session.write_trajectory_document(route, root_document, cast(Any, status))
        assert selected_path.read_bytes() == root_bytes
        child_path = route.run_dir / "children" / f"{child_id}.json"
        child_document = TrajectoryDocumentSnapshot(child_id, child_path, child_bytes)
        session.write_trajectory_document(route, child_document, cast(Any, status))
        assert child_path.read_bytes() == child_bytes
        assert selected_path.read_bytes() == root_bytes
        assert untouched_path.read_bytes() == untouched_bytes
        frozen = session.freeze(_snapshot(session_id, (root_document, child_document), status=status))
        _publish(session, frozen)

    assert selected_path.read_bytes() == root_bytes
    assert untouched_path.read_bytes() == untouched_bytes
    assert (source / ".daydream" / "runs" / session_id / "children" / f"{child_id}.json").read_bytes() == child_bytes


@pytest.mark.parametrize("disposition", ["complete", "partial_evidence", "rollback"])
async def test_artifact_disposition_controls_external_trajectory_independent_of_write_status(
    tmp_path: Path,
    source: Path,
    disposition: str,
) -> None:
    requested = tmp_path / f"external-{disposition}" / "trajectory.json"
    requested.parent.mkdir()
    requested.write_bytes(b"prior full")
    requested.chmod(0o640)
    requested.with_suffix(requested.suffix + ".partial").write_bytes(b"prior partial")
    session_id = f"disposition-{disposition}"
    payload = _payload(session_id, extra={"partial": disposition == "partial_evidence"})

    async with open_artifact_session(_work(source), session_id=session_id) as session:
        route = session.register_trajectory_output(requested)
        document = TrajectoryDocumentSnapshot(session_id, requested, payload)
        session.write_trajectory_document(route, document, "complete")
        frozen = session.freeze(_snapshot(session_id, (document,)))
        _publish(session, frozen, artifact_visibility.ArtifactDisposition(disposition))

    expected = b"prior full" if disposition == "rollback" else payload
    assert requested.read_bytes() == expected
    expected_mode = 0o640 if disposition == "rollback" else 0o600
    assert stat.S_IMODE(requested.stat().st_mode) == expected_mode


async def test_findings_delivery_and_finalization_merge_are_closed_host_boundaries(
    tmp_path: Path,
    source: Path,
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


async def test_current_entry_replacement_is_transferred_and_preserved_during_detach(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_public_artifacts(source)
    target = source / ".review-output.md"
    replacement = b"unique concurrent review bytes"
    replacer = _ReplaceAtTransfer(target, replacement)

    monkeypatch.setattr(artifact_visibility, "_transfer_observer", replacer, raising=False)
    with pytest.raises(ArtifactVisibilityError, match="changed|conflict"):
        async with open_artifact_session(_work(source), session_id="current-detach"):
            pass
    assert replacer.observed is True
    assert target.read_bytes() == replacement
    owner = _owner(source)
    assert (owner.artifact_state_root / "canonical" / ".review-output.md").read_bytes() == b"review output\n"


async def test_transfer_conflict_never_replaces_a_second_concurrent_occupant(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_public_artifacts(source)
    target = source / ".review-output.md"
    first = b"first concurrent review bytes"
    second = b"second concurrent review bytes"

    def install_first(path: Path, _stage: Path) -> None:
        if path == target:
            path.unlink()
            path.write_bytes(first)

    real_os = os

    class SecondOccupantAtRestore:
        injected = False

        def __getattr__(self, name: str) -> Any:
            return getattr(real_os, name)

        def replace(self, source_path: Any, destination_path: Any) -> None:
            if Path(destination_path) == target:
                target.write_bytes(second)
                self.injected = True
            real_os.replace(source_path, destination_path)

        def link(self, source_path: Any, destination_path: Any, **kwargs: Any) -> None:
            if Path(destination_path) == target:
                target.write_bytes(second)
                self.injected = True
            real_os.link(source_path, destination_path, **kwargs)

    raced_os = SecondOccupantAtRestore()
    monkeypatch.setattr(artifact_visibility, "_transfer_observer", install_first)
    monkeypatch.setattr(artifact_visibility, "os", raced_os)
    with pytest.raises(ArtifactVisibilityError, match="changed|conflict"):
        async with open_artifact_session(_work(source), session_id="double-current-detach"):
            pass

    assert raced_os.injected is True
    assert target.read_bytes() == second
    retained = [path.read_bytes() for path in source.parent.glob(f".daydream-transfer-*/entries/{target.name}")]
    assert first in retained
    with pytest.raises(ArtifactVisibilityError, match="conflict"):
        async with open_artifact_session(_work(source), session_id="double-current-reopen"):
            pass
    assert target.read_bytes() == second
    retained_after = [path.read_bytes() for path in source.parent.glob(f".daydream-transfer-*/entries/{target.name}")]
    assert first in retained_after


@pytest.mark.parametrize("label", [OutputLabel.FINDINGS_OUTPUT, OutputLabel.DUMP_DIRECTORY])
async def test_current_entry_replacement_is_preserved_during_publication(
    source: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: OutputLabel,
) -> None:
    """A concurrent occupant of a published destination is a conflict, never overwritten."""
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
            monkeypatch.setattr(artifact_visibility, "_transfer_observer", replacer, raising=False)
            _publish(session, frozen)

    assert replacer.observed is True
    assert target.read_bytes() == replacement


# ``<action>`` to child entrypoint, with the positions of its non-Path arguments.
_HELPERS: dict[str, tuple[Callable[..., None], tuple[int, ...]]] = {
    "lock": (_lock_child, ()),
    "transaction": (_transaction_child, (3,)),
    "recover": (_recover_child, ()),
    "production-transaction": (_production_transaction_child, (3,)),
    "production-lock": (_production_lock_child, ()),
    "production-external": (_production_external_child, (3, 4)),
    "external-observe-fifo": (_external_fifo_observation_child, ()),
}


async def test_transfer_intents_are_durable_before_first_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every transfer intent is persisted before the first entry move (#1162).

    The former per-entry rewrite fsynced the whole ledger before each move;
    batching the writes must keep at least that guarantee, so when the first
    ``os.replace`` fires the stage's ``intents.json`` must already contain the
    complete, validated intent set for the detach.
    """
    source = tmp_path / "source"
    _init_repo(source)
    seeded = _seed_public_artifacts(source)

    observed_at_first_move: dict[str, Any] = {}

    def capture_first_move(path: Path, moved: Path) -> None:
        if "stage" in observed_at_first_move:
            return
        stage = next(
            (candidate for candidate in moved.parents if (candidate / "stage-owner.json").is_file()),
            None,
        )
        observed_at_first_move["stage"] = stage
        observed_at_first_move["path"] = path
        observed_at_first_move["intents"] = (
            json.loads((stage / "intents.json").read_text(encoding="utf-8"))
            if stage is not None and (stage / "intents.json").is_file()
            else None
        )

    monkeypatch.setattr(artifact_visibility, "_transfer_observer", capture_first_move)
    async with open_artifact_session(_work(source), session_id="intents-durability"):
        pass

    assert observed_at_first_move, "the detach must transfer at least one entry"
    intents = observed_at_first_move["intents"]
    assert intents is not None, "intents.json must be durable before the first move"
    assert intents["schema_version"] == artifact_visibility._SCHEMA_VERSION
    expected_files = sorted(entry.path for entry in seeded if entry.kind == "file")
    assert expected_files, "seeded public artifacts must include files"
    assert sorted(cast(str, intent["relative"]) for intent in intents["intents"]) == expected_files
    first_source = Path(cast(str, observed_at_first_move["path"]))
    assert str(first_source) in [cast(str, intent["source"]) for intent in intents["intents"]]


async def test_routed_path_accessors_resolve_explicit_session_over_fallbacks(
    tmp_path: Path,
) -> None:
    """Single-channel routing matrix for artifact_dir_for/review_output_path_for (#1162).

    Pins the #1162 seam: an explicit ``session`` wins, the bound channel
    (``_SESSION``) is honored for extension steps, the documented standalone
    legacy path remains the no-session compatibility story, and strict
    callers can fail closed with ``allow_standalone=False``.
    """
    source = tmp_path / "source"
    _init_repo(source)
    work = _work(source)

    # No session bound: documented legacy standalone path, unchanged.
    assert artifact_dir_for(work.repo) == source / ".daydream"
    assert review_output_path_for(work.repo) == source / ".review-output.md"

    async with open_artifact_session(work, session_id="routing-seam") as session:
        # Explicit session wins even though the ContextVar is also bound.
        assert artifact_dir_for(work.repo, session=session) == session.daydream_dir
        assert (
            review_output_path_for(work.repo, session=session)
            == session.review_output
        )
        # Bound channel (the documented extension contract) still routes.
        assert artifact_dir_for(work.repo) == session.daydream_dir
        assert review_output_path_for(work.repo) == session.review_output

    # Session closed: bound channel is empty again, standalone path resumes.
    assert artifact_dir_for(work.repo) == source / ".daydream"

    # Explicit fail-closed seam for strict callers (e.g. a future composition
    # root that must never silently write the public tree).
    with pytest.raises(ArtifactVisibilityError, match="standalone artifact routing"):
        artifact_dir_for(work.repo, allow_standalone=False)
    with pytest.raises(ArtifactVisibilityError, match="standalone artifact routing"):
        review_output_path_for(work.repo, allow_standalone=False)

    # A bound session satisfies allow_standalone=False (it is a real session).
    async with open_artifact_session(work, session_id="routing-strict") as session:
        assert artifact_dir_for(work.repo, allow_standalone=False) == session.daydream_dir
        assert (
            review_output_path_for(work.repo, allow_standalone=False)
            == session.review_output
        )


def _main() -> None:
    action, *arguments = sys.argv[1:]
    if action not in _HELPERS:
        raise SystemExit(f"unknown storage-spike helper action: {action}")
    handler, strings = _HELPERS[action]
    handler(*(value if index in strings else Path(value) for index, value in enumerate(arguments)))


def test_create_private_directory_rejects_symlinked_leaf(tmp_path: Path) -> None:
    """A symlinked final component fails closed at the leaf validation.

    The leaf keeps the ``validate_private_directory`` contract ("must be a
    real directory") exactly as before the consolidation; ancestors are
    covered by the restored ancestry scan.
    """
    target = tmp_path / "storage" / "runs"
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside-secret"
    outside.mkdir()
    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactVisibilityError, match="real directory"):
        artifact_visibility._create_private_directory(target)

    assert outside.is_dir()
    assert list(outside.iterdir()) == []


def test_create_private_directory_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    """A symlink anywhere in the ancestry fails closed (#1162 review finding).

    The collapsed helper must keep the ancestry lstat scan the deleted
    ``_ensure_private_directory`` enforced: creation must never mkdir through
    a symlinked ancestor or chmod through the link.
    """
    outside = tmp_path / "outside-secret"
    outside.mkdir()
    (tmp_path / "storage").mkdir()
    (tmp_path / "storage" / "live-link").symlink_to(outside, target_is_directory=True)
    # The missing-child walk stops at the first existing ancestor (the
    # symlink itself); the ancestry scan must reject it fail-closed.
    linked_target = tmp_path / "storage" / "live-link" / "runs" / "deep"

    with pytest.raises(ArtifactVisibilityError, match="symlink"):
        artifact_visibility._create_private_directory(linked_target)

    assert outside.is_dir()
    assert list(outside.iterdir()) == []
    assert not (outside / "runs").exists()


if __name__ == "__main__":
    _main()
