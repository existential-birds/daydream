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
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pytest

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


def _transition(journal: Path, state: str, kill_at: str, marker: Path) -> None:
    _atomic_json(journal, {"schema_version": 1, "state": state})
    if state != kill_at:
        return
    marker.write_text(state, encoding="ascii")
    _fsync_file(marker)
    _fsync_dir(marker.parent)
    while True:
        signal.pause()


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


def _projection_matches(root: Path, entries: tuple[_Entry, ...]) -> bool:
    try:
        return _manifest(root) == entries
    except FileNotFoundError:
        return False


def _paths_overlap(left: Path, right: Path) -> bool:
    first = left.resolve(strict=False)
    second = right.resolve(strict=False)
    return first == second or first in second.parents or second in first.parents


def test_lock_process_rejects_second_process_and_releases_on_death(tmp_path: Path) -> None:
    lock_path = tmp_path / "workspace.lock"
    marker = tmp_path / "lock-ready"
    process = _spawn_helper("lock", str(lock_path), str(marker))
    try:
        _wait_for_marker(process, marker)
        contender = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)

        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
        released = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(released, fcntl.LOCK_UN)
        finally:
            os.close(released)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_recovery_spike_rejects_symlink_without_following(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _init_repo(source)
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
    transition: str,
) -> None:
    source = tmp_path / "source"
    _init_repo(source)
    expected = _seed_public_artifacts(source)
    runtime = tmp_path / "operator-runtime"
    first_repo = tmp_path / "ephemeral-one"
    second_repo = tmp_path / "ephemeral-two"
    _git(source, "worktree", "add", "--detach", str(first_repo), "HEAD")
    _git(source, "worktree", "add", "--detach", str(second_repo), "HEAD")
    common_dir = _git_common_dir(source)
    assert _git_common_dir(first_repo) == common_dir
    assert _git_common_dir(second_repo) == common_dir
    workspace_key = _workspace_key(source)
    marker = tmp_path / f"reached-{transition}"

    process = _spawn_helper(
        "transaction",
        str(source),
        str(first_repo),
        str(runtime),
        transition,
        str(marker),
    )
    try:
        _wait_for_marker(process, marker)
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    state = runtime / workspace_key
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


def test_runtime_ancestry_is_disjoint_from_real_repository_cwd(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    _init_repo(repo)
    runtime = Path.home() / ".daydream" / "runtime"

    assert not _paths_overlap(runtime, repo)
    assert _paths_overlap(repo, repo / ".daydream" / "runtime")
    assert _paths_overlap(repo.parent, repo)


def _main() -> None:
    action, *arguments = sys.argv[1:]
    if action == "lock":
        lock_path, marker = map(Path, arguments)
        _lock_child(lock_path, marker)
        return
    if action == "transaction":
        source_arg, repo_arg, runtime_arg, transition, marker_arg = arguments
        _transaction_child(
            Path(source_arg), Path(repo_arg), Path(runtime_arg), transition, Path(marker_arg)
        )
        return
    if action == "recover":
        source_path, repo_path, runtime_path = map(Path, arguments)
        _recover_child(source_path, repo_path, runtime_path)
        return
    raise SystemExit(f"unknown storage-spike helper action: {action}")


if __name__ == "__main__":
    _main()
