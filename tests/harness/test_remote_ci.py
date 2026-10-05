"""Regression tests for the real-push/no-CI fixture handshake."""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from pathlib import Path

import pytest

from tests.harness.fake_gh import FakeGh
from tests.harness.remote_ci import (
    _wait_for_pushed_sha,
    finish_remote_ci_fake,
    start_remote_ci_fake,
)
from tests.test_integration import _wait_for_remote_ci_pids


def _wait_until_exists(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path.name}")
        time.sleep(0.01)

def test_pushed_sha_reader_ignores_existing_empty_marker(tmp_path: Path) -> None:
    """Shell redirection publishes an empty file before its delayed write."""
    sha_path = tmp_path / "push sha.txt"
    opened_path = tmp_path / "opened"
    release_path = tmp_path / "release"
    expected_sha = "a" * 40
    stop = threading.Event()
    reader: threading.Thread | None = None
    writer = subprocess.Popen(  # noqa: S603 - fixed test shell and arguments
        ["/bin/sh", "-c",
            ('exec 3>"$1"\n' 'touch "$2"\n' 'while [ ! -f "$3" ]; do sleep 0.01; done\n' 'printf "%s\\n" "$4" >&3\n'),
            "delayed-writer", str(sha_path), str(opened_path), str(release_path), expected_sha,
        ]
    )
    try:
        _wait_until_exists(opened_path)
        assert sha_path.exists()
        assert sha_path.read_bytes() == b""

        observed: list[str | None] = []
        reader = threading.Thread(target=lambda: observed.append(_wait_for_pushed_sha(sha_path, stop)), daemon=True,)
        reader.start()
        time.sleep(0.05)
        assert reader.is_alive(), "an empty-but-existing marker must not be ready"

        release_path.write_text("go\n")
        reader.join(timeout=5)
        assert not reader.is_alive()
        assert observed == [expected_sha]
    finally:
        stop.set()
        release_path.touch()
        writer.wait(timeout=5)
        if reader is not None:
            reader.join(timeout=5)

def test_pushed_sha_reader_stops_without_a_ready_marker(tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()

    assert _wait_for_pushed_sha(tmp_path / "missing.sha", stop) is None

def test_integration_seeder_waits_for_complete_sha_before_publishing(tmp_path: Path, fake_gh: FakeGh,) -> None:
    """A partial final-path SHA must not seed endpoints or release the hook."""
    project = tmp_path / "project"
    (project / ".git" / "hooks").mkdir(parents=True)
    hook_marker = tmp_path / "pre push hook ran"
    sha_path = hook_marker.with_name(hook_marker.name + " sha")
    ready_path = hook_marker.with_name(hook_marker.name + " ready")
    opened_path = tmp_path / "partial opened"
    release_path = tmp_path / "release remainder"
    expected_sha = "a" * 40
    seeder = start_remote_ci_fake(project, fake_gh, hook_marker, outcome="no_ci")
    writer = subprocess.Popen(  # noqa: S603 - fixed test shell and arguments
        ["/bin/sh", "-c",
            (
                'exec 3>"$1"\n'
                'printf "%s" "$4" >&3\n'
                'touch "$2"\n'
                'while [ ! -f "$3" ]; do sleep 0.01; done\n'
                'printf "%s\\n" "$5" >&3\n'
            ), "partial-sha-writer", str(sha_path), str(opened_path), str(release_path), expected_sha[:20],
            expected_sha[20:],
        ]
    )
    try:
        _wait_until_exists(opened_path)
        assert sha_path.read_text() == expected_sha[:20]
        deadline = time.monotonic() + 1
        while seeder.thread.is_alive() and not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert seeder.thread.is_alive(), "partial SHA incorrectly completed the seeder"
        assert not ready_path.exists(), "partial SHA incorrectly released the pre-push hook"

        release_path.write_text("go\n")
        seeder.thread.join(timeout=5)
        assert not seeder.thread.is_alive()
        assert seeder.errors == []
        assert ready_path.read_text() == "ready\n"
        responses = fake_gh._read_responses()
        assert any(expected_sha in key for key in responses)
        assert not any(expected_sha[:20] in key and expected_sha not in key for key in responses)
    finally:
        seeder.stop.set()
        release_path.touch()
        writer.wait(timeout=5)
        seeder.thread.join(timeout=5)

def test_integration_seeder_stops_cleanly_when_push_never_starts(tmp_path: Path, fake_gh: FakeGh,) -> None:
    project = tmp_path / "project"
    (project / ".git" / "hooks").mkdir(parents=True)
    hook_marker = tmp_path / "unused hook marker"
    seeder = start_remote_ci_fake(project, fake_gh, hook_marker, outcome="no_ci")
    finish_remote_ci_fake(seeder)

    assert not hook_marker.with_name(hook_marker.name + " ready").exists()

def test_integration_seeder_joins_and_surfaces_external_fake_failure(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / ".git" / "hooks").mkdir(parents=True)
    hook_marker = tmp_path / "failed hook marker"
    def fail_to_seed(*args: object, **kwargs: object) -> None:
        raise RuntimeError("external fake rejected seed")
    monkeypatch.setattr(fake_gh, "set_response", fail_to_seed)
    seeder = start_remote_ci_fake(project, fake_gh, hook_marker, outcome="no_ci")
    hook_marker.with_name(hook_marker.name + " sha").write_text("b" * 40 + "\n")
    ready_path = hook_marker.with_name(hook_marker.name + " ready")
    _wait_until_exists(ready_path)
    with pytest.raises(AssertionError):
        finish_remote_ci_fake(seeder)

    assert not seeder.thread.is_alive()
    assert ready_path.read_text() == "failed\n"
    assert len(seeder.errors) == 1
    assert isinstance(seeder.errors[0], RuntimeError)

@pytest.mark.asyncio
async def test_pid_wait_fails_when_runner_ends_before_push(tmp_path: Path) -> None:
    async def already_done() -> int:
        return 7
    runner_task = asyncio.create_task(already_done())
    await runner_task
    with pytest.raises(AssertionError, match="runner exited 7 before reaching the remote-CI push boundary",):
        await _wait_for_remote_ci_pids(
            tmp_path / "missing pids", sha_path=tmp_path / "missing sha", runner_task=runner_task,
        )
