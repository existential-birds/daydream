"""Regression tests for the real-push/no-CI fixture handshake."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from tests.harness.remote_ci import _wait_for_pushed_sha


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
        [
            "/bin/sh",
            "-c",
            (
                'exec 3>"$1"\n'
                'touch "$2"\n'
                'while [ ! -f "$3" ]; do sleep 0.01; done\n'
                'printf "%s\\n" "$4" >&3\n'
            ),
            "delayed-writer",
            str(sha_path),
            str(opened_path),
            str(release_path),
            expected_sha,
        ]
    )
    try:
        _wait_until_exists(opened_path)
        assert sha_path.exists()
        assert sha_path.read_bytes() == b""

        observed: list[str | None] = []
        reader = threading.Thread(
            target=lambda: observed.append(_wait_for_pushed_sha(sha_path, stop)),
            daemon=True,
        )
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
