"""Deterministic unit coverage for docker_daemon_is_available().

These tests replace the subprocess boundary with monkeypatch, so they run in a
Docker-less environment and never construct images or contact a daemon.
"""

from __future__ import annotations

import subprocess

import pytest
from conftest import docker_daemon_is_available


@pytest.mark.parametrize(
    "returncode, expected",
    [(0, True), (1, False)],
    ids=["daemon-reachable", "daemon-unreachable"],
)
def test_docker_daemon_is_available_mirrors_docker_info_returncode(
    monkeypatch: pytest.MonkeyPatch, returncode: int, expected: bool
) -> None:
    """A return code of 0 means the client reached its daemon; anything else is unavailable."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert docker_daemon_is_available() is expected


def test_docker_daemon_is_available_false_when_client_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client that fails to launch (FileNotFoundError) is treated as unavailable."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert docker_daemon_is_available() is False