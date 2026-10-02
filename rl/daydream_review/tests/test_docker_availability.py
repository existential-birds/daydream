"""Probe Docker availability at the subprocess boundary without contacting a daemon or building images."""

from __future__ import annotations

import subprocess

import pytest
from conftest import docker_daemon_is_available


def _patch_subprocess_run(monkeypatch: pytest.MonkeyPatch, returncode: int = 0,) -> None:

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


@pytest.mark.parametrize(
    "returncode, expected", [(0, True), (1, False)], ids=["daemon-reachable", "daemon-unreachable"],
)
def test_docker_daemon_is_available_mirrors_docker_info_returncode(
    monkeypatch: pytest.MonkeyPatch, returncode: int, expected: bool
) -> None:

    _patch_subprocess_run(monkeypatch, returncode)
    assert docker_daemon_is_available() is expected

def test_docker_daemon_is_available_false_when_client_missing(monkeypatch: pytest.MonkeyPatch,) -> None:

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert docker_daemon_is_available() is False
