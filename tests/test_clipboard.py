# tests/test_clipboard.py
"""Tests for daydream.clipboard.copy_to_clipboard detection + fallback."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from daydream import clipboard
from daydream.clipboard import copy_to_clipboard


@pytest.mark.parametrize(
    ("available", "text", "expected_argv"),
    [
        pytest.param({"pbcopy"}, "hello", ["pbcopy"], id="pbcopy-success"),
        pytest.param(
            {"pbcopy", "xclip", "xsel", "clip.exe"},
            "x",
            ["pbcopy"],
            id="pbcopy-first",
        ),
        pytest.param(
            {"xclip", "xsel", "clip.exe"},
            "x",
            ["xclip", "-selection", "clipboard"],
            id="xclip-before-xsel",
        ),
        pytest.param(
            {"xsel", "clip.exe"},
            "x",
            ["xsel", "--clipboard", "--input"],
            id="xsel-before-clip-exe",
        ),
        pytest.param({"clip.exe"}, "x", ["clip.exe"], id="clip-exe-fallback"),
    ],
)
def test_copy_to_clipboard_success_and_selection(
    monkeypatch: pytest.MonkeyPatch,
    available: set[str],
    text: str,
    expected_argv: list[str],
) -> None:
    detected = {cmd: f"/usr/bin/{cmd}" for cmd in available}
    monkeypatch.setattr(clipboard.shutil, "which", detected.get)
    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert copy_to_clipboard(text) is True
    assert captured["argv"] == expected_argv
    assert captured["input"] == text


def test_copy_to_clipboard_no_mechanism(monkeypatch: pytest.MonkeyPatch) -> None:
    """No clipboard tool on PATH → returns False without invoking run."""
    monkeypatch.setattr(clipboard.shutil, "which", lambda cmd: None)

    invoked = False

    def fake_run(*args: Any, **kwargs: Any) -> None:
        nonlocal invoked
        invoked = True

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert copy_to_clipboard("hello") is False
    assert invoked is False


@pytest.mark.parametrize(
    ("tool", "error"),
    [
        pytest.param(
            "pbcopy",
            subprocess.CalledProcessError(returncode=1, cmd=["pbcopy"]),
            id="called-process-error",
        ),
        pytest.param("xclip", OSError("exec failed"), id="os-error"),
    ],
)
def test_copy_to_clipboard_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    error: BaseException,
) -> None:
    monkeypatch.setattr(clipboard.shutil, "which", {tool: f"/usr/bin/{tool}"}.get)

    def fake_run(argv: list[str], **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert copy_to_clipboard("hello") is False
