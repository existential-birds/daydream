# tests/test_clipboard.py
"""Tests for daydream.clipboard.copy_to_clipboard detection + fallback."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from daydream import clipboard
from daydream.clipboard import copy_to_clipboard


def test_copy_to_clipboard_pbcopy_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """pbcopy on PATH → run is invoked and True returned."""
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda cmd: "/usr/bin/pbcopy" if cmd == "pbcopy" else None,
    )

    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert copy_to_clipboard("hello") is True
    assert captured["argv"] == ["pbcopy"]
    assert captured["input"] == "hello"


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


def test_copy_to_clipboard_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subprocess raising CalledProcessError → returns False, swallows error."""
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda cmd: "/usr/bin/pbcopy" if cmd == "pbcopy" else None,
    )

    def fake_run(argv: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=argv)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert copy_to_clipboard("hello") is False


def test_copy_to_clipboard_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subprocess raising OSError (e.g. exec failure) → returns False."""
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None,
    )

    def fake_run(argv: list[str], **kwargs: Any) -> None:
        raise OSError("exec failed")

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert copy_to_clipboard("hello") is False


def test_copy_to_clipboard_detection_order_pbcopy_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pbcopy wins over xclip/xsel/clip.exe when all are present."""
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda cmd: f"/usr/bin/{cmd}",  # every tool "exists"
    )

    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert copy_to_clipboard("x") is True
    assert captured["argv"][0] == "pbcopy"


def test_copy_to_clipboard_detection_order_xclip_before_xsel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pbcopy missing, xclip present → xclip is used (before xsel)."""

    def which(cmd: str) -> str | None:
        if cmd == "pbcopy":
            return None
        return f"/usr/bin/{cmd}"

    monkeypatch.setattr(clipboard.shutil, "which", which)

    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert copy_to_clipboard("x") is True
    assert captured["argv"] == ["xclip", "-selection", "clipboard"]


def test_copy_to_clipboard_detection_order_xsel_before_clip_exe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only xsel and clip.exe present → xsel is chosen first."""

    def which(cmd: str) -> str | None:
        if cmd in ("xsel", "clip.exe"):
            return f"/usr/bin/{cmd}"
        return None

    monkeypatch.setattr(clipboard.shutil, "which", which)

    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert copy_to_clipboard("x") is True
    assert captured["argv"] == ["xsel", "--clipboard", "--input"]


def test_copy_to_clipboard_clip_exe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only clip.exe present → it is used."""

    def which(cmd: str) -> str | None:
        return "/usr/bin/clip.exe" if cmd == "clip.exe" else None

    monkeypatch.setattr(clipboard.shutil, "which", which)

    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    assert copy_to_clipboard("x") is True
    assert captured["argv"] == ["clip.exe"]
