"""Tests for daydream.clipboard.copy_to_clipboard detection + fallback."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import pytest

from daydream import clipboard
from daydream.clipboard import copy_to_clipboard


def test_copy_to_clipboard_selects_first_tool_and_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the first available command with text on stdin; a missing or failing tool returns False."""
    monkeypatch.setattr(shutil, "which", {cmd: f"/usr/bin/{cmd}" for cmd in ("xsel", "clip.exe")}.get)
    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.update(argv=argv, input=kwargs.get("input"), timeout=kwargs.get("timeout"))
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert copy_to_clipboard("x") is True
    assert captured == {"argv": ["xsel", "--clipboard", "--input"], "input": "x",
                        "timeout": clipboard._CLIPBOARD_TIMEOUT_SECONDS}

    captured.clear()
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert copy_to_clipboard("hello") is False
    assert captured == {}, "no clipboard tool on PATH must not invoke run"

    def failing_run(argv: list[str], **kwargs: Any) -> None:
        raise OSError("exec failed")

    monkeypatch.setattr(shutil, "which", {"xclip": "/usr/bin/xclip"}.get)
    monkeypatch.setattr(subprocess, "run", failing_run)
    assert copy_to_clipboard("hello") is False
