"""Tests for daydream.clipboard.copy_to_clipboard detection + fallback."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream import clipboard
from daydream.clipboard import copy_to_clipboard
from tests.harness.table_cases import isolated_case


def test_copy_to_clipboard_success_and_selection(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    """Select the first available clipboard command and pass text on standard input."""
    table_root = tmp_path
    row_index = 0
    cases: list[tuple[set[str], str, list[str]]] = [
        ({"pbcopy"}, "hello", ["pbcopy"]),
        ({"pbcopy", "xclip", "xsel", "clip.exe"}, "x", ["pbcopy"]),
        ({"xclip", "xsel", "clip.exe"}, "x", ["xclip", "-selection", "clipboard"]),
        ({"xsel", "clip.exe"}, "x", ["xsel", "--clipboard", "--input"]),
        ({"clip.exe"}, "x", ["clip.exe"]),
    ]
    for available, text, expected_argv in cases:
        row_index += 1
        with subtests.test(text=text):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                monkeypatch = case.monkeypatch
                detected = {cmd: f"/usr/bin/{cmd}" for cmd in available}
                monkeypatch.setattr(shutil, "which", detected.get)
                captured: dict[str, Any] = {}

                def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
                    captured["argv"] = argv
                    captured["input"] = kwargs.get("input")
                    captured["timeout"] = kwargs.get("timeout")
                    return subprocess.CompletedProcess(argv, returncode=0)

                monkeypatch.setattr(subprocess, "run", fake_run)

                assert copy_to_clipboard(text) is True
                assert captured["argv"] == expected_argv
                assert captured["input"] == text
                assert captured["timeout"] == clipboard._CLIPBOARD_TIMEOUT_SECONDS

def test_copy_to_clipboard_no_mechanism(monkeypatch: pytest.MonkeyPatch) -> None:
    """No clipboard tool on PATH → returns False without invoking run."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    invoked = False
    def fake_run(*args: Any, **kwargs: Any) -> None:
        nonlocal invoked
        invoked = True
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert copy_to_clipboard("hello") is False
    assert invoked is False

def test_copy_to_clipboard_subprocess_failure(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    """Return failure when the selected clipboard command cannot complete."""
    table_root = tmp_path
    row_index = 0
    cases: list[tuple[str, BaseException]] = [
        ("pbcopy", subprocess.CalledProcessError(returncode=1, cmd=["pbcopy"])),
        ("xclip", OSError("exec failed")),
        ("pbcopy", subprocess.TimeoutExpired(cmd=["pbcopy"], timeout=clipboard._CLIPBOARD_TIMEOUT_SECONDS)),
    ]
    for tool, error in cases:
        row_index += 1
        with subtests.test(tool=tool):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                monkeypatch = case.monkeypatch
                monkeypatch.setattr(shutil, "which", {tool: f"/usr/bin/{tool}"}.get)

                def fake_run(argv: list[str], **kwargs: Any) -> None:
                    """Raise the parameterized failure at the subprocess boundary."""
                    raise error

                monkeypatch.setattr(subprocess, "run", fake_run)
                assert copy_to_clipboard("hello") is False
