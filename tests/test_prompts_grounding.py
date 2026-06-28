"""Unit tests for the shared cwd-grounding instruction constant."""

from __future__ import annotations

from pathlib import Path

from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION


def test_grounding_instruction_contains_cwd_warning() -> None:
    text = CWD_GROUNDING_INSTRUCTION
    assert "git worktree" in text
    assert "git rev-parse --show-toplevel" in text
    assert "git rev-parse --git-common-dir" in text
    assert "git worktree list" in text


def test_grounding_instruction_formats_cwd() -> None:
    cwd = Path("/tmp/some/linked/worktree")
    rendered = CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)
    assert str(cwd) in rendered
