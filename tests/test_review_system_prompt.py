"""Tests for cwd-grounding in the review system prompt builders."""

from __future__ import annotations

from pathlib import Path

from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION
from daydream.prompts.review_system_prompt import (
    CodebaseMetadata,
    build_pr_review_prompt,
    build_review_system_prompt,
)


def _metadata() -> CodebaseMetadata:
    return CodebaseMetadata(
        file_count=10,
        total_tokens=1000,
        languages=["Python"],
        largest_files=[("daydream/runner.py", 500)],
        changed_files=["daydream/runner.py"],
    )


def test_review_prompt_contains_cwd_grounding() -> None:
    cwd = Path("/tmp/linked/worktree")
    prompt = build_review_system_prompt(_metadata(), cwd=cwd)
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=cwd) in prompt
    assert str(cwd) in prompt


def test_pr_review_prompt_contains_cwd_grounding() -> None:
    cwd = Path("/tmp/linked/worktree")
    prompt = build_pr_review_prompt(
        _metadata(),
        pr_title="Add taste parser",
        pr_description="Implements the parser",
        cwd=cwd,
    )
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=cwd) in prompt
    assert str(cwd) in prompt
