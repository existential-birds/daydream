"""Unit tests for the shared cwd-grounding instruction constant."""

from __future__ import annotations

from pathlib import Path

from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION, REPOSITORY_SEARCH_GUIDANCE, REVIEW_STOPPING_GUIDANCE


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

def test_review_stopping_guidance_bounds_candidate_work_and_environment_setup() -> None:
    assert "one pass" in REVIEW_STOPPING_GUIDANCE
    assert "Do not install" in REVIEW_STOPPING_GUIDANCE
    assert "missing dependencies" in REVIEW_STOPPING_GUIDANCE


def test_search_guidance_preserves_evidence_and_bounds_fallback() -> None:
    # Shipped prompt contract; behavioral semantics belong to repository_search tests.
    assert "tracked dotfiles" in REPOSITORY_SEARCH_GUIDANCE
    assert "nonignored untracked" in REPOSITORY_SEARCH_GUIDANCE
    assert "fallback when rg is unavailable" in REPOSITORY_SEARCH_GUIDANCE
    assert "--expand named/path --reason" in REPOSITORY_SEARCH_GUIDANCE
    assert "including grep, are allowed" in REPOSITORY_SEARCH_GUIDANCE
