"""Real-path test for cwd grounding in a linked git worktree (issue #221).

Drives the real ``pre_scan`` exploration pipeline against an actual linked git
worktree whose sibling main worktree contains different content at the same
relative paths. Only the backend is mocked. Asserts that the specialist prompts
carry cwd-absolute paths under the LINKED worktree (never the main worktree) and
the cwd-grounding instruction — the deterministic contract the fix locks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import anyio

from daydream.backends import Backend, ResultEvent
from daydream.exploration_runner import pre_scan
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
)
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION
from tests.harness.backend import ScriptedBackend


def test_pre_scan_grounds_specialists_to_linked_worktree(linked_worktree: tuple[Path, Path]) -> None:
    main_repo, linked = linked_worktree

    # Sanity: the trap is real — services/taste/ exists only in the linked worktree.
    assert (linked / "services" / "taste" / "parser.go").exists()
    assert not (main_repo / "services" / "taste").exists()

    diff_text = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["git", "diff", "main...HEAD"],  # noqa: S607 - git is a trusted command
        cwd=linked,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # 4 changed files => parallel tier => all three specialists run.
    assert "services/taste/parser.go" in diff_text

    backend = ScriptedBackend(
        responses_by_schema=[
            (
                PATTERN_SCANNER_SCHEMA,
                [
                    ResultEvent(
                        structured_output={"conventions": [], "guidelines": []}, continuation=None
                    )
                ],
            ),
            (
                DEPENDENCY_TRACER_SCHEMA,
                [
                    ResultEvent(
                        structured_output={"affected_files": [], "dependencies": []}, continuation=None
                    )
                ],
            ),
            (
                TEST_MAPPER_SCHEMA,
                [ResultEvent(structured_output={"affected_files": []}, continuation=None)],
            ),
        ]
    )

    async def run_pre_scan() -> None:
        await pre_scan(cast(Backend, backend), linked, diff_text)

    anyio.run(run_pre_scan)

    assert backend.prompts, "expected specialist prompts to be captured"
    joined = "\n".join(backend.prompts)

    # Every specialist prompt carries the cwd-grounding instruction rooted at the
    # linked worktree.
    for prompt in backend.prompts:
        assert CWD_GROUNDING_INSTRUCTION.format(cwd=linked) in prompt

    # Specialist file lists are cwd-absolute under the LINKED worktree.
    assert str(linked / "services" / "taste" / "parser.go") in joined

    # The main worktree path never leaks into any prompt (its sibling, where
    # services/taste/ does not exist).
    assert str(main_repo) not in joined
