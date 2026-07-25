"""Real-path test for cwd grounding in a linked git worktree (issue #221).

Drives the real ``pre_scan`` exploration pipeline against an actual linked git
worktree whose sibling main worktree contains different content at the same
relative paths. Only the backend is mocked. Asserts that the specialist prompts
carry cwd-absolute paths under the LINKED worktree (never the main worktree) and
the cwd-grounding instruction — the deterministic contract the fix locks.
"""

from __future__ import annotations

import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import anyio

from daydream.backends import AgentEvent, ResultEvent
from daydream.exploration_runner import pre_scan
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
)
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION


class _PromptCapturingBackend:
    """Backend stub: captures every prompt and returns schema-shaped output."""

    model = "mock-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def execute(  # type: ignore[no-untyped-def]
        self,
        cwd,
        prompt,
        output_schema=None,
        continuation=None,
        agents=None,
        max_turns=None,
        read_only=False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.prompts.append(prompt)
        result: dict[str, Any] = {}
        if output_schema == PATTERN_SCANNER_SCHEMA:
            result = {"conventions": [], "guidelines": []}
        elif output_schema == DEPENDENCY_TRACER_SCHEMA:
            result = {"affected_files": [], "dependencies": []}
        elif output_schema == TEST_MAPPER_SCHEMA:
            result = {"affected_files": []}
        yield ResultEvent(structured_output=result, continuation=None)

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


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

    backend = _PromptCapturingBackend()

    async def run_pre_scan() -> None:
        await pre_scan(backend, linked, diff_text)

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
