"""Ground pre_scan in a linked worktree whose sibling has different same-path content.

Only the backend is doubled. Specialist prompts must name the linked worktree's
absolute paths and include the cwd-grounding instruction.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.exploration_runner import pre_scan
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
)
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION, REPOSITORY_SEARCH_GUIDANCE
from tests.harness.backend import ScriptedBackend


def test_pre_scan_grounds_specialists_to_linked_worktree(linked_worktree: tuple[Path, Path]) -> None:
    main_repo, linked = linked_worktree

    # Sanity: the trap is real — services/taste/ exists only in the linked worktree.
    assert (linked / "services" / "taste" / "parser.go").exists()
    assert not (main_repo / "services" / "taste").exists()

    diff_text = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["git", "diff", "main...HEAD"],  # noqa: S607 - git is a trusted command
        cwd=linked, capture_output=True, text=True, check=True,
    ).stdout
    # 4 changed files => parallel tier => all three specialists run.
    assert "services/taste/parser.go" in diff_text

    backend = ScriptedBackend(responses_by_schema=[(PATTERN_SCANNER_SCHEMA,
                [ResultEvent(structured_output={"conventions": [], "guidelines": []}, continuation=None)],
            ), (DEPENDENCY_TRACER_SCHEMA,
                [ResultEvent(structured_output={"affected_files": [], "dependencies": []}, continuation=None)],
            ), (TEST_MAPPER_SCHEMA, [ResultEvent(structured_output={"affected_files": []}, continuation=None)],),
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
        assert REPOSITORY_SEARCH_GUIDANCE in prompt

    # Specialist file lists are cwd-absolute under the LINKED worktree.
    assert str(linked / "services" / "taste" / "parser.go") in joined

    # The main worktree path never leaks into any prompt (its sibling, where
    # services/taste/ does not exist).
    assert str(main_repo) not in joined


async def test_runner_search_cli_keeps_linked_scope_and_custom_strategy(
    linked_worktree: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner-owned prompt delivery + installed CLI/file set, only provider doubled.

    Existing pre_scan coverage cannot detect a helper that reroots via shared Git
    metadata, loses tracked ignored evidence, or lacks the production CLI dispatch.
    """
    from dataclasses import replace

    from daydream import runner
    from daydream.artifact_visibility import private_root_locations
    from daydream.review_profile import ResolvedProfile, build_default_profile
    from tests.deep_orchestrator.empty_synthesis_support import empty_review_config
    from tests.harness.git_helpers import git
    from tests.harness.stub_backend import review_stage_result

    main, linked = linked_worktree
    (linked / ".gitignore").write_text("deps/\nbuild/\n")
    files = {
        "services/taste/untracked.go": "// search-token untracked\n",
        ".rules/policy": "search-token linked dotfile\n",
        "deps/tracked.go": "// search-token tracked ignored\n",
        "deps/pkg/generated.go": "// search-token generated\n",
        "build/noise": "search-token misleading\n",
    }
    for name, value in files.items():
        target = linked / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    git(linked, "add", ".rules")
    git(linked, "add", "-f", "deps/tracked.go")
    (main / ".rules").mkdir()
    (main / ".rules/policy").write_text("search-token WRONG SIBLING\n")
    (linked / "services/taste/outside").symlink_to(main / ".rules/policy")
    profile = build_default_profile()
    strategies = dict(profile.strategies)
    strategies["discovery.per_stack"] = replace(
        strategies["discovery.per_stack"],
        content="CUSTOM_SEARCH_STRATEGY " + strategies["discovery.per_stack"].content,
    )
    profile = replace(profile, strategies=strategies)
    observed: list[dict[str, Any]] = []
    def respond(cwd: Path, prompt: str, schema: Any, *args: Any) -> list[Any]:
        if "CUSTOM_SEARCH_STRATEGY" in prompt and not observed:
            assert REPOSITORY_SEARCH_GUIDANCE in prompt
            # Remove rg entirely while retaining the real Git executable.
            bin_dir = tmp_path / "git-only-bin"
            bin_dir.mkdir()
            import shutil
            git_path = shutil.which("git")
            assert git_path is not None
            (bin_dir / "git").symlink_to(git_path)
            env = dict(os.environ, PATH=str(bin_dir))
            command = [sys.executable, "-m", "daydream", "search", "search-token", "services/taste", ".rules", "deps"]
            for suffix in ([], ["--expand", "deps/pkg", "--reason", "trace generated caller"]):
                completed = subprocess.run(
                    command + suffix, cwd=cwd, env=env, capture_output=True, text=True, timeout=30,
                )
                assert completed.returncode == 2  # outward symlink is disclosed as skipped
                observed.append(json.loads(completed.stdout))
        return [TextEvent(text="Review complete."),
            ResultEvent(structured_output=review_stage_result(prompt, []) if schema else None, continuation=None)]
    backend = ScriptedBackend(responder=respond)
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    trajectory = tmp_path / "trajectory.json"
    config = empty_review_config(linked, trajectory, shallow=True, stack="go",
        review_profile=ResolvedProfile(profile=profile, source_kind="test"))
    assert await runner.run(config, private_roots=private_root_locations(base=tmp_path / "private")) == 0
    assert len(observed) == 2
    normal, expanded = observed
    expected = {"services/taste/untracked.go", ".rules/policy", "deps/tracked.go"}
    assert {hit["path"] for hit in normal["matches"]} == expected
    assert {hit["path"] for hit in expanded["matches"]} == expected | {"deps/pkg/generated.go"}
    assert normal["skipped_files"] == 1
    assert all("WRONG SIBLING" not in hit["text"] for result in observed for hit in result["matches"])
    assert expanded["expansion"]["reason"] == "trace generated caller"
    assert trajectory.is_file()
    assert json.loads(trajectory.read_text())["extra"]["profile_digest"] == profile.digest
    assert not (main / ".review-output.md").exists()
