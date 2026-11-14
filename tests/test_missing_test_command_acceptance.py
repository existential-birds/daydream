"""Issue #1491: command readiness gates fixes before any suite or repair runs.

Runner journeys resolve and persist their own recipes. The supplementary phase
rows exercise genuine recipe reload, because runner intentionally resolves anew.
Only the external backend is replaced in the runner regressions.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.config_file import load_file_config
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.phases import phase_test_and_heal
from daydream.run_context import InteractionPolicy, RunContext
from daydream.runner import run
from daydream.test_execution import (
    ResolvedFact,
    load_test_recipe,
    persist_test_recipe,
    recipe_to_payload,
    resolve_test_recipe,
)
from tests.deep_orchestrator.support import _scan_phase_events, _scan_trajectory_extra
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import commit, git, work_context
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import MakeConfig, _fix_prompts


def _arrange(target: Path, monkeypatch: pytest.MonkeyPatch) -> StubBackend:
    (target / "api.py").write_text("VALUE = 2\n")
    (target / "requirements.txt").write_text("pytest\n")
    git(target, "add", "api.py", "requirements.txt")
    commit(target, "test: actionable Python change and manifest candidate")
    backend = StubBackend(target)
    backend.review_claims = [{"title": "Sample issue", "file": "api.py", "line": 1}]
    backend.fix_edit_line = "# unauthorized fix turn ran\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    return backend


def _state(target: Path) -> tuple[str, str, dict[str, bytes]]:
    tracked = git(target, "ls-files").splitlines()
    paths = [*tracked, "owner-untracked.txt"]
    return git(target, "rev-parse", "HEAD"), git(target, "diff", "--cached", "--binary"), {
        name: (target / name).read_bytes() for name in paths
    }


async def test_runner_rejects_missing_canonical_before_fix_or_test(
    linked_worktree: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """An empty-argv file command enters the real TOML loader and stops before any fix or suite.

    The rejection is persisted as a no-execution verdict and recipe; the manifest
    candidate is never executed. Command-miss reasons have owner-level proof.
    """
    reason, source = "empty-argv", "config"
    _, target = linked_worktree
    backend = _arrange(target, monkeypatch)
    (target / ".daydream.toml").write_text('test_command = \'""\'\n')
    (target / "base.txt").write_text("owner dirty bytes\n")
    (target / "owner-untracked.txt").write_text("owner untracked bytes\n")
    before = _state(target)
    counter = tmp_path / "candidate-runs"
    candidate = tmp_path / "pytest"
    candidate.write_text(f"#!/bin/sh\nprintf ran >> '{counter}'\nexit 0\n")
    candidate.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    trajectory = tmp_path / "trajectory.json"
    config = make_config(target, output_mode="loop", assume="yes", test_command=None,
                         file_config=load_file_config(target), trajectory_path=trajectory)
    rc = await run(config)
    assert rc != 0, "unconfigured validation cannot authorize a successful fix flow"
    assert _state(target) == before, "configuration rejection changed owner source/index/HEAD"
    assert not counter.exists(), "the manifest candidate was executed without admission"
    assert backend.test_suite_calls == 0
    assert not _fix_prompts(backend)
    assert not any(call["prompt"].startswith("The tests failed") for call in backend.calls)
    phases = _scan_trajectory_extra(target / ".daydream", trajectory, "daydream_phase")
    assert not {"test", "fix"}.intersection(phases)
    starts = _scan_phase_events(target / ".daydream", trajectory, "phase_start")
    assert not {"test", "test-execution", "fix"}.intersection(event["phase"] for event in starts)
    deep = target / ".daydream" / "deep"
    verdict = json.loads((deep / "test-verdict.json").read_text())
    assert verdict["status"] == "no-execution"
    assert verdict["passed"] is False and verdict["retries"] == 0 and verdict["ignored"] is False
    diagnostic: dict[str, Any] = verdict["diagnostic"]
    assert diagnostic["reason_code"] == reason and diagnostic["source"] == source
    assert "--test-command" in diagnostic["message"]
    assert "[tool.daydream]" in diagnostic["message"]
    assert verdict["repairs"] == []
    assert verdict["attempts"][0]["kind"] == "no-execution"
    loaded = load_test_recipe(deep)
    assert loaded is not None and not loaded.command.resolved
    assert loaded.command.diagnostic is not None
    assert loaded.command.diagnostic.reason_code == reason
    assert loaded.command.diagnostic.source == source
    assert loaded.candidate is not None and loaded.candidate.argv == ("pytest",)
    assert not (deep / "repair-job.json").exists()
    assert not (deep / "repair-checkpoint.json").exists()


async def test_public_test_phase_returns_no_execution_without_repair(
    linked_worktree: tuple[Path, Path], tmp_path: Path, make_config: MakeConfig,
) -> None:
    """Public phase covers direct config and loaded current/legacy/derived/stale recipes.

    Legacy/stale cases model actual untrusted historical bytes at the loader;
    these supplement the genuine producer runner journey above.
    """
    _, target = linked_worktree
    config = make_config(target, assume="yes", test_command="python 'private-token")
    before = git(target, "rev-parse", "HEAD"), git(target, "diff", "--binary"), git(target, "diff", "--cached")

    def capture() -> str:
        return git_ops.tree_key(git_ops.snapshot_worktree_delta(
            target, "HEAD", preexisting_untracked={}, preexisting_gitlinks=(),
        ))

    for recipe_state, reason, source in (("direct", "malformed", "cli"), ("loaded", "malformed", "cli"),
                                         ("legacy", "unresolved", "unresolved"),
                                         ("derived", "unresolved", "derived"), ("stale", "malformed", "cli")):
        recipe = None
        if recipe_state != "direct":
            persisted = tmp_path / recipe_state
            resolved = resolve_test_recipe(config.file_config, config, repo_root=target)
            path = persist_test_recipe(persisted, resolved)
            if recipe_state == "derived":
                persist_test_recipe(persisted, replace(resolved, command=ResolvedFact(("true",), "derived")))
            elif recipe_state in {"legacy", "stale"}:
                payload = recipe_to_payload(resolved)
                payload["format_version"] = 2 if recipe_state == "legacy" else 999
                payload["command"] = {"source": "unresolved", "value": None}
                path.write_text(json.dumps(payload))
            recipe = load_test_recipe(persisted)
            assert (recipe is None) is (recipe_state == "stale"), recipe_state
        backend, fixer = ScriptedBackend(), ScriptedBackend()
        result = await phase_test_and_heal(
            backend, work_context(target, run_id="configuration-no-execution"), config=config,
            session_id="configuration-no-execution", capture_tree_key=capture,
            footprint=AuthorizedFixFootprint(frozenset({"services/taste/parser.go"}), 1),
            repair_backend=fixer, recipe=recipe, allow_standalone=True,
            run_context=RunContext(InteractionPolicy(assume="yes", interactive=False)),
        )
        assert not result.passed and not result.proceed and not result.ignored, recipe_state
        assert result.retries == 0 and result.repairs == (), recipe_state
        assert result.diagnostic is not None, recipe_state
        assert (result.diagnostic.reason_code, result.diagnostic.source) == (reason, source), recipe_state
        assert "private-token" not in result.diagnostic.message
        assert len(result.attempts) == 1 and result.attempts[0].kind == "no-execution"
        assert result.attempts[0].identity is None
        assert result.attempts[0].input_tree_key == result.attempts[0].output_tree_key
        assert backend.call_count == fixer.call_count == 0
    assert before == (
        git(target, "rev-parse", "HEAD"), git(target, "diff", "--binary"), git(target, "diff", "--cached"),
    )
    assert not (target / ".daydream" / "deep" / "repair-job.json").exists()
    assert not list(target.glob(".daydream/**/handoff*.md"))
