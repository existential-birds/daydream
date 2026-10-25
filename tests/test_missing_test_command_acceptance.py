"""Issue #1491: command readiness gates fixes before any suite or repair runs.

Runner journeys resolve and persist their own recipes. The supplementary phase
rows exercise genuine recipe reload, because runner intentionally resolves anew.
Only the external backend is replaced in the runner regressions.
"""
from __future__ import annotations

import json
import os
import shlex
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent
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
from tests.deep_orchestrator.support import _only_archived_run, _scan_phase_events, _scan_trajectory_extra
from tests.harness.backend import ScriptedBackend
from tests.harness.execution import host_test_command
from tests.harness.git_helpers import commit, git, work_context
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import MakeConfig, _fix_prompts, _merge_item


def _arrange(target: Path, monkeypatch: pytest.MonkeyPatch) -> StubBackend:
    (target / "api.py").write_text("VALUE = 2\n")
    (target / "requirements.txt").write_text("pytest\n")
    git(target, "add", "api.py", "requirements.txt")
    commit(target, "test: actionable Python change and manifest candidate")
    backend = StubBackend(target)
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    backend.fix_edit_line = "# unauthorized fix turn ran\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    return backend


def _state(target: Path) -> tuple[str, str, dict[str, bytes]]:
    tracked = git(target, "ls-files").splitlines()
    paths = [*tracked, "owner-untracked.txt"]
    return git(target, "rev-parse", "HEAD"), git(target, "diff", "--cached", "--binary"), {
        name: (target / name).read_bytes() for name in paths
    }


@pytest.mark.parametrize(("command", "file_value", "reason", "source", "journeys"), [
    pytest.param(None, None, "missing", "config", 2, id="absent-two-runner-journeys"),
    pytest.param(" \t ", None, "blank", "cli", 1, id="whitespace-cli"),
    pytest.param("python 'private-token", None, "malformed", "cli", 1, id="malformed-cli"),
    pytest.param('""', None, "empty-argv", "cli", 1, id="empty-argv-cli"),
    pytest.param(123, None, "invalid-type", "cli", 1, id="invalid-cli-type"),
    pytest.param("daydream-no-such-suite-command", None, "unavailable", "cli", 1, id="unavailable-executable"),
    pytest.param(None, '" \t "', "blank", "config", 1, id="whitespace-file"),
    pytest.param(None, '"python \'private-token"', "malformed", "config", 1, id="malformed-file"),
    pytest.param(None, "123", "invalid-type", "config", 1, id="invalid-file-type"),
    pytest.param(None, '\'""\'', "empty-argv", "config", 1, id="empty-argv-file"),
    pytest.param(" ", '"true"', "blank", "cli", 1, id="invalid-cli-wins-valid-file"),
])
async def test_runner_rejects_missing_canonical_before_fix_or_test(
    linked_worktree: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, archive_dir: Path, command: object, file_value: str | None,
    reason: str, source: str, journeys: int,
) -> None:
    """Runner resolves anew each journey; actual artifacts prove first-run persistence.

    File rows enter the real TOML loader before runner.run; the missing row runs
    twice on the same linked worktree. Loaded phase recipes have separate proof.
    """
    _, target = linked_worktree
    backend = _arrange(target, monkeypatch)
    if file_value is not None:
        (target / ".daydream.toml").write_text(f"test_command = {file_value}\n")
    (target / "base.txt").write_text("owner dirty bytes\n")
    (target / "owner-untracked.txt").write_text("owner untracked bytes\n")
    before = _state(target)
    counter = tmp_path / "candidate-runs"
    candidate = tmp_path / "pytest"
    candidate.write_text(f"#!/bin/sh\nprintf ran >> '{counter}'\nexit 0\n")
    candidate.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    config = make_config(target, stack="python", assume="yes", test_command=command,
                         file_config=load_file_config(target), archive=journeys == 2)
    for journey in range(journeys):
        trajectory = tmp_path / f"trajectory-{journey}.json"
        config.trajectory_path = trajectory
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
        assert "private-token" not in json.dumps(verdict)
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
        if journeys == 2 and journey == 0:
            archived = json.loads((_only_archived_run(archive_dir) / "manifest.json").read_text())
            test_state = archived["phase_states"]["test"]
            assert test_state["ran"] is False and test_state["status"] == "failed"
            assert test_state["diagnostic"] == diagnostic


@pytest.mark.parametrize(("recipe_state", "reason", "source"), [
    ("direct", "malformed", "cli"), ("loaded", "malformed", "cli"),
    ("legacy", "unresolved", "unresolved"), ("derived", "unresolved", "derived"),
    ("stale", "malformed", "cli"),
])
async def test_public_test_phase_returns_no_execution_without_repair(
    linked_worktree: tuple[Path, Path], tmp_path: Path, make_config: MakeConfig,
    recipe_state: str, reason: str, source: str,
) -> None:
    """Public phase covers direct config and loaded current/legacy/derived recipes.

    Legacy/stale rows model actual untrusted historical bytes at the loader;
    these supplement the genuine producer runner journeys above.
    """
    _, target = linked_worktree
    config = make_config(target, assume="yes", test_command="python 'private-token")
    recipe = None
    if recipe_state != "direct":
        resolved = resolve_test_recipe(config.file_config, config, repo_root=target)
        path = persist_test_recipe(tmp_path / "persisted", resolved)
        if recipe_state == "derived":
            persist_test_recipe(tmp_path / "persisted", replace(
                resolved, command=ResolvedFact(("true",), "derived"),
            ))
        elif recipe_state in {"legacy", "stale"}:
            payload = recipe_to_payload(resolved)
            payload["format_version"] = 2 if recipe_state == "legacy" else 999
            payload["command"] = {"source": "unresolved", "value": None}
            path.write_text(json.dumps(payload))
        recipe = load_test_recipe(tmp_path / "persisted")
        assert (recipe is None) is (recipe_state == "stale")
    backend, fixer = ScriptedBackend(), ScriptedBackend()
    before = git(target, "rev-parse", "HEAD"), git(target, "diff", "--binary"), git(target, "diff", "--cached")

    def capture() -> str:
        return git_ops.tree_key(git_ops.snapshot_worktree_delta(
            target, "HEAD", preexisting_untracked={}, preexisting_gitlinks=(),
        ))

    result = await phase_test_and_heal(
        backend, work_context(target, run_id="configuration-no-execution"), config=config,
        session_id="configuration-no-execution", capture_tree_key=capture,
        footprint=AuthorizedFixFootprint(frozenset({"services/taste/parser.go"}), 1),
        repair_backend=fixer, recipe=recipe, allow_standalone=True,
        run_context=RunContext(InteractionPolicy(assume="yes", interactive=False)),
    )
    assert not result.passed and not result.proceed and not result.ignored
    assert result.retries == 0 and result.repairs == ()
    assert result.diagnostic is not None
    assert result.diagnostic.reason_code == reason and result.diagnostic.source == source
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


@pytest.mark.parametrize(("mode", "output", "passed", "source"), [
    ("pass", "12 failed, 0 passed", True, "cli"),
    ("pass", "12 failed, 0 passed", True, "config"),
    ("all", "12 passed, 0 failed", False, "cli"),
])
async def test_configured_runner_uses_host_exit_status(
    linked_worktree: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, mode: Literal["pass", "all"], output: str, passed: bool, source: str,
) -> None:
    """Configured controls execute real scripts; persuasive output cannot change exits."""
    _, target = linked_worktree
    backend = _arrange(target, monkeypatch)
    command = host_test_command(tmp_path / "host-suite", mode=mode, output=output)
    if passed:
        # The configured-command control needs no retained edit; meaningful
        # repair/checkpoint/fix controls below exercise mutating configured runs.
        backend.fix_edit_line = None
    file_command = command if source == "config" else "false"
    (target / ".daydream.toml").write_text(
        f"test_command = {json.dumps(file_command)}\ntest_required_suites = ['python', 'rl']\n")
    rc = await run(make_config(target, stack="python", assume="yes",
                              test_command=command if source == "cli" else None,
                              test_required_suites=["explicit"] if source == "config" else [],
                              file_config=load_file_config(target)))
    recipe = load_test_recipe(target / ".daydream/deep")
    assert recipe is not None
    assert recipe.command.value == tuple(shlex.split(command)) and recipe.command.source == source
    assert recipe.declared == (("explicit",) if source == "config" else ("python", "rl"))
    verdict = json.loads((target / ".daydream/deep/test-verdict.json").read_text())
    assert verdict["passed"] is passed
    assert verdict["status"] == "executed"
    assert verdict["diagnostic"] is None
    assert int((tmp_path / "host-suite/suite-count.txt").read_text()) >= 1
    assert backend.test_suite_calls == 0
    assert all(attempt["kind"] == "host" and attempt["passed"] is passed
               and attempt["command_source"] == source for attempt in verdict["attempts"])
    if passed:
        assert rc == 0
    else:
        assert rc != 0
        assert verdict["retries"] == 1 and len(verdict["repairs"]) == 1
        assert len([call for call in backend.calls if call["prompt"].startswith("The tests failed")]) == 1


async def test_missing_command_resume_preserves_genuine_repair_checkpoint(
    linked_worktree: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A failed first runner produces a job, then the operator restores reviewed source.

    That real Git action makes --start-at fix eligible without changing artifacts;
    the configuration rejection must preserve both durable job and checkpoint.
    """
    _, target = linked_worktree
    backend = _arrange(target, monkeypatch)
    backend.heal_fix_authorized = "api.py"
    backend.heal_fix_authorized_line = "\n# candidate repair\n"
    backend.heal_fix_partial = "interrupted candidate diagnosis"
    backend.runaway_fix_sleep_s = 0.05
    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    # Resume requires a clean checkout independently of the diff fingerprint.
    # Track this sentinel; the fresh-run matrix above owns untracked preservation.
    (target / "owner-untracked.txt").write_text("owner sentinel\n")
    git(target, "add", "owner-untracked.txt")
    commit(target, "test: track the operator sentinel before repair journey")
    initial_source = _state(target)[2]
    command = host_test_command(tmp_path / "prior-suite", mode="all")
    assert await run(make_config(target, stack="python", assume="yes", test_command=command)) != 0
    deep = target / ".daydream" / "deep"
    job = deep / "repair-job.json"
    checkpoint = deep / "repair-checkpoint.json"
    saved_job, saved_checkpoint = job.read_bytes(), checkpoint.read_bytes()
    assert json.loads(saved_job)["executions"] > 0
    assert "candidate repair" in json.loads(saved_checkpoint)["candidate_patch"]
    prior_count = (tmp_path / "prior-suite/suite-count.txt").read_text()
    # An operator restores the reviewed tracked files and preexisting sentinel,
    # while keeping the coordinator's original durable artifacts untouched.
    for name, content in initial_source.items():
        (target / name).write_bytes(content)
    before = _state(target)
    backend.calls.clear()

    rc = await run(make_config(target, stack="python", assume="yes", start_at="fix", test_command=None))

    assert rc != 0
    assert _state(target) == before
    assert job.read_bytes() == saved_job and checkpoint.read_bytes() == saved_checkpoint
    assert (tmp_path / "prior-suite/suite-count.txt").read_text() == prior_count
    assert not _fix_prompts(backend) and backend.test_suite_calls == 0
    assert not any(call["prompt"].startswith("The tests failed") for call in backend.calls)
    verdict = json.loads((deep / "test-verdict.json").read_text())
    assert verdict["status"] == "no-execution" and verdict["retries"] == 0
    assert verdict["repairs"] == [] and verdict["diagnostic"]["reason_code"] == "missing"
    loaded = load_test_recipe(deep)
    assert loaded is not None and not loaded.command.resolved
    suite, fixer = ScriptedBackend(), ScriptedBackend()
    result = await phase_test_and_heal(
        suite, work_context(target, run_id="noexecution-loaded-job"), config=make_config(target, test_command=None),
        session_id="noexecution-loaded-job", capture_tree_key=lambda: git(target, "write-tree"),
        footprint=AuthorizedFixFootprint(frozenset({"api.py"}), 1), repair_backend=fixer,
        recipe=loaded, allow_standalone=True,
        run_context=RunContext(InteractionPolicy(assume="yes", interactive=False)),
    )
    assert result.diagnostic is not None and not result.passed and not result.proceed
    assert result.retries == 0 and result.repairs == ()
    assert suite.call_count == fixer.call_count == 0
    assert job.read_bytes() == saved_job and checkpoint.read_bytes() == saved_checkpoint


async def test_late_unavailable_command_preserves_executed_failure_and_repair_history(
    linked_worktree: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, archive_dir: Path,
) -> None:
    """A real red suite may heal once, but losing its executable stops the retry.

    This distinct runner boundary protects already-produced host/repair evidence:
    no-execution describes the final attempt, not the entire run's history.
    """
    _, target = linked_worktree
    _arrange(target, monkeypatch)
    executable = tmp_path / "canonical-suite"
    counter = tmp_path / "suite-executions.txt"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf 'ran\\n' >> {shlex.quote(str(counter))}\n"
        "printf '1 failed\\n'\nexit 1\n"
    )
    executable.chmod(0o755)

    class LosingExecutableBackend(StubBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if prompt.startswith("The tests failed"):
                executable.unlink()
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = LosingExecutableBackend(target)
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    backend.heal_fix_authorized = "api.py"
    backend.heal_fix_authorized_line = "\n# actual authorized suite repair\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    head_before = git(target, "rev-parse", "HEAD")
    rc = await run(make_config(target, stack="python", assume="yes", archive=True,
                               test_command=shlex.join([str(executable)])))

    assert rc != 0
    assert counter.read_text().splitlines() == ["ran"]
    assert git(target, "rev-parse", "HEAD") == head_before
    assert git(target, "diff", "--cached") == ""
    assert "actual authorized suite repair" in (target / "api.py").read_text()
    assert backend.test_suite_calls == 0
    assert len([call for call in backend.calls if call["prompt"].startswith("The tests failed")]) == 1
    assert not any("read-only failure-summarizer" in call["prompt"].lower() for call in backend.calls)
    deep = target / ".daydream" / "deep"
    verdict = json.loads((deep / "test-verdict.json").read_text())
    assert verdict["passed"] is False and verdict["ignored"] is False
    assert verdict["diagnostic"]["reason_code"] == "unavailable"
    assert verdict["diagnostic"]["source"] == "cli"
    assert [attempt["kind"] for attempt in verdict["attempts"]] == ["host", "no-execution"]
    assert verdict["attempts"][0]["identity"]["outcome"] == "failed"
    checkpoint = json.loads((deep / "repair-checkpoint.json").read_text())
    assert "actual authorized suite repair" in checkpoint["candidate_patch"]
    archived = json.loads((_only_archived_run(archive_dir) / "manifest.json").read_text())
    archived_test = archived["phase_states"]["test"]
    assert (len(verdict["repairs"]), archived_test["ran"]) == (1, True)
    assert "api.py" in verdict["repairs"][0]["changed_paths"]
    assert not (target / ".daydream-heal-fix-applied").exists()
    assert verdict["repairs"][0]["checkpoint_ref"] == "repair-checkpoint.json"
    assert archived_test["status"] == "failed" and archived_test["diagnostic"] == verdict["diagnostic"]
