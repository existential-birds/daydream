"""Integration tests for the full review-fix-test flow."""
import asyncio
import json
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import (
    AgentEvent,
)
from daydream.backends.codex import CodexBackend
from daydream.remote_ci import RemoteCITarget, pending_remote_ci_verdict, write_remote_ci_handoff
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, commit as _commit, git as _git, init_repo as _init_repo
from tests.harness.phase_backend import PhaseDispatchBackend
from tests.harness.processes import wait_for_process_group_exit
from tests.harness.remote_ci import (
    finish_remote_ci_fake,
    seed_pr_identity,
    start_remote_ci_fake,
)
from tests.test_deep_orchestrator import _install_stub_backend, _silence
from tests.test_runner import _fix_item, _seed_fix_resume


def _load_trajectory(project: Path, session_id: str) -> dict[str, Any]:
    """Load the recorded trajectory for ``session_id`` from ``project``'s run dir."""
    path = project / ".daydream" / "runs" / session_id / "trajectory.json"
    return cast(dict[str, Any], json.loads(path.read_text()))

def _remote_ci_phase_end(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Return the single ``remote-ci`` ``phase_end`` event, asserting uniqueness."""
    remote_ends: list[dict[str, Any]] = [
        event
        for event in trajectory["extra"]["phase_events"]
        if event["phase"] == "remote-ci" and event["event"] == "phase_end"
    ]
    assert len(remote_ends) == 1
    return remote_ends[0]


_FULL_FLOW_ISSUE = {"id": 1, "description": "Add type hints to function", "file": "main.py", "line": 1}


class _WorktreeMutatingBackend(PhaseDispatchBackend):
    """Fake only the backend; fix turns perform the requested real edit."""

    async def execute(
        self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        if prompt.startswith("Fix this issue") or prompt.startswith("Fix these"):
            (cwd / "main.py").write_text("def hello() -> str:\n    return 'world'\n")
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event

def _remote_ci_push_project(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    """Create a spaced-path checkout pushed through a truthful GitHub URL."""
    project = tmp_path / "remote ci project"
    project.mkdir()
    (project / "main.py").write_text("def hello():\n    return 'world'\n")
    _init_repo(project)
    _git(project, "add", "main.py")
    _commit(project, "base")
    _git(project, "checkout", "-b", "feature")
    (project / "main.py").write_text("def hello():\n    return 'universe'\n")
    _git(project, "add", "main.py")
    _commit(project, "feature")
    remote = bare_remote(tmp_path / "remote ci origin.git")
    raw_remote = "https://github.com/fork-user/project.git"
    _git(project, "config", f"url.{remote.resolve().as_uri()}.insteadOf", raw_remote)
    _git(project, "remote", "add", "origin", raw_remote)
    marker = tmp_path / "pre push hook ran"
    return project, remote, marker, raw_remote

async def _wait_for_remote_ci_pids(path: Path, *, sha_path: Path, runner_task: asyncio.Task[int]) -> dict[str, int]:
    while not sha_path.exists():
        if runner_task.done():
            try:
                result = runner_task.result()
            except BaseException as exc:
                raise AssertionError("runner ended before reaching the remote-CI push boundary") from exc
            raise AssertionError(f"runner exited {result} before reaching the remote-CI push boundary")
        await asyncio.sleep(0.01)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if runner_task.done():
            try:
                result = runner_task.result()
            except BaseException as exc:
                raise AssertionError("runner ended before the blocking remote-CI process started") from exc
            raise AssertionError(f"runner exited {result} before the blocking remote-CI process started")
        try:
            value = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            await asyncio.sleep(0.01)
            continue
        if isinstance(value.get("direct"), int) and isinstance(value.get("grandchild"), int):
            return {"direct": value["direct"], "grandchild": value["grandchild"]}
        await asyncio.sleep(0.01)
    raise AssertionError("blocking remote CI process did not publish process ids")

def _live_deep_dir(artifact_runtime_root: Path) -> Path:
    """Locate the sole active session: remote CI evidence lives in its private tree."""
    matches = sorted(artifact_runtime_root.glob("*/runs/*/live/.daydream/deep"))
    assert len(matches) == 1, f"expected exactly one live deep dir, got {matches}"
    return matches[0]

def _shallow_pr_config(make_config: Callable[..., "RunConfig"], project: Path, **overrides: object) -> "RunConfig":
    """Shallow PR-process config for the remote-CI tests, shared defaults seeded."""
    kwargs: dict[str, object] = {
        "stack": "python", "quiet": True, "shallow": True, "assume": "yes",
        "archive": True, "test_command": "true", "pr_number": 7, "pr_repo": "base-user/project",
    }
    kwargs.update(overrides)
    return make_config(project, **kwargs)

@pytest.mark.parametrize("resume", [True], ids=["resume-stale-handoff"])
async def test_runner_remote_ci_cancellation_persists_verdict_and_handoff(
    tmp_path: Path, artifact_runtime_root: Path, install_backend: Callable[[object], object],
    make_config: Callable[..., "RunConfig"], fake_gh: FakeGh, resume: bool,
) -> None:
    """Cancellation reaps the real gh process before durable operator state."""
    project, _remote, hook_marker, _raw_remote = _remote_ci_push_project(tmp_path)
    old_sha = _git(project, "rev-parse", "HEAD")
    deep = project / ".daydream" / "deep"
    deep.mkdir(parents=True)
    verdict_path = deep / "remote-ci-verdict.json"
    handoff_path = deep / "remote-ci-handoff.json"
    write_remote_ci_handoff(
        handoff_path,
        pending_remote_ci_verdict(
            RemoteCITarget(
                target_dir=project, base_repository="base-user/project", base_ref="main",
                head_repository="fork-user/project", head_ref="feature", pr_number=7,
                pr_url="https://github.com/base-user/project/pull/7", remote="origin", pushed_sha=old_sha,
            )
        ), session_id="old-session",
    )
    unrelated = deep / "operator-notes.json"
    unrelated.write_bytes(b'{"keep":"operator notes"}\n')
    if resume:
        _seed_fix_resume(project, [_fix_item()])
    seed_pr_identity(fake_gh, head_sha=old_sha)
    seeder = start_remote_ci_fake(project, fake_gh, hook_marker, outcome="blocking")
    install_backend(_WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]]))
    task = asyncio.create_task(
        run(_shallow_pr_config( make_config, project, archive=False, start_at="fix" if resume else "ttt", ))
    )
    pids: dict[str, int] | None = None
    try:
        pids = await _wait_for_remote_ci_pids(
            hook_marker.with_name(hook_marker.name + " pids"),
            sha_path=hook_marker.with_name(hook_marker.name + " sha"), runner_task=task,
        )
        # Mid-run artifacts stay private; the public tree is republished at run end.
        live_deep = _live_deep_dir(artifact_runtime_root)
        pending = json.loads((live_deep / "remote-ci-verdict.json").read_text())
        assert pending["status"] == "pending"
        assert pending["session_id"] != "old-session"
        assert pending["target"]["pushed_sha"] != old_sha
        assert not (live_deep / "remote-ci-handoff.json").exists(), (
            "the new attempt retained old-SHA guidance"
        )
        if resume:
            assert (live_deep / "operator-notes.json").read_bytes() == (
                b'{"keep":"operator notes"}\n'
            )
    finally:
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish_remote_ci_fake(seeder)
            if pids is not None:
                await wait_for_process_group_exit(pids["direct"])
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "cancelled"
    assert verdict["target"]["pushed_sha"] == _git(project, "rev-parse", "HEAD")
    assert json.loads(handoff_path.read_text())["status"] == "cancelled"
    verdict_bytes = verdict_path.read_bytes()
    calls = len(fake_gh.process_calls())
    await asyncio.sleep(0.05)
    assert verdict_path.read_bytes() == verdict_bytes
    assert len(fake_gh.process_calls()) == calls
    trajectory = _load_trajectory(project, verdict["session_id"])
    assert trajectory["session_id"] == verdict["session_id"]
    end = _remote_ci_phase_end(trajectory)
    assert end["status"] == "cancelled"
    assert end["reason_code"] == "cancelled"
    assert end["metadata"]["stop_reason"] == "cancelled"

def _two_commit_repo(repo: Path, filename: str, before: str, after: str, branch: str) -> Path:
    """A repo with *filename* committed on ``main`` and modified on *branch*."""
    _init_repo(repo)
    (repo / filename).write_text(before)
    _git(repo, "add", ".")
    _commit(repo, "init")
    _git(repo, "checkout", "-b", branch)
    (repo / filename).write_text(after)
    _git(repo, "add", ".")
    _commit(repo, "change")
    return repo


async def test_run_comment_pr_lookup_failure_and_absence_exit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., "RunConfig"], fake_gh: FakeGh,
) -> None:
    """A failed (auth) or empty PR lookup stops comment mode before model dispatch or posting."""
    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")
    _silence(monkeypatch)
    backend = _install_stub_backend(monkeypatch, tmp_path)
    errors: list[str] = []
    monkeypatch.setattr("daydream.runner.print_error", lambda _console, _title, message: errors.append(message))
    absent = "GraphQL: Could not resolve to a PullRequest with the number of 7. (repository.pullRequest)"
    for outcome, expected in (("auth_failure", "authentication required"), ("absence", "No target PR found")):
        errors.clear()
        fake_gh.set_response("pr-view", value={"__error__": expected if outcome == "auth_failure" else absent})
        assert await run(make_config(tmp_path, output_mode="comment", pr_number=7)) == 1, outcome
        assert any(expected in message for message in errors), (outcome, errors)
        assert outcome == "absence" or all("No target PR found" not in message for message in errors), outcome
    assert not backend.calls
    assert fake_gh.calls("POST", "repos/acme/widgets/pulls/7/reviews") == []

async def test_codex_backend_raises_on_agents(tmp_path: Path) -> None:
    """CodexBackend.execute() refuses agents= with NotImplementedError."""
    backend = CodexBackend(model="fixture-model")
    with pytest.raises(NotImplementedError, match="Codex backend does not support exploration"):
        async for _ in backend.execute(tmp_path, "prompt", agents={"x": object()}):
            pass


def test_remote_ci_evidence_parsing_and_rendering_name_every_non_green_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required policy unions pinned app ids, check states fail closed, and every non-green context is named.

    The runner journeys only reach the passing and cancelled shapes; a failed verdict
    must still render its failing, pending, missing, and advisory contexts.
    """
    from types import SimpleNamespace

    from daydream.deep.remote_ci_steps import _print_remote_ci_result
    from daydream.remote_ci.evidence import RequiredContext
    from daydream.remote_ci.parsing import _parse_check_state, parse_required_policy

    checks = [{"context": "test", "app_id": 7}, {"context": "lint", "app_id": None}]
    policy = parse_required_policy([], {"strict": True, "contexts": ["lint", "test"], "checks": checks})
    assert policy.strict is True
    assert policy.contexts == (RequiredContext("lint", None), RequiredContext("test", 7))
    assert _parse_check_state("queued", None) == ("pending", "queued")
    for status, conclusion in (("queued", "success"), ("unknown", None), ("completed", "unknown")):
        with pytest.raises(ValueError):
            _parse_check_state(status, conclusion)
    printed: list[str] = []
    monkeypatch.setattr("daydream.deep.remote_ci_steps.print_info", lambda _c, message: printed.append(str(message)))
    monkeypatch.setattr("daydream.deep.remote_ci_steps.print_warning", lambda _c, message: printed.append(message))
    verdict: Any = SimpleNamespace(
        target=None, status="failed", reason="required check failed", evidence_sha="a" * 40,
        failing_contexts=("test",), pending_contexts=("lint",), missing_contexts=("docs",),
        advisory_observations=(SimpleNamespace(context="coverage", state="fail"),
                               SimpleNamespace(context="bench", state="pass")),
        urls=("https://github.test/run/1",),
    )
    _print_remote_ci_result(verdict)
    assert printed == ["Remote CI result: failed — required check failed", f"Remote CI evidence SHA: {'a' * 40}",
                       "Failing CI: test", "Pending CI: lint", "Missing CI: docs", "Advisory CI not green: coverage",
                       "CI details: https://github.test/run/1"]
