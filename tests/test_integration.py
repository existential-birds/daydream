"""Integration tests for the full review-fix-test flow."""
import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator, Callable
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console

from daydream import remote_ci
from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend
from daydream.deep import fix_steps
from daydream.exploration import ExplorationContext
from daydream.phases import phase_alternative_review
from daydream.pr_review import PRInfo
from daydream.remote_ci import RemoteCITarget, pending_remote_ci_verdict, write_remote_ci_handoff
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext
from daydream.runner import run
from daydream.trajectory import DaydreamPhase
from daydream.ui import NEON_THEME
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, commit as _commit, git as _git, init_repo as _init_repo
from tests.harness.phase_backend import PhaseDispatchBackend
from tests.harness.processes import wait_for_process_group_exit
from tests.harness.remote_ci import (
    NoCIRemote,
    finish_remote_ci_fake,
    seed_pr_identity,
    start_remote_ci_fake,
)
from tests.harness.stub_backend import force_interactive, install_stub_backend, review_stage_result, silence
from tests.test_deep_orchestrator import _install_stub_backend, _silence
from tests.test_runner import _fix_item, _seed_fix_resume

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

#: The byte-identical CostEvent + ResultEvent tail every scripted stream ends with.
_TERMINAL_EVENTS: tuple[AgentEvent, ...] = (
    CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
    ResultEvent(structured_output=None, continuation=None),
)

_COMMENT_PR = PRInfo(
    number=7, head_sha="0" * 40, base_sha="1" * 40, base_ref="main", head_ref="feature", owner="acme", repo="widgets",
    url="https://example/pr/7",
)

def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text for assertion comparisons."""
    return _ANSI_ESCAPE.sub("", text)

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

async def render_agent(
    monkeypatch: pytest.MonkeyPatch, events: list[Any], *, quiet: bool, prompt: str = "Test prompt",
    color_system: str | None = None,
) -> str:
    """Run scripted events through a pinned terminal and return raw ANSI output.

    force_terminal, width=120, and NEON_THEME remove host display differences.
    Styling assertions use raw output; text assertions call strip_ansi."""
    output = StringIO()
    extra: dict[str, Any] = {} if color_system is None else {"color_system": color_system}
    monkeypatch.setattr(
        "daydream.agent.console",
        Console(file=output, force_terminal=True, no_color=False, width=120, theme=NEON_THEME, **extra),
    )
    await run_agent(
        ScriptedBackend(events=events, model="mock-model"), Path("/tmp"), prompt, phase=DaydreamPhase.REVIEW,
        run_context=RunContext(InteractionPolicy(quiet=quiet)),
    )
    return output.getvalue()

async def test_five_thinking_panels_render_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five consecutive thinking panels render once each, in order."""
    thoughts = [f"thought {i} of the stream" for i in range(5)]
    events: list[Any] = [
        *[ThinkingEvent(text=t) for t in thoughts], ResultEvent(structured_output=None, continuation=None),
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=False))
    assert plain_text.count("Thinking") == 5, "each thought must render its Thinking title once"
    cursor = -1
    for t in thoughts:
        idx = plain_text.find(t)
        assert idx != -1, f"thought {t!r} did not render"
        assert idx > cursor, f"thought {t!r} rendered out of order"
        cursor = idx


@pytest.fixture
def target_project(tmp_path: Path) -> Path:
    """Create a real two-commit feature branch satisfying workspace preflight."""
    return _two_commit_repo(
        tmp_path / "test_project", "main.py",
        "def hello():\n    return 'world'\n",
        "def hello():\n    return 'universe'\n",
        "feature",
    )

async def test_full_fix_flow(
    target_project: Path, tmp_path: Path, install_backend: Callable[[object], object],
    make_config: Callable[..., 'RunConfig'], no_ci_remote: NoCIRemote,
) -> None:
    """The shallow flow writes a report, applies a fix, tests it, and commits."""
    install_backend(_WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]]))
    # The real push and ls-remote verification need a bare origin.
    no_ci_remote.connect(target_project, bare_remote(tmp_path / "origin.git"))
    head_before = _git(target_project, "rev-parse", "HEAD")
    config = make_config(
        target_project, stack="python", quiet=True, shallow=True, assume="yes", pr_number=no_ci_remote.pr_number,
        pr_repo=no_ci_remote.base_repository,
    )
    exit_code = await run(config)
    assert exit_code == 0
    report = target_project / ".review-output.md"
    assert report.is_file()
    report_text = report.read_text()
    assert "main.py:1" in report_text
    assert "Add type hints to function" in report_text
    assert (target_project / ".daydream" / "deep" / "merged-items.json").is_file()
    assert json.loads((target_project / ".daydream" / "deep" / "test-verdict.json").read_text())["passed"]
    assert "-> str" in (target_project / "main.py").read_text()
    assert _git(target_project, "rev-parse", "HEAD") != head_before

@pytest.mark.parametrize("protect_untracked_related", [False, True])
async def test_fix_commit_includes_pre_gate_authorized_unstaged_edit(
    tmp_path: Path, install_backend: Callable[[object], object], make_config: Callable[..., 'RunConfig'],
    protect_untracked_related: bool, no_ci_remote: NoCIRemote,
) -> None:
    """Real runner commits the complete authorized HEAD-relative result."""
    repo = tmp_path / "pre-gate-authorized"
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n")
    (repo / "b.py").write_text("B = 1\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    _git(repo, "checkout", "-b", "feature")
    (repo / "a.py").write_text("A = 2\n")
    _git(repo, "add", "a.py")
    _commit(repo, "feature")
    # Reviewed and authorized before the fix gate, but never touched by the
    # fixer.  This must still be selected relative to the original HEAD.
    (repo / "a.py").write_text("A = 3\n")
    if protect_untracked_related:
        (repo / "scratch.py").write_text("PRIVATE_USER_DRAFT = 1\n")
    remote = bare_remote(tmp_path / "pre-gate-origin.git")
    no_ci_remote.connect(repo, remote)
    issue = {"id": 1, "description": "Update the related value", "file": "b.py", "line": 1}
    related_files = ["a.py", "scratch.py"] if protect_untracked_related else ["a.py"]

    class RelatedOnlyBackend(PhaseDispatchBackend):
        async def execute(
            self, cwd: Any, prompt: str, *args: Any, **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            schema = args[0] if args else kwargs.get("output_schema")
            fields = schema.get("properties", {}) if schema else {}
            if "items" in fields:
                yield ResultEvent(structured_output={"items": [{
                    **issue, "severity": "medium", "confidence": "HIGH", "rationale": "Related values disagree",
                    "evidence": "a.py:1 and b.py:1", "lens": "per-stack", "source_uids": ["python:1"],
                    "related_files": related_files,
                }]}, continuation=None)
                return
            if "issues" in fields and ("evaluate the implementation" in prompt.lower()
                                       or "repository-wide interactions" in prompt.lower()):
                yield ResultEvent(structured_output={"issues": []}, continuation=None)
                return
            if prompt.startswith("Fix this issue") or prompt.startswith("Fix these"):
                (Path(cwd) / "b.py").write_text("B = 2\n")
                if protect_untracked_related:
                    (Path(cwd) / "scratch.py").write_text("PRIVATE_USER_DRAFT = 2\n")
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    install_backend(RelatedOnlyBackend(parse_results=[[issue]]))
    exit_code = await run(
        make_config(
            repo, stack="python", quiet=True, latency_profile="forensic", assume="yes",
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0
    assert _git(repo, "show", "HEAD:a.py") == "A = 3"
    assert _git(repo, "show", "HEAD:b.py") == "B = 2"
    assert set(_git(repo, "show", "--pretty=", "--name-only", "HEAD").splitlines()) == {"a.py", "b.py"}
    remote_head = _git(remote, "rev-parse", "refs/heads/feature")
    assert remote_head == _git(repo, "rev-parse", "HEAD")
    if protect_untracked_related:
        assert (repo / "scratch.py").read_text() == "PRIVATE_USER_DRAFT = 1\n"
        assert "scratch.py" not in _git(remote, "ls-tree", "--name-only", remote_head).splitlines()
        assert "PRIVATE_USER_DRAFT" not in (repo / ".daydream" / "recommended.patch" ).read_text()

async def test_shallow_staged_fix_preflight_preserves_review_evidence_and_git_state(
    target_project: Path, install_backend: Callable[[object], object], make_config: Callable[..., 'RunConfig'],
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A fresh real run rejects a review-time staged edit without erasing proof."""
    stale_bytes = b'{"session_id":"review-phase","passed":true}\n'
    staged_source = b"def hello():\n    return 'staged during review'\n"
    class ReviewStagingBackend(PhaseDispatchBackend):
        staged_index: str | None = None
        async def execute(
            self, cwd: Any, prompt: str, *args: Any, **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            prompt_lower = prompt.lower()
            review_markers = (
                "inclusion obligation", "full change spans", "language-agnostic review practices",
                "assigned to this stack", "repository-wide interactions",
            )
            if self.staged_index is None and any(marker in prompt_lower for marker in review_markers):
                repo = Path(cwd)
                deep = repo / ".daydream" / "deep"
                deep.mkdir(parents=True, exist_ok=True)
                (deep / "test-verdict.json").write_bytes(stale_bytes)
                (repo / "main.py").write_bytes(staged_source)
                _git(repo, "add", "main.py")
                self.staged_index = _git(repo, "write-tree")
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
    backend = ReviewStagingBackend(parse_results=[[_FULL_FLOW_ISSUE]])
    install_backend(backend)
    deep = target_project / ".daydream" / "deep"
    stale = deep / "test-verdict.json"
    assert not stale.exists()
    head_before = _git(target_project, "rev-parse", "HEAD")
    exit_code = await run(make_config( target_project, stack="python", quiet=True, shallow=True, assume="yes", ))
    assert exit_code == 1
    output = capfd.readouterr().out
    console_file = getattr(fix_steps, "console").file
    if isinstance(console_file, StringIO):
        output += console_file.getvalue()
    assert "Cannot start the fix cycle with staged changes" in output, (
        output,
        "\n".join(backend.call_log),
        _git(target_project, "status", "--short"),
    )
    assert backend.staged_index is not None
    assert _git(target_project, "write-tree") == backend.staged_index
    assert _git(target_project, "rev-parse", "HEAD") == head_before
    assert (target_project / "main.py").read_bytes() == staged_source
    assert stale.read_bytes() == stale_bytes
    assert not any(prompt.startswith("fix this issue") or prompt.startswith("fix these") for prompt in backend.call_log)

class _WorktreeMutatingBackend(PhaseDispatchBackend):
    """Fake only the backend; fix/commit turns perform the requested real Git edits."""

    async def execute(
        self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        if prompt.startswith("Fix this issue") or prompt.startswith("Fix these"):
            (cwd / "main.py").write_text("def hello() -> str:\n    return 'world'\n")
        elif prompt.startswith("The daydream changes are already staged"):
            run_id = prompt.split("Daydream-Run: ", 1)[1].splitlines()[0]
            version = prompt.split("Daydream-Version: ", 1)[1].splitlines()[0]
            # The index is pre-staged by _do_commit (issue #543) — commit it.
            _commit(cwd, f"fix: add type hints\n\nDaydream-Run: {run_id}\nDaydream-Version: {version}")
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

async def test_runner_remote_ci_red_fails_after_real_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_backend: Callable[[object], object],
    make_config: Callable[..., "RunConfig"], fake_gh: FakeGh, archive_dir: Path,
) -> None:
    """A locally green real push cannot complete without exact remote CI."""
    project, remote, hook_marker, raw_remote = _remote_ci_push_project(tmp_path)
    old_sha = _git(project, "rev-parse", "HEAD")
    seed_pr_identity(fake_gh, head_sha=old_sha)
    seeder = start_remote_ci_fake(project, fake_gh, hook_marker, outcome="failed")
    rendered = StringIO()
    monkeypatch.setattr("daydream.deep.remote_ci_steps.console", Console(file=rendered, width=160))
    install_backend(_WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]]))
    try:
        exit_code = await run(_shallow_pr_config(make_config, project))
    finally:
        finish_remote_ci_fake(seeder)
    assert exit_code == 1
    new_sha = _git(project, "rev-parse", "HEAD")
    assert new_sha != old_sha
    assert _git(project, "config", "--get", "remote.origin.url") == raw_remote
    assert _git(remote, "rev-parse", "refs/heads/feature") == new_sha
    assert hook_marker.read_text() == "ran\n"
    verdict_path = project / ".daydream" / "deep" / "remote-ci-verdict.json"
    assert verdict_path.is_file()
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "failed"
    assert verdict["target"]["pushed_sha"] == new_sha
    assert verdict["evidence_sha"] == new_sha
    assert verdict["failing_contexts"] == ["Build [matrix] (app 10)"]
    assert [item["context"] for item in verdict["advisory_observations"]] == ["Lint [optional]"]
    assert verdict["urls"] == [
        "https://github.com/base-user/project/actions/runs/[matrix]/7",
        "https://github.com/base-user/project/actions/runs/[optional]/8",
    ]
    assert "top-secret" not in verdict_path.read_text()
    handoff = json.loads((project / ".daydream" / "deep" / "remote-ci-handoff.json").read_text())
    assert handoff["status"] == "failed"
    assert handoff["target"]["pushed_sha"] == new_sha
    output = rendered.getvalue()
    assert "Failing CI: Build [matrix] (app 10)" in output
    assert "Advisory CI not green: Lint [optional]" in output
    assert "https://github.com/base-user/project/actions/runs/[matrix]/7" in output
    assert "https://github.com/base-user/project/actions/runs/[optional]/8" in output
    assert "\\[matrix]" not in output
    assert "\\[optional]" not in output
    assert "Commit and push complete" not in output
    assert "Exact pushed-SHA remote CI passed" not in output
    manifests = list((archive_dir / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["phase_states"]["test"]["status"] == "succeeded"
    assert manifest["phase_states"]["push"]["status"] == "succeeded"
    assert manifest["phase_states"]["remote_ci"]["status"] == "failed"
    assert manifest["pipeline_status"] == "failed"

@pytest.mark.parametrize("ci_variant", ["delayed", "merge-delayed"])
async def test_runner_remote_ci_replaces_stale_and_waits_for_exact_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_backend: Callable[[object], object],
    make_config: Callable[..., "RunConfig"], fake_gh: FakeGh, archive_dir: Path, ci_variant: str,
) -> None:
    """Old green evidence cannot satisfy a new push that registers later."""
    project, remote, hook_marker, _raw_remote = _remote_ci_push_project(tmp_path)
    old_sha = _git(project, "rev-parse", "HEAD")
    seed_pr_identity(fake_gh, head_sha=old_sha)
    stale = project / ".daydream" / "deep" / "remote-ci-verdict.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        json.dumps(
            {"schema_version": 1, "session_id": "old-session", "status": "passed", "target": {"pushed_sha": old_sha}}
        )
    )
    seeder = start_remote_ci_fake(
        project, fake_gh, hook_marker, outcome=ci_variant
    )
    monkeypatch.setattr(
        remote_ci, "DEFAULT_LIMITS",
        remote_ci.RemoteCILimits(poll_seconds=0.01, discovery_seconds=60, completion_seconds=60, request_seconds=5),
    )
    install_backend(_WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]]))
    try:
        exit_code = await run(_shallow_pr_config(make_config, project))
    finally:
        finish_remote_ci_fake(seeder)
    assert exit_code == 0
    new_sha = _git(project, "rev-parse", "HEAD")
    assert new_sha != old_sha
    assert _git(remote, "rev-parse", "refs/heads/feature") == new_sha
    verdict = json.loads(stale.read_text())
    assert verdict["status"] == "passed"
    assert verdict["session_id"] != "old-session"
    assert verdict["target"]["pushed_sha"] == new_sha
    expected_evidence = "e" * 40 if ci_variant == "merge-delayed" else new_sha
    assert verdict["evidence_sha"] == expected_evidence
    assert verdict["polling"]["stable_polls"] >= 2
    assert not (stale.parent / "remote-ci-handoff.json").exists()
    api_calls = [call.endpoint for call in fake_gh.calls("GET")]
    assert not any(old_sha in endpoint for endpoint in api_calls)
    if ci_variant == "merge-delayed":
        assert any(expected_evidence in endpoint for endpoint in api_calls)
    manifests = list((archive_dir / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["phase_states"]["test"]["status"] == "succeeded"
    assert manifest["phase_states"]["push"]["status"] == "succeeded"
    assert manifest["phase_states"]["remote_ci"]["status"] == "succeeded"
    assert manifest["pipeline_status"] == "succeeded"
    trajectory = json.loads((manifests[0].parent / "trajectory.json").read_text())
    end = _remote_ci_phase_end(trajectory)
    assert end["status"] == "succeeded"
    assert "reason_code" not in end
    assert end["metadata"]["stop_reason"] == "passed"

async def test_runner_remote_ci_keyboard_interrupt_preserves_interrupted_phase_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_backend: Callable[[object], object],
    make_config: Callable[..., "RunConfig"], fake_gh: FakeGh,
) -> None:
    """A host interrupt stays distinct after its cancelled handoff is persisted."""
    project, remote, hook_marker, _raw_remote = _remote_ci_push_project(tmp_path)
    old_sha = _git(project, "rev-parse", "HEAD")
    seed_pr_identity(fake_gh, head_sha=old_sha)
    seeder = start_remote_ci_fake(project, fake_gh, hook_marker, outcome="delayed")
    async def interrupt_fetch(
        _self: remote_ci.GitHubRemoteCIFetcher, _target: remote_ci.RemoteCITarget, *, budget: Any,
    ) -> remote_ci.RemoteCISnapshot:
        del budget
        raise KeyboardInterrupt
    monkeypatch.setattr(remote_ci.GitHubRemoteCIFetcher, "fetch", interrupt_fetch)
    install_backend(_WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]]))
    try:
        with pytest.raises(KeyboardInterrupt):
            await run(_shallow_pr_config(make_config, project, archive=False))
    finally:
        finish_remote_ci_fake(seeder)
    new_sha = _git(project, "rev-parse", "HEAD")
    assert new_sha != old_sha
    assert _git(remote, "rev-parse", "refs/heads/feature") == new_sha
    assert hook_marker.read_text() == "ran\n"
    deep = project / ".daydream" / "deep"
    verdict = json.loads((deep / "remote-ci-verdict.json").read_text())
    handoff = json.loads((deep / "remote-ci-handoff.json").read_text())
    assert verdict["status"] == "cancelled"
    assert verdict["target"]["pushed_sha"] == new_sha
    assert handoff["status"] == "cancelled"
    assert handoff["target"]["pushed_sha"] == new_sha
    verdict_bytes = (deep / "remote-ci-verdict.json").read_bytes()
    await asyncio.sleep(0.05)
    assert (deep / "remote-ci-verdict.json").read_bytes() == verdict_bytes
    trajectory = _load_trajectory(project, verdict["session_id"])
    end = _remote_ci_phase_end(trajectory)
    assert end["status"] == "cancelled"
    assert end["reason_code"] == "cancelled"
    assert end["metadata"]["stop_reason"] == "interrupted"

@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume-stale-handoff"])
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

async def test_shallow_commits_when_operator_ignores_red_suite(
    monkeypatch: pytest.MonkeyPatch, feature_branch_repo: Path, tmp_path: Path,
    install_backend: Callable[[object], object], make_config: Callable[..., 'RunConfig'],
    silence_console: Callable[..., None], no_ci_remote: NoCIRemote,
) -> None:
    """Ignoring a red suite keeps proceed=True while passed=False.

    Real heal and commit phases must then honor the commit approval and persist
    the backend edit in Git, exiting successfully."""
    # stdin answers, in order: intent confirmation, decline the optional PR
    # review post, the apply-fixes gate, the heal menu ("3" = ignore and
    # continue), and the commit gate.
    monkeypatch.setattr("sys.stdin", StringIO("y\nn\ny\n3\ny\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)
    silence_console("daydream.runner")
    silence_console("daydream.deep.orchestrator")
    silence_console("daydream.deep.review_steps")
    silence_console("daydream.deep.merge_steps")
    silence_console("daydream.deep.diagram_steps")
    silence_console("daydream.deep.fix_steps")
    silence_console("daydream.ui")
    install_backend(_WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]], tests_pass=False))
    # The real push and ls-remote verification need a bare origin.
    no_ci_remote.connect(feature_branch_repo, bare_remote(tmp_path / "origin.git"))
    head_before = _git(feature_branch_repo, "rev-parse", "HEAD")
    config = make_config(
        feature_branch_repo, stack="python", quiet=True, shallow=True, non_interactive=False, output_mode="loop",
        pr_number=no_ci_remote.pr_number, pr_repo=no_ci_remote.base_repository,
    )
    exit_code = await run(config)
    assert exit_code == 0, "choice '3' must continue the run, not abort it"
    assert _git(feature_branch_repo, "rev-parse", "HEAD") != head_before, (
        "the ignored-red-suite run never committed"
    )
    assert "-> str" in _git(feature_branch_repo, "show", "HEAD:main.py"), (
        "the fix was reverted instead of committed"
    )
    assert "Daydream-Run:" in _git(feature_branch_repo, "log", "-1", "--format=%B")

async def test_glob_tool_panel_displays_file_count_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render the full Glob lifecycle, streamed text cursor, and stable thinking title."""
    tool_use_id = "test-glob-lifecycle-123"
    glob_result = """/project/src/main.py
/project/src/utils/helper.py
/project/tests/test_main.py"""
    events = [
        ThinkingEvent(text="Analyzing the project structure..."),
        TextEvent(text="I'll search for Python files in the project."),
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "**/*.py", "path": "/project"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False), *_TERMINAL_EVENTS,
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=False, prompt="Test prompt for Glob tool"))
    assert "Thinking" in plain_text
    assert "Analyzing the project structure" in plain_text
    assert "I'll search for Python files" in plain_text
    assert "Glob" in plain_text
    assert "**/*.py" in plain_text
    assert "Found 3 files" in plain_text
    assert "main.py" in plain_text
    assert "helper.py" in plain_text
    assert "test_main.py" in plain_text

async def test_glob_tool_panel_singular_file_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LiveToolPanel shows singular 'file' for 1 result in normal mode."""
    tool_use_id = "test-glob-singular-456"
    glob_result = "/project/main.py"
    events = [
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "*.py"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False), *_TERMINAL_EVENTS,
    ]
    output_text = await render_agent(monkeypatch, events, quiet=False)
    assert "Found 1 file" in output_text
    assert "Found 1 files" not in output_text
    assert "main.py" in output_text

async def test_glob_tool_panel_truncates_long_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LiveToolPanel truncates long Glob results in normal mode."""
    tool_use_id = "test-glob-truncate-789"
    # 25 files exceeds max_lines=20 from _build_result_content_internal.
    mock_files = [f"/project/src/module{i}.py" for i in range(25)]
    glob_result = "\n".join(mock_files)
    events = [
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "**/*.py"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False), *_TERMINAL_EVENTS,
    ]
    output_text = await render_agent(monkeypatch, events, quiet=False)
    assert "Found 25 files" in output_text
    assert "and 5 more" in output_text  # 25 total - 20 displayed

async def test_quiet_mode_shows_header_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that quiet mode shows header only (no output section)."""
    tool_use_id = "test-output-panel-001"
    read_result = "def hello():\n    return 'world'"
    events = [
        ToolStartEvent(id=tool_use_id, name="Read", input={"file_path": "/project/main.py"}),
        ToolResultEvent(id=tool_use_id, output=read_result, is_error=False), *_TERMINAL_EVENTS,
    ]
    output_text = await render_agent(monkeypatch, events, quiet=True)
    plain_text = strip_ansi(output_text)
    assert "Read" in plain_text
    assert "/project/main.py" in plain_text
    assert "Output" not in plain_text
    assert "hello" not in plain_text
    assert "world" not in plain_text
    assert "╭" in output_text or "│" in output_text

async def test_quiet_mode_bash_panel_shows_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quiet-mode Bash panel shows the command (issue #1108), like Read shows file_path."""
    tool_use_id = "test-bash-quiet-1108"
    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "git diff --stat"}),
        ToolResultEvent(id=tool_use_id, output="", is_error=False), *_TERMINAL_EVENTS,
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=True))
    assert "$ git diff --stat" in plain_text

async def test_quiet_mode_bash_panel_renders_more_than_bare_name_without_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A description-less Bash call renders more than the bare `🔨 Bash` line."""
    tool_use_id = "test-bash-quiet-nodesc-1108"
    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "ls -la /tmp"}),
        ToolResultEvent(id=tool_use_id, output="", is_error=False), *_TERMINAL_EVENTS,
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=True))
    assert "ls -la /tmp" in plain_text or "ls -la /tm" in plain_text
    assert "$ ls" in plain_text

async def test_quiet_mode_bash_panel_redacts_command_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Panel command is redacted via redact_structured_text before display."""
    tool_use_id = "test-bash-quiet-redact-1108"
    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "DB_PASSWORD=hunter2 make db-up"}),
        ToolResultEvent(id=tool_use_id, output="", is_error=False), *_TERMINAL_EVENTS,
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=True))
    assert "hunter2" not in plain_text
    assert "$ DB_PASSWORD=[REDACTED_ENV_VAR] make db-up" in plain_text

async def test_quiet_mode_bash_panel_redacts_before_truncating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secret redaction happens on the complete command before the 200-char slice."""
    # Redact before slicing at 200: the raw secret straddles that boundary,
    # while the redaction marker starts at column 194 and remains visible.
    filler = "x" * 180
    command = f"{filler} token=opaque-test-12345"
    tool_use_id = "test-bash-redact-order-1108"
    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": command}),
        ToolResultEvent(id=tool_use_id, output="", is_error=False), *_TERMINAL_EVENTS,
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=True))
    assert "opaque-test-12345" not in plain_text, "secret must never reach the panel"
    assert "opaque" not in plain_text, "a partial secret must not survive truncation"
    assert 'token="[REDA' in plain_text, "the truncated redaction marker must reach the panel"
    assert "[REDACTED_CREDENTIAL]" not in plain_text, "the complete command must be redacted before slicing"

async def test_quiet_mode_empty_result_shows_header_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that quiet mode shows header only for empty results (no output section)."""
    tool_use_id = "test-empty-result-002"
    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "true"}),
        ToolResultEvent(id=tool_use_id, output="", is_error=False), *_TERMINAL_EVENTS,
    ]
    output_text = await render_agent(monkeypatch, events, quiet=True)
    assert "Bash" in output_text
    assert "Output" not in output_text  # quiet mode: header only
    assert "╭" in output_text or "│" in output_text

async def test_quiet_mode_error_shows_header_with_red_border(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that quiet mode shows header only with red border for errors."""
    tool_use_id = "test-error-result-003"
    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "false"}),
        ToolResultEvent(id=tool_use_id, output="Command failed with exit code 1", is_error=True), *_TERMINAL_EVENTS,
    ]
    # Force truecolor for consistent RGB color codes across environments.
    output_text = await render_agent(monkeypatch, events, quiet=True, color_system="truecolor")
    assert "Bash" in output_text
    assert "Command failed" not in output_text  # quiet mode: header only, no error body
    assert "╭" in output_text or "│" in output_text
    assert re.search(r"\x1b\[[0-9;]*38;2;255;85;85(?:;[0-9;]+)?m[╭│╰─]", output_text), (
        "the failed tool's border must render red, not merely carry ANSI styling"
    )

async def test_skill_tool_panel_collapses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress the redundant Skill launch output already named in its header."""
    tool_use_id = "test-skill-collapse-001"
    events = [
        ToolStartEvent(id=tool_use_id, name="Skill", input={"skill": "review-python"}),
        ToolResultEvent(id=tool_use_id, output="Launching skill: review-python", is_error=False), *_TERMINAL_EVENTS,
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=True))
    assert "Skill" in plain_text
    assert "review-python" in plain_text
    # No Output panel: the header already shows the skill name, so the redundant
    # "Launching skill:" result is suppressed.
    assert "Output" not in plain_text
    assert "Launching skill:" not in plain_text

async def test_concurrent_tool_panels_display_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple tool starts before any result must render all results without corruption."""
    # 3 concurrent commands: all started before any complete.
    events = [
        ToolStartEvent(id="cmd-1", name="shell", input={"command": "git diff -- file1.py"}),
        ToolStartEvent(id="cmd-2", name="shell", input={"command": "git diff -- file2.py"}),
        ToolStartEvent(id="cmd-3", name="shell", input={"command": "git diff -- file3.py"}),
        ToolResultEvent(id="cmd-1", output="+added line in file1", is_error=False),
        ToolResultEvent(id="cmd-2", output="+added line in file2", is_error=False),
        ToolResultEvent(id="cmd-3", output="+added line in file3", is_error=False), *_TERMINAL_EVENTS,
    ]
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=False))
    assert "+added line in file1" in plain_text
    assert "+added line in file2" in plain_text
    assert "+added line in file3" in plain_text
    assert "git diff -- file1.py" in plain_text
    assert "git diff -- file2.py" in plain_text
    assert "git diff -- file3.py" in plain_text

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


@pytest.mark.parametrize("pr_number", [None, 7], ids=["branch", "explicit"])
async def test_run_comment_resolves_pr_through_real_cli_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., "RunConfig"], fake_gh: FakeGh,
    pr_number: int | None,
) -> None:
    """Production runner, PR assembly, and posting use one explicit checkout."""
    _two_commit_repo(tmp_path, "api.py", "print('hello')", "print('world')", "feat/test")
    head = _git(tmp_path, "rev-parse", "HEAD")
    fake_gh.serve_pr_view(
        {
            "number": 7, "title": "Change greeting", "body": "", "state": "OPEN", "headRefName": "feat/test",
            "baseRefName": "main", "headRefOid": head, "url": "https://github.test/acme/widgets/pull/7",
            "headRepository": {"name": "widgets", "nameWithOwner": "acme/widgets"},
            "headRepositoryOwner": {"login": "acme"},
        }
    )
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tmp_path)
    exit_code = await run(make_config(tmp_path, output_mode="comment", pr_number=pr_number))
    assert exit_code == 0
    items = json.loads((tmp_path / ".daydream" / "deep" / "merged-items.json").read_text())["items"]
    assert items
    assert any(item.get("file") for item in items)
    assert (tmp_path / ".daydream" / "diff.patch").is_file()
    review_calls = fake_gh.calls("POST", "repos/acme/widgets/pulls/7/reviews")
    assert len(review_calls) == 1
    assert review_calls[0].payload["commit_id"] == head
    assert review_calls[0].payload["comments"][0]["path"] == "api.py"
    assert review_calls[0].payload["comments"][0]["path"] in {item.get("file") for item in items}
    assert fake_gh.process_calls()
    assert {call.cwd for call in fake_gh.process_calls()} == {tmp_path.resolve()}
    list_calls = [
        call for call in fake_gh.process_calls() if call.argv[1:3] == ["pr", "list"]
    ]
    if pr_number is None:
        assert len(list_calls) == 1
        assert "baseRefOid" not in list_calls[0].argv

@pytest.mark.parametrize("pr_number", [None, 7], ids=["branch", "explicit"])
@pytest.mark.parametrize("outcome", ["auth_failure", "schema_failure", "malformed_head", "absence"])
async def test_run_comment_pr_lookup_failure_and_absence_exit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., "RunConfig"], fake_gh: FakeGh,
    pr_number: int | None, outcome: str,
) -> None:
    """Both lookup modes fail closed without attempting a review POST."""
    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")
    if outcome == "malformed_head":
        fake_gh.serve_pr_view({
            "number": 7, "title": "Change greeting", "body": "", "state": "OPEN",
            "headRefName": "feat/test", "baseRefName": "main", "headRefOid": _git(tmp_path, "rev-parse", "HEAD"),
            "url": "https://github.test/acme/widgets/pull/7",
            "headRepository": {}, "headRepositoryOwner": {"login": "acme"},
        })
    elif pr_number is None and outcome == "absence":
        fake_gh.set_response("pr-list", value=[])
    else:
        diagnostic = (
            "authentication required"
            if outcome == "auth_failure"
            else 'Unknown JSON field: "futureCompatibilityFloor"'
        )
        if outcome == "absence":
            diagnostic = (
                "GraphQL: Could not resolve to a PullRequest with the number of 7. "
                "(repository.pullRequest)"
            )
        fake_gh.set_response("pr-view", value={"__error__": diagnostic})
        if pr_number is None:
            fake_gh.set_response("pr-list", value={"__error__": diagnostic})
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tmp_path)
    errors: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr("daydream.pr_review.print_error", lambda _console, _title, message: errors.append(message))
    monkeypatch.setattr("daydream.pr_review.print_warning", lambda _console, message: warnings.append(message))
    exit_code = await run(make_config(tmp_path, output_mode="comment", pr_number=pr_number))
    assert exit_code == 1
    assert fake_gh.calls("POST", "repos/acme/widgets/pulls/7/reviews") == []
    messages = errors + warnings
    if outcome != "absence":
        expected = (
            "authentication required"
            if outcome == "auth_failure"
            else "invalid PR row" if outcome == "malformed_head" else "futureCompatibilityFloor"
        )
        assert any(expected in message for message in messages)
        assert all("No open PR found" not in message for message in messages)
    else:
        expected = "No open PR found" if pr_number is None else "PR #7 not found"
        assert any(expected in message for message in messages)

async def test_run_comment_does_not_prompt_for_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_backend: Callable[[object], object],
    silence_console: Callable[..., None], make_config: Callable[..., 'RunConfig'], fake_gh: FakeGh,
) -> None:
    """--comment mode should never prompt for skill selection."""
    _two_commit_repo(tmp_path, "f.txt", "a", "b", "feat")
    def response(cwd: Path, prompt: str, *_args: Any) -> list[AgentEvent]:
        return [TextEvent(text="Intent: changes f.txt."),
                ResultEvent(structured_output=review_stage_result(prompt, []), continuation=None)]

    install_backend(ScriptedBackend(responder=response, model="mock-model"))
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda _td, **_kwargs: _COMMENT_PR)
    silence_console("daydream.ui")
    silence_console("daydream.runner")
    # Trap: skill selection must never prompt in --comment mode.
    def runner_prompt_trap(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Should not prompt for skill selection in --comment mode")
    monkeypatch.setattr("daydream.run_context._prompt_user", runner_prompt_trap)
    config = make_config(tmp_path, output_mode="comment")
    exit_code = await run(config)
    assert exit_code == 0


async def test_run_comment_submission_failure_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., 'RunConfig'], fake_gh: Any,
) -> None:
    """A gh-process failure propagates through real classification/posting as exit 1."""
    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tmp_path)
    fake_pr = _COMMENT_PR
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda _td, **_kwargs: fake_pr)
    fake_gh.set_response("POST", "repos/acme/widgets/pulls/7/reviews", {"__error__": "HTTP 500"})
    config = make_config(tmp_path, output_mode="comment")
    exit_code = await run(config)
    assert exit_code == 1, "comment mode must fail when the review post fails"

async def test_run_loop_submission_failure_warns_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., 'RunConfig'], fake_gh: Any,
) -> None:
    """Optional loop posting may fail while the approved-post/declined-fix run exits 0."""
    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")
    silence(monkeypatch, prompts=False)
    install_stub_backend(monkeypatch, tmp_path)
    force_interactive(monkeypatch)
    fake_pr = _COMMENT_PR
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda _td, **_kwargs: fake_pr)
    fake_gh.set_response("POST", "repos/acme/widgets/pulls/7/reviews", {"__error__": "HTTP 500"})
    # Approve the PR-post gate but decline the apply-fixes gate so the run ends
    # after the report is written (no fix cycle / commit).
    def _gate_prompt(console: Any, message: str, default: str = "") -> str:
        if "apply fix" in message.lower():
            return "n"
        return "y"
    monkeypatch.setattr("daydream.run_context._prompt_user", _gate_prompt)
    config = make_config(tmp_path, output_mode="loop")
    exit_code = await run(config)
    assert exit_code == 0, "deep loop must warn-and-continue on a failed PR post"
    # The report the review produced is still on disk (the fix gate declined).
    assert (tmp_path / ".review-output.md").exists()


async def test_run_populates_exploration_context(
    monkeypatch: pytest.MonkeyPatch, multi_stack_target: Path, make_config: Callable[..., 'RunConfig'],
) -> None:
    """Populate exploration context before fan-out on a four-file parallel-tier run.

    The review must receive the actual on-disk exploration directory."""
    (multi_stack_target / "extra.py").write_text("VALUE = 2\n")
    _git(multi_stack_target, "add", ".")
    _commit(multi_stack_target, "add extra")
    _silence(monkeypatch)
    backend = _install_stub_backend(monkeypatch, multi_stack_target, enable_exploration=True)
    config = make_config(multi_stack_target, shallow=True)
    exit_code = await run(config)
    assert exit_code == 0
    assert isinstance(config.exploration_context, ExplorationContext)
    exploration = multi_stack_target / ".daydream" / "exploration"
    assert exploration.is_dir()
    review_prompts = [call["prompt"] for call in backend.calls if call.get("output_schema")
                      and "you are reviewing the" in call["prompt"].lower()]
    assert review_prompts
    from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents
    expected_summary = (multi_stack_target / ".daydream/exploration/summary.md").read_text()
    assert all(supporting_contents(prompt)["exploration-summary"] == expected_summary for prompt in review_prompts)

async def test_codex_backend_raises_on_agents(tmp_path: Path) -> None:
    """CodexBackend.execute() refuses agents= with NotImplementedError."""
    backend = CodexBackend(model="fixture-model")
    with pytest.raises(NotImplementedError, match="Codex backend does not support exploration"):
        async for _ in backend.execute(tmp_path, "prompt", agents={"x": object()}):
            pass

async def test_alternative_review_surfaces_confidence_and_rationale(
    tmp_path: Path, make_work: Callable[..., WorkContext]
) -> None:
    """Parsed alternative-review issues retain schema-enforced confidence and rationale."""
    enriched_trust_issue = {
        "id": 1, "title": "t", "description": "x", "recommendation": "y", "severity": "high", "files": ["a.py"],
        "confidence": "HIGH", "rationale": "verified by Convention snake_case_modules", "evidence": "a.py:1",
    }
    def _issue_backend(payload: dict[str, Any]) -> ScriptedBackend:
        return ScriptedBackend(
            events=[TextEvent(text="ok"), ResultEvent(structured_output=payload, continuation=None)],
            model="test-model",
        )
    work = make_work(tmp_path)
    diff_path = tmp_path / "diff.txt"
    diff_path.write_text("diff")
    trust_backend = _issue_backend({"issues": [enriched_trust_issue]})
    trust_issues = await phase_alternative_review(
        trust_backend, work, diff_path, "intent summary", exploration_dir=tmp_path,
    )
    assert trust_issues
    assert "confidence" in trust_issues[0]
    assert "rationale" in trust_issues[0]
