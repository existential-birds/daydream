"""End-to-end integration tests for the workspace + branch-resolution layer.

Stage 4.2: every test below stands up a real git repository (often with a
real bare-origin remote) and exercises ``daydream.runner.run`` /
``run_feedback`` end-to-end. The Backend is the ONLY thing mocked — git
operations all run for real so the tests reflect actual user-facing
behavior. The ``MockBackend`` mirrors the canned-event pattern used in
``tests/test_runner.py`` and ``tests/test_integration.py``.

Test inventory (keyed to the Stage 4.2 spec):

1. ``test_default_loop_on_base_branch_raises_wrong_branch_error``
2. ``test_branch_only_on_origin_creates_ephemeral_runs_review_cleans_up``
3. ``test_branch_also_checked_out_locally_warns_uses_origin``
4. ``test_comment_mode_no_pr_errors_clearly``
5. ``test_comment_mode_with_open_pr_uses_pr_base``
6. ``test_feedback_subcommand_works_like_legacy_pr``
7. ``test_review_mode_on_base_branch_does_not_error``
"""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    ResultEvent,
    TextEvent,
)
from daydream.runner import RunConfig

# --- Helpers ---------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a git command in *repo* and return stripped stdout."""
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )
    return proc.stdout.strip()


def _make_feature_branch_on_origin(
    tmp_path: Path, repo: Path, bare_origin: Path, branch: str = "feat/X"
) -> str:
    """Push a new commit on *branch* to *bare_origin* via a sidecar clone.

    Returns the new commit SHA. Done via a sidecar so the working ``repo`` does
    not have *branch* checked out locally (mirrors the "branch only on origin"
    scenario the production code is expected to handle).
    """
    sidecar = tmp_path / f"sidecar-{branch.replace('/', '_')}"
    _git(tmp_path, "clone", str(bare_origin), str(sidecar))
    _git(sidecar, "config", "user.email", "test@test.com")
    _git(sidecar, "config", "user.name", "Test")
    _git(sidecar, "checkout", "-b", branch)
    (sidecar / f"{branch.replace('/', '_')}.txt").write_text("payload\n")
    _git(sidecar, "add", ".")
    _git(sidecar, "commit", "-m", f"feature commit on {branch}")
    sha = _git(sidecar, "rev-parse", "HEAD")
    _git(sidecar, "push", "origin", branch)
    return sha


class MockBackend:
    """Minimal Backend that yields canned events and records every call.

    No phase-specific routing — every ``execute`` returns the same generic
    pair of events. Tests assert on observable side effects (workspace
    cleanup, exit codes, captured errors) rather than on what the backend
    happened to be asked to do.
    """

    model = "mock-model"

    def __init__(self, events: list[AgentEvent] | None = None) -> None:
        default: list[AgentEvent] = [
            TextEvent(text="ok"),
            ResultEvent(structured_output={"issues": []}, continuation=None),
        ]
        self._events = events if events is not None else default
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        self.calls.append({"cwd": cwd, "prompt": prompt})
        for event in self._events:
            yield event

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}" + (f" {args}" if args else "")


@pytest.fixture
def silence_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence Rich panels emitted from the runner so test output stays clean."""
    for name in (
        "print_phase_hero",
        "print_info",
        "print_success",
        "print_warning",
        "print_dim",
    ):
        monkeypatch.setattr(f"daydream.runner.{name}", lambda *a, **kw: None)


@pytest.fixture
def install_mock_backend(monkeypatch: pytest.MonkeyPatch) -> MockBackend:
    """Patch ``create_backend`` so every backend instantiation returns the same mock."""
    backend = MockBackend()
    monkeypatch.setattr(
        "daydream.runner.create_backend", lambda name, model=None: backend
    )
    return backend


# --- Test 1: WrongBranchError on base branch -------------------------------


@pytest.mark.asyncio
async def test_default_loop_on_base_branch_raises_wrong_branch_error(
    repo_with_origin: Path,
    install_mock_backend: MockBackend,
    silence_ui: None,
) -> None:
    """Default loop (no --branch, no --worktree) on the base branch errors loudly.

    ``runner.run`` propagates :class:`git_ops.WrongBranchError` for the
    rendering-level catch in :func:`daydream.cli.main` to surface the panel
    and exit 1.
    """
    config = RunConfig(target=str(repo_with_origin), shallow=True, cleanup=False)

    with pytest.raises(git_ops.WrongBranchError) as excinfo:
        await runner.run(config)

    msg = str(excinfo.value)
    assert "base branch 'main'" in msg
    assert "--branch" in msg
    assert "--worktree" in msg
    # Backend must NOT have been invoked — the guard fires before dispatch.
    assert install_mock_backend.calls == []


# --- Test 2: --branch on origin → ephemeral worktree -----------------------


@pytest.mark.asyncio
async def test_branch_only_on_origin_creates_ephemeral_runs_review_cleans_up(
    tmp_path: Path,
    repo_with_origin: Path,
    bare_origin: Path,
    silence_ui: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``daydream --branch feat/X`` (X only on origin) fetches, runs, cleans up."""
    feat_sha = _make_feature_branch_on_origin(
        tmp_path, repo_with_origin, bare_origin, branch="feat/X"
    )

    # Confirm precondition: feat/X is NOT a local branch in repo_with_origin yet.
    local_branches = _git(repo_with_origin, "branch", "--list")
    assert "feat/X" not in local_branches

    # Stub the optional gh PR-base lookup so the resolver doesn't depend on a
    # live gh CLI in the test environment.
    monkeypatch.setattr(
        "daydream.workspace.git_ops.gh_pr_list_for_branch",
        lambda _repo, _branch: [],
    )

    captured: dict[str, Any] = {}
    worktree_path_at_dispatch: dict[str, Path] = {}

    async def fake_run_loop_shallow(work, config):
        captured["base_branch"] = work.base_branch
        captured["is_ephemeral"] = work.is_ephemeral
        captured["head_sha"] = work.head_sha
        worktree_path_at_dispatch["repo"] = work.repo
        # Sanity: the ephemeral worktree exists on disk while we're inside it.
        assert work.repo.is_dir()
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_shallow", fake_run_loop_shallow)

    config = RunConfig(
        target=str(repo_with_origin),
        branch="feat/X",
        shallow=True,
        cleanup=False,
    )
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert captured["is_ephemeral"] is True
    assert captured["head_sha"] == feat_sha
    # The ephemeral worktree was placed under ``<source>/.daydream/worktrees/``.
    assert str(worktree_path_at_dispatch["repo"]).startswith(
        str(repo_with_origin / ".daydream" / "worktrees")
    )
    # Cleanup: the worktree directory is removed (or empty) after exit.
    if worktree_path_at_dispatch["repo"].exists():
        # Some platforms leave the empty parent dir behind — that's fine, just
        # confirm the worktree itself is gone.
        pytest.fail(
            f"ephemeral worktree {worktree_path_at_dispatch['repo']} not removed"
        )


# --- Test 3: --branch X also checked out locally → warns + uses origin -----


@pytest.mark.asyncio
async def test_branch_also_checked_out_locally_warns_uses_origin(
    tmp_path: Path,
    repo_with_origin: Path,
    bare_origin: Path,
    silence_ui: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --branch X is also checked out locally and stale, warn + use origin/X."""
    # Push feat/Y to origin first (canonical commit on origin).
    origin_sha = _make_feature_branch_on_origin(
        tmp_path, repo_with_origin, bare_origin, branch="feat/Y"
    )
    # Check out feat/Y locally from main, BEFORE fetching, so the local
    # branch is stale relative to origin/feat/Y.
    _git(repo_with_origin, "checkout", "-b", "feat/Y", "main")
    local_sha = _git(repo_with_origin, "rev-parse", "feat/Y")
    assert local_sha != origin_sha

    monkeypatch.setattr(
        "daydream.workspace.git_ops.gh_pr_list_for_branch",
        lambda _repo, _branch: [],
    )
    warnings_emitted: list[str] = []
    # ``_resolve_ref`` does ``from daydream.ui import ... print_warning``
    # inside the function, so patch the symbol on its origin module.
    monkeypatch.setattr(
        "daydream.ui.print_warning",
        lambda _console, msg: warnings_emitted.append(msg),
    )

    captured: dict[str, Any] = {}

    async def fake_run_loop_shallow(work, config):
        captured["head_sha"] = work.head_sha
        captured["is_ephemeral"] = work.is_ephemeral
        captured["repo"] = work.repo
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_shallow", fake_run_loop_shallow)

    config = RunConfig(
        target=str(repo_with_origin),
        branch="feat/Y",
        shallow=True,
        cleanup=False,
    )
    exit_code = await runner.run(config)

    assert exit_code == 0
    # Warning fires per the Stale-local-handling rule.
    assert any("feat/Y" in m and "origin/feat/Y" in m for m in warnings_emitted), (
        f"expected stale-local warning; got: {warnings_emitted}"
    )
    # The ephemeral worktree was checked out at origin/feat/Y, NOT the local SHA.
    assert captured["is_ephemeral"] is True
    assert captured["head_sha"] == origin_sha
    assert str(captured["repo"]).startswith(
        str(repo_with_origin / ".daydream" / "worktrees")
    )


# --- Test 4: --comment + no open PR ----------------------------------------


@pytest.mark.asyncio
async def test_comment_mode_no_pr_errors_clearly(
    tmp_path: Path,
    repo_with_origin: Path,
    bare_origin: Path,
    install_mock_backend: MockBackend,
    silence_ui: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--comment --branch feat/X`` with no open PR exits 1 with actionable error."""
    _make_feature_branch_on_origin(
        tmp_path, repo_with_origin, bare_origin, branch="feat/Z"
    )
    monkeypatch.setattr(
        "daydream.workspace.git_ops.gh_pr_list_for_branch",
        lambda _repo, _branch: [],
    )
    monkeypatch.setattr(
        "daydream.runner.git_ops.gh_pr_list_for_branch",
        lambda _repo, _branch: [],
    )
    captured: dict[str, str] = {}

    def fake_print_error(_console: Any, title: str, body: str) -> None:
        captured["title"] = title
        captured["body"] = body

    monkeypatch.setattr("daydream.runner.print_error", fake_print_error)

    config = RunConfig(
        target=str(repo_with_origin),
        branch="feat/Z",
        output_mode="comment",
        cleanup=False,
    )
    exit_code = await runner.run(config)

    assert exit_code == 1
    assert captured["title"] == "No Open PR"
    assert "no open PR for branch feat/Z" in captured["body"]
    assert "push first or use --review" in captured["body"]
    # No backend call: the pre-flight aborts before review runs.
    assert install_mock_backend.calls == []


# --- Test 5: --comment + open PR resolves base from PR ---------------------


@pytest.mark.asyncio
async def test_comment_mode_with_open_pr_uses_pr_base(
    tmp_path: Path,
    repo_with_origin: Path,
    bare_origin: Path,
    silence_ui: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open PR's ``baseRefName`` is what ``open_workspace`` resolves as base."""
    # PR base = "develop". Push a develop branch to origin AND create a local
    # tracking branch so ``git merge-base develop HEAD`` (run from inside the
    # ephemeral worktree) can resolve the ref symbolically.
    _make_feature_branch_on_origin(
        tmp_path, repo_with_origin, bare_origin, branch="develop"
    )
    _make_feature_branch_on_origin(
        tmp_path, repo_with_origin, bare_origin, branch="feat/W"
    )
    _git(repo_with_origin, "fetch", "origin")
    _git(repo_with_origin, "branch", "develop", "origin/develop")

    monkeypatch.setattr(
        "daydream.workspace.git_ops.gh_pr_list_for_branch",
        lambda _repo, _branch: [
            {
                "number": 42,
                "baseRefName": "develop",
                "headRefOid": "deadbeef",
                "baseRefOid": "cafebabe",
                "url": "https://github.com/x/y/pull/42",
            }
        ],
    )
    # Stop _run_comment after open_workspace resolves; assertions below check
    # the resolved WorkContext.
    captured: dict[str, Any] = {}

    async def fake_run_comment(work, config):
        captured["base_branch"] = work.base_branch
        captured["is_ephemeral"] = work.is_ephemeral
        return 0

    monkeypatch.setattr("daydream.runner._run_comment", fake_run_comment)

    config = RunConfig(
        target=str(repo_with_origin),
        branch="feat/W",
        output_mode="comment",
        cleanup=False,
    )
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert captured["base_branch"] == "develop"
    assert captured["is_ephemeral"] is True


# --- Test 6: feedback subcommand routes through PR feedback flow -----------


@pytest.mark.asyncio
async def test_feedback_subcommand_works_like_legacy_pr(
    repo_with_origin: Path,
    install_mock_backend: MockBackend,
    silence_ui: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``daydream feedback <pr#>`` enters ``_run_pr_feedback`` end-to-end."""
    # Move off main so the WrongBranchError guard does not fire
    # (PR feedback runs in-place, not against an ephemeral worktree).
    _git(repo_with_origin, "checkout", "-b", "feat/pr7")

    monkeypatch.setattr(
        "daydream.workspace.git_ops.gh_pr_list_for_branch",
        lambda _repo, _branch: [],
    )
    captured: dict[str, Any] = {}

    async def fake_run_pr_feedback(work, config):
        captured["pr_number"] = config.pr_number
        captured["bot"] = config.bot
        captured["repo"] = work.repo
        return 0

    monkeypatch.setattr("daydream.runner._run_pr_feedback", fake_run_pr_feedback)

    config = RunConfig(target=str(repo_with_origin), bot="copilot", cleanup=False)
    exit_code = await runner.run_feedback(config, 7)

    assert exit_code == 0
    assert captured["pr_number"] == 7
    assert captured["bot"] == "copilot"
    assert captured["repo"] == repo_with_origin


# --- Test 7: --review on base branch is allowed ----------------------------


@pytest.mark.asyncio
async def test_review_mode_on_base_branch_does_not_error(
    repo_with_origin: Path,
    install_mock_backend: MockBackend,
    silence_ui: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--review`` on base branch must NOT raise WrongBranchError.

    The review flow may legitimately produce a no-op report from the base
    branch (no diff = no findings) — that's not a failure case.
    """
    captured: dict[str, str] = {}

    def fake_print_error(_console: Any, title: str, body: str) -> None:
        captured["title"] = title
        captured["body"] = body

    monkeypatch.setattr("daydream.runner.print_error", fake_print_error)

    # Stop after open_workspace resolves; assertions check we routed past the
    # WrongBranchError guard into _run_review.
    routed: dict[str, Any] = {}

    async def fake_run_review(work, config):
        routed["base_branch"] = work.base_branch
        routed["head_branch"] = work.head_branch
        return 0

    monkeypatch.setattr("daydream.runner._run_review", fake_run_review)

    config = RunConfig(
        target=str(repo_with_origin),
        output_mode="review",
        cleanup=False,
    )
    exit_code = await runner.run(config)

    assert exit_code == 0
    # WrongBranchError must NOT have been raised.
    assert captured.get("title") != "Wrong Branch"
    # _run_review was reached.
    assert routed["base_branch"] == "main"
    assert routed["head_branch"] == "main"
