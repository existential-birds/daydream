"""Workspace/branch integration with real Git repositories and bare remotes.

Runner tests cover remote-only branches, private ephemeral cleanup, PR base
selection, branch guards, and mode dispatch. Backend/dispatch seams isolate
review execution while workspace and Git operations remain real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from daydream import git_ops, runner
from daydream.backends import BackendExecutionInput, ResultEvent, TextEvent
from daydream.github_app import GitHubExecutionInput
from daydream.run_config import RunConfig
from daydream.run_context import current_run_context
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import git as _git

# --- Helpers ---------------------------------------------------------------


def _make_feature_branch_on_origin(tmp_path: Path, bare_origin: Path, branch: str = "feat/X") -> str:
    """Push a branch from a sidecar clone and return its SHA, leaving the main
    working repository without a local checkout of that branch.
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


def _stub_run_deep(monkeypatch: pytest.MonkeyPatch, on_dispatch: Callable[[Any, Any], None]) -> None:
    """Capture the real workspace/config dispatch and return success."""

    async def _stub(
        config: RunConfig, work: WorkContext, *, run_artifacts: Any, run_context: Any,
        github_execution: GitHubExecutionInput, backend_execution: BackendExecutionInput | None, allow_standalone: bool,
    ) -> int:
        assert run_context is current_run_context()
        assert backend_execution is None
        assert not allow_standalone
        on_dispatch(work, config)
        return 0

    monkeypatch.setattr("daydream.deep.orchestrator.run_deep", _stub)


@pytest.fixture(autouse=True)
def _silence_runner_ui(silence_console: Callable[..., None]) -> None:
    """Silence Rich panels emitted from the runner so test output stays clean."""
    silence_console("daydream.runner")


@pytest.fixture
def install_mock_backend(monkeypatch: pytest.MonkeyPatch) -> ScriptedBackend:
    """Patch ``create_backend`` so every backend instantiation returns the same mock."""
    backend = ScriptedBackend(
        events=[TextEvent(text="ok"), ResultEvent(structured_output={"issues": []}, continuation=None)],
        model="mock-model",
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    return backend


# --- Test 1: WrongBranchError on base branch -------------------------------

@pytest.mark.asyncio
async def test_default_loop_on_base_branch_raises_wrong_branch_error(
    repo_with_origin: Path, install_mock_backend: ScriptedBackend,
) -> None:
    """Default loop on the base branch propagates WrongBranchError for CLI rendering."""
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
    tmp_path: Path, repo_with_origin: Path, bare_origin: Path, artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``daydream --branch feat/X`` (X only on origin) fetches, runs, cleans up."""
    feat_sha = _make_feature_branch_on_origin(tmp_path, bare_origin, branch="feat/X")

    # Confirm precondition: feat/X is NOT a local branch in repo_with_origin yet.
    local_branches = _git(repo_with_origin, "branch", "--list")
    assert "feat/X" not in local_branches

    # Stub the optional gh PR-base lookup so the resolver doesn't depend on a
    # live gh CLI in the test environment.
    monkeypatch.setattr("daydream.workspace.git_ops.gh_pr_list_for_branch", lambda _repo, _branch, **_kwargs: [],)

    captured: dict[str, Any] = {}
    worktree_path_at_dispatch: dict[str, Path] = {}

    def _capture(work: Any, config: Any) -> None:
        captured["base_branch"] = work.base_branch
        captured["is_ephemeral"] = work.is_ephemeral
        captured["head_sha"] = work.head_sha
        worktree_path_at_dispatch["repo"] = work.repo
        # Sanity: the ephemeral worktree exists on disk while we're inside it.
        assert work.repo.is_dir()

    _stub_run_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), branch="feat/X", shallow=True, cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert captured["is_ephemeral"] is True
    assert captured["head_sha"] == feat_sha
    # The ephemeral worktree is a source-owned private operational worktree
    # under the artifact-private base (``<base>/workspaces/<key>/operational``),
    # disjoint from the repo-eligible tree — never under the public source.
    private_base = artifact_runtime_root.parent
    assert str(worktree_path_at_dispatch["repo"]).startswith(str(private_base / "workspaces"))
    assert not worktree_path_at_dispatch["repo"].is_relative_to(repo_with_origin)
    # Cleanup: the worktree directory is removed after exit.
    if worktree_path_at_dispatch["repo"].exists():
        pytest.fail(f"ephemeral worktree {worktree_path_at_dispatch['repo']} not removed")


# --- Test 3: --branch X also checked out locally → warns + uses origin -----

@pytest.mark.asyncio
async def test_branch_also_checked_out_locally_warns_uses_origin(
    tmp_path: Path, repo_with_origin: Path, bare_origin: Path, artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --branch X is also checked out locally and stale, warn + use origin/X."""
    # Push feat/Y to origin first (canonical commit on origin).
    origin_sha = _make_feature_branch_on_origin(tmp_path, bare_origin, branch="feat/Y")
    # Check out feat/Y locally from main, BEFORE fetching, so the local
    # branch is stale relative to origin/feat/Y.
    _git(repo_with_origin, "checkout", "-b", "feat/Y", "main")
    local_sha = _git(repo_with_origin, "rev-parse", "feat/Y")
    assert local_sha != origin_sha

    monkeypatch.setattr("daydream.workspace.git_ops.gh_pr_list_for_branch", lambda _repo, _branch, **_kwargs: [],)
    warnings_emitted: list[str] = []
    # ``_resolve_ref`` does ``from daydream.ui import ... print_warning``
    # inside the function, so patch the symbol on its origin module.
    monkeypatch.setattr("daydream.ui.print_warning", lambda _console, msg: warnings_emitted.append(msg),)

    captured: dict[str, Any] = {}

    def _capture(work: Any, config: Any) -> None:
        captured["head_sha"] = work.head_sha
        captured["is_ephemeral"] = work.is_ephemeral
        captured["repo"] = work.repo

    _stub_run_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), branch="feat/Y", shallow=True, cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    # Warning fires per the Stale-local-handling rule.
    assert any("feat/Y" in m and "origin/feat/Y" in m for m in warnings_emitted), (
        f"expected stale-local warning; got: {warnings_emitted}"
    )
    # The ephemeral worktree was checked out at origin/feat/Y, NOT the local SHA.
    assert captured["is_ephemeral"] is True
    assert captured["head_sha"] == origin_sha
    assert str(captured["repo"]).startswith(str(artifact_runtime_root.parent / "workspaces"))


# --- Test 4: --comment + no open PR ----------------------------------------

@pytest.mark.asyncio
async def test_comment_mode_without_open_pr_runs_deep_flow(
    tmp_path: Path, repo_with_origin: Path, bare_origin: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comment mode resolves an origin-only branch and dispatches deep review without a PR."""
    _make_feature_branch_on_origin(tmp_path, bare_origin, branch="feat/Z")
    monkeypatch.setattr("daydream.workspace.git_ops.gh_pr_list_for_branch", lambda _repo, _branch, **_kwargs: [],)
    monkeypatch.setattr("daydream.runner.git_ops.gh_pr_list_for_branch", lambda _repo, _branch, **_kwargs: [],)
    captured: dict[str, Any] = {}

    def _capture(work: Any, config: Any) -> None:
        captured["is_ephemeral"] = work.is_ephemeral
        captured["head_sha"] = work.head_sha
        captured["output_mode"] = config.output_mode

    _stub_run_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), branch="feat/Z", output_mode="comment", cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert captured["is_ephemeral"] is True
    assert captured["output_mode"] == "comment"


# --- Test 5: --comment + open PR resolves base from PR ---------------------

@pytest.mark.asyncio
async def test_comment_mode_with_open_pr_uses_pr_base(
    tmp_path: Path, repo_with_origin: Path, bare_origin: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open PR's ``baseRefName`` is what ``open_workspace`` resolves as base."""
    # PR base = "develop". Push a develop branch to origin AND create a local
    # tracking branch so ``git merge-base develop HEAD`` (run from inside the
    # ephemeral worktree) can resolve the ref symbolically.
    _make_feature_branch_on_origin(tmp_path, bare_origin, branch="develop")
    _make_feature_branch_on_origin(tmp_path, bare_origin, branch="feat/W")
    _git(repo_with_origin, "fetch", "origin")
    _git(repo_with_origin, "branch", "develop", "origin/develop")

    monkeypatch.setattr("daydream.workspace.git_ops.gh_pr_list_for_branch",
        lambda _repo, _branch, **_kwargs: [{
                "number": 42, "baseRefName": "develop", "headRefOid": "deadbeef", "baseRefOid": "cafebabe",
                "url": "https://github.com/x/y/pull/42",
            }
        ],
    )
    # Stop run_deep after open_workspace resolves; assertions below check
    # the resolved WorkContext.
    captured: dict[str, Any] = {}

    def _capture(work: Any, config: Any) -> None:
        captured["base_branch"] = work.base_branch
        captured["is_ephemeral"] = work.is_ephemeral

    _stub_run_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), branch="feat/W", output_mode="comment", cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert captured["base_branch"] == "develop"
    assert captured["is_ephemeral"] is True


# --- Test 6: --review on base branch is allowed ----------------------------

@pytest.mark.asyncio
async def test_review_mode_on_base_branch_does_not_error(
    repo_with_origin: Path, install_mock_backend: ScriptedBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review mode permits a no-diff base-branch report without WrongBranchError."""
    captured: dict[str, str] = {}

    def fake_print_error(_console: Any, title: str, body: str) -> None:
        captured["title"] = title
        captured["body"] = body

    monkeypatch.setattr("daydream.runner.print_error", fake_print_error)

    # Stop after open_workspace resolves; assertions check we routed past the
    # WrongBranchError guard into the deep flow.
    routed: dict[str, Any] = {}

    def _capture(work: Any, config: Any) -> None:
        routed["base_branch"] = work.base_branch
        routed["head_branch"] = work.head_branch

    _stub_run_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), output_mode="review", cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    # WrongBranchError must NOT have been raised.
    assert captured.get("title") != "Wrong Branch"
    # run_deep was reached.
    assert routed["base_branch"] == "main"
    assert routed["head_branch"] == "main"
