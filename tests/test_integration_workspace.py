"""Workspace/branch integration with real Git repositories and bare remotes.

Runner tests cover remote-only branches, private ephemeral cleanup, PR base
selection, branch guards, and mode dispatch. Backend/dispatch seams isolate
review execution while workspace and Git operations remain real.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pytest
from rich.console import Console

from daydream import git_ops, runner
from daydream.backends import ResultEvent, TextEvent
from daydream.flows.engine import BackendFactory
from daydream.github_app import GitHubExecutionInput
from daydream.run_config import RunConfig
from daydream.run_context import current_run_context
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git as _git, tracked_source_state
from tests.test_github_app_integration import _minimal_backend
from tests.test_workspace import _push_origin_commit_via_sidecar

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


def _stub_run_loop_deep(monkeypatch: pytest.MonkeyPatch, on_dispatch: Callable[[Any, Any], None]) -> None:
    """Capture the real workspace/config dispatch and return success."""

    async def _stub(
        work: Any, config: Any, run_artifacts: Any = None, *, run_context: Any, github_execution: GitHubExecutionInput,
        backend_factory: BackendFactory | None,
    ) -> int:
        assert run_context is current_run_context()
        assert backend_factory is None
        on_dispatch(work, config)
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_deep", _stub)


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

    _stub_run_loop_deep(monkeypatch, _capture)

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
    first_origin_sha = _make_feature_branch_on_origin(tmp_path, bare_origin, branch="feat/Y")
    _git(repo_with_origin, "fetch", "origin")
    _git(repo_with_origin, "checkout", "--track", "-b", "feat/Y", "origin/feat/Y")
    assert _git(repo_with_origin, "rev-parse", "HEAD") == first_origin_sha
    _push_origin_commit_via_sidecar(tmp_path, bare_origin, branch="feat/Y")
    origin_sha = _push_origin_commit_via_sidecar(tmp_path, bare_origin, branch="feat/Y")
    assert _git(repo_with_origin, "rev-parse", "origin/feat/Y") == first_origin_sha
    assert first_origin_sha != origin_sha
    assert _git(bare_origin, "rev-list", "--count", f"{first_origin_sha}..{origin_sha}") == "2"
    monkeypatch.setattr("daydream.workspace.git_ops.gh_pr_list_for_branch", lambda _repo, _branch, **_kwargs: [])
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=200, height=25)
    monkeypatch.setattr("daydream.agent.console", rec)

    captured: dict[str, Any] = {}

    def _capture(work: Any, config: Any) -> None:
        captured["head_sha"] = work.head_sha
        captured["is_ephemeral"] = work.is_ephemeral
        captured["repo"] = work.repo
        captured["head_branch"] = work.head_branch
        captured["base_branch"] = work.base_branch
        assert _git(work.repo, "rev-parse", "HEAD") == origin_sha

    _stub_run_loop_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), branch="feat/Y", shallow=True, cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    out = rec.export_text()
    assert "feat/Y is checked out in cwd" in out
    assert "2 commits behind origin/feat/Y" in out
    assert "reviewing origin/feat/Y" in out
    # The ephemeral worktree was checked out at origin/feat/Y, NOT the local SHA.
    assert captured["is_ephemeral"] is True
    assert captured["head_sha"] == origin_sha
    assert str(captured["repo"]).startswith(str(artifact_runtime_root.parent / "workspaces"))

    assert (captured["head_branch"], captured["base_branch"]) == (None, "main")
    assert not captured["repo"].exists()


# --- Test 4: --comment + no open PR ----------------------------------------

@pytest.mark.asyncio
async def test_comment_mode_without_open_pr_rejects_before_dispatch(
    tmp_path: Path, repo_with_origin: Path, bare_origin: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """An origin-only branch without a PR cannot dispatch committed comment review."""
    _make_feature_branch_on_origin(tmp_path, bare_origin, branch="feat/Z")
    fake_gh.set_response("pr-list", value=[])
    captured: dict[str, Any] = {}
    errors: list[str] = []
    monkeypatch.setattr("daydream.runner.print_error", lambda _console, _title, message: errors.append(message))

    def _capture(work: Any, config: Any) -> None:
        captured["is_ephemeral"] = work.is_ephemeral
        captured["head_sha"] = work.head_sha
        captured["output_mode"] = config.output_mode

    _stub_run_loop_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), branch="feat/Z", output_mode="comment", cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 1
    assert captured == {}
    assert any("No target PR found" in message for message in errors)


# --- Test 5: --comment + open PR resolves base from PR ---------------------

@pytest.mark.asyncio
async def test_comment_mode_with_open_pr_uses_pr_base(
    tmp_path: Path, repo_with_origin: Path, bare_origin: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """An origin-only PR pins its head, comparison and distinct base tip before dispatch."""
    # PR base = "develop". Push a develop branch to origin AND create a local
    # tracking branch so ``git merge-base develop HEAD`` (run from inside the
    # ephemeral worktree) can resolve the ref symbolically.
    base_tip = _make_feature_branch_on_origin(tmp_path, bare_origin, branch="develop")
    head = _make_feature_branch_on_origin(tmp_path, bare_origin, branch="feat/W")
    _git(repo_with_origin, "fetch", "origin")
    _git(repo_with_origin, "branch", "develop", "origin/develop")

    comparison = _git(repo_with_origin, "merge-base", "develop", head)
    fake_gh.serve_pr_view({
        "number": 42, "headRefName": "feat/W", "baseRefName": "develop", "headRefOid": head,
        "headRepository": {"name": "widgets", "nameWithOwner": "acme/widgets"},
        "headRepositoryOwner": {"login": "acme"}, "url": "https://github.com/acme/widgets/pull/42",
    })
    fake_gh.set_response("GET", "repos/acme/widgets/pulls/42", {
        "head": {"sha": head}, "base": {"sha": base_tip},
    })
    # Stop _run_loop_deep after open_workspace resolves; assertions below check
    # the resolved WorkContext.
    captured: dict[str, Any] = {}

    def _capture(work: Any, config: Any) -> None:
        captured["base_branch"] = work.base_branch
        captured["is_ephemeral"] = work.is_ephemeral
        captured["head_sha"] = work.head_sha
        captured["pr"] = work.review_pr
        captured["pr_number"] = config.pr_number
        captured["pr_repo"] = config.pr_repo

    _stub_run_loop_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), branch="feat/W", output_mode="comment", cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert captured["base_branch"] == comparison
    assert captured["is_ephemeral"] is True
    assert captured["head_sha"] == head
    assert captured["pr"].base_ref == "develop"
    assert captured["pr"].base_sha == comparison
    assert captured["pr"].pr_base_sha == base_tip
    assert captured["pr_number"] == 42
    assert captured["pr_repo"] == "acme/widgets"
    assert config.pr_number is None
    assert config.pr_repo is None


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

    _stub_run_loop_deep(monkeypatch, _capture)

    config = RunConfig(target=str(repo_with_origin), output_mode="review", cleanup=False,)
    exit_code = await runner.run(config)

    assert exit_code == 0
    # WrongBranchError must NOT have been raised.
    assert captured.get("title") != "Wrong Branch"
    # _run_loop_deep was reached.
    assert routed["base_branch"] == _git(repo_with_origin, "rev-parse", "main")
    assert routed["head_branch"] == "main"


@pytest.mark.parametrize("route,base_kind", [
    ("in-place", "default"), ("in-place", "named"), ("in-place", "sha"),
    ("worktree", "default"), ("branch", "default"),
])
async def test_local_review_pins_remote_preferred_base_without_upstream(
    repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, route: str, base_kind: str,
) -> None:
    """Capture must retain the diff resolver's base policy, including explicit SHAs.

    Resolver unit tests do not cover the runner taking the workspace's different
    merge-base as its accepted source and expanding the actual review inventory.
    """
    repo = repo_with_origin
    original_base = _git(repo, "rev-parse", "main")
    _git(repo, "branch", "--unset-upstream", "main")
    _git(repo, "checkout", "-b", "feature")
    (repo / "upstream.py").write_text("UPSTREAM = 1\n")
    _git(repo, "add", "upstream.py")
    _git(repo, "commit", "-m", "already merged upstream")
    remote_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "push", "origin", "HEAD:main")
    (repo / "feature.py").write_text("FEATURE = 1\n")
    _git(repo, "add", "feature.py")
    _git(repo, "commit", "-m", "feature change")
    _git(repo, "push", "origin", "feature")
    assert _git(repo, "rev-parse", "main") == original_base
    assert _git(repo, "rev-parse", "origin/main") == remote_tip
    fake_gh.set_response("pr-list", value=[])
    backend = _minimal_backend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    before = tracked_source_state(repo)
    base = original_base if base_kind == "sha" else "main" if base_kind == "named" else None
    config = RunConfig(target=str(repo), output_mode="review", non_interactive=True, shallow=True,
                       stack="python", archive=False, base=base, force_worktree=route == "worktree",
                       branch="feature" if route == "branch" else None)

    assert await runner.run(config) == 0

    assert backend.calls
    diff = (repo / ".daydream/diff.patch").read_text()
    coverage = json.loads((repo / ".daydream/deep/review-coverage.json").read_text())
    expected_base = original_base if base_kind == "sha" else remote_tip
    assert coverage["analyzed_revision"]["merge_base_sha"] == expected_base
    assert "diff --git a/feature.py b/feature.py" in diff
    assert ("diff --git a/upstream.py b/upstream.py" in diff) == (base_kind == "sha")
    assigned = {path for scope in coverage["planned_scopes"] for path in scope["files"]}
    assert assigned == ({"feature.py", "upstream.py"} if base_kind == "sha" else {"feature.py"})
    assert tracked_source_state(repo) == before
