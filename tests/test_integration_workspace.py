"""Workspace/branch integration with real Git repositories and bare remotes.

Runner tests cover remote-only branches, private ephemeral cleanup, PR base
selection, branch guards, and mode dispatch. Backend/dispatch seams isolate
review execution while workspace and Git operations remain real.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from daydream import runner
from daydream.flows.engine import BackendFactory
from daydream.github_app import GitHubExecutionInput
from daydream.run_config import RunConfig
from daydream.run_context import current_run_context
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git as _git, tracked_source_state
from tests.test_github_app_integration import _minimal_backend

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


# --- Test 1: WrongBranchError on base branch -------------------------------

# --- Test 2: --branch on origin → ephemeral worktree -----------------------

# --- Test 3: --branch X also checked out locally → warns + uses origin -----

# --- Test 4: --comment + no open PR ----------------------------------------

# --- Test 5: --comment + open PR resolves base from PR ---------------------


# --- Test 6: --review on base branch is allowed ----------------------------

@pytest.mark.parametrize("route,base_kind", [

    ("branch", "default"),
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
