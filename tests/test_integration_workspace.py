"""Workspace base selection through the real runner, Git repositories, and a bare remote."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from daydream import runner
from daydream.run_config import RunConfig
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git as _git, tracked_source_state
from tests.test_github_app_integration import _minimal_backend


@pytest.fixture(autouse=True)
def _silence_runner_ui(silence_console: Callable[..., None]) -> None:
    """Silence Rich panels emitted from the runner so test output stays clean."""
    silence_console("daydream.runner")


async def test_local_review_pins_remote_preferred_base_without_upstream(
    repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """Capture must retain the diff resolver's remote-preferred default base for ``--branch``.

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
    config = RunConfig(target=str(repo), output_mode="review", non_interactive=True, shallow=True,
                       stack="python", archive=False, branch="feature")

    assert await runner.run(config) == 0

    assert backend.calls
    diff = (repo / ".daydream/diff.patch").read_text()
    coverage = json.loads((repo / ".daydream/deep/review-coverage.json").read_text())
    assert coverage["analyzed_revision"]["merge_base_sha"] == remote_tip
    assert "diff --git a/feature.py b/feature.py" in diff
    assert "diff --git a/upstream.py b/upstream.py" not in diff
    assert {path for scope in coverage["planned_scopes"] for path in scope["files"]} == {"feature.py"}
    assert tracked_source_state(repo) == before
