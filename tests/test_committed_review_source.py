"""Real Git checkout transforms must not split committed source identity."""

from pathlib import Path
from typing import Any

import pytest

from daydream import agent, git_ops
from daydream.committed_review import CommittedReviewRequest, run_committed_review
from daydream.git_ops.source import FrozenSourceRejected
from tests.harness.git_helpers import (
    commit,
    git,
    init_repo,
    tracked_source_state,
    write_and_stage,
)


async def test_committed_review_refuses_clean_checkout_with_transformed_native_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reversible filter cannot let native reads contradict canonical source.

    The native fresh-context journey protects dirty-source isolation, but has no
    checkout filters: its detached snapshot and raw Git blobs already agree.
    This regression must stop at capture, before any model is dispatched.
    """
    repo = tmp_path / "source"
    init_repo(repo)
    git(repo, "config", "filter.review.clean", "sed 's/NATIVE/CANONICAL/g'")
    git(repo, "config", "filter.review.smudge", "sed 's/CANONICAL/NATIVE/g'")
    write_and_stage(repo, ".gitattributes", "*.py filter=review\n")
    write_and_stage(repo, "api.py", "VALUE = 'CANONICAL base'\n")
    base = commit(repo, "base")
    write_and_stage(repo, "api.py", "VALUE = 'CANONICAL head'\n")
    head = commit(repo, "change")

    # Git's own clean-status check accepts the transformed worktree. That is
    # precisely why status cannot stand in for raw blob/native byte coherence.
    control = tmp_path / "transformed-control"
    git_ops.worktree_add(repo, control, head, detach=True)
    try:
        assert git(control, "status", "--porcelain") == ""
        assert (control / "api.py").read_bytes() == b"VALUE = 'NATIVE head'\n"
        assert git(control, "show", f"{head}:api.py") == "VALUE = 'CANONICAL head'"
    finally:
        git_ops.worktree_remove(repo, control, force=True)

    (repo / "api.py").write_bytes(b"VALUE = 'DIRTY_USER_BYTES'\n")
    original = tracked_source_state(repo)
    original_worktrees = git(repo, "worktree", "list", "--porcelain")

    async def unexpected_dispatch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("incoherent source reached model dispatch")

    monkeypatch.setattr(agent, "run_agent", unexpected_dispatch)
    with pytest.raises(FrozenSourceRejected, match="api.py.*differs.*captured Git blob"):
        await run_committed_review(CommittedReviewRequest(repo, base))
    assert tracked_source_state(repo) == original
    assert git(repo, "worktree", "list", "--porcelain") == original_worktrees


def test_committed_diff_uses_raw_commit_endpoints_despite_local_diff_drivers(tmp_path: Path) -> None:
    """Committed diff identity excludes both driver output and dirty tracked bytes.

    The generic diff boundary intentionally includes worktree changes; existing
    tests of that API cannot enforce this committed-only source contract.
    """
    repo = tmp_path / "source"
    init_repo(repo)
    write_and_stage(repo, ".gitattributes", "*.py diff=review\n")
    write_and_stage(repo, "api.py", "VALUE = 'base'\n")
    base = commit(repo, "base")
    write_and_stage(repo, "api.py", "VALUE = 'head'\n")
    head = commit(repo, "change")
    (repo / "api.py").write_text("VALUE = 'dirty'\n")
    git(repo, "config", "diff.review.textconv", "echo DRIVER_OUTPUT")
    git(repo, "config", "diff.external", "false")

    patch = git_ops.committed_diff(repo, base, head)

    assert "-VALUE = 'base'" in patch
    assert "+VALUE = 'head'" in patch
    assert "dirty" not in patch and "DRIVER_OUTPUT" not in patch
