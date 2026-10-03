"""Native accepted repair sessions for focused phase tests."""

from typing import Any

from daydream import git_ops
from daydream.deep.fix_state import EvidenceKey, FixCycleState, RepairCandidate, RetainedTreeSnapshot
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.run_config import RunConfig
from daydream.test_execution import TestRecipe
from daydream.workspace import WorkContext
from tests.harness.git_helpers import commit, git, init_repo


def repair_session(
    work: WorkContext,
    *,
    paths: frozenset[str] = frozenset(),
    config: Any = None,
    recipe: TestRecipe | None = None,
    session_id: str = "s",
) -> FixCycleState:
    """Capture a real baseline and an explicitly authorized candidate."""
    if not (work.repo / ".git").exists():
        init_repo(work.repo)
    if not git(work.repo, "rev-parse", "--verify", "HEAD", check=False):
        (work.repo / ".repair-fixture").write_text("repair baseline\n")
        git(work.repo, "add", ".repair-fixture")
        commit(work.repo, "test: repair baseline")
    head = git_ops.head_sha(work.repo)
    session = FixCycleState(
        work=work,
        config=config or RunConfig(target=str(work.repo)),
        recipe=recipe,
        session_id=session_id,
        stable_ref=head,
        stable_head=head,
        initial_index=git_ops.snapshot_index(work.repo),
        preexisting_untracked={},
        preexisting_gitlinks=(),
        footprint=AuthorizedFixFootprint(run_allowed_paths=paths, policy_revision=1),
    )
    snapshot = RetainedTreeSnapshot(
        paths=paths,
        states=git_ops.snapshot_worktree_paths(work.repo, paths),
        tree_key=session.capture_key(),
        verifier_patch="",
        recommended_patch=b"",
    )
    session.candidate = RepairCandidate(snapshot, EvidenceKey(snapshot.tree_key, 1), {})
    return session
