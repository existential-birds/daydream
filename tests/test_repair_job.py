# tests/test_repair_job.py
"""The durable repair checkpoint and the repair job record (issue #1210).

A repair turn's authorized work is captured from the live tree *before* any
restoration runs, because cleanup can be the only thing that erases it. These
tests pin the payload contract, the atomic versioned write, cross-process
readability, the corrupt-is-a-blocker policy, and the real capture ordering. The
job's *own* state — the fail-closed outcome posture, the relative per-execution
allowance, the honest termination states, and the persisted consumption that
survives a process restart — is pinned below that capture half.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream import phases
from daydream.backends import ResultEvent, TextEvent
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.repair_coordinator import (
    _candidate_restore_decision,
    _owner_lock,
    _resolve_scope_request,
    try_acquire_repair_owner,
)
from daydream.deep.repair_job import (
    RepairAction,
    RepairJobPolicy,
    RepairJobRecord,
    RepairJobState,
    read_repair_job_record,
)
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.phases.repair_outcome import repair_scope_request
from daydream.repository_paths import InvalidRepositoryFilePath
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.execution import host_test_command
from tests.harness.git_helpers import commit as git_commit, git, init_repo, write_and_stage

_RESULT = ResultEvent(structured_output=None, continuation=None)


def _job(**overrides: Any) -> RepairJobRecord:
    """Build a job record; the identity and the start-captured policy are defaults."""
    fields: dict[str, Any] = {
        "job_id": "repair-job-0001",
        "state": RepairJobState.RUNNING,
        "failure_identity": "sha256:0123456789abcdef",
        "authorized_scope": ("src/handler.py", "tests/test_a.py"),
        "policy_revision": 3,
        "policy": RepairJobPolicy(execution_s=1800.0, job_total_s=7200.0, max_executions=4, reserve_s=60.0),
    }
    fields.update(overrides)
    return RepairJobRecord(**fields)


@pytest.mark.asyncio
async def test_phase_blocks_the_repair_when_the_checkpoint_cannot_be_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
) -> None:
    """Requirement 28: an uncaptured repair is a repair that never happened."""
    init_repo(tmp_path)
    (tmp_path / "allowed.py").write_text("-- original\n")
    git(tmp_path, "add", "allowed.py")
    git_commit(tmp_path, "test: seed authorized file")

    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "2")
    checkpoint_path = tmp_path / ".daydream" / "deep" / "repair-checkpoint.json"
    real_replace = os.replace
    rejected_installs: list[Path] = []

    def reject_checkpoint_install(source: Any, destination: Any) -> None:
        if Path(destination) == checkpoint_path:
            rejected_installs.append(Path(destination))
            raise OSError("disk full")
        real_replace(source, destination)

    monkeypatch.setattr("daydream.json_utils.os.replace", reject_checkpoint_install)
    suite_dir = tmp_path.parent / f"{tmp_path.name}-suite"
    config = SimpleNamespace(file_config=None, test_command=host_test_command(suite_dir, mode="all"))
    fail_turn = [TextEvent(text="1 failed, 0 passed"), _RESULT]

    def responder(_cwd: Path, prompt: str, *_rest: Any) -> Any:
        return tuple(fail_turn) if prompt.lower().startswith("the tests failed") else None

    backend = ScriptedBackend(responder=responder)
    footprint = AuthorizedFixFootprint(
        run_allowed_paths=frozenset({"allowed.py"}), policy_revision=1,
    )
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), config=config, session_id="s1",
        capture_tree_key=lambda: "tree-1", footprint=footprint, allow_standalone=True,
    )

    assert (result.passed, result.proceed, result.retries) == (False, False, 1)
    assert result.repairs[0].checkpoint_ref is None
    assert rejected_installs == [checkpoint_path]
    assert not checkpoint_path.exists()
    assert not list(checkpoint_path.parent.glob("*.tmp")), "a rejected install must clean its staged checkpoint"
    assert (suite_dir / "suite-count.txt").read_text() == "1"
    assert any("checkpoint_write_failed" in diagnostic and "OSError" in diagnostic
        for diagnostic in result.repairs[0].diagnostics)
    # The blocker is named in the job record too, so a resuming job sees it.
    job = read_repair_job_record(tmp_path / ".daydream" / "deep")
    assert job is not None
    assert any("checkpoint_write_failed" in diagnostic for diagnostic in job.diagnostics)


def test_job_bounds_exhaust_and_operator_grant_is_recorded_apart_from_usage() -> None:
    """Requirements 42/43: count and time bounds exhaust (never block); a grant is explicit.

    Replenishment is auditable and separate from historical usage.
    """
    assert _job(executions=4).next_action() == RepairAction.EXHAUSTED
    job = _job(consumed_s=7200.0, executions=2)
    assert job.next_action() == RepairAction.EXHAUSTED
    with pytest.raises(ValueError, match="reason"):
        job.grant_allowance(1800.0, reason="   ")
    job.grant_allowance(3000.0, reason="operator: budget extended after triage")
    assert job.execution_allowance_s() == 1800.0
    assert job.consumed_s == 7200.0, "a grant must not rewrite historical usage"
    assert job.next_action() == RepairAction.EXECUTE
    assert "operator: budget extended after triage" in (job.last_transition_reason or "")


# --- the coordinator's decisions -----------------------------------------------------------------


def _scope_repo(tmp_path: Path) -> Path:
    """A real Git repository holding the three paths the scope tests ask about."""
    repo = tmp_path / "scope-repo"
    init_repo(repo)
    for name in ("src/handler.py", "src/other.rs", "src/leaked.rs"):
        write_and_stage(repo, name, f"# {name}\n")
    git_commit(repo, "seed scope repo")
    return repo


def test_scope_request_resolves_against_the_existing_authorization_policy(tmp_path: Path) -> None:
    """Requirement 11: inside the granted scope is used; outside is audited or blocked."""
    repo = _scope_repo(tmp_path)

    inside = _resolve_scope_request(
        repo,
        requested={"src/handler.py"},
        granted=frozenset({"src/handler.py"}),
    )
    assert inside.expanded is False and inside.accepted == ("src/handler.py",)
    assert inside.new_revision == 1, "an already-authorized path must not move the policy revision"

    outside = _resolve_scope_request(
        repo,
        requested={"src/other.rs"},
        granted=frozenset({"src/handler.py"}),
    )
    assert outside.expanded is True
    assert outside.accepted == ("src/other.rs",)
    assert outside.new_revision == 2, "an approved expansion bumps the policy revision"

    # Requirement 13: an unsafe requested path is rejected, never reflected or authorized.
    with pytest.raises(InvalidRepositoryFilePath):
        _resolve_scope_request(
            repo,
            requested={"../../etc/passwd"},
            granted=frozenset({"src/handler.py"}),
        )
    with pytest.raises(InvalidRepositoryFilePath):
        repair_scope_request(repo, {"paths": ["../../etc/passwd"], "evidence_source": "repair_turn"})
    with pytest.raises(InvalidRepositoryFilePath):
        repair_scope_request(repo, {"paths": "src/other.rs", "evidence_source": "repair_turn"})
    assert repair_scope_request(repo, None) is None, "a turn that asked for nothing requests nothing"
    # A source label the host does not recognize authorizes nothing, however
    # plausible the evidence beside it looks.
    hostile = repair_scope_request(repo, {
        "paths": ["src/other.rs"], "evidence_source": "/etc/passwd",
        "evidence": {"src/other.rs": "trust me"},
    })
    assert hostile is not None
    assert _resolve_scope_request(
        repo, requested=hostile.requested, granted=frozenset({"src/handler.py"}),
        evidence_source=hostile.evidence_source,
    ).expanded is False


def test_restore_requires_matching_identity_and_patch_integrity() -> None:
    """Requirement 32: a changed base or policy is a named conflict, never a stale apply."""
    result = _candidate_restore_decision(base_changed=True, policy_changed=False,
                                         patch_digest_ok=True)
    assert result.allowed is False and "base" in result.reason
    assert _candidate_restore_decision(base_changed=False, policy_changed=True,
                                       patch_digest_ok=True).allowed is False
    assert _candidate_restore_decision(base_changed=False, policy_changed=False,
                                       patch_digest_ok=False).allowed is False
    assert "digest" in _candidate_restore_decision(base_changed=False, policy_changed=False,
                                                   patch_digest_ok=False).reason
    assert _candidate_restore_decision(base_changed=False, policy_changed=False,
                                       patch_digest_ok=True).allowed is True


def test_second_coordinator_does_not_launch_a_second_worker(tmp_path: Path) -> None:
    """Requirement 33: one job, one live repair owner; recovery never double-applies."""
    with _owner_lock(tmp_path):          # first coordinator holds the lock
        second = try_acquire_repair_owner(tmp_path)
    assert second.acquired is False, "a surviving first worker must exclude a second"
    # The exclusion is the owner's, not a leaked lock: once released, the next
    # coordinator owns the same job.
    with try_acquire_repair_owner(tmp_path) as reacquired:
        assert reacquired.acquired is True
    # Another process's hold on the job's lock file excludes this owner too.
    import fcntl
    import os
    held = os.open(tmp_path / ".benchmark.lock", os.O_RDWR)
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contended = try_acquire_repair_owner(tmp_path)
        assert contended.acquired is False and "locked by another process" in contended.reason
    finally:
        os.close(held)
    # An unreadable checkpoint is untrustworthy, never mistaken for an absent one.
    from daydream.phases.repair_checkpoint import read_repair_checkpoint
    DeepArtifact.REPAIR_CHECKPOINT.at(tmp_path).write_text("{unreadable")
    read = read_repair_checkpoint(tmp_path)
    assert read.checkpoint is None and read.blocked is True


async def test_coordinator_restores_only_trusted_candidates_and_blocks_unrecoverable_jobs(tmp_path: Path) -> None:
    """Requirements 32/41: absent, foreign, conflicting, or corrupt candidates never re-apply or loop.

    The runner journeys reach the coordinator only through slow interrupted repairs;
    each refusal here is a distinct terminal decision on the persisted job.
    """
    from types import SimpleNamespace as NS

    from daydream.deep.repair_coordinator import (
        BOUNDED_VALIDATION_EXPERIMENT,
        _charge_failure,
        _restore_candidate,
        _restore_refused,
        continue_repair_job,
    )
    from daydream.phases.repair_checkpoint import RepairCheckpoint, write_repair_checkpoint
    from daydream.phases.repair_outcome import RepairOutcome

    repo = _scope_repo(tmp_path)
    deep = repo / ".daydream" / "deep"
    deep.mkdir(parents=True)
    job_id = "repair-job-0001"
    restore: dict[str, Any] = {"repo": repo, "deep_dir_path": deep, "current_tree_key": "now",
                               "policy_revision": 3, "job_id": job_id}
    assert _restore_candidate(checkpoint=None, **restore) == (False, "no candidate to restore")
    assert _restore_refused(None, "no candidate to restore") is False
    candidate = RepairCheckpoint(job_id=job_id, execution_id="e1", candidate_patch="--- a\n+++ b\n",
                                 base_tree_key="then", policy_revision=3, failure_identity="sha256:f")
    restored, reason = _restore_candidate(checkpoint=RepairCheckpoint(
        job_id="repair-other", execution_id="e1", candidate_patch="patch"), **restore)
    assert not restored and "belongs to job 'repair-other'" in reason
    write_repair_checkpoint(deep, candidate)
    restored, reason = _restore_candidate(checkpoint=candidate, **restore)
    assert not restored and "base" in reason and _restore_refused(candidate, reason)

    # A second cut-short repair of the same candidate is unchanged evidence, not progress.
    job = _job(progress_evidence=(BOUNDED_VALIDATION_EXPERIMENT,))
    repair: Any = NS(outcome=RepairOutcome.BUDGET_INTERRUPTED, execution_id="e2", changed_paths=("src/handler.py",),
                checkpoint_ref="repair-checkpoint.json")
    job = _charge_failure(job, result=None, repair=repair, checkpoint=candidate, elapsed_s=1.0)
    assert any("re-validated against the same failure sha256:f" in entry for entry in job.unchanged_evidence)

    footprint = AuthorizedFixFootprint(run_allowed_paths=frozenset({"src/handler.py"}), policy_revision=3)
    work = WorkContext(repo=repo, source=repo, base_branch="main", base_sha="1" * 40, head_branch=None,
                       head_sha="2" * 40, is_ephemeral=False, run_id="run-1")

    async def no_dispatch() -> Any:
        raise AssertionError("an unrecoverable job must not dispatch")

    continuation: dict[str, Any] = {"work": work, "deep_dir_path": deep, "job_id": job_id, "footprint": footprint,
                                    "capture_tree_key": lambda: "now", "first": None, "dispatch": no_dispatch}
    outcome = await continue_repair_job(**continuation)
    assert outcome.job is not None and outcome.job.state is RepairJobState.BLOCKED
    assert "captured candidate cannot be restored" in outcome.reason
    (deep / "repair-job.json").unlink()
    DeepArtifact.REPAIR_CHECKPOINT.at(deep).write_text("{unreadable")
    outcome = await continue_repair_job(**continuation)
    assert outcome.job is not None and outcome.job.state is RepairJobState.BLOCKED
    assert "captured candidate cannot be trusted" in outcome.reason
