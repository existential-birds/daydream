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

import hashlib
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from daydream import phases
from daydream.backends import ResultEvent, TextEvent
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
    record_diagnostic,
    write_repair_job_record,
)
from daydream.deep.settings import _resolve_non_negative_float
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.phases.repair_checkpoint import (
    RepairCheckpoint,
    read_repair_checkpoint,
    write_repair_checkpoint,
)
from daydream.phases.repair_outcome import repair_scope_request
from daydream.repository_paths import InvalidRepositoryFilePath
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import commit as git_commit, git, init_repo, write_and_stage

_RESULT = ResultEvent(structured_output=None, continuation=None)


def _checkpoint(**overrides: Any) -> RepairCheckpoint:
    """Build one checkpoint; every field except the identity is a keyword default."""
    fields: dict[str, Any] = {
        "job_id": "repair-job-0001",
        "execution_id": "repair-job-0001:execution:1",
        "candidate_patch": "diff --git a/allowed.py b/allowed.py\n+repair edit\n",
        "base_tree_key": "tree-base",
        "retained_tree_key": "tree-retained",
        "authorized_scope": ("allowed.py", "helper.py"),
        "policy_revision": 3,
        "failure_identity": "sha256:0123456789abcdef",
        "test_command": "pytest -q",
        "test_cwd": ".",
        "focused_results": ("1 failed, 0 passed",),
        "completed_experiments": (),
        "disproven_hypotheses": ("missing import",),
        "next_experiment": None,
        "backend_name": "scripted",
        "model": "test-model",
        "consumed_budget": {"wall_budget_s": 1800.0, "elapsed_s": 0.3},
    }
    fields.update(overrides)
    return RepairCheckpoint(**fields)


def _policy(**overrides: Any) -> RepairJobPolicy:
    """Build a job policy; the four bounds are keyword defaults."""
    fields: dict[str, Any] = {
        "execution_s": 1800.0,
        "job_total_s": 7200.0,
        "max_executions": 4,
        "reserve_s": 60.0,
    }
    fields.update(overrides)
    return RepairJobPolicy(**fields)


def _job(**overrides: Any) -> RepairJobRecord:
    """Build a job record; the identity and the start-captured policy are defaults."""
    fields: dict[str, Any] = {
        "job_id": "repair-job-0001",
        "state": RepairJobState.RUNNING,
        "failure_identity": "sha256:0123456789abcdef",
        "authorized_scope": ("src/handler.py", "tests/test_a.py"),
        "policy_revision": 3,
        "policy": _policy(),
    }
    fields.update(overrides)
    return RepairJobRecord(**fields)


def _limit_config(**attrs: object) -> Any:
    """Minimal RunConfig stand-in carrying only the scalar under test."""
    return SimpleNamespace(file_config=None, **attrs)


def test_unwritable_checkpoint_is_a_blocker_not_a_silent_clean_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 28: a checkpoint the host cannot persist blocks the repair."""
    monkeypatch.setattr(
        "daydream.phases.repair_checkpoint.atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        write_repair_checkpoint(tmp_path, _checkpoint())
    assert not list(tmp_path.glob("*.tmp")), "a failed write leaves no staging file behind"


def test_job_record_merges_a_diagnostic_instead_of_clobbering_state(tmp_path: Path) -> None:
    """The read-modify-write shape: a later write keeps an earlier execution's evidence."""
    write_repair_job_record(tmp_path, RepairJobRecord(job_id="repair-s1", executions=1))
    second = record_diagnostic(tmp_path, "repair-s1", "checkpoint_write_failed: OSError: disk full")
    assert second is not None
    assert second.job_id == "repair-s1"
    assert second.state is RepairJobState.RUNNING
    assert second.executions == 1, "the diagnostic merge dropped the earlier execution"
    assert second.diagnostics == ("checkpoint_write_failed: OSError: disk full",)
    assert read_repair_job_record(tmp_path) == second


def test_corrupt_job_record_is_not_read_as_an_empty_job(tmp_path: Path) -> None:
    """A foreign or corrupt record is reported as absent, never as fresh state."""
    assert read_repair_job_record(tmp_path) is None
    (tmp_path / "repair-job.json").write_text("{not json")
    assert read_repair_job_record(tmp_path) is None


@pytest.mark.asyncio
async def test_phase_blocks_the_repair_when_the_checkpoint_cannot_be_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    silence_console: Callable[..., None],
) -> None:
    """Requirement 28: an uncaptured repair is a repair that never happened."""
    silence_console("daydream.ui")
    init_repo(tmp_path)
    (tmp_path / "allowed.py").write_text("-- original\n")
    git(tmp_path, "add", "allowed.py")
    git_commit(tmp_path, "test: seed authorized file")

    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "2")
    monkeypatch.setattr(
        "daydream.phases.testing.write_repair_checkpoint",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    fail_turn = [TextEvent(text="1 failed, 0 passed"), _RESULT]

    def responder(_cwd: Path, prompt: str, *_rest: Any) -> Any:
        return tuple(fail_turn) if prompt.lower().startswith("the tests failed") else None

    backend = ScriptedBackend(responder=responder)
    footprint = AuthorizedFixFootprint(
        run_allowed_paths=frozenset({"allowed.py"}), policy_revision=1,
    )
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), session_id="s1",
        capture_tree_key=lambda: "tree-1", footprint=footprint, allow_standalone=True,
    )

    assert (result.passed, result.proceed, result.retries) == (False, False, 1)
    assert result.repairs[0].checkpoint_ref is None
    assert any("checkpoint_write_failed" in diagnostic and "OSError" in diagnostic
        for diagnostic in result.repairs[0].diagnostics)
    # The blocker is named in the job record too, so a resuming job sees it.
    job = read_repair_job_record(tmp_path / ".daydream" / "deep")
    assert job is not None
    assert any("checkpoint_write_failed" in diagnostic for diagnostic in job.diagnostics)


@pytest.mark.asyncio
async def test_checkpoint_captures_authorized_work_before_the_guard_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext],
    silence_console: Callable[..., None],
) -> None:
    """Requirements 25/28: cleanup can never erase the only copy."""
    silence_console("daydream.ui")
    init_repo(tmp_path)
    (tmp_path / "allowed.py").write_text("-- original\n")
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "0001_init.sql").write_text("-- generated original\n")
    git(tmp_path, "add", "allowed.py", "migrations/0001_init.sql")
    git_commit(tmp_path, "test: seed authorized and generated files")

    monkeypatch.setattr("daydream.config.DEFAULT_WALL_BUDGET_S", 0.3)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "2")

    async def repair(**_kwargs: Any) -> AsyncIterator[TextEvent]:
        # The spike's confirmed sequence: authorized edit, authorized new file,
        # then a stall to the wall budget with no terminal ResultEvent.
        (tmp_path / "allowed.py").write_text("-- repair edit\n")
        (tmp_path / "helper.py").write_text("# new helper\n")
        while True:
            yield TextEvent(text="PARTIAL-DIAGNOSIS")
            await anyio.sleep(0)

    backend = ScriptedBackend(responder=lambda cwd, prompt, *_: repair()
                              if prompt.lower().startswith("the tests failed") else None)
    footprint = AuthorizedFixFootprint(
        run_allowed_paths=frozenset({"allowed.py", "helper.py", "migrations/0001_init.sql"}),
        policy_revision=1,
    )
    result = await phases.phase_test_and_heal(
        backend, make_work(tmp_path), session_id="s1",
        capture_tree_key=lambda: "tree-1", footprint=footprint,
        allow_standalone=True,
    )

    assert (result.passed, result.proceed) == (False, False)
    read = read_repair_checkpoint(tmp_path / ".daydream" / "deep")
    assert read.blocked is False, read.reason
    checkpoint = read.checkpoint
    assert checkpoint is not None
    # Both authorized artifacts the interrupted turn produced are in the capture,
    # read from the live tree before the guard could restore anything.
    assert "repair edit" in checkpoint.candidate_patch
    assert "helper.py" in checkpoint.candidate_patch
    # The capture is the interrupted turn's own identity, not a later attempt's.
    assert checkpoint.execution_id == f"repair-s1:execution:{result.retries}"
    assert checkpoint.authorized_scope == ("allowed.py", "helper.py", "migrations/0001_init.sql")
    assert checkpoint.backend_name == "scripted"
    assert checkpoint.base_tree_key == "tree-1"
    assert checkpoint.next_experiment, "a bounded follow-up must be named for the resuming job"
    # The stored payload is verifiable on its own terms.
    stored = json.loads(
        (tmp_path / ".daydream" / "deep" / "repair-checkpoint.json").read_text(encoding="utf-8")
    )
    assert stored["patch_digest"] == hashlib.sha256(
        stored["candidate_patch"].encode()).hexdigest()
    assert result.repairs[0].checkpoint_ref == "repair-checkpoint.json"


def test_execution_deadline_is_the_smallest_remaining_allowance_less_reserve() -> None:
    """Requirement 36: process-local, derived per execution."""
    job = _job(consumed_s=500.0, policy=_policy(execution_s=1800, job_total_s=7200,
                                                 max_executions=4, reserve_s=120))
    assert job.execution_allowance_s() == pytest.approx(min(1800, 7200 - 500 - 120))


def test_no_progress_reports_blocked_naming_the_unchanged_evidence() -> None:
    """Requirement 41 + the Should Have: the signal names what it saw."""
    job = _job(unchanged_evidence=("focused test:tests/test_a.py", "src/handler.py"))
    job.record_execution(progress=False, next_experiment=None)
    assert job.next_action() == RepairAction.BLOCKED
    assert "tests/test_a.py" in (job.last_transition_reason or "")


def test_more_output_or_changed_bytes_alone_is_not_progress() -> None:
    """Requirement 40: a real defect alone is not progress."""
    job = _job()
    job.record_execution(progress=False, next_experiment=None,
                         completed_experiments=[], tool_calls=400, changed_paths=("src/a.py",))
    assert job.next_action() == RepairAction.BLOCKED


@pytest.mark.parametrize(("raw_value", "expected"), [(-5, 1800.0), ("nonsense", 1800.0),
                                                      (0.0, 0.0), (900.0, 900.0)])
def test_invalid_job_limits_degrade_to_the_default(raw_value: object, expected: float) -> None:
    """The existing resolver family degrades invalid input to the default."""
    cfg = _limit_config(repair_execution_wall_s=raw_value)
    assert _resolve_non_negative_float(cfg, "repair_execution_wall_s", 1800.0) == expected


def test_execution_count_ceiling_is_exhausted_not_blocked() -> None:
    """Requirement 42: a bounded job stops on its own bounds, not on a defect."""
    job = _job(executions=4, policy=_policy(max_executions=4))
    assert job.next_action() == RepairAction.EXHAUSTED


def test_operator_grant_is_recorded_apart_from_usage_and_never_silent() -> None:
    """Requirement 43: replenishment is explicit, auditable, and separate from history."""
    job = _job(consumed_s=7200.0, executions=2, policy=_policy())
    assert job.next_action() == RepairAction.EXHAUSTED
    with pytest.raises(ValueError, match="reason"):
        job.grant_allowance(1800.0, reason="   ")
    job.grant_allowance(1800.0, reason="operator: budget extended after triage")
    assert job.consumed_s == 7200.0, "a grant must not rewrite historical usage"
    assert job.next_action() == RepairAction.EXECUTE
    assert "operator: budget extended after triage" in (job.last_transition_reason or "")


def test_cost_ceiling_is_exhausted_not_blocked() -> None:
    """Requirement 42: cost is a bound like time and count, and reports as exhausted."""
    job = _job(cumulative_cost_usd=5.01, policy=_policy(max_cost_usd=5.0))
    assert job.next_action() == RepairAction.EXHAUSTED


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


def test_scope_request_path_merely_appearing_in_test_output_grants_nothing(tmp_path: Path) -> None:
    """Requirement 10: test output is not evidence of authorization."""
    repo = _scope_repo(tmp_path)
    result = _resolve_scope_request(
        repo,
        requested={"src/leaked.rs"},
        granted=frozenset({"src/handler.py"}),
        evidence_source="test_output",
    )
    assert result.expanded is False
    assert result.reason == "insufficient_evidence"
    assert result.accepted == (), "a blocked path is never accepted, only recorded"
    assert result.new_revision == 1


def test_scope_request_validation_is_fail_closed_on_untrusted_paths(tmp_path: Path) -> None:
    """Requirement 13: an unsafe requested path is rejected, never reflected or authorized."""
    repo = _scope_repo(tmp_path)
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
    assert try_acquire_repair_owner(tmp_path).acquired is True

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_checkpoint_is_readable_from_a_separate_interpreter(tmp_path: Path) -> None:
    """Requirements 27/48: cross-process readability, no inherited in-memory state."""
    write_repair_checkpoint(tmp_path, _checkpoint())
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys;from pathlib import Path;"
         "sys.path.insert(0,'.');"
         "from daydream.phases.repair_checkpoint import read_repair_checkpoint;"
         "print(read_repair_checkpoint(Path(sys.argv[1])).checkpoint.payload()['job_id'])",
         str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "repair-job-0001"
