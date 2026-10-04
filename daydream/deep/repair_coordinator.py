"""One repair job's continuation: who runs the next execution, on what evidence, how often.

``phases/testing.py`` runs *one* bounded test-and-heal interaction and returns.
A repair job outlives that call: a turn the host cut short (budget, transport,
cancellation) leaves authorized work on the tree and a durable checkpoint behind
it, and the job owes another bounded execution. This module owns that
continuation and nothing else does — no phase, and no backend, dispatches a
second execution.

Four decisions live here, and each one is deliberately small enough to test on
its own:

**Who may continue.** Exactly one live repair owner per job
(:func:`try_acquire_repair_owner`, :func:`_owner_lock`). A second coordinator
never runs a second worker: a surviving first owner excludes it, in this process
through an owner registry and across processes through the existing exclusive
workspace ``flock``. A *resuming* process is not a second owner — the crashed
one is gone, the kernel released its lock — which is exactly what makes crash
recovery safe.

**Whether an interrupted repair may be continued at all.** Only a cut-short
repair (``BUDGET_INTERRUPTED``) enters bounded validation, and it is admitted
*once*: a completed experiment the host observed, not a claim. A second
execution that repeats the same candidate against the same failure is charged as
no progress, so the job goes ``BLOCKED`` with the unchanged evidence named
instead of looping on a red suite. A repair the host could not vouch for, an
infrastructure failure, and a plain completed-but-still-red suite are not
continuation progress at all.

**Whether a green execution completed the job.** Only a canonical validation
does: the passing execution's output tree key must equal the retained tree the
worktree still holds right now. A model claim, a candidate patch, or changed
bytes never complete a job.

**Whether captured work may be restored.** A captured candidate is applied only
to the exact tree it was captured against, under the same authorization policy
revision, with the stored patch digest re-verified against the file on disk. A
base that moved, a policy that moved, or a patch that does not apply is a named
conflict, never a best-effort apply — and an already-present candidate is
recognized (reverse check) rather than re-applied, so recovery is idempotent.

Scope is resolved the same way: only a repair turn's *own* request, carrying
per-path evidence from that turn, may widen the authorization policy. A path
that merely appears in test output is not evidence of authorization
(:func:`_resolve_scope_request`), and an uncanonicalizable request grants
nothing at all.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import daydream.agent as agent
import daydream.ui as ui
from daydream import git_ops
from daydream.benchmark.storage import LockContentionError, WorkspaceLock
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.repair_job import (
    RepairAction,
    RepairJobPolicy,
    RepairJobRecord,
    RepairJobState,
    read_repair_job_record,
    record_diagnostic,
    write_repair_job_record,
)
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.json_utils import read_json_object
from daydream.phases.repair_checkpoint import RepairCheckpoint, read_repair_checkpoint
from daydream.phases.repair_outcome import RepairOutcome, repair_scope_request
from daydream.phases.test_evidence import RepairAttemptEvidence, TestAndHealResult
from daydream.repository_paths import (
    InvalidRepositoryFilePath,
    canonicalize_repository_file_path,
)
from daydream.workspace import WorkContext

#: The host-observed experiment that admits an interrupted candidate to bounded
#: validation. A *name*, recorded in the job, so "we gave the candidate one more
#: shot" is visible after the fact — and its second occurrence is the evidence
#: that the job is going in circles.
BOUNDED_VALIDATION_EXPERIMENT = "bounded validation of the interrupted candidate"

#: The experiment a completed job ran last, recorded for the same reason.
FINAL_VALIDATION_EXPERIMENT = "final suite validation of the retained tree"

#: Evidence sources that may widen the authorization policy. A repair turn's own
#: request is the only one: it is the turn saying which file the failure it was
#: handed actually lives in. Everything else — test output, a diff scan, a
#: filename that merely appeared in a stack trace — describes a *symptom*, and
#: authorizing a path because a symptom mentioned it is how a run walks out of
#: its sandbox.
_AUTHORIZING_EVIDENCE_SOURCES = frozenset({"repair_turn"})

#: Owner registries are keyed by the deep artifact directory, which is the job's
#: identity: two coordinators over the same directory are the same job.
_OWNERS: dict[Path, str] = {}


# --- scope ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeRequestResolution:
    """The host's answer to one repair turn's scope request.

    ``accepted`` is what the turn may work on *now*: its already-authorized paths
    plus, only when ``expanded``, the ones this request is entitled to add.
    ``new_revision`` is the policy revision the grant would produce, which is
    ``policy_revision`` unchanged whenever nothing was widened — a request that
    does not move the policy must not pretend to.
    """

    accepted: tuple[str, ...]
    expanded: bool
    new_revision: int
    reason: str
    requested: tuple[str, ...] = ()


def _resolve_scope_request(
    repo: Path,
    *,
    requested: object,
    granted: frozenset[str] | set[str],
    evidence_source: str = "repair_turn",
    policy_revision: int = 1,
    evidence: Mapping[str, str] | None = None,
) -> ScopeRequestResolution:
    """Resolve a repair turn's requested paths against the current authorization.

    Every requested value is canonicalized through ``repository_paths`` first, so
    an absolute path, a traversal, or a symlink crossing raises
    :class:`~daydream.repository_paths.InvalidRepositoryFilePath` instead of
    becoming a narrower-than-asked authorization. A path already in the grant is
    accepted without moving the policy. A path outside it is accepted *only* when
    this request carries authorizing evidence — the turn's own scope request
    block — and never on the strength of test output.
    """
    if isinstance(requested, (str, bytes)) or not isinstance(requested, (list, tuple, set, frozenset)):
        raise InvalidRepositoryFilePath("invalid scope request paths")
    normalized = tuple(dict.fromkeys(
        canonicalize_repository_file_path(repo, value) for value in requested
    ))
    outside = tuple(path for path in normalized if path not in granted)
    if not normalized:
        return ScopeRequestResolution((), False, policy_revision, "nothing requested", ())
    if not outside:
        return ScopeRequestResolution(
            normalized, False, policy_revision, "already authorized", normalized,
        )
    if evidence_source not in _AUTHORIZING_EVIDENCE_SOURCES:
        return ScopeRequestResolution(
            tuple(path for path in normalized if path in granted), False, policy_revision,
            "insufficient_evidence", normalized,
        )
    return ScopeRequestResolution(
        normalized, True, policy_revision + 1, "", normalized,
    )


def _apply_scope_request(
    job: RepairJobRecord,
    repair: RepairAttemptEvidence,
    *,
    repo: Path,
    deep: Path,
    footprint: AuthorizedFixFootprint,
) -> None:
    """Record the turn's scope request on the job and widen the policy when entitled.

    The request is persisted either way — an honored, a rejected, and an
    unparseable request are all evidence a reader needs. A malformed payload is
    a named diagnostic rather than an exception out of a post-hoc audit: the
    repair itself already happened, and failing here would erase its record.
    """
    payload = repair.scope_request
    if payload is None:
        return
    try:
        request = repair_scope_request(repo, payload)
    except InvalidRepositoryFilePath as exc:
        record_diagnostic(deep, job.job_id, f"scope_request_malformed: {exc}")
        return
    if request is None or not request.requested:
        return
    try:
        resolution = _resolve_scope_request(
            repo,
            requested=request.requested,
            granted=footprint.run_allowed_paths,
            evidence_source=request.evidence_source,
            policy_revision=footprint.policy_revision,
            evidence=request.evidence,
        )
    except InvalidRepositoryFilePath as exc:
        record_diagnostic(deep, job.job_id, f"scope_request_rejected: {exc}")
        return
    if resolution.expanded:
        for path in resolution.accepted:
            footprint.authorize_scope_request(
                repo, path,
                phase="test_heal", round_number=None,
                reason=request.evidence.get(path) or (
                    f"repair turn {repair.execution_id} requested this path from the failure it worked on"
                ),
            )
        ui.print_info(
            agent.console,
            f"Repair turn {repair.execution_id} widened the authorized scope to "
            f"{len(resolution.accepted)} path(s) at policy revision {resolution.new_revision}.",
        )
    elif resolution.accepted:
        ui.print_dim(
            agent.console,
            f"Repair turn {repair.execution_id} requested {len(resolution.requested)} path(s); "
            f"{len(resolution.accepted)} already authorized.",
        )
    job.scope_request = {
        "execution_id": repair.execution_id,
        "requested": list(resolution.requested),
        "accepted": list(resolution.accepted),
        "expanded": resolution.expanded,
        "reason": resolution.reason,
        "evidence_source": request.evidence_source,
    }
    job.policy_revision = footprint.policy_revision
    job.authorized_scope = tuple(sorted(footprint.run_allowed_paths))


# --- restoration ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateRestoreDecision:
    """Whether captured work may be re-applied, and which identity check said no.

    ``reason`` names the failed check, because "restoration refused" without the
    identity that moved is not a diagnosis a resuming operator can act on.
    """

    allowed: bool
    reason: str


def _candidate_restore_decision(
    *, base_changed: bool, policy_changed: bool, patch_digest_ok: bool,
) -> CandidateRestoreDecision:
    """Fail closed on any identity mismatch, naming the check that failed.

    Requirement 32: the base tree, the authorization policy revision, and the
    stored patch digest are all part of the candidate's identity. A candidate
    whose base moved would be applied to a tree it was never measured against; a
    candidate whose policy moved was authorized under rules that no longer hold.
    """
    if base_changed:
        return CandidateRestoreDecision(False, "the worktree no longer matches the candidate's base tree")
    if policy_changed:
        return CandidateRestoreDecision(
            False, "the authorization policy revision changed since the candidate was captured",
        )
    if not patch_digest_ok:
        return CandidateRestoreDecision(False, "the stored patch digest does not match the captured patch")
    return CandidateRestoreDecision(True, "")


def _stored_digest_ok(deep_dir_path: Path, checkpoint: RepairCheckpoint) -> bool:
    """Re-verify the stored digest against the file on disk, immediately before a write."""
    raw = read_json_object(DeepArtifact.REPAIR_CHECKPOINT.at(deep_dir_path))
    return isinstance(raw, Mapping) and raw.get("patch_digest") == checkpoint.patch_digest


def _restore_candidate(
    *,
    repo: Path,
    deep_dir_path: Path,
    checkpoint: RepairCheckpoint | None,
    current_tree_key: str,
    policy_revision: int,
    job_id: str,
) -> tuple[bool, str]:
    """Restore a captured candidate onto a tree whose identity the host verified.

    Returns ``(restored, reason)``. Idempotent by construction: a candidate that
    is already present is recognized through a reverse check rather than
    re-applied, so a second recovery attempt of the same job is a no-op instead
    of a failure. An empty candidate is nothing to restore — an authorized turn
    that changed nothing is a real capture, not a missing one.
    """
    if checkpoint is None or not checkpoint.candidate_patch.strip():
        return False, "no candidate to restore"
    if checkpoint.job_id != job_id:
        return False, f"the stored checkpoint belongs to job {checkpoint.job_id!r}, not {job_id!r}"
    known = {key for key in (checkpoint.base_tree_key, checkpoint.retained_tree_key) if key}
    decision = _candidate_restore_decision(
        base_changed=bool(known) and current_tree_key not in known,
        policy_changed=checkpoint.policy_revision != policy_revision,
        patch_digest_ok=_stored_digest_ok(deep_dir_path, checkpoint),
    )
    if not decision.allowed:
        return False, decision.reason
    patch = checkpoint.candidate_patch.encode("utf-8", errors="surrogateescape")
    if git_ops.worktree_patch_applies(repo, patch):
        git_ops.apply_worktree_patch(repo, patch)
        return True, "candidate patch reapplied to its captured base tree"
    if git_ops.worktree_patch_applies(repo, patch, reverse=True):
        return False, "candidate patch is already present in the worktree"
    return False, "the candidate patch does not apply to the captured base tree"


# --- ownership ------------------------------------------------------------------------------------


@dataclass
class RepairOwnerHandle:
    """A live repair owner's claim on one job, or the reason there is none.

    Acquisition never throws: a run that finds another owner must be able to
    report the contended job and carry on with the other run's evidence, which
    is not the same as being the second worker for it.
    """

    root: Path
    owner_id: str
    acquired: bool
    reason: str = ""
    _lock: WorkspaceLock | None = field(default=None, repr=False)

    def release(self) -> None:
        """Release the claim. Idempotent; the lock file is deliberately left in place."""
        lock, self._lock = self._lock, None
        _OWNERS.pop(self.root, None)
        if lock is not None:
            lock.__exit__(None, None, None)

    def __enter__(self) -> RepairOwnerHandle:
        return self

    def __exit__(self, *_exc: object) -> Literal[False]:
        self.release()
        return False


def try_acquire_repair_owner(root: Path) -> RepairOwnerHandle:
    """Claim exclusive ownership of the repair job rooted at *root*.

    Two layers, because either alone is insufficient. ``WorkspaceLock`` gives a
    cross-process exclusive ``flock`` — and is reentrant *within* a process,
    which is exactly wrong here: a second coordinator in the same interpreter
    would be handed the first one's lock. The owner registry closes that gap, so
    "one live owner" means one owner in either dimension.
    """
    root = Path(root)
    held = _OWNERS.get(root)
    if held is not None:
        return RepairOwnerHandle(
            root, "", False, f"another repair owner is already live for this job: {root}",
        )
    lock = WorkspaceLock(root, blocking=False)
    try:
        # WorkspaceLock has no non-context acquire; this is its documented entry.
        lock.__enter__()
    except LockContentionError as exc:
        return RepairOwnerHandle(root, "", False, str(exc))
    owner_id = uuid.uuid4().hex
    _OWNERS[root] = owner_id
    return RepairOwnerHandle(root, owner_id, True, "", lock)


@contextmanager
def _owner_lock(root: Path) -> Iterator[RepairOwnerHandle]:
    """Hold the repair job's ownership for the duration of the block, or raise.

    :class:`~daydream.benchmark.storage.LockContentionError` is the contended
    answer, so a caller that genuinely cannot continue says so instead of
    proceeding as though it owned the job.
    """
    handle = try_acquire_repair_owner(root)
    if not handle.acquired:
        raise LockContentionError(handle.reason)
    try:
        yield handle
    finally:
        handle.release()


# --- continuation ---------------------------------------------------------------------------------


def repair_job_id(session_id: str) -> str:
    """The repair job's identity for one test session (``phases/testing.py``'s own id)."""
    return f"repair-{session_id}"


@dataclass(frozen=True)
class RepairContinuationResult:
    """What the coordinator did with the job, and what the caller should believe.

    ``result`` is the *latest* execution's typed result, so a caller that
    re-dispatched gets the same evidence a caller that did not. ``job`` is the
    persisted record, or ``None`` when the run never had a repair to own.
    """

    continued: bool
    result: TestAndHealResult | None
    job: RepairJobRecord | None
    reason: str
    executions_dispatched: int = 0

    @property
    def passed(self) -> bool:
        """Whether the latest execution passed *and* the job completed canonically."""
        return self.result is not None and self.result.passed and self.job is not None \
            and self.job.state is RepairJobState.COMPLETED


def _last_repair(result: TestAndHealResult | None) -> RepairAttemptEvidence | None:
    """The most recent repair turn of one execution, or ``None`` when it ran none."""
    return result.repairs[-1] if result is not None and result.repairs else None


def _validated_against(result: TestAndHealResult, capture_tree_key: Callable[[], str]) -> bool:
    """Whether this passing execution validated the tree the worktree still holds.

    Canonical validation, requirement 45: the last attempt's own output tree key
    must equal the retained tree as it stands now. A pass against some other
    tree — a tree the test itself mutated, or one a later edit moved — is not
    validation of anything the run would keep.
    """
    if not result.attempts:
        return False
    return result.attempts[-1].output_tree_key == capture_tree_key()


async def continue_repair_job(
    *,
    work: WorkContext,
    deep_dir_path: Path,
    job_id: str,
    footprint: AuthorizedFixFootprint,
    capture_tree_key: Callable[[], str],
    first: TestAndHealResult | None,
    dispatch: Callable[[], Awaitable[TestAndHealResult]],
    policy: RepairJobPolicy | None = None,
    cost_usd: float = 0.0,
    first_elapsed_s: float = 0.0,
    granted_allowance_s: float = 0.0,
) -> RepairContinuationResult:
    """Continue one repair job until it completes, blocks, or exhausts its bounds.

    ``first`` is the execution the caller already ran (``None`` when this process
    is resuming from disk), and ``dispatch`` runs exactly one more bounded
    execution against the same test phase. The loop cannot repeat indefinitely:
    every pass is charged to the job, the job's bounds are its own, and unchanged
    evidence ends it.

    ``cost_usd`` is what the *caller* observed for the executions this call
    charged, and ``first_elapsed_s`` how long its own execution took. The typed
    result carries neither billing nor a duration, so a caller that measured them
    passes them in and a caller that could not leaves them at their defaults: an
    under-charge, never an over-charge, with the execution ceiling still binding.

    ``granted_allowance_s`` is the operator's *additional finite* allowance
    (requirement 43). It is applied only to a job that ran out of time, never to
    one that went blocked, and only because someone asked for it: the seconds the
    job already spent stay spent and the grant is recorded beside them.
    """
    deep = Path(deep_dir_path)
    stored = read_repair_job_record(deep)
    checkpoint_read = read_repair_checkpoint(deep)
    if first is None or not first.repairs:
        if stored is None and checkpoint_read.checkpoint is None and not checkpoint_read.blocked:
            # Nothing to continue and nothing to resume: a green run that never
            # repaired writes no job record and takes no lock, so the artifact
            # set of an ordinary run is unchanged by this module.
            return RepairContinuationResult(False, first, None, "no repair to continue")

    try:
        with _owner_lock(deep):
            return await _continue_locked(
                work=work, deep=deep, job_id=job_id, footprint=footprint,
                capture_tree_key=capture_tree_key, first=first, dispatch=dispatch,
                policy=policy, cost_usd=cost_usd, first_elapsed_s=first_elapsed_s,
                granted_allowance_s=granted_allowance_s,
            )
    except LockContentionError as exc:
        ui.print_warning(
            agent.console,
            f"Not continuing the repair job: {exc}",
        )
        return RepairContinuationResult(
            False, first, read_repair_job_record(deep), f"repair owner contention: {exc}",
        )


async def _continue_locked(
    *,
    work: WorkContext,
    deep: Path,
    job_id: str,
    footprint: AuthorizedFixFootprint,
    capture_tree_key: Callable[[], str],
    first: TestAndHealResult | None,
    dispatch: Callable[[], Awaitable[TestAndHealResult]],
    policy: RepairJobPolicy | None,
    cost_usd: float,
    first_elapsed_s: float,
    granted_allowance_s: float,
) -> RepairContinuationResult:
    """The continuation loop, run by the job's single live owner."""
    job = _load_job(deep, job_id=job_id, footprint=footprint, policy=policy)
    if job.state in (RepairJobState.BLOCKED, RepairJobState.EXHAUSTED, RepairJobState.COMPLETED):
        return RepairContinuationResult(
            False, first, job, f"the persisted job is already {job.state.value}",
        )
    result = first
    dispatched = 0
    measured = first_elapsed_s
    while True:
        # Re-read every pass, not only at the start: a checkpoint that became
        # untrustworthy *during* the job is just as unrecoverable as one that
        # was corrupt when the job loaded, and "nothing to restore" would throw
        # the only copy of an interrupted repair away without saying so.
        blocked = _corrupt_checkpoint_reason(deep)
        if blocked is not None:
            reason = f"the captured candidate cannot be trusted: {blocked}"
            return _settle(
                _persist(deep, job.with_state(RepairJobState.BLOCKED, reason)), result, dispatched, reason,
            )
        repair = _last_repair(result)
        if result is not None and repair is not None:
            _apply_scope_request(job, repair, repo=work.repo, deep=deep, footprint=footprint)
        checkpoint = _usable_checkpoint(deep, job_id=job_id)
        elapsed = _elapsed(result, repair, measured)

        if result is not None and result.passed and _validated_against(result, capture_tree_key):
            _charge_completion(job, elapsed_s=elapsed, cost_usd=cost_usd, repair=repair)
            return _settle(
                _persist(deep, job.with_state(
                    RepairJobState.COMPLETED,
                    f"execution {job.executions} validated the retained tree with a passing suite",
                )),
                result, dispatched, "the repair job completed on a validated green execution",
            )

        if result is not None:
            # Only an execution that actually ran is charged. A resuming process
            # starts with no result at all: its job already carries the execution
            # the previous process charged, and charging the absence of one again
            # would invent a failure nobody observed.
            job = _charge_failure(
                job, result=result, repair=repair, checkpoint=checkpoint, elapsed_s=elapsed,
                cost_usd=cost_usd,
            )
            job = _persist(deep, job)
            action = job.next_action()
            if action is not RepairAction.EXECUTE:
                resumed = _apply_operator_grant(job, action, granted_allowance_s)
                if resumed is None:
                    return _settle(
                        job, result, dispatched, job.last_transition_reason or f"the job is {action.value}",
                    )
                job = _persist(deep, resumed)
                if job.next_action() is not RepairAction.EXECUTE:
                    return _settle(
                        job, result, dispatched,
                        job.last_transition_reason or f"the job is {job.state.value}",
                    )

        restored, reason = _restore_candidate(
            repo=work.repo, deep_dir_path=deep, checkpoint=checkpoint,
            current_tree_key=capture_tree_key(),
            policy_revision=footprint.policy_revision, job_id=job_id,
        )
        if not restored and _restore_refused(checkpoint, reason):
            # A conflict is the job's end, not a retry: the base it was captured
            # against is gone, and no bounded execution can rebuild it. A refusal
            # to *find* anything to restore is not a conflict.
            blocked_reason = f"captured candidate cannot be restored: {reason}"
            return _settle(
                _persist(deep, job.with_state(RepairJobState.BLOCKED, blocked_reason)),
                result, dispatched, blocked_reason,
            )
        if restored:
            ui.print_info(agent.console, f"Repair job restored captured work: {reason}")
        # The validating state is persisted *before* the dispatch, so a process
        # that dies inside it leaves a job that says which execution was owed and
        # why -- rather than a record that claims the work was still untouched.
        job = _persist(deep, job.with_state(
            RepairJobState.VALIDATING,
            f"execution {job.executions + 1} runs the {BOUNDED_VALIDATION_EXPERIMENT}",
        ))
        started = time.monotonic()
        result = await dispatch()
        measured = time.monotonic() - started
        dispatched += 1


def _apply_operator_grant(
    job: RepairJobRecord, action: RepairAction, allowance_s: float,
) -> RepairJobRecord | None:
    """Apply an operator's additional finite allowance, or return ``None``.

    Only a *time* exhaustion is grantable. A blocked job stopped because its
    evidence did not narrow, and handing it more seconds would buy nothing but
    a repeated non-result, so a blocked job is never replenished -- with or
    without an operator's grant. An exhausted job is replenished only when an
    operator actually asked for it, and the grant lands in its own field, so the
    consumption this job already paid for is still on the record. The grant is
    also written to ``diagnostics``, so it stays auditable after the job moves on
    to a state whose own reason says something else.
    """
    if action is not RepairAction.EXHAUSTED or allowance_s <= 0.0:
        return None
    exhausted = job.last_transition_reason or "exhaustion"
    job.grant_allowance(allowance_s, reason=f"the operator granted {allowance_s:g}s more after {exhausted}")
    granted = job.last_transition_reason or ""
    job.diagnostics = (*job.diagnostics, f"operator_grant: {granted}")
    return job.with_state(
        RepairJobState.READY_TO_RESUME,
        f"resumed by operator grant ({granted}); execution {job.executions + 1} is owed",
    )


def _restore_refused(checkpoint: RepairCheckpoint | None, reason: str) -> bool:
    """Whether a restore refusal is a conflict (blocking) or simply nothing to do."""
    if checkpoint is None or not checkpoint.candidate_patch.strip():
        return False
    return "already present" not in reason


def _corrupt_checkpoint_reason(deep: Path) -> str | None:
    """Name an untrustworthy checkpoint, or ``None`` when there is nothing wrong."""
    read = read_repair_checkpoint(deep)
    return read.reason if read.blocked else None


def _usable_checkpoint(deep: Path, *, job_id: str) -> RepairCheckpoint | None:
    """The captured candidate this job may restore, re-read fresh every pass.

    A new interrupted execution overwrites the checkpoint, so a value read before
    the dispatch is a stale identity: the loop re-reads it after every execution
    rather than carrying one forward.
    """
    read = read_repair_checkpoint(deep)
    if read.blocked or read.checkpoint is None:
        return None
    return read.checkpoint if read.checkpoint.job_id == job_id else None


def _elapsed(
    result: TestAndHealResult | None, repair: RepairAttemptEvidence | None, fallback: float,
) -> float:
    """This execution's wall seconds, from the host's own measurement where it has one.

    Relative seconds only: an absolute clock reading is never persisted, so a
    record written by a previous process stays meaningful.
    """
    if repair is not None:
        return max(0.0, float(repair.execution_elapsed_s))
    if result is not None:
        return max(0.0, fallback)
    return 0.0


def _load_job(
    deep: Path, *, job_id: str, footprint: AuthorizedFixFootprint, policy: RepairJobPolicy | None,
) -> RepairJobRecord:
    """The persisted job, or a new one bound to the policy the run granted.

    Requirement 44: a job's bounds are stored when it starts, so a resumed job
    reads its own captured policy instead of re-resolving whatever the current
    configuration says.
    """
    stored = read_repair_job_record(deep)
    if stored is not None and stored.job_id == job_id:
        return stored
    if stored is not None:
        record_diagnostic(
            deep, job_id,
            f"job_record_for_other_job_ignored: {stored.job_id!r} (this job is {job_id!r})",
        )
    return RepairJobRecord(
        job_id=job_id,
        policy=policy or RepairJobPolicy(),
        policy_revision=footprint.policy_revision,
        authorized_scope=tuple(sorted(footprint.run_allowed_paths)),
    )


def _persist(deep: Path, job: RepairJobRecord) -> RepairJobRecord:
    """Write the job record and read it back as the record the next step sees.

    Persisted *before* the next dispatch starts, so a process that dies mid
    execution leaves a job that knows an execution was owed and how far it got.
    """
    write_repair_job_record(deep, job)
    return read_repair_job_record(deep) or job


def _settle(
    job: RepairJobRecord, result: TestAndHealResult | None, dispatched: int, reason: str,
) -> RepairContinuationResult:
    """Shape the caller's answer, and say so out loud when the job did not finish."""
    if job.state is not RepairJobState.COMPLETED and job.executions:
        ui.print_warning(
            agent.console,
            f"Repair job {job.job_id} ended {job.state.value}: {reason}",
        )
    return RepairContinuationResult(dispatched > 0, result, job, reason, dispatched)


def _charge_completion(
    job: RepairJobRecord, *, elapsed_s: float, cost_usd: float,
    repair: RepairAttemptEvidence | None,
) -> None:
    """Charge the validating execution, with the repair that led to it as evidence."""
    completed = [FINAL_VALIDATION_EXPERIMENT]
    if repair is not None:
        completed.insert(0, repair.execution_id)
    job.record_execution(
        progress=True, next_experiment=None, completed_experiments=completed,
        elapsed_s=elapsed_s, cost_usd=cost_usd,
    )


def _charge_failure(
    job: RepairJobRecord,
    *,
    result: TestAndHealResult | None,
    repair: RepairAttemptEvidence | None,
    checkpoint: RepairCheckpoint | None,
    elapsed_s: float,
    cost_usd: float,
) -> RepairJobRecord:
    """Charge one unfinished execution and resolve the job's next state.

    Only a cut-short repair is continuation progress, and only the first time:
    the host-observed bounded validation is recorded as a completed experiment, so
    a repeat of the same candidate against the same failure carries no new
    experiment and the job blocks instead of looping.
    """
    if repair is not None and repair.checkpoint_ref:
        # The record points at the artifact the work actually lives in, so a
        # resuming process reads the candidate from the file, not from a claim.
        job.checkpoint_ref = repair.checkpoint_ref
    if checkpoint is not None:
        job.failure_identity = job.failure_identity or checkpoint.failure_identity
    if repair is None:
        # The execution ran and produced no repair evidence at all. There is
        # nothing to continue from, so the job records the absence and stops.
        job.record_execution(
            progress=False, next_experiment=None,
            unchanged_evidence=(
                f"execution ran {len(result.attempts) if result is not None else 0} attempt(s) "
                f"and no repair turn",
            ),
            elapsed_s=elapsed_s, cost_usd=cost_usd,
        )
        return job
    if repair.outcome is RepairOutcome.EXECUTION_ERROR:
        # An infrastructure failure is not diagnostic evidence about the code
        # under repair. Charging it as progress would buy a retry with a budget
        # the failure just proved does not work.
        job.record_execution(
            progress=False, next_experiment=repair_checkpoint_next(checkpoint),
            unchanged_evidence=(f"execution {repair.execution_id} failed before producing evidence",),
            infrastructure_failed=True,
            elapsed_s=elapsed_s, cost_usd=cost_usd,
        )
        return job
    if repair.outcome is not RepairOutcome.BUDGET_INTERRUPTED:
        job.record_execution(
            progress=False, next_experiment=repair_checkpoint_next(checkpoint),
            unchanged_evidence=(f"repair turn {repair.execution_id} ended {repair.outcome.value}",),
            changed_paths=repair.changed_paths, elapsed_s=elapsed_s, cost_usd=cost_usd,
        )
        return job

    failure = "" if checkpoint is None else checkpoint.failure_identity
    digest = "" if checkpoint is None else checkpoint.patch_digest[:12]
    if BOUNDED_VALIDATION_EXPERIMENT in job.progress_evidence:
        # The candidate already had its one bounded validation. Running it again
        # against the same failure is the loop requirement 41 forbids, so the
        # unchanged evidence is named and the job blocks.
        job.record_execution(
            progress=False, next_experiment=repair_checkpoint_next(checkpoint),
            unchanged_evidence=(
                f"candidate {digest or 'unknown'} re-validated against the same failure "
                f"{failure or 'unrecorded'}",
            ),
            changed_paths=repair.changed_paths, elapsed_s=elapsed_s, cost_usd=cost_usd,
        )
        return job
    job.record_execution(
        progress=True, next_experiment=repair_checkpoint_next(checkpoint),
        completed_experiments=[BOUNDED_VALIDATION_EXPERIMENT],
        candidate_patch_confirmed=bool(checkpoint is not None and checkpoint.candidate_patch.strip()),
        changed_paths=repair.changed_paths, elapsed_s=elapsed_s, cost_usd=cost_usd,
    )
    return job


def repair_checkpoint_next(checkpoint: RepairCheckpoint | None) -> str | None:
    """The experiment the captured turn named as next, when it named one."""
    return None if checkpoint is None else checkpoint.next_experiment


__all__ = [
    "BOUNDED_VALIDATION_EXPERIMENT",
    "FINAL_VALIDATION_EXPERIMENT",
    "CandidateRestoreDecision",
    "RepairContinuationResult",
    "RepairOwnerHandle",
    "ScopeRequestResolution",
    "continue_repair_job",
    "repair_job_id",
    "try_acquire_repair_owner",
]
