"""The durable repair job record (issue #1210).

One artifact, one writer, one meaning — the read-modify-write shape of
``deep/routing_record.py``. ``repair-job.json`` is the job's own state: which
execution ran last, what the job has consumed, what bound it was granted, what
the repair turn asked for, and every named degradation of the host's accounting
(including a checkpoint that could not be persisted). A later execution merges
into it, so a resume never erases an earlier execution's evidence.

Two properties are structural rather than checked later:

* **Fail closed.** A job that is not ``COMPLETED`` is *incapable* of reporting a
  passing verdict or authorizing a commit (:attr:`RepairJobRecord.cannot_report_green`).
* **Relative time only.** The record persists consumed seconds, never an absolute
  clock reading. :meth:`RepairJobRecord.execution_allowance_s` derives each
  execution's allowance from that persisted consumption, so a deadline is
  process-local by construction and a record written by a previous process is
  still meaningful.

It is *not* an input to the dispatch decision in this module: the checkpoint
(``phases/repair_checkpoint.py``) owns the captured work, and the coordinator
owns the decision. This record exists so the decision and the outcome survive the
process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from daydream.config import (
    DEFAULT_REPAIR_EXECUTION_WALL_S,
    DEFAULT_REPAIR_JOB_WALL_S,
    DEFAULT_REPAIR_MAX_EXECUTIONS,
)
from daydream.deep.artifacts import DeepArtifact
from daydream.json_utils import atomic_write_json, read_json_object

#: Bump whenever the job record shape changes so a stale file can never be read
#: as the current contract (mirrors ``EVIDENCE_REUSE_FORMAT``).
REPAIR_JOB_FORMAT: int = 1

#: Held back from the job total so the final validating execution has somewhere
#: to run. Not operator-configurable: a reserve an operator can zero is not a
#: reserve.
REPAIR_JOB_RESERVE_S: float = 60.0


def repair_job_id(session_id: str) -> str:
    """The repair job's identity for one test session.

    One definition, because the producer (``phases/testing.py``) and the consumer
    (``deep/repair_coordinator.py``) must agree byte-for-byte: a checkpoint
    captured under a different spelling is unrestorable, which would silently
    block the whole job.
    """
    return f"repair-{session_id}"


class RepairJobState(StrEnum):
    """Where a repair job stands; every terminal state names why in its reason.

    ``running`` is the in-flight state. ``paused`` and ``ready_to_resume`` are
    both non-terminal: the job owes another bounded execution. ``validating`` is
    the final execution that re-runs the suite. ``completed`` is the only state
    from which a green verdict or a commit is even representable.
    """

    RUNNING = "running"
    PAUSED = "paused"
    READY_TO_RESUME = "ready_to_resume"
    VALIDATING = "validating"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"


class RepairAction(StrEnum):
    """What the host may do with a job next; never "repeat indefinitely"."""

    EXECUTE = "execute"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class RepairJobPolicy:
    """The bounds a job was granted, captured when the job started.

    Requirement 44: these are stored, not re-resolved per execution, so a resumed
    job can never quietly inherit a broader policy than it was granted.
    """

    execution_s: float = DEFAULT_REPAIR_EXECUTION_WALL_S
    job_total_s: float = DEFAULT_REPAIR_JOB_WALL_S
    max_executions: int = DEFAULT_REPAIR_MAX_EXECUTIONS
    reserve_s: float = REPAIR_JOB_RESERVE_S
    max_cost_usd: float | None = None

    def payload(self) -> dict[str, Any]:
        """JSON-serializable form of the captured policy."""
        return {
            "execution_s": self.execution_s,
            "job_total_s": self.job_total_s,
            "max_executions": self.max_executions,
            "reserve_s": self.reserve_s,
            "max_cost_usd": self.max_cost_usd,
        }

    @classmethod
    def from_payload(cls, raw: object) -> RepairJobPolicy:
        """Rebuild from a stored policy; any missing or malformed bound is the default."""
        if not isinstance(raw, Mapping):
            return cls()
        default = cls()
        execution_s = _finite_float(raw.get("execution_s"))
        job_total_s = _finite_float(raw.get("job_total_s"))
        reserve_s = _finite_float(raw.get("reserve_s"))
        max_executions = _plain_int(raw.get("max_executions"))
        max_cost_usd = _finite_float(raw.get("max_cost_usd"))
        return cls(
            execution_s=default.execution_s if execution_s is None else max(0.0, execution_s),
            job_total_s=default.job_total_s if job_total_s is None else max(0.0, job_total_s),
            max_executions=default.max_executions if max_executions is None
            else max(0, max_executions),
            reserve_s=default.reserve_s if reserve_s is None else max(0.0, reserve_s),
            max_cost_usd=None if max_cost_usd is None else max(0.0, max_cost_usd),
        )


@dataclass
class RepairJobRecord:
    """What one repair job has run, what it has consumed, and what it still owes.

    Evidence fields are *names*, never contents: the bounded evidence itself lives
    in the checkpoint this record points at. Mutability is deliberate — an
    execution records its outcome in place, the way the job's own timeline runs.
    """

    job_id: str
    state: RepairJobState = RepairJobState.RUNNING
    failure_identity: str = ""
    authorized_scope: tuple[str, ...] = ()
    policy_revision: int = 0
    policy: RepairJobPolicy = field(default_factory=RepairJobPolicy)
    #: Cumulative *job* wall seconds already spent, across every execution and
    #: process. The only time value ever persisted.
    consumed_s: float = 0.0
    cumulative_cost_usd: float = 0.0
    #: Additional allowance an operator granted. Recorded separately from usage
    #: so replenishment is visible and an exhausted job is never silently topped up.
    granted_allowance_s: float = 0.0
    executions: int = 0
    scope_request: Mapping[str, Any] | None = None
    unchanged_evidence: tuple[str, ...] = ()
    progress_evidence: tuple[str, ...] = ()
    next_experiment: str | None = None
    completed_experiments: tuple[str, ...] = ()
    disproven_hypotheses: tuple[str, ...] = ()
    last_transition_reason: str | None = None
    checkpoint_ref: str | None = None
    diagnostics: tuple[str, ...] = ()

    # -- fail-closed posture -----------------------------------------------------------------

    @property
    def cannot_report_green(self) -> bool:
        """Whether this job is structurally barred from reporting a passing verdict.

        The same fail-closed posture bars the job from authorizing a commit: only
        ``completed`` may do either.
        """
        return self.state is not RepairJobState.COMPLETED

    # -- budget arithmetic ------------------------------------------------------------------

    def execution_allowance_s(self) -> float:
        """Relative wall seconds the next execution may spend, floored at zero.

        Requirement 36: the smallest of the per-execution ceiling and what the job
        total still has left after this job's persisted consumption and the
        reserve. Never an absolute instant — the caller adds its own process-local
        clock start, which is why no timestamp is ever persisted.
        """
        remaining = (
            self.policy.job_total_s
            - self.consumed_s
            - self.policy.reserve_s
            + self.granted_allowance_s
        )
        return max(0.0, min(self.policy.execution_s, remaining))

    def _exhaustion_reason(self) -> str | None:
        """Name the bound that stops the job, or ``None`` while it may still run.

        Time first, then count, then cost: a simultaneous breach resolves to one
        deterministic reason rather than whichever check happened to run first.
        """
        if self.execution_allowance_s() <= 0.0:
            return (
                f"job wall allowance exhausted after {self.consumed_s:.1f}s consumed "
                f"(total {self.policy.job_total_s:.1f}s, reserve {self.policy.reserve_s:.1f}s)"
            )
        if self.executions >= self.policy.max_executions:
            return f"execution ceiling reached: {self.executions}/{self.policy.max_executions}"
        cap = self.policy.max_cost_usd
        if cap is not None and self.cumulative_cost_usd >= cap:
            return f"job cost allowance exhausted: ${self.cumulative_cost_usd:.2f} of ${cap:.2f}"
        return None

    def grant_allowance(self, extra_s: float, *, reason: str) -> None:
        """Record an operator's additional finite allowance (requirement 43).

        A reason is mandatory: replenishment must be auditable, and historical
        usage is never rewritten. Grants compose, so an exhausted job only runs
        again because someone said so.
        """
        if not reason.strip():
            raise ValueError("a repair allowance grant must state its reason")
        self.granted_allowance_s += max(0.0, extra_s)
        self.last_transition_reason = f"allowance granted: {reason.strip()}"

    # -- outcome classification -------------------------------------------------------------

    def record_execution(
        self, *, progress: bool, next_experiment: str | None,
        completed_experiments: Sequence[str] = (),
        disproven_hypotheses: Sequence[str] = (),
        unchanged_evidence: Sequence[str] = (),
        candidate_patch_confirmed: bool = False,
        infrastructure_failed: bool = False,
        tool_calls: int = 0,
        changed_paths: Sequence[str] = (),
        elapsed_s: float = 0.0,
        cost_usd: float = 0.0,
    ) -> RepairAction:
        """Charge one finished execution to the job and resolve its termination state.

        Requirement 40: the caller's ``progress`` claim is necessary but not
        sufficient. Progress requires a *discriminating* result — a completed
        experiment, or a candidate patch a focused test supports. More tool calls,
        more output, or changed file bytes alone are not progress, and an
        infrastructure failure is never diagnostic progress. Returns the action
        the job may take next, which is also its new state.
        """
        self.executions += 1
        self.consumed_s += max(0.0, elapsed_s)
        self.cumulative_cost_usd += max(0.0, cost_usd)
        self.completed_experiments = _extend(self.completed_experiments, completed_experiments)
        self.disproven_hypotheses = _extend(self.disproven_hypotheses, disproven_hypotheses)
        self.next_experiment = next_experiment

        observed = tuple(unchanged_evidence)
        if observed:
            self.unchanged_evidence = _extend(self.unchanged_evidence, observed)
        discriminating = bool(completed_experiments) or candidate_patch_confirmed
        made_progress = progress and discriminating and not infrastructure_failed

        if made_progress:
            self.unchanged_evidence = ()
            observed_progress = tuple(completed_experiments) or observed or (
                "candidate patch confirmed by a focused test",
            )
            self.progress_evidence = _extend(self.progress_evidence, observed_progress)
            self.state = RepairJobState.READY_TO_RESUME
            self.last_transition_reason = (
                f"execution {self.executions} made progress: " + "; ".join(observed_progress)
            )
        else:
            self.state = RepairJobState.BLOCKED
            self.last_transition_reason = self._no_progress_reason(
                tool_calls=tool_calls, changed_paths=tuple(changed_paths),
                infrastructure_failed=infrastructure_failed,
            )

        exhausted = self._exhaustion_reason()
        if exhausted is not None:
            self.state = RepairJobState.EXHAUSTED
            self.last_transition_reason = f"execution {self.executions}: {exhausted}"
        return self.next_action()

    def _no_progress_reason(
        self, *, tool_calls: int, changed_paths: tuple[str, ...], infrastructure_failed: bool,
    ) -> str:
        """Name the unchanged evidence (requirement 41) plus what the turn did instead."""
        if infrastructure_failed:
            head = "infrastructure failure, which is never diagnostic progress"
        else:
            head = "no discriminating result"
        seen = ", ".join(self.unchanged_evidence) or "none recorded"
        tail = (
            f"{tool_calls} tool call(s) touched {len(changed_paths)} path(s) "
            f"({', '.join(changed_paths) if changed_paths else 'none'}) without narrowing the failure"
        )
        return f"{head}; unchanged evidence: {seen}; {tail}"

    def next_action(self) -> RepairAction:
        """Resolve what the host may do next, honestly and boundedly.

        A blocked or exhausted job keeps its checkpoint and its evidence; neither
        ever commits, pushes, or repeats.
        """
        if self.state is RepairJobState.COMPLETED:
            return RepairAction.COMPLETED
        if self.state in (RepairJobState.BLOCKED, RepairJobState.EXHAUSTED):
            return RepairAction(self.state.value)
        if self._exhaustion_reason() is not None:
            return RepairAction.EXHAUSTED
        return RepairAction.EXECUTE

    def with_state(self, state: RepairJobState, reason: str) -> RepairJobRecord:
        """Return a copy moved to ``state`` with a non-empty reason (never in place)."""
        if not reason.strip():
            raise ValueError("every repair job transition must state its reason")
        return replace(self, state=state, last_transition_reason=reason.strip())

    # -- persistence ------------------------------------------------------------------------

    def payload(self) -> dict[str, Any]:
        """JSON-serializable form carrying the format version and the captured policy."""
        return {
            "format_version": REPAIR_JOB_FORMAT,
            "job_id": self.job_id,
            "state": self.state.value,
            "failure_identity": self.failure_identity,
            "authorized_scope": list(self.authorized_scope),
            "policy_revision": self.policy_revision,
            "policy": self.policy.payload(),
            "consumed_s": self.consumed_s,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "granted_allowance_s": self.granted_allowance_s,
            "executions": self.executions,
            "scope_request": dict(self.scope_request) if self.scope_request is not None else None,
            "unchanged_evidence": list(self.unchanged_evidence),
            "progress_evidence": list(self.progress_evidence),
            "next_experiment": self.next_experiment,
            "completed_experiments": list(self.completed_experiments),
            "disproven_hypotheses": list(self.disproven_hypotheses),
            "last_transition_reason": self.last_transition_reason,
            "checkpoint_ref": self.checkpoint_ref,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> RepairJobRecord:
        """Rebuild a record from a stored payload; the job id is the only requirement.

        Every other field degrades to its default rather than raising, so a record
        written by an older execution still resumes. It is *not* a silent reset:
        whatever is readable is preserved, and the caller decides what an
        unreadable record means.
        """
        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("repair job record is missing its job_id")
        scope_request = raw.get("scope_request")
        return cls(
            job_id=job_id,
            state=_state(raw.get("state")),
            failure_identity=_opt_str(raw.get("failure_identity")) or "",
            authorized_scope=_str_tuple(raw.get("authorized_scope")),
            policy_revision=_plain_int(raw.get("policy_revision")) or 0,
            policy=RepairJobPolicy.from_payload(raw.get("policy")),
            consumed_s=max(0.0, _finite_float(raw.get("consumed_s")) or 0.0),
            cumulative_cost_usd=max(0.0, _finite_float(raw.get("cumulative_cost_usd")) or 0.0),
            granted_allowance_s=max(0.0, _finite_float(raw.get("granted_allowance_s")) or 0.0),
            executions=max(0, _plain_int(raw.get("executions")) or 0),
            scope_request=dict(scope_request) if isinstance(scope_request, Mapping) else None,
            unchanged_evidence=_str_tuple(raw.get("unchanged_evidence")),
            progress_evidence=_str_tuple(raw.get("progress_evidence")),
            next_experiment=_opt_str(raw.get("next_experiment")),
            completed_experiments=_str_tuple(raw.get("completed_experiments")),
            disproven_hypotheses=_str_tuple(raw.get("disproven_hypotheses")),
            last_transition_reason=_opt_str(raw.get("last_transition_reason")),
            checkpoint_ref=_opt_str(raw.get("checkpoint_ref")),
            diagnostics=_str_tuple(raw.get("diagnostics")),
        )


def read_repair_job_record(deep_dir_path: Path) -> RepairJobRecord | None:
    """Return the stored job record, or ``None`` when absent, foreign, or corrupt.

    A corrupt job record is deliberately not an empty one: ``read_json_object``
    cannot tell a corrupt file from an absent one, and inventing a fresh job here
    is exactly the silent-absence failure requirement 6 forbids. This returns
    ``None`` and leaves the "is recovery blocked?" judgement to the coordinator,
    which can name the corrupt path as a blocker.
    """
    raw = read_json_object(DeepArtifact.REPAIR_JOB.at(deep_dir_path))
    if not raw or raw.get("format_version") != REPAIR_JOB_FORMAT:
        return None
    try:
        return RepairJobRecord.from_payload(raw)
    except (TypeError, ValueError):
        return None


def write_repair_job_record(deep_dir_path: Path, job: RepairJobRecord) -> Path:
    """Replace the job record atomically and return the written path."""
    path = DeepArtifact.REPAIR_JOB.at(deep_dir_path)
    atomic_write_json(path, job.payload(), trailing_newline=True)
    return path


def record_diagnostic(deep_dir_path: Path, job_id: str, diagnostic: str) -> RepairJobRecord | None:
    """Append one named host degradation to an existing job record; return it, or ``None``.

    Used for failures the job cannot recover from inside its own turn — notably a
    repair checkpoint that could not be persisted. The write is best-effort by
    design only because its *caller* is the blocking outcome: this returns
    ``None`` when even the diagnostic could not be recorded.

    Requirement 28: the blocker is named in the job record too, so a resuming
    job sees why it stopped. That is why this records even when no record exists
    yet — the checkpoint-write failure this is called for happens *before* the
    job is created.

    Requirement 44 is honoured by the reader instead: a record minted here has
    ``policy_revision == 0`` and an empty ``authorized_scope`` (both filled from
    defaults), while a real job record is stamped by the coordinator with the
    run's footprint. :func:`daydream.deep.repair_coordinator._load_job` therefore
    never adopts such a record as the job's captured grant — it keeps the
    diagnostics and builds the job with the policy the run actually granted. So
    writing one here cannot silently replace the run's configured bounds.

    A diagnostic about a record belonging to another job is refused rather than
    relabelling it into this job's grant.
    """
    existing = read_repair_job_record(deep_dir_path)
    if existing is not None and existing.job_id != job_id:
        return None
    # Annotate the decoded record in place and persist it whole. ``replace`` on
    # the already-normalized record is what re-encoding through the payload used
    # to achieve, so every earlier execution's slices survive untouched.
    job = replace(
        existing if existing is not None else RepairJobRecord(job_id=job_id),
        diagnostics=(*(existing.diagnostics if existing is not None else ()), diagnostic),
    )
    try:
        write_repair_job_record(deep_dir_path, job)
    except OSError:
        return None
    return job


def _state(value: object) -> RepairJobState:
    if isinstance(value, RepairJobState):
        return value
    if isinstance(value, str):
        try:
            return RepairJobState(value)
        except ValueError:
            return RepairJobState.RUNNING
    return RepairJobState.RUNNING


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _extend(existing: tuple[str, ...], added: Sequence[str]) -> tuple[str, ...]:
    """Append names not already recorded, preserving order."""
    return (*existing, *(item for item in added if isinstance(item, str) and item not in existing))


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _plain_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


__all__ = [
    "REPAIR_JOB_FORMAT",
    "REPAIR_JOB_RESERVE_S",
    "RepairAction",
    "RepairJobPolicy",
    "RepairJobRecord",
    "RepairJobState",
    "read_repair_job_record",
    "record_diagnostic",
    "repair_job_id",
    "write_repair_job_record",
]
