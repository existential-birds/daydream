"""Unified neutral models for the durable review service (Plan 008, consolidated).

This module is the single home for the frozen data contracts the review service
shares across every consuming leaf: the exact-candidate job model, the policy /
evaluation model, and the controller's persisted row. It deliberately carries
**no vendor fields**: no Sprites, Coder, Kubernetes, provider, lease, pod/VM, or
worker-asserted infrastructure identity ever appears in a common model.

Terminology (per the public plan): ``backend`` is the Daydream model-agent
driver (Claude/Codex/Pi), ``provider`` is the model endpoint provider, and
``executor`` is the compute/workspace adapter. None of these overload the
``daydream.Backend`` agent driver.

Contract families:

- ``ReviewTargetV1`` / ``ReviewJobV1`` — the immutable, exact-candidate job that
  a service-mode worker runs (REVIEW_TARGET_V1). ``from_dict`` enforces
  ``additionalProperties=False`` semantics at every nesting level.
- Policy models — ``ReviewTarget`` / ``ReviewPolicy`` / ``RoundRecord`` /
  ``LensInventory`` / ``SourceOfTruth`` and the enums the fail-closed
  ``PolicyEvaluator`` and the trusted publisher consume.
- Controller models — ``CandidateTarget`` / ``JobSpec`` / ``ControllerRecord``
  and the verdict / capability constants the durable controller persists.

The canonical execution types (:class:`daydream.executors.contract.ExecutionRef`,
``ExecutionSnapshot``, ``ArtifactEnvelope``) are defined by the conformance suite
in :mod:`daydream.executors.contract`; the controller consumes those through its
port rather than re-defining them here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from daydream.service.states import ServiceState

# ==========================================================================
# REVIEW_TARGET_V1 / immutable job (leaf-A, worker/artifact)
# ==========================================================================

_HEX_RE = re.compile(r"[0-9a-fA-F]+")


def _validate_hex(value: object, length: int, name: str) -> None:
    """Reject *value* unless it is a *length*-character hex string."""
    if not isinstance(value, str) or len(value) != length or not _HEX_RE.fullmatch(value):
        raise ValueError(f"{name} must be a {length}-character hex string")


def _validate_iso_utc(value: object, name: str) -> None:
    """Reject *value* unless it parses as a timezone-aware UTC ISO 8601 string."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO 8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 UTC string: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC: {value!r}")


def _required(data: dict[str, Any], key: str) -> Any:
    """Return ``data[key]``, raising ``ValueError`` when the field is absent."""
    if key not in data:
        raise ValueError(f"missing required field {key!r}")
    return data[key]


@dataclass(frozen=True)
class ReviewTargetV1:
    """The exact, immutable review target.

    Attributes:
        target_kind: ``"pr_head"`` (a pull-request head) or ``"merge_group"``
            (an opaque merge-group identity).
        repo: ``owner/repo`` slug.
        candidate_sha: Exact 40-hex candidate commit SHA (the review target).
        candidate_tree_digest: Exact tree digest of the candidate commit.
        base_sha: Exact base commit SHA the diff is computed against.
        pr_numbers: PR numbers relevant to a ``pr_head`` target; must be
            non-empty for ``pr_head`` and empty for ``merge_group``.
        merge_group_id: Opaque merge-group identity; required for
            ``merge_group`` and forbidden for ``pr_head``.
        full_diff_digest: Canonical SHA-256 (64-hex) of the full
            ``base_sha..candidate_sha`` diff.
        protected_config_ref: Ref/SHA the trusted review config came from, or
            None when no protected config source applies.
        protected_config_digest: Digest of the protected config content.
        invalidation_id: Opaque invalidate-on-next-job identifier so a fresh
            target can provably supersede a stale one.
    """

    target_kind: Literal["pr_head", "merge_group"]
    repo: str
    candidate_sha: str
    candidate_tree_digest: str
    base_sha: str
    full_diff_digest: str
    invalidation_id: str
    pr_numbers: tuple[int, ...] = ()
    merge_group_id: str | None = None
    protected_config_ref: str | None = None
    protected_config_digest: str | None = None

    def __post_init__(self) -> None:
        if self.target_kind not in ("pr_head", "merge_group"):
            raise ValueError(
                f"unknown target_kind {self.target_kind!r}; expected 'pr_head' or 'merge_group'"
            )
        if not isinstance(self.repo, str) or "/" not in self.repo or self.repo.startswith("/"):
            raise ValueError(f"repo must be an 'owner/repo' slug, got {self.repo!r}")
        _validate_hex(self.candidate_sha, 40, "candidate_sha")
        _validate_hex(self.candidate_tree_digest, 40, "candidate_tree_digest")
        _validate_hex(self.base_sha, 40, "base_sha")
        _validate_hex(self.full_diff_digest, 64, "full_diff_digest")
        if not isinstance(self.invalidation_id, str) or not self.invalidation_id:
            raise ValueError("invalidation_id must be a non-empty string")

        if self.target_kind == "pr_head":
            if not isinstance(self.pr_numbers, tuple) or not self.pr_numbers:
                raise ValueError("pr_head target requires non-empty pr_numbers")
            if not all(isinstance(n, int) and n > 0 for n in self.pr_numbers):
                raise ValueError("pr_numbers must be positive integers")
            if self.merge_group_id is not None:
                raise ValueError("pr_head target must not carry a merge_group_id")
        else:
            if not isinstance(self.merge_group_id, str) or not self.merge_group_id:
                raise ValueError("merge_group target requires a non-empty merge_group_id")
            if self.pr_numbers:
                raise ValueError("merge_group target must have empty pr_numbers")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (lists for tuples)."""
        return {
            "target_kind": self.target_kind,
            "repo": self.repo,
            "candidate_sha": self.candidate_sha,
            "candidate_tree_digest": self.candidate_tree_digest,
            "base_sha": self.base_sha,
            "pr_numbers": list(self.pr_numbers),
            "merge_group_id": self.merge_group_id,
            "full_diff_digest": self.full_diff_digest,
            "protected_config_ref": self.protected_config_ref,
            "protected_config_digest": self.protected_config_digest,
            "invalidation_id": self.invalidation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewTargetV1:
        """Deserialize strictly: unknown fields and missing required fields raise."""
        unknown = set(data) - _TARGET_KEYS
        if unknown:
            raise ValueError(f"unexpected fields on ReviewTargetV1: {sorted(unknown)}")
        pr_numbers = _required(data, "pr_numbers")
        if isinstance(pr_numbers, list):
            pr_numbers = tuple(pr_numbers)
        return cls(
            target_kind=_required(data, "target_kind"),
            repo=_required(data, "repo"),
            candidate_sha=_required(data, "candidate_sha"),
            candidate_tree_digest=_required(data, "candidate_tree_digest"),
            base_sha=_required(data, "base_sha"),
            pr_numbers=pr_numbers,
            merge_group_id=data.get("merge_group_id"),
            full_diff_digest=_required(data, "full_diff_digest"),
            invalidation_id=_required(data, "invalidation_id"),
            protected_config_ref=data.get("protected_config_ref"),
            protected_config_digest=data.get("protected_config_digest"),
        )


@dataclass(frozen=True)
class ReviewJobV1:
    """An immutable, validated service review job wrapping one target.

    Attributes:
        job_id: Controller-side job identifier.
        idempotency_key: Dedup key; identical jobs re-run to one artifact.
        target: The exact review target (see :class:`ReviewTargetV1`).
        effective_config_digest: Digest of the trusted effective config the
            reviewer bundle ran under.
        reviewer_bundle_digest: Digest of the immutable reviewer bundle.
        required_lenses: Every lens that MUST complete; non-empty.
        round: Logical full-review round, >= 1.
        attempt: Logical attempt within the round, >= 1.
        deadline: ISO 8601 UTC deadline.
        created_at: ISO 8601 UTC creation timestamp.
    """

    job_id: str
    idempotency_key: str
    target: ReviewTargetV1
    effective_config_digest: str
    reviewer_bundle_digest: str
    required_lenses: tuple[str, ...]
    round: int
    attempt: int
    deadline: str
    created_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        if not isinstance(self.effective_config_digest, str) or not self.effective_config_digest:
            raise ValueError("effective_config_digest must be a non-empty string")
        if not isinstance(self.reviewer_bundle_digest, str) or not self.reviewer_bundle_digest:
            raise ValueError("reviewer_bundle_digest must be a non-empty string")
        if (
            not isinstance(self.required_lenses, tuple)
            or not self.required_lenses
            or not all(isinstance(lens, str) and lens for lens in self.required_lenses)
        ):
            raise ValueError("required_lenses must be a non-empty tuple of non-empty strings")
        if not isinstance(self.round, int) or self.round < 1:
            raise ValueError("round must be an integer >= 1")
        if not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("attempt must be an integer >= 1")
        _validate_iso_utc(self.deadline, "deadline")
        _validate_iso_utc(self.created_at, "created_at")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (target nested, lenses as a list)."""
        return {
            "job_id": self.job_id,
            "idempotency_key": self.idempotency_key,
            "target": self.target.to_dict(),
            "effective_config_digest": self.effective_config_digest,
            "reviewer_bundle_digest": self.reviewer_bundle_digest,
            "required_lenses": list(self.required_lenses),
            "round": self.round,
            "attempt": self.attempt,
            "deadline": self.deadline,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewJobV1:
        """Deserialize strictly: unknown fields and missing required fields raise."""
        unknown = set(data) - _JOB_KEYS
        if unknown:
            raise ValueError(f"unexpected fields on ReviewJobV1: {sorted(unknown)}")
        target = _required(data, "target")
        if not isinstance(target, dict):
            raise ValueError("target must be an object")
        required_lenses = _required(data, "required_lenses")
        if isinstance(required_lenses, list):
            required_lenses = tuple(required_lenses)
        return cls(
            job_id=_required(data, "job_id"),
            idempotency_key=_required(data, "idempotency_key"),
            target=ReviewTargetV1.from_dict(target),
            effective_config_digest=_required(data, "effective_config_digest"),
            reviewer_bundle_digest=_required(data, "reviewer_bundle_digest"),
            required_lenses=required_lenses,
            round=_required(data, "round"),
            attempt=_required(data, "attempt"),
            deadline=_required(data, "deadline"),
            created_at=_required(data, "created_at"),
        )


# ==========================================================================
# Policy / evaluation models (leaf-E, publisher-policy)
# ==========================================================================

SCHEMA_VERSION = 1

PayloadKind = Literal["pr_head", "merge_group"]


class TargetKind(Enum):
    """The kind of review candidate. Unknown target kinds are rejected at admission."""

    PR_HEAD = "pr_head"
    MERGE_GROUP = "merge_group"


class TerminalOutcome(Enum):
    """A round's terminal result. Only CLEAN can ever contribute to success."""

    CLEAN = "clean"
    FINDINGS = "findings"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"


class PolicyDecision(Enum):
    """Fail-closed terminal policy decision."""

    SUCCESS = "success"
    FAIL = "fail"


class SourceOfTruth:
    """A protected configuration source: base/default-branch ref, SHA, and digest.

    Attributes:
        ref: The git ref the protected policy snapshot was resolved from.
        sha: The commit SHA the snapshot was read at.
        digest: Canonical effective-config digest of that snapshot.
    """

    __slots__ = ("ref", "sha", "digest")

    def __init__(self, *, ref: str, sha: str, digest: str) -> None:
        self.ref = ref
        self.sha = sha
        self.digest = digest

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SourceOfTruth)
            and self.ref == other.ref
            and self.sha == other.sha
            and self.digest == other.digest
        )

    def __hash__(self) -> int:
        return hash((self.ref, self.sha, self.digest))

    def __repr__(self) -> str:
        return f"SourceOfTruth(ref={self.ref!r}, sha={self.sha!r})"


class ReviewTarget:
    """The exact candidate a review job must review and a Check binds to.

    Distinguishes ``pr_head`` from ``merge_group`` and always carries the exact
    candidate SHA and tree digest, base SHA, relevant PR/merge-group identity,
    the canonical full-diff digest, the protected config source, and an
    invalidation id used to cancel superseded work. A merge-group job may never
    substitute a constituent PR head; that is enforced by the target kind.
    """

    __slots__ = (
        "repo",
        "kind",
        "candidate_sha",
        "candidate_tree",
        "base_sha",
        "pr_number",
        "merge_group_id",
        "diff_digest",
        "config_source",
        "invalidation_id",
        "resolved_at",
    )

    def __init__(
        self,
        *,
        repo: str,
        kind: TargetKind,
        candidate_sha: str | None,
        candidate_tree: str,
        base_sha: str,
        pr_number: int | None,
        merge_group_id: str | None,
        diff_digest: str,
        config_source: SourceOfTruth,
        invalidation_id: str,
        resolved_at: datetime | None = None,
    ) -> None:
        self.repo = repo
        self.kind = kind
        self.candidate_sha = candidate_sha
        self.candidate_tree = candidate_tree
        self.base_sha = base_sha
        self.pr_number = pr_number
        self.merge_group_id = merge_group_id
        self.diff_digest = diff_digest
        self.config_source = config_source
        self.invalidation_id = invalidation_id
        self.resolved_at = resolved_at or datetime.now(timezone.utc)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """The full identity this target binds to (SHA, tree, diff, config digest)."""
        return (
            self.candidate_sha or "",
            self.candidate_tree,
            self.diff_digest,
            self.config_source.digest,
        )

    @property
    def payload_kind(self) -> PayloadKind:
        return self.kind.value

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ReviewTarget)
            and self.repo == other.repo
            and self.kind == other.kind
            and self.candidate_sha == other.candidate_sha
            and self.candidate_tree == other.candidate_tree
            and self.base_sha == other.base_sha
            and self.pr_number == other.pr_number
            and self.merge_group_id == other.merge_group_id
            and self.diff_digest == other.diff_digest
            and self.config_source == other.config_source
        )

    def __hash__(self) -> int:
        return hash((self.repo, self.kind, self.candidate_sha, self.candidate_tree, self.diff_digest))

    def __repr__(self) -> str:
        return (
            f"ReviewTarget(repo={self.repo!r}, kind={self.kind.value!r}, "
            f"sha={self.candidate_sha!r})"
        )


class LensInventory:
    """The complete lens inventory a full review must cover.

    Attributes:
        required: Every lens a round must complete to count as a full review.
    """

    __slots__ = ("required",)

    def __init__(self, *, required: set[str]) -> None:
        if not required:
            raise ValueError("a review policy must require at least one lens")
        self.required = frozenset(required)

    def is_subset(self, completed: set[str] | frozenset[str]) -> bool:
        return self.required.issubset(completed)

    def missing(self, completed: set[str] | frozenset[str]) -> set[str]:
        return set(self.required - frozenset(completed))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LensInventory) and self.required == other.required

    def __hash__(self) -> int:
        return hash(self.required)


class ReviewPolicy:
    """The protected service policy the evaluator applies.

    Logical fields (non-secret): backend/model provider/model, required full
    rounds, complete-lens policy, immutable reviewer bundle, deadlines and hard
    budgets, executor name plus namespaced non-secret options, publisher name,
    exact Check identity, and the protected config source plus its canonical
    effective digest. No value here is hard-coded by any evaluator.
    """

    __slots__ = (
        "backend",
        "provider",
        "model",
        "required_rounds",
        "complete_lens",
        "executor",
        "concurrent_rounds",
        "immutable_reviewer_bundle",
        "deadline_s",
        "hard_budget_s",
        "publisher",
        "check_name",
        "source",
    )

    def __init__(
        self,
        *,
        backend: str,
        provider: str,
        model: str,
        required_rounds: int,
        complete_lens: LensInventory,
        executor: str,
        concurrent_rounds: bool,
        immutable_reviewer_bundle: str,
        deadline_s: float,
        hard_budget_s: float,
        publisher: str,
        check_name: str,
        source: SourceOfTruth,
    ) -> None:
        if required_rounds <= 0:
            raise ValueError("required_rounds must be >= 1")
        if deadline_s <= 0 or hard_budget_s <= 0:
            raise ValueError("deadlines and budgets must be positive")
        self.backend = backend
        self.provider = provider
        self.model = model
        self.required_rounds = required_rounds
        self.complete_lens = complete_lens
        self.executor = executor
        self.concurrent_rounds = concurrent_rounds
        self.immutable_reviewer_bundle = immutable_reviewer_bundle
        self.deadline_s = deadline_s
        self.hard_budget_s = hard_budget_s
        self.publisher = publisher
        self.check_name = check_name
        self.source = source

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ReviewPolicy)
            and self.backend == other.backend
            and self.provider == other.provider
            and self.model == other.model
            and self.required_rounds == other.required_rounds
            and self.complete_lens == other.complete_lens
            and self.executor == other.executor
            and self.concurrent_rounds == other.concurrent_rounds
            and self.immutable_reviewer_bundle == other.immutable_reviewer_bundle
            and self.deadline_s == other.deadline_s
            and self.hard_budget_s == other.hard_budget_s
            and self.publisher == other.publisher
            and self.check_name == other.check_name
            and self.source == other.source
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.backend,
                self.provider,
                self.model,
                self.required_rounds,
                self.complete_lens,
                self.executor,
                self.concurrent_rounds,
                self.immutable_reviewer_bundle,
                self.deadline_s,
                self.hard_budget_s,
                self.publisher,
                self.check_name,
            )
        )

    def __repr__(self) -> str:
        return (
            f"ReviewPolicy(rounds={self.required_rounds}, backend={self.backend!r}, "
            f"executor={self.executor!r}, publisher={self.publisher!r}, "
            f"check={self.check_name!r})"
        )


class RoundRecord:
    """One completed (or terminal) review round, captured from a worker artifact.

    Passively reported by the worker; the trusted controller validates and binds
    it to its execution ref. It carries the target identity the round actually
    reviewed, completed lenses, the process outcome, the finding count, and
    whether the artifact set was complete.
    """

    __slots__ = (
        "attempt_id",
        "target",
        "outcome",
        "completed_lenses",
        "finding_count",
        "partial_artifacts",
        "execution_ref",
    )

    def __init__(
        self,
        *,
        attempt_id: str,
        target: ReviewTarget,
        outcome: TerminalOutcome,
        completed_lenses: set[str],
        finding_count: int = 0,
        partial_artifacts: bool = False,
        execution_ref: str = "",
    ) -> None:
        if finding_count < 0:
            raise ValueError("finding_count must be >= 0")
        self.attempt_id = attempt_id
        self.target = target
        self.outcome = outcome
        self.completed_lenses = frozenset(completed_lenses)
        self.finding_count = finding_count
        self.partial_artifacts = partial_artifacts
        self.execution_ref = execution_ref

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RoundRecord) and self.attempt_id == other.attempt_id

    def __hash__(self) -> int:
        return hash(self.attempt_id)


# ==========================================================================
# Controller models (leaf-B, controller-admission)
# ==========================================================================


@dataclass(frozen=True)
class CandidateTarget:
    """Identifies the exact candidate a controller job reviews.

    Bound to an exact candidate SHA/tree and a protected config source; a push
    or replaced merge-group candidate bumps ``invalidation_id`` so all earlier
    rounds are rejected before they can publish.
    """

    target_kind: str
    repo: str
    candidate_sha: str
    tree_digest: str
    base_sha: str
    invalidation_id: int


@dataclass(frozen=True)
class ExecutionRef:
    """Opaque execution reference returned by ``ReviewExecutor.start``.

    The controller stores this as an opaque value and binds collected artifact
    hashes to it. It must never parse ``opaque_handle`` — on restart it hands the
    handle back to the registered executor via ``inspect``.
    """

    executor_kind: str
    adapter_version: str
    opaque_handle: str
    attempt_id: str


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Passive artifact a worker produces; validated by the controller.

    Carries bounded hashes + timestamps and a terminal worker verdict. It must
    not contain an executor/provider/lease or other worker-asserted
    infrastructure identity — the controller binds hashes to the separately
    stored ``ExecutionRef``.
    """

    worker_verdict: str
    completed_lenses: frozenset[str] = frozenset()
    artifact_hashes: tuple[str, ...] = ()
    blocked: bool = False


@dataclass(frozen=True)
class ExecutionSnapshot:
    """What ``ReviewExecutor.inspect`` reports about a stored opaque ref.

    Lets a restarting controller reconcile the persisted state to reality
    without ever parsing the handle. The controller only consults ``running``
    and ``terminal``; the canonical conformance suite in
    :mod:`daydream.executors.contract` defines a richer status-shaped snapshot
    that hermetic adapters return, which the executor bridge normalizes onto
    this surface.
    """

    running: bool
    terminal: bool
    detail: tuple[str, ...] = ()


@dataclass(frozen=True)
class JobSpec:
    """The immutable, complete description of one review job.

    Attributes:
        job_id: Stable, idempotent identity for the job (one per candidate+round).
        candidate: The exact candidate this job reviews.
        service: Logical service/repository namespace for per-service budgets.
        backend: Daydream agent driver name (``"claude"``/``"codex"``/``"pi"``).
        provider: Model endpoint provider name.
        model: Concrete model id the provider will serve.
        required_lenses: Complete lens inventory the round must cover; a missing
            lens is an infrastructure error, never a clean verdict.
        attempt_number: Logical attempt within the round (>= 1).
    """

    job_id: str
    candidate: CandidateTarget
    service: str
    backend: str
    provider: str
    model: str
    required_lenses: frozenset[str] = frozenset()
    attempt_number: int = 1


@dataclass(frozen=True)
class ControllerRecord:
    """The durable controller record persisted via the storage port.

    Binds the opaque execution reference *separately* from collected artifact
    hashes, so a stale or late artifact for a superseded attempt can be rejected
    without disturbing the live execution reference.
    """

    job_id: str
    spec: JobSpec
    state: ServiceState = ServiceState.QUEUED
    execution_ref: ExecutionRef | None = None
    artifact_hashes: tuple[str, ...] = ()
    worker_verdict: str | None = None
    retries_used: int = 0
    superseded_by: int | None = None
    # Logical webhook/trigger causal markers; opaque to this leaf.
    blocked: bool = False
    completed_lenses: frozenset[str] = frozenset()
    trigger_ref: str | None = None
    # Storage-slot reserved for the store leaf (CAS/lease/attempt-history).
    store_fields: dict[str, str] = field(default_factory=dict)


# Worker verdicts — the only terminal outcomes a passive artifact may assert.
VERDICT_CLEAN = "clean"
VERDICT_FINDINGS = "findings"
VERDICT_INFRA_ERROR = "infra_error"
VERDICT_CANCELLED = "cancelled"

VALID_WORKER_VERDICTS = frozenset(
    {VERDICT_CLEAN, VERDICT_FINDINGS, VERDICT_INFRA_ERROR, VERDICT_CANCELLED}
)


# Execution capabilities every admitted executor must prove (DAYSERVICE V1).
CAPABILITY_EXCLUSIVE_WORKSPACE = "exclusive_workspace"
CAPABILITY_NO_AMBIENT_CREDENTIALS = "no_ambient_credentials"
CAPABILITY_SOURCE_READ_ONLY = "source_read_only"
CAPABILITY_BOUNDED_EGRESS = "bounded_egress"
CAPABILITY_DURABLE_EXECUTION_IDENTITY = "durable_execution_identity"
CAPABILITY_STRONG_CANCEL = "strong_cancel"
CAPABILITY_DETERMINISTIC_RELEASE = "deterministic_release"
CAPABILITY_RESTART_RECONCILIATION = "restart_reconciliation"

REQUIRED_CAPABILITIES = frozenset(
    {
        CAPABILITY_EXCLUSIVE_WORKSPACE,
        CAPABILITY_NO_AMBIENT_CREDENTIALS,
        CAPABILITY_SOURCE_READ_ONLY,
        CAPABILITY_BOUNDED_EGRESS,
        CAPABILITY_DURABLE_EXECUTION_IDENTITY,
        CAPABILITY_STRONG_CANCEL,
        CAPABILITY_DETERMINISTIC_RELEASE,
        CAPABILITY_RESTART_RECONCILIATION,
    }
)


_TARGET_KEYS = frozenset(ReviewTargetV1.__dataclass_fields__)
_JOB_KEYS = frozenset(ReviewJobV1.__dataclass_fields__)
