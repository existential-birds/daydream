"""Immutable neutral models for the durable review service (Plan 008 Step 3).

These are the typed values the controller persists and the storage/executor
ports exchange. They deliberately carry **no vendor fields**: no Sprites, Coder,
Kubernetes, provider, or worker-asserted infrastructure identity ever appears
in a common model. The ``ExecutionRef`` is opaque — the controller binds it to
collected artifact hashes but never parses its handle; only the registered
executor may inspect it.

Terminology (per the public plan): ``backend`` is the Daydream model-agent
driver (Claude/Codex/Pi), ``provider`` is the Pi/model endpoint provider, and
``executor`` is the compute/workspace adapter (Sprites, Coder, local,
Kubernetes). None of these overload the ``daydream.Backend`` agent driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from daydream.service.states import ServiceState


@dataclass(frozen=True)
class CandidateTarget:
    """Identifies the exact candidate a review round must cover.

    Mirrors the frozen ``REVIEW_TARGET_V1`` boundaries needed by the controller:
    a merge-authorizing job is bound to an exact candidate SHA/tree and one
    protected config source; a push or replaced merge-group candidate changes
    ``invalidation_id`` so all earlier rounds are rejected before they can
    publish.

    Attributes:
        target_kind: ``"pr_head"`` or ``"merge_group"``. Unknown kinds fail admission.
        repo: Declared ``owner/repo`` slug.
        candidate_sha: Exact candidate commit SHA to review.
        tree_digest: Exact tree digest of ``candidate_sha``.
        base_sha: Base/default-branch SHA to diff against.
        invalidation_id: Monotonic identity bumped on every superseding push so
            earlier candidates can be shown stale.
    """

    target_kind: str
    repo: str
    candidate_sha: str
    tree_digest: str
    base_sha: str
    invalidation_id: int


@dataclass(frozen=True)
class JobSpec:
    """The immutable, complete description of one review job.

    Attributes:
        job_id: Stable, idempotent identity for the job (one per candidate+round).
        candidate: The exact candidate this job reviews.
        service: Logical service/repository namespace for per-service budgets.
        backend: Daydream agent driver name (``"claude"``/``"codex"``/``"pi"``).
        provider: Model endpoint provider name (e.g. ``"nous"``/``"anthropic"``).
        model: Concrete model id the provider will serve.
        required_lenses: Complete lens inventory the round must cover; a missing
            lens is an infrastructure error, never a clean verdict.
        round_number and attempt_number are logical identities supplied by the
        policy evaluator; this leaf binds and persists them without interpreting
        the round count.
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
class ExecutionRef:
    """Opaque execution reference returned by ``ReviewExecutor.start``.

    The controller stores this as an opaque value and binds collected artifact
    hashes to it. It must never parse ``opaque_handle`` — on restart it hands the
    handle back to the registered executor via ``inspect``.

    Attributes:
        executor_kind: Which executor produced the ref (e.g. ``"local"``).
        adapter_version: The adapter's schema/behavior version.
        opaque_handle: Executor-private handle; opaque to the controller.
        attempt_id: Identity of the attempt this ref belongs to.
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

    Attributes:
        worker_verdict: ``clean`` | ``findings`` | ``infra_error`` | ``cancelled``.
        completed_lenses: Which required lenses the artifact actually covered.
        artifact_hashes: Bounded SHA-256 hashes of the raw artifact blobs.
        blocked: True when a blocking finding is present (findings even on exit 0).
    """

    worker_verdict: str
    completed_lenses: frozenset[str] = frozenset()
    artifact_hashes: Tuple[str, ...] = ()
    blocked: bool = False


@dataclass(frozen=True)
class ExecutionSnapshot:
    """What ``ReviewExecutor.inspect`` reports about a stored opaque ref.

    Lets a restarting controller reconcile the persist-ed state to reality
    without ever parsing the handle.

    Attributes:
        running: Whether the execution still appears active.
        terminal: Whether the execution has reached a terminal worker outcome.
        detail: Opaque adapter detail (never parsed by the controller).
    """

    running: bool
    terminal: bool
    detail: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControllerRecord:
    """The durable controller record persisted via the storage port.

    Binds the opaque execution reference *separately* from collected artifact
    hashes, so a stale or late artifact for a superseded attempt can be
    rejected without disturbing the live execution reference.

    Attributes:
        job_id: Idempotent job identity.
        spec: The immutable job spec.
        state: Current neutral state of the job.
        execution_ref: Opaque execution reference, or None before start.
        artifact_hashes: Collected artifact hashes, or empty before collect.
        worker_verdict: Terminal worker verdict once collected, else None.
        retries_used: Number of infra-failure retries consumed.
        superseded_by: invalidation_id that superseded this job, if any.
    """

    job_id: str
    spec: JobSpec
    state: ServiceState = ServiceState.QUEUED
    execution_ref: ExecutionRef | None = None
    artifact_hashes: Tuple[str, ...] = ()
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
