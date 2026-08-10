"""DAYDREAM_SERVICE_V1 executor contract: neutral models + ``ReviewExecutor`` port.

These are the *common* types every registered executor adapter (Local,
Coder, Kubernetes, Sprites, ...) programs against and every conformance test
asserts. No vendor/SDK/worker-asserted infrastructure identity is allowed to
appear here:

- ``ExecutionRef`` is an opaque handle wrapped in a typed envelope. The
  controller never parses ``opaque_handle``; it only stores it and feeds it
  back to the originating executor.
- ``ExecutionSnapshot`` carries lifecycle state plus a bounded status. It has
  no fields for a pod, VM, lease, container, workspace hostname, or any other
  adapter-specific execution resource.
- ``ArtifactEnvelope`` carries the bounded review outcome (outcome enum,
  completed lens names, a content hash). It deliberately excludes adapter
  identity, credentials, and raw artifact contents.

Vendor errors are mapped *by the adapter* into the neutral ``ExecutorError``
hierarchy through :func:`map_vendor_error`; they never leak out of an adapter.

The port set is frozen (contract ``DAYDREAM_SERVICE_V1``):

``start(job) -> ExecutionRef``, ``inspect(ref) -> ExecutionSnapshot``,
``cancel(ref) -> None``, ``collect(ref) -> ArtifactEnvelope``,
``release(ref, disposition) -> None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

# Versioned service contract this module implements (independent of the
# DAYDREAM_EXT_API extension contract; both versioned separately).
DAYDREAM_SERVICE_V1: int = 1
MIN_SUPPORTED_DAYDREAM_SERVICE_V1: int = 1


class ExecutorCapability(str, Enum):
    """Capabilities an executor may declare; missing required ones fail admission."""

    EXCLUSIVE_WORKSPACE = "exclusive_workspace"
    NO_AMBIENT_CREDENTIALS = "no_ambient_credentials"
    SOURCE_READ_ONLY = "source_read_only"
    BOUNDED_EGRESS = "bounded_egress"
    DURABLE_EXECUTION_IDENTITY = "durable_execution_identity"
    STRONG_CANCEL = "strong_cancel"
    DETERMINISTIC_RELEASE = "deterministic_release"
    RESTART_RECONCILIATION = "restart_reconciliation"


# Every capability a merge-authorizing executor must prove. The controller
# admits an executor only when this set is a subset of the declared set; a
# missing capability fails admission (contract STOP condition).
REQUIRED_CAPABILITIES: frozenset[ExecutorCapability] = frozenset(ExecutorCapability)


class ExecutionStatus(str, Enum):
    """Neutral lifecycle position of one execution, as seen by :meth:`inspect`."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COLLECTING = "collecting"
    EVALUATED = "evaluated"
    RELEASED = "released"
    CANCELLED = "cancelled"
    INFRA_ERROR = "infra_error"


# Statuses from which an execution can no longer transition on its own.
_TERMINAL_STATUSES: frozenset[ExecutionStatus] = frozenset(
    {ExecutionStatus.EVALUATED, ExecutionStatus.RELEASED, ExecutionStatus.CANCELLED, ExecutionStatus.INFRA_ERROR}
)


def is_terminal(status: ExecutionStatus) -> bool:
    """Return True when *status* is a terminal lifecycle position."""
    return status in _TERMINAL_STATUSES


class ExecutionOutcome(str, Enum):
    """Bounded verdict an execution reports through :meth:`collect`."""

    CLEAN = "clean"
    FINDINGS = "findings"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ExecutionRef:
    """Typed identity of one execution.

    ``opaque_handle`` is produced and consumed only by the originating
    executor; the controller must never parse it. ``attempt_id`` makes the
    reference unique across retries of the same logical job.

    Attributes:
        executor_kind: Adapter kind, e.g. ``"local"``. Stable, non-secret.
        adapter_version: The adapter's DAYDREAM_SERVICE_V1 conformance version.
        opaque_handle: Opaque storage/lifecycle pointer owned by the adapter.
        attempt_id: Logical attempt identity; scopes a retry to a new ref.
    """

    executor_kind: str
    adapter_version: int
    opaque_handle: str
    attempt_id: str


@dataclass(frozen=True)
class ExecutionSnapshot:
    """Lifecycle observation returned by :meth:`ReviewExecutor.inspect`.

    No vendor/SDK/worker-asserted infrastructure identity is carried here —
    by design. Status transitions follow the neutral lifecycle:
    ``queued -> starting -> running -> collecting -> evaluated`` with
    ``cancelled`` / ``infra_error`` as side exits and ``released`` terminal.
    """

    ref: ExecutionRef
    status: ExecutionStatus
    started_at_iso: str | None = None
    completed_at_iso: str | None = None


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Bounded result of an execution, retrieved through :meth:`collect`.

    Carries only the review outcome surface: the terminal outcome, the
    completed lens inventory, and a content hash of the artifacts. Adapter
    identity, credentials, and raw artifact bytes are deliberately absent.
    """

    ref: ExecutionRef
    outcome: ExecutionOutcome
    completed_lenses: tuple[str, ...] = ()
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class ExecutorJob:
    """Neutral job handed to :meth:`ReviewExecutor.start`.

    ``payload`` is a bounded mapping the adapter interprets with its own
    vocabulary; it must never be a common-schema authority. Identity fields
    (``idempotency_key``, ``attempt_id``) are executor-agnostic.
    """

    attempt_id: str
    idempotency_key: str = ""
    payload: Mapping[str, object] = field(default_factory=dict)


class ExecutorError(Exception):
    """Base error for the executor seam. Adapters never raise raw vendor errors."""


class ExecutorCapabilityError(ExecutorError):
    """Capability admission failed: a required capability is undeclared."""


class UnknownExecutionError(ExecutorError):
    """The reference is unknown to the executor (never started or already forgotten)."""


class ExecutorInfrastructureError(ExecutorError):
    """A neutral infrastructure error, mapped from a vendor/SDK failure.

    Attributes:
        vendor_cause: The original exception when one was mapped (never
            re-raised as itself; kept for logging/triage only).
    """

    def __init__(self, message: str, *, vendor_cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.vendor_cause = vendor_cause


class CancelError(ExecutorError):
    """Execution could not be cancelled cleanly."""


def map_vendor_error(message: str, exc: BaseException) -> ExecutorError:
    """Map an adapter-internal vendor/SDK exception to a neutral executor error.

    Adapters must never let a vendor exception escape: the controller and
    conformance suite only understand :class:`ExecutorError`. This keeps
    vendor error types, SDK objects, and handles out of common models while
    preserving the original for triage.
    """
    return ExecutorInfrastructureError(message, vendor_cause=exc)


def require_capabilities(
    declared: set[ExecutorCapability] | frozenset[ExecutorCapability],
    *,
    kind: str,
    required: frozenset[ExecutorCapability] = REQUIRED_CAPABILITIES,
) -> None:
    """Admit an executor by capability set, raising on any missing required one."""
    missing = required - frozenset(declared)
    if missing:
        names = ", ".join(sorted(cap.value for cap in missing))
        raise ExecutorCapabilityError(
            f"executor {kind!r} lacks required capabilities: {names}; capability admission fails"
        )
