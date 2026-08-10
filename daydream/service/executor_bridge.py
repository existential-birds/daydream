"""Bridge between the canonical executor contract and the controller port.

The controller (Plan 008 leaf-B) programs against a narrow, controller-shaped
``ReviewExecutor`` surface: ``start(spec)``, ``collect`` returning a
``service.models.ArtifactEnvelope``, and ``inspect`` returning a
``service.models.ExecutionSnapshot`` with ``running``/``terminal``. The
conformance-tested executor adapters (leaf-D) implement the canonical
DAYDREAM_SERVICE_V1 port in :mod:`daydream.executors` instead: ``start(ExecutorJob)``
returning an opaque ``executors.contract.ExecutionRef``, plus canonical
``ExecutionSnapshot`` (status-based) and ``ArtifactEnvelope`` (outcome-based).

This module is the single seam that normalizes a canonical ``ReviewExecutor``
onto the controller-facing surface, so the controller can drive any registered
adapter (Local, Scripted, Sprites, ...) without leaking canonical/vendor types
into ``ControllerRecord``. It deliberately keeps the controller's opaque ref and
the canonical opaque ref as separate values — the controller stores the
controller-shaped ref and the bridge keeps the canonical handle to resolve it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daydream.executors import (
    ArtifactEnvelope as CanonicalArtifactEnvelope,
)
from daydream.executors import (
    ExecutionOutcome,
    ExecutionStatus,
    ExecutorJob,
)
from daydream.executors import (
    ExecutionRef as CanonicalExecutionRef,
)
from daydream.executors import (
    ExecutionSnapshot as CanonicalExecutionSnapshot,
)
from daydream.executors import (
    ReviewExecutor as CanonicalReviewExecutor,
)
from daydream.executors.contract import is_terminal
from daydream.service.models import (
    VERDICT_CANCELLED,
    VERDICT_CLEAN,
    VERDICT_FINDINGS,
    VERDICT_INFRA_ERROR,
    ArtifactEnvelope,
    ExecutionRef,
    ExecutionSnapshot,
    JobSpec,
)

# Verdict the canonical execution outcome maps to on the controller surface.
_OUTCOME_TO_VERDICT = {
    ExecutionOutcome.CLEAN: VERDICT_CLEAN,
    ExecutionOutcome.FINDINGS: VERDICT_FINDINGS,
    ExecutionOutcome.INFRA_ERROR: VERDICT_INFRA_ERROR,
    ExecutionOutcome.CANCELLED: VERDICT_CANCELLED,
}


@dataclass
class _Binding:
    """Controller-shaped ref + the canonical ref + whether it was released."""

    controller_ref: ExecutionRef
    canonical_ref: CanonicalExecutionRef
    released: bool = False
    job: ExecutorJob | None = None


class ExecutionBridge:
    """Present a canonical ``ReviewExecutor`` to the controller seam.

    Args:
        executor: A canonical ``daydream.executors.protocol.ReviewExecutor``.
        resolve_attempt_id: Optional callable mapping a ``JobSpec`` to the logical
            attempt id used to scope the canonical execution (defaults to
            ``spec.attempt_number`` as a string).
    """

    def __init__(
        self,
        executor: CanonicalReviewExecutor,
        *,
        resolve_attempt_id: Any = None,
    ) -> None:
        self._executor = executor
        self._resolve_attempt_id = resolve_attempt_id
        self._bindings: dict[str, _Binding] = {}

    @property
    def kind(self) -> str:
        return self._executor.kind

    def _attempt_id(self, spec: JobSpec) -> str:
        if self._resolve_attempt_id is not None:
            return self._resolve_attempt_id(spec)
        return str(spec.attempt_number)

    async def start(self, spec: JobSpec) -> ExecutionRef:
        """Begin a canonical execution for *spec*, returning a controller ref."""
        attempt_id = self._attempt_id(spec)
        idempotency_key = getattr(spec, "job_id", "") or spec.job_id
        job = ExecutorJob(
            attempt_id=attempt_id,
            idempotency_key=idempotency_key,
            payload={
                "job_id": spec.job_id,
                "service": spec.service,
                "backend": spec.backend,
                "provider": spec.provider,
                "model": spec.model,
                "required_lenses": sorted(spec.required_lenses),
                "candidate": {
                    "target_kind": spec.candidate.target_kind,
                    "repo": spec.candidate.repo,
                    "candidate_sha": spec.candidate.candidate_sha,
                    "candidate_tree_digest": spec.candidate.tree_digest,
                    "base_sha": spec.candidate.base_sha,
                    "invalidation_id": spec.candidate.invalidation_id,
                },
            },
        )
        canonical = await self._executor.start(job)
        controller_ref = ExecutionRef(
            executor_kind=canonical.executor_kind,
            adapter_version=str(canonical.adapter_version),
            opaque_handle=canonical.opaque_handle,
            attempt_id=canonical.attempt_id,
        )
        self._bindings[controller_ref.opaque_handle] = _Binding(
            controller_ref=controller_ref,
            canonical_ref=canonical,
            job=job,
        )
        return controller_ref

    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot:
        """Report live state of *ref* as a controller snapshot."""
        canonical = self._resolve(ref)
        snapshot = await self._executor.inspect(canonical)
        return self._normalize_snapshot(snapshot)

    async def cancel(self, ref: ExecutionRef) -> None:
        """Strongly cancel *ref*."""
        canonical = self._resolve(ref)
        await self._executor.cancel(canonical)

    async def collect(self, ref: ExecutionRef) -> ArtifactEnvelope:
        """Collect the bounded artifacts for *ref* as a controller envelope."""
        canonical = self._resolve(ref)
        envelope = await self._executor.collect(canonical)
        return self._normalize_envelope(envelope)

    async def release(self, ref: ExecutionRef, disposition: str) -> None:
        """Deterministically release *ref*."""
        binding = self._bindings.get(ref.opaque_handle)
        if binding is not None and not binding.released:
            await self._executor.release(binding.canonical_ref, disposition)
            binding.released = True

    def _resolve(self, ref: ExecutionRef) -> CanonicalExecutionRef:
        binding = self._bindings.get(ref.opaque_handle)
        if binding is None:
            # Unknown to this bridge; build a canonical ref purely from the
            # controller-shaped ref so the executor can still report/refuse it.
            return CanonicalExecutionRef(
                executor_kind=ref.executor_kind,
                adapter_version=int(ref.adapter_version) if ref.adapter_version.isdigit() else 1,
                opaque_handle=ref.opaque_handle,
                attempt_id=ref.attempt_id,
            )
        return binding.canonical_ref

    @staticmethod
    def _normalize_snapshot(snapshot: CanonicalExecutionSnapshot) -> ExecutionSnapshot:
        status: ExecutionStatus = snapshot.status
        running = status in (
            ExecutionStatus.QUEUED,
            ExecutionStatus.STARTING,
            ExecutionStatus.RUNNING,
            ExecutionStatus.COLLECTING,
            ExecutionStatus.EVALUATED,
        )
        terminal = (
            status in (ExecutionStatus.RELEASED, ExecutionStatus.CANCELLED, ExecutionStatus.INFRA_ERROR)
            or is_terminal(status)
        )
        return ExecutionSnapshot(running=running, terminal=terminal, detail=(status.value,))

    @staticmethod
    def _normalize_envelope(envelope: CanonicalArtifactEnvelope) -> ArtifactEnvelope:
        verdict = _OUTCOME_TO_VERDICT.get(envelope.outcome, VERDICT_INFRA_ERROR)
        blocked = envelope.outcome is ExecutionOutcome.FINDINGS
        hashes: tuple[str, ...] = ()
        if envelope.artifact_sha256:
            hashes = (envelope.artifact_sha256,)
        return ArtifactEnvelope(
            worker_verdict=verdict,
            completed_lenses=frozenset(envelope.completed_lenses),
            artifact_hashes=hashes,
            blocked=blocked,
        )
