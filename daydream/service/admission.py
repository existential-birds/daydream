"""Global admission + retry budgets for the review service (Plan 008 Step 3).

The controller consults this before dispatch so that fleet/global, per-service,
per-backend, and per-model-provider limits can never be exceeded — the core
property that "executor count cannot multiply Pi calls without bound." A single
``ReviewExecutor.start`` consumes one slot in every applicable bucket; each
must have capacity or the job waits (no unbounded fan-out).

Retry policy is strict: only a *classified infrastructure failure* may be
retried with a new attempt, and only while a per-scope retry budget is
unexhausted. Finding verdicts, test failures, and cancellation are never
retried at the admission layer. On retry exhaustion the job is routed to an
operator instead of thrashing the same worker profile.

All state here is process-local admission bookkeeping (compare-and-set against
the execution's own progress lives in the storage port / store leaf). Missing
capability fails admission in ``AdmissionController.admit_executor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from daydream.service.models import REQUIRED_CAPABILITIES, JobSpec


@dataclass(frozen=True)
class Budgets:
    """Limits for each admission bucket.

    Each field is a ``dict`` of	max-concurrent-slot values. ``None`` (or an
    absent key) means "unbounded" for that particular scope. The special key
    ``"*"`` is the fleet/global default applied to every job subject to a
    bucket when no narrower key matches.

    Attributes:
        fleet: Global cap on concurrent executions across the whole service.
        per_service: Cap on concurrent executions per service namespace.
        per_backend: Cap on concurrent executions per agent backend (claude/codex/pi).
        per_model_provider: Cap on concurrent executions per model provider.
        retries: Max infra-failure retry attempts per (scope, key) before routing to an operator.
    """

    fleet: int | None = None
    per_service: dict[str, int] = field(default_factory=dict)
    per_backend: dict[str, int] = field(default_factory=dict)
    per_model_provider: dict[str, int] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)


@dataclass
class AdmissionController:
    """Tracks in-flight executions and admits jobs only when every bucket allows.

    This is deliberately single-process and intentionally small: it enforces
    the *admission* invariant (bounded Pi fan-out) while the durable CAS/lease
    bookkeeping lives in the storage port's store leaf. Multi-process admission
    would share a counter in the store; that is a store-leaf concern.
    """

    budgets: Budgets = field(default_factory=Budgets)

    # in-flight counts by bucket key.
    _fleet_in_flight: int = 0
    _service_in_flight: dict[str, int] = field(default_factory=dict)
    _backend_in_flight: dict[str, int] = field(default_factory=dict)
    _provider_in_flight: dict[str, int] = field(default_factory=dict)
    # infra retries consumed per (scope,key).
    _retries_used: dict[tuple[str, str], int] = field(default_factory=dict)

    # -- admission --------------------------------------------------------

    def can_start(self, spec: JobSpec) -> str | None:
        """Return a denial reason, or ``None`` when the job may be dispatched.

        ``None`` means every applicable bucket has spare capacity and the job
        may begin. A non-None return is the human-facing reason (e.g.
        ``"fleet 3/2 saturated"``) so the controller can back off, not fail.
        """
        if self.budgets.fleet is not None and self._fleet_in_flight >= self.budgets.fleet:
            return f"fleet saturated ({self._fleet_in_flight}/{self.budgets.fleet})"
        service_cap = self._bucket_cap(self.budgets.per_service, spec.service)
        if service_cap is not None and self._service_in_flight.get(spec.service, 0) >= service_cap:
            return f"service '{spec.service}' saturated ({self._service_in_flight.get(spec.service, 0)}/{service_cap})"
        backend_cap = self._bucket_cap(self.budgets.per_backend, spec.backend)
        if backend_cap is not None and self._backend_in_flight.get(spec.backend, 0) >= backend_cap:
            return f"backend '{spec.backend}' saturated ({self._backend_in_flight.get(spec.backend, 0)}/{backend_cap})"
        provider_cap = self._bucket_cap(self.budgets.per_model_provider, spec.provider)
        if provider_cap is not None and self._provider_in_flight.get(spec.provider, 0) >= provider_cap:
            used = self._provider_in_flight.get(spec.provider, 0)
            return f"provider '{spec.provider}' saturated ({used}/{provider_cap})"
        return None

    def start(self, spec: JobSpec) -> None:
        """Consume one slot in every applicable bucket for ``spec``.

        Callers must gate on ``can_start`` first; this method asserts capacity
        rather than silently over-committing.
        """
        reason = self.can_start(spec)
        if reason is not None:
            raise AdmissionDenied(reason)
        self._fleet_in_flight += 1
        self._service_in_flight[spec.service] = self._service_in_flight.get(spec.service, 0) + 1
        self._backend_in_flight[spec.backend] = self._backend_in_flight.get(spec.backend, 0) + 1
        self._provider_in_flight[spec.provider] = self._provider_in_flight.get(spec.provider, 0) + 1

    def release(self, spec: JobSpec) -> None:
        """Free one slot in every bucket after an execution terminates."""
        self._fleet_in_flight = max(0, self._fleet_in_flight - 1)
        self._service_in_flight[spec.service] = max(0, self._service_in_flight.get(spec.service, 0) - 1)
        self._backend_in_flight[spec.backend] = max(0, self._backend_in_flight.get(spec.backend, 0) - 1)
        self._provider_in_flight[spec.provider] = max(0, self._provider_in_flight.get(spec.provider, 0) - 1)

    # -- retry budget -----------------------------------------------------

    def infra_retry_available(self, spec: JobSpec) -> bool:
        """Return True when an infra failure may still be retried for ``spec``.

        Only infrastructure failures are retried here; findings and
        cancellation never route through this gate. The budget is enforced per
        scope key so a single runaway service cannot eat the fleet's retries.
        The lookup prefers a service-specific cap, falls back to a ``"*"``
        wildcard, and fails closed (no retries) when neither is configured.
        """
        max_retries = self.budgets.retries.get(spec.service, self.budgets.retries.get("*", 0))
        used = self._retries_used.get(("service", spec.service), 0)
        return used < max_retries

    def record_infra_retry(self, spec: JobSpec) -> None:
        """Consume one infra retry for ``spec``'s service scope."""
        key = ("service", spec.service)
        self._retries_used[key] = self._retries_used.get(key, 0) + 1

    # -- capability admission (executor) -----------------------------------

    def admit_executor(self, capabilities: frozenset[str]) -> str | None:
        """Return ``None`` if all required capabilities are present, else the missing capability.

        Missing capability fails admission (STOP condition from the public
        plan): an executor that cannot prove isolation, no-ambient-credentials,
        source read-only, brokered model access, strong cancellation,
        deterministic release, and restart reconciliation is never admitted.
        """
        missing = REQUIRED_CAPABILITIES - capabilities
        if not missing:
            return None
        return sorted(missing)[0]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _bucket_cap(bucket: dict[str, int], key: str) -> int | None:
        """Return the cap for ``key`` in ``bucket``, honoring a ``"*"`` wildcard."""
        if key in bucket:
            return bucket[key]
        return bucket.get("*")

    def in_flight(self) -> InFlightSnapshot:
        """Snapshot of in-flight counts for diagnostics/debugging."""
        return InFlightSnapshot(
            fleet=self._fleet_in_flight,
            service=dict(self._service_in_flight),
            backend=dict(self._backend_in_flight),
            provider=dict(self._provider_in_flight),
        )


@dataclass
class InFlightSnapshot:
    """A diagnostic snapshot of concurrent execution counts per bucket."""

    fleet: int
    service: dict[str, int]
    backend: dict[str, int]
    provider: dict[str, int]


class AdmissionDenied(Exception):
    """A job was not admitted (a bucket was already saturated)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
