"""Admission + retry budget tests (Plan 008 Step 3, leaf-B).

These pin the invariant that fleet/global, per-service, per-backend, and
per-model-provider budgets bound concurrent execution, and that only classified
infrastructure failure is retried (bounded), never findings or cancellation.
"""

from __future__ import annotations

import pytest

from daydream.service.admission import AdmissionController, AdmissionDenied, Budgets
from daydream.service.models import REQUIRED_CAPABILITIES, CandidateTarget, JobSpec


def _job(service: str = "svc", backend: str = "pi", provider: str = "nous") -> JobSpec:
    return JobSpec(
        job_id="j1",
        candidate=CandidateTarget(
            target_kind="pr_head",
            repo="owner/repo",
            candidate_sha="c" * 40,
            tree_digest="t" * 40,
            base_sha="b" * 40,
            invalidation_id=1,
        ),
        service=service,
        backend=backend,
        provider=provider,
        model="deepseek-v4-flash-0731",
    )


def test_fleet_budget_saturates_and_restores() -> None:
    gate = AdmissionController(Budgets(fleet=1))
    assert gate.can_start(_job()) is None
    gate.start(_job())
    denied = gate.can_start(_job())
    assert denied is not None and "fleet" in denied
    gate.release(_job())
    assert gate.can_start(_job()) is None


def test_per_service_budget_is_independent_across_services() -> None:
    gate = AdmissionController(Budgets(per_service={"svc-a": 1}))
    gate.start(_job(service="svc-a"))
    assert gate.can_start(_job(service="svc-a")) is not None
    assert gate.can_start(_job(service="svc-b")) is None


def test_wildcard_bucket_cap_is_a_per_key_default() -> None:
    gate = AdmissionController(Budgets(per_backend={"*": 1}))
    gate.start(_job(backend="pi"))
    # Same key saturates; a distinct key keeps its own wildcard-derived slot.
    assert gate.can_start(_job(backend="pi")) is not None
    assert gate.can_start(_job(backend="claude")) is None


def test_per_backend_fanout_is_bounded() -> None:
    """Two concurrent Pi jobs to the same backend must not both be admitted."""
    gate = AdmissionController(Budgets(per_backend={"pi": 1}))
    gate.start(_job(backend="pi"))
    assert gate.can_start(_job(backend="pi")) is not None
    assert gate.can_start(_job(backend="codex")) is None


def test_per_model_provider_is_bounded() -> None:
    gate = AdmissionController(Budgets(per_model_provider={"nous": 1}))
    gate.start(_job(provider="nous"))
    assert gate.can_start(_job(provider="nous")) is not None
    assert gate.can_start(_job(provider="anthropic")) is None


def test_all_buckets_must_grant_for_admission() -> None:
    """Admission is the AND across fleet, service, backend, and provider."""
    gate = AdmissionController(
        Budgets(
            fleet=10,
            per_service={"svc": 10},
            per_backend={"pi": 2},
            per_model_provider={"nous": 2},
        )
    )
    spec = _job()
    gate.start(spec)
    gate.start(_job(service="other"))  # second concurrent under pi/nous caps
    # pi and nous are now both at 2; a third must be denied even though service is free.
    denied = gate.can_start(_job())
    assert denied is not None
    assert "pi" in denied or "provider" in denied


def test_start_asserts_capacity() -> None:
    gate = AdmissionController(Budgets(fleet=1))
    gate.start(_job())
    with pytest.raises(AdmissionDenied):
        gate.start(_job())


def test_infra_retry_budget_is_per_service_and_bounded() -> None:
    gate = AdmissionController(Budgets(retries={"svc": 2}))
    assert gate.infra_retry_available(_job(service="svc")) is True
    gate.record_infra_retry(_job(service="svc"))
    gate.record_infra_retry(_job(service="svc"))
    assert gate.infra_retry_available(_job(service="svc")) is False
    # A service with no configured budget fails closed (no retries).
    assert gate.infra_retry_available(_job(service="other")) is False


def test_global_retry_budget_default() -> None:
    gate = AdmissionController(Budgets(retries={"*": 1}))
    assert gate.infra_retry_available(_job()) is True
    gate.record_infra_retry(_job())
    assert gate.infra_retry_available(_job()) is False


def test_zero_retry_budget_never_retries() -> None:
    gate = AdmissionController(Budgets(retries={"svc": 0}))
    assert gate.infra_retry_available(_job(service="svc")) is False


def test_executor_capability_gate() -> None:
    gate = AdmissionController()
    # Missing one required capability fails admission with the missing name.
    weakened = REQUIRED_CAPABILITIES - {"strong_cancel"}
    missing = gate.admit_executor(frozenset(weakened))
    assert missing == "strong_cancel"
    # Full capability set is admitted.
    assert gate.admit_executor(REQUIRED_CAPABILITIES) is None


def test_missing_execution_capability_fails_closed() -> None:
    """An executor missing restart reconciliation is never admitted."""
    gate = AdmissionController()
    assert gate.admit_executor(frozenset({"exclusive_workspace"})) is not None


def test_in_flight_snapshot_shape() -> None:
    budgets = Budgets(fleet=3, per_service={"a": 3}, per_backend={"pi": 3}, per_model_provider={"n": 3})
    gate = AdmissionController(budgets)
    gate.start(_job(service="a", backend="pi", provider="n"))
    snap = gate.in_flight()
    assert snap.fleet == 1
    assert snap.service == {"a": 1}
    assert snap.backend == {"pi": 1}
    assert snap.provider == {"n": 1}
