"""Truthful coverage derivation and strict terminal-result invariants."""
import copy
from collections.abc import Callable
from typing import Any

import pytest

from daydream.review_result import (
    ReasonCode,
    ReviewCoverage,
    reason_for_exception,
    validate_terminal_result,
)
from tests.harness.review_result import review_coverage


@pytest.mark.parametrize("case,expected", [
    ("complete", "complete"), ("failed", "failed"), ("partial", "incomplete"),
    ("sibling", "incomplete"), ("projection", "failed"), ("dynamic", "incomplete"),
    ("cancelled", "incomplete"), ("alternatives", "incomplete"), ("partial-alternatives", "incomplete"),
    ("no-diff", "complete"), ("unknown", "failed"), ("intent-only", "failed"), ("missing-projection", "failed"),
])
def test_terminal_state_requires_positive_usable_coverage(case: str, expected: str) -> None:
    phases = {"no-diff": ("no_diff",), "unknown": (), "intent-only": ("intent",)}
    c = review_coverage(scope_ids=() if case in phases else ("python", "structure"),
                        phases=phases.get(case, ("merge",)))
    failed_pipeline = case in {"sibling", "projection", "partial-alternatives", "missing-projection"}
    pipeline = "failed" if failed_pipeline else "completed"
    if case in {"complete", "projection", "dynamic"}:
        for scope in c.scopes:
            c.record_scope(scope, "complete")
        c.record_phase("merge", "complete", noop=True)
    elif case in {"failed", "partial"}:
        c.record_scope("python", "incomplete" if case == "partial" else "failed",
                       reasons=["host_wall_budget_exhaustion" if case == "partial" else "backend_failure"],
                       partial_evidence=case == "partial")
        c.record_phase("merge", "complete", noop=True)
    elif case == "sibling":
        c.record_scope("python", "complete")
        c.record_scope("structure", "failed", reasons=["backend_failure"])
        c.record_phase("merge", "failed", reasons=["synthesis_failure"])
    elif case == "cancelled":
        c.record_scope("python", "complete")
        pipeline = "cancelled"
    elif case == "alternatives":
        for scope in c.scopes:
            c.record_scope(scope, "failed", reasons=["backend_failure"])
        c.record_phase("merge", "complete", noop=True)
    if case == "dynamic":
        c.require_phase("adjudication")
    elif case in {"alternatives", "partial-alternatives"}:
        c.require_phase("alternatives")
        c.record_phase("alternatives", "incomplete" if case == "partial-alternatives" else "complete",
                       reasons=["host_wall_budget_exhaustion"] if case == "partial-alternatives" else [],
                       usable_evidence=True)
    elif case == "no-diff":
        c.record_phase("no_diff", "complete", noop=True)
    elif case == "intent-only":
        c.record_phase("intent", "complete")
    elif case == "missing-projection":
        c.require_phase("findings")
        c.record_phase("findings", "failed", reasons=["missing_artifact"])
    result = c.finalize(pipeline, projection_valid=case not in {"projection", "missing-projection"})
    assert result["analysis_state"] == expected
    assert [scope["scope_id"] for scope in result["stack_outcomes"]] == sorted(c.scopes)
    if case == "complete":
        assert result["reason_codes"] == []
    elif case in {"failed", "partial"}:
        assert result["uncovered_stacks"] == ["python", "structure"]
    elif case == "cancelled":
        assert "interruption" in result["reason_codes"]
    elif case in {"unknown", "intent-only"}:
        assert result["reason_codes"] == ["coverage_unknown"]
    elif case == "missing-projection":
        assert "missing_artifact" in result["reason_codes"] and "malformed_artifact" not in result["reason_codes"]


@pytest.mark.parametrize("mutate", [
    lambda r: r["stack_outcomes"].pop(),
    lambda r: r["planned_scopes"].append(copy.deepcopy(r["planned_scopes"][0])),
    lambda r: r.update(analysis_state="complete"),
    lambda r: r["reason_codes"].append("invented_reason"),
    lambda r: r.update(completed_stacks=["structure"]),
    lambda r: r["stack_outcomes"][0].update(partial_evidence=True),
])
def test_terminal_rejects_inconsistent_or_unknown_evidence(mutate: Callable[[dict[str, Any]], None]) -> None:
    c = review_coverage()
    c.record_scope("python", "complete")
    result = c.finalize("failed")
    mutate(result)
    with pytest.raises(ValueError):
        validate_terminal_result(result)


@pytest.mark.parametrize("bad", [None, [], {"schema_version": 1}, {"schema_version": True}])
def test_corrupt_coverage_is_a_controlled_validation_failure(bad: Any) -> None:
    with pytest.raises(ValueError):
        ReviewCoverage.from_dict(bad)


def test_checked_roundtrip_and_immutable_terminal() -> None:
    c = review_coverage(files=("a.py", "b.py"))
    c.record_scope("python", "incomplete", reasons=["host_tool_budget_exhaustion"], partial_evidence=True,
                   diagnostic="budget stopped: OPENROUTER_API_KEY=secret")
    assert "secret" not in c.diagnostics["scopes"]["python"]
    restored = ReviewCoverage.from_dict(c.to_dict())
    assert restored.to_dict() == c.to_dict()
    corrupted = c.to_dict()
    for name in ("planned_scopes", "stack_outcomes"):
        corrupted[name][0]["files"].reverse()
    with pytest.raises(ValueError):
        ReviewCoverage.from_dict(corrupted)
    assert not restored.is_finalized
    result = restored.finalize("completed")
    assert result["analysis_state"] == "incomplete" and restored.is_finalized
    persisted = restored.to_dict()
    result["analysis_state"] = "complete"
    restored.scopes["python"]["status"] = "uncovered"
    restored.run_id = "late-other-run"
    restored.diagnostics["scopes"]["python"] = "late diagnostic"
    assert restored.is_finalized and restored.to_dict() == persisted
    with pytest.raises(AttributeError):
        setattr(restored, "is_finalized", False)
    with pytest.raises(ValueError, match="frozen"):
        restored.record_scope("python", "failed", reasons=["backend_failure"])


def test_identity_and_invalid_recording_are_rejected() -> None:
    c = review_coverage()
    with pytest.raises(ValueError, match="head"):
        validate_terminal_result(c.finalize("completed"), expected_head_sha="different-head")
    with pytest.raises(ValueError, match="duplicate"):
        review_coverage(scope_ids=("same", "same"))
    with pytest.raises(ValueError, match="unknown"):
        review_coverage().record_scope("extra", "complete")
    with pytest.raises(ValueError, match="usable"):
        review_coverage().record_phase("merge", "failed", reasons=["synthesis_failure"], usable_evidence=True)


@pytest.mark.parametrize("category,declared", [(None, None), ("AUTH_CONFIG", None),
    (None, "provider-specific-error"), (None, 429), (None, {"provider": "error"})])
def test_exception_classification_uses_host_attributes(category: str | None, declared: Any) -> None:
    class ProviderError(RuntimeError):
        reason_code = declared
    error = ProviderError("invalid api key")
    if category:
        error.category = category  # type: ignore[attr-defined]
    expected = ReasonCode.AUTHENTICATION_FAILURE if category else ReasonCode.BACKEND_FAILURE
    assert reason_for_exception(error) == expected
