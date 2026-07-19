"""Tests for improve-flow finding prioritization."""

from typing import Any

import pytest

from daydream.improve.prioritize import (
    aggregate_cross_service,
    leverage_score,
    order_by_leverage,
    partition_direction,
)


def _f(**overrides: Any) -> dict[str, Any]:
    finding = {
        "fingerprint": "0" * 64,
        "path": "src/example.py",
        "line": 1,
        "placement": "inline",
        "title": "Example finding",
        "body": "Example body",
        "severity": "MED",
        "confidence": "HIGH",
        "is_cross_stack": False,
        "impact": "MED",
        "effort": "M",
        "risk": "LOW",
        "leverage": None,
        "category": "correctness",
        "services": [],
        "provenance": "inherited",
    }
    finding.update(overrides)
    if "evidence" not in overrides:
        finding["evidence"] = [f"{finding['path']}:{finding['line']}"]
    return finding


def test_leverage_orders_impact_over_effort_discounted() -> None:
    hi = _f(impact="HIGH", effort="S", confidence="HIGH", risk="LOW")
    lo = _f(impact="HIGH", effort="L", confidence="LOW", risk="HIGH")
    assert leverage_score(hi) == pytest.approx(3.0)
    assert [f is hi for f in order_by_leverage([lo, hi])][0]


def test_high_confidence_security_floats_above_equal_leverage() -> None:
    sec = _f(category="security", confidence="HIGH", impact="MED", effort="M")
    bug = _f(category="correctness", confidence="HIGH", impact="MED", effort="M")
    assert order_by_leverage([bug, sec])[0] is sec


def test_direction_findings_partition_separately() -> None:
    defects, direction = partition_direction(
        [_f(category="direction"), _f(category="correctness")]
    )
    assert [f["category"] for f in defects] == ["correctness"]
    assert [f["category"] for f in direction] == ["direction"]


def test_equal_leverage_same_category_preserves_input_order() -> None:
    first = _f(title="First")
    second = _f(title="Second")
    assert order_by_leverage([first, second]) == [first, second]


def test_same_pattern_across_services_aggregates_to_one_finding() -> None:
    a = _f(
        title="Unbounded query in list endpoint",
        path="apps/billing/api.py",
        services=["billing"],
    )
    b = _f(
        title="Unbounded query in the list endpoint",
        path="apps/catalog/api.py",
        services=["catalog"],
    )
    merged = aggregate_cross_service([a, b])
    assert len(merged) == 1
    assert set(merged[0]["services"]) == {"billing", "catalog"}
    assert len(merged[0]["evidence"]) >= 2


def test_distinct_findings_do_not_aggregate() -> None:
    assert (
        len(
            aggregate_cross_service(
                [_f(title="SQL injection"), _f(title="Slow CI cache")]
            )
        )
        == 2
    )
