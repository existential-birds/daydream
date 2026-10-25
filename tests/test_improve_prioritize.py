"""Tests for improve-flow finding prioritization."""

from typing import Any

import pytest

from daydream.improve import prioritize
from daydream.improve.prioritize import (
    aggregate_cross_service,
    leverage_score,
    order_by_leverage,
)


def _f(**overrides: Any) -> dict[str, Any]:
    finding = {
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
    if "fingerprint" not in overrides:
        finding["fingerprint"] = f"{finding['path']}::{finding['title']}"
    if "evidence" not in overrides:
        finding["evidence"] = [f"{finding['path']}:{finding['line']}"]
    return finding


def test_prioritization_orders_by_leverage_then_security_shape_and_stability(subtests: pytest.Subtests) -> None:
    with subtests.test(contract="leverage_orders_impact_over_effort_discounted"):
        hi = _f(impact="HIGH", effort="S", confidence="HIGH", risk="LOW")
        lo = _f(impact="HIGH", effort="L", confidence="LOW", risk="HIGH")
        assert leverage_score(hi) == pytest.approx(3.0)
        assert [f is hi for f in order_by_leverage([lo, hi])][0]
    with subtests.test(contract="high_confidence_security_floats_above_equal_leverage"):
        sec = _f(category="security", confidence="HIGH", impact="MED", effort="M")
        bug = _f(category="correctness", confidence="HIGH", impact="MED", effort="M")
        assert order_by_leverage([bug, sec])[0] is sec
    with subtests.test(contract="equal_leverage_same_category_preserves_input_order"):
        first = _f(title="First")
        second = _f(title="Second")
        assert order_by_leverage([first, second]) == [first, second]
    with subtests.test(contract="equal_leverage_prefers_subtractive_change_shape"):
        additive = _f(title="Add an adapter", change_shape="additive")
        consolidate = _f(title="Consolidate adapters", change_shape="consolidate")
        reuse = _f(title="Reuse the shared adapter", change_shape="reuse")
        delete = _f(title="Delete the redundant adapter", change_shape="delete")
        assert order_by_leverage([additive, consolidate, reuse, delete]) == [
            delete,
            reuse,
            consolidate,
            additive,
        ]


def test_package_compatibility_requires_a_shared_concern_and_consistent_reuse_target(subtests: pytest.Subtests) -> None:
    with subtests.test(contract="same_pattern_across_services_aggregates_to_one_finding"):
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
    with subtests.test(contract="distinct_findings_do_not_aggregate"):
        assert len(aggregate_cross_service([_f(title="SQL injection"), _f(title="Slow CI cache")])) == 2
    with subtests.test(contract="shared_partition_alone_does_not_merge"):
        a = _f(
            title="Unbounded query in the list endpoint",
            path="frontend/src/alpha/list.tsx",
            services=[],
            partition="frontend/src/alpha",
        )
        b = _f(
            title="Unbounded query in the detail endpoint",
            path="frontend/src/alpha/table.tsx",
            services=[],
            partition="frontend/src/alpha",
        )
        assert len(aggregate_cross_service([a, b])) == 2
    with subtests.test(contract="canonical_reuse_target_is_a_strong_cross_category_grouping_key"):
        local_parser = _f(
            fingerprint="fp-parser",
            title="Remove the local header parser",
            path="src/api/headers.py",
            category="tech-debt",
            impact="LOW",
            effort="S",
            risk="LOW",
            confidence="HIGH",
            maintenance_signals=["reuse_existing", "dead_code"],
            change_shape="delete",
            reuse_target="repo:src/http.py#parse_headers",
        )
        duplicated_tests = _f(
            fingerprint="fp-tests",
            title="Drive header cases through the shared parser",
            path="tests/test_headers.py",
            category="tests",
            impact="HIGH",
            effort="L",
            risk="HIGH",
            confidence="LOW",
            maintenance_signals=["reuse_existing", "duplicated_test_structure"],
            change_shape="additive",
            reuse_target="repo:src/http.py#parse_headers",
        )

        packages = aggregate_cross_service([duplicated_tests, local_parser])

        assert len(packages) == 1
        package = packages[0]
        assert package["categories"] == ["tech-debt", "tests"]
        assert package["maintenance_signals"] == [
            "dead_code",
            "duplicated_test_structure",
            "reuse_existing",
        ]
        assert package["reuse_target"] == "repo:src/http.py#parse_headers"
        assert package["change_shape"] == "additive"
        assert (package["impact"], package["effort"]) == ("HIGH", "L")
        assert (package["risk"], package["confidence"]) == ("HIGH", "LOW")
        assert package["locations"] == [
            "src/api/headers.py:1",
            "tests/test_headers.py:1",
        ]
    with subtests.test(contract="same_file_without_shared_concern_does_not_form_a_package"):
        findings = [
            _f(title="Delete a self-evident comment", category="tech-debt"),
            _f(title="Fix authentication bypass", category="security"),
        ]
        assert len(aggregate_cross_service(findings)) == 2
    with subtests.test(contract="same_file_and_generic_signal_still_require_semantic_similarity"):
        findings = [
            _f(
                fingerprint="fp-wrapper",
                title="Remove the redundant request wrapper",
                maintenance_signals=["overengineered_structure"],
            ),
            _f(
                fingerprint="fp-cache",
                title="Collapse duplicate cache bookkeeping",
                maintenance_signals=["overengineered_structure"],
            ),
        ]
        assert len(aggregate_cross_service(findings)) == 2
    with subtests.test(contract="conflicting_reuse_targets_never_form_one_package"):
        findings = [
            _f(
                fingerprint="fp-stdlib",
                title="Replace the local URL parser",
                reuse_target="stdlib:urllib.parse.urlparse",
            ),
            _f(
                fingerprint="fp-library",
                title="Replace the local URL parser",
                reuse_target="dep:yarl:URL",
            ),
        ]
        assert len(aggregate_cross_service(findings)) == 2
    with subtests.test(contract="package_compatibility_is_all_pairs_not_transitive"):
        common = {
            "category": "tests",
            "maintenance_signals": ["duplicated_test_structure"],
        }
        left = _f(
            **common,
            path="tests/test_catalog_setup.py",
            fingerprint="fp-left",
            title="Repeated catalog fixture setup",
        )
        bridge = _f(
            **common,
            path="tests/test_catalog_cases.py",
            fingerprint="fp-bridge",
            title="Repeated catalog fixture setup and assertions",
        )
        right = _f(
            **common,
            path="tests/test_catalog_assertions.py",
            fingerprint="fp-right",
            title="Repeated catalog assertions",
        )

        packages = aggregate_cross_service([left, bridge, right])

        assert len(packages) == 2
        assert sorted(len(package["members"]) for package in packages) == [1, 2]
        # The bridge must join before either endpoint, so the final join has
        # to agree with two existing members rather than a singleton group.
        bridge.update(impact="HIGH", effort="S")
        packages = aggregate_cross_service([left, bridge, right])
        assert sorted(len(package["members"]) for package in packages) == [1, 2]
        assert all(
            not {"fp-left", "fp-right"} <= set(package["member_fingerprints"])
            for package in packages
        )


def test_package_caps_keep_deterministic_distinct_identities(subtests: pytest.Subtests) -> None:
    with subtests.test(contract="work_package_caps_and_identity_are_deterministic"):
        findings = [
            _f(
                fingerprint=f"fp-{index}",
                title=f"Use shared parser at call site {index}",
                path=f"src/call_{index}.py",
                reuse_target="repo:src/parser.py#parse",
            )
            for index in range(7)
        ]
        forward = aggregate_cross_service(findings)
        reverse = aggregate_cross_service(list(reversed(findings)))
        assert [len(package["members"]) for package in forward] == [5, 2]
        assert [(package["package_fingerprint"], package["member_fingerprints"]) for package in forward] == [
            (package["package_fingerprint"], package["member_fingerprints"]) for package in reverse
        ]
    with subtests.test(contract="same_path_cap_splits_have_distinct_deterministic_package_ids"):
        findings = [
            _f(
                fingerprint=f"fp-{index:02d}",
                title=f"Use shared parser for request variant {index:02d}",
                path="src/requests.py",
                line=1,
                category="tech-debt",
                maintenance_signals=["reuse_existing"],
                reuse_target="repo:src/parser.py#parse",
            )
            for index in range(11)
        ]

        forward = aggregate_cross_service(findings)
        reverse = aggregate_cross_service(list(reversed(findings)))

        assert [len(package["members"]) for package in forward] == [5, 5, 1]
        assert len({package["package_fingerprint"] for package in forward}) == 3
        assert [(package["package_fingerprint"], package["member_fingerprints"]) for package in forward] == [
            (package["package_fingerprint"], package["member_fingerprints"]) for package in reverse
        ]


def test_member_aliases_survive_wording_and_expose_membership_drift() -> None:
    first = _f(
        fingerprint="volatile-first",
        title="Repeated catalog setup should use the fixture",
        path="tests/test_catalog.py",
        line=12,
        category="tests",
        maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )
    reworded = _f(
        fingerprint="volatile-reworded",
        title="Consolidate the catalog test setup",
        path="tests/test_catalog.py",
        line=12,
        category="tests",
        maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )
    added = _f(
        fingerprint="volatile-added",
        title="Consolidate repeated catalog assertions",
        path="tests/test_catalog.py",
        line=30,
        category="tests",
        maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )

    before = aggregate_cross_service([first])[0]
    after = aggregate_cross_service([reworded])[0]
    expanded = aggregate_cross_service([reworded, added])[0]

    assert before["member_aliases"] == after["member_aliases"]
    assert before["package_fingerprint"] == after["package_fingerprint"]
    assert set(after["member_aliases"]) < set(expanded["member_aliases"])
    assert expanded["package_fingerprint"] != after["package_fingerprint"]


def test_prioritize_unknown_severity_maps_explicitly() -> None:
    """P-BOUNDARY (R6.2): unknown severity does not silently pass through as
    ``"CRITICAL"``/raw — the axis mapper returns ``None`` and the caller omits
    the axis instead of promoting to a conservative fallback."""
    assert prioritize._map_axis_severity(None) is None
    assert prioritize._map_axis_severity("high") == "HIGH"
    assert prioritize._map_axis_severity("weird") is None
    # The axis vocabulary is 3-level; the off-canonical "CRITICAL" must not
    # pass through as raw, exactly as the docstring asserts.
    assert prioritize._map_axis_severity("CRITICAL") is None
