"""Tests for improve-flow finding prioritization."""

from typing import Any

import pytest

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


def test_leverage_orders_impact_over_effort_discounted() -> None:
    hi = _f(impact="HIGH", effort="S", confidence="HIGH", risk="LOW")
    lo = _f(impact="HIGH", effort="L", confidence="LOW", risk="HIGH")
    assert leverage_score(hi) == pytest.approx(3.0)
    assert [f is hi for f in order_by_leverage([lo, hi])][0]


def test_high_confidence_security_floats_above_equal_leverage() -> None:
    sec = _f(category="security", confidence="HIGH", impact="MED", effort="M")
    bug = _f(category="correctness", confidence="HIGH", impact="MED", effort="M")
    assert order_by_leverage([bug, sec])[0] is sec


def test_equal_leverage_same_category_preserves_input_order() -> None:
    first = _f(title="First")
    second = _f(title="Second")
    assert order_by_leverage([first, second]) == [first, second]


def test_equal_leverage_prefers_subtractive_change_shape() -> None:
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


def test_cross_partition_findings_merge_like_cross_service() -> None:
    a = _f(
        title="Unbounded query in the list endpoint",
        path="frontend/src/alpha/view.tsx",
        services=[],
        partition="frontend/src/alpha",
    )
    b = _f(
        title="Unbounded query in the list endpoint",
        path="frontend/src/beta/view.tsx",
        services=[],
        partition="frontend/src/beta",
    )
    merged = aggregate_cross_service([a, b])
    assert len(merged) == 1
    assert set(merged[0]["partitions"]) == {
        "frontend/src/alpha",
        "frontend/src/beta",
    }
    assert "frontend/src/alpha" in merged[0]["body"]
    assert "frontend/src/beta" in merged[0]["body"]


def test_shared_partition_alone_does_not_merge() -> None:
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


def test_service_and_partition_stamps_compose() -> None:
    a = _f(
        title="Unbounded query in the list endpoint",
        path="apps/billing/api.py",
        services=["billing"],
        partition="billing",
    )
    b = _f(
        title="Unbounded query in the list endpoint",
        path="apps/catalog/api.py",
        services=["catalog"],
        partition="catalog",
    )
    merged = aggregate_cross_service([a, b])
    assert len(merged) == 1
    assert set(merged[0]["services"]) == {"billing", "catalog"}
    assert set(merged[0]["partitions"]) == {"billing", "catalog"}


def test_reworded_duplicates_at_the_same_path_become_one_package() -> None:
    first = _f(
        fingerprint="fp-first",
        title="Repeated catalog fixture setup obscures behavior",
        path="tests/test_catalog.py",
        category="tests",
    )
    second = _f(
        fingerprint="fp-second",
        title="Catalog fixture setup is repeated in every test",
        path="tests/test_catalog.py",
        category="tests",
    )

    packages = aggregate_cross_service([first, second])

    assert len(packages) == 1
    assert packages[0]["member_fingerprints"] == ["fp-first", "fp-second"]
    assert len(packages[0]["members"]) == 2
    assert packages[0]["fingerprint"] == packages[0]["package_fingerprint"]


def test_rewording_does_not_change_semantic_singleton_package_identity() -> None:
    before = _f(
        fingerprint="volatile-before",
        title="Repeated catalog test setup should use the shared fixture",
        path="tests/test_catalog.py",
        category="tests",
        maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )
    after = _f(
        fingerprint="volatile-after",
        title="Consolidate duplicated setup in catalog tests",
        path="tests/test_catalog.py",
        category="tests",
        maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )

    before_package = aggregate_cross_service([before])[0]
    after_package = aggregate_cross_service([after])[0]

    assert before_package["package_fingerprint"] == after_package["package_fingerprint"]
    assert before_package["member_fingerprints"] == ["volatile-before"]
    assert after_package["member_fingerprints"] == ["volatile-after"]


def test_canonical_reuse_target_is_a_strong_cross_category_grouping_key() -> None:
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


def test_same_file_without_shared_concern_does_not_form_a_package() -> None:
    findings = [
        _f(title="Delete a self-evident comment", category="tech-debt"),
        _f(title="Fix authentication bypass", category="security"),
    ]

    assert len(aggregate_cross_service(findings)) == 2


def test_same_file_and_generic_signal_still_require_semantic_similarity() -> None:
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


def test_conflicting_reuse_targets_never_form_one_package() -> None:
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


def test_package_compatibility_is_all_pairs_not_transitive() -> None:
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


def test_work_package_caps_and_identity_are_deterministic() -> None:
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


def test_same_path_cap_splits_have_distinct_deterministic_package_ids() -> None:
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
