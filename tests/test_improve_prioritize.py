"""Tests for improve-flow finding prioritization."""

from typing import Any

from daydream.improve.prioritize import (
    aggregate_cross_service,
)


def _f(**overrides: Any) -> dict[str, Any]:
    finding = {
        "path": "src/example.py", "line": 1, "placement": "inline", "title": "Example finding", "body": "Example body",
        "severity": "MED", "confidence": "HIGH", "is_cross_stack": False, "impact": "MED", "effort": "M", "risk": "LOW",
        "leverage": None, "category": "correctness", "services": [], "provenance": "inherited",
    }
    finding.update(overrides)
    if "fingerprint" not in overrides:
        finding["fingerprint"] = f"{finding['path']}::{finding['title']}"
    if "evidence" not in overrides:
        finding["evidence"] = [f"{finding['path']}:{finding['line']}"]
    return finding

def test_canonical_reuse_target_is_a_strong_cross_category_grouping_key() -> None:
    findings = [
        _f(fingerprint="billing", title="Unbounded query in list endpoint", path="apps/billing/api.py",
            services=["billing"], severity="high"),
        _f(fingerprint="catalog", title="Unbounded query in the list endpoint", path="apps/catalog/api.py",
            services=["catalog"]),
        _f(fingerprint="parser", title="Remove the local header parser", path="src/api/headers.py",
            category="tech-debt", impact="LOW", effort="S", maintenance_signals=["reuse_existing", "dead_code"],
            change_shape="delete", reuse_target="repo:src/http.py#parse_headers"),
        _f(fingerprint="parser-tests", title="Drive header cases through the shared parser",
            path="tests/test_headers.py", category="tests", impact="HIGH", effort="L", risk="HIGH", confidence="LOW",
            maintenance_signals=["reuse_existing", "duplicated_test_structure"], change_shape="additive",
            reuse_target="repo:src/http.py#parse_headers"),
        _f(fingerprint="comment", title="Delete a self-evident comment", category="tech-debt"),
        _f(fingerprint="auth", title="Fix authentication bypass", category="security"),
        _f(fingerprint="wrapper", title="Remove the redundant request wrapper",
            maintenance_signals=["overengineered_structure"]),
        _f(fingerprint="cache", title="Collapse duplicate cache bookkeeping",
            maintenance_signals=["overengineered_structure"]),
        _f(fingerprint="stdlib", title="Replace the local URL parser", reuse_target="stdlib:urllib.parse.urlparse"),
        _f(fingerprint="library", title="Replace the local URL parser", reuse_target="dep:yarl:URL"),
        _f(fingerprint="list", title="Route subscriptions leak in the list component",
            path="frontend/src/alpha/list.tsx",
            partition="frontend/src/alpha"),
        _f(fingerprint="detail", title="Route subscriptions leak in the detail component",
            path="frontend/src/alpha/table.tsx",
            partition="frontend/src/alpha"),
        _f(fingerprint="left", title="Repeated catalog fixture setup", path="tests/test_catalog_setup.py",
            category="tests", maintenance_signals=["duplicated_test_structure"]),
        _f(fingerprint="bridge", title="Repeated catalog fixture setup and assertions",
            path="tests/test_catalog_cases.py", category="tests", maintenance_signals=["duplicated_test_structure"]),
        _f(fingerprint="right", title="Repeated catalog assertions", path="tests/test_catalog_assertions.py",
            category="tests", maintenance_signals=["duplicated_test_structure"]),
    ]
    packages = {tuple(package["member_fingerprints"]): package for package in aggregate_cross_service(findings)}
    assert set(packages) == {
        ("billing", "catalog"), ("parser", "parser-tests"), ("comment",), ("auth",), ("wrapper",), ("cache",),
        ("stdlib",), ("library",), ("list",), ("detail",), ("bridge", "right"), ("left",),
    }
    shared = packages[("billing", "catalog")]
    assert shared["services"] == ["billing", "catalog"]
    assert shared["evidence"] == ["apps/billing/api.py:1", "apps/catalog/api.py:1"]
    assert shared["severity"] == "HIGH"
    package = packages[("parser", "parser-tests")]
    assert package["categories"] == ["tech-debt", "tests"]
    assert package["maintenance_signals"] == ["dead_code", "duplicated_test_structure", "reuse_existing"]
    assert package["reuse_target"] == "repo:src/http.py#parse_headers"
    assert package["change_shape"] == "additive"
    assert (package["impact"], package["effort"], package["risk"], package["confidence"]) == (
        "HIGH", "L", "HIGH", "LOW",
    )
    assert package["locations"] == ["src/api/headers.py:1", "tests/test_headers.py:1"]


def test_member_aliases_survive_wording_and_expose_membership_drift() -> None:
    first = _f(fingerprint="volatile-first", title="Repeated catalog setup should use the fixture",
        path="tests/test_catalog.py", line=12, category="tests", maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )
    reworded = _f(
        fingerprint="volatile-reworded", title="Consolidate the catalog test setup", path="tests/test_catalog.py",
        line=12, category="tests", maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )
    added = _f(
        fingerprint="volatile-added", title="Consolidate repeated catalog assertions", path="tests/test_catalog.py",
        line=30, category="tests", maintenance_signals=["duplicated_test_structure"],
        reuse_target="repo:tests/conftest.py#catalog_fixture",
    )

    before = aggregate_cross_service([first])[0]
    after = aggregate_cross_service([reworded])[0]
    expanded, bare = aggregate_cross_service([reworded, added, {"title": "x"}])

    assert before["member_aliases"] == after["member_aliases"]
    assert before["package_fingerprint"] == after["package_fingerprint"]
    assert set(after["member_aliases"]) < set(expanded["member_aliases"])
    assert expanded["package_fingerprint"] != after["package_fingerprint"]
    assert expanded["member_aliases"][0] == "member:33dc485da5dce8f6160b9c0e2ebb1dee85078f795929502a6bf7b16babe3dd6c"
    assert expanded["member_aliases"][1] == "member:ec9652f854fb905fdc5744fb08bcbd18681e45f153b1b1cdda54fba0cb54e602"
    assert bare["member_aliases"] == ["member:aee8bc487098438d8abf22bd68eacb59752131851a25ca9430116254d459890e"]
