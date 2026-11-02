"""Posted finding bodies disclose host location distrust instead of demoting silently."""
from __future__ import annotations

from daydream.pr_review import parsed_issues_from_items


def test_unverified_location_demotion_is_visible_on_the_posted_body() -> None:
    """An uncorroborated citation keeps its demotion evidence on the posted body.

    Runner placement tests cite in-hunk lines, so they never carry the distrust
    signal that the reviewer must be able to see on the PR.
    """
    base = {"file": "main.py", "line": 9, "description": "Module lacks a rollback barrier",
            "rationale": "No barrier guards this module", "severity": "low", "confidence": "HIGH",
            "location_distrust": True}
    demoted, unknown = parsed_issues_from_items([{**base, "severity_before_demotion": "high"}, base])
    assert demoted.body.split("\n\n") == ["**Severity:** low", "**Confidence:** HIGH",
                                          "**Location:** unverified citation (severity demoted from high)",
                                          "No barrier guards this module"]
    assert "**Location:** unverified citation (severity demoted)" in unknown.body
    assert demoted.location_distrust and demoted.severity_before_demotion == "high"
