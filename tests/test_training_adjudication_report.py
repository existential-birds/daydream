"""Report separates outcome-bearing coverage from silver/task-only coverage (AC 5)."""

import pytest

from daydream.training.adjudication.report import build_report


def test_report_separates_outcome_bearing_from_task_only_and_flags_as_of() -> None:
    items = [
        {
            "record_id": "r1",
            "disposition": "accepted",
            "profile": "pr_review",
            "stack": "python",
            "tier": "gold",
            "posterior_eligible": True,
            "observations": [],
            "evidence_after_as_of": False,
        },
        {
            "record_id": "r2",
            "disposition": "rejected",
            "profile": "local_branch",
            "stack": "rust",
            "tier": "task-only",
            "posterior_eligible": False,
            "observations": [],
            "evidence_after_as_of": False,
        },
        {
            "record_id": "r3",
            "disposition": "accepted",
            "profile": "pr_review",
            "stack": "python",
            "tier": "gold",
            "posterior_eligible": False,
            "observations": [],
            "evidence_after_as_of": True,
        },
    ]
    report = build_report(items)
    cov = report["outcome_coverage"]
    # Only eligible pr_review gold counts, excluding after-as-of and task-only records.
    assert cov["adjudicated"] == 1
    assert cov["total"] == 1  # r3 can never adjudicate, so it never feeds the denominator
    assert report["evidence_after_as_of"] == ["r3"]
    gate = report["admission_gate"]
    assert gate["outcome_bearing_total"] == 1
    # Non-posterior-eligible or after-as-of records cannot block the outcome denominator.
    assert gate["passes_80pct"] is True
    assert gate["class_balance_ok"] is False  # sole outcome-bearing record is accepted-only


def test_missing_required_field_raises_value_error() -> None:
    with pytest.raises(ValueError, match="record_id"):
        build_report([{"disposition": "accepted", "profile": "pr_review", "stack": "python"}])
    with pytest.raises(ValueError, match="disposition"):
        build_report([{"record_id": "a" * 64, "profile": "pr_review", "stack": "python"}])
