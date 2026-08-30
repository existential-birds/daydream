"""Report separates outcome-bearing coverage from silver/task-only coverage (AC 5)."""
from typing import Any

from daydream.training.adjudication.report import build_report


def _item(
    rid: str,
    disposition: str,
    profile: str = "pr_review",
    stack: str = "python",
    raters: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    return {"record_id": rid, "disposition": disposition, "profile": profile, "stack": stack,
            "tier": "gold" if disposition in {"accepted", "rejected"} else "task-only",
            "observations": [{"role": r, "disposition": d}
                             for (r, d) in raters]}


def test_silver_task_only_never_counts_toward_outcome_coverage() -> None:
    items = [
        _item("a" * 64, "accepted", raters=(("rater", "accepted"),)),
        _item("b" * 64, "task-only", raters=()),            # silver/task-only stratum
        _item("d" * 64, "task-only", raters=()),            # never in the 80% denominator
    ]
    report = build_report(items)
    assert report["outcome_coverage"] == {"adjudicated": 1, "total": 1}  # task-only excluded
    assert report["silver_task_only_count"] == 2
    assert report["class_balance"] == {"accepted": 1, "rejected": 0}
    assert report["unresolved"] == 0


def test_inter_rater_agreement_counts_only_multi_rater_items() -> None:
    items = [
        _item("a" * 64, "accepted", raters=(("rater", "accepted"), ("rater2", "accepted"))),
        _item("b" * 64, "task-only", raters=(("rater", "accepted"), ("rater2", "rejected"))),
        _item("e" * 64, "rejected", raters=(("rater", "rejected"),)),
    ]
    report = build_report(items)
    assert report["inter_rater"] == {"items": 1, "agreeing": 0}  # the disputed pair counted


def test_report_is_deterministic_and_stratified() -> None:
    items = [_item(f"{i:064x}", "ambiguous", stack=s, profile=p)
             for i, (s, p) in enumerate([("python", "pr_review"), ("rust", "pr_review"),
                                         ("python", "task")])]
    r1 = build_report(list(reversed(items)))
    r2 = build_report(items)
    assert r1 == r2  # determinism regardless of input order
    assert r1["strata"][("python", "pr_review")] == 1
    assert r1["strata"][("python", "task")] == 1


def test_missing_required_field_raises_value_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="record_id"):
        build_report([{"disposition": "accepted", "profile": "pr_review", "stack": "python"}])
    with pytest.raises(ValueError, match="disposition"):
        build_report([{"record_id": "a" * 64, "profile": "pr_review", "stack": "python"}])
