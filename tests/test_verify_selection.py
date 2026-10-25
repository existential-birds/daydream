from typing import Any, cast

import pytest

from daydream.deep.adjudication_provenance import RecordProvenance
from daydream.deep.verify_selection import (
    SELECTION_RULE_VERSION,
    SelectionConfig,
    SelectionDecision,
    UnknownRiskCategoryError,
    changed_text_at,
    item_content_digest,
    plan_reuse,
    resolve_selection_config,
    select_items,
)

_DIFF = """diff --git a/auth.py b/auth.py
--- a/auth.py
+++ b/auth.py
@@ -1,3 +1,4 @@
 def login(user):
+    if not user.token: raise PermissionError
     return user
"""

def test_cited_changed_lines_are_the_added_text_of_the_covering_hunk() -> None:
    assert "permissionerror" in changed_text_at(_DIFF, "auth.py", 2).lower()
    assert changed_text_at(_DIFF, "auth.py", 3).strip() == ""  # unchanged line
    assert changed_text_at(_DIFF, "other.py", 2) == ""  # file not in the diff
    assert changed_text_at("", "auth.py", 1) == ""  # no diff at all


def _item(item_uid: str, **over: object) -> dict[str, object]:
    base = {"item_uid": item_uid, "id": 1, "lens": "per-stack", "file": "a.py", "line": 1,
        "severity": "low", "confidence": "HIGH", "description": "routine cleanup",
        "rationale": "pure rename", "evidence": "a.py:1 renamed helper",
    }
    return {**base, **over}


def _provenance(uid: str, *, verdict_bound: bool = True, revised_fields: tuple[str, ...] = ()) -> RecordProvenance:
    return RecordProvenance(uid, ("arbiter",), verdict_bound, True, revised_fields)

def test_absent_global_provenance_cannot_certify_a_routine_item() -> None:
    decisions = select_items([_item("item:1")], provenance={}, diff_text=_DIFF,
        config=SelectionConfig(verify_all=False, extra_categories=()))
    assert [(d.item_uid, d.selected, d.reason_code) for d in decisions] == [
        ("item:1", True, "unadjudicated:no_provenance")]

def test_only_a_strong_adjudicated_routine_item_skips() -> None:
    """One selection batch retains each sibling's identity, distinct reason and input order."""
    items = [
        _item("item:1"), _item("item:2", lens="cross-stack"),
        _item("item:3", description="the handler must authenticate the caller's token"),
        _item("item:4", description="this path can deadlock while holding the mutex"),
        _item("item:5", description="the change runs `alter table users`"),
        _item("item:6", description="the grpc endpoint gains a field"),
        _item("item:7", description="see migrations/0042_add_flag.py"),
        _item("item:8", confidence="MEDIUM"), _item("item:9", evidence="n/a"),
        _item("item:10", location_distrust=True), _item("item:11"), _item("item:12"),
    ]
    for index, item in enumerate(items, 1):
        item["id"] = index
    provenance = {str(item["item_uid"]): _provenance(str(item["item_uid"])) for item in items}
    provenance["item:11"] = _provenance("item:11", verdict_bound=False)
    provenance["item:12"] = _provenance("item:12", revised_fields=("severity",))
    config = resolve_selection_config(verify_all=False, extra_categories=["security", "migration", "security"])
    decisions = select_items(items, provenance=provenance, diff_text=_DIFF, config=config)
    assert [(d.item_uid, d.item_id, d.selected, d.reason_code) for d in decisions] == [
        ("item:1", 1, False, "strongly_evidenced_adjudicated_routine"), ("item:2", 2, True, "cross_stack"),
        ("item:3", 3, True, "risk_category:security"), ("item:4", 4, True, "risk_category:concurrency"),
        ("item:5", 5, True, "risk_category:persistence"),
        ("item:6", 6, True, "risk_category:public-interface"),
        ("item:7", 7, True, "risk_category:migration"), ("item:8", 8, True, "weak_evidence:confidence"),
        ("item:9", 9, True, "weak_evidence:placeholder_evidence"),
        ("item:10", 10, True, "weak_evidence:located_beyond_tolerance"),
        ("item:11", 11, True, "unadjudicated:verdict_unbound"), ("item:12", 12, True, "materially_revised"),
    ]

def test_absent_diff_selects_even_a_strong_routine_item() -> None:
    # Uniform failure direction: an unreadable or absent diff.patch is an absent
    # input (``_read_diff_text`` degrades to ""), so mandatory risk categories
    # grounded on changed text cannot be ruled out and the item selects -- never
    # skips, whatever else the item's own evidence says (verify_selection:29-31).
    decisions = select_items([_item("item:1")], provenance={"item:1": _provenance("item:1")},
        diff_text="", config=SelectionConfig(verify_all=False, extra_categories=()),
    )
    assert decisions[0].selected is True
    assert decisions[0].reason_code == "unreadable_diff"

def _decision(item_uid: str, *, digest: str, item_id: int = 1) -> SelectionDecision:
    return SelectionDecision(
        item_uid=item_uid, item_id=item_id, selected=True, reason_code="unadjudicated:no_provenance",
        reason="no recorded adjudication", provenance={}, content_digest=digest,
    )

def test_reused_verdict_is_rekeyed_to_the_current_item_id() -> None:
    # A resume that renumbers merged ids keeps the verdict (durable uid plus the
    # id-invariant content digest) but the prior body's ``issue_id`` is stale.
    # ``_attach_verdicts`` binds verdicts by the current ``id`` -> ``issue_id``
    # join, so the reused body must be re-keyed or it binds to whichever item now
    # holds the old number (or is silently unmatched).
    decisions = [_decision("item:1", digest="d1", item_id=7)]
    prior = {"rule_version": SELECTION_RULE_VERSION,
        "decisions": [{"item_uid": "item:1", "item_id": 2, "content_digest": "d1"}],
        "verdicts": [{"issue_id": 2, "verdict": "consistent", "evidence": "e", "unverified_assumptions": []}],
    }
    reused, to_verify = plan_reuse(prior, decisions)
    assert reused["item:1"]["issue_id"] == 7
    assert reused["item:1"]["verdict"] == "consistent"
    assert to_verify == []

def test_a_moved_rule_version_invalidates_every_prior_decision() -> None:
    prior = {"rule_version": SELECTION_RULE_VERSION + 1, "decisions": [{"item_uid": "item:1", "content_digest": "d1"}],
        "verdicts": [{"issue_id": 1}],
    }
    reused, to_verify = plan_reuse(prior, [_decision("item:1", digest="d1")])
    assert reused == {} and [d.item_uid for d in to_verify] == ["item:1"]

def test_unresolved_prior_verdicts_are_reverified_and_absent_prior_is_a_miss() -> None:
    prior = {"rule_version": SELECTION_RULE_VERSION, "decisions": [{"item_uid": "item:1", "content_digest": "d1"}],
        "verdicts": [{"issue_id": 1, "verdict": "contradicts", "evidence": "e"}],
    }
    reused, to_verify = plan_reuse(prior, [_decision("item:1", digest="d1")])
    assert reused == {} and [d.item_uid for d in to_verify] == ["item:1"]
    assert plan_reuse(None, [_decision("item:1", digest="d1")]) == ({}, [_decision("item:1", digest="d1")])
    assert plan_reuse(cast(Any, []), [_decision("item:1", digest="d1")])[0] == {}

def test_unknown_extra_category_fails_loudly_and_digest_is_content_only() -> None:
    with pytest.raises(UnknownRiskCategoryError, match="public_interface"):
        resolve_selection_config(verify_all=False, extra_categories=["public_interface"])
    a = _item("item:1")
    assert item_content_digest(a) == item_content_digest({**a, "id": 7, "verifier_verdict": "consistent"})
    assert item_content_digest(a) != item_content_digest({**a, "evidence": "a.py:1 rewritten"})
