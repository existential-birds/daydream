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

@pytest.mark.parametrize(("item_over", "prov", "expected_reason"),
    [({"lens": "cross-stack"}, _provenance("item:1"), "cross_stack"),
        ({"description": "auth token check missing"}, _provenance("item:1"), "risk_category:security"),
        ({"confidence": "MEDIUM"}, _provenance("item:1"), "weak_evidence:confidence"),
        ({"evidence": "n/a"}, _provenance("item:1"), "weak_evidence:placeholder_evidence"),
        ({"location_distrust": True}, _provenance("item:1"), "weak_evidence:located_beyond_tolerance"),
        ({}, None, "unadjudicated:no_provenance"),
        ({}, _provenance("item:1", verdict_bound=False), "unadjudicated:verdict_unbound"),
        ({}, _provenance("item:1", revised_fields=("severity",)), "materially_revised"),
    ],
)
def test_every_mandatory_select_branch_is_reachable(
    item_over: dict[str, object], prov: RecordProvenance | None, expected_reason: str
) -> None:
    decisions = select_items([_item("item:1", **item_over)], provenance={} if prov is None else {"item:1": prov},
        diff_text="", config=SelectionConfig(verify_all=False, extra_categories=()),
    )
    assert decisions[0].selected is True
    assert decisions[0].reason_code == expected_reason

def test_only_a_strong_adjudicated_routine_item_skips() -> None:
    # The diff is present and readable but never covers the cited file (a.py), so
    # the changed-line signal is absent while the input itself is not: the skip
    # branch may certify routine strength. An *absent* diff is an absent input
    # and selects instead (see the invariant test below).
    decisions = select_items([_item("item:1")], provenance={"item:1": _provenance("item:1")},
        diff_text=_DIFF, config=SelectionConfig(verify_all=False, extra_categories=()),
    )
    assert decisions[0].selected is False
    assert decisions[0].reason_code == "strongly_evidenced_adjudicated_routine"

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
    with pytest.raises(UnknownRiskCategoryError):
        resolve_selection_config(verify_all=False, extra_categories=["nope"])
    a = _item("item:1")
    assert item_content_digest(a) == item_content_digest({**a, "id": 7, "verifier_verdict": "consistent"})
    assert item_content_digest(a) != item_content_digest({**a, "evidence": "a.py:1 rewritten"})
