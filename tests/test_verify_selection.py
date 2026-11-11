from dataclasses import replace
from typing import Any, cast

from daydream.deep.adjudication_provenance import RecordProvenance
from daydream.deep.verify_selection import (
    SELECTION_RULE_VERSION,
    SelectionConfig,
    SelectionDecision,
    item_content_digest,
    plan_reuse,
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

def _item(item_uid: str, **over: object) -> dict[str, object]:
    base = {"item_uid": item_uid, "id": 1, "lens": "per-stack", "file": "a.py", "line": 1,
        "severity": "low", "confidence": "HIGH", "description": "routine cleanup",
        "rationale": "pure rename", "evidence": "a.py:1 renamed helper",
    }
    return {**base, **over}


def _provenance(uid: str, *, verdict_bound: bool = True, revised_fields: tuple[str, ...] = ()) -> RecordProvenance:
    return RecordProvenance(uid, ("arbiter",), verdict_bound, True, revised_fields)

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
    config = SelectionConfig(verify_all=False)
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
    # Uniform failure direction: an unreadable or absent diff.patch is an absent
    # input (``_read_diff_text`` degrades to ""), so mandatory risk categories
    # grounded on changed text cannot be ruled out and the item selects -- never
    # skips, whatever else the item's own evidence says (verify_selection:29-31).
    decisions = select_items([_item("item:1")], provenance={"item:1": _provenance("item:1")},
        diff_text="", config=SelectionConfig(verify_all=False),
    )
    assert decisions[0].selected is True
    assert decisions[0].reason_code == "unreadable_diff"

def _decision(item_uid: str, *, digest: str, item_id: int = 1) -> SelectionDecision:
    return SelectionDecision(
        item_uid=item_uid, item_id=item_id, selected=True, reason_code="unadjudicated:no_provenance",
        reason="no recorded adjudication", provenance={}, content_digest=digest,
    )

def test_prior_verdict_reuse_requires_rule_uid_digest_and_a_resolved_verdict() -> None:
    """Resolved verdicts are reused under the current id; every other prior is a miss."""
    prior = {"rule_version": SELECTION_RULE_VERSION, "decisions": [
        {"item_uid": "item:1", "content_digest": "d1", "item_id": 4}, {"item_uid": "item:2", "content_digest": "d2"},
        {"item_uid": "item:3", "content_digest": "stale"}, "invalid"],
        "verdicts": [{"issue_id": 4, "verdict": "consistent", "evidence": "e"}, "invalid", {"issue_id": True}],
    }
    current = [_decision("item:1", digest="d1", item_id=9), _decision("item:2", digest="d2", item_id=2),
               _decision("item:3", digest="d3", item_id=3)]
    reused, to_verify = plan_reuse(prior, current)
    assert reused == {"item:1": {"issue_id": 9, "verdict": "consistent", "evidence": "e"}}
    assert [d.item_uid for d in to_verify] == ["item:2", "item:3"]
    assert plan_reuse({**prior, "decisions": "invalid"}, current) == ({}, current)
    # The verify phase reads prior selection plus verdicts from its artifact; a foreign block misses.
    from daydream.deep.fix_steps import _attach_verdicts
    from daydream.phases.verify import _merge_reused_verdicts, _prior_selection_payload
    assert _prior_selection_payload({"selection": {"rule_version": 1}, "verdicts": "invalid"}) == {
        "rule_version": 1, "verdicts": []}
    assert _prior_selection_payload({"selection": "invalid"}) is None
    shown = {"issue_id": 2, "verdict": "uncertain", "evidence": "shown"}
    stray = [{"issue_id": n, "verdict": "contradicts", "evidence": "stray"} for n in (3, 9, 99, "2")]
    decisions = [*current[:2], replace(current[2], selected=False)]
    verdicts = _merge_reused_verdicts([reused["item:1"]], [shown, *stray], decisions, reused)
    assert verdicts == [reused["item:1"], shown]
    findings = [_item("item:1", id=9), _item("item:2", id=2), _item("item:3", id=3)]
    assert _attach_verdicts(findings, {"verdicts": verdicts}) == [
        {**_item("item:1", id=9), "verifier_verdict": "consistent", "evidence": "e", "unverified_assumptions": []},
        {**_item("item:2", id=2), "verifier_verdict": "uncertain", "evidence": "shown", "unverified_assumptions": []},
        _item("item:3", id=3)]
    prior = {"rule_version": SELECTION_RULE_VERSION + 1, "decisions": [{"item_uid": "item:1", "content_digest": "d1"}],
        "verdicts": [{"issue_id": 1}],
    }
    reused, to_verify = plan_reuse(prior, [_decision("item:1", digest="d1")])
    assert reused == {} and [d.item_uid for d in to_verify] == ["item:1"]
    prior = {"rule_version": SELECTION_RULE_VERSION, "decisions": [{"item_uid": "item:1", "content_digest": "d1"}],
        "verdicts": [{"issue_id": 1, "verdict": "contradicts", "evidence": "e"}],
    }
    reused, to_verify = plan_reuse(prior, [_decision("item:1", digest="d1")])
    assert reused == {} and [d.item_uid for d in to_verify] == ["item:1"]
    assert plan_reuse(None, [_decision("item:1", digest="d1")]) == ({}, [_decision("item:1", digest="d1")])
    assert plan_reuse(cast(Any, []), [_decision("item:1", digest="d1")])[0] == {}

    a = _item("item:1")
    assert item_content_digest(a) == item_content_digest({**a, "id": 7, "verifier_verdict": "consistent"})
    assert item_content_digest(a) != item_content_digest({**a, "evidence": "a.py:1 rewritten"})
