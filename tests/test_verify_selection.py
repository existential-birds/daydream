from dataclasses import replace

from daydream.deep.verify_selection import SelectionConfig, SelectionDecision, item_content_digest, select_items

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


def test_selective_verification_selects_every_item_with_its_own_reason() -> None:
    """One selection batch retains each sibling's identity, distinct reason and input order."""
    items = [
        _item("item:1"), _item("item:2", lens="cross-stack"),
        _item("item:3", description="the handler must authenticate the caller's token"),
        _item("item:4", description="this path can deadlock while holding the mutex"),
        _item("item:5", description="the change runs `alter table users`"),
        _item("item:6", description="the grpc endpoint gains a field"),
        _item("item:7", description="see migrations/0042_add_flag.py"),
        _item("item:8", confidence="MEDIUM"), _item("item:9", evidence="n/a"),
        _item("item:10", location_distrust=True), _item("item:11", lens="structural"),
    ]
    for index, item in enumerate(items, 1):
        item["id"] = index
    decisions = select_items(items, diff_text=_DIFF, config=SelectionConfig(verify_all=False))
    assert [(d.item_uid, d.item_id, d.selected, d.reason_code) for d in decisions] == [
        # No reviewer certifies a finding as routine, so a strong routine finding is verified too.
        ("item:1", 1, True, "unadjudicated:no_provenance"), ("item:2", 2, True, "cross_stack"),
        ("item:3", 3, True, "risk_category:security"), ("item:4", 4, True, "risk_category:concurrency"),
        ("item:5", 5, True, "risk_category:persistence"),
        ("item:6", 6, True, "risk_category:public-interface"),
        ("item:7", 7, True, "risk_category:migration"), ("item:8", 8, True, "weak_evidence:confidence"),
        ("item:9", 9, True, "weak_evidence:placeholder_evidence"),
        ("item:10", 10, True, "weak_evidence:located_beyond_tolerance"),
        ("item:11", 11, False, "exempt:structural"),
    ]
    assert all(d.provenance == {} and d.verdict_reused is False for d in decisions)

def _decision(item_uid: str, *, digest: str, item_id: int = 1) -> SelectionDecision:
    return SelectionDecision(
        item_uid=item_uid, item_id=item_id, selected=True, reason_code="unadjudicated:no_provenance",
        reason="no recorded adjudication", provenance={}, content_digest=digest,
    )

def test_fresh_verdicts_bind_only_to_the_findings_asked_about() -> None:
    """Verdicts for findings the verifier was not shown are dropped before they reach a finding."""
    from daydream.deep.fix_steps import _attach_verdicts
    from daydream.phases.verify import _asked_verdicts
    current = [_decision("item:1", digest="d1", item_id=9), _decision("item:2", digest="d2", item_id=2),
               _decision("item:3", digest="d3", item_id=3)]
    shown = {"issue_id": 2, "verdict": "uncertain", "evidence": "shown"}
    stray = [{"issue_id": n, "verdict": "contradicts", "evidence": "stray"} for n in (3, 99, "2")]
    decisions = [*current[:2], replace(current[2], selected=False)]
    verdicts = _asked_verdicts([shown, *stray], decisions)
    assert verdicts == [shown]
    findings = [_item("item:1", id=9), _item("item:2", id=2), _item("item:3", id=3)]
    assert _attach_verdicts(findings, {"verdicts": verdicts}) == [
        _item("item:1", id=9),
        {**_item("item:2", id=2), "verifier_verdict": "uncertain", "evidence": "shown", "unverified_assumptions": []},
        _item("item:3", id=3)]

    a = _item("item:1")
    assert item_content_digest(a) == item_content_digest({**a, "id": 7, "verifier_verdict": "consistent"})
    assert item_content_digest(a) != item_content_digest({**a, "evidence": "a.py:1 rewritten"})
