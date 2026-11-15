from dataclasses import replace

from daydream.deep.verify_selection import SelectionDecision, select_items


def _item(item_uid: str, **over: object) -> dict[str, object]:
    base = {"item_uid": item_uid, "id": 1, "lens": "per-stack", "file": "a.py", "line": 1,
        "severity": "low", "confidence": "HIGH", "description": "routine cleanup",
        "rationale": "pure rename", "evidence": "a.py:1 renamed helper",
    }
    return {**base, **over}


def test_every_finding_is_verified() -> None:
    """One selection batch retains each sibling's identity and input order; no lens keeps a finding out."""
    items = [_item("item:1"), _item("item:2", lens="cross-stack"), _item("item:3", confidence="MEDIUM", evidence="n/a"),
             _item("item:4", lens="structural"), _item("item:5", lens="wonder"), _item("item:6", id="six")]
    for index, item in enumerate(items[:5], 1):
        item["id"] = index
    decisions = select_items(items)
    assert [(d.item_uid, d.item_id, d.selected, d.reason_code) for d in decisions] == [
        ("item:1", 1, True, "verify_all"), ("item:2", 2, True, "verify_all"), ("item:3", 3, True, "verify_all"),
        ("item:4", 4, True, "verify_all"), ("item:5", 5, True, "verify_all"),
        ("item:6", None, True, "verify_all"),
    ]
    # The stored decision carries what the fix cycle reads, and nothing for skipping or reuse.
    assert decisions[0].as_dict() == {"item_uid": "item:1", "item_id": 1, "selected": True,
                                      "reason_code": "verify_all", "reason": "every finding is verified"}

def _decision(item_uid: str, *, item_id: int = 1) -> SelectionDecision:
    return SelectionDecision(
        item_uid=item_uid, item_id=item_id, selected=True, reason_code="verify_all",
        reason="every finding is verified",
    )

def test_fresh_verdicts_bind_only_to_the_findings_asked_about() -> None:
    """Verdicts for findings the verifier was not shown are dropped before they reach a finding."""
    from daydream.deep.fix_steps import _attach_verdicts
    from daydream.phases.verify import _asked_verdicts
    current = [_decision("item:1", item_id=9), _decision("item:2", item_id=2), _decision("item:3", item_id=3)]
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
