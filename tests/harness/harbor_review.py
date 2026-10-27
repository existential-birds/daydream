"""Scripted model answers shared by native Harbor and packaged-container proofs."""
from __future__ import annotations

import json
from typing import Any


def assignment_from_request(request: dict[str, Any]) -> dict[str, Any]:
    user, = [message for message in request["messages"] if message["role"] == "user"]
    content = user["content"]
    text = content if isinstance(content, str) else "\n".join(block.get("text", "") for block in content)
    assignment: dict[str, Any] = json.loads(text.split("\nCommitted review assignment:\n", 1)[1].split(
        "\nEnd committed review assignment.\n", 1)[0])
    return assignment


def answer_review(
    request: dict[str, Any], *, empty: bool = False, partial: bool = False,
) -> list[tuple[str, str, dict[str, Any]]]:
    assignment = assignment_from_request(request)
    kind = assignment["kind"]
    if not any(message["role"] == "tool" for message in request["messages"]):
        return [(f"{kind}-read", "read", {"path": "feature.py"})]
    refs_by_target = {part["target_id"]: [{
        "source_id": next(option["source_id"] for option in part["source_options"] if option["side"] == "new"),
        "source_range": None, "basis": "supplied_diff",
    }] for part in assignment["assignments"]}
    feature_part = next(part for part in assignment["assignments"] if part["file"] == "feature.py")
    refs = refs_by_target[feature_part["target_id"]]
    if kind == "discovery":
        payload = {"targets": [{"target_id": target_id, "status": "reviewed", "reason": "Captured source checked.",
                                "evidence_refs": target_refs} for target_id, target_refs in refs_by_target.items()],
                   "claims": [] if empty else [{"uid": "", "revision": 0, "source_uids": [],
                       "title": "Captured feature has a reproducible defect", "payload": {
                           "claim": "The changed feature loses the required second value.",
                           "trigger": "Use the changed feature with the second input.",
                           "consequence": "The caller receives the wrong value.",
                           "grounds": "The captured second line demonstrates the changed value.",
                           "file": "feature.py", "line": 2, "evidence_refs": refs}}]}
    else:
        claim, = assignment["claims"]
        payload = {"verdicts": [{"uid": claim["uid"], "revision": claim["revision"],
            "verdict": "unresolved" if partial else "confirmed", "grounds": "Fresh captured source checked.",
            "evidence_refs": refs}], "claims": []}
    return [(f"{kind}-submit", "structured_output", payload)]
