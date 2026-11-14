"""Recommendation-verifier selection.

Every canonical item gets one decision, in order. Each finding is verified; only an
item of a verdict-exempt lens stays out of the verifier.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from daydream.deep.records import item_uid
from daydream.json_utils import canonical_json

#: On-disk selection-rule version, stored in the verdicts artifact.
SELECTION_RULE_VERSION: int = 1

#: The verifier-relevant components of one canonical item, and only those.
_DIGEST_FIELDS: tuple[str, ...] = (
    "item_uid",
    "lens",
    "file",
    "line",
    "severity",
    "confidence",
    "description",
    "rationale",
    "evidence",
)

#: Lenses the verifier never adjudicates, with the exemption reason each records.
_EXEMPT_LENSES: dict[str, str] = {
    "structural": "exempt:structural",
    "wonder": "exempt:wonder",
}


@dataclass(frozen=True)
class SelectionDecision:
    """One canonical item's selection verdict, with the evidence that produced it."""

    item_uid: str
    item_id: int | None
    selected: bool
    reason_code: str
    reason: str
    provenance: dict[str, Any]
    content_digest: str
    verdict_reused: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe artifact body for this decision (Pattern A)."""
        return {
            "item_uid": self.item_uid,
            "item_id": self.item_id,
            "selected": self.selected,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "provenance": self.provenance,
            "content_digest": self.content_digest,
            "verdict_reused": self.verdict_reused,
        }


def item_content_digest(item: Mapping[str, Any]) -> str:
    """Hash verifier-relevant fields, excluding positional IDs and attached verdicts; missing fields hash as None."""
    payload = {field: item.get(field) for field in _DIGEST_FIELDS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def select_items(items: Sequence[Mapping[str, Any]]) -> list[SelectionDecision]:
    """Return exactly one decision per input item, in input order."""
    decisions = []
    for item in items:
        data = dict(item) if isinstance(item, Mapping) else {}
        item_id = data.get("id")
        lens = str(data.get("lens", ""))
        exempt = _EXEMPT_LENSES.get(lens)
        decisions.append(SelectionDecision(
            item_uid=item_uid(data),
            item_id=item_id if isinstance(item_id, int) and not isinstance(item_id, bool) else None,
            selected=exempt is None,
            reason_code=exempt or "verify_all",
            reason=f"lens {lens!r} is verdict-exempt" if exempt else "every finding is verified",
            provenance={},
            content_digest=item_content_digest(data),
        ))
    return decisions


__all__ = [
    "SELECTION_RULE_VERSION",
    "SelectionDecision",
    "item_content_digest",
    "select_items",
]
