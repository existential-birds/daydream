"""Recommendation-verifier selection.

Every canonical item gets one decision, in order, and every finding is verified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from daydream.deep.records import item_uid


@dataclass(frozen=True)
class SelectionDecision:
    """One canonical item's selection verdict, with the evidence that produced it."""

    item_uid: str
    item_id: int | None
    selected: bool
    reason_code: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe artifact body for this decision (Pattern A)."""
        return {
            "item_uid": self.item_uid,
            "item_id": self.item_id,
            "selected": self.selected,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


def select_items(items: Sequence[Mapping[str, Any]]) -> list[SelectionDecision]:
    """Return exactly one decision per input item, in input order."""
    decisions = []
    for item in items:
        data = dict(item) if isinstance(item, Mapping) else {}
        item_id = data.get("id")
        decisions.append(SelectionDecision(
            item_uid=item_uid(data),
            item_id=item_id if isinstance(item_id, int) and not isinstance(item_id, bool) else None,
            selected=True,
            reason_code="verify_all",
            reason="every finding is verified",
        ))
    return decisions


__all__ = [
    "SelectionDecision",
    "select_items",
]
