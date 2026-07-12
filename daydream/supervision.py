"""Runtime findings and tool supervision helpers."""

from __future__ import annotations

from typing import Any


_REVISABLE_FINDING_FIELDS = (
    "severity",
    "confidence",
    "description",
    "rationale",
    "evidence",
)

FindingVerdictEvent = tuple[int, str, str]


def revise_finding_fields(record: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Apply only verdict fields that may revise a finding in place."""
    for field in _REVISABLE_FINDING_FIELDS:
        if field in verdict:
            record[field] = verdict[field]


def apply_findings_verdicts(
    items: list[dict[str, Any]],
    verdicts: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[FindingVerdictEvent]]:
    """Apply id-keyed findings verdicts in one fail-open pass."""
    kept: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    events: list[FindingVerdictEvent] = []

    for item in items:
        item_id = item.get("id")
        verdict = verdicts.get(item_id) if isinstance(item_id, int) else None
        if verdict is None or verdict.get("id") != item_id:
            kept.append(item)
            continue

        action = verdict.get("action")
        if action == "drop":
            events.append((item_id, action, str(verdict.get("reason", ""))))
        elif action == "edit":
            revise_finding_fields(item, verdict)
            kept.append(item)
            events.append((item_id, action, str(verdict.get("reason", ""))))
        elif action == "hold":
            held.append(item)
            events.append((item_id, action, str(verdict.get("reason", ""))))
        else:
            kept.append(item)

    return kept, held, events
