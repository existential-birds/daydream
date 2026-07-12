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


def revise_finding_fields(record: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Apply only verdict fields that may revise a finding in place."""
    for field in _REVISABLE_FINDING_FIELDS:
        if field in verdict:
            record[field] = verdict[field]
