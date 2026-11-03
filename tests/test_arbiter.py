"""Arbiter and suppression selection over mixed-stack review records.

Arbitration selects high severity or divergent severity at a shared location
across distinct stacks. Canonical stack identity prevents alternate source
spellings from creating a false contest.
"""

from __future__ import annotations

import pytest

from daydream.deep.arbiter import select_arbiter_targets, select_suppression_targets
from daydream.deep.records import stack_name_from_records_source


def _rec(file: str, line: int, severity: str, uid: str | None = None) -> dict[str, object]:
    """Build a finding; fixture scopes supply host UIDs before selection."""
    record: dict[str, object] = {
        "id": 1, "description": f"{severity} finding at {file}:{line}", "file": file, "line": line,
        "severity": severity, "confidence": "MEDIUM", "rationale": "because",
    }
    if uid is not None:
        record["uid"] = uid
    return record


def _rec_conf(file: str, line: int, severity: str, confidence: str) -> dict[str, object]:
    rec = _rec(file, line, severity)
    rec["confidence"] = confidence
    return rec

def _selected(records: list[dict[str, object]], scopes: list[str], **options: object) -> list[int]:
    for ordinal, (record, scope) in enumerate(zip(records, scopes, strict=True), 1):
        record.setdefault("uid", f"{stack_name_from_records_source(scope)}:{ordinal}")
    return select_arbiter_targets(records, **options)  # type: ignore[arg-type]


# Suppression selects borderline severity/confidence outside arbiter exclusions.

def test_target_selection_requires_scope_uids_known_classes_and_honors_contested_location() -> None:
    """The profile's ``Arbitration.contested_location`` gates the contested branch of arbiter selection: disabling it
    leaves only the severity branch, so divergent-severity multi-stack collisions no longer route to the arbiter."""
    with pytest.raises(ValueError):
        select_suppression_targets([_rec_conf("a.py", 1, "low", "LOW")], confidence_classes=("LOW", "GUESSED"))
    # Stack identity comes solely from the host UID.
    with pytest.raises(ValueError, match="scope UIDs"):
        select_arbiter_targets([_rec("api.py", 10, "medium"), _rec("api.py", 10, "low")])
    assert select_arbiter_targets([]) == []
    records = [_rec("api.py", 10, "high"), _rec("api.py", 10, "medium")]
    sources = ["python", "react"]
    # Default on: the medium finding is contested with the high one and is selected.
    assert _selected(records, sources) == [0, 1]
    # Knob off: only the severity branch selects; the contested medium drops out.
    assert _selected(records, sources, contested_location=False) == [0]
    # Severity branch still fully live when the contested branch is disabled:
    # lowering min_severity to medium pulls the medium finding back in.
    assert _selected(records, sources, min_severity="medium", contested_location=False) == [0, 1]
