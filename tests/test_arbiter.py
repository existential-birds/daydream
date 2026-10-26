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


def test_arbiter_membership_retains_file_boundaries_and_missing_severity() -> None:
    """One population retains empty severity, file isolation, and whole-file agreement."""
    bare = {"id": 1, "description": "d", "file": "missing.py", "line": 1}
    records = [dict(bare), dict(bare),
        _rec("line.py", 88, "medium"), _rec("other.py", 0, "low"),
        _rec("whole.py", 0, "medium"), _rec("whole.py", 0, "high"),
        _rec("agree.py", 88, "medium"), _rec("agree.py", 0, "medium"),
    ]
    sources = ["python", "react", "python", "structure", "python", "structure", "python", "structure"]
    assert _selected(records, sources, contested_only=[3, 5, 7]) == [4, 5]


# Suppression selects borderline severity/confidence outside arbiter exclusions.

def test_suppression_rejects_unknown_confidence_class() -> None:

    records = [_rec_conf("a.py", 1, "low", "LOW")]
    with pytest.raises(ValueError):
        select_suppression_targets(records, confidence_classes=("LOW", "GUESSED"))

# Stack identity comes solely from the host UID; fixture labels cannot change it.
# Bare names and stack-<name>-records.json paths can identify the same stack.

def test_uid_less_contests_are_rejected() -> None:
    for records in ([_rec("api.py", 10, "medium"), _rec("api.py", 10, "low")], []):
        if records:
            with pytest.raises(ValueError, match="scope UIDs"):
                select_arbiter_targets(records)
        else:
            assert select_arbiter_targets(records) == []

def test_select_suppression_targets_honors_severity_classes_knob() -> None:
    records = [{"severity": "low", "file": "a.py", "line": 1}, {"severity": "medium", "file": "b.py", "line": 2},
        {"severity": "low", "confidence": "LOW", "file": "c.py", "line": 3},
    ]
    # Default ("low",): low-severity records selected; medium not; LOW-confidence still selected.
    assert select_suppression_targets(records) == [0, 2]
    # Knob widened to include medium.
    assert select_suppression_targets(records, severity_classes=("low", "medium")) == [0, 1, 2]

def test_select_suppression_targets_honors_confidence_classes_knob() -> None:
    """The profile's ``Suppression.confidence_classes`` governs the confidence branch live, not a hardcoded LOW
    predicate (fail-open fix): widening to include MEDIUM routes otherwise-borderline MEDIUM-confidence findings
    to the suppression pass, narrowing to HIGH excludes LOW-confidence ones."""
    records = [{"severity": "medium", "confidence": "LOW", "file": "a.py", "line": 1},
        {"severity": "medium", "confidence": "MEDIUM", "file": "b.py", "line": 2},
        {"severity": "medium", "confidence": "HIGH", "file": "c.py", "line": 3},
    ]
    # Default ("LOW",): LOW-confidence selected; MEDIUM- and HIGH-confidence not.
    assert select_suppression_targets(records) == [0]
    # Widened to include MEDIUM: now selects LOW- and MEDIUM-confidence.
    assert select_suppression_targets(records, confidence_classes=("LOW", "MEDIUM")) == [0, 1]
    # Narrowed to HIGH (or any other non-LOW selection) must NOT silently fall
    # back to the old LOW-only branch -- the knob is live in both directions.
    assert select_suppression_targets(records, confidence_classes=("HIGH",)) == [2]

def test_select_arbiter_targets_honors_contested_location_knob() -> None:
    """The profile's ``Arbitration.contested_location`` gates the contested branch of arbiter selection: disabling it
    leaves only the severity branch, so divergent-severity multi-stack collisions no longer route to the arbiter."""
    records = [_rec("api.py", 10, "high"), _rec("api.py", 10, "medium")]
    sources = ["python", "react"]
    # Default on: the medium finding is contested with the high one and is selected.
    assert _selected(records, sources) == [0, 1]
    # Knob off: only the severity branch selects; the contested medium drops out.
    assert _selected(records, sources, contested_location=False) == [0]
    # Severity branch still fully live when the contested branch is disabled:
    # lowering min_severity to medium pulls the medium finding back in.
    assert _selected(records, sources, min_severity="medium", contested_location=False) == [0, 1]
