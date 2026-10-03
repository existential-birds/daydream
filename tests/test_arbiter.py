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


def test_select_arbiter_targets_honors_min_severity_knob() -> None:
    records = [{"severity": "medium", "file": "a.py", "line": 1}, {"severity": "high", "file": "b.py", "line": 2},
        {"severity": "low", "file": "c.py", "line": 3},
    ]
    sources = ["s1", "s2", "s3"]
    # Default unchanged: only the high record is selected.
    assert _selected(records, sources) == [1]
    # Knob lowered: medium is now arbitrated too.
    assert _selected(records, sources, min_severity="medium") == [0, 1]

def test_mixed_severity_multi_stack_collision_selects_high_and_contested() -> None:
    # Index map:
    #  0 python  api.py:10  high     -> selected (high severity)
    #  1 react   api.py:10  low      -> selected (contested: same loc, 2 stacks, divergent sev)
    #  2 python  util.py:5  medium   -> NOT selected (uncontested, not high)
    #  3 go      util.py:5  medium   -> NOT selected (same loc + 2 stacks but AGREEING severity)
    #  4 react   App.tsx:1  low      -> NOT selected (uncontested low)
    #  5 python  App.tsx:1  low      -> NOT selected (same loc, 2 stacks, but agreeing severity)
    records = [_rec("api.py", 10, "high"), _rec("api.py", 10, "low"), _rec("util.py", 5, "medium"),
        _rec("util.py", 5, "medium"), _rec("App.tsx", 1, "low"), _rec("App.tsx", 1, "low"),
    ]
    sources = ["python", "react", "python", "go", "react", "python"]

    selected = _selected(records, sources)

    # 0 (high) and 1 (contested with 0 at api.py:10) selected; nothing else.
    assert selected == [0, 1]

def test_same_location_single_stack_is_not_contested() -> None:
    # Two divergent-severity records at the same loc but from the SAME stack:
    # not contested (contested requires >=2 distinct stacks). Neither is high.
    records = [_rec("a.py", 3, "medium"), _rec("a.py", 3, "low")]
    sources = ["python", "python"]
    assert _selected(records, sources) == []

def test_all_low_uncontested_selects_nothing() -> None:
    records = [_rec("a.py", 1, "low"), _rec("b.py", 2, "low"), _rec("c.py", 3, "medium")]
    sources = ["python", "react", "go"]
    assert _selected(records, sources) == []

def test_high_severity_always_selected_even_when_alone() -> None:
    records = [_rec("a.py", 1, "low"), _rec("b.py", 2, "high")]
    sources = ["python", "react"]
    assert _selected(records, sources) == [1]

def test_missing_severity_only_selectable_via_contested() -> None:
    # Missing severity cannot select a record or diverge when both values are absent.
    bare = {"id": 1, "description": "d", "file": "x.py", "line": 1}
    assert _selected([dict(bare), dict(bare)], ["python", "react"]) == []

def test_missing_scope_uid_raises() -> None:
    with pytest.raises(ValueError, match="scope UIDs"):
        select_arbiter_targets([_rec("a.py", 1, "high")])


# Suppression selects borderline severity/confidence outside arbiter exclusions.

def test_suppression_selects_low_confidence_and_low_severity_uncontested() -> None:
    # Index map (no exclusions):
    #  0 low-severity MEDIUM-confidence   -> selected (low severity)
    #  1 medium-severity LOW-confidence   -> selected (LOW confidence)
    #  2 medium-severity MEDIUM-confidence-> NOT selected (borderline on neither axis)
    records = [_rec_conf("a.py", 1, "low", "MEDIUM"), _rec_conf("b.py", 2, "medium", "LOW"),
        _rec_conf("c.py", 3, "medium", "MEDIUM"),
    ]
    assert select_suppression_targets(records) == [0, 1]

def test_suppression_excludes_arbiter_targets() -> None:
    # A high finding + a LOW-confidence uncontested finding. The arbiter takes the
    # high one; suppression must take ONLY the borderline one, never the high.
    records = [_rec_conf("api.py", 10, "high", "HIGH"), _rec_conf("util.py", 5, "low", "LOW")]
    sources = ["python", "react"]
    arbiter_targets = _selected(records, sources)
    assert arbiter_targets == [0]
    assert select_suppression_targets(records, arbiter_targets) == [1]

def test_suppression_excludes_contested_low_finding() -> None:
    # Contested low-severity records belong to arbitration, not suppression.
    records = [
        _rec_conf("api.py", 10, "high", "HIGH"),  # 0 contested + high
        _rec_conf("api.py", 10, "low", "LOW"),    # 1 contested (excluded despite low)
        _rec_conf("b.py", 2, "low", "LOW"),       # 2 borderline uncontested -> selected
    ]
    sources = ["python", "react", "go"]
    arbiter_targets = _selected(records, sources)
    assert arbiter_targets == [0, 1]
    assert select_suppression_targets(records, arbiter_targets) == [2]

def test_suppression_selects_nothing_when_all_medium_uncontested() -> None:
    records = [_rec_conf("a.py", 1, "medium", "MEDIUM"), _rec_conf("b.py", 2, "medium", "HIGH")]
    assert select_suppression_targets(records) == []

def test_suppression_default_exclude_is_empty() -> None:
    # Called without an exclude set, every borderline record is selected.
    records = [_rec_conf("a.py", 1, "low", "LOW")]
    assert select_suppression_targets(records) == [0]

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

def test_suppression_rejects_unknown_confidence_class() -> None:

    records = [_rec_conf("a.py", 1, "low", "LOW")]
    with pytest.raises(ValueError):
        select_suppression_targets(records, confidence_classes=("LOW", "GUESSED"))

def test_contested_only_records_skip_the_severity_branch() -> None:
    """Structural records marked contested_only require a contest even at high severity."""
    records = [_rec("api.py", 10, "high"), _rec("api.py", 20, "high")]
    sources = ["python", "structure"]
    assert _selected(records, sources) == [0, 1]
    assert _selected(records, sources, contested_only=[1]) == [0]

def test_contested_only_record_is_still_selected_when_contested() -> None:
    """Severity exemption still permits a genuine cross-stack location contest."""
    records = [_rec("svc/config.yaml", 29, "medium"), _rec("svc/config.yaml", 29, "high")]
    sources = ["python", "structure"]
    assert _selected(records, sources, contested_only=[1]) == [0, 1]

def test_whole_file_record_contests_every_line_in_that_file() -> None:
    """A whole-file line:0 anchor contests any line-anchored twin in the same file."""
    records = [_rec("svc/loader.py", 88, "medium"), _rec("svc/loader.py", 0, "high")]
    sources = ["python", "structure"]
    assert _selected(records, sources, contested_only=[1]) == [0, 1]

def test_whole_file_record_does_not_reach_across_files() -> None:
    """The widening is per-file: a whole-file finding contests only its own file."""
    records = [_rec("svc/loader.py", 88, "medium"), _rec("svc/other.py", 0, "low")]
    sources = ["python", "structure"]
    assert _selected(records, sources, contested_only=[1]) == []

def test_two_whole_file_records_contest_each_other() -> None:
    """Whole-file records can contest one another without any line-anchored finding."""
    records = [_rec("svc/loader.py", 0, "medium"), _rec("svc/loader.py", 0, "high")]
    sources = ["python", "structure"]
    assert _selected(records, sources, contested_only=[1]) == [0, 1]

def test_whole_file_record_agreeing_on_severity_is_not_contested() -> None:
    """The widening changes grouping only; divergent severity is still required."""
    records = [_rec("svc/loader.py", 88, "medium"), _rec("svc/loader.py", 0, "medium")]
    sources = ["python", "structure"]
    assert _selected(records, sources, contested_only=[1]) == []

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


# Stack identity comes solely from the host UID; fixture labels cannot change it.
# Bare names and stack-<name>-records.json paths can identify the same stack.

def test_one_stack_spelled_two_ways_is_not_contested() -> None:
    """Different source spellings cannot make one stack's divergent severities a contest."""
    records = [_rec("api.py", 10, "medium", uid="python:1"), _rec("api.py", 10, "low", uid="python:2")]
    sources = ["stack-python-records.json", "python"]
    assert _selected(records, sources) == []

def test_genuine_cross_stack_contest_survives_normalization() -> None:
    """Source normalization preserves genuinely distinct stacks and their contest."""
    records = [_rec("api.py", 10, "medium", uid="python:1"), _rec("api.py", 10, "low", uid="react:1")]
    sources = ["stack-python-records.json", "react"]
    assert _selected(records, sources) == [0, 1]

def test_uid_less_contests_are_rejected() -> None:
    for records in ([_rec("api.py", 10, "medium"), _rec("api.py", 10, "low")], []):
        if records:
            with pytest.raises(ValueError, match="scope UIDs"):
                select_arbiter_targets(records)
        else:
            assert select_arbiter_targets(records) == []


def test_uid_outranks_the_source_tag() -> None:
    """Birth UIDs outrank re-derived source tags when determining distinct stacks."""
    records = [_rec("api.py", 10, "medium", uid="python:1"), _rec("api.py", 10, "low", uid="react:1")]
    sources = ["stack-python-records.json", "stack-python-records.json"]
    assert _selected(records, sources) == [0, 1]
