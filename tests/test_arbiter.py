"""Unit tests for the scoped-arbiter selection predicate (issue #168).

``select_arbiter_targets`` decides which parsed per-stack records reach the
expensive Opus arbiter: every high-severity record, plus every record at a
``(file, line)`` location *contested* across >=2 stacks with divergent severity.
Low/medium uncontested findings must never be selected — that is the cost split.

These tests drive the predicate against the structural shape it exists to
handle: a mixed-severity, multi-stack, same-``file:line`` collision, alongside
the near-miss shapes (same location but one stack; same location but agreeing
severity) that must NOT trip contested selection.
"""

from __future__ import annotations

from daydream.deep.arbiter import select_arbiter_targets, select_suppression_targets


def _rec(file: str, line: int, severity: str) -> dict[str, object]:
    return {
        "id": 1,
        "description": f"{severity} finding at {file}:{line}",
        "file": file,
        "line": line,
        "severity": severity,
        "confidence": "MEDIUM",
        "rationale": "because",
    }


def _rec_conf(file: str, line: int, severity: str, confidence: str) -> dict[str, object]:
    rec = _rec(file, line, severity)
    rec["confidence"] = confidence
    return rec


def test_mixed_severity_multi_stack_collision_selects_high_and_contested() -> None:
    # Index map:
    #  0 python  api.py:10  high     -> selected (high severity)
    #  1 react   api.py:10  low      -> selected (contested: same loc, 2 stacks, divergent sev)
    #  2 python  util.py:5  medium   -> NOT selected (uncontested, not high)
    #  3 go      util.py:5  medium   -> NOT selected (same loc + 2 stacks but AGREEING severity)
    #  4 react   App.tsx:1  low      -> NOT selected (uncontested low)
    #  5 python  App.tsx:1  low      -> NOT selected (same loc, 2 stacks, but agreeing severity)
    records = [
        _rec("api.py", 10, "high"),
        _rec("api.py", 10, "low"),
        _rec("util.py", 5, "medium"),
        _rec("util.py", 5, "medium"),
        _rec("App.tsx", 1, "low"),
        _rec("App.tsx", 1, "low"),
    ]
    sources = ["python", "react", "python", "go", "react", "python"]

    selected = select_arbiter_targets(records, sources)

    # 0 (high) and 1 (contested with 0 at api.py:10) selected; nothing else.
    assert selected == [0, 1]


def test_same_location_single_stack_is_not_contested() -> None:
    # Two divergent-severity records at the same loc but from the SAME stack:
    # not contested (contested requires >=2 distinct stacks). Neither is high.
    records = [_rec("a.py", 3, "medium"), _rec("a.py", 3, "low")]
    sources = ["python", "python"]
    assert select_arbiter_targets(records, sources) == []


def test_all_low_uncontested_selects_nothing() -> None:
    records = [_rec("a.py", 1, "low"), _rec("b.py", 2, "low"), _rec("c.py", 3, "medium")]
    sources = ["python", "react", "go"]
    assert select_arbiter_targets(records, sources) == []


def test_high_severity_always_selected_even_when_alone() -> None:
    records = [_rec("a.py", 1, "low"), _rec("b.py", 2, "high")]
    sources = ["python", "react"]
    assert select_arbiter_targets(records, sources) == [1]


def test_missing_severity_only_selectable_via_contested() -> None:
    # A record with no severity field (the legacy FEEDBACK_SCHEMA shape) is never
    # "high", so it can only be pulled in by a contested collision. Here both
    # records at x.py:1 lack severity -> severities collapse to {""} -> not
    # contested -> nothing selected.
    bare = {"id": 1, "description": "d", "file": "x.py", "line": 1}
    assert select_arbiter_targets([dict(bare), dict(bare)], ["python", "react"]) == []


def test_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        select_arbiter_targets([_rec("a.py", 1, "high")], ["python", "react"])


# Issue #232: precision-mode suppression selection predicate.
#
# ``select_suppression_targets`` picks the borderline, uncontested findings the
# arbiter never sees: LOW-confidence and/or low-severity records NOT in the
# arbiter's exclusion set. High/contested records (the arbiter's job) must never
# be selected here.


def test_suppression_selects_low_confidence_and_low_severity_uncontested() -> None:
    # Index map (no exclusions):
    #  0 low-severity MEDIUM-confidence   -> selected (low severity)
    #  1 medium-severity LOW-confidence   -> selected (LOW confidence)
    #  2 medium-severity MEDIUM-confidence-> NOT selected (borderline on neither axis)
    records = [
        _rec_conf("a.py", 1, "low", "MEDIUM"),
        _rec_conf("b.py", 2, "medium", "LOW"),
        _rec_conf("c.py", 3, "medium", "MEDIUM"),
    ]
    sources = ["python", "react", "go"]
    assert select_suppression_targets(records, sources) == [0, 1]


def test_suppression_excludes_arbiter_targets() -> None:
    # A high finding + a LOW-confidence uncontested finding. The arbiter takes the
    # high one; suppression must take ONLY the borderline one, never the high.
    records = [
        _rec_conf("api.py", 10, "high", "HIGH"),
        _rec_conf("util.py", 5, "low", "LOW"),
    ]
    sources = ["python", "react"]
    arbiter_targets = select_arbiter_targets(records, sources)
    assert arbiter_targets == [0]
    assert select_suppression_targets(records, sources, arbiter_targets) == [1]


def test_suppression_excludes_contested_low_finding() -> None:
    # A low-severity finding that is CONTESTED (same loc, 2 stacks, divergent
    # severity) reaches the arbiter, so it must be excluded from suppression even
    # though it is low severity.
    records = [
        _rec_conf("api.py", 10, "high", "HIGH"),  # 0 contested + high
        _rec_conf("api.py", 10, "low", "LOW"),    # 1 contested (excluded despite low)
        _rec_conf("b.py", 2, "low", "LOW"),       # 2 borderline uncontested -> selected
    ]
    sources = ["python", "react", "go"]
    arbiter_targets = select_arbiter_targets(records, sources)
    assert arbiter_targets == [0, 1]
    assert select_suppression_targets(records, sources, arbiter_targets) == [2]


def test_suppression_selects_nothing_when_all_medium_uncontested() -> None:
    records = [_rec_conf("a.py", 1, "medium", "MEDIUM"), _rec_conf("b.py", 2, "medium", "HIGH")]
    sources = ["python", "react"]
    assert select_suppression_targets(records, sources) == []


def test_suppression_default_exclude_is_empty() -> None:
    # Called without an exclude set, every borderline record is selected.
    records = [_rec_conf("a.py", 1, "low", "LOW")]
    assert select_suppression_targets(records, ["python"]) == [0]


def test_suppression_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        select_suppression_targets([_rec("a.py", 1, "low")], ["python", "react"])
