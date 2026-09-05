"""Unit tests for the scoped-arbiter selection predicate (issue #168).

``select_arbiter_targets`` decides which parsed per-stack records reach the
expensive Opus arbiter: every high-severity record, plus every record at a
``(file, line)`` location *contested* across >=2 stacks with divergent severity.
Low/medium uncontested findings must never be selected — that is the cost split.

These tests drive the predicate against the structural shape it exists to
handle: a mixed-severity, multi-stack, same-``file:line`` collision, alongside
the near-miss shapes (same location but one stack; same location but agreeing
severity) that must NOT trip contested selection.

Issue #1111 adds the "distinct stacks" half of that predicate: ``_stack_name``
resolves each record to ONE spelling of its owning stack, so a single stack
tagged two ways cannot masquerade as a contest.
"""

from __future__ import annotations

from daydream.deep.arbiter import select_arbiter_targets, select_suppression_targets


def _rec(file: str, line: int, severity: str, uid: str | None = None) -> dict[str, object]:
    """Build a parsed per-stack record.

    ``uid`` is the host-minted ``stack:ordinal`` identity (issue #1111). It is
    optional because most tests here drive selection off the parallel ``sources``
    list alone, which is also the shape a record written by a pre-``uid`` run has
    on disk; the stack-normalization tests pass it explicitly.
    """
    record: dict[str, object] = {
        "id": 1,
        "description": f"{severity} finding at {file}:{line}",
        "file": file,
        "line": line,
        "severity": severity,
        "confidence": "MEDIUM",
        "rationale": "because",
    }
    if uid is not None:
        record["uid"] = uid
    return record


def _rec_conf(file: str, line: int, severity: str, confidence: str) -> dict[str, object]:
    rec = _rec(file, line, severity)
    rec["confidence"] = confidence
    return rec


def test_select_arbiter_targets_honors_min_severity_knob() -> None:
    records = [
        {"severity": "medium", "file": "a.py", "line": 1},
        {"severity": "high", "file": "b.py", "line": 2},
        {"severity": "low", "file": "c.py", "line": 3},
    ]
    sources = ["s1", "s2", "s3"]
    # Default unchanged: only the high record is selected.
    assert select_arbiter_targets(records, sources) == [1]
    # Knob lowered: medium is now arbitrated too.
    assert select_arbiter_targets(records, sources, min_severity="medium") == [0, 1]


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


def test_select_suppression_targets_honors_severity_classes_knob() -> None:
    records = [
        {"severity": "low", "file": "a.py", "line": 1},
        {"severity": "medium", "file": "b.py", "line": 2},
        {"severity": "low", "confidence": "LOW", "file": "c.py", "line": 3},
    ]
    sources = ["s1", "s2", "s3"]
    # Default ("low",): low-severity records selected; medium not; LOW-confidence still selected.
    assert select_suppression_targets(records, sources) == [0, 2]
    # Knob widened to include medium.
    assert select_suppression_targets(records, sources, severity_classes=("low", "medium")) == [0, 1, 2]


def test_select_suppression_targets_honors_confidence_classes_knob() -> None:
    """The profile's ``Suppression.confidence_classes`` governs the confidence
    branch live, not a hardcoded LOW predicate (fail-open fix): widening to
    include MEDIUM routes otherwise-borderline MEDIUM-confidence findings to the
    suppression pass, narrowing to HIGH excludes LOW-confidence ones."""
    records = [
        {"severity": "medium", "confidence": "LOW", "file": "a.py", "line": 1},
        {"severity": "medium", "confidence": "MEDIUM", "file": "b.py", "line": 2},
        {"severity": "medium", "confidence": "HIGH", "file": "c.py", "line": 3},
    ]
    sources = ["s1", "s2", "s3"]
    # Default ("LOW",): LOW-confidence selected; MEDIUM- and HIGH-confidence not.
    assert select_suppression_targets(records, sources) == [0]
    # Widened to include MEDIUM: now selects LOW- and MEDIUM-confidence.
    assert select_suppression_targets(
        records, sources, confidence_classes=("LOW", "MEDIUM")
    ) == [0, 1]
    # Narrowed to HIGH (or any other non-LOW selection) must NOT silently fall
    # back to the old LOW-only branch -- the knob is live in both directions.
    assert select_suppression_targets(records, sources, confidence_classes=("HIGH",)) == [2]


def test_suppression_rejects_unknown_confidence_class() -> None:
    import pytest

    records = [_rec_conf("a.py", 1, "low", "LOW")]
    with pytest.raises(ValueError):
        select_suppression_targets(records, ["python"], confidence_classes=("LOW", "GUESSED"))


def test_contested_only_records_skip_the_severity_branch() -> None:
    """``contested_only`` indices are invisible to the severity branch (#1103).

    The deep pipeline passes the structural meta-stack's records here so they can
    contest a language finding without every high-severity structural finding
    also being pulled into arbitration on its own.
    """
    records = [_rec("api.py", 10, "high"), _rec("api.py", 20, "high")]
    sources = ["python", "structure"]
    assert select_arbiter_targets(records, sources) == [0, 1]
    assert select_arbiter_targets(records, sources, contested_only=[1]) == [0]


def test_contested_only_record_is_still_selected_when_contested() -> None:
    """Issue #1103 case A: exempting the severity branch does not exempt contest.

    Same file, same line, two distinct stacks, divergent severity -- the pair the
    contested branch exists for. The structural side must come back even though
    it opted out of severity-based selection.
    """
    records = [_rec("svc/config.yaml", 29, "medium"), _rec("svc/config.yaml", 29, "high")]
    sources = ["python", "structure"]
    assert select_arbiter_targets(records, sources, contested_only=[1]) == [0, 1]


def test_whole_file_record_contests_every_line_in_that_file() -> None:
    """Issue #1103 case B: ``line: 0`` is a whole-file anchor, not line zero.

    A whole-file finding is about the entire file, so it co-locates with each
    line reported in that file. Grouping strictly by ``(file, line)`` filed it
    under its own key and the contested branch could never fire against the
    line-anchored twin.
    """
    records = [_rec("svc/loader.py", 88, "medium"), _rec("svc/loader.py", 0, "high")]
    sources = ["python", "structure"]
    assert select_arbiter_targets(records, sources, contested_only=[1]) == [0, 1]


def test_whole_file_record_does_not_reach_across_files() -> None:
    """The widening is per-file: a whole-file finding contests only its own file."""
    records = [_rec("svc/loader.py", 88, "medium"), _rec("svc/other.py", 0, "low")]
    sources = ["python", "structure"]
    assert select_arbiter_targets(records, sources, contested_only=[1]) == []


def test_two_whole_file_records_contest_each_other() -> None:
    """Two ``line: 0`` findings from different stacks still form one group.

    Routing whole-file records out of the ``(file, line)`` grouping must not lose
    the case where a file has nothing BUT whole-file findings.
    """
    records = [_rec("svc/loader.py", 0, "medium"), _rec("svc/loader.py", 0, "high")]
    sources = ["python", "structure"]
    assert select_arbiter_targets(records, sources, contested_only=[1]) == [0, 1]


def test_whole_file_record_agreeing_on_severity_is_not_contested() -> None:
    """The widening changes grouping only; divergent severity is still required."""
    records = [_rec("svc/loader.py", 88, "medium"), _rec("svc/loader.py", 0, "medium")]
    sources = ["python", "structure"]
    assert select_arbiter_targets(records, sources, contested_only=[1]) == []


def test_select_arbiter_targets_honors_contested_location_knob() -> None:
    """The profile's ``Arbitration.contested_location`` gates the contested
    branch of arbiter selection: disabling it leaves only the severity branch,
    so divergent-severity multi-stack collisions no longer route to the arbiter."""
    records = [
        _rec("api.py", 10, "high"),
        _rec("api.py", 10, "medium"),
    ]
    sources = ["python", "react"]
    # Default on: the medium finding is contested with the high one and is selected.
    assert select_arbiter_targets(records, sources) == [0, 1]
    # Knob off: only the severity branch selects; the contested medium drops out.
    assert select_arbiter_targets(records, sources, contested_location=False) == [0]
    # Severity branch still fully live when the contested branch is disabled:
    # lowering min_severity to medium pulls the medium finding back in.
    assert select_arbiter_targets(
        records, sources, min_severity="medium", contested_location=False
    ) == [0, 1]


# Issue #1111: "two or more DISTINCT stacks" needs one canonical spelling per
# stack. ``source`` has two -- the ``stack-<name>-records.json`` filename used by
# every path that loads records off disk, and a bare stack name used by the
# uncovered sweep's in-memory append -- so comparing raw ``source`` strings made
# one stack count as two and marked uncontested locations contested, routing
# low/medium findings to the expensive Opus arbiter that the cost split exists to
# keep away from it. ``_stack_name`` prefers the stack half of the record's
# ``uid`` and normalizes ``source`` only as the uid-less fallback.


def test_one_stack_spelled_two_ways_is_not_contested() -> None:
    """A single stack tagged both ways must not fake a cross-stack contest.

    Same location, divergent severity, neither record high: the ONLY thing that
    could select either is the contested branch, and it must not fire -- both
    records are ``python``, however the pipeline happened to tag them.
    """
    records = [
        _rec("api.py", 10, "medium", uid="python:1"),
        _rec("api.py", 10, "low", uid="python:2"),
    ]
    sources = ["stack-python-records.json", "python"]
    assert select_arbiter_targets(records, sources) == []


def test_genuine_cross_stack_contest_survives_normalization() -> None:
    """Normalizing the spelling must not also erase real contests.

    The mirror of the test above with the same two ``source`` spellings, but the
    records belong to different stacks -- the case the contested branch exists
    for. Collapsing every record onto one key would silently disable it.
    """
    records = [
        _rec("api.py", 10, "medium", uid="python:1"),
        _rec("api.py", 10, "low", uid="react:1"),
    ]
    sources = ["stack-python-records.json", "react"]
    assert select_arbiter_targets(records, sources) == [0, 1]


def test_uid_less_records_fall_back_to_normalized_source() -> None:
    """A record with no ``uid`` is grouped by its normalized ``source``.

    Records written by a run from before ``uid`` existed reach the predicate
    without one. ``stack_name_from_uid("")`` is ``""``, so without the ``source``
    fallback every such record would collapse onto one empty pseudo-stack and
    the contested branch would UNDER-count -- silently dropping real
    cross-stack contests out of arbitration. Both spellings still normalize, so
    the one-stack-two-ways case stays uncontested here too.
    """
    two_stacks = [_rec("api.py", 10, "medium"), _rec("api.py", 10, "low")]
    assert select_arbiter_targets(two_stacks, ["stack-python-records.json", "react"]) == [0, 1]
    one_stack = [_rec("api.py", 10, "medium"), _rec("api.py", 10, "low")]
    assert select_arbiter_targets(one_stack, ["stack-python-records.json", "python"]) == []


def test_uid_outranks_the_source_tag() -> None:
    """The ``uid`` wins when the two disagree, because it is minted at birth.

    ``source`` is re-derived from whichever file a record was loaded out of;
    ``uid`` is stamped once, when the record is created, and travels with the
    record through the structural partition, adjudication drops and record
    rewrites. Two records sharing a ``source`` tag but carrying uids from
    different stacks are two stacks, and the contest must fire.
    """
    records = [
        _rec("api.py", 10, "medium", uid="python:1"),
        _rec("api.py", 10, "low", uid="react:1"),
    ]
    sources = ["stack-python-records.json", "stack-python-records.json"]
    assert select_arbiter_targets(records, sources) == [0, 1]


def test_uncovered_sweep_record_is_its_own_stack() -> None:
    """The sweep's bare ``uncovered`` tag names a real, distinct stack.

    The uncovered-file sweep is the pipeline's second record-birth site and tags
    its records with a bare stack name rather than a records filename. Both
    spellings normalize to themselves, so a sweep finding colliding with a
    per-stack finding is a genuine two-stack contest and must be arbitrated.
    """
    records = [
        _rec("api.py", 10, "medium", uid="python:1"),
        _rec("api.py", 10, "low", uid="uncovered:1"),
    ]
    sources = ["stack-python-records.json", "uncovered"]
    assert select_arbiter_targets(records, sources) == [0, 1]
