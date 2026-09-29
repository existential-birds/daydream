"""The host-stamped adjudication provenance ledger (issue #735).

The ledger records, per canonical record uid, what the adjudication passes did
to it: whether it was targeted at all, whether a verdict bound to it, whether
it survived, and which revisable fields were materially rewritten. It is
written only by the pass that applies the verdicts, and read fail-open by the
verify-selection predicate.
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream.deep.adjudication_provenance import (
    PROVENANCE_FORMAT,
    RecordProvenance,
    find_revision_delta,
    load_provenance,
    record_provenance,
)
from daydream.deep.artifacts import adjudication_provenance_path
from daydream.deep.merge_steps import _apply_adjudication_verdicts


def test_a_revision_that_changes_a_revisable_field_is_recorded(tmp_path: Path) -> None:
    before = {
        "uid": "python:1",
        "severity": "high",
        "confidence": "HIGH",
        "description": "d",
        "rationale": "r",
        "evidence": "e",
    }
    after = {**before, "severity": "low", "description": "rewritten"}
    assert find_revision_delta(before, after) == ("severity", "description")
    assert find_revision_delta(before, dict(before)) == ()

    record_provenance(
        tmp_path,
        pass_name="arbiter",
        outcomes=[
            RecordProvenance("python:1", ("arbiter",), True, True, ("severity", "description")),
            RecordProvenance("python:2", ("arbiter",), False, True, ()),
        ],
    )
    recorded = load_provenance(tmp_path)
    assert recorded["python:1"].materially_revised is True
    assert recorded["python:1"].confirmed is True
    assert recorded["python:2"].verdict_bound is False and recorded["python:2"].confirmed is False
    assert recorded["python:2"].targeted is True
    assert json.loads(adjudication_provenance_path(tmp_path).read_text())["format"] == PROVENANCE_FORMAT


def test_apply_adjudication_verdicts_reports_confirmation_and_revision() -> None:
    """A bound keep-verdict confirms; an unbound or missing one does not."""
    records = [
        {
            "uid": "python:1",
            "severity": "high",
            "confidence": "HIGH",
            "description": "a",
            "rationale": "r",
            "evidence": "e",
        },
        {
            "uid": "python:2",
            "severity": "high",
            "confidence": "HIGH",
            "description": "b",
            "rationale": "r",
            "evidence": "e",
        },
        {
            "uid": "python:3",
            "severity": "high",
            "confidence": "HIGH",
            "description": "c",
            "rationale": "r",
            "evidence": "e",
        },
    ]
    verdicts = {
        1: {"arb_id": 1, "keep": True, "severity": "low"},
        2: {"arb_id": 99, "keep": True},  # echoed id mismatch -> unconfirmed
    }
    _kept, _sources, outcomes = _apply_adjudication_verdicts(
        records,
        ["stack-python-records.json"] * 3,
        [0, 1, 2],
        verdicts,
        pass_name="arbiter",
        id_field="arb_id",
        fail_closed=False,
    )

    by_uid = {o.uid: o for o in outcomes}
    assert by_uid["python:1"].verdict_bound is True
    assert by_uid["python:1"].revised_fields == ("severity",)
    assert by_uid["python:1"].confirmed is True
    assert by_uid["python:2"].verdict_bound is False and by_uid["python:2"].confirmed is False
    assert by_uid["python:3"].verdict_bound is False  # no verdict returned


def test_absent_or_malformed_ledger_loads_as_no_provenance(tmp_path: Path) -> None:
    assert load_provenance(tmp_path) == {}
    adjudication_provenance_path(tmp_path).write_text("{not json")
    assert load_provenance(tmp_path) == {}
