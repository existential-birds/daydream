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
from typing import Any

from daydream.deep.adjudication_provenance import (
    PROVENANCE_FORMAT,
    RecordProvenance,
    find_revision_delta,
    load_provenance,
    record_provenance,
)
from daydream.deep.adjudication_steps import _apply_adjudication_verdicts
from daydream.deep.artifacts import DeepArtifact


def test_a_revision_that_changes_a_revisable_field_is_recorded(tmp_path: Path) -> None:
    before = {"uid": "python:1", "severity": "high", "confidence": "HIGH", "description": "d", "rationale": "r",
        "evidence": "e",
    }
    after = {**before, "severity": "low", "description": "rewritten"}
    assert find_revision_delta(before, after) == ("severity", "description")
    assert find_revision_delta(before, dict(before)) == ()

    record_provenance(tmp_path, pass_name="arbiter",
        outcomes=[RecordProvenance("python:1", ("arbiter",), True, True, ("severity", "description")),
            RecordProvenance("python:2", ("arbiter",), False, True, ()),
        ],
    )
    recorded = load_provenance(tmp_path)
    assert recorded["python:1"].materially_revised is True
    assert recorded["python:1"].confirmed is True
    assert recorded["python:2"].verdict_bound is False and recorded["python:2"].confirmed is False
    assert recorded["python:2"].targeted is True
    assert json.loads(DeepArtifact.ADJUDICATION_PROVENANCE.at(tmp_path).read_text())["format"] == PROVENANCE_FORMAT

def test_apply_adjudication_verdicts_reports_confirmation_and_revision() -> None:
    """A bound keep-verdict confirms; an unbound or missing one does not."""
    records = [{"uid": "python:1", "severity": "high", "confidence": "HIGH", "description": "a", "rationale": "r",
            "evidence": "e",
        }, {"uid": "python:2", "severity": "high", "confidence": "HIGH", "description": "b", "rationale": "r",
            "evidence": "e",
        }, {"uid": "python:3", "severity": "high", "confidence": "HIGH", "description": "c", "rationale": "r",
            "evidence": "e",
        },
    ]
    verdicts: dict[int, dict[str, Any]] = {1: {"arb_id": 1, "keep": True, "severity": "low"},
        2: {"arb_id": 99, "keep": True},  # echoed id mismatch -> unconfirmed
    }
    _kept, outcomes = _apply_adjudication_verdicts(
        records, [0, 1, 2], verdicts, pass_name="arbiter", id_field="arb_id",
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
    DeepArtifact.ADJUDICATION_PROVENANCE.at(tmp_path).write_text("{not json")
    assert load_provenance(tmp_path) == {}

def test_a_body_missing_kept_parses_as_unkept_not_confirmed() -> None:
    """A missing ``kept`` is an absent input and must not read as confirmation.

    The writer always emits an explicit ``kept``, so a body that omits it is malformed or foreign: defaulting it
    to True would let a partially readable ledger certify a record as confirmed and silently skip its
    verification, against the ledger's documented fail-open read path."""
    parsed = RecordProvenance.from_dict({"uid": "python:1", "passes": ["arbiter"], "verdict_bound": True})
    assert parsed is not None
    assert parsed.kept is False
    assert parsed.confirmed is False

def test_two_passes_accumulate_additively_in_one_ledger(tmp_path: Path) -> None:
    """The additive merge path the real run relies on (arbiter, then suppression).

    The second ``record_provenance`` call must fold into the first record: ``passes`` union in first-seen order,
    ``verdict_bound`` OR, ``kept`` AND, and ``revised_fields`` union in declaration order."""
    record_provenance(
        tmp_path, pass_name="arbiter", outcomes=[RecordProvenance("python:1", ("arbiter",), True, True, ("severity",))],
    )
    record_provenance(tmp_path, pass_name="suppression",
        outcomes=[RecordProvenance("python:1", ("suppression",), False, False, ("evidence",))],
    )
    merged = load_provenance(tmp_path)["python:1"]
    assert merged.passes == ("arbiter", "suppression")
    assert merged.verdict_bound is True
    assert merged.kept is False
    assert merged.revised_fields == ("severity", "evidence")
