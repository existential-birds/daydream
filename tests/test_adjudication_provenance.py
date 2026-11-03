"""The host-stamped adjudication provenance ledger (issue #735).

The ledger records, per canonical record uid, what the adjudication passes did
to it: whether it was targeted at all, whether a verdict bound to it, whether
it survived, and which revisable fields were materially rewritten. It is
written only by the pass that applies the verdicts, and read fail-open by the
verify-selection predicate.
"""

from __future__ import annotations

from typing import Any

from daydream.deep.adjudication_steps import _apply_adjudication_verdicts


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
