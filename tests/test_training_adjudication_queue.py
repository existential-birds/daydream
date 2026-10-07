"""Adjudication queue build: deterministic ordering over projector adjudication entries."""

from pathlib import Path

import pytest

from daydream.training.adjudication import queue as queue_module
from daydream.training.adjudication.queue import build_queue
from daydream.training.corpus_projection.projector import project_findings
from daydream.training.record_identity import record_finding_id


def _session(sid: str, fingerprint: str, disposition: str, digest: str) -> dict[str, object]:
    return {
        "session_id": sid,
        "trajectory_id": f"{sid}-traj",
        "segment_id": f"{sid}-seg",
        "resolutions": [
            {
                "fingerprint": fingerprint,
                "item_uid": fingerprint,
                "disposition": disposition,
                "evidence": [{"reply_id": "r1", "body_sha256": "abc"}],
                "evidence_digest": digest,
                "evidence_digest_scheme": "canonical-json-v1",
                "profile": "pr_review",
                "stack": "python",
            }
        ],
    }


def _sessions_with_accepted_and_unanswered() -> list[dict[str, object]]:  # existing-shape helper
    return [_session("s1", "fp-a", "accepted", "d-gold"), _session("s2", "fp-b", "unanswered", "d2")]


def test_build_queue_include_decisive_returns_complete_set(tmp_path: Path) -> None:
    sessions = [*_sessions_with_accepted_and_unanswered(),
                _session("s3", "fp-c", "rejected", "d-rejected"),
                _session("s4", "fp-d", "ambiguous", "d-ambiguous")]
    open_only = build_queue(sessions)
    assert sorted(str(i["disposition"]) for i in open_only) == ["ambiguous", "unanswered"]

    complete = build_queue(sessions, include_decisive=True)
    assert sorted(str(i["disposition"]) for i in complete) == ["accepted", "ambiguous", "rejected", "unanswered"]
    assert {str(i["status"]) for i in complete} <= {"open", "reopened"}


def test_queue_is_deterministic_and_covers_all_non_decisive_states() -> None:
    # NOTE: 'accepted' disposition with evidence would be gold — the queue must EXCLUDE it.
    sessions = [
        _session("s2", "fp-b", "unanswered", "d2"),
        _session("s1", "fp-b", "ambiguous", "d1"),
        _session("s1", "fp-a", "accepted", "d-gold"),
        _session("s0", "fp-m", "missing", "d3"),
    ]
    items_a = build_queue(sessions)
    items_b = build_queue(list(reversed(sessions)))
    assert [i["record_id"] for i in items_a] == [i["record_id"] for i in items_b]
    assert [i["record_id"] for i in items_a] == sorted(str(i["record_id"]) for i in items_a)
    assert all(i["disposition"] in {"ambiguous", "unanswered", "missing"} for i in items_a)
    assert all(i["record_id"] != record_finding_id("s1", "s1-traj", "s1-seg", "fp-a") for i in items_a)


def test_digest_drift_reopens_item_and_missing_digest_fails_closed() -> None:
    record_id_val = record_finding_id("s1", "s1-traj", "s1-seg", "fp-a")
    prior = {
        "record_id": record_id_val,
        "role": "rater",
        "disposition": "accepted",
        "evidence_digest": "d" * 64,
        "labeler": "alice",
        "observed_at": "2026-08-30T10:00:00+00:00",
        "review_required": True,  # e.g. a stored model-suggested label
    }
    fresh = build_queue([_session("s1", "fp-a", "ambiguous", "d" * 64)], prior_observations={record_id_val: prior})
    assert fresh[0]["status"] == "open" and fresh[0]["prior_disposition"] is None
    assert fresh[0]["review_required"] is True  # stored flag propagates to the queue item
    drifted = build_queue([_session("s1", "fp-a", "ambiguous", "e" * 64)], prior_observations={record_id_val: prior})
    assert drifted[0]["status"] == "reopened" and drifted[0]["prior_disposition"] == "accepted"
    assert drifted[0]["review_required"] is True
    auto = dict(prior, role="automatic")
    auto_item = build_queue([_session("s1", "fp-a", "ambiguous", "e" * 64)], prior_observations={record_id_val: auto})
    assert auto_item[0]["status"] == "open"
    session = _session("s1", "fp-a", "ambiguous", "e" * 64)
    del session["resolutions"][0]["evidence_digest"]  # type: ignore[index]
    with pytest.raises(ValueError, match="fp-a"):
        build_queue([session])


def test_decisive_adjudication_entry_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session("s1", "fp-a", "ambiguous", "d1")

    def _forge_decisive(s: dict[str, object], **_kw: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        # Forge a decisive record outside the projector to prove the queue guard is independent.
        _, adjudication = project_findings(s, return_adjudication=True)
        return [], [dict(e, disposition="accepted") for e in adjudication]

    monkeypatch.setattr(queue_module, "project_findings", _forge_decisive)
    with pytest.raises(ValueError, match="fp-a"):
        build_queue([session])
