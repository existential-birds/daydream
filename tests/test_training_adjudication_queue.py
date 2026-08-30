"""Adjudication queue build: deterministic ordering over projector adjudication entries."""
import pytest

from daydream.training.adjudication.queue import build_queue
from daydream.training.corpus_v2.identity import record_id


def _session(
    sid: str, fingerprint: str, disposition: str, digest: str
) -> dict[str, object]:
    return {
        "session_id": sid, "trajectory_id": f"{sid}-traj", "segment_id": f"{sid}-seg",
        "resolutions": [{
            "fingerprint": fingerprint, "disposition": disposition,
            "evidence": [{"reply_id": "r1", "body_sha256": "abc"}],
            "evidence_digest": digest, "profile": "pr_review", "stack": "python",
        }],
    }

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
    assert items_a[0]["record_id"] == record_id("s1", "s1-traj", "s1-seg", "fp-b")  # sorted by record_id
    assert all(i["disposition"] in {"ambiguous", "unanswered", "missing"} for i in items_a)
    assert all(i["record_id"] != record_id("s1", "s1-traj", "s1-seg", "fp-a") for i in items_a)


def test_digest_drift_reopens_item_and_missing_digest_fails_closed() -> None:
    record_id_val = record_id("s1", "s1-traj", "s1-seg", "fp-a")
    prior = {
        "record_id": record_id_val,
        "role": "rater",
        "disposition": "accepted",
        "evidence_digest": "d" * 64,
        "labeler": "alice",
        "observed_at": "2026-08-30T10:00:00+00:00",
        "review_required": True,  # e.g. a stored model-suggested label
    }
    # Same digest: judgment stands, item stays open with no prior disposition.
    fresh = build_queue([_session("s1", "fp-a", "ambiguous", "d" * 64)], prior_observations={record_id_val: prior})
    assert fresh[0]["status"] == "open" and fresh[0]["prior_disposition"] is None
    assert fresh[0]["review_required"] is True  # stored flag propagates to the queue item
    # Digest drift: item reopens, prior disposition carried as provenance.
    drifted = build_queue([_session("s1", "fp-a", "ambiguous", "e" * 64)], prior_observations={record_id_val: prior})
    assert drifted[0]["status"] == "reopened" and drifted[0]["prior_disposition"] == "accepted"
    assert drifted[0]["review_required"] is True
    # Automatic (non-human) prior never reopens.
    auto = dict(prior, role="automatic")
    auto_item = build_queue([_session("s1", "fp-a", "ambiguous", "e" * 64)], prior_observations={record_id_val: auto})
    assert auto_item[0]["status"] == "open"
    # A fresh evidence entry without evidence_digest fails closed, naming the fingerprint.
    session = _session("s1", "fp-a", "ambiguous", "e" * 64)
    del session["resolutions"][0]["evidence_digest"]  # type: ignore[index]
    with pytest.raises(ValueError, match="fp-a"):
        build_queue([session])


def test_decisive_adjudication_entry_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.training.adjudication import queue as queue_module
    from daydream.training.corpus_v2.projector import project_findings

    session = _session("s1", "fp-a", "ambiguous", "d1")

    def _forge_decisive(
        s: dict[str, object], **_kw: object
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        # The real projector routes only task-only dispositions to the
        # adjudication set; forge a decisive entry to prove the queue
        # builder's own fail-closed disposition guard holds.
        _, adjudication = project_findings(s, return_adjudication=True)
        return [], [dict(e, disposition="accepted") for e in adjudication]

    monkeypatch.setattr(queue_module, "project_findings", _forge_decisive)
    with pytest.raises(ValueError, match="fp-a"):
        build_queue([session])
