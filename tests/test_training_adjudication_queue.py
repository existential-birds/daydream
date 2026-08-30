"""Adjudication queue build: deterministic ordering over projector adjudication entries."""
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
    ]
    items_a = build_queue(sessions)
    items_b = build_queue(list(reversed(sessions)))
    assert [i["record_id"] for i in items_a] == [i["record_id"] for i in items_b]
    assert items_a[0]["record_id"] == record_id("s1", "s1-traj", "s1-seg", "fp-b")  # sorted by record_id
    assert all(i["disposition"] in {"ambiguous", "unanswered", "missing"} for i in items_a)
    assert all(i["record_id"] != record_id("s1", "s1-traj", "s1-seg", "fp-a") for i in items_a)
