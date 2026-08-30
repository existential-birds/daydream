"""Precedence: explicit adjudicator > latest human rater > automatic; conflicts stay non-gold."""
from collections.abc import Mapping
from typing import Any

from daydream.training.adjudication.precedence import effective_adjudication, has_rater_conflict

R1 = "b" * 64


def _obs(
    disposition: str,
    labeler: str,
    role: str = "rater",
    digest: str = "d" * 64,
    observed: str = "2026-08-30T10:00:00+00:00",
) -> dict[str, Any]:
    return {
        "record_id": R1,
        "disposition": disposition,
        "evidence_digest": digest,
        "labeler": labeler,
        "role": role,
        "evidence": [{"reply_id": "r1"}],
        "observed_at": "2026-08-30T10:00:00+00:00" if observed == "auto" else observed,
    }


def test_latest_human_rater_wins_over_automatic() -> None:
    auto = _obs("ambiguous", "classifier-r1", observed="auto")
    human = _obs("accepted", "alice", observed="2026-08-30T11:00:00+00:00")
    assert effective_adjudication([auto, human])["disposition"] == "accepted"
    assert effective_adjudication([auto, human])["labeler"] == "alice"


def test_explicit_adjudicator_resolution_beats_later_rater() -> None:
    rater = _obs("rejected", "alice", observed="2026-08-30T11:00:00+00:00")
    adjudicator = _obs("accepted", "chief", role="adjudicator", observed="2026-08-30T10:30:00+00:00")
    assert effective_adjudication([rater, adjudicator])["labeler"] == "chief"


def test_conflicting_raters_without_adjudicator_are_non_gold() -> None:
    obs = [
        _obs("accepted", "alice", observed="2026-08-30T10:00:00+00:00"),
        _obs("rejected", "bob", observed="2026-08-30T11:00:00+00:00"),
    ]
    result = effective_adjudication(obs)
    assert has_rater_conflict(obs) is True
    assert result["gold_eligible"] is False  # AC 6: stays non-gold until adjudicated
    assert result["conflict"] is True


def test_conflict_resolved_by_adjudicator_is_gold_eligible_again() -> None:
    obs = [
        _obs("accepted", "alice", observed="2026-08-30T10:00:00+00:00"),
        _obs("rejected", "bob", observed="2026-08-30T11:00:00+00:00"),
        _obs("accepted", "chief", role="adjudicator", observed="2026-08-30T12:00:00+00:00"),
    ]
    result = effective_adjudication(obs)
    assert result["conflict"] is False and result["gold_eligible"] is True


def test_conflicting_raters_fixture_order_is_stable() -> None:
    # Determinism: the same observations in any input order resolve identically.
    obs_a: list[Mapping[str, Any]] = [
        _obs("accepted", "alice", observed="2026-08-30T10:00:00+00:00"),
        _obs("rejected", "bob", observed="2026-08-30T11:00:00+00:00"),
    ]
    assert effective_adjudication(obs_a) == effective_adjudication(list(reversed(obs_a)))


def test_digest_change_requeues_prior_judgment() -> None:
    from daydream.training.adjudication.precedence import reopen_on_digest_change
    human = _obs("accepted", "alice", digest="d" * 64)
    # Same digest: judgment stands.
    assert reopen_on_digest_change(human, current_digest="d" * 64) is False
    # Digest drifted: item must reopen, not silently reuse the judgment.
    assert reopen_on_digest_change(human, current_digest="e" * 64) is True


def test_model_suggested_queue_item_never_gold_eligible_unreviewed() -> None:
    from daydream.training.adjudication.observations import append_observation  # noqa: F401
    # Model suggested 'accepted' on a queue item: stored with review_required=True.
    obs = {"record_id": "c" * 64, "disposition": "accepted", "evidence_digest": "d" * 64,
           "labeler": "reply-classifier-980-r1", "role": "model-suggested",
           "rationale": "classifier says accept", "valid_at": "2026-08-30T10:00:00+00:00",
           "observed_at": "2026-08-30T10:00:00+00:00", "rubric_version": "984-adjudicate-r1",
           "evidence": [{"reply_id": "r1"}],
           "review_required": True}
    result = effective_adjudication([obs])
    assert result["gold_eligible"] is False  # review-required ⇒ not gold, whatever the disposition
    assert result["review_required"] is True
