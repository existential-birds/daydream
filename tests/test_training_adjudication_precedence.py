"""Precedence: explicit adjudicator > latest human rater > automatic; conflicts stay non-gold."""

from typing import Any

from daydream.training.adjudication.precedence import (
    effective_adjudication,
    has_rater_conflict,
    reopen_on_digest_change,
)

R1 = "b" * 64


def _obs(
    disposition: str,
    labeler: str,
    role: str = "rater",
    digest: str = "d" * 64,
    observed: str = "2026-08-30T10:00:00+00:00",
) -> dict[str, Any]:
    return {
        "observation_id": f"{labeler}:{role}:{observed}:{digest}:{disposition}",
        "record_id": R1,
        "disposition": disposition,
        "evidence_digest": digest,
        "labeler": labeler,
        "role": role,
        "evidence": [{"reply_id": "r1"}],
        "observed_at": observed,
    }


def test_latest_human_rater_wins_over_automatic() -> None:
    auto = _obs("ambiguous", "classifier-r1", role="automatic", observed="2026-08-30T09:00:00+00:00")
    human = _obs("accepted", "alice", observed="2026-08-30T11:00:00+00:00")
    assert effective_adjudication([auto, human])["disposition"] == "accepted"
    assert effective_adjudication([auto, human])["labeler"] == "alice"
    assert effective_adjudication([auto])["role"] == "automatic"


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
    assert result == effective_adjudication(list(reversed(obs)))


def test_conflict_resolved_by_adjudicator_is_gold_eligible_again() -> None:
    obs = [
        _obs("accepted", "alice", observed="2026-08-30T10:00:00+00:00"),
        _obs("rejected", "bob", observed="2026-08-30T11:00:00+00:00"),
        _obs("accepted", "chief", role="adjudicator", observed="2026-08-30T12:00:00+00:00"),
    ]
    result = effective_adjudication(obs)
    assert result["conflict"] is False and result["gold_eligible"] is True


def test_digest_change_requeues_prior_judgment() -> None:
    human = _obs("accepted", "alice", digest="d" * 64)
    assert reopen_on_digest_change(human, current_digest="d" * 64) is False
    assert reopen_on_digest_change(human, current_digest="e" * 64) is True


def test_precedence_orders_fractional_utc_timestamps_chronologically() -> None:
    earlier = _obs("rejected", "alice", observed="2026-08-30T10:00:00Z")
    later = _obs("accepted", "bob", observed="2026-08-30T10:00:00.500000+00:00")
    assert effective_adjudication([earlier, later])["disposition"] == "accepted"
    assert effective_adjudication([later, earlier])["disposition"] == "accepted"
