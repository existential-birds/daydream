"""Tests for labeler version constants and reply evidence digest."""

from daydream.training import labeler_versions as lv
from daydream.training import reward


def test_versions_are_independent() -> None:
    """The four version axes exist and never alias reward.REWARD_VERSION."""
    assert lv.RUBRIC_SCHEMA_VERSION != lv.LABELER_POLICY_VERSION
    assert lv.LABELER_POLICY_VERSION != lv.REPLY_CLASSIFIER_VERSION
    assert lv.REPLY_EVIDENCE_DIGEST_FORMAT == "sha256/1"
    assert lv.LABELER_POLICY_VERSION != reward.REWARD_VERSION
    assert lv.RUBRIC_SCHEMA_VERSION != reward.REWARD_VERSION


def test_evidence_digest_is_deterministic() -> None:
    """reply_evidence_digest is a stable sha256 over the canonical evidence JSON."""
    replies = [{"id": 1, "body": "fixed in abc"}, {"id": 2, "body": "fixed too"}]
    a = lv.reply_evidence_digest(replies)
    b = lv.reply_evidence_digest(list(reversed(replies)))
    c = lv.reply_evidence_digest([{"id": 1, "body": "not fixed"}])
    assert a == b and a != c
    assert len(a) == 64
