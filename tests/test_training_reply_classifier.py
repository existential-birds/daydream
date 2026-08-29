from typing import Any

import pytest

from daydream.training.reply_classifier import (
    REPLY_CLASSIFIER_VERSION,
    classify_replies,
    classify_reply,
    is_qualifying_author,
)


def _reply(body: str, login: str = "maintainer", assoc: str = "MEMBER", bot: str = "User") -> dict[str, Any]:
    return {"user": {"login": login, "type": bot}, "author_association": assoc, "body": body}


@pytest.mark.parametrize(
    "body,expected",
    [
        ("Fixed in abc123", "accepted"),                    # M22: fixed-in-sha
        ("Good catch, merged the fix", "accepted"),         # explicit agree
        ("Not applicable — this path is unreachable", "rejected"),   # M22
        ("This is already handled upstream", "rejected"),   # M22
        ("False positive, the linter is wrong here", "rejected"),    # M22
        ("This behavior is intentional", "rejected"),       # M22
        ("I disagree with the finding; the docs say otherwise", "rejected"),  # factual disagreement
        ("Thanks for the review!", "ambiguous"),            # bare ack
        ("Could you elaborate on line 12?", "ambiguous"),   # question
        ("Deferring this to the next milestone", "ambiguous"),       # deferral
        ("Won't fix, but not because it's wrong", "ambiguous"),      # won't-fix w/o dispute
        ("not fixed yet, still reproduces", "ambiguous"),   # M22: negation fails closed
        ("", "ambiguous"),                                  # empty body
        ("Fixed in abc123. Though the other finding was a false positive.", "ambiguous"),  # mixed direction
    ],
)
def test_classify_directional_rules(body: str, expected: str) -> None:
    assert classify_reply(_reply(body)) == expected


@pytest.mark.parametrize(
    "reply,expected",
    [
        (_reply("Fixed in abc123", assoc="NONE"), "accepted"),      # PR-author path covered below
        (_reply("Fixed in abc123", bot="Bot", login="dependabot[bot]"), "ambiguous"),  # bot excluded
        (_reply("Fixed in abc123", login="", assoc="NONE"), "ambiguous"),  # empty author excluded
        (_reply("Fixed in abc123", login="daydream-agent"), "ambiguous"),  # daydream self-reply excluded
    ],
)
def test_qualifying_author_gates_decisive_labels(reply: dict[str, Any], expected: str) -> None:
    assert classify_reply(reply) == expected


def test_qualifying_author_rules() -> None:
    """PR author, OWNER/MEMBER/COLLABORATOR, or formal-review author qualify (M6)."""
    assert is_qualifying_author(_reply("x", assoc="OWNER"), pr_author_logins={"someone"}) is True
    assert is_qualifying_author(_reply("x", assoc="COLLABORATOR"), pr_author_logins=set()) is True
    assert is_qualifying_author(_reply("x", login="alice", assoc="NONE"), pr_author_logins={"alice"}) is True
    # formal review author: passed in via review_author_logins
    assert (
        is_qualifying_author(_reply("x", assoc="NONE"), pr_author_logins=set(), review_author_logins={"bob"}) is False
    )
    assert (
        is_qualifying_author(
            _reply("x", login="bob", assoc="NONE"), pr_author_logins=set(), review_author_logins={"bob"}
        )
        is True
    )
    assert is_qualifying_author(_reply("x", bot="Bot", login="ci[bot]"), pr_author_logins=set()) is False


def test_conflicting_replies_fail_closed_to_ambiguous() -> None:
    """One accept + one reject from qualifying humans ⇒ ambiguous (M22)."""
    replies = [
        {"id": 1, **_reply("Fixed in abc123")},
        {"id": 2, **_reply("False positive", login="other", assoc="NONE")},
    ]
    assert classify_replies(replies) == "ambiguous"


def test_bot_only_replies_yield_no_decisive_label() -> None:
    """Bot-only replies never produce accepted/rejected (M22)."""
    replies = [{"id": 1, **_reply("Fixed in abc123", bot="Bot", login="app[bot]"), "is_self_reply": False}]
    assert classify_replies(replies) == "ambiguous"


def test_self_replies_yield_no_decisive_label() -> None:
    """The daydream account replying to its own finding ⇒ ambiguous (M5/M22)."""
    replies = [{"id": 1, **_reply("Fixed in abc123", login="daydream-agent"), "is_self_reply": True}]
    assert classify_replies(replies) == "ambiguous"


def test_classifier_version_is_exported() -> None:
    """The rule version ties classifications to re-fetch identity (M13/M14)."""
    assert REPLY_CLASSIFIER_VERSION
