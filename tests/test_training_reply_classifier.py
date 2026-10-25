from collections.abc import Sequence
from typing import Any

import pytest

from daydream.training.reply_classifier import (
    classify_reply,
    is_qualifying_author,
)


def _reply(body: str, login: str = "maintainer", assoc: str = "MEMBER", bot: str = "User") -> dict[str, Any]:
    return {"user": {"login": login, "type": bot}, "author_association": assoc, "body": body}


def test_classify_directional_rules(subtests: pytest.Subtests) -> None:
    cases: Sequence[tuple[str, str]] = [
        ("Fixed in abc123", "accepted"),  # M22: fixed-in-sha
        ("Good catch, merged the fix", "accepted"),  # explicit agree
        ("Not applicable — this path is unreachable", "rejected"),  # M22
        ("This is already handled upstream", "rejected"),  # M22
        ("False positive, the linter is wrong here", "rejected"),  # M22
        ("This behavior is intentional", "rejected"),  # M22
        ("I disagree with the finding; the docs say otherwise", "rejected"),  # factual disagreement
        ("Thanks for the review!", "ambiguous"),  # bare ack
        ("Could you elaborate on line 12?", "ambiguous"),  # question
        ("Deferring this to the next milestone", "ambiguous"),  # deferral
        ("Won't fix, but not because it's wrong", "ambiguous"),  # won't-fix w/o dispute
        ("not fixed yet, still reproduces", "ambiguous"),  # M22: negation fails closed
        ("", "ambiguous"),  # empty body
        ("Fixed in abc123. Though the other finding was a false positive.", "ambiguous"),  # mixed direction
        # Negating a rejection phrase stays ambiguous.
        ("This is not a false positive — it reproduces on main.", "ambiguous"),
        ("This is not intentional; it's a real bug.", "ambiguous"),
        ("This is not applicable to this PR.", "rejected"),  # affirm 'not applicable' is still a rejection
    ]
    for case_index, (body, expected) in enumerate(cases):
        with subtests.test(body=body, expected=expected):
            assert classify_reply(_reply(body)) == expected


def test_qualifying_author_gates_decisive_labels(subtests: pytest.Subtests) -> None:
    cases: Sequence[tuple[dict[str, Any], str]] = [
        (_reply("Fixed in abc123", bot="Bot", login="dependabot[bot]"), "ambiguous"),  # bot excluded
        (_reply("Fixed in abc123", login="", assoc="NONE"), "ambiguous"),  # empty author excluded
        (_reply("Fixed in abc123", login="daydream-agent"), "ambiguous"),  # daydream self-reply excluded
    ]
    case_ids_0 = ["reply1-ambiguous", "reply2-ambiguous", "reply3-ambiguous"]
    for case_index, (reply, expected) in enumerate(cases):
        with subtests.test(msg=case_ids_0[case_index], reply=reply, expected=expected):
            assert classify_reply(reply) == expected


def test_qualifying_author_rules() -> None:
    assert is_qualifying_author(_reply("x", assoc="OWNER"), pr_author_logins={"someone"}) is True
    assert is_qualifying_author(_reply("x", assoc="COLLABORATOR"), pr_author_logins=set()) is True
    assert is_qualifying_author(_reply("x", login="alice", assoc="NONE"), pr_author_logins={"alice"}) is True
    assert (
        is_qualifying_author(_reply("x", assoc="NONE"), pr_author_logins=set(), review_author_logins={"bob"}) is False
    )
    assert (is_qualifying_author(
            _reply("x", login="bob", assoc="NONE"), pr_author_logins=set(), review_author_logins={"bob"}
        )
        is True
    )
    assert is_qualifying_author(_reply("x", bot="Bot", login="ci[bot]"), pr_author_logins=set()) is False
