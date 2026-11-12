from typing import Any

from daydream.training.reply_classifier import classify_reply, qualification_reason


def _reply(body: str, login: str = "maintainer", assoc: str = "MEMBER", bot: str = "User") -> dict[str, Any]:
    return {"user": {"login": login, "type": bot}, "author_association": assoc, "body": body}


def test_qualifying_author_rules() -> None:
    assert qualification_reason(_reply("x", assoc="OWNER"), {"someone"}) == "assoc:OWNER"
    assert qualification_reason(_reply("x", assoc="COLLABORATOR"), set()) == "assoc:COLLABORATOR"
    assert qualification_reason(_reply("x", login="alice", assoc="NONE"), {"alice"}) == "pr_author"
    assert qualification_reason(_reply("x", assoc="NONE"), set(), {"bob"}) == "excluded:non-qualifying"
    assert qualification_reason(_reply("x", login="bob", assoc="NONE"), set(), {"bob"}) == "review_author"
    assert qualification_reason(_reply("x", bot="Bot", login="ci[bot]"), set()) == "excluded:bot"
    # Blank lines between a decisive reply's lines carry no direction of their own.
    assert classify_reply(_reply("Good catch.\n\n   \nApplied in the next commit.")) == "accepted"
