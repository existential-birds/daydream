from typing import Any

from daydream.training.reply_classifier import is_qualifying_author


def _reply(body: str, login: str = "maintainer", assoc: str = "MEMBER", bot: str = "User") -> dict[str, Any]:
    return {"user": {"login": login, "type": bot}, "author_association": assoc, "body": body}


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
