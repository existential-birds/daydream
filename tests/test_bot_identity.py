"""One source of truth for the [bot]-tolerant login comparator (issue #254)."""

from daydream.bot_identity import bot_login_matches, bot_stem


def test_bot_stem_strips_bot_suffix_and_lowercases() -> None:
    assert bot_stem("daydream[bot]") == "daydream"
    assert bot_stem("Daydream") == "daydream"          # GraphQL drops [bot]
    assert bot_stem(None) == ""


def test_bot_login_matches_across_rest_and_graphql_forms() -> None:
    # --bot value is the bare slug (DAYDREAM_BOT_HANDLE shape).
    assert bot_login_matches("daydream[bot]", "daydream") is True   # REST form
    assert bot_login_matches("daydream", "daydream") is True        # GraphQL form
    assert bot_login_matches("evil-attacker", "daydream") is False
    assert bot_login_matches(None, "daydream") is False
