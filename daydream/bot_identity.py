"""Single source of truth for the ``[bot]``-tolerant login comparator (issue #254).

``bot_login_matches`` / ``bot_stem`` compare GitHub login strings tolerant of
the REST/GraphQL ``[bot]`` suffix mismatch. They live here — a zero-dependency
leaf mirroring the ``daydream/prompts/<name>.py`` one-constant-per-module
precedent — so ``daydream.reconcile`` (the author-filter fix for forged
``daydream-finding`` markers) can consume the comparator without ``reconcile``
pulling benchmark deps or risking an import cycle.

Exports:
    bot_login_matches: ``[bot]``-suffix-tolerant login comparison.
    bot_stem: login's comparison stem — ``[bot]`` suffix dropped, lowercased.
"""

from __future__ import annotations

__all__ = ["bot_login_matches", "bot_stem"]


def bot_login_matches(login: str | None, bot: str) -> bool:
    """Match a bot login tolerant of GitHub's REST/GraphQL ``[bot]`` mismatch.

    REST ``user.login`` keeps the ``[bot]`` suffix (``coderabbitai[bot]``);
    GraphQL ``author.login`` drops it (``coderabbitai``). Compare on the
    stripped, lowercased stem so both forms match one ``--bot`` value.
    """
    return bot_stem(login) == bot_stem(bot)


def bot_stem(login: str | None) -> str:
    """Return a login's comparison stem: ``[bot]`` suffix dropped, lowercased."""
    return (login or "").removesuffix("[bot]").lower()
