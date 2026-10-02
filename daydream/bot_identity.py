"""Dependency-free GitHub bot-login comparison shared by trust checks."""

from __future__ import annotations

__all__ = ["bot_login_matches", "bot_stem"]


def bot_login_matches(login: str | None, bot: str) -> bool:
    """Compare lowercased stems across the REST/GraphQL [bot] suffix difference."""
    return bot_stem(login) == bot_stem(bot)


def bot_stem(login: str | None) -> str:
    """Return a login's comparison stem: ``[bot]`` suffix dropped, lowercased."""
    return (login or "").removesuffix("[bot]").lower()
