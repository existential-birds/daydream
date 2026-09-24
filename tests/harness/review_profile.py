"""Shared default-profile strategy lookup for prompt tests."""

from __future__ import annotations

from dataclasses import replace

from daydream.review_profile import ResolvedProfile, build_default_profile


def default_strategy(stage: str) -> str:
    return build_default_profile().strategies[stage].content


def exploration_strategies() -> dict[str, str]:
    """Return the default profile's exploration.* strategy content by key."""
    return {
        name: value.content
        for name, value in build_default_profile().strategies.items()
        if name.startswith("exploration.")
    }


def independent_alternatives_profile() -> ResolvedProfile:
    """Explicitly request the separate design pass in integration fixtures."""
    profile = build_default_profile()
    strategies = dict(profile.strategies)
    strategies["alternatives"] = replace(
        strategies["alternatives"], content=strategies["alternatives"].content + "\nCustom independent design policy.",
    )
    return ResolvedProfile(profile=replace(profile, strategies=strategies), source_kind="test")


def independent_exploration_profile(
    resolved: ResolvedProfile | None = None,
) -> ResolvedProfile:
    """Request model exploration when testing specialist dispatch and lifecycle."""
    profile = resolved.profile if resolved is not None else build_default_profile()
    strategies = dict(profile.strategies)
    key = "exploration.dependency_trace"
    strategies[key] = replace(
        strategies[key], content=strategies[key].content + "\nCustom dependency exploration policy.",
    )
    return ResolvedProfile(profile=replace(profile, strategies=strategies), source_kind="test")
