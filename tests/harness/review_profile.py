"""Shared default-profile strategy lookup for prompt tests."""

from __future__ import annotations

from dataclasses import replace

from daydream.review_profile import ResolvedProfile, build_default_profile


def default_strategy(stage: str) -> str:
    from daydream import review_profile as _rp

    return _rp.build_default_profile().strategies[stage].content


def independent_alternatives_profile() -> ResolvedProfile:
    """Explicitly request the separate design pass in integration fixtures."""
    profile = build_default_profile()
    strategies = dict(profile.strategies)
    strategies["alternatives"] = replace(
        strategies["alternatives"], content=strategies["alternatives"].content + "\nCustom independent design policy.",
    )
    return ResolvedProfile(profile=replace(profile, strategies=strategies), source_kind="test")
