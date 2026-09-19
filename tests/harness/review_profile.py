"""Shared default-profile strategy lookup for prompt tests."""

from __future__ import annotations


def default_strategy(stage: str) -> str:
    from daydream import review_profile as _rp

    return _rp.build_default_profile().strategies[stage].content
