"""Pure latency-profile decision tests (issue #732).

The decision is a value: profiles, risk floors, group planning. Nothing here
touches the filesystem, the clock, or the environment -- see MH3.
"""
from __future__ import annotations

from daydream.deep.latency import (
    DEFAULT_LATENCY_PROFILE,
    FAIL_SAFE_LATENCY_PROFILE,
    LATENCY_PROFILES,
    PROFILE_ROUTES,
    WONDER_ROUTES,
    resolve_latency_profile,
)


def test_every_profile_names_a_route_and_the_ladder_is_monotone() -> None:
    assert LATENCY_PROFILES == ("fast", "balanced", "forensic")
    assert set(PROFILE_ROUTES) == set(LATENCY_PROFILES)
    assert all(PROFILE_ROUTES[name].profile == name for name in LATENCY_PROFILES)
    assert (
        (PROFILE_ROUTES["fast"].group_max_targets, PROFILE_ROUTES["fast"].legacy_trivial_tier_gate),
        (PROFILE_ROUTES["balanced"].group_max_targets, PROFILE_ROUTES["balanced"].legacy_trivial_tier_gate),
        (PROFILE_ROUTES["forensic"].group_max_targets, PROFILE_ROUTES["forensic"].legacy_trivial_tier_gate),
    ) == ((4, False), (3, False), (0, True))
    ranks = {
        name: (
            WONDER_ROUTES.index(PROFILE_ROUTES[name].wonder),
            ("medium", "high", "xhigh").index(PROFILE_ROUTES[name].arbiter_effort),
        )
        for name in LATENCY_PROFILES
    }
    assert ranks["fast"] <= ranks["balanced"] <= ranks["forensic"]


def test_forensic_is_todays_route_and_balanced_is_the_default() -> None:
    forensic = PROFILE_ROUTES["forensic"]
    assert (forensic.wonder, forensic.arbiter_effort) == ("high", "xhigh")
    assert forensic.arbiter_sharded is False
    assert PROFILE_ROUTES["balanced"].wonder != "skip"
    assert PROFILE_ROUTES["balanced"].arbiter_sharded is True
    assert DEFAULT_LATENCY_PROFILE == "balanced"
    assert FAIL_SAFE_LATENCY_PROFILE == "forensic"


def test_recognized_profile_preserves_request_and_source() -> None:
    resolved = resolve_latency_profile("fast", source="config")
    assert (resolved.profile, resolved.requested, resolved.source) == ("fast", "fast", "config")
    assert (resolved.fail_safe, resolved.reason) == (False, None)


def test_unknown_profile_fails_safe_upward_and_says_so() -> None:
    resolved = resolve_latency_profile("turbo", source="cli")
    assert resolved.profile == "forensic"
    assert resolved.fail_safe is True
    assert "turbo" in (resolved.reason or "")


def test_absent_profile_uses_the_default_without_a_fail_safe_record() -> None:
    resolved = resolve_latency_profile(None, source="default")
    assert (resolved.profile, resolved.fail_safe, resolved.source) == ("balanced", False, "default")
