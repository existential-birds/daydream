"""Pure latency-profile decisions for deep review.

Profiles set monotone effort floors: risk may raise a route, never lower it.
Unknown profiles fail safe to the highest (forensic) route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LatencyProfile = Literal["fast", "balanced", "forensic"]
WonderRoute = Literal["skip", "medium", "high"]
ArbiterEffort = Literal["medium", "high", "xhigh"]

LATENCY_PROFILES: tuple[LatencyProfile, ...] = ("fast", "balanced", "forensic")
DEFAULT_LATENCY_PROFILE: LatencyProfile = "balanced"
FAIL_SAFE_LATENCY_PROFILE: LatencyProfile = "forensic"
WONDER_ROUTES: tuple[WonderRoute, ...] = ("skip", "medium", "high")
ARBITER_EFFORTS: tuple[ArbiterEffort, ...] = ("medium", "high", "xhigh")


@dataclass(frozen=True)
class LatencyRoute:
    profile: LatencyProfile
    wonder: WonderRoute
    arbiter_effort: ArbiterEffort
    arbiter_sharded: bool
    group_max_targets: int
    legacy_trivial_tier_gate: bool


PROFILE_ROUTES: dict[LatencyProfile, LatencyRoute] = {
    "fast": LatencyRoute("fast", "skip", "medium", True, 4, False),
    "balanced": LatencyRoute("balanced", "medium", "high", True, 3, False),
    "forensic": LatencyRoute("forensic", "high", "xhigh", False, 0, True),
}


@dataclass(frozen=True)
class ProfileResolution:
    profile: LatencyProfile
    requested: str | None
    source: str
    fail_safe: bool
    reason: str | None


def resolve_latency_profile(requested: str | None, *, source: str) -> ProfileResolution:
    """Resolve a profile without raising; reject unknown names upward to forensic."""
    if requested is None:
        return ProfileResolution(DEFAULT_LATENCY_PROFILE, None, source, False, None)
    if requested in PROFILE_ROUTES:
        return ProfileResolution(PROFILE_ROUTES[requested].profile, requested, source, False, None)
    return ProfileResolution(
        FAIL_SAFE_LATENCY_PROFILE, requested, source, True, f"Unknown latency profile: {requested!r}"
    )
