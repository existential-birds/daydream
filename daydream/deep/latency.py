"""Pure latency-profile decisions for deep review.

Profiles set monotone effort floors: risk may raise a route, never lower it.
Unknown profiles fail safe to the highest (forensic) route.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

LatencyProfile = Literal["fast", "balanced", "forensic"]
WonderRoute = Literal["skip", "medium", "high"]
ArbiterEffort = Literal["medium", "high", "xhigh"]

LATENCY_PROFILES: tuple[LatencyProfile, ...] = ("fast", "balanced", "forensic")
DEFAULT_LATENCY_PROFILE: LatencyProfile = "balanced"
FAIL_SAFE_LATENCY_PROFILE: LatencyProfile = "forensic"
WONDER_ROUTES: tuple[WonderRoute, ...] = ("skip", "medium", "high")
ARBITER_EFFORTS: tuple[ArbiterEffort, ...] = ("medium", "high", "xhigh")

# Calibration surfaces: match only changed content and post-state paths, never context.
_SECURITY_TRIGGERS = (
    "authenticate", "authorization", "authz", "password", "secret", "credential", "token", "permission",
)
_CONCURRENCY_TRIGGERS = (
    "threading.lock", "asyncio.lock", "mutex", "semaphore", "deadlock", "atomic", "race condition",
)
_PERSISTENCE_TRIGGERS = ("alter table", "create table", "drop table", "begin;", "commit;", "rollback;", "transaction")
_INTERFACE_TRIGGERS = ("@app.route", "@router.", "openapi", "proto3", "endpoint", "api/v", "grpc")
_MIGRATION_TRIGGERS = ("migrations/", "migration", "alembic", "schema_version")


@dataclass(frozen=True)
class DiffSignals:
    diff_lines: int
    diff_bytes: int
    changed_files: int
    stack_count: int
    security_surface: bool
    concurrency_surface: bool
    persistence_surface: bool
    interface_surface: bool
    migration_surface: bool


def diff_signals(*, diff: str, changed_files: int, stack_count: int) -> DiffSignals:
    """Extract surface signals only from changed lines and post-state file paths."""
    changed = "\n".join(
        line for line in diff.lower().splitlines()
        if line.startswith("+++ b/") or (line.startswith(("+", "-")) and not line.startswith(("+++ ", "--- ")))
    )
    return DiffSignals(
        diff_lines=diff.count("\n") + bool(diff and not diff.endswith("\n")),
        diff_bytes=len(diff.encode("utf-8")),
        changed_files=changed_files,
        stack_count=stack_count,
        security_surface=any(trigger in changed for trigger in _SECURITY_TRIGGERS),
        concurrency_surface=any(trigger in changed for trigger in _CONCURRENCY_TRIGGERS),
        persistence_surface=any(trigger in changed for trigger in _PERSISTENCE_TRIGGERS),
        interface_surface=any(trigger in changed for trigger in _INTERFACE_TRIGGERS),
        migration_surface=any(trigger in changed for trigger in _MIGRATION_TRIGGERS),
    )


@dataclass(frozen=True)
class FindingSignals:
    high_severity: bool
    contested: bool


@dataclass(frozen=True)
class RiskSummary:
    size_score: int
    breadth_score: int
    floors: tuple[str, ...]
    wonder_floor: WonderRoute
    arbiter_floor: ArbiterEffort


def summarize_risk(signals: DiffSignals, findings: FindingSignals | None = None) -> RiskSummary:
    """Record scale for audit, but escalate only for surfaces or findings."""
    if signals.diff_lines > 2000 or signals.diff_bytes > 65536:
        size_score = 2
    elif signals.diff_lines > 200 or signals.diff_bytes > 8192:
        size_score = 1
    else:
        size_score = 0
    floors = [
        name
        for name in (
            "security_surface", "concurrency_surface", "persistence_surface", "interface_surface", "migration_surface"
        )
        if getattr(signals, name)
    ]
    if findings is not None:
        if findings.high_severity:
            floors.append("high_severity_findings")
        if findings.contested:
            floors.append("contested_findings")
    return RiskSummary(
        size_score=size_score,
        breadth_score=min(1, signals.stack_count // 2),
        floors=tuple(sorted(set(floors))),
        wonder_floor="high" if floors else "skip",
        arbiter_floor="high" if floors else "medium",
    )


def _ladder_max[T](ladder: tuple[T, ...], left: T, right: T) -> T:
    return ladder[max(ladder.index(left), ladder.index(right))]


def route_for(profile: LatencyProfile, summary: RiskSummary) -> LatencyRoute:
    """Raise a profile's route to mandatory risk floors without changing other choices."""
    base = PROFILE_ROUTES[profile]
    return replace(
        base,
        wonder=_ladder_max(WONDER_ROUTES, base.wonder, summary.wonder_floor),
        arbiter_effort=_ladder_max(ARBITER_EFFORTS, base.arbiter_effort, summary.arbiter_floor),
    )


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
class WonderDecision:
    """What the wonder pass does, why, and at what effort (``None`` when it does not run)."""

    outcome: str
    effort: str | None
    reason: str


def wonder_decision(
    route: LatencyRoute, summary: RiskSummary, *, folded: bool, tier: str
) -> WonderDecision:
    """Decide whether to run the wonder pass, purely and totally.

    Rule order: a folded design lens never runs its own pass; next, the legacy
    trivial-diff gate applies only under a route that asks for it (forensic) and
    only when no mandatory floor forbids the skip (MH5); then the route's own
    wonder choice; otherwise the pass runs at the route's effort, naming any
    floor that raised it above the profile.
    """
    if folded:
        return WonderDecision("folded", None, "design alternatives folded into the structural review")
    if tier == "skip" and route.legacy_trivial_tier_gate and not summary.floors:
        return WonderDecision("skip", None, "trivial diff (<=1 changed file)")
    if route.wonder == "skip":
        return WonderDecision("skip", None, f"profile {route.profile} routes wonder to skip")
    raised = f" (raised by {', '.join(summary.floors)})" if summary.floors else ""
    return WonderDecision(
        "run", route.wonder, f"profile {route.profile} routes wonder to run at {route.wonder}{raised}"
    )


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
