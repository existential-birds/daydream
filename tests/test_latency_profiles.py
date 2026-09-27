"""Pure latency-profile decision tests (issue #732).

The decision is a value: profiles, risk floors, group planning. Nothing here
touches the filesystem, the clock, or the environment -- see MH3.
"""
from __future__ import annotations

import pytest

from daydream.deep.latency import (
    ARBITER_EFFORTS,
    DEFAULT_LATENCY_PROFILE,
    FAIL_SAFE_LATENCY_PROFILE,
    LATENCY_PROFILES,
    PROFILE_ROUTES,
    WONDER_ROUTES,
    DiffSignals,
    FindingSignals,
    diff_signals,
    resolve_latency_profile,
    route_for,
    summarize_risk,
    wonder_decision,
)


def _signals(diff: str, *, files: int = 4, stacks: int = 2) -> DiffSignals:
    return diff_signals(diff=diff, changed_files=files, stack_count=stacks)


ROUTINE_DIFF = "diff --git a/notes.md b/notes.md\n--- a/notes.md\n+++ b/notes.md\n+wording\n"


@pytest.mark.parametrize(
    ("diff", "floor"),
    [
        ("+def authenticate(user):\n", "security_surface"),
        ("+    threading.Lock()\n", "concurrency_surface"),
        ("+ALTER TABLE users ADD COLUMN ttl int;\n", "persistence_surface"),
        ("+@app.route('/v1/things')\n", "interface_surface"),
        ("+    migrations/0007_ttl.py\n", "migration_surface"),
    ],
)
def test_each_surface_family_forbids_the_cheapest_route(diff: str, floor: str) -> None:
    summary = summarize_risk(_signals(diff))
    assert floor in summary.floors
    route = route_for("fast", summary)
    assert route.wonder != "skip" and route.arbiter_effort != "medium"
    assert route.wonder == "high" and route.arbiter_effort == "high"


def test_high_severity_or_contested_findings_forbid_the_cheapest_route() -> None:
    for findings in (FindingSignals(high_severity=True, contested=False), FindingSignals(False, True)):
        summary = summarize_risk(_signals(ROUTINE_DIFF), findings)
        route = route_for("fast", summary)
        assert route.wonder != "skip" and route.arbiter_effort != "medium"


def test_file_count_alone_can_neither_route_nor_escalate() -> None:
    """MH4: constant file count, different signals, different routes."""
    routine = summarize_risk(_signals(ROUTINE_DIFF, files=500))
    assert routine.floors == ()
    assert route_for("fast", routine) == PROFILE_ROUTES["fast"]
    risky = summarize_risk(_signals(ROUTINE_DIFF + "+BEGIN;\n", files=1))
    assert route_for("fast", risky).wonder == "high"


def test_escalation_is_monotone_and_profile_can_only_raise_the_floor() -> None:
    """MH6: escalation never lowers; profile rank never inverts."""
    protected = summarize_risk(_signals(ROUTINE_DIFF + "+shop_backend_secret = load()\n"))
    for profile in ("fast", "balanced", "forensic"):
        base = PROFILE_ROUTES[profile]
        raised = route_for(profile, protected)
        assert WONDER_ROUTES.index(raised.wonder) >= WONDER_ROUTES.index(base.wonder)
        assert ARBITER_EFFORTS.index(raised.arbiter_effort) >= ARBITER_EFFORTS.index(base.arbiter_effort)
    assert route_for("balanced", protected).wonder == "high"


def test_diff_signals_scan_changed_lines_and_paths_not_context() -> None:
    signals = _signals(" context authenticate threading.Lock ALTER TABLE\n+++ b/api/openapi.yaml\n-proto3\n")
    assert signals.interface_surface
    assert not signals.security_surface
    assert signals.concurrency_surface is False
    assert signals.persistence_surface is False
    assert _signals("diff --git a/x b/x\n+++ b/schema.proto3\n").interface_surface
    assert _signals("diff --git a/x b/x\n+++ b/x\n-authenticate()\n").security_surface
    assert not _signals("diff --git a/authenticate b/authenticate\n context ALTER TABLE\n").security_surface
    assert not _signals("+def ordinary():\n+class Regular:\n+public field\n").security_surface


def test_risk_scores_are_recorded_but_do_not_set_floors() -> None:
    signals = _signals("+routine\n" * 2001, files=500, stacks=4)
    assert (signals.diff_lines, signals.diff_bytes, signals.changed_files, signals.stack_count) == (
        2001, len("+routine\n".encode()) * 2001, 500, 4
    )
    summary = summarize_risk(signals)
    assert (summary.size_score, summary.breadth_score) == (2, 1)
    assert (summary.floors, summary.wonder_floor, summary.arbiter_floor) == ((), "skip", "medium")
    assert summarize_risk(_signals("+é", stacks=1)).size_score == 0
    assert summarize_risk(_signals("+x\n" * 201)).size_score == 1
    assert summarize_risk(_signals("+x\n" * 200)).size_score == 0
    assert summarize_risk(_signals("+é" * 4100)).size_score == 1


def test_every_risk_floor_is_sorted_and_deduplicated() -> None:
    summary = summarize_risk(
        _signals("+authenticate()\n+authenticate()\n+CREATE TABLE things\n"), FindingSignals(True, True)
    )
    assert summary.floors == tuple(sorted(set(summary.floors)))
    assert summary.floors == (
        "contested_findings", "high_severity_findings", "persistence_surface", "security_surface"
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


def test_wonder_decision_is_pure_over_route_summary_tier_and_fold() -> None:
    summary = summarize_risk(diff_signals(diff="+x\n", changed_files=1, stack_count=1))
    fast = route_for("fast", summary)
    assert wonder_decision(fast, summary, folded=True, tier="parallel").outcome == "folded"

    # MH5: a mandatory floor vetoes the skip a fast route would otherwise take.
    risky = summarize_risk(diff_signals(diff="+threading.Lock()\n", changed_files=1, stack_count=1))
    vetoed = wonder_decision(route_for("fast", risky), risky, folded=False, tier="parallel")
    assert (vetoed.outcome, vetoed.effort) == ("run", "high")
    assert "concurrency_surface" in vetoed.reason

    assert wonder_decision(fast, summary, folded=False, tier="parallel").outcome == "skip"
    # A5: the legacy trivial-diff gate survives only under forensic.
    assert wonder_decision(route_for("forensic", summary), summary, folded=False, tier="skip").outcome == "skip"
    assert wonder_decision(route_for("balanced", summary), summary, folded=False, tier="skip").outcome == "run"
