"""Pure latency-profile decision tests (issue #732).

The decision is a value: profiles, risk floors, group planning. Nothing here
touches the filesystem, the clock, or the environment -- see MH3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.adjudication_steps import _load_group_verdicts, _persist_group_verdicts
from daydream.deep.arbiter import partition_arbiter_targets
from daydream.deep.artifacts import (
    DeepArtifact,
    arbiter_group_verdicts_path,
)
from daydream.deep.latency import (
    ARBITER_EFFORTS,
    PROFILE_ROUTES,
    WONDER_ROUTES,
    DiffSignals,
    FindingSignals,
    PlannedGroup,
    diff_signals,
    resolve_latency_profile,
    route_for,
    summarize_risk,
    wonder_decision,
)


def _rec(uid: str, file: str, line: int, severity: str = "medium") -> dict[str, object]:
    return {"uid": uid, "file": file, "line": line, "severity": severity}

def test_a_location_is_never_split_across_groups() -> None:
    records = [_rec(f"py:{i}", "api.py", 10 if i < 3 else 20) for i in range(1, 5)]
    groups = partition_arbiter_targets(records, [0, 1, 2, 3], edges={}, max_targets=2)
    same_location = [g for g in groups if "py:1" in g.target_uids][0]
    assert {"py:1", "py:2", "py:3"} <= set(same_location.target_uids)


def test_groups_are_order_independent_and_identity_is_stable() -> None:
    records = [_rec("py:1", "api.py", 10), _rec("react:1", "App.tsx", 4), _rec("py:2", "api.py", 10),
        _rec("py:3", "api.py", 12),
    ]
    edges = {"api.py": {"App.tsx"}}          # co-located via one import edge
    first = partition_arbiter_targets(records, [0, 1, 2, 3], edges=edges, max_targets=2)
    shuffled = partition_arbiter_targets(records, [3, 1, 2, 0], edges=edges, max_targets=2)
    assert [g.target_uids for g in first] == [g.target_uids for g in shuffled]
    assert [g.group_id for g in first] == ["arbiter-group-0", "arbiter-group-1"]
    assert sum(len(g.target_uids) for g in first) == 4

def _signals(diff: str, *, files: int = 4, stacks: int = 2) -> DiffSignals:
    return diff_signals(diff=diff, changed_files=files, stack_count=stacks)


ROUTINE_DIFF = "diff --git a/notes.md b/notes.md\n--- a/notes.md\n+++ b/notes.md\n+wording\n"

@pytest.mark.parametrize(("diff", "floor"),
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
    assert summary.floors == ()
    assert route_for("fast", summary).arbiter_effort == "medium"
    assert summarize_risk(_signals("+é", stacks=1)).size_score == 0
    assert summarize_risk(_signals("+x\n" * 201)).size_score == 1
    assert summarize_risk(_signals("+x\n" * 200)).size_score == 0
    assert summarize_risk(_signals("+é" * 4100)).size_score == 1

def test_unknown_profile_fails_safe_upward_and_says_so() -> None:
    resolved = resolve_latency_profile("turbo", source="cli")
    assert resolved.profile == "forensic"
    assert resolved.fail_safe is True
    assert "turbo" in (resolved.reason or "")

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

@pytest.mark.parametrize('fault',
                         ['identity', 'null-targets', 'id-binding', 'boolean', 'missing-field', 'missing-verdict',
                          'execution-contract', 'record-evidence'])
def test_group_cache_rejects_malformed_bound_verdicts(tmp_path: Path, fault: str) -> None:
    group = PlannedGroup('arbiter-group-0', ('python:1',), 'high', 'policy')
    verdict: dict[str, Any] = {'arb_id': 1, 'keep': True, 'severity': 'high', 'confidence': 'HIGH',
                               'description': 'finding', 'rationale': 'verified', 'evidence': 'a.py:1'}
    _persist_group_verdicts(tmp_path, group, {1: verdict}, contract="execution-contract", records_digest="records")
    loaded = _load_group_verdicts(tmp_path, group, contract="execution-contract", records_digest="records")
    assert loaded == {1: verdict}
    path = arbiter_group_verdicts_path(tmp_path, group.group_id)
    payload = json.loads(path.read_text())
    if fault == 'execution-contract':
        payload['contract'] = 'foreign'
    elif fault == 'record-evidence':
        payload['input_digest'] = payload['settled_digest'] = 'foreign'
    elif fault == 'identity':
        payload['group_id'] = 'other'
    elif fault == 'null-targets':
        payload['target_uids'] = None
    elif fault == 'id-binding':
        payload['verdicts']['1']['arb_id'] = 2
    elif fault == 'boolean':
        payload['verdicts']['1']['keep'] = 'no'
    elif fault == 'missing-field':
        del payload['verdicts']['1']['evidence']
    else:
        payload['verdicts'] = {}
    path.write_text(json.dumps(payload))
    assert _load_group_verdicts(tmp_path, group, contract="execution-contract", records_digest="records") is None


@pytest.mark.parametrize('change', ['model', 'effort', 'schema'])
def test_arbiter_contract_binds_actual_execution(tmp_path: Path, make_config: Any, make_work: Any,
                                               monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from daydream.deep.adjudication_steps import _arbiter_plan_component
    from daydream.deep.latency import ArbiterPlan
    from daydream.deep.state import DeepState
    from daydream.extensions import get_registry
    from daydream.flows.engine import FlowContext
    from tests.harness.review_result import review_coverage

    path = tmp_path / 'intent.md'
    path.write_text('current context')
    backend = SimpleNamespace(model='first-model', reasoning_effort='high')
    ctx = FlowContext(make_config(tmp_path), make_work(tmp_path), get_registry(),
        data={'review_coverage': review_coverage(), 'intent_path': path, 'alts_path': path,
              'exploration_dir': None, 'intent_summary': 'current context'})
    monkeypatch.setattr(ctx, 'backend_for', lambda _: backend)
    monkeypatch.setattr(ctx, 'backend_for_effort', lambda *_: backend)
    group = PlannedGroup('arbiter-group-0', ('python:1',), 'high', 'policy')
    plan = ArbiterPlan(True, (group,), 'policy')
    before = _arbiter_plan_component(ctx, DeepState(ctx.data), plan, None)
    if change == 'model':
        backend.model = 'second-model'
    elif change == 'effort':
        plan = replace(plan, groups=(replace(group, effort='medium'),))
        backend.reasoning_effort = 'medium'
    else:
        monkeypatch.setattr('daydream.deep.adjudication_steps.ARBITER_SCHEMA', {'type': 'object'})
    after = _arbiter_plan_component(ctx, DeepState(ctx.data), plan, None)
    assert before['groups'][0]['target_uids'] == after['groups'][0]['target_uids']
    assert before['groups'][0]['contract'] != after['groups'][0]['contract']


@pytest.mark.parametrize('phase', ['arbiter', 'suppression'])
def test_whole_adjudication_proof_requires_current_positive_stage(
    tmp_path: Path, phase: str, make_config: Any, make_work: Any,
) -> None:
    from daydream.deep.adjudication_steps import _completed_adjudication, _digest
    from daydream.deep.records import RecordPool
    from daydream.deep.state import DeepState
    from tests.harness.review_result import review_coverage

    coverage = review_coverage(scope_ids=(), phases=('arbiter', 'suppression'))
    for stage in coverage.phases:
        coverage.record_phase(stage, 'complete', noop=True)
    state = DeepState({'dd': tmp_path, 'record_pool': RecordPool({}, {}), 'review_coverage': coverage})
    from daydream.extensions import get_registry
    from daydream.flows.engine import FlowContext

    ctx = FlowContext(make_config(tmp_path), make_work(tmp_path), get_registry())
    contract = {'plan': None, 'precision_mode': True, 'suppression': 'execution-contract'}
    DeepArtifact.ADJUDICATION_COMPLETE.at(tmp_path).write_text(json.dumps({**contract, 'records': _digest([])}))
    assert _completed_adjudication(ctx, state, contract)
    if phase == 'arbiter':
        coverage.record_phase('arbiter', 'complete', noop=False)
        assert not _completed_adjudication(ctx, state, contract)
    coverage.record_phase(phase, 'uncovered', reasons=('coverage_unknown',))
    assert not _completed_adjudication(ctx, state, contract)
