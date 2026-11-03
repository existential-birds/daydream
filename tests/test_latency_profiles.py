"""Pure latency-profile decision tests (issue #732).

The decision is a value: profiles, risk floors, group planning. Nothing here
touches the filesystem, the clock, or the environment -- see MH3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.deep.adjudication_steps import _load_group_verdicts, _persist_group_verdicts
from daydream.deep.arbiter import partition_arbiter_targets
from daydream.deep.artifacts import (
    DeepArtifact,
    arbiter_group_verdicts_path,
)
from daydream.deep.latency import (
    DiffSignals,
    FindingSignals,
    PlannedGroup,
    diff_signals,
    route_for,
    summarize_risk,
    wonder_decision,
)


def _rec(uid: str, file: str, line: int, severity: str = "medium") -> dict[str, object]:
    return {"uid": uid, "file": file, "line": line, "severity": severity}

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


def test_wonder_decision_is_pure_over_route_summary_tier_and_fold() -> None:
    # High-severity or contested findings forbid the cheapest route.
    for findings in (FindingSignals(high_severity=True, contested=False), FindingSignals(False, True)):
        route = route_for("fast", summarize_risk(_signals(ROUTINE_DIFF), findings))
        assert route.wonder != "skip" and route.arbiter_effort != "medium"
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

def test_group_cache_rejects_malformed_bound_verdicts(tmp_path: Path) -> None:
    group = PlannedGroup('arbiter-group-0', ('python:1',), 'high', 'policy')
    verdict: dict[str, Any] = {'arb_id': 1, 'keep': True, 'severity': 'high', 'confidence': 'HIGH',
                               'description': 'finding', 'rationale': 'verified', 'evidence': 'a.py:1'}
    path = arbiter_group_verdicts_path(tmp_path, group.group_id)

    def foreign_contract(payload: dict[str, Any]) -> None:
        payload['contract'] = 'foreign'

    def null_targets(payload: dict[str, Any]) -> None:
        payload['target_uids'] = None

    def rebind_id(payload: dict[str, Any]) -> None:
        payload['verdicts']['1']['arb_id'] = 2

    def drop_field(payload: dict[str, Any]) -> None:
        del payload['verdicts']['1']['evidence']

    def drop_verdict(payload: dict[str, Any]) -> None:
        payload['verdicts'] = {}

    for fault in (foreign_contract, null_targets, rebind_id, drop_field, drop_verdict):
        _persist_group_verdicts(tmp_path, group, {1: verdict}, contract="execution-contract", records_digest="records")
        loaded = _load_group_verdicts(tmp_path, group, contract="execution-contract", records_digest="records")
        assert loaded == {1: verdict}
        payload = json.loads(path.read_text())
        fault(payload)
        path.write_text(json.dumps(payload))
        assert _load_group_verdicts(tmp_path, group, contract="execution-contract", records_digest="records") is None


def test_whole_adjudication_proof_requires_current_positive_stage(
    tmp_path: Path, make_config: Any, make_work: Any,
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
    coverage.record_phase('arbiter', 'complete', noop=False)
    assert not _completed_adjudication(ctx, state, contract)
    coverage.record_phase('arbiter', 'uncovered', reasons=('coverage_unknown',))
    assert not _completed_adjudication(ctx, state, contract)
