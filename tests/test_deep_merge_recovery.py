"""Recover malformed cross-stack merges into partial, resumable reports.

Coverage includes current envelope validation and bare-list rejection, structured error context, surviving
per-stack artifacts, typed synthesis failure outcomes, and fix/merge resume.
Salvage dedup uses host UIDs; malformed unbound pairs reject.
Phase tests mock only the backend; integration uses runner.run.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.artifacts import (
    DeepArtifact,
    deep_dir,
)
from daydream.deep.merge_steps import _drop_cross_stack_duplicates, _step_cross_stack_merge, _step_load_items
from daydream.deep.reuse_store import ReuseCache, review_cache_dir
from daydream.extensions import get_registry
from daydream.flows.engine import FlowContext
from daydream.review_budget import review_warnings
from daydream.review_result import AnalyzedRevision, PlannedScope, ReasonCode, ReviewCoverage
from daydream.run_config import RunConfig
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.review_result import record_pool, review_coverage
from tests.harness.trajectory import make_recorder


@pytest.fixture(autouse=True)
def _no_open_pr(fake_gh: FakeGh) -> None:
    fake_gh.set_response("pr-list", value=[])


async def test_empty_merge_cold_reuse_and_resume_preserve_coverage_and_lifecycle(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    """A deterministic merge remains a complete #733 unit and a resumable phase."""
    dd = deep_dir(tmp_path, allow_standalone=True)
    records = dd / "stack-python-records.json"
    records.write_text('{"issues": []}')
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    intent = dd / "intent.md"
    intent.write_text("Review the new API contract")
    backend = ScriptedBackend(events=[AssertionError("empty synthesis must not dispatch")])
    cache = ReuseCache(review_cache_dir(dd), run_id="empty-synthesis")
    coverage = ReviewCoverage("empty-synthesis", AnalyzedRevision("a" * 40, "b" * 40, "c" * 64),
        [PlannedScope("python", "python", ("api.py",)), PlannedScope("react", "react", ("App.tsx",))], ["merge"])
    coverage.record_scope("python", "incomplete", reasons=("host_tool_budget_exhaustion",),
                          partial_evidence=True, diagnostic="review budget exhausted")
    coverage.record_scope("react", "failed", reasons=(ReasonCode.HOST_WALL_BUDGET_EXHAUSTION,),
                          diagnostic="budget exhausted: wall deadline")
    ctx = FlowContext(
        RunConfig(target=str(tmp_path), review_cache_enabled=True), make_work(tmp_path), get_registry(),
        data={
            "dd": dd, "alts_path": alternatives, "intent_path": intent,
            "record_pool": record_pool(dd, paths=[records]),
            "exploration_dir": None, "reuse_cache": cache,
            "review_coverage": coverage,
        },
        allow_standalone_artifacts=True, _backend_factory=lambda *args: backend,
    )
    recorder = make_recorder(tmp_path)
    async with recorder:
        # Each successful synthesis must supersede its own stale salvage and
        # budget stop while retaining independent missing reviewer coverage.
        coverage.record_phase('merge', 'failed', reasons=('synthesis_failure',), diagnostic='old timeout')
        assert await _step_cross_stack_merge(ctx) is None
        assert json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text()) == {"items": []}
        assert json.loads(DeepArtifact.DEDUP_CANDIDATES.at(dd).read_text()) == {
            "record_alt_pairs": [], "record_duplicate_pairs": [],
        }
        assert coverage.unfinished_scopes == {"react": "budget exhausted: wall deadline",
                                          "python": "review budget exhausted"}
        assert review_warnings(dd) == ("python: review budget exhausted", "react: budget exhausted: wall deadline")
        manifests = list((cache.store_dir / "entries").glob("*/manifest.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        assert manifest["unit"] == "merge"
        assert {"merged-items.json", "dedup-candidates.json"} <= set(manifest["payload"])
        cold_items = DeepArtifact.MERGED_ITEMS.at(dd).read_bytes()

        # Same completed records restore through the ordinary reuse contract.
        DeepArtifact.MERGED_ITEMS.at(dd).write_text('{"items": [{"description": "stale result"}]}')
        assert await _step_cross_stack_merge(ctx) is None
        assert DeepArtifact.MERGED_ITEMS.at(dd).read_bytes() == cold_items
        provenance = cache.store_dir / "provenance" / "empty-synthesis.json"
        trace = json.loads(provenance.read_text())
        assert trace["units"]["merge"]["outcome"] == "hit"
        reused_trace = provenance.read_bytes()

        # An explicit merge resume recomputes the host output without looking
        # up or republishing cache units and still records a successful phase.
        ctx.config.start_at = "merge"
        DeepArtifact.MERGED_ITEMS.at(dd).write_text('{"items": [{"description": "stale result"}]}')
        assert await _step_cross_stack_merge(ctx) is None
        assert DeepArtifact.MERGED_ITEMS.at(dd).read_bytes() == cold_items
        assert provenance.read_bytes() == reused_trace
        assert await _step_load_items(ctx) is None
        report = ctx.data["merged_report"].read_text()
        assert "react" in report
        assert "python" in report
    assert backend.call_count == 0
    events = [event.to_dict() for event in recorder._phase_events]
    merge_ends = [event for event in events if event["phase"] == "merge" and event["event"] == "phase_end"]
    assert len(merge_ends) == 3
    assert all(event["status"] == "succeeded" for event in merge_ends)
    terminal = coverage.finalize("completed")
    assert terminal["analysis_state"] == "incomplete"
    assert terminal["failed_stacks"] == ["react"]
    assert terminal["phase_outcomes"] == [{"phase": "merge", "status": "complete", "reason_codes": [],
                                           "noop": True, "usable_evidence": False}]


@pytest.mark.parametrize('diagnostics', [{'scopes': {}, 'phases': {'extra': 'failure'}},
                                       ])
def test_coverage_rejects_malformed_failure_diagnostics(diagnostics: Any) -> None:
    payload = review_coverage().to_dict()
    payload['diagnostics'] = diagnostics
    with pytest.raises(ValueError):
        ReviewCoverage.from_dict(payload)

def test_merge_salvage_rejects_dedup_without_current_identity(tmp_path: Path) -> None:
    payloads: tuple[dict[str, Any], ...] = ({"record_duplicate_pairs": None}, {"record_duplicate_pairs": [{}]})
    for payload in payloads:
        DeepArtifact.DEDUP_CANDIDATES.at(tmp_path).write_text(json.dumps(payload))
        records = [{"id": 1, "file": "api.py", "description": "Sample issue", "uid": "generic:1"},
                   {"id": 1, "file": "api.py", "description": "Sample issue", "uid": "python:1"}]
        with pytest.raises(ValueError):
            _drop_cross_stack_duplicates(tmp_path, records)
        assert [record["uid"] for record in records] == ["generic:1", "python:1"]
