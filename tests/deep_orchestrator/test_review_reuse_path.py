"""Real runner tests of cache publication and survival across fresh runs."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import stat
import weakref
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream.backends import AgentEvent
from daydream.deep import review_steps
from daydream.phases.review import phase_per_stack_reviews
from daydream.review_stage_inputs import StageInputFactory
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _arbiter_stacks,
    _count_review_prompts,
)
from tests.harness.review_profile import (
    independent_alternatives_profile,
)
from tests.harness.review_result import saved_coverage
from tests.harness.stub_backend import force_interactive, install_stub_backend, silence
from tests.test_deep_orchestrator import MakeConfig


def _records_bytes(target: Path) -> dict[str, bytes]:
    """The canonical review artifact set (A7), keyed by artifact basename."""
    deep = target / ".daydream" / "deep"
    paths = sorted(deep.glob("stack-*-records.json"))
    merged = deep / "merged-items.json"
    if merged.is_file():
        paths.append(merged)
    return {path.name: path.read_bytes() for path in paths}


_PER_STACK_PROMPT = re.compile(r"you are reviewing the (\S+) stack", re.IGNORECASE)
_INTENT_DISCRIMINATOR = "understand the intent of these changes"
_WONDER_DISCRIMINATOR = "evaluate the implementation"


def _count_unit_prompts(calls: list[dict[str, object]], needle: str) -> int:
    """How many captured calls carried the given production prompt discriminator."""
    return sum(1 for call in calls if needle in str(call.get("prompt", "")).lower())


def _reused_stacks(deep: Path) -> set[str]:
    """The shard units this run's provenance recorded with a hit outcome."""
    record = _latest_provenance(deep)
    units = cast(dict[str, object], record.get("units") or {})
    return {unit.removeprefix("shard:")
        for unit, trace in units.items()
        if unit.startswith("shard:")
        and isinstance(trace, dict)
        and trace.get("outcome") == "hit"
    }


def _newest_provenance_path(deep: Path) -> Path:
    """The most recently written per-run provenance record in the store."""
    provenance = deep.parent / "review-cache" / "provenance"
    records = sorted(provenance.glob("*.json"), key=lambda path: path.stat().st_mtime)
    assert records, "the run must record its reuse provenance inside the store"
    return records[-1]


def _latest_provenance(deep: Path) -> dict[str, object]:
    """The most recently written per-run provenance record in the store."""
    record: dict[str, object] = json.loads(_newest_provenance_path(deep).read_text(encoding="utf-8"))
    return record


_MERGE_DISCRIMINATOR = "cross-stack merge agent"


def _review_surface_prompts(calls: list[dict[str, object]],) -> list[dict[str, object]]:
    """Captured calls that performed deep review work (the paid review surface).

    Filters by the production prompt discriminators, so an unrelated fix or test
    call never masks a genuine review call and a warm run that still calls a
    reviewer fails the assertion instead of passing vacuously.
    """
    surface: list[dict[str, object]] = []
    for call in calls:
        prompt = str(call.get("prompt", "")).lower()
        if (_PER_STACK_PROMPT.search(prompt)
            or _ARBITER_DISCRIMINATOR in prompt
            or _INTENT_DISCRIMINATOR in prompt
            or _WONDER_DISCRIMINATOR in prompt
            or _MERGE_DISCRIMINATOR in prompt
        ):
            surface.append(call)
    return surface


async def test_identical_rerun_restores_completed_units_and_current_run_evidence(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Every warm contract retains exact bytes, positive evidence, and no paid review calls.

    A denied optional rationale cleanup on the warm run must not admit stale same-source rationale.
    """
    silence(monkeypatch)
    force_interactive(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_by_stack = _arbiter_stacks({"python": "high", "react": "high", "generic": "high"})
    stub.merge_echo_records = True
    config = make_config(multi_stack_target, output_mode="loop", non_interactive=False,
                         review_profile=independent_alternatives_profile(), latency_profile="balanced")
    assert await run(config) == 0
    deep = multi_stack_target / ".daydream" / "deep"
    names = ["intent.md", "alternatives.json", "dedup-candidates.json"]
    canonical = {**_records_bytes(multi_stack_target), **{name: (deep / name).read_bytes() for name in names}}
    failures = saved_coverage(deep).unfinished_scopes
    first = json.loads((deep / "review-coverage.json").read_text())
    attribution = {scope['scope_id']: scope['target_decisions'] for scope in first['stack_outcomes']}
    assert all(attribution.values()), 'completed scopes must retain source-attributed target assertions'
    for scope_id, decisions in attribution.items():
        records = json.loads((deep / f'stack-{scope_id}-records.json').read_text())
        assert 'target_decisions' not in records, 'retained attribution must not inflate later model inputs'
        cached = list((deep.parent / 'review-cache/entries').glob(f'*/stack-{scope_id}-records-attribution.json'))
        assert cached, 'completed cache entries require hash-verified source attribution'
        for path in cached:
            manifest = json.loads((path.parent / 'manifest.json').read_text())
            assert manifest['payload'][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert json.loads(path.read_text()) == {'target_decisions': decisions}
        assert all(any(ref['role'] == 'assigned' for ref in decision['evidence_refs']) for decision in decisions)
    obstructed: list[Path] = []
    prior_rationale = (deep / 'stack-python-report-context.json').read_bytes()
    unlink = Path.unlink

    def obstruct_optional_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        if (path.name == 'stack-python-report-context.json'
                and tuple(parent.name for parent in path.parents[:3]) == ('deep', '.daydream', 'live')
                and not obstructed):
            obstructed.append(path)
            path.write_bytes(prior_rationale)
            raise PermissionError('optional report cleanup denied')
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', obstruct_optional_cleanup)
    assert _count_unit_prompts(stub.calls, _INTENT_DISCRIMINATOR) == 1
    assert _count_unit_prompts(stub.calls, _WONDER_DISCRIMINATOR) == 1
    assert _count_arbiter_prompts(stub.calls) >= 1
    stub.calls.clear()
    assert await run(config) == 0
    assert _review_surface_prompts(stub.calls) == [], "the warm run dispatched fresh reviewer work"
    assert {name: (deep / name).read_bytes() for name in canonical} == canonical
    assert saved_coverage(deep).unfinished_scopes == failures
    second = json.loads((deep / "review-coverage.json").read_text())
    assert first["run_id"] != second["run_id"]
    assert all(first[field] == second[field] for field in ("analyzed_revision", "planned_scopes", "stack_outcomes"))
    assert all(scope["status"] == "complete" for scope in second["stack_outcomes"])
    context = json.loads((deep / 'review-report-context.json').read_text())
    python = next(entry for entry in context['scopes'] if entry['scope_id'] == 'python')
    assert obstructed and python['review_context'] is None, 'failed cleanup must not admit stale rationale'
    assert (deep / 'stack-python-report-context.json').read_bytes() == prior_rationale
    assert all(entry['originating_run_id'] == first['run_id'] for entry in context['scopes'])
    units = cast(dict[str, Any], _latest_provenance(deep)["units"])
    for name, grounding in (("intent", {"exploration"}), ("alternatives", {"intent", "exploration"}),
                            ("arbiter", {"intent", "alternatives", "exploration"})):
        assert units[name]["outcome"] == "hit" and set(units[name]["grounding_status"]) == grounding
    assert sorted(path.name for path in deep.glob("arbiter-*-complete.marker"))
    # A directory-fd lookup avoids this kernel's stale per-thread negative dentry
    # after the artifact-session publication worker replaces the public report.
    parent_fd = os.open(multi_stack_target, os.O_RDONLY)
    try:
        assert stat.S_ISREG(os.stat(".review-output.md", dir_fd=parent_fd).st_mode)
        report_fd = os.open(".review-output.md", os.O_RDONLY, dir_fd=parent_fd)
        try:
            assert "## Coverage" not in os.read(report_fd, 1 << 20).decode("utf-8")
        finally:
            os.close(report_fd)
    finally:
        os.close(parent_fd)


_ARBITER_DISCRIMINATOR = "you are the arbiter"


def _count_arbiter_prompts(calls: list[dict[str, object]]) -> int:
    """How many captured calls carried the production arbiter prompt."""
    return _count_unit_prompts(calls, _ARBITER_DISCRIMINATOR)


def _damage_source_attribution(decisions: list[dict[str, Any]], damage: str) -> bool:
    """Drop one retained target without relying on shape, revision, or hash rejection."""
    assert damage == "attribution-missing-target"
    if len(decisions) < 2:
        return False
    decisions.pop()
    return True


async def test_cache_population_restores_warm_bytes_then_recomputes_only_independently_damaged_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry-local proof failures cannot invalidate an untouched sibling or establish false cache coverage."""
    from tests.deep_orchestrator.test_review_investigation import InvestigationRun
    from tests.harness.git_helpers import seed_feature_branch
    from tests.harness.stub_backend import review_stage_state

    # Manifest-level coverage/origin/format proofs belong to the reuse-store owner test; these damages
    # reach the per-stack restore gates that only a real shard lookup exercises.
    damages = ["legacy", "missing-attribution", "attribution-missing-target"]
    repo = tmp_path / "cache_population"
    before = {f"{name}/module_{index}.py": "VALUE = 0\n" for name in [*damages, "healthy"] for index in range(2)}
    after = {path: "VALUE = 1\n" for path in before}
    after["healthy/module_0.py"] = "VALUE = '" + "界" * 11000 + "'\n"
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    config = dict(
        output_mode='loop',
        findings_out=None,
        deep_shard_enabled=True,
        deep_shard_max_files=2,
        deep_shard_max_bytes=10**9,
        deep_shard_fanout_cap=32,
        review_cache_enabled=True,
    )
    assert await review.run(**config) == 0
    deep = repo / ".daydream/deep"
    first = saved_coverage(deep)
    by_name = {
        scope.files[0].split("/")[0]: scope.scope_id for scope in first.planned_scopes if scope.scope_id != "structure"
    }
    assert set(by_name) == {*damages, "healthy"}
    original = {scope: outcome["target_decisions"] for scope, outcome in first.scopes.items()}
    fragments = [
        decision
        for decision in original[by_name["healthy"]]
        if any(ref["fragment"] is not None for ref in decision["evidence_refs"])
    ]
    assert len(fragments) > 1
    assert all(
        ref["source_range"] is None and ref["fragment"]["bytes"] > 0
        for decision in fragments
        for ref in decision["evidence_refs"]
    )
    canonical = _records_bytes(repo)
    entries = deep.parent / "review-cache/entries"
    manifests = {json.loads(path.read_text())["unit"]: path for path in entries.glob("*/manifest.json")}
    unknown = entries / ("a" * 64)
    unknown.mkdir()
    optional = {}
    for name, fault in [("legacy", "missing"), ("missing-attribution", "corrupt"),
                        ("attribution-missing-target", "nested")]:
        scope = by_name[name]
        manifest = manifests["shard:" + scope]
        path = manifest.parent / f"stack-{scope}-report-context.json"
        assert path.name not in json.loads(manifest.read_text())["payload"]
        optional[scope] = fault
        if fault == "missing":
            path.unlink()
        else:
            path.write_text("{corrupt" if fault == "corrupt" else "[" * 30000 + "0" + "]" * 30000)
    review.backend.calls.clear()
    assert await review.run(**config) == 0
    assert _review_surface_prompts(review.backend.calls) == []
    assert _records_bytes(repo) == canonical and unknown.is_dir()
    second = saved_coverage(deep)
    assert first.run_id != second.run_id and first.revision == second.revision
    assert second.scopes == first.scopes
    context = json.loads((deep / "review-report-context.json").read_text())
    contexts = {scope["scope_id"]: scope for scope in context["scopes"]}
    assert all(contexts[scope]["review_context"] is None for scope in optional)
    assert all(scope["originating_run_id"] == first.run_id for scope in contexts.values())

    for damage in damages:
        manifest_path = manifests["shard:" + by_name[damage]]
        manifest = json.loads(manifest_path.read_text())
        if damage == "legacy":
            manifest.pop("coverage")
        else:
            payload_path = next(manifest_path.parent / name for name in manifest["payload"]
                                if name.endswith("-records-attribution.json"))
            if damage == "missing-attribution":
                manifest["payload"].pop(payload_path.name)
            else:
                payload = json.loads(payload_path.read_text())
                assert _damage_source_attribution(payload["target_decisions"], damage)
                payload_path.write_text(json.dumps(payload))
                manifest["payload"][payload_path.name] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    # One fan-out slot: each miss builds scope-local inputs only once admitted, while hits validate without waiting.
    review.backend.fanout_concurrency = 1
    hits = {by_name["healthy"], "structure"}
    live: weakref.WeakSet[StageInputFactory] = weakref.WeakSet()
    built: list[str] = []
    waited: list[str] = []
    planned: list[str] = []
    construct = StageInputFactory.__init__

    async def planned_review(backend: Any, work: Any, stacks: list[Any], **options: Any) -> Any:
        planned.extend(stack.stack_name for stack in stacks)
        return await phase_per_stack_reviews(backend, work, stacks, **options)

    def tracked(inputs: StageInputFactory, *args: Any, **kwargs: Any) -> None:
        construct(inputs, *args, **kwargs)
        live.add(inputs)
        built.append(inputs.stack.stack_name)

    async def admitted(stage: dict[str, Any], output: dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        with anyio.fail_after(10):
            while not hits <= set(built) or (gc.collect(), len(live))[1] != 1:
                await anyio.sleep(0.01)
        if stage["scope_id"] not in waited:
            waited.append(stage["scope_id"])
        events: tuple[AgentEvent, ...] = ()
        for event in events:
            yield event

    monkeypatch.setattr(StageInputFactory, "__init__", tracked)
    monkeypatch.setattr(review_steps, "phase_per_stack_reviews", planned_review)
    for name in damages:
        review.backend.script(by_name[name])(admitted)
    review.backend.calls.clear()
    assert await review.run(**config) == 0
    actual = {
        stage["scope_id"] for call in review.backend.calls if (stage := review_stage_state(call["prompt"])) is not None
    }
    assert actual == set(waited) == {by_name[name] for name in damages}
    # Admission follows arrival at the limiter: misses without an entry in planned order, then
    # scopes whose entry failed validation, in planned order once that validation finishes.
    revalidated = {scope for scope in waited if built.count(scope) == 2}
    assert waited == ([scope for scope in planned if scope in set(waited) - revalidated]
                      + [scope for scope in planned if scope in revalidated])
    assert revalidated and revalidated != set(waited)
    assert _reused_stacks(deep) == {by_name["healthy"], "structure"}
    current = saved_coverage(deep)
    assert all(scope["status"] == "complete" for scope in current.scopes.values())
    assert {scope: outcome["target_decisions"] for scope, outcome in current.scopes.items()} == original
    assert unknown.is_dir()
    review.backend.calls.clear()
    assert await review.run(start_at="merge", **config) == 0
    assert _count_review_prompts(review.backend.calls) == 0
    assert {scope: outcome["target_decisions"] for scope, outcome in saved_coverage(deep).scopes.items()} == original


async def test_resume_rejects_missing_or_mismatched_coverage(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """Resume requires a valid same-snapshot inventory and diff identity; bytes alone are insufficient."""
    from tests.test_deep_orchestrator import _pin_findings_pr

    silence(monkeypatch)
    force_interactive(monkeypatch)
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    fixing = {"output_mode": "loop", "non_interactive": False, "pr_number": pr.number}
    assert await run(make_config(multi_stack_target, **fixing)) == 0
    deep = multi_stack_target / ".daydream" / "deep"
    path, key = deep / "review-coverage.json", deep / "diff-key"
    coverage = path.read_bytes()
    report = multi_stack_target / '.review-output.md'
    report_bytes = report.read_bytes()
    for missing in (path, key):
        missing.unlink()
        stub.calls.clear()
        assert await run(make_config(multi_stack_target, start_at="merge", **fixing)) == 1
        assert stub.calls == []
        assert not missing.exists()
        assert report.read_bytes() == report_bytes
        path.write_bytes(coverage)


def test_reuse_summary_reports_reused_units_that_ran_under_moved_grounding() -> None:
    """The one-line reuse outcome discloses moved grounding instead of implying fresh equivalence.

    Warm runner journeys reuse under unchanged grounding, so the MH16 disclosure never appears there.
    """
    from daydream.deep.review_steps import _reuse_summary_line

    record = {'units': {'shard:python': {'outcome': 'hit', 'grounding': {'intent': {'moved': True}}},
                        'intent': {'outcome': 'miss'}, 'bogus': 'unreadable'},
              'store': {'entries': 2, 'bytes': 10}}
    assert _reuse_summary_line(record) == (
        'Review reuse: 3 units (1 hit, 1 miss); shard: 1; reused: shard:python; store: 2 entries, 10 bytes; '
        'grounding moved in 1 reused')


def test_reuse_entry_without_matching_coverage_proof_or_lineage_is_a_named_miss(tmp_path: Path) -> None:
    """Payload-valid entries are still misses when their coverage proof or origin lineage is unusable.

    The runner cache journey damages only per-stack restore payloads; every manifest-level proof
    (legacy, corrupt, incomplete, foreign head/diff/scope, malformed origin) is decided here.
    """
    from daydream.deep import reuse_store
    from tests.test_reuse_key import _identity

    store = reuse_store.ReuseCache(tmp_path / "review-cache", run_id="run-1", session_id="s1")
    proof: dict[str, Any] = {"schema_version": 1, "unit": "shard:python", "status": "complete",
             "analyzed_revision": {"head_sha": "h" * 40, "diff_key": "d" * 64},
             "planned_scopes": [{"scope_id": "python", "files": ["api.py"]}]}
    key = "a" * 64
    store.store(key, unit="shard:python", payload={"stack-python-records.json": b"{}"}, components={},
                identity=_identity(), grounding={}, grounding_status={}, coverage=proof)
    manifest_path = reuse_store.entry_manifest_path(store.store_dir, key)
    original = json.loads(manifest_path.read_text())
    (tmp_path / "out").mkdir()
    assert reuse_store.lookup_reuse_entry(store, "shard:python", key, tmp_path / "out",
                                          expected_coverage=proof) is not None
    damaged_proofs = [None, "invalid proof", {**proof, "status": "incomplete"},
                      {**proof, "analyzed_revision": {**proof["analyzed_revision"], "head_sha": "f" * 40}},
                      {**proof, "analyzed_revision": {**proof["analyzed_revision"], "diff_key": "f" * 64}},
                      {**proof, "planned_scopes": [{"scope_id": "python", "files": ["foreign.py"]}]}]
    for damaged in damaged_proofs:
        manifest = {key_: value for key_, value in original.items() if key_ != "coverage"}
        if damaged is not None:
            manifest["coverage"] = damaged
        manifest_path.write_text(json.dumps(manifest))
        assert reuse_store.lookup_reuse_entry(store, "shard:python", key, tmp_path / "out",
                                              expected_coverage=proof) is None
        unit = store.provenance()["units"]["shard:python"]
        assert (unit["outcome"], unit["reason"]) == ("miss", "complete coverage proof absent or mismatched")
    manifest_path.write_text(json.dumps({**original, "origin": "malformed lineage"}))
    miss = store.lookup(key)
    assert isinstance(miss, reuse_store.ReuseMiss) and miss.reason == "manifest origin unreadable"
