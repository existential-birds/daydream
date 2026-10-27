"""Real runner tests of cache publication and survival across fresh runs."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

import daydream
from daydream import git_ops
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep import reuse_store
from daydream.deep.render import render_report
from daydream.phases import build_commit_message
from daydream.review_profile import resolve_profile
from daydream.runner import run
from tests.deep_orchestrator.empty_synthesis_support import empty_review_config
from tests.deep_orchestrator.precision_overlap_support import PrecisionOverlapBackend
from tests.deep_orchestrator.support import (
    _arbiter_stacks,
    _count_merge_prompts,
    _count_review_prompts,
)
from tests.harness.fake_gh import FakeGh
from tests.harness.review_profile import (
    independent_alternatives_profile,
    independent_exploration_profile,
)
from tests.harness.review_result import saved_coverage
from tests.harness.stub_backend import install_stub_backend
from tests.test_deep_orchestrator import MakeConfig, _merge_item


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


def _reviewed_stacks(calls: list[dict[str, object]]) -> set[str]:
    """Stack names whose per-stack review prompt ran in this call list."""
    reviewed: set[str] = set()
    for call in calls:
        prompt = call.get("prompt")
        if isinstance(prompt, str):
            match = _PER_STACK_PROMPT.search(prompt)
            if match is not None:
                reviewed.add(match.group(1))
    return reviewed


def _entry_count_for_unit(deep: Path, unit: str) -> int:
    """How many content-addressed entries the store holds for ``unit``."""
    entries = deep.parent / "review-cache" / "entries"
    count = 0
    for entry in entries.iterdir():
        try:
            manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("unit") == unit:
            count += 1
    return count


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


@pytest.mark.parametrize(
    "unit", ["arbiter", "rationale-directory", "rationale-permission", "author-present", "author-absent"]
)
async def test_identical_rerun_restores_completed_units_and_current_run_evidence(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    unit: str,
    fake_gh: FakeGh,
) -> None:
    """Every warm contract retains exact bytes, positive evidence, and no paid review calls."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    options: dict[str, Any] = {}
    author_case = unit.startswith('author-')
    if author_case:
        options['pr_number'] = 7
        fake_gh.serve_pr_view({'state': 'OPEN', 'headRefOid': git_ops.head_sha(multi_stack_target),
                             'body': 'Current author grounding' if unit == 'author-present' else None})
    if unit == "arbiter":
        options["review_profile"] = independent_alternatives_profile()
    if unit == "arbiter":
        options["latency_profile"] = "balanced"
        stub.parse_by_stack = _arbiter_stacks({"python": "high", "react": "high", "generic": "high"})
        stub.merge_echo_records = True
    config = make_config(multi_stack_target, **options)
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
    author_faults: list[Path] = []
    stale_author = {'source': 'pull_request_body', 'head_sha': first['analyzed_revision']['head_sha'],
                    'body': 'Stale optional author assertion', 'truncated': False}
    if author_case:
        from daydream import json_utils
        stage = json_utils._stage_bytes
        def fail_author_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
            if path.name == 'review-author-context.json':
                author_faults.append(path)
                path.write_text(json.dumps(stale_author))
                raise OSError('optional author context staging failed')
            return stage(path, content, **kwargs)
        monkeypatch.setattr(json_utils, '_stage_bytes', fail_author_stage)
    obstructed: list[Path] = []
    prior_rationale = (deep / 'stack-python-report-context.json').read_bytes()
    if unit in {"rationale-directory", "rationale-permission"}:
        unlink = Path.unlink

        def obstruct_optional_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
            if (path.name == 'stack-python-report-context.json'
                    and tuple(parent.name for parent in path.parents[:3]) == ('deep', '.daydream', 'live')
                    and not obstructed):
                obstructed.append(path)
                if unit == 'rationale-directory':
                    unlink(path, missing_ok=True)
                    path.mkdir()
                    (path / 'preserve.txt').write_text('user obstruction')
                else:
                    path.write_bytes(prior_rationale)
                    raise PermissionError('optional report cleanup denied')
            unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'unlink', obstruct_optional_cleanup)
    if unit == "arbiter":
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
    if author_case:
        assert len(author_faults) == 1
        context = json.loads((deep / 'review-report-context.json').read_text())
        assert context['author_reported'] is None
        assert json.loads((deep / 'review-author-context.json').read_text()) == stale_author
        assert 'Stale optional author assertion' not in (multi_stack_target / '.review-output.md').read_text()
    if unit.startswith('rationale-'):
        context = json.loads((deep / 'review-report-context.json').read_text())
        assert context['scopes']
        if obstructed:
            python = next(entry for entry in context['scopes'] if entry['scope_id'] == 'python')
            assert python['review_context'] is None, 'failed cleanup must not admit stale same-source rationale'
            preserved = deep / 'stack-python-report-context.json'
            if unit == 'rationale-directory':
                assert (preserved / 'preserve.txt').read_text() == 'user obstruction'
            else:
                assert preserved.read_bytes() == prior_rationale
        else:
            assert unit not in {'rationale-directory', 'rationale-permission'}, 'cleanup fault was not reached'
            assert all(entry['review_context'] is None for entry in context['scopes'])
        assert all(entry['originating_run_id'] == first['run_id'] for entry in context['scopes'])
    units = cast(dict[str, Any], _latest_provenance(deep)["units"])
    if unit == "arbiter":
        for name, grounding in (("intent", {"exploration"}), ("alternatives", {"intent", "exploration"})):
            assert units[name]["outcome"] == "hit" and set(units[name]["grounding_status"]) == grounding
    if unit == "arbiter":
        assert units["arbiter"]["outcome"] == "hit"
        assert set(units["arbiter"]["grounding_status"]) == {"intent", "alternatives", "exploration"}
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


async def test_obsolete_completed_synthesis_is_replaced_and_current_output_reused(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """Old host-appended duplicates must miss even when all contributing bytes match.

    The identical-rerun test proves reuse within one contract; it cannot catch a
    completed result produced before synthesis became authoritative. Keep the
    old entry valid and complete so only the changed synthesis policy rejects it.
    """
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    structural_description = "A cache hit retains the previous snapshot's boundary decision"
    stub.parse_by_stack = {"structure": {
        "description": structural_description, "severity": "medium", "confidence": "MEDIUM",
    }}
    authored = _merge_item(1, "api.py", "medium", desc="Reused state crosses the request ownership boundary")
    authored["source_uids"] = ["python:1", "structure:1"]
    stub.merge_items = [authored]
    config = make_config(multi_stack_target, shallow_fanout_threshold=0)
    assert await run(config) == 0
    deep = multi_stack_target / ".daydream" / "deep"
    assert all(scope["status"] == "complete" for scope in saved_coverage(deep).scopes.values())
    records = {name: data for name, data in _records_bytes(multi_stack_target).items()
               if name.startswith("stack-")}
    structural = json.loads(records["stack-structure-records.json"])["issues"][0]
    old_items = [dict(authored, item_uid="item:1"),
                 dict(structural, id=2, item_uid="item:2", lens="structural", source_uids=[structural["uid"]])]
    old_payload = {
        "merged-items.json": json.dumps({"items": old_items}, indent=2).encode(),
        "review-output.md": render_report(old_items).encode(),
    }
    entries = deep.parent / "review-cache" / "entries"
    manifest_path = next(path for path in entries.glob("*/manifest.json")
                         if json.loads(path.read_text())["unit"] == "merge")
    manifest = json.loads(manifest_path.read_text())
    manifest["components"].pop("synthesis_policy", None)
    old_key = hashlib.sha256(json.dumps({
        "format": manifest["format"], "unit": manifest["unit"], "components": manifest["components"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest["key"] = old_key
    for name, content in old_payload.items():
        (manifest_path.parent / name).write_bytes(content)
        manifest["payload"][name] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    old_entry = entries / old_key
    if manifest_path.parent != old_entry:
        manifest_path.parent.rename(old_entry)
    assert (old_entry / "complete.marker").is_file()
    assert manifest["coverage"]["status"] == "complete"

    stub.calls.clear()
    assert await run(config) == 0
    assert _count_merge_prompts(stub.calls) == 1, "obsolete completed synthesis was restored"
    assert {name: (deep / name).read_bytes() for name in records} == records
    current = json.loads((deep / "merged-items.json").read_text())["items"]
    assert [item["description"] for item in current] == [authored["description"]]
    assert structural_description not in (multi_stack_target / ".review-output.md").read_text()
    canonical = (deep / "merged-items.json").read_bytes()
    assert cast(dict[str, Any], _latest_provenance(deep)["units"])["merge"]["outcome"] != "hit"

    stub.calls.clear()
    assert await run(config) == 0
    assert _review_surface_prompts(stub.calls) == []
    assert (deep / "merged-items.json").read_bytes() == canonical
    assert cast(dict[str, Any], _latest_provenance(deep)["units"])["merge"]["outcome"] == "hit"

    # An explicit resume must consume saved reviewer records and overwrite the
    # obsolete canonical projection rather than resurrect either cache result.
    for name, content in old_payload.items():
        (deep / name).write_bytes(content)
    stub.calls.clear()
    assert await run(make_config(multi_stack_target, start_at="merge", shallow_fanout_threshold=0)) == 0
    assert _count_merge_prompts(stub.calls) == 1
    assert _count_review_prompts(stub.calls) == 0
    assert {name: (deep / name).read_bytes() for name in records} == records
    assert (deep / "merged-items.json").read_bytes() == canonical
    assert structural_description not in (multi_stack_target / ".review-output.md").read_text()


async def test_precision_overlap_publishes_only_joined_whole_unit_and_restores_both_passes(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm precision run must restore the joined result without either adjudicator.

    Existing whole-arbiter reuse coverage has no suppression sibling, so an entry
    published as soon as arbitration finishes could silently restore unsuppressed records.
    """
    backend = PrecisionOverlapBackend(multi_stack_target, blocked=True)
    backend.fanout_concurrency = 4
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    config = empty_review_config(multi_stack_target, tmp_path / "cached-overlap.json", precision_mode=True,
        latency_profile="balanced", review_profile=independent_alternatives_profile(), review_cache_enabled=True)

    async def drive() -> None:
        assert await run(config) == 0

    with anyio.fail_after(60):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(drive)
            await backend.started["arbiter"].wait()
            await backend.started["suppression"].wait()
            live = backend.input_paths["suppression"].parent
            backend.release["arbiter"].set()
            await backend.closed["arbiter"].wait()
            await anyio.wait_all_tasks_blocked()
            assert not backend.closed["suppression"].is_set()
            assert not (live / "adjudication-complete.marker").exists()
            assert _entry_count_for_unit(live, "arbiter") == 0
            backend.release["suppression"].set()

    deep = multi_stack_target / ".daydream/deep"
    canonical = _records_bytes(multi_stack_target)
    completion = (deep / "adjudication-complete.marker").read_bytes()
    records = [row for name, content in canonical.items() if name.startswith("stack-")
               for row in json.loads(content)["issues"]]
    uids = {row["uid"] for row in records}
    assert {"python:1", "react:2"} <= uids
    assert not {"generic:1", "python:2", "react:1"} & uids
    assert _entry_count_for_unit(deep, "arbiter") == 1
    assert sorted(deep.glob("arbiter-group-*-complete.marker"))

    backend.calls.clear()
    backend.inputs.clear()
    assert await run(config) == 0
    assert backend.inputs == {}, "a whole-unit hit must skip both precision adjudicators"
    assert _count_arbiter_prompts(backend.calls) == 0
    assert _count_unit_prompts(backend.calls, "you are the suppression reviewer") == 0
    assert _records_bytes(multi_stack_target) == canonical
    assert (deep / "adjudication-complete.marker").read_bytes() == completion
    assert cast(dict[str, Any], _latest_provenance(deep)["units"])["arbiter"]["outcome"] == "hit"
    coverage = saved_coverage(deep)
    assert all(coverage.phases[phase]["status"] == "complete" for phase in ("arbiter", "suppression"))


@pytest.mark.parametrize('field', ['reviewer_investigation_budget_s', 'reviewer_tool_call_budget'])
async def test_configured_reviewer_policy_invalidates_reuse_and_records_resolved_identity(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, field: str,
) -> None:
    """A local-policy change must not serve a completed review under incompatible limits."""
    from tests.harness.stub_backend import review_stage_state

    stub = install_stub_backend(monkeypatch, multi_stack_target)
    profile_path = tmp_path / 'review-profile.toml'
    digests = []
    deep = multi_stack_target / '.daydream' / 'deep'
    for index, value in enumerate((480, 900, 900)):
        profile_path.write_text(f'[pipeline]\n{field} = {value}\n')
        resolved = resolve_profile(explicit_path=str(profile_path))
        trajectory_path = tmp_path / f'policy-{index}.json'
        stub.calls.clear()
        assert await run(make_config(multi_stack_target, review_profile_path=profile_path,
                                    trajectory_path=trajectory_path)) == 0
        extra = json.loads(trajectory_path.read_text())['extra']
        assert extra['profile_digest'] == resolved.digest
        assert extra['profile_source_kind'] == 'explicit'
        digests.append(extra['profile_digest'])
        reviewed_scopes = {stage['scope_id'] for call in stub.calls
                           if (stage := review_stage_state(call['prompt'])) is not None}
        if index < 2:
            assert reviewed_scopes == {'python', 'react', 'generic', 'structure'}
            assert _reused_stacks(deep) == set()
        else:
            assert reviewed_scopes == set()
            assert _reused_stacks(deep) == {'python', 'react', 'generic', 'structure'}
        assert saved_coverage(deep).unfinished_scopes == {}
    assert digests[0] != digests[1] == digests[2]


async def test_identical_rerun_reviews_no_stack_and_a_leaf_edit_invalidates_bound_coverage(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """An identical run reuses every shard; changed diff invalidates the bound coverage inventory."""
    stub = install_stub_backend(monkeypatch, shard_many_python_target)
    run_config = make_config(shard_many_python_target, deep_shard_enabled=True, deep_shard_max_files=1,
                             deep_shard_max_bytes=10**9)
    assert await run(run_config) == 0
    assert _count_review_prompts(stub.calls) >= 2  # a sharded first run reviews several shards
    first_records = _records_bytes(shard_many_python_target)  # {artifact name: bytes}
    stub.calls.clear()
    assert await run(run_config) == 0
    assert _count_review_prompts(stub.calls) == 0  # exact hit: no per-stack review calls
    assert _records_bytes(shard_many_python_target) == first_records  # restored byte-for-byte
    target = shard_many_python_target / "mod0.py"
    target.write_text("def f0():\n    return 'edited'\n")
    stub.calls.clear()
    assert await run(run_config) == 0
    assert _count_review_prompts(stub.calls) >= 2  # changed snapshot cannot reuse prior completion claims


async def test_automatic_plan_change_invalidates_whole_stack_reuse_at_unchanged_revision(
    sibling_frontier_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.deep_orchestrator.test_review_completion import ReviewRun, scopes

    review = ReviewRun(sibling_frontier_target, tmp_path, monkeypatch)
    assert await review.run(deep_shard_enabled=False, review_cache_enabled=True) == 0
    first = review.load()['terminal_result']
    assert {scope['scope_id'] for scope in first['stack_outcomes']} == {'python', 'structure'}
    deep = review.repo / '.daydream/deep'
    assert _entry_count_for_unit(deep, 'shard:python') == 1

    review.backend.calls.clear()
    assert await review.run(review_cache_enabled=True) == 0
    data = review.load()
    second = data['terminal_result']
    assert first['analyzed_revision'] == second['analyzed_revision']
    assert list(scopes(data)) == ['python#0', 'python#1', 'python#2', 'structure']
    assert _reviewed_stacks(review.backend.calls) == {'python#0', 'python#1', 'python#2'}
    assert _reused_stacks(deep) == set(), 'Structure also depends on the full planned-scope inventory'
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    records = _records_bytes(review.repo)
    review.backend.calls.clear()
    assert await review.run(review_cache_enabled=True) == 0
    assert _count_review_prompts(review.backend.calls) == 0
    assert _reused_stacks(deep) == set(scopes(data))
    assert _records_bytes(review.repo) == records


async def test_editing_a_recorded_frontier_file_invalidates_snapshot_bound_coverage(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """Frontier edits move the analyzed diff, invalidating every previous completion claim."""
    stub = install_stub_backend(monkeypatch, sibling_frontier_target)
    config = make_config(sibling_frontier_target, deep_shard_enabled=True, deep_shard_max_files=1,
                         deep_shard_max_bytes=10**9)
    assert await run(config) == 0
    deep = sibling_frontier_target / ".daydream" / "deep"
    scopes = {}
    for call in stub.calls:
        prompt = str(call["prompt"])
        scope = _PER_STACK_PROMPT.search(prompt)
        assigned = re.search(r"Assigned files: ([^\n]+)", prompt)
        if scope is None or assigned is None:
            continue
        frontier = re.search(r"review targets\): ([^\n]+)\.", prompt)
        scopes[scope.group(1)] = {"assigned_files": assigned.group(1).split(", "),
            "frontier_files": frontier.group(1).split(", ") if frontier else [],
        }
    assert (deep / f"stack-{STRUCTURE_STACK_NAME}-records.json").is_file()
    scopes[STRUCTURE_STACK_NAME] = {"assigned_files": [], "frontier_files": []}
    namers = {name for name, rec in scopes.items() if "core.py" in rec["frontier_files"]}
    assert namers, "fixture must ground at least one shard with core.py"
    assert all("core.py" not in scopes[name]["assigned_files"] for name in namers), (
        "a frontier namer must not own the edited file, or its miss would not prove the frontier is keyed"
    )
    owner = {name
        for name, rec in scopes.items()
        if "core.py" in rec["assigned_files"] and name != STRUCTURE_STACK_NAME
    }
    (sibling_frontier_target / "core.py").write_text("SHARED = 'edited'\n")
    stub.calls.clear()
    assert await run(config) == 0
    assert namers | owner
    expected_miss = set(scopes) - {STRUCTURE_STACK_NAME}
    assert _reviewed_stacks(stub.calls) == expected_miss
    assert _reused_stacks(deep) == set()
    # The captured full diff participates in coverage-bound keys, including
    # the structural meta-stack whose completion also belongs to this snapshot.
    for name in namers:
        assert _entry_count_for_unit(deep, f"shard:{name}") == 2
    assert _entry_count_for_unit(deep, f"shard:{STRUCTURE_STACK_NAME}") == 2


_ARBITER_DISCRIMINATOR = "you are the arbiter"


def _count_arbiter_prompts(calls: list[dict[str, object]]) -> int:
    """How many captured calls carried the production arbiter prompt."""
    return _count_unit_prompts(calls, _ARBITER_DISCRIMINATOR)


async def test_fix_loop_commit_recomputes_every_snapshot_bound_scope(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A real Daydream fix commit moves HEAD and requires positive coverage of the new snapshot."""
    stub = install_stub_backend(monkeypatch, shard_many_python_target, enable_exploration=True)
    config = make_config(shard_many_python_target, deep_shard_enabled=True, deep_shard_max_files=1,
                         deep_shard_max_bytes=10**9)
    assert await run(config) == 0
    touched = shard_many_python_target / "mod0.py"
    touched.write_text("def f0():\n    return 'fixed'\n")
    git_ops.commit_paths(shard_many_python_target, [Path("mod0.py")],
                         build_commit_message(items=[{"file": "mod0.py", "description": "fix f0"}],
                                              run_id="fix-loop-1", version=daydream.__version__))
    # The aggregation units' subject is the contributing record *bytes* (A5), so
    # the gate only means something if the recompute changes them. The stub's
    # default finding is content-independent; make the recomputed shard emit a
    # HIGH, differently-worded finding so the merge and arbiter keys move.
    stub.parse_by_stack = {"python#0": {"severity": "high", "confidence": "HIGH",
                                        "file": "mod0.py", "line": 1,
                                        "description": "regression after fix"}}
    stub.calls.clear()
    with caplog.at_level(logging.INFO):
        assert await run(config) == 0
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Review reuse:")]
    assert summary and "0 hit" in summary[-1] and "shard:" in summary[-1]
    deep = shard_many_python_target / ".daydream" / "deep"
    assert len(_reviewed_stacks(stub.calls)) >= 2
    assert _count_arbiter_prompts(stub.calls) >= 1             # records changed -> arbiter recomputes
    assert _count_merge_prompts(stub.calls) >= 1               # merged set consumes records
    provenance = json.loads(reuse_store.provenance_path(
        shard_many_python_target / ".daydream" / "review-cache", _newest_provenance_path(deep).stem).read_text())
    reused = [k for k, v in provenance["units"].items() if k.startswith("shard:") and v["outcome"] == "hit"]
    assert not reused, "previous-head completion claims cannot establish current coverage"
    assert all(value["outcome"] == "miss" for name, value in provenance["units"].items() if name.startswith("shard:"))



async def test_no_review_cache_disables_the_store_and_bypasses_the_exploration_cache(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """--no-review-cache disables store reads/writes and pre-scan reuse, recording the bypass."""
    # The real pre-scan must be live for the bypass to be observable: the warm
    # run leaves a cache-key on disk, so a disabled run that still consulted the
    # pre-scan cache would emit no exploration specialist calls.
    stub = install_stub_backend(monkeypatch, multi_stack_target, enable_exploration=True)
    profile = independent_exploration_profile()
    assert await run(make_config(multi_stack_target, review_profile=profile)) == 0
    store = multi_stack_target / ".daydream" / "review-cache"
    entries_before = sorted(p.name for p in (store / "entries").iterdir())
    assert entries_before, "the warm run must have populated the store"
    stub.calls.clear()
    assert await run(make_config(multi_stack_target, review_profile=profile, review_cache_enabled=False)) == 0
    assert _count_review_prompts(stub.calls) > 0, "a disabled run must do the work"
    assert _count_unit_prompts(stub.calls, "dependency-tracer") > 0, (
        "--no-review-cache must bypass the exploration pre-scan cache and recompute it"
    )
    assert sorted(p.name for p in (store / "entries").iterdir()) == entries_before
    latest = _latest_provenance(multi_stack_target / ".daydream" / "deep")
    units = cast(dict[str, dict[str, object]], latest["units"])
    assert units["exploration"]["outcome"] == "disabled"
    assert any(v["outcome"] == "disabled" for v in units.values())


_ATTRIBUTION_DAMAGE = ["attribution-target", "attribution-path", "attribution-range", "attribution-part",
                       "attribution-artifact", "attribution-missing-target"]


def _damage_source_attribution(decisions: list[dict[str, Any]], damage: str) -> bool:
    """Corrupt frozen correspondence without relying on shape, revision, or hash rejection."""
    if damage == "attribution-missing-target":
        if len(decisions) < 2:
            return False
        decisions.pop()
        return True
    target = decisions[0]
    reference = target['evidence_refs'][0]
    if damage == "attribution-target":
        target['target_id'] = 'foreign-target.py'
    elif damage == "attribution-path":
        reference['path'] = 'foreign-source.py'
    elif damage == "attribution-range":
        reference['source_range'] = {'start': 999999, 'end': 999999}
        reference['fragment'] = None
    elif damage == "attribution-part":
        reference['part_sha256'] = 'f' * 64
    else:
        reference['artifact_sha256'] = 'f' * 64
    return True


async def test_cache_population_restores_warm_bytes_then_recomputes_only_independently_damaged_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry-local proof failures cannot invalidate an untouched sibling or establish false cache coverage."""
    from tests.deep_orchestrator.test_review_investigation import InvestigationRun
    from tests.harness.git_helpers import seed_feature_branch
    from tests.harness.stub_backend import review_stage_state

    damages = [
        "legacy",
        "corrupt",
        "incomplete",
        "head",
        "diff",
        "scope",
        "origin",
        "partial_payload",
        "malformed_payload",
        "unstaged",
        "prior-staged",
        "missing-attribution",
        "invalid-attribution",
        *_ATTRIBUTION_DAMAGE,
    ]
    repo = tmp_path / "cache_population"
    before = {f"{name}/module_{index}.py": "VALUE = 0\n" for name in [*damages, "healthy"] for index in range(2)}
    after = {path: "VALUE = 1\n" for path in before}
    after["healthy/module_0.py"] = "VALUE = '" + "界" * 11000 + "'\n"
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    config = dict(
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
    for name, fault in [("legacy", "missing"), ("corrupt", "corrupt"), ("incomplete", "nested")]:
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
        if damage in {"unstaged", "prior-staged"}:
            if damage == "unstaged":
                manifest["components"].pop("staged_review_contract")
            else:
                manifest["components"]["staged_review_contract"] = 11
            key = hashlib.sha256(
                json.dumps(
                    {"format": manifest["format"], "unit": manifest["unit"], "components": manifest["components"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            manifest["key"] = key
        elif damage == "legacy":
            manifest.pop("coverage")
        elif damage == "corrupt":
            manifest["coverage"] = "invalid proof"
        elif damage == "incomplete":
            manifest["coverage"]["status"] = "incomplete"
        elif damage == "origin":
            manifest["origin"] = "malformed lineage"
        elif damage in {"head", "diff"}:
            manifest["coverage"]["analyzed_revision"]["head_sha" if damage == "head" else "diff_key"] = "f" * (
                40 if damage == "head" else 64
            )
        elif damage == "scope":
            manifest["coverage"]["planned_scopes"][0]["files"] = ["foreign.py"]
        else:
            attribution = damage not in {"partial_payload", "malformed_payload"}
            suffix = "-records-attribution.json" if attribution else "-records.json"
            payload_path = next(manifest_path.parent / name for name in manifest["payload"] if name.endswith(suffix))
            payload = json.loads(payload_path.read_text())
            if damage == "missing-attribution":
                manifest["payload"].pop(payload_path.name)
            else:
                if damage == "invalid-attribution":
                    payload["target_decisions"][0]["evidence_refs"][0]["revision"] = "0" * 40
                elif damage in _ATTRIBUTION_DAMAGE:
                    assert _damage_source_attribution(payload["target_decisions"], damage)
                elif damage == "partial_payload":
                    payload["incomplete"] = True
                else:
                    payload["issues"] = ["invalid authored finding"]
                payload_path.write_text(json.dumps(payload))
                manifest["payload"][payload_path.name] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        if damage in {"unstaged", "prior-staged"}:
            manifest_path.parent.rename(entries / manifest["key"])
    review.backend.calls.clear()
    assert await review.run(**config) == 0
    actual = {
        stage["scope_id"] for call in review.backend.calls if (stage := review_stage_state(call["prompt"])) is not None
    }
    assert actual == {by_name[name] for name in damages}
    assert _reused_stacks(deep) == {by_name["healthy"], "structure"}
    current = saved_coverage(deep)
    assert all(scope["status"] == "complete" for scope in current.scopes.values())
    assert {scope: outcome["target_decisions"] for scope, outcome in current.scopes.items()} == original
    assert unknown.is_dir()
    review.backend.calls.clear()
    assert await review.run(start_at="merge", **config) == 0
    assert _count_review_prompts(review.backend.calls) == 0
    assert {scope: outcome["target_decisions"] for scope, outcome in saved_coverage(deep).scopes.items()} == original


async def test_aggregate_cached_attribution_regains_all_frozen_references_within_the_published_bound(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """The shared publication budget remains independent of entry-local corruption."""
    run_limit = 256 * 1024
    monkeypatch.setattr('daydream.review_result.RUN_TARGET_DECISIONS_MAX_BYTES', run_limit)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    config = make_config(multi_stack_target)
    assert await run(config) == 0
    deep = multi_stack_target / ".daydream/deep"
    original = {scope: outcome["target_decisions"] for scope, outcome in saved_coverage(deep).scopes.items()}
    for manifest_path in (deep.parent / "review-cache/entries").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        for name in manifest["payload"]:
            if not name.endswith("-records-attribution.json") or manifest["unit"] == "shard:structure":
                continue
            path = manifest_path.parent / name
            payload = json.loads(path.read_text())
            payload["target_decisions"][0]["reason"] = "r" * 90000
            path.write_text(json.dumps(payload))
            manifest["payload"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    stub.calls.clear()
    assert await run(config) == 0 and _count_review_prompts(stub.calls) > 0
    coverage = saved_coverage(deep).to_dict()
    for outcome in coverage["stack_outcomes"]:
        decisions = [
            {key: value for key, value in decision.items() if key != "reason"}
            for decision in outcome["target_decisions"]
        ]
        expected = [
            {key: value for key, value in decision.items() if key != "reason"}
            for decision in original[outcome["scope_id"]]
        ]
        assert decisions == expected
    core = copy.deepcopy(coverage)
    for outcome in core["stack_outcomes"]:
        outcome.pop("target_decisions", None)

    def size(value: dict[str, Any]) -> int:
        return len(json.dumps({"terminal_result": value}, indent=2, ensure_ascii=False).encode())

    assert size(coverage) - size(core) <= run_limit


@pytest.mark.parametrize("damage", ["missing", "corrupt", "head", "diff", "scope", "phase",
                                    "missing-attribution", "invalid-attribution", "aggregate-attribution",
                                    "attribution-path"])
async def test_resume_rejects_missing_or_mismatched_coverage(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, damage: str, tmp_path: Path,
) -> None:
    """Resume requires a valid same-snapshot inventory; bytes alone are insufficient."""
    from tests.test_deep_orchestrator import _pin_findings_pr

    if damage == 'aggregate-attribution':
        monkeypatch.setattr('daydream.review_result.RUN_TARGET_DECISIONS_MAX_BYTES', 256 * 1024)
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    initial = tmp_path / "initial-review.json"
    assert await run(make_config(multi_stack_target, findings_out=str(initial), pr_number=pr.number)) == 0
    path = multi_stack_target / ".daydream" / "deep" / "review-coverage.json"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{broken")
    else:
        evidence = json.loads(path.read_text())
        if damage in {"head", "diff"}:
            field = "head_sha" if damage == "head" else "diff_key"
            evidence["analyzed_revision"][field] = "f" * (40 if damage == "head" else 64)
        elif damage == "scope":
            evidence["planned_scopes"][0]["files"] = ["changed-assignment.py"]
        elif damage == 'missing-attribution':
            evidence['stack_outcomes'][0].pop('target_decisions')
        elif damage == 'invalid-attribution':
            evidence['stack_outcomes'][0]['target_decisions'][0]['evidence_refs'][0]['revision'] = '0' * 40
        elif damage in _ATTRIBUTION_DAMAGE:
            damaged = [_damage_source_attribution(scope['target_decisions'], damage)
                       for scope in evidence['stack_outcomes']]
            assert any(damaged), 'corruption must reach a retained source assertion'
        elif damage == 'aggregate-attribution':
            for scope in evidence['stack_outcomes']:
                if scope['scope_id'] != 'structure':
                    scope['target_decisions'][0]['reason'] = 'r' * 90000
        else:
            evidence["required_phases"].append("unknown-stage")
        path.write_text(json.dumps(evidence))
    stub.calls.clear()
    output = tmp_path / "resume-review.json"
    assert await run(make_config(multi_stack_target, start_at="merge", findings_out=str(output),
                                pr_number=pr.number)) == 1
    assert _review_surface_prompts(stub.calls) == []
    assert not output.exists(), "rejected resume cannot manufacture version 2 coverage from unproven evidence"
    assert initial.is_file()


@pytest.mark.parametrize("recovery", ["resume", "cache"])
@pytest.mark.parametrize("scope", ["python", "structure"])
async def test_successful_scope_rerun_clears_prior_failure_for_same_revision(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recovery: str, scope: str,
) -> None:
    """Unfinished scopes require successful investigation before resume or cache reuse."""
    from daydream.backends import AgentEvent, ResultEvent
    from tests.deep_orchestrator.test_review_completion import ReviewRun
    from tests.harness.git_helpers import git
    from tests.harness.stub_backend import review_stage_result, review_stage_state

    review = ReviewRun(multi_stack_target, tmp_path, monkeypatch)
    backend = review.backend

    source = {name: (review.repo / name).read_bytes() for name in git(review.repo, "ls-files").splitlines()}

    def unfinished_scope(prompt: str) -> list[AgentEvent] | None:
        stage = review_stage_state(prompt)
        if stage is None or stage["scope_id"] != scope:
            return None
        result = review_stage_result(prompt, [])
        for target in result["targets"]:
            target.update(status="not_reviewed", reason="Dependency conclusion is unfinished")
        return [ResultEvent(structured_output=result, continuation=None)]

    if recovery == "cache":
        backend.responder = unfinished_scope
    else:
        backend.fail_stack = scope
    options: dict[str, Any] = {"review_cache_enabled": recovery == "cache"}
    assert await review.run(**options) == 0
    first_data = review.load()
    first = first_data["terminal_result"]
    assert first_data["findings"] == []
    deep = multi_stack_target / ".daydream/deep"
    persisted = saved_coverage(deep)
    assert persisted.to_dict()["phase_outcomes"] == first["phase_outcomes"]
    if scope == "structure":
        alternatives = persisted.phases["alternatives"]
        assert alternatives["status"] == ("incomplete" if recovery == "cache" else "failed")
        assert alternatives["reason_codes"] == persisted.scopes[scope]["reason_codes"]
        assert alternatives["noop"] is alternatives["usable_evidence"] is False
        if recovery == "cache":
            assert persisted.scopes[scope]["partial_evidence"] is True
        assert "folded into Structure" in persisted.diagnostics["phases"]["alternatives"]
        assert persisted.diagnostics["scopes"][scope] in persisted.diagnostics["phases"]["alternatives"]
    assert first["analysis_state"] == "incomplete"
    report = (review.repo / '.review-output.md').read_text()
    assert '<code>incomplete</code>' in report and first['run_id'] in report
    assert first['analyzed_revision']['head_sha'] in report
    first_origins = {entry['scope_id']: entry['originating_run_id'] for entry in first_data['review_context']['scopes']}

    assert first["failed_stacks"] == ([] if recovery == "cache" else [scope])
    if recovery == "cache":
        assert saved_coverage(deep).unfinished_scopes.keys() == {scope}
        assert _entry_count_for_unit(deep, f"shard:{scope}") == 0
        sibling = "structure" if scope == "python" else "python"
        assert _entry_count_for_unit(deep, f"shard:{sibling}") == 1
    else:
        backend.calls.clear()
        assert await review.run(**options, start_at="merge") == 0
        resumed = review.load()
        assert resumed["findings"] == first_data["findings"]
        assert resumed['review_context']['scopes'] == first_data['review_context']['scopes']
        report = (review.repo / '.review-output.md').read_text()
        assert '<code>incomplete</code>' in report and resumed['terminal_result']['run_id'] in report

        for field in ("analysis_state", "failed_stacks", "phase_outcomes", "stack_outcomes"):
            assert resumed["terminal_result"][field] == first[field]
        assert saved_coverage(deep).diagnostics == persisted.diagnostics
        assert _review_surface_prompts(backend.calls) == []
        assert not any(review_stage_state(call["prompt"]) is not None for call in backend.calls)
        options["start_at"] = "per-stack"
    backend.fail_stack = backend.responder = None
    backend.calls.clear()
    assert await review.run(**options) == 0
    second_data = review.load()
    second = second_data["terminal_result"]
    assert second_data["findings"] == first_data["findings"]
    current = saved_coverage(deep)
    assert current.to_dict()["phase_outcomes"] == second["phase_outcomes"]
    assert current.phases["alternatives"]["status"] == "complete"
    assert current.phases["alternatives"]["noop"] is True
    assert current.phases["alternatives"]["reason_codes"] == []
    assert "alternatives" not in current.diagnostics["phases"]
    assert _count_unit_prompts(backend.calls, _WONDER_DISCRIMINATOR) == 0
    if recovery == "resume":
        assert {stage["scope_id"] for call in backend.calls
                if (stage := review_stage_state(call["prompt"])) is not None} == {
                    "python", "react", "generic", "structure"}
    assert {name: (review.repo / name).read_bytes() for name in source} == source
    assert second["analysis_state"] == "complete" and second["failed_stacks"] == []
    report = (review.repo / '.review-output.md').read_text()
    assert '<code>complete</code>' in report and second['run_id'] in report
    assert report.count('## Review outcome') == 1
    assert 'Assigned changed behavior reviewed' in report
    if recovery == 'cache':
        second_origins = {entry['scope_id']: entry['originating_run_id']
                          for entry in second_data['review_context']['scopes']}
        assert second_origins[sibling] == first_origins[sibling]

    assert "backend_failure" not in second["reason_codes"]
    assert first["analyzed_revision"] == second["analyzed_revision"]
    if recovery == "cache":
        stages = [stage for call in backend.calls
                  if (stage := review_stage_state(call["prompt"])) is not None]
        assert {stage["scope_id"] for stage in stages} == {scope}
        assert _entry_count_for_unit(deep, f"shard:{scope}") == 1
        assert saved_coverage(deep).unfinished_scopes == {}
        backend.calls.clear()
        assert await review.run(**options) == 0
        assert _count_review_prompts(backend.calls) == 0
        assert _reused_stacks(deep) == {"python", "react", "generic", "structure"}
        assert saved_coverage(deep).unfinished_scopes == {}
