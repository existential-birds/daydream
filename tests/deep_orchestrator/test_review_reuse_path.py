"""Real runner tests of cache publication and survival across fresh runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, cast

import pytest

import daydream
from daydream import git_ops
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep import reuse_store
from daydream.phases import build_commit_message
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _arbiter_stacks,
    _count_merge_prompts,
    _count_review_prompts,
)
from tests.harness.review_profile import (
    independent_alternatives_profile,
    independent_exploration_profile,
)
from tests.harness.review_result import saved_coverage
from tests.harness.stub_backend import install_stub_backend
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


@pytest.mark.parametrize("unit", ["all-units", "independent", "arbiter"])
async def test_identical_rerun_restores_completed_units_and_current_run_evidence(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, unit: str,
) -> None:
    """Every warm contract retains exact bytes, positive evidence, and no paid review calls."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    options: dict[str, Any] = {}
    if unit in {"independent", "arbiter"}:
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
    if unit == "independent":
        assert _count_unit_prompts(stub.calls, _INTENT_DISCRIMINATOR) == 1
        assert _count_unit_prompts(stub.calls, _WONDER_DISCRIMINATOR) == 1
    elif unit == "arbiter":
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
    units = cast(dict[str, Any], _latest_provenance(deep)["units"])
    if unit == "independent":
        for name, grounding in (("intent", {"exploration"}), ("alternatives", {"intent", "exploration"})):
            assert units[name]["outcome"] == "hit" and set(units[name]["grounding_status"]) == grounding
    elif unit == "arbiter":
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

async def test_store_directory_survives_a_fresh_run_and_is_readable_by_the_next(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """Published review-cache survives detachment and reseeding of the next live root."""
    install_stub_backend(monkeypatch, multi_stack_target)
    assert await run(make_config(multi_stack_target)) == 0
    store = multi_stack_target / ".daydream" / "review-cache"
    assert store.is_dir(), "store must be published beside .daydream/deep"
    provenance = multi_stack_target / ".daydream" / "review-cache" / "provenance"
    records = list(provenance.glob("*.json"))
    assert records, "the run must record its reuse provenance inside the store"
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["units"]["exploration"]["outcome"] in {"reused", "regenerated"}

    (store / "entries").mkdir(exist_ok=True)
    (store / "entries" / ("a" * 64)).mkdir()
    assert await run(make_config(multi_stack_target)) == 0
    assert (store / "entries" / ("a" * 64)).is_dir(), "a fresh run must not wipe the store"


async def test_identical_rerun_reviews_no_stack_and_a_leaf_edit_invalidates_bound_coverage(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """An identical run reuses every shard; changed diff invalidates the bound coverage inventory."""
    stub = install_stub_backend(monkeypatch, shard_many_python_target)
    run_config = make_config(shard_many_python_target, deep_shard_enabled=True, deep_shard_max_files=1,
                             deep_shard_max_bytes=10**9)
    assert await run(run_config) == 0
    assert _count_review_prompts(stub.calls) >= 2          # a sharded first run reviews several shards
    first_records = _records_bytes(shard_many_python_target)   # {artifact name: bytes}
    stub.calls.clear()
    assert await run(run_config) == 0
    assert _count_review_prompts(stub.calls) == 0          # exact hit: no per-stack review calls
    assert _records_bytes(shard_many_python_target) == first_records   # restored byte-for-byte
    target = shard_many_python_target / "mod0.py"
    target.write_text("def f0():\n    return 'edited'\n")
    stub.calls.clear()
    assert await run(run_config) == 0
    assert _count_review_prompts(stub.calls) >= 2  # changed snapshot cannot reuse prior completion claims


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
    assert await run(config) == 0
    deep = shard_many_python_target / ".daydream" / "deep"
    assert len(_reviewed_stacks(stub.calls)) >= 2
    assert _count_arbiter_prompts(stub.calls) >= 1             # records changed -> arbiter recomputes
    assert _count_merge_prompts(stub.calls) >= 1               # merged set consumes records
    provenance = json.loads(reuse_store.provenance_path(
        shard_many_python_target / ".daydream" / "review-cache", _newest_provenance_path(deep).stem).read_text())
    reused = [k for k, v in provenance["units"].items() if k.startswith("shard:") and v["outcome"] == "hit"]
    assert not reused, "previous-head completion claims cannot establish current coverage"
    assert all(value["outcome"] == "miss" for name, value in provenance["units"].items() if name.startswith("shard:"))

async def test_run_reports_snapshot_bound_reuse_misses_after_head_moves(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    install_stub_backend(monkeypatch, shard_many_python_target, enable_exploration=True)
    config = make_config(shard_many_python_target, deep_shard_enabled=True, deep_shard_max_files=1,
                         deep_shard_max_bytes=10**9)
    assert await run(config) == 0
    (shard_many_python_target / "mod0.py").write_text("def f0():\n    return 'fixed'\n")
    git_ops.commit_paths(shard_many_python_target, [Path("mod0.py")],
                         build_commit_message(items=[], run_id="fix-loop-2", version=daydream.__version__))
    with caplog.at_level(logging.INFO):
        assert await run(config) == 0
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Review reuse:")]
    assert summary, "the run must report its reuse outcome"
    assert "0 hit" in summary[-1] and "shard:" in summary[-1]


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


@pytest.mark.parametrize("damage", ["legacy", "corrupt", "incomplete", "head", "diff", "scope", "origin",
                                     "partial_payload", "malformed_payload", "unstaged", "prior-staged",
                                     "contract-3", "contract-4", "contract-5", "contract-6"])
async def test_legacy_cache_without_coverage_proof_recomputes_review(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    damage: str,
) -> None:
    """Pre-v2 completed bytes cannot establish complete current review coverage."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    config = make_config(multi_stack_target)
    assert await run(config) == 0
    entries = multi_stack_target / ".daydream" / "review-cache" / "entries"
    downgraded = 0
    for manifest_path in entries.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if damage in {"unstaged", "prior-staged", "contract-3", "contract-4", "contract-5", "contract-6"}:
            if not manifest["unit"].startswith("shard:"):
                continue
            if damage == "unstaged":
                manifest["components"].pop("staged_review_contract")
            else:
                manifest["components"]["staged_review_contract"] = {
                    "contract-3": 3, "contract-4": 4, "contract-5": 5, "contract-6": 6,
                }.get(damage, 2)
                assert manifest['coverage']['status'] == 'complete'
            legacy_key = hashlib.sha256(json.dumps({
                "format": manifest["format"], "unit": manifest["unit"], "components": manifest["components"],
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest["key"] = legacy_key
            manifest_path.write_text(json.dumps(manifest))
            manifest_path.parent.rename(entries / legacy_key)
            downgraded += 1
            continue
        if damage == "legacy":
            manifest.pop("coverage", None)
        elif damage == "corrupt":
            manifest["coverage"] = "invalid proof"
        elif damage == "incomplete":
            manifest["coverage"]["status"] = "incomplete"
        elif damage == "origin":
            manifest["origin"] = "malformed lineage"
        elif damage in {"partial_payload", "malformed_payload"}:
            for name in manifest["payload"]:
                if not name.startswith("stack-") or not name.endswith("-records.json"):
                    continue
                payload_path = manifest_path.parent / name
                payload = json.loads(payload_path.read_text())
                if damage == "partial_payload":
                    payload["incomplete"] = True
                else:
                    payload["issues"] = ["invalid authored finding"]
                payload_path.write_text(json.dumps(payload))
                manifest["payload"][name] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        elif damage in {"head", "diff"}:
            field = "head_sha" if damage == "head" else "diff_key"
            manifest["coverage"]["analyzed_revision"][field] = "f" * (40 if damage == "head" else 64)
        else:
            manifest["coverage"]["planned_scopes"][0]["files"] = ["foreign.py"]
        manifest_path.write_text(json.dumps(manifest))
    assert damage not in {"unstaged", "prior-staged", "contract-3", "contract-4",
                          "contract-5", "contract-6"} or downgraded > 0
    stub.calls.clear()
    assert await run(config) == 0
    assert _count_review_prompts(stub.calls) > 0
    if damage not in {"partial_payload", "malformed_payload", "unstaged", "prior-staged",
                      "contract-3", "contract-4", "contract-5", "contract-6"}:
        assert _count_merge_prompts(stub.calls) > 0
    if damage in {"unstaged", "prior-staged", "contract-3", "contract-4", "contract-5", "contract-6"}:
        assert _reused_stacks(multi_stack_target / ".daydream/deep") == set()
    coverage = json.loads((multi_stack_target / ".daydream" / "deep" / "review-coverage.json").read_text())
    assert all(outcome["status"] == "complete" for outcome in coverage["stack_outcomes"])


@pytest.mark.parametrize("damage", ["missing", "corrupt", "head", "diff", "scope", "phase"])
async def test_resume_rejects_missing_or_mismatched_coverage(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, damage: str, tmp_path: Path,
) -> None:
    """Resume requires a valid same-snapshot inventory; bytes alone are insufficient."""
    from tests.test_deep_orchestrator import _pin_findings_pr

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
async def test_successful_scope_rerun_clears_prior_failure_for_same_revision(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recovery: str,
) -> None:
    """Unfinished scopes require successful investigation before resume or cache reuse."""
    from daydream.backends import AgentEvent, ResultEvent
    from tests.deep_orchestrator.test_review_completion import ReviewRun
    from tests.harness.stub_backend import review_stage_result

    review = ReviewRun(multi_stack_target, tmp_path, monkeypatch)
    backend = review.backend

    def unfinished_python(prompt: str) -> list[AgentEvent] | None:
        if "python stack" not in prompt.lower():
            return None
        result = review_stage_result(prompt, [])
        for target in result["targets"]:
            target.update(status="not_reviewed", reason="Dependency conclusion is unfinished")
        return [ResultEvent(structured_output=result, continuation=None)]

    if recovery == "cache":
        backend.responder = unfinished_python
    else:
        backend.fail_stack = "python"
    options: dict[str, Any] = {"review_cache_enabled": recovery == "cache"}
    assert await review.run(**options) == 0
    first = review.load()["terminal_result"]
    deep = multi_stack_target / ".daydream/deep"
    assert first["analysis_state"] == "incomplete"
    assert first["failed_stacks"] == ([] if recovery == "cache" else ["python"])
    if recovery == "cache":
        assert saved_coverage(deep).unfinished_scopes.keys() == {"python"}
        assert _entry_count_for_unit(deep, "shard:python") == 0
        assert _entry_count_for_unit(deep, "shard:structure") == 1
    else:
        options["start_at"] = "per-stack"
    backend.fail_stack = backend.responder = None
    backend.calls.clear()
    assert await review.run(**options) == 0
    second = review.load()["terminal_result"]
    assert second["analysis_state"] == "complete" and second["failed_stacks"] == []
    assert "backend_failure" not in second["reason_codes"]
    assert first["analyzed_revision"] == second["analyzed_revision"]
    if recovery == "cache":
        assert _reviewed_stacks(backend.calls) == {"python"}
        assert _entry_count_for_unit(deep, "shard:python") == 1
        assert saved_coverage(deep).unfinished_scopes == {}
        backend.calls.clear()
        assert await review.run(**options) == 0
        assert _count_review_prompts(backend.calls) == 0
        assert _reused_stacks(deep) == {"python", "react", "generic", "structure"}
        assert saved_coverage(deep).unfinished_scopes == {}
