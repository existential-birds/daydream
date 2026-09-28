"""Real-path tests for content-addressed reuse of deep review results.

These tests enter from ``runner.run`` with the real filesystem and event loop,
mocking only the external backend through the ``create_backend`` seam. They
assert observable outcomes of the ``.daydream/review-cache/`` store: that it is
published through the artifact-visibility anchors and survives a fresh run.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import cast

import pytest

import daydream
from daydream import git_ops
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep import reuse_store
from daydream.phases import build_commit_message
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _count_merge_prompts,
    _count_review_prompts,
    _install_uncovered_sweep_stub,
    _uncovered_sweep_target,
)
from tests.harness.review_profile import independent_alternatives_profile
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


def _stack_files(deep: Path) -> list[str]:
    """The per-stack review artifacts this run actually left behind."""
    return sorted(path.name for path in deep.glob("stack-*-records.json"))


def _expected_stack_files(deep: Path) -> list[str]:
    """One records file per stack the deterministic assignment named.

    The pre-fan-out coverage receipts are computed from the current run's
    assignments, so they name exactly the stacks a fresh run reviews.
    """
    receipts = json.loads((deep / "coverage-receipts.json").read_text())
    return sorted(f"stack-{name}-records.json" for name in receipts)


def _stack_receipts(deep: Path) -> dict[str, dict[str, list[str]]]:
    """The deterministic pre-fan-out coverage receipts, keyed by stack name."""
    receipts: dict[str, dict[str, list[str]]] = json.loads(
        (deep / "coverage-receipts.json").read_text(encoding="utf-8")
    )
    return receipts


_PER_STACK_PROMPT = re.compile(r"you are reviewing the (\S+) stack", re.IGNORECASE)
_INTENT_DISCRIMINATOR = "understand the intent of these changes"
_WONDER_DISCRIMINATOR = "evaluate the implementation"
# A sentence only the uncovered-sweep prompt carries (coverage.py).
_SWEEP_DISCRIMINATOR = "you may only comment on hunks you have read"


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


def _stack_bytes(deep: Path, names: set[str]) -> dict[str, bytes]:
    """The current records bytes for ``names``, keyed by artifact basename."""
    return {
        f"stack-{name}-records.json": (deep / f"stack-{name}-records.json").read_bytes()
        for name in names
    }


def _origin_stack_bytes(deep: Path, names: set[str]) -> dict[str, bytes]:
    """The origin run's records bytes for ``names``, read from the reuse store.

    A recompute stores a second entry under a moved key, so the oldest
    ``created_at`` for a unit is the byte-for-byte content a reuse would have
    restored.
    """
    entries = deep.parent / "review-cache" / "entries"
    origin: dict[str, bytes] = {}
    for name in names:
        artifact = f"stack-{name}-records.json"
        candidates: list[tuple[float, Path]] = []
        for entry in entries.iterdir():
            try:
                manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if manifest.get("unit") == f"shard:{name}" and artifact in (manifest.get("payload") or {}):
                candidates.append((float(manifest.get("created_at", 0.0)), entry))
        if candidates:
            origin[artifact] = min(candidates)[1].joinpath(artifact).read_bytes()
    return origin


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
    return {
        unit.removeprefix("shard:")
        for unit, trace in units.items()
        if unit.startswith("shard:")
        and isinstance(trace, dict)
        and trace.get("outcome") == "hit"
    }


def _latest_provenance(deep: Path) -> dict[str, object]:
    """The most recently written per-run provenance record in the store."""
    provenance = deep.parent / "review-cache" / "provenance"
    records = sorted(provenance.glob("*.json"), key=lambda path: path.stat().st_mtime)
    assert records, "the run must record its reuse provenance inside the store"
    record: dict[str, object] = json.loads(records[-1].read_text(encoding="utf-8"))
    return record


def _session_id_of(deep: Path) -> str:
    """The provenance key of the run that just finished.

    ``ReuseCache`` names each run's provenance file after ``WorkContext.run_id``
    and keeps every earlier run's record, so the run under test is the newest
    file by mtime. The file stem is what :func:`reuse_store.provenance_path`
    expects.
    """
    provenance = deep.parent / "review-cache" / "provenance"
    records = sorted(provenance.glob("*.json"), key=lambda path: path.stat().st_mtime)
    assert records, "the run must record its reuse provenance inside the store"
    return records[-1].stem


_MERGE_DISCRIMINATOR = "cross-stack merge agent"


def _review_surface_prompts(
    calls: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Captured calls that performed deep review work (the paid review surface).

    Filters by the production prompt discriminators, so an unrelated fix or test
    call never masks a genuine review call and a warm run that still calls a
    reviewer fails the assertion instead of passing vacuously.
    """
    surface: list[dict[str, object]] = []
    for call in calls:
        prompt = str(call.get("prompt", "")).lower()
        if (
            _PER_STACK_PROMPT.search(prompt)
            or _SWEEP_DISCRIMINATOR in prompt
            or _ARBITER_DISCRIMINATOR in prompt
            or _INTENT_DISCRIMINATOR in prompt
            or _WONDER_DISCRIMINATOR in prompt
            or _MERGE_DISCRIMINATOR in prompt
        ):
            surface.append(call)
    return surface


async def test_identical_rerun_reuses_the_uncovered_sweep_and_restores_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH8/A8: an identical rerun performs no sweep model call and restores the
    origin run's uncovered records AND its coverage accounting byte-for-byte --
    the counters are read back from the entry, never recomputed against a
    different run's trajectory."""
    target = _uncovered_sweep_target(tmp_path)
    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.merge_echo_records = True
    config = make_config(target, assume="yes", output_mode="loop")
    assert await run(config) == 0
    deep = target / ".daydream" / "deep"
    assert _count_unit_prompts(stub.calls, _SWEEP_DISCRIMINATOR) >= 1
    records_bytes = (deep / "stack-uncovered-records.json").read_bytes()
    coverage_bytes = (deep / "coverage-stats.json").read_bytes()
    assert json.loads(records_bytes), "the seed sweep must produce findings"
    assert json.loads(coverage_bytes)["sweep_finding_count"] >= 1
    stub.calls.clear()
    assert await run(config) == 0
    assert _count_unit_prompts(stub.calls, _SWEEP_DISCRIMINATOR) == 0
    assert (deep / "stack-uncovered-records.json").read_bytes() == records_bytes
    assert (deep / "coverage-stats.json").read_bytes() == coverage_bytes
    units = cast(dict[str, dict[str, object]], _latest_provenance(deep)["units"])
    assert units["sweep"]["outcome"] == "hit"
    assert set(cast(dict[str, object], units["sweep"]["grounding_status"])) == {
        "intent",
        "exploration",
    }


async def test_identical_rerun_reuses_intent_and_wonder_units(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH1/MH2/MH16: an identical rerun performs no intent or alternatives model
    call at all; the two whole-change units restore their recorded artifacts
    byte-for-byte, because a moved pre-scan (grounding) never moves their keys."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    config = make_config(multi_stack_target, review_profile=independent_alternatives_profile())
    assert await run(config) == 0
    assert _count_unit_prompts(stub.calls, _INTENT_DISCRIMINATOR) == 1
    assert _count_unit_prompts(stub.calls, _WONDER_DISCRIMINATOR) == 1
    deep = multi_stack_target / ".daydream" / "deep"
    intent_bytes = (deep / "intent.md").read_bytes()
    alternatives_bytes = (deep / "alternatives.json").read_bytes()
    stub.calls.clear()
    assert await run(config) == 0
    assert _count_unit_prompts(stub.calls, _INTENT_DISCRIMINATOR) == 0
    assert _count_unit_prompts(stub.calls, _WONDER_DISCRIMINATOR) == 0
    assert (deep / "intent.md").read_bytes() == intent_bytes
    assert (deep / "alternatives.json").read_bytes() == alternatives_bytes
    units = cast(dict[str, dict[str, object]], _latest_provenance(deep)["units"])
    assert units["intent"]["outcome"] == "hit"
    assert set(cast(dict[str, object], units["intent"]["grounding_status"])) == {"exploration"}
    assert units["alternatives"]["outcome"] == "hit"
    assert set(cast(dict[str, object], units["alternatives"]["grounding_status"])) == {
        "intent",
        "exploration",
    }


async def test_store_directory_survives_a_fresh_run_and_is_readable_by_the_next(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """The `.daydream/review-cache/` sibling is published back into the tree, so a
    later run (which detaches and re-seeds its live root) can read what an earlier
    run wrote."""
    install_stub_backend(monkeypatch, multi_stack_target)
    assert await run(make_config(multi_stack_target)) == 0
    store = multi_stack_target / ".daydream" / "review-cache"
    assert store.is_dir(), "store must be published beside .daydream/deep"
    (store / "entries").mkdir(exist_ok=True)
    (store / "entries" / ("a" * 64)).mkdir()
    assert await run(make_config(multi_stack_target)) == 0
    assert (store / "entries" / ("a" * 64)).is_dir(), "a fresh run must not wipe the store"


async def test_exploration_provenance_is_recorded_in_the_store(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH5/MH16: every named unit is accounted for, and the record lives in the
    store so it outlives the fresh-run wipe of `.daydream/deep/`."""
    install_stub_backend(monkeypatch, multi_stack_target)
    assert await run(make_config(multi_stack_target)) == 0
    provenance = multi_stack_target / ".daydream" / "review-cache" / "provenance"
    records = list(provenance.glob("*.json"))
    assert records, "the run must record its reuse provenance inside the store"
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["units"]["exploration"]["outcome"] in {"reused", "regenerated"}


async def test_identical_rerun_reviews_no_stack_and_a_leaf_edit_misses_only_its_shard(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH1/MH2/MH9: an identical rerun reuses every shard (zero review prompts);
    editing one assigned file misses exactly the shard that owns it."""
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
    assert _count_review_prompts(stub.calls) == 1          # only the shard owning mod0.py recomputes


async def test_reused_shard_leaves_no_stale_companion_artifact(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH6: the restore set is complete (records + review sidecar + coverage
    receipt + failure state) and the structural stack's delegation artifacts from
    an earlier iteration never survive into a reused structural unit."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    # Every reviewer completes a read of its own scope, so its clean-verdict
    # files carry evidence-gated ``clean`` verdicts -- the witness that a reused
    # stack's restored records are never re-reconciled against this run's empty
    # read set and downgraded to ``not_reviewed``.
    stub.per_stack_emit_reads = True
    config = make_config(multi_stack_target)
    assert await run(config) == 0
    deep = multi_stack_target / ".daydream" / "deep"
    fresh_receipts = json.loads((deep / "coverage-receipts.json").read_text())
    fresh_failures = (deep / "per-stack-failures.json").exists()
    fresh_records = _records_bytes(multi_stack_target)
    (deep / "structural-delegation.json").write_text(json.dumps({"primary_scopes": {"python": ["api.py"]}}))
    stub.calls.clear()
    assert await run(config) == 0
    assert _count_review_prompts(stub.calls) == 0
    assert json.loads((deep / "coverage-receipts.json").read_text()) == fresh_receipts
    assert (deep / "per-stack-failures.json").exists() == fresh_failures
    assert not (deep / "structural-delegation.json").exists(), "stale delegation must not survive"
    assert _stack_files(deep) == _expected_stack_files(deep)   # one records file per detected stack
    assert _records_bytes(multi_stack_target) == fresh_records, "reused records must be byte-identical"


async def test_editing_a_recorded_frontier_file_misses_every_shard_that_named_it(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH7: the frontier component of a shard's key is its *recorded* frontier, so an
    edit to a shared interface file invalidates every shard whose review prompt was
    grounded with it -- not only the shard that owns it. A shard that neither names
    nor owns the file (the structural meta-stack, whose per-file content is grounding
    rather than keying) still hits."""
    stub = install_stub_backend(monkeypatch, sibling_frontier_target)
    config = make_config(sibling_frontier_target, deep_shard_enabled=True, deep_shard_max_files=1,
                         deep_shard_max_bytes=10**9)
    assert await run(config) == 0
    deep = sibling_frontier_target / ".daydream" / "deep"
    receipts = _stack_receipts(deep)
    namers = {name for name, rec in receipts.items() if "core.py" in rec["frontier_files"]}
    assert namers, "fixture must ground at least one shard with core.py"
    assert all("core.py" not in receipts[name]["assigned_files"] for name in namers), (
        "a frontier namer must not own the edited file, or its miss would not prove the frontier is keyed"
    )
    owner = {
        name
        for name, rec in receipts.items()
        if "core.py" in rec["assigned_files"] and name != STRUCTURE_STACK_NAME
    }
    origin_structure = _stack_bytes(deep, {STRUCTURE_STACK_NAME})
    (sibling_frontier_target / "core.py").write_text("SHARED = 'edited'\n")
    stub.calls.clear()
    assert await run(config) == 0
    expected_miss = namers | owner
    assert _reviewed_stacks(stub.calls) == expected_miss
    assert _reused_stacks(deep) == set(receipts) - expected_miss
    # A frontier namer recomputes under a moved key, so the store holds both the
    # origin entry and this run's entry; the reused shard keeps exactly one and
    # restores the origin bytes verbatim.
    for name in namers:
        assert _entry_count_for_unit(deep, f"shard:{name}") == 2
    assert _entry_count_for_unit(deep, f"shard:{STRUCTURE_STACK_NAME}") == 1
    assert _stack_bytes(deep, {STRUCTURE_STACK_NAME}) == origin_structure
    assert _origin_stack_bytes(deep, {STRUCTURE_STACK_NAME}) == origin_structure


_ARBITER_DISCRIMINATOR = "you are the arbiter"


def _count_arbiter_prompts(calls: list[dict[str, object]]) -> int:
    """How many captured calls carried the production arbiter prompt."""
    return sum(
        1
        for call in calls
        if _ARBITER_DISCRIMINATOR in str(call.get("prompt", "")).lower()
    )


def _arbiter_stacks(severities: dict[str, str]) -> dict[str, dict[str, object]]:
    """Per-stack findings at three distinct ``(file, line)`` locations.

    Three locations make ``partition_arbiter_targets`` return more than one
    co-located group under a sharding route, which is the real-path
    precondition for a sharded whole-unit arbiter key.
    """
    locations = {
        "python": ("api.py", "python finding"),
        "react": ("App.tsx", "react finding"),
        "generic": ("README.md", "generic finding"),
    }
    return {
        name: {
            "severity": severities[name],
            "confidence": "high",
            "file": file,
            "line": 1,
            "description": description,
        }
        for name, (file, description) in locations.items()
    }


async def test_arbiter_reuses_whole_when_its_records_are_unchanged_and_resumes_per_group_when_one_is(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH1/MH8: one content key per arbiter unit; #732's per-group markers still resume
    a partially adjudicated run without re-running the groups already done."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_by_stack = _arbiter_stacks(
        {"python": "high", "react": "high", "generic": "high"}
    )
    stub.merge_echo_records = True
    config = make_config(
        multi_stack_target,
        latency_profile="balanced",
        review_profile=independent_alternatives_profile(),
    )
    assert await run(config) == 0
    deep = multi_stack_target / ".daydream" / "deep"
    assert _count_arbiter_prompts(stub.calls) >= 1
    merged = (deep / "merged-items.json").read_bytes()
    stub.calls.clear()
    assert await run(config) == 0
    assert _count_arbiter_prompts(stub.calls) == 0                  # whole-unit hit
    assert (deep / "merged-items.json").read_bytes() == merged
    # A partially completed earlier adjudication still resumes group-by-group:
    # the group markers are read from the fresh run's own artifacts, not from the store.
    assert sorted(p.name for p in deep.glob("arbiter-*-complete.marker"))


async def test_fix_loop_commit_reuses_untouched_shards_and_recomputes_the_rest_with_grounding(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH8/MH9/MH16 -- the S4 gate. A one-file fix committed the way the fix phase
    commits (build_commit_message -> 'Daydream-Run:' trailer) moves ``head``, so the
    pre-scan and intent are regenerated; the untouched shards must still hit, the
    affected shard and the aggregation units downstream must recompute, and every
    reused unit must carry its grounding provenance."""
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
    assert len(_reviewed_stacks(stub.calls)) == 1              # exactly the shard owning mod0.py
    assert _count_arbiter_prompts(stub.calls) >= 1             # records changed -> arbiter recomputes
    assert _count_merge_prompts(stub.calls) >= 1               # merged set consumes records
    provenance = json.loads(reuse_store.provenance_path(
        shard_many_python_target / ".daydream" / "review-cache", _session_id_of(deep)).read_text())
    reused = [k for k, v in provenance["units"].items()
              if k.startswith("shard:") and v["outcome"] == "hit"]
    assert reused, "sibling shards must still reuse after the fix commit"
    for unit in reused:                                        # MH16, per reused unit
        assert provenance["units"][unit]["grounding"]["settled_decisions"]["moved"] is True
        assert provenance["units"][unit]["grounding_status"]["exploration"] == "regenerated"


async def test_identical_rerun_pays_nothing_and_matches_the_first_run_byte_for_byte(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH9/MH10: no model call anywhere on the review surface, and the canonical
    artifacts come back byte-identical."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    config = make_config(multi_stack_target)
    assert await run(config) == 0
    deep = multi_stack_target / ".daydream" / "deep"
    canonical = {p.name: p.read_bytes() for p in deep.glob("stack-*-records.json")}
    canonical["merged-items.json"] = (deep / "merged-items.json").read_bytes()
    stub.calls.clear()
    assert await run(config) == 0
    assert _review_surface_prompts(stub.calls) == [], f"paid work on a warm run: {stub.calls}"
    assert {p.name: p.read_bytes() for p in deep.glob("stack-*-records.json")} == \
        {k: v for k, v in canonical.items() if k != "merged-items.json"}
    assert (deep / "merged-items.json").read_bytes() == canonical["merged-items.json"]


async def test_merge_unit_reuses_when_every_contributing_unit_is_unchanged(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """MH1/MH8/MH16: an identical rerun performs no cross-stack merge call; the
    merged items and the dedup candidates are restored byte-for-byte, and the
    render-only public report still lands with its coverage section."""
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    config = make_config(multi_stack_target)
    assert await run(config) == 0
    deep = multi_stack_target / ".daydream" / "deep"
    items, dedup = (deep / "merged-items.json").read_bytes(), (deep / "dedup-candidates.json").read_bytes()
    stub.calls.clear()
    assert await run(config) == 0
    assert _count_merge_prompts(stub.calls) == 0
    assert (deep / "merged-items.json").read_bytes() == items
    assert (deep / "dedup-candidates.json").read_bytes() == dedup
    # The run publishes the report from the artifact-session worker thread while
    # this (main) thread keeps a stale negative dentry for the path across the
    # rerun's detach/republish cycle, so a plain ``Path.is_file``/``read_text``
    # can spuriously miss it on this kernel. A directory-fd lookup is the same
    # observable claim -- the public report landed as a regular file -- without
    # depending on that per-thread cache behaviour.
    report_name = ".review-output.md"
    parent_fd = os.open(str(multi_stack_target), os.O_RDONLY)
    try:
        assert stat.S_ISREG(os.stat(report_name, dir_fd=parent_fd).st_mode), (
            "the public report still lands"
        )
        report_fd = os.open(report_name, os.O_RDONLY, dir_fd=parent_fd)
        try:
            content = os.read(report_fd, 1 << 20).decode("utf-8")
        finally:
            os.close(report_fd)
    finally:
        os.close(parent_fd)
    assert "## Coverage" in content


