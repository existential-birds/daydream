"""Real-path tests for content-addressed reuse of deep review results.

These tests enter from ``runner.run`` with the real filesystem and event loop,
mocking only the external backend through the ``create_backend`` seam. They
assert observable outcomes of the ``.daydream/review-cache/`` store: that it is
published through the artifact-visibility anchors and survives a fresh run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from daydream.config import STRUCTURE_STACK_NAME
from daydream.runner import run
from tests.deep_orchestrator.support import _count_review_prompts
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
    return json.loads((deep / "coverage-receipts.json").read_text(encoding="utf-8"))


_PER_STACK_PROMPT = re.compile(r"you are reviewing the (\S+) stack", re.IGNORECASE)


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
    provenance = deep.parent / "review-cache" / "provenance"
    records = sorted(provenance.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not records:
        return set()
    record = json.loads(records[-1].read_text(encoding="utf-8"))
    return {
        unit.removeprefix("shard:")
        for unit, trace in (record.get("units") or {}).items()
        if unit.startswith("shard:")
        and isinstance(trace, dict)
        and trace.get("outcome") == "hit"
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
