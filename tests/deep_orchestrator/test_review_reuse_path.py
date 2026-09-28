"""Real-path tests for content-addressed reuse of deep review results.

These tests enter from ``runner.run`` with the real filesystem and event loop,
mocking only the external backend through the ``create_backend`` seam. They
assert observable outcomes of the ``.daydream/review-cache/`` store: that it is
published through the artifact-visibility anchors and survives a fresh run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
