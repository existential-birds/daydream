"""Deterministic deep-sharding + sibling-frontier coverage canary (issue #763).

Real-path, model-free tests proving that enabled deep sharding and
sibling-frontier (``dependency_frontier_read``) coverage bind and hold their
regression invariants through the real ``runner.run`` pipeline, using the
cross-importing ``sibling_frontier_target`` fixture.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import install_stub_backend

if TYPE_CHECKING:
    from daydream.runner import RunConfig

MakeConfig = Callable[..., "RunConfig"]

# Deterministic deep-sharding bounds the live canaries lock (issue #763).
# ``CANARY_MAX_BYTES`` is set BELOW the fixture's file-bound packing so the
# changed-byte budget -- not the 5-file ceiling -- is the binding constraint:
# a byte-budget regression in sharding._pack_shards would silently pack
# oversized shards past this budget and the canary's AC2 byte check fails.
CANARY_MAX_FILES = 5
CANARY_MAX_BYTES = 700
CANARY_FANOUT_CAP = 16
CANARY_FRONTIER_MAX = 8


def _diff_change_bytes(diff: str) -> dict[str, int]:
    """Per-file changed-byte sizes, mirroring ``sharding._file_change_bytes``.

    Splits ``diff`` into ``diff --git`` blocks (the shared ``_DIFF_BLOCK_SPLIT``
    contract) and records each block's encoded byte length under its post-state
    path. Files absent from the map size as 1 byte, exactly as ``_pack_shards``
    sizes them, so the AC2 byte check compares like with like.
    """
    sizes: dict[str, int] = {}
    for block in re.split(r"^(?=diff --git )", diff, flags=re.M):
        m = re.match(r"^diff --git a/(.+?) b/", block)
        if m is not None:
            sizes.setdefault(m.group(1), len(block.encode("utf-8")))
    return sizes


def _shard_assigning(
    receipts: dict[str, dict[str, list[str]]], filename: str
) -> tuple[str, dict[str, list[str]]]:
    """The (name, receipt) of the python shard that assigns ``filename``.

    Membership lookup rather than a bare ``python#N`` subscript, so a change to
    the canary repo or diff ordering that re-buckets files degrades to a
    targeted assertion failure instead of an opaque :class:`KeyError`.
    """
    for name, r in receipts.items():
        if name.startswith("python#") and filename in r["assigned_files"]:
            return name, r
    raise AssertionError(f"{filename!r} is not assigned to any python shard")


async def _drive_canary(
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    *,
    unread: frozenset[str] = frozenset(),
    parse_by_stack: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Run one enabled-sharding deep canary through the real ``runner.run``.

    Shared harness preamble for the two live canary tests (issue #763):
    installs the stub backend (reading every assigned file except ``unread``),
    drives a single deep run at the locked sharding bounds, and returns the
    ``.daydream/deep`` output dir against which both tests assert.
    """
    from daydream.runner import run

    stub = install_stub_backend(monkeypatch, target)
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = unread
    stub.parse_by_stack = parse_by_stack
    exit_code = await run(make_config(
        target,
        deep_shard_enabled=True,
        deep_shard_max_files=CANARY_MAX_FILES,
        deep_shard_max_bytes=CANARY_MAX_BYTES,
        deep_shard_fanout_cap=CANARY_FANOUT_CAP,
        deep_shard_frontier_max=CANARY_FRONTIER_MAX,
    ))
    assert exit_code == 0
    return target / ".daydream" / "deep"


def test_sibling_frontier_target_shape(sibling_frontier_target: Path) -> None:
    """The canary fixture has 13 changed python files, all importing core.py."""
    repo = sibling_frontier_target
    pys = {p.name for p in repo.glob("*.py")}
    assert pys == {"core.py"} | {f"mod{i}.py" for i in range(12)}
    # All files changed on the feature branch (diff vs main is non-empty).
    changed = set(
        _git(repo, "diff", "--name-only", "main..HEAD").splitlines()
    )
    assert changed == {p.name for p in repo.glob("*.py")}
    # Real, tree-sitter-parseable cross-file import edges are physically present.
    assert all(
        "from core import core_helper" in (repo / f"mod{i}.py").read_text()
        for i in range(12)
    )


async def test_deep_canary_sharding_and_sibling_frontier(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """AC1-5/7: enabled sharding at deterministic bounds yields >=2 shards,
    each within the file AND changed-byte budgets (the byte budget binding),
    with non-empty sibling frontier, dependency_frontier_read credit, and a
    fail-open sweep for the unread frontier file."""
    # mod3.py's HIGH finding keys off the shard that assigns it; under the
    # locked byte budget that is python#2 (asserted below by membership lookup,
    # so shard-name drift surfaces a targeted message, not a bare subscript).
    deep = await _drive_canary(
        sibling_frontier_target, monkeypatch, make_config,
        unread=frozenset({"mod5.py"}),
        parse_by_stack={
            "python#2": {"severity": "high", "confidence": "HIGH",
                         "file": "mod3.py", "line": 1,
                         "description": "finding on mod3"}
        },
    )

    # AC1: >=2 stack-python#N review descriptors.
    shards = sorted(p for p in deep.glob("stack-python#*-review.md"))
    assert len(shards) >= 2

    # AC3: coverage-receipts.json records assigned/inline/frontier per shard.
    receipts = json.loads((deep / "coverage-receipts.json").read_text())
    py_receipts = {k: v for k, v in receipts.items() if k.startswith("python#")}
    assert py_receipts, "no python shards in receipts"
    for r in py_receipts.values():
        assert set(r) == {"assigned_files", "inline_files", "frontier_files"}
    # Non-empty sibling frontier on at least one shard (the canary's point).
    assert any(r["frontier_files"] for r in py_receipts.values())

    # AC2: every shard is within BOTH the per-shard file bound and the
    # changed-byte budget. The byte bound is the binding constraint at the
    # locked CANARY_MAX_BYTES (shards pack to 2-3 files, not the 5-file
    # ceiling), so a byte-budget regression in _pack_shards fails here.
    diff = _git(sibling_frontier_target, "diff", "main..HEAD")
    sizes = _diff_change_bytes(diff)
    for r in py_receipts.values():
        assigned = r["assigned_files"]
        assert len(assigned) <= CANARY_MAX_FILES
        assert sum(sizes.get(f, 1) for f in assigned) <= CANARY_MAX_BYTES

    stats = json.loads((deep / "coverage-stats.json").read_text())
    pre = stats["pre_sweep"]
    cbe = pre["coverage_by_evidence"]

    # AC4: >=1 file credited dependency_frontier_read, absent from uncovered.
    assert cbe["dependency_frontier_read"] >= 1
    assert "core.py" not in pre["uncovered_files"]  # sibling frontier file, read by its owning shard

    # Clean vs finding contrast: the shard owning mod3.py reports has_findings;
    # the one owning mod0.py is clean. Looked up by membership, so re-bucketing
    # cannot surface as an opaque KeyError.
    name3, _ = _shard_assigning(receipts, "mod3.py")
    rec3 = json.loads((deep / f"stack-{name3}-records.json").read_text())
    verdicts3 = {e["path"]: e["verdict"] for e in rec3["verdicts"]}
    assert verdicts3["mod3.py"] == "has_findings"
    name0, _ = _shard_assigning(receipts, "mod0.py")
    rec0 = json.loads((deep / f"stack-{name0}-records.json").read_text())
    verdicts0 = {e["path"]: e["verdict"] for e in rec0["verdicts"]}
    assert verdicts0["mod0.py"] == "clean"

    # AC5: the unread frontier file stays uncovered and is dispatched to sweep.
    assert "mod5.py" in pre["uncovered_files"]
    assert "mod5.py" in stats["attempted_files"]


async def test_deep_canary_no_redundant_sweep_when_all_covered(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """AC6: when every changed file carries valid evidence, pre_sweep
    uncovered_files is empty and the sweep is skipped."""
    deep = await _drive_canary(
        sibling_frontier_target, monkeypatch, make_config,
    )
    stats = json.loads((deep / "coverage-stats.json").read_text())
    pre = stats["pre_sweep"]
    assert pre["uncovered_files"] == []          # no file lacks evidence
    assert stats["attempted_files"] == []        # sweep dispatched nothing
    assert pre["coverage_ratio"] == 1.0          # full coverage pre-sweep


def test_deep_canary_golden_fixtures_are_well_formed() -> None:
    """Well-formedness check on the committed golden receipt/stat fixtures.

    NOT a golden reproduction: the live-run invariants are pinned by
    ``test_deep_canary_sharding_and_sibling_frontier`` and
    ``test_deep_canary_no_redundant_sweep_when_all_covered`` above, which
    each drive a fresh ``runner.run`` and assert against live
    ``.daydream/deep`` output. This check guards the COMMITTED fixtures from
    a corrupting edit: the archive must still look like a valid deterministic
    canary outcome (>=2 shards, a non-empty frontier, every shard within the
    5-file bound, non-trivial dependency-frontier evidence, and mod5.py
    uncovered then attempted). It compares shape, not bytes, so benign
    shard-name drift can't break it.
    """
    golden_receipt = json.loads(
        (Path(__file__).parent / "fixtures" / "deep" / "coverage-receipts.json").read_text()
    )
    golden_stats = json.loads(
        (Path(__file__).parent / "fixtures" / "deep" / "coverage-stats.json").read_text()
    )
    py_shards = {k: v for k, v in golden_receipt.items() if k.startswith("python#")}
    assert len(py_shards) >= 2
    assert any(r["frontier_files"] for r in py_shards.values())
    assert all(len(r["assigned_files"]) <= 5 for r in py_shards.values())
    pre = golden_stats["pre_sweep"]
    assert pre["coverage_by_evidence"]["dependency_frontier_read"] >= 1
    assert "mod5.py" in pre["uncovered_files"]
    assert "mod5.py" in golden_stats["attempted_files"]
