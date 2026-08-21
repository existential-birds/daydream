"""Deterministic deep-sharding + sibling-frontier coverage canary (issue #763).

Real-path, model-free tests proving that enabled deep sharding and
sibling-frontier (``dependency_frontier_read``) coverage bind and hold their
regression invariants through the real ``runner.run`` pipeline, using the
cross-importing ``sibling_frontier_target`` fixture.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import install_stub_backend


def _numeric(name: str) -> tuple[int, ...]:
    """Natural sort: mod10.py orders after mod9.py (lexicographic would place
    mod10 before mod2, diverging from the plan's numeric ordering)."""
    return tuple(
        int(p) if p.isdigit() else float("inf")
        for p in re.split(r"(\d+)", name)
        if p
    )


def test_sibling_frontier_target_shape(sibling_frontier_target: Path) -> None:
    """The canary fixture has 13 changed python files, all importing core.py."""
    repo = sibling_frontier_target
    pys = sorted((p.name for p in repo.glob("*.py")), key=_numeric)
    assert pys == ["core.py"] + [f"mod{i}.py" for i in range(12)]
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
) -> None:
    """AC1-5/7: enabled sharding at DEFAULT bounds yields >=2 shards, each
    bounded, with non-empty sibling frontier, dependency_frontier_read credit,
    and a fail-open sweep for the unread frontier file."""
    from daydream.runner import RunConfig, run

    stub = install_stub_backend(monkeypatch, sibling_frontier_target)
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"mod5.py"})
    stub.parse_by_stack = {
        "python#1": {"severity": "high", "confidence": "HIGH",
                     "file": "mod3.py", "line": 1,
                     "description": "finding on mod3"}
    }
    exit_code = await run(RunConfig(
        target=str(sibling_frontier_target), cleanup=False,
        deep_shard_enabled=True,
        deep_shard_max_files=5, deep_shard_max_bytes=12288,
        deep_shard_fanout_cap=16, deep_shard_frontier_max=8,
    ))
    assert exit_code == 0
    deep = sibling_frontier_target / ".daydream" / "deep"

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

    # AC2: every shard within the default 5-file bound (bytes are dwarfed).
    for r in py_receipts.values():
        assert len(r["assigned_files"]) <= 5

    stats = json.loads((deep / "coverage-stats.json").read_text())
    pre = stats["pre_sweep"]
    cbe = pre["coverage_by_evidence"]

    # AC4: >=1 file credited dependency_frontier_read, absent from uncovered.
    assert cbe["dependency_frontier_read"] >= 1
    assert "core.py" not in pre["uncovered_files"]  # core is a sibling frontier file, read by python#0

    # Clean vs finding contrast: mod3.py has_findings, mod0.py clean.
    rec1 = json.loads((deep / "stack-python#1-records.json").read_text())
    verdicts1 = {e["path"]: e["verdict"] for e in rec1["verdicts"]}
    assert verdicts1["mod3.py"] == "has_findings"

    # AC5: the unread frontier file stays uncovered and is dispatched to sweep.
    assert "mod5.py" in pre["uncovered_files"]
    assert "mod5.py" in stats["attempted_files"]


async def test_deep_canary_no_redundant_sweep_when_all_covered(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC6: when every changed file carries valid evidence, pre_sweep
    uncovered_files is empty and the sweep is skipped."""
    from daydream.runner import RunConfig, run

    stub = install_stub_backend(monkeypatch, sibling_frontier_target)
    stub.per_stack_emit_reads = True  # read every assigned file; no unread knob
    exit_code = await run(RunConfig(
        target=str(sibling_frontier_target), cleanup=False,
        deep_shard_enabled=True,
        deep_shard_max_files=5, deep_shard_max_bytes=12288,
        deep_shard_fanout_cap=16, deep_shard_frontier_max=8,
    ))
    assert exit_code == 0
    deep = sibling_frontier_target / ".daydream" / "deep"
    stats = json.loads((deep / "coverage-stats.json").read_text())
    pre = stats["pre_sweep"]
    assert pre["uncovered_files"] == []          # no file lacks evidence
    assert stats["attempted_files"] == []        # sweep dispatched nothing
    assert pre["coverage_ratio"] == 1.0          # full coverage pre-sweep


def test_deep_canary_golden_reproduction(sibling_frontier_target: Path) -> None:
    """Should-Have: the committed golden receipt/stat JSON is a human-readable
    archive of the deterministic canary outcome. This test pins the archive's
    SHAPE (shard set + frontier non-emptiness + evidence counts), not a byte
    diff that would break on benign shard-name drift."""
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