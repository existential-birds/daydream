"""Real-path canary for bounded deep sharding and sibling-frontier context."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from daydream.runner import run
from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import StubBackend, install_stub_backend

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

async def _drive_canary(target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, *,
    parse_by_stack: dict[str, dict[str, object]] | None = None,
) -> tuple[Path, StubBackend]:
    """Run one enabled-sharding deep canary through the real ``runner.run``.

    Shared harness preamble for the two live canary tests (issue #763):
    installs the stub backend,
    drives a single deep run at the locked sharding bounds, and returns the
    ``.daydream/deep`` output dir against which both tests assert.
    """
    stub = install_stub_backend(monkeypatch, target)
    stub.parse_by_stack = parse_by_stack
    exit_code = await run(make_config(target, deep_shard_enabled=True, deep_shard_max_files=CANARY_MAX_FILES,
            deep_shard_max_bytes=CANARY_MAX_BYTES, deep_shard_fanout_cap=CANARY_FANOUT_CAP,
            deep_shard_frontier_max=CANARY_FRONTIER_MAX,
        )
    )
    assert exit_code == 0
    return target / ".daydream" / "deep", stub

def test_sibling_frontier_target_shape(sibling_frontier_target: Path) -> None:
    """The canary fixture has 13 changed python files, all importing core.py."""
    repo = sibling_frontier_target
    pys = {p.name for p in repo.glob("*.py")}
    assert pys == {"core.py"} | {f"mod{i}.py" for i in range(12)}
    # All files changed on the feature branch (diff vs main is non-empty).
    changed = set(_git(repo, "diff", "--name-only", "main..HEAD").splitlines())
    assert changed == {p.name for p in repo.glob("*.py")}
    # Real, tree-sitter-parseable cross-file import edges are physically present.
    assert all("from core import core_helper" in (repo / f"mod{i}.py").read_text() for i in range(12))

async def test_deep_canary_sharding_and_sibling_frontier(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """Shards retain their byte/file limits, frontier context, and findings."""
    deep, stub = await _drive_canary(sibling_frontier_target, monkeypatch, make_config,
        parse_by_stack={
            "python#2": {
                "severity": "high", "confidence": "HIGH", "file": "mod3.py", "line": 1,
                "description": "finding on mod3",
            }
        },
    )

    # AC1: >=2 stack-python#N review descriptors.
    shards = sorted(p for p in deep.glob("stack-python#*-review.md"))
    assert len(shards) >= 2

    assignments: dict[str, list[str]] = {}
    frontier_contexts: list[str] = []
    for call in stub.calls:
        prompt = call["prompt"]
        scope = re.search(r"You are reviewing the (python#\d+) stack\. Assigned files: ([^\n]+)", prompt)
        if scope:
            assignments[scope.group(1)] = scope.group(2).split(", ")
            frontier = re.search(r"review targets\): ([^\n]+)\.", prompt)
            if frontier:
                frontier_contexts.append(frontier.group(1))
    assert len(assignments) == len(shards)
    assert frontier_contexts, "sharded reviewers lost cross-shard source context"
    assert set().union(*(set(files) for files in assignments.values())) == {
        "core.py", *(f"mod{i}.py" for i in range(12))
    }

    diff = _git(sibling_frontier_target, "diff", "main..HEAD")
    sizes = _diff_change_bytes(diff)
    for assigned in assignments.values():
        assert len(assigned) <= CANARY_MAX_FILES
        assert sum(sizes.get(file, 1) for file in assigned) <= CANARY_MAX_BYTES

    records = json.loads((deep / "stack-python#2-records.json").read_text())
    assert any(record["file"] == "mod3.py" for record in records["issues"])
    merged = json.loads((deep / "merged-items.json").read_text())
    assert merged["items"]
    assert "verdicts" not in records
    assert not (deep / "coverage-receipts.json").exists()
    assert not (deep / "coverage-stats.json").exists()
    assert not (deep / "stack-uncovered-records.json").exists()
    assert not list(deep.glob("uncovered-*-review.md"))
    assert not any("uncovered file sweep" in call["prompt"].lower() for call in stub.calls)
