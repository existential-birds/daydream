"""Real-path canary for bounded deep sharding and sibling-frontier context."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from daydream.runner import run
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import StubBackend, install_stub_backend

if TYPE_CHECKING:
    from daydream.run_config import RunConfig

MakeConfig = Callable[..., "RunConfig"]

# Set the byte limit below file-bound packing so a byte-budget regression cannot pass the canary.
CANARY_MAX_FILES = 5
CANARY_MAX_BYTES = 700
CANARY_FANOUT_CAP = 16
CANARY_FRONTIER_MAX = 8

def _diff_change_bytes(diff: str) -> dict[str, int]:
    """Independently measure header-inclusive UTF-8 diff blocks for the sharding assertions."""
    sizes: dict[str, int] = {}
    for block in re.split(r"^(?=diff --git )", diff, flags=re.M):
        m = re.match(r"^diff --git a/(.+?) b/", block)
        if m is not None:
            sizes.setdefault(m.group(1), len(block.encode("utf-8")))
    return sizes

async def _drive_canary(target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, *,
    parse_by_stack: dict[str, dict[str, object]] | None = None,
    enabled: bool | None = True,
    clean: bool = False,
    frontier_max: int = CANARY_FRONTIER_MAX,
) -> tuple[Path, StubBackend]:
    """Run the real deep entrypoint with fixed sharding bounds; return artifact directory and backend."""
    stub = install_stub_backend(monkeypatch, target)
    if clean:
        stub = EmptyReviewBackend(target)
        monkeypatch.setattr('daydream.runner.create_backend', lambda *_args, **_kwargs: stub)
    stub.parse_by_stack = parse_by_stack
    exit_code = await run(make_config(target, deep_shard_enabled=enabled, deep_shard_max_files=CANARY_MAX_FILES,
            deep_shard_max_bytes=CANARY_MAX_BYTES, deep_shard_fanout_cap=CANARY_FANOUT_CAP,
            deep_shard_frontier_max=frontier_max,
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

@pytest.mark.parametrize(('enabled', 'with_finding', 'parser_failed', 'frontier_max'),
                         [(True, True, False, 8), (None, True, False, 2), (None, False, False, 8),
                          (None, False, True, 8), (None, False, False, 0)],
                         ids=['explicit', 'automatic', 'automatic-clean', 'automatic-parser-unavailable',
                              'automatic-zero-frontier'])
async def test_deep_canary_sharding_and_sibling_frontier(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    enabled: bool | None, with_finding: bool, parser_failed: bool, frontier_max: int,
) -> None:
    """Shards retain their byte/file limits, frontier context, and findings."""
    if parser_failed:
        import tree_sitter_python

        from daydream.tree_sitter_index import runtime

        def unavailable_grammar() -> None:
            raise RuntimeError('external Python grammar unavailable')

        # Force this run through the external constructor instead of a parser
        # cached by an earlier test; production still owns its fail-open path.
        monkeypatch.setattr(runtime, '_PARSER_CACHE', {})
        monkeypatch.setattr(tree_sitter_python, 'language', unavailable_grammar)
    deep, stub = await _drive_canary(sibling_frontier_target, monkeypatch, make_config,
        parse_by_stack={
            "python#2": {
                "severity": "high", "confidence": "HIGH", "file": "mod3.py", "line": 1,
                "description": "finding on mod3",
            }
        } if with_finding else {}, enabled=enabled, clean=not with_finding, frontier_max=frontier_max,
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
                supporting = frontier.group(1).split(', ')
                assert supporting == sorted(supporting) and len(supporting) <= frontier_max
                assert set(supporting).isdisjoint(assignments[scope.group(1)])
                frontier_contexts.append(frontier.group(1))
    assert len(assignments) == len(shards)
    if parser_failed or frontier_max == 0:
        assert frontier_contexts == []
    else:
        assert frontier_contexts, "sharded reviewers lost cross-shard source context"
    expected = {"core.py", *(f"mod{i}.py" for i in range(12))}
    assert Counter(file for files in assignments.values() for file in files) == Counter(expected)
    assert list(assignments) == [f'python#{index}' for index in range(len(assignments))]

    diff = _git(sibling_frontier_target, "diff", "main..HEAD")
    sizes = _diff_change_bytes(diff)
    for assigned in assignments.values():
        assert len(assigned) <= CANARY_MAX_FILES
        assert sum(sizes.get(file, 1) for file in assigned) <= CANARY_MAX_BYTES

    records = json.loads((deep / "stack-python#2-records.json").read_text())
    assert any(record["file"] == "mod3.py" for record in records["issues"]) is with_finding
    merged = json.loads((deep / "merged-items.json").read_text())
    assert bool(merged["items"]) is with_finding
    assert "verdicts" not in records
    coverage = json.loads((deep / 'review-coverage.json').read_text())
    assert all(scope['status'] == 'complete' for scope in coverage['stack_outcomes'])
    primary = [path for scope in coverage['planned_scopes'] if scope['scope_id'] != 'structure'
               for path in scope['files']]
    assert Counter(primary) == Counter(expected)
    structural = next(scope for scope in coverage['planned_scopes'] if scope['scope_id'] == 'structure')
    assert set(structural['files']) == expected
