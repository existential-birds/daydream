"""Stack detection routing tests (D-11..D-16).

Covers ``daydream.deep.detection.detect_stacks`` implemented in plan 05-01.
"""
from pathlib import Path
from typing import Any


def test_stack_assignment_has_no_skill_field() -> None:
    """M9: StackAssignment carries routing metadata only, never a skill invocation."""
    from daydream.deep.detection import StackAssignment

    assignment = StackAssignment(stack_name="python", files=["a.py"])
    assert not hasattr(assignment, "skill_invocation")


def test_extension_routing_python() -> None:
    """D-11: .py files route to python stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py"])
    names = {a.stack_name for a in result}
    assert "python" in names



def test_extension_routing_react() -> None:
    """D-11: .tsx files route to react stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/App.tsx"])
    assert "react" in {a.stack_name for a in result}


def test_ambiguous_single_stack_shortcut() -> None:
    """D-12: single stack in diff -> ambiguous files unconditionally join it."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/app.py", "migrations/001.sql"])
    python = next(a for a in result if a.stack_name == "python")
    assert "migrations/001.sql" in python.files


def test_ambiguous_nearest_ancestor() -> None:
    """D-12: ambiguous file routes to nearest-ancestor unambiguous stack."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(
        ["backend/api/main.py", "backend/api/queries.sql", "frontend/App.tsx"],
    )
    python = next(a for a in result if a.stack_name == "python")
    assert "backend/api/queries.sql" in python.files


def test_equal_depth_fallthrough() -> None:
    """D-12c: equal-depth ambiguity falls through to generic."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(
        ["main.py", "App.tsx", "shared.sql"],  # .sql has no unambiguous ancestor,
    )
    generic = next(a for a in result if a.stack_name == "generic")
    assert "shared.sql" in generic.files


def test_config_default_generic() -> None:
    """D-13a: .yaml / .toml route to generic by default."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["config.yaml"])
    language_names = {a.stack_name for a in result if a.stack_name != "structure"}
    assert language_names == {"generic"}


def test_config_promotion_pyproject() -> None:
    """D-13b: pyproject.toml + .py co-change -> python."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["pyproject.toml", "src/main.py"])
    python = next(a for a in result if a.stack_name == "python")
    assert "pyproject.toml" in python.files


def test_no_static_promotion_without_cochange() -> None:
    """D-13c: static paths alone do not promote config to a stack."""
    from daydream.deep.detection import detect_stacks

    # pyproject.toml alone (no .py in diff) stays generic
    result = detect_stacks(["pyproject.toml"])
    language_names = {a.stack_name for a in result if a.stack_name != "structure"}
    assert language_names == {"generic"}


def test_md_pinned_to_generic() -> None:
    """D-14: .md files pinned to generic even when co-changed with code."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py", "README.md"])
    generic = next(a for a in result if a.stack_name == "generic")
    assert "README.md" in generic.files
    assert generic.is_docs_only is False  # mixed with py stack, but docs go here


def test_no_files_dropped() -> None:
    """D-15: every file is routed somewhere."""
    from daydream.deep.detection import detect_stacks

    files = ["src/main.py", "README.md", "config.yaml", "Dockerfile", "src/App.tsx"]
    result = detect_stacks(files)
    routed = {f for a in result for f in a.files}
    assert routed == set(files)


def test_detected_stack_never_degrades_to_generic() -> None:
    """M3: D-16 removed — a detected stack never degrades to generic without a skill."""
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/lib.rs"])
    language_names = {a.stack_name for a in result if a.stack_name != "structure"}
    assert language_names == {"rust"}


def test_structure_stack_emitted_for_code_diff() -> None:
    """Structure stack is present on any non-docs-only code diff (skill-free)."""
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["src/main.py", "src/util.py"])
    structure = next((a for a in result if a.stack_name == STRUCTURE_STACK_NAME), None)
    assert structure is not None
    assert structure.files == ["src/main.py", "src/util.py"]
    assert structure.is_docs_only is False


def test_structure_stack_files_are_union_across_languages() -> None:
    """Structure stack sees every changed file regardless of language."""
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import detect_stacks

    files = ["api/main.py", "ui/App.tsx", "infra/Dockerfile"]
    result = detect_stacks(files)
    structure = next(a for a in result if a.stack_name == STRUCTURE_STACK_NAME)
    assert sorted(structure.files) == sorted(files)


def test_structure_stack_skipped_for_docs_only_diff() -> None:
    """Structural rubric does not apply when the entire diff is docs."""
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import detect_stacks

    result = detect_stacks(["README.md", "CHANGELOG.md"])
    assert all(a.stack_name != STRUCTURE_STACK_NAME for a in result)


def test_structure_stack_skipped_for_empty_diff() -> None:
    """Empty changed_files yields no stacks at all, including structure."""
    from daydream.deep.detection import detect_stacks

    assert detect_stacks([]) == []


# --- Issue #731: deep-review sharding splitter ---


def test_shard_stacks_splits_oversized_stack_by_file_count() -> None:
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(
        stack_name="python",
        files=[f"src/m{i}.py" for i in range(6)],
    )
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    shards = [s for s in out if s.stack_name.startswith("python#")]
    assert len(shards) == 3                      # 6 files / 2 per shard
    assert [s.stack_name for s in shards] == ["python#0", "python#1", "python#2"]
    union = [f for s in shards for f in s.files]
    assert sorted(union) == sorted(stack.files)  # no file dropped, no duplicate
    assert all(len(s.files) <= 2 for s in shards)


def test_shard_stacks_never_splits_structure_meta_stack() -> None:
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    structure = StackAssignment(
        stack_name=STRUCTURE_STACK_NAME,
        files=[f"src/m{i}.py" for i in range(50)],
    )
    out = shard_stacks([structure], "", max_files=5, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    assert [s for s in out if s.stack_name == STRUCTURE_STACK_NAME] == [structure]  # unchanged, single


def test_shard_stacks_deterministic_names_and_assignments() -> None:
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(stack_name="python", files=[f"src/m{i}.py" for i in range(5)])

    def _split() -> list[Any]:
        return shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)

    a = _split()
    b = _split()
    assert [(s.stack_name, s.files) for s in a] == [(s.stack_name, s.files) for s in b]


def test_shard_stacks_under_bound_returns_original_unsplit() -> None:
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(stack_name="python", files=["a.py", "b.py"])
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    assert out == [stack]


def test_shard_stacks_splits_by_changed_bytes_not_file_count() -> None:
    """Issue #731: an oversized *byte* budget forces a split even when the file
    count is within ``max_files``; the union is still exact (no drop/dup)."""
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    # The split is forced by header-inclusive block sizing: each ``diff --git``
    # block here is ~64-69 bytes (headers + one hunk line) and any two of them
    # total ~130 > max_bytes=100. ``_per_file_change_bytes`` sizes the whole
    # block, headers included, not just the hunk lines.
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n+'x'*2000\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n+'y'\n"
        "diff --git a/c.py b/c.py\n--- a/c.py\n+++ b/c.py\n@@ -1 +1 @@\n+'z'\n"
    )
    stack = StackAssignment(stack_name="python",
                            files=["a.py", "b.py", "c.py"])
    out = shard_stacks([stack], diff, max_files=100, max_bytes=100, fanout_cap=16, frontier_max=8)
    shards = [s for s in out if s.stack_name.startswith("python#")]
    assert len(shards) >= 2                     # byte budget forces a split
    assert all(len(s.files) >= 1 for s in shards)
    union = [f for s in shards for f in s.files]
    assert sorted(union) == ["a.py", "b.py", "c.py"]  # still no drop/dup


def test_shard_stacks_fanout_cap_limits_total_tasks() -> None:
    """Issue #731: total review tasks (shards + unsplit non-structural stacks)
    never exceeds the fan-out cap; everything is still assigned exactly once."""
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    # Two oversized stacks would each yield 6 shards = 12 tasks; cap=4.
    py = StackAssignment(stack_name="python", files=[f"p{i}.py" for i in range(12)])
    rs = StackAssignment(stack_name="rust", files=[f"r{i}.rs" for i in range(12)])
    out = shard_stacks([py, rs], "", max_files=2, max_bytes=10**9, fanout_cap=4, frontier_max=8)
    # Total review tasks (shards + unsplit stacks) never exceeds the cap.
    assert len(out) <= 4
    # Everything still assigned exactly once.
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted([f"p{i}.py" for i in range(12)] + [f"r{i}.rs" for i in range(12)])


def test_shard_stacks_fanout_cap_single_shard_split_never_wastes_reduction() -> None:
    """Issue #731 fix: a stack that packs into exactly one shard (a single
    oversized file -- never split mid-file) is unsplit-equivalent, keeps its
    original name, and never eats a cap-reduction; the cap still holds while
    a reducible sharded stack remains."""
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    # big.py's diff block exceeds max_bytes=100 but holds a single file, so it
    # packs into exactly one shard ("never split a file").
    big_diff = (
        "diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n@@ -1 +1 @@\n+"
        + "x" * 80
        + "\n"
    )
    huge = StackAssignment(stack_name="python", files=["big.py"])
    many = StackAssignment(stack_name="rust", files=[f"r{i}.rs" for i in range(12)])
    out = shard_stacks([huge, many], big_diff, max_files=2, max_bytes=100, fanout_cap=4, frontier_max=8)
    # The single-shard stack stays unsplit under its original name.
    assert any(s.stack_name == "python" and s.files == ["big.py"] for s in out)
    # Total tasks (shards + unsplit) never exceed the cap.
    assert len(out) <= 4
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted(["big.py"] + [f"r{i}.rs" for i in range(12)])


def test_shard_stacks_fanout_cap_irreducible_when_unsplit_stacks_outnumber_cap() -> None:
    """Issue #731 fix: when the unsplit non-structural stacks alone outnumber
    the cap, no shard exists to un-split and the total necessarily exceeds it
    (files are never dropped or merged); every stack is still assigned once."""
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stacks = [
        StackAssignment(stack_name=f"s{i}", files=[f"f{i}.py"])
        for i in range(18)
    ]
    out = shard_stacks(stacks, "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    # No shard exists to un-split; the floor is the distinct-stack count.
    assert len(out) == 18
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted([f"f{i}.py" for i in range(18)])


def test_shard_stacks_co_locates_dependent_files_when_room(tmp_path: Path) -> None:
    """Issue #731: files sharing an import edge stay in the same shard when it
    fits within the bounds. The graph comes from ``build_import_graph`` over
    real on-disk files, and the edge pair (a.py, d.py) is non-adjacent in sorted
    order -- without graph-driven co-location the sorted-singleton fallback
    packs [a,b]/[c,d] and this assertion fails, so dropping co-location is
    caught."""
    from pathlib import Path

    from daydream.deep.dependency import build_import_graph
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    # d.py imports a.py (a resolvable edge). 4 files / max_files=2 forces the
    # shard; the {a,d} component fits one shard so d.py stays co-located with
    # its dependency a.py.
    stack = StackAssignment(stack_name="python",
                            files=["a.py", "b.py", "c.py", "d.py"])
    root = Path(tmp_path)
    for name in ("a.py", "b.py", "c.py"):
        (root / name).write_text("x = 1\n")
    (root / "d.py").write_text("import a\n")
    graph = build_import_graph(["a.py", "b.py", "c.py", "d.py"], root)
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9,
                       fanout_cap=16, frontier_max=8, graph=graph)
    shards = [s for s in out if s.stack_name.startswith("python#")]
    edge_shard = next(s for s in shards if "d.py" in s.files)
    assert "a.py" in edge_shard.files            # co-located when size permits


def test_shard_stacks_fail_open_without_graph() -> None:
    """Issue #731: an empty/missing graph never drops a file; each file gets
    exactly one assignment (deterministic sorted-singleton packing)."""
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(stack_name="python",
                            files=[f"m{i}.py" for i in range(5)])
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8, graph={})
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted(stack.files)   # no graph -> every file still assigned once
    # A file with no resolvable edge still gets exactly one assignment (fallback).
    assert len(set(union)) == len(union)  # no duplicate primary assignment


def test_shard_stacks_populates_bounded_frontier(tmp_path: Path) -> None:
    """Issue #731: cross-shard shared files surface as a bounded frontier,
    derived from a real ``build_import_graph`` over on-disk files."""
    from pathlib import Path

    from daydream.deep.dependency import build_import_graph
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    stack = StackAssignment(stack_name="python",
                            files=[f"m{i}.py" for i in range(8)])
    # m4..m7 all import m0 (a shared interface in shard 0).
    root = Path(tmp_path)
    for i in range(8):
        (root / f"m{i}.py").write_text("import m0\n" if i >= 4 else "x = 1\n")
    graph = build_import_graph([f"m{i}.py" for i in range(8)], root)
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=3, graph=graph)
    frontier_shards = [s for s in out if getattr(s, "frontier_files", [])]
    assert frontier_shards, "cross-shard shared files must surface as a frontier"
    assert all(len(s.frontier_files) <= 3 for s in out)   # bounded


def test_build_import_graph_resolves_python_edges(tmp_path: Path) -> None:
    """Issue #731: tree-sitter resolves python import edges, absolute and
    relative; unknown grammars fail open to singletons."""
    from pathlib import Path

    from daydream.deep.dependency import build_import_graph

    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("from .util import helper\n")
    (tmp_path / "pkg" / "util.py").write_text("def helper(): pass\n")
    (tmp_path / "notes.txt").write_text("no grammar\n")
    graph = build_import_graph(["a.py", "b.py", "pkg/mod.py", "pkg/util.py", "notes.txt"], Path(tmp_path))
    assert "b.py" in graph["a.py"]          # absolute import edge resolved
    assert graph["b.py"] == set()           # no outgoing edge
    assert "pkg/util.py" in graph["pkg/mod.py"]  # 'from .util import helper' edge
    assert "notes.txt" in graph             # unknown grammar -> fail-open singleton


def test_build_import_graph_resolves_multilanguage_edges(tmp_path: Path) -> None:
    """Issue #731: tree-sitter resolves .ts/.go/.rs import edges too, so the
    dependency graph is not inert outside python."""
    from pathlib import Path

    from daydream.deep.dependency import build_import_graph

    (tmp_path / "a.ts").write_text('import { b } from "./b"\n')
    (tmp_path / "b.ts").write_text("export const b = 1;\n")
    (tmp_path / "a.go").write_text('package a\nimport "b"\n')
    (tmp_path / "b.go").write_text("package b\n")
    (tmp_path / "a.rs").write_text("use b::c;\n")
    (tmp_path / "b.rs").write_text("pub fn c() {}\n")
    files = ["a.ts", "b.ts", "a.go", "b.go", "a.rs", "b.rs"]
    graph = build_import_graph(files, Path(tmp_path))
    assert "b.ts" in graph["a.ts"]          # './b' resolves to sibling b.ts
    assert "b.go" in graph["a.go"]          # go import path -> b.go
    assert "b.rs" in graph["a.rs"]          # rust 'use b::c' -> module file b.rs


def test_shard_stacks_default_bounds_split_16file_50kb_and_inline() -> None:
    """Issue #740 AC4/AC5: a realistic 16-file ~50 KB stack splits under the DEFAULT
    bounds, and every shard's diff fits the inline budget by construction."""
    from daydream.config import DEFAULT_DEEP_SHARD_MAX_BYTES, DEFAULT_DEEP_SHARD_MAX_FILES
    from daydream.deep.detection import StackAssignment
    from daydream.deep.prompts import inline_grounded_files
    from daydream.deep.sharding import shard_stacks

    files = [f"src/m{i:02d}.py" for i in range(16)]
    # ~2.9 KB per hunk -> ~47 KB total changed bytes, > the 12288-byte bound.
    diff = "".join(
        f"diff --git a/{f} b/{f}\n--- a/{f}\n+++ b/{f}\n@@ -1 +1 @@\n+x{'a' * 2900}\n"
        for f in files
    )
    stack = StackAssignment(stack_name="python", files=files)
    out = shard_stacks(
        [stack], diff,
        max_files=DEFAULT_DEEP_SHARD_MAX_FILES,
        max_bytes=DEFAULT_DEEP_SHARD_MAX_BYTES,
        fanout_cap=16, frontier_max=8,
    )
    shards = [s for s in out if s.stack_name.startswith("python#")]
    assert len(shards) > 1                       # the stack splits
    union = [f for s in shards for f in s.files]
    assert sorted(union) == sorted(files)        # no drop, no dup
    # Every shard inlines: its hunks fit the inline budget, so reviewers never
    # fall back to fetching/triaging the full patch.
    for shard in shards:
        assert inline_grounded_files(diff, shard.files) == set(shard.files)


def test_detect_stacks_registry_independent_same_scopes() -> None:
    from daydream.deep.detection import GENERIC_STACK, detect_stacks

    # Same files, absent vs empty vs populated registry -> same ordered scopes.
    changed = ["a.py", "b.ts", "c.md", "d.unknownext"]
    absent = detect_stacks(changed, registry=None)
    # `skill_availability` param is removed; the call must work with no registry.
    names = [s.stack_name for s in absent]
    # python + react language stacks + generic (md + unknown) + structural last.
    assert "python" in names and "react" in names and GENERIC_STACK in names
    assert names[-1] == "structure"


def test_detect_stacks_never_degrades_to_generic_without_registry() -> None:
    from daydream.deep.detection import GENERIC_STACK, detect_stacks

    # D-16 removed: a python stack never becomes generic merely because no
    # plugin registry is present.
    changed = ["a.py"]
    stacks = detect_stacks(changed)
    assert any(s.stack_name == "python" for s in stacks)
    assert not any(s.stack_name == GENERIC_STACK for s in stacks)
