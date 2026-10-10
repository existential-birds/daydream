"""Stack ownership, sharding bounds, and import-graph routing."""
from pathlib import Path

import pytest

from daydream.deep.dependency import build_import_graph
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.sharding import shard_stacks
from daydream.extensions import Registry, StackRule


@pytest.mark.parametrize(("files", "stack_name", "member", "docs_only"),
    [pytest.param(["src/main.py"], "python", "src/main.py", None, id="python-main"),
        pytest.param(["src/App.tsx"], "react", "src/App.tsx", None, id="react-app"),
        pytest.param(["src/app.py", "migrations/001.sql"], "python", "migrations/001.sql", None,
            id="ambiguous-shortcut"),
        pytest.param(["backend/api/main.py", "backend/api/queries.sql", "frontend/App.tsx"], "python",
            "backend/api/queries.sql", None, id="ambiguous-nearest-ancestor"),
        pytest.param(["main.py", "App.tsx", "shared.sql"], "generic", "shared.sql", None, id="equal-depth-fallthrough"),
        pytest.param(["pyproject.toml", "src/main.py"], "python", "pyproject.toml", None, id="config-promotion"),
        pytest.param(["src/main.py", "README.md"], "generic", "README.md", False, id="md-pinned-to-generic"),
    ],
)
def test_stack_membership_routing(files: list[str], stack_name: str, member: str, docs_only: bool | None) -> None:
    """detect_stacks routes each file to the owned stack, and docs-only follows the mixed diff."""
    result = detect_stacks(files)
    stack = next(a for a in result if a.stack_name == stack_name)
    assert member in stack.files
    if docs_only is not None:
        assert stack.is_docs_only is docs_only


def test_mixed_frontend_and_repository_infrastructure_routes_separately() -> None:
    """Shelfspace #2826's Node/CI changes must not inflate the React review."""
    frontend = ["frontend/app/components/modals/__tests__/CreateShelfModal.test.tsx",
        "frontend/app/components/shelf/ShelfNameInput.tsx",
        "frontend/tests/components/taste-reveal/RevealFlowContainer.test.tsx",
    ]
    infrastructure = [".github/workflows/daydream.yml", "Makefile", "docs/README.md",
        "docs/daydream-review.md", "quality-workspaces.json",
        "scripts/daydream-workflow.test.mjs", "scripts/github-actions-workflow-policy.mjs",
    ]
    files = frontend + infrastructure
    stacks = {stack.stack_name: stack.files for stack in detect_stacks(files)}
    assert set(stacks["react"]) == set(frontend)
    assert set(stacks["generic"]) == set(infrastructure)
    assert set(stacks["structure"]) == set(files)

def test_infrastructure_defaults_preserve_explicit_and_nested_ownership() -> None:
    registry = Registry()
    registry.add_stack(StackRule("build", ("scripts/custom.cjs",)))
    react_files = ["frontend/App.tsx", "package.json", "tsconfig.json",
        "frontend/fixtures/data.json", "frontend/Makefile", "frontend/helpers.mjs",
    ]
    generic_files = ["Makefile", "quality.json", "scripts/check.cjs", "scripts/ci/check.mjs"]
    stacks = {stack.stack_name: stack.files
        for stack in detect_stacks(react_files + generic_files + ["scripts/custom.cjs"], registry=registry)
    }
    assert set(stacks["react"]) == set(react_files)
    assert set(stacks["generic"]) == set(generic_files)
    assert stacks["build"] == ["scripts/custom.cjs"]


@pytest.mark.parametrize(("files", "expected"),
    [pytest.param(["config.yaml"], {"generic"}, id="D-13a-config-default-generic"),
        pytest.param(["pyproject.toml"], {"generic"}, id="D-13c-no-static-promotion"),
        pytest.param(["src/lib.rs"], {"rust"}, id="M3-never-degrades-to-generic"),
    ],
)
def test_language_classification(files: list[str], expected: set[str]) -> None:
    """D-13a/D-13c/M3: config paths default to generic; a detected language never degrades."""
    result = detect_stacks(files)
    language_names = {a.stack_name for a in result if a.stack_name != "structure"}
    assert language_names == expected


def test_no_files_dropped() -> None:
    files = ["src/main.py", "README.md", "config.yaml", "Dockerfile", "src/App.tsx"]
    result = detect_stacks(files)
    routed = {f for a in result for f in a.files}
    assert routed == set(files)

def test_shard_stacks_fanout_cap_limits_total_tasks() -> None:
    """The fan-out cap includes shards and unsplit stacks; all files remain assigned once."""
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
    """One oversized file stays unsplit with its original name, leaving cap reduction to other stacks."""

    # big.py exceeds the byte cap but cannot split mid-file.
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
    """Distinct unsplit stacks set the minimum fan-out; enforcing a lower cap cannot drop or merge them."""
    stacks = [StackAssignment(stack_name=f"s{i}", files=[f"f{i}.py"]) for i in range(18)]
    out = shard_stacks(stacks, "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8)
    # No shard exists to un-split; the floor is the distinct-stack count.
    assert len(out) == 18
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted([f"f{i}.py" for i in range(18)])

def test_shard_stacks_co_locates_dependent_files_when_room(tmp_path: Path) -> None:
    """Use a real, non-adjacent import edge so sorted singleton packing cannot pass accidentally."""

    # The {a,d} dependency component fits a shard despite non-adjacent sorted names.
    stack = StackAssignment(stack_name="python", files=["a.py", "b.py", "c.py", "d.py"])
    root = Path(tmp_path)
    for name in ("a.py", "b.py", "c.py"):
        (root / name).write_text("x = 1\n")
    (root / "d.py").write_text("import a\n")
    graph = build_import_graph(["a.py", "b.py", "c.py", "d.py"], root)
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8, graph=graph)
    shards = [s for s in out if s.stack_name.startswith("python#")]
    edge_shard = next(s for s in shards if "d.py" in s.files)
    assert "a.py" in edge_shard.files            # co-located when size permits

def test_shard_stacks_fail_open_without_graph() -> None:
    """Missing edges fall back to deterministic packing with exactly one assignment per file."""
    stack = StackAssignment(stack_name="python", files=[f"m{i}.py" for i in range(5)])
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8, graph={})
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted(stack.files)   # no graph -> every file still assigned once
    # A file with no resolvable edge still gets exactly one assignment (fallback).
    assert len(set(union)) == len(union)  # no duplicate primary assignment

def test_build_import_graph_resolves_python_edges(tmp_path: Path) -> None:
    """Resolve absolute and relative Python imports; unknown grammars remain singleton nodes."""
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
    """Resolve TypeScript, Go, and Rust edges as well as Python."""
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


@pytest.mark.parametrize('cap', [0, 1, 3, 4])
def test_coarsening_merges_adjacent_lowest_weights_and_recomputes_frontiers(cap: int) -> None:
    py = StackAssignment(stack_name='python', files=['a.py', 'b.py', 'c.py', 'd.py'])
    other = StackAssignment(stack_name='generic', files=['README.md'])
    diff = ''.join(f'diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n+{body}\n'
                   for path, body in [('a.py', 'x' * 500), ('b.py', 'x'), ('c.py', 'x'), ('d.py', 'x' * 600)])
    result = shard_stacks([py, other], diff, max_files=1, max_bytes=10000, fanout_cap=cap,
                          frontier_max=8, graph={'b.py': {'c.py', 'README.md'}})
    python = [stack for stack in result if stack.stack_name.startswith('python')]
    expected = ([['a.py', 'b.py', 'c.py', 'd.py']] if cap <= 1 else
                [['a.py', 'b.py', 'c.py'], ['d.py']] if cap == 3 else
                [['a.py'], ['b.py', 'c.py'], ['d.py']])
    assert [stack.files for stack in python] == expected
    assert [stack.stack_name for stack in python] == (
        ['python'] if len(expected) == 1 else [f'python#{i}' for i in range(len(expected))])
    assert next(stack for stack in python if 'b.py' in stack.files).frontier_files == ['README.md']
    assert sum(len(stack.files) for stack in result) == 5
