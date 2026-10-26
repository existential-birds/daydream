"""Stack ownership, sharding bounds, and import-graph routing."""

import json
import os
from collections import Counter
from pathlib import Path

import pytest

from daydream.deep.dependency import build_import_graph
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.sharding import shard_stacks
from daydream.extensions import Registry, StackRule


def test_mixed_repository_routes_source_metadata_and_extension_evidence(tmp_path: Path) -> None:
    """Each package supplies its own evidence; one changed inventory retains every file exactly once."""
    sources = {
        "static-import": 'import {useState} from "react";',
        "require": 'const React = require("react");',
        "jsx-runtime": 'import {jsx} from "react/jsx-runtime";',
        "jsx-package": "export const App = () => <div/>;",
        "jsx-config": "export const App = () => <div/>;",
        "pragma": "/** @jsxImportSource react */ const App = () => <div/>;",
        "mode-only": "/** @jsxRuntime automatic */ const App = () => <div/>;",
        "mode-package": "/** @jsxRuntime automatic */ const App = () => <div/>;",
        "mode-config": "/** @jsxRuntime automatic */ const App = () => <div/>;",
        "mode-preact": "/** @jsxRuntime automatic */ const App = () => <div/>;",
        "import-equals": 'import React = require("react");',
        "mixed-import-equals": 'import React from "react"; import Vue = require("vue");',
        "both-import-equals": 'import React = require("react"); import Vue = require("vue");',
        "angular": 'import {Component} from "@angular/core";',
        "vue": 'import {ref} from "vue";',
        "svelte": 'import {onMount} from "svelte";',
        "preact": "/** @jsxImportSource preact */ const App = () => <div/>;",
        "mixed-import": 'import React from "react"; import {ref} from "vue";',
        "mixed-package": "export const App = () => <div/>;",
        "config-wins": "export const App = () => <div/>;",
        "non-jsx": "export const twice = (n: number) => n * 2;",
        "decoy": '// import React from "react";\nconst text = `import React from "react"`;',
        "jsx-alone": "export const App = () => <div/>;",
    }
    metadata: dict[str, object] = {f"cases/{name}/package.json": {} for name in sources}
    for name in ("jsx-package", "mode-package", "non-jsx", "config-wins"):
        metadata[f"cases/{name}/package.json"] = {"dependencies": {"react": "*"}}
    metadata["cases/mixed-package/package.json"] = {"dependencies": {"react": "*", "vue": "*"}}
    for name in ("jsx-config", "mode-config"):
        metadata[f"cases/{name}/tsconfig.json"] = {"compilerOptions": {"jsx": "react-jsx"}}
    for name in ("mode-preact", "config-wins"):
        metadata[f"cases/{name}/tsconfig.json"] = {"compilerOptions": {"jsxImportSource": "preact"}}
    bodies = {f"cases/{name}/App.tsx": source for name, source in sources.items()}
    bodies.update({name: json.dumps(value) for name, value in metadata.items()})
    bodies.update(
        {
            "backend/api/main.py": "VALUE = 1\n",
            "backend/api/queries.sql": "SELECT 1;\n",
            "pyproject.toml": "[project]\n",
            "migrations/001.sql": "SELECT 1;\n",
            "src/lib.rs": "pub fn run() {}\n",
            "README.md": "# Changed docs\n",
            "config.yaml": "enabled: true\n",
            "Dockerfile": "FROM scratch\n",
            "shared.sql": "SELECT 1;\n",
            "scripts/custom.cjs": "// Explicit extension owner\n",
            "unowned/App.tsx": "export const App = () => <div/>;",
        }
    )
    for name, body in bodies.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    registry = Registry()
    registry.add_stack(StackRule("build", ("scripts/custom.cjs",)))

    standalone = detect_stacks(["config.yaml", "pyproject.toml"], target_dir=tmp_path)
    assert [(stack.stack_name, stack.files) for stack in standalone] == [
        ("generic", ["config.yaml", "pyproject.toml"]),
        ("structure", ["config.yaml", "pyproject.toml"]),
    ]
    # A package-only change owns its ambiguous migration before other stacks join the diff.
    initial = detect_stacks(["pyproject.toml", "backend/api/main.py", "migrations/001.sql"], target_dir=tmp_path)
    assert [(s.stack_name, s.files) for s in initial] == [
        ("python", ["backend/api/main.py", "migrations/001.sql", "pyproject.toml"]),
        ("structure", ["backend/api/main.py", "migrations/001.sql", "pyproject.toml"]),
    ]
    result = detect_stacks(list(reversed(bodies)), target_dir=tmp_path, registry=registry)
    react_names = {
        "static-import",
        "require",
        "jsx-runtime",
        "jsx-package",
        "jsx-config",
        "pragma",
        "mode-package",
        "mode-config",
        "import-equals",
    }
    react = {f"cases/{name}/App.tsx" for name in react_names} | {
        "cases/jsx-package/package.json",
        "cases/mode-package/package.json",
        "cases/non-jsx/package.json",
        "cases/config-wins/package.json",
        "cases/jsx-config/tsconfig.json",
        "cases/mode-config/tsconfig.json",
    }
    python = {"backend/api/main.py", "backend/api/queries.sql", "pyproject.toml"}
    expected = {
        "build": ["scripts/custom.cjs"],
        "python": sorted(python),
        "react": sorted(react),
        "rust": ["src/lib.rs"],
        "generic": sorted(set(bodies) - react - python - {"scripts/custom.cjs", "src/lib.rs"}),
        "structure": sorted(bodies),
    }
    assert {stack.stack_name: stack.files for stack in result} == expected
    assert all(not stack.is_docs_only for stack in result)
    primary = [path for stack in result if stack.stack_name != "structure" for path in stack.files]
    assert Counter(primary) == Counter(bodies.keys())


def test_unavailable_package_evidence_cannot_borrow_a_valid_siblings_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each failed source/metadata read has its own generic scope beside a real React source."""
    packages = (
        "deleted",
        "malformed",
        "invalid-workspace-date",
        "cycle",
        "external-extends",
        "unreadable",
        "metadata-symlink",
        "escape",
    )
    bodies = {f"{name}/App.tsx": "export const App = () => <div/>;" for name in packages if name != "deleted"}
    bodies.update({f"{name}/package.json": '{"dependencies":{"react":"*"}}' for name in packages})
    bodies.update(
        {
            "react.ts": 'import React from "react";',
            "malformed/package.json": "{broken",
            "invalid-workspace-date/pnpm-workspace.yaml": 'packages: ["apps/*"]\ninvalid: 2026-99-99\n',
            "cycle/tsconfig.json": '{"extends":"./base.json"}',
            "cycle/base.json": '{"extends":"./tsconfig.json"}',
            "external-extends/tsconfig.json": '{"extends":"framework-config"}',
        }
    )
    for name, body in bodies.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    outside_source = tmp_path.parent / f"{tmp_path.name}-outside.tsx"
    outside_source.write_text('import React from "react";')
    outside_metadata = tmp_path.parent / f"{tmp_path.name}-package.json"
    outside_metadata.write_text('{"dependencies":{"react":"*"}}')
    escaped = tmp_path / "escape/App.tsx"
    escaped.unlink()
    escaped.symlink_to(outside_source)
    metadata_link = tmp_path / "metadata-symlink/package.json"
    metadata_link.unlink()
    metadata_link.symlink_to(outside_metadata)
    original_open = os.open
    opened: list[Path] = []

    def inspect_open(candidate: Path, flags: int) -> int:
        opened.append(candidate)
        if candidate == tmp_path / "unreadable/package.json":
            raise PermissionError("package unavailable")
        return original_open(candidate, flags)

    monkeypatch.setattr("daydream.deep.framework_detection.os.open", inspect_open)
    changed = ["react.ts", *(f"{name}/App.tsx" for name in packages), f"../{outside_source.name}"]
    result = detect_stacks(changed, target_dir=tmp_path)
    assert {stack.stack_name: stack.files for stack in result} == {
        "react": ["react.ts"],
        "generic": sorted(changed[1:]),
        "structure": sorted(changed),
    }
    assert tmp_path / "unreadable/package.json" in opened
    assert not {escaped, metadata_link, outside_source, outside_metadata}.intersection(opened)


def test_workspace_inventory_keeps_each_unowned_child_generic(tmp_path: Path) -> None:
    """One changed inventory crosses independent JSON/YAML workspace and unsupported-glob boundaries."""
    patterns = [
        "apps/web",
        "apps/*",
        "apps/**",
        "apps/w?b",
        "apps/[wm]*",
        "apps/{web,mobile}",
        "apps/@(web|mobile)",
        r"apps\*",
    ]
    changed = []
    for kind in ("package.json", "pnpm-workspace.yaml"):
        for index, pattern in enumerate(patterns):
            root = tmp_path / f"{kind}-{index}"
            (root / "apps/web").mkdir(parents=True)
            package: dict[str, object] = {"dependencies": {"react": "*"}}
            if kind == "package.json":
                package["workspaces"] = [pattern]
            else:
                (root / kind).write_text("packages: " + json.dumps([pattern]) + "\n")
            (root / "package.json").write_text(json.dumps(package))
            (root / "apps/web/App.tsx").write_text("const App = () => <div/>;")
            changed.append(f"{root.name}/apps/web/App.tsx")
    healthy = tmp_path / "healthy"
    healthy.mkdir()
    (healthy / "package.json").write_text('{"dependencies":{"react":"*"}}')
    (healthy / "App.tsx").write_text("const App = () => <div/>;")

    result = detect_stacks([*reversed(changed), "healthy/App.tsx"], target_dir=tmp_path)
    assert {stack.stack_name: stack.files for stack in result} == {
        "react": ["healthy/App.tsx"],
        "generic": sorted(changed),
        "structure": sorted([*changed, "healthy/App.tsx"]),
    }


def test_mixed_frontend_and_repository_infrastructure_routes_separately(tmp_path: Path) -> None:
    """Only React is present: repository defaults cannot hide behind another language."""
    frontend = ["frontend/app/components/modals/__tests__/CreateShelfModal.test.tsx",
        "frontend/app/components/shelf/ShelfNameInput.tsx",
        "frontend/tests/components/taste-reveal/RevealFlowContainer.test.tsx", "frontend/App.tsx",
    ]
    react_files = frontend + ["package.json", "tsconfig.json", "frontend/fixtures/data.json", "frontend/Makefile"]
    infrastructure = [".github/workflows/daydream.yml", "Makefile", "docs/README.md",
        "docs/daydream-review.md", "quality-workspaces.json", "quality.json",
        "scripts/daydream-workflow.test.mjs", "scripts/github-actions-workflow-policy.mjs",
        "scripts/check.cjs", "scripts/ci/check.mjs", "frontend/helpers.mjs",
    ]
    for path in frontend:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('import React from "react"; export const View = () => <div/>;')
    (tmp_path / "package.json").write_text('{"dependencies":{"react":"*"}}')
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions":{"jsx":"react-jsx"}}')
    files = react_files + infrastructure
    stacks = {stack.stack_name: stack.files for stack in detect_stacks(files, target_dir=tmp_path)}
    assert set(stacks["react"]) == set(react_files)
    assert set(stacks["generic"]) == set(infrastructure)
    assert set(stacks["structure"]) == set(files)



def test_local_packages_and_explicit_config_rules(tmp_path: Path) -> None:
    """A sibling React file and root dependency cannot overwrite another package or an explicit rule."""
    bodies = {
        "package.json": {"workspaces": ["packages/*"], "dependencies": {"react": "*"}},
        "tsconfig.json": {"compilerOptions": {"jsx": "react-jsx"}},
        "packages/react/package.json": {"dependencies": {"react": "*"}},
        "packages/vue/package.json": {"dependencies": {"vue": "*"}},
        "packages/node/package.json": {},
        "packages/web/package.json": {},
        "packages/web/tsconfig.json": {"extends": "../../tsconfig.json"},
        "packages/react/tsconfig.json": {"compilerOptions": {"jsx": "react-jsx"}},
        "packages/vue/tsconfig.json": {"compilerOptions": {"jsxImportSource": "vue"}},
    }
    files = list(bodies)
    for path, body in bodies.items():
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(body))
    for package in ("react", "vue", "node", "web"):
        path = f"packages/{package}/App.tsx"
        (tmp_path / path).write_text("export const App = () => <div/>;")
        files.append(path)
    registry = Registry()
    registry.add_stack(StackRule("custom", ("packages/react/tsconfig.json", "packages/node/package.json")))
    scopes = {s.stack_name: s.files for s in detect_stacks(files, target_dir=tmp_path, registry=registry)}
    assert scopes["react"] == [
        "package.json", "packages/react/App.tsx", "packages/react/package.json",
        "packages/web/App.tsx", "packages/web/tsconfig.json", "tsconfig.json",
    ]
    assert scopes["custom"] == ["packages/node/package.json", "packages/react/tsconfig.json"]
    assert scopes["generic"] == sorted(set(files) - set(scopes["react"]) - set(scopes["custom"]))
    assert Counter(f for s, paths in scopes.items() if s != "structure" for f in paths) == Counter(files)


@pytest.mark.parametrize("flag", ["O_NOFOLLOW", "O_NONBLOCK"])
def test_framework_missing_safe_open_capability_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str,
) -> None:
    """Unavailable safe-open flags retain generic source/config coverage without unsafe reads."""
    (tmp_path / "App.tsx").write_text('import React from "react"; const App = () => <div/>;')
    (tmp_path / "package.json").write_text('{"dependencies":{"react":"*"}}')
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions":{"jsx":"react-jsx"}}')
    monkeypatch.delattr(os, flag, raising=False)
    files = ["App.tsx", "package.json", "tsconfig.json"]
    result = detect_stacks(files, target_dir=tmp_path)
    primary = [stack for stack in result if stack.stack_name != "structure"]
    assert [(stack.stack_name, stack.files) for stack in primary] == [("generic", files)]
    assert Counter(path for stack in primary for path in stack.files) == Counter(files)
    assert next(stack.files for stack in result if stack.stack_name == "structure") == files



@pytest.mark.parametrize("limit", ["source-count", "source-bytes", "metadata-count", "metadata-bytes", "total-bytes"])
def test_framework_inspection_budget_boundaries(tmp_path: Path, limit: str) -> None:
    """Exact defaults admit complete evidence; the first over-budget file receives generic coverage."""
    files = []
    expected_react = []
    expected_generic = []
    if limit in {"source-count", "source-bytes", "total-bytes"}:
        count = 257 if limit == "source-count" else 17 if limit == "total-bytes" else 2
        for index in range(count):
            path = f"source{index:03}.ts"
            text = 'import React from "react";\n'
            if limit != "source-count":
                size = 256 * 1024 + (index if limit == "source-bytes" else 0)
                text += "//" + " " * (size - len(text) - 2)
            (tmp_path / path).write_text(text)
            files.append(path)
        expected_react = files[:-1]
        expected_generic = files[-1:]
    else:
        count = 129 if limit == "metadata-count" else 2
        for index in range(count):
            directory = tmp_path / f"pkg{index:03}"
            directory.mkdir()
            metadata = '{"dependencies":{"react":"*"}}'
            if limit == "metadata-bytes":
                metadata += " " * (64 * 1024 + index - len(metadata))
            (directory / "package.json").write_text(metadata)
            (directory / "App.tsx").write_text("const App = () => <div/>;")
            files.append(f"{directory.name}/App.tsx")
        expected_react = files[:-1]
        expected_generic = files[-1:]
    # Reverse caller input: budgets and output must follow sorted paths.
    result = detect_stacks(list(reversed(files)), target_dir=tmp_path)
    scopes = {s.stack_name: s.files for s in result}
    assert scopes["react"] == expected_react
    assert scopes["generic"] == expected_generic
    assert Counter(f for s, paths in scopes.items() if s != "structure" for f in paths) == Counter(files)


def test_explicit_config_rule_survives_language_promotion() -> None:
    registry = Registry()
    registry.add_stack(StackRule("build", ("pyproject.toml", "package.json", "tsconfig.json")))
    files = ["main.py", "pyproject.toml", "package.json", "tsconfig.json"]
    scopes = {s.stack_name: s.files for s in detect_stacks(files, registry=registry)}
    assert scopes["build"] == ["package.json", "pyproject.toml", "tsconfig.json"]
    assert scopes["python"] == ["main.py"]



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
    go = StackAssignment(stack_name="go", files=[f"g{i}.go" for i in range(12)])
    out = shard_stacks([huge, many, go], big_diff, max_files=2, max_bytes=100, fanout_cap=4, frontier_max=8)
    # The single-shard stack stays unsplit under its original name.
    assert any(s.stack_name == "python" and s.files == ["big.py"] for s in out)
    # Total tasks (shards + unsplit) never exceed the cap.
    assert len(out) <= 4
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted(["big.py"] + [f"r{i}.rs" for i in range(12)] + [f"g{i}.go" for i in range(12)])
    rust_files = [f for s in out if s.stack_name.partition("#")[0] == "rust" for f in s.files]
    go_files = [f for s in out if s.stack_name.partition("#")[0] == "go" for f in s.files]
    assert sorted(rust_files) == sorted(f"r{i}.rs" for i in range(12))
    assert sorted(go_files) == sorted(f"g{i}.go" for i in range(12))


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
    assert "a.py" in edge_shard.files  # co-located when size permits


def test_shard_stacks_fail_open_without_graph() -> None:
    """Missing edges fall back to deterministic packing with exactly one assignment per file."""
    stack = StackAssignment(stack_name="python", files=[f"m{i}.py" for i in range(5)])
    out = shard_stacks([stack], "", max_files=2, max_bytes=10**9, fanout_cap=16, frontier_max=8, graph={})
    union = [f for s in out for f in s.files]
    assert sorted(union) == sorted(stack.files)  # no graph -> every file still assigned once
    # A file with no resolvable edge still gets exactly one assignment (fallback).
    assert len(set(union)) == len(union)  # no duplicate primary assignment


def test_build_import_graph_resolves_python_edges(tmp_path: Path) -> None:
    """One native graph resolves each language and retains no-edge/unknown siblings."""
    bodies = {
        "a.py": "import b\n", "b.py": "x = 1\n",
        "pkg/mod.py": "from .util import helper\n", "pkg/util.py": "def helper(): pass\n",
        "notes.txt": "no grammar\n",
        "a.ts": 'import { b } from "./b"\n', "b.ts": "export const b = 1;\n",
        "a.go": 'package a\nimport "b"\n', "b.go": "package b\n",
        "a.rs": "use b::c;\n", "b.rs": "pub fn c() {}\n",
    }
    for name, source in bodies.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    graph = build_import_graph(list(bodies), tmp_path)
    assert graph == {
        "a.py": {"b.py"}, "b.py": set(),
        "pkg/mod.py": {"pkg/util.py"}, "pkg/util.py": set(), "notes.txt": set(),
        "a.ts": {"b.ts"}, "b.ts": set(), "a.go": {"b.go"}, "b.go": set(),
        "a.rs": {"b.rs"}, "b.rs": set(),
    }



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
