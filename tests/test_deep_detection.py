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

    # Only React is present: repository infrastructure cannot hide behind another language.
    root = tmp_path / "frontend-only"
    frontend = ["frontend/app/components/__tests__/Modal.test.tsx", "frontend/app/shelf/Input.tsx", "frontend/App.tsx"]
    react_files = [*frontend, "package.json", "tsconfig.json", "frontend/fixtures/data.json", "frontend/Makefile"]
    infrastructure = [".github/workflows/daydream.yml", "Makefile", "docs/README.md", "quality.json",
                      "scripts/check.cjs", "scripts/ci/check.mjs", "frontend/helpers.mjs"]
    for path in frontend:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text('import React from "react"; export const View = () => <div/>;')
    (root / "package.json").write_text('{"dependencies":{"react":"*"}}')
    (root / "tsconfig.json").write_text('{"compilerOptions":{"jsx":"react-jsx"}}')
    stacks = {stack.stack_name: stack.files for stack in detect_stacks(react_files + infrastructure, target_dir=root)}
    assert (set(stacks["react"]), set(stacks["generic"])) == (set(react_files), set(infrastructure))


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


def test_framework_missing_safe_open_capability_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable safe-open flags retain generic source/config coverage without unsafe reads."""
    (tmp_path / "App.tsx").write_text('import React from "react"; const App = () => <div/>;')
    (tmp_path / "package.json").write_text('{"dependencies":{"react":"*"}}')
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions":{"jsx":"react-jsx"}}')
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    files = ["App.tsx", "package.json", "tsconfig.json"]
    result = detect_stacks(files, target_dir=tmp_path)
    primary = [stack for stack in result if stack.stack_name != "structure"]
    assert [(stack.stack_name, stack.files) for stack in primary] == [("generic", files)]
    assert Counter(path for stack in primary for path in stack.files) == Counter(files)
    assert next(stack.files for stack in result if stack.stack_name == "structure") == files


def test_framework_inspection_budget_boundaries(tmp_path: Path) -> None:
    """Exact defaults admit complete evidence; the first over-budget file receives generic coverage."""
    for limit in ("source-count", "source-bytes", "metadata-count"):
        root = tmp_path / limit
        root.mkdir()
        files = []
        if limit == "metadata-count":
            for index in range(129):
                directory = root / f"pkg{index:03}"
                directory.mkdir()
                (directory / "package.json").write_text('{"dependencies":{"react":"*"}}')
                (directory / "App.tsx").write_text("const App = () => <div/>;")
                files.append(f"{directory.name}/App.tsx")
        for index in range(0 if limit == "metadata-count" else 257 if limit == "source-count" else 2):
            path = f"source{index:03}.ts"
            text = 'import React from "react";\n'
            if limit == "source-bytes":
                size = 256 * 1024 + index
                text += "//" + " " * (size - len(text) - 2)
            (root / path).write_text(text)
            files.append(path)
        # Reverse caller input: budgets and output must follow sorted paths.
        result = detect_stacks(list(reversed(files)), target_dir=root)
        scopes = {s.stack_name: s.files for s in result}
        assert scopes["react"] == files[:-1]
        assert scopes["generic"] == files[-1:]
        assert Counter(f for s, paths in scopes.items() if s != "structure" for f in paths) == Counter(files)


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


@pytest.mark.parametrize('cap', [3, ])
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
