"""Unit tests for daydream.tree_sitter_index.detect_affected_files()."""

import inspect
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.tree_sitter_index import (
    _MAX_IMPORTERS,
    BRANCH_NODE_TYPES,
    PYTHON_DEF_QUERY,
    TERMINAL_CALL_NAMES,
    TERMINAL_MACRO_NAMES,
    TERMINAL_NODE_TYPES,
    _def_query_for_language,
    _diagram_def_query_for_language,
    _walk,
    branch_statement_lines,
    definitions_in_file,
    detect_affected_files,
    get_parser,
    is_branch_line,
    is_terminal_line,
    language_for_path,
)
from tests.conftest import _make_repo_with_main
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import configure_identity as _configure_identity
from tests.harness.git_helpers import git as _git

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


def _modified_diff(path: str) -> str:
    """Minimal unified diff marking *path* as modified."""
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1,2 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )


def _importers(results: Any) -> set[str]:
    return {r.path for r in results if r.role == "imported_by"}


def _materialize(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create files under tmp_path with the given relative paths and contents."""
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


def test_detect_affected_files_rows_are_static_provenance(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "widget.py").write_text("x = 1\n")
    (tmp_path / "app.py").write_text("import pkg.widget\n")
    results = detect_affected_files(_modified_diff("app.py"), tmp_path, depth=1)
    assert results, "static resolution must produce rows"
    assert all(f.provenance == "static" for f in results)


def test_python_impact_surface(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "daydream_demo/__init__.py": "",
            "daydream_demo/api.py": (
                '"""API module."""\nfrom .models import User\n\ndef get_user():\n    return User()\n'
            ),
            "daydream_demo/models.py": '"""Models module."""\n\nclass User:\n    pass\n',
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    paths_by_role = {(r.path, r.role) for r in results}
    assert ("daydream_demo/api.py", "modified") in paths_by_role
    assert ("daydream_demo/models.py", "modified") in paths_by_role
    # The api.py -> models.py forward edge must be exact.
    assert ("daydream_demo/models.py", "imports") in paths_by_role
    assert len(results) >= 2


_SHARED_FIXTURES: dict[str, str] = {
    "package/__init__.py": "",
    "package/models.py": '"""Models module."""\n\nclass User:\n    pass\n',
    "package/feature/__init__.py": "",
}


@pytest.mark.parametrize(
    "api_rel, files, expected_import_path",
    [
        # `from ..models` in package/feature/api.py resolves to the parent package.
        (
            "package/feature/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/api.py": (
                    '"""API module."""\nfrom ..models import User\n\ndef get_user():\n    return User()\n'
                ),
            },
            "package/models.py",
        ),
        # `from ...models` in package/feature/nested/api.py resolves to the grandparent package.
        (
            "package/feature/nested/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/nested/__init__.py": "",
                "package/feature/nested/api.py": (
                    '"""API module."""\nfrom ...models import User\n\ndef get_user():\n    return User()\n'
                ),
            },
            "package/models.py",
        ),
        # `from . import something` resolves to the current package's __init__.py
        # AND the sibling module package/feature/something.py, because the
        # capture now spans the whole statement, not just the relative import
        # node (the dot).
        (
            "package/feature/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/api.py": (
                    '"""API module."""\nfrom . import something\n\nthing = something.thing\n'
                ),
                "package/feature/something.py": "",
            },
            {"package/feature/__init__.py", "package/feature/something.py"},
        ),
        # `from ....something import name` with 4+ dots ascends to the great-grandparent package.
        (
            "package/feature/nested/deep/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/nested/__init__.py": "",
                "package/feature/nested/deep/__init__.py": "",
                "package/something.py": '"""Ancestor sibling module."""\n\nthing = 1\n',
                "package/feature/nested/deep/api.py": (
                    '"""API module."""\nfrom ....something import thing\n\nthing\n'
                ),
            },
            "package/something.py",
        ),
    ],
)
def test_python_multilevel_relative_imports(
    tmp_path: Path,
    api_rel: str,
    files: dict[str, str],
    expected_import_path: str | set[str],
) -> None:
    repo = _materialize(tmp_path, files)
    results = detect_affected_files(_modified_diff(api_rel), repo, depth=1)
    imports_pairs = {(r.path, r.role) for r in results if r.role == "imports"}
    if isinstance(expected_import_path, str):
        assert imports_pairs == {(expected_import_path, "imports")}
    else:
        assert imports_pairs == {(p, "imports") for p in expected_import_path}


@pytest.mark.parametrize(
    "api_rel, files, expected_import_paths",
    [
        # R3 primary fix: `from .. import services` in package/feature/api.py
        # resolves to BOTH the parent package __init__.py AND the imported
        # sibling package's __init__.py — and must NOT include the importer's
        # own package/feature/__init__.py (C3).
        pytest.param(
            "package/feature/api.py",
            _SHARED_FIXTURES
            | {
                "package/services/__init__.py": "",
                "package/feature/api.py": (
                    '"""API module."""\nfrom .. import services\n'
                ),
            },
            {"package/__init__.py", "package/services/__init__.py"},
            id="parent-package-import",
        ),
    ],
)
def test_python_parent_relative_imports(
    tmp_path: Path,
    api_rel: str,
    files: dict[str, str],
    expected_import_paths: set[str],
) -> None:
    repo = _materialize(tmp_path, files)
    results = detect_affected_files(_modified_diff(api_rel), repo, depth=1)
    imports_paths = {r.path for r in results if r.role == "imports"}
    assert imports_paths == expected_import_paths


def test_typescript_impact_surface(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "typescript_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "src/api.ts": (
                '// API module\nimport { User } from "./models";\n\n'
                "export function getUser(): User {\n  return new User();\n}\n"
            ),
            "src/models.ts": "// Models module\n\nexport class User {}\n",
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    assert any(r.path == "src/api.ts" and r.role == "modified" for r in results)
    assert any(r.path == "src/models.ts" and r.role == "modified" for r in results)
    assert ("src/models.ts", "imports") in {
        (r.path, r.role) for r in results
    }


def test_go_imports_reuse_one_package_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _materialize(
        tmp_path,
        {
            "cmd/alpha.go": 'package main\n\nimport "example.com/models"\n',
            "cmd/beta.go": 'package main\n\nimport "example.com/helpers"\n',
            "models/user.go": "package models\n",
            "models/order.go": "package models\n",
            "helpers/format.go": "package helpers\n",
        },
    )
    real_rglob = Path.rglob
    calls = 0

    def one_go_traversal(self: Path, pattern: str) -> Any:
        nonlocal calls
        if self == repo and pattern == "*.go":
            calls += 1
            if calls > 1:
                raise AssertionError("repo_root.rglob('*.go') called more than once")
            return real_rglob(self, pattern)
        raise AssertionError(f"unexpected rglob: {pattern!r}")

    monkeypatch.setattr(Path, "rglob", one_go_traversal)

    results = detect_affected_files(
        _modified_diff("cmd/alpha.go") + _modified_diff("cmd/beta.go"), repo, depth=1
    )
    imports_paths = {r.path for r in results if r.role == "imports"}
    assert imports_paths == {"models/user.go", "models/order.go", "helpers/format.go"}
    assert {r.path for r in results if r.role == "modified"} == {"cmd/alpha.go", "cmd/beta.go"}


def test_rust_impact_surface(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "rust_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "src/api.rs": "// api module\nuse crate::models::User;\n\npub fn get_user() -> User {\n    User\n}\n",
            "src/models.rs": "// models module\n\npub struct User;\n",
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    assert any(r.path == "src/api.rs" and r.role == "modified" for r in results)
    assert any(r.path == "src/models.rs" and r.role == "modified" for r in results)
    assert ("src/models.rs", "imports") in {
        (r.path, r.role) for r in results
    }


def test_default_depth_is_one() -> None:
    sig = inspect.signature(detect_affected_files)
    assert sig.parameters["depth"].default == 1


def test_unsupported_language_gets_modified_role(tmp_path: Path) -> None:
    diff_text = (
        "diff --git a/lib/foo.rb b/lib/foo.rb\n"
        "index 1111111..2222222 100644\n"
        "--- a/lib/foo.rb\n"
        "+++ b/lib/foo.rb\n"
        "@@ -1,1 +1,2 @@\n"
        " class Foo\n"
        "+  def bar; end\n"
    )
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "foo.rb").write_text("class Foo\n  def bar; end\nend\n")
    results = detect_affected_files(diff_text, tmp_path, depth=1)
    assert len(results) == 1
    assert results[0].path == "lib/foo.rb"
    assert results[0].role == "modified"


def test_deleted_file_does_not_raise_filenotfound(tmp_path: Path) -> None:
    diff_text = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-print('bye')\n"
    )
    results = detect_affected_files(diff_text, tmp_path, depth=1)
    assert len(results) == 1
    assert results[0].path == "gone.py"
    assert results[0].role == "modified"


# --- Reverse-edge (importers) behavior: real git repo -----------------------


def test_reverse_edge_finds_code_importer(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "widget.py").write_text("x = 1\ny = 2\n")
    (repo / "caller.py").write_text("from pkg.widget import W\n\nW()\n")
    _git(repo, "add", "pkg/widget.py", "caller.py")
    _commit(repo, "add widget + caller")

    results = detect_affected_files(_modified_diff("pkg/widget.py"), repo, depth=1)
    assert "caller.py" in _importers(results)


def test_reverse_edge_skips_generic_stem(tmp_path: Path) -> None:
    # "app" is a generic stem: a bare grep would match unrelated prose/code.
    repo = _make_repo_with_main(tmp_path)
    (repo / "app.py").write_text("x = 1\ny = 2\n")
    (repo / "unrelated.py").write_text("# the app starts here\nrun_app()\n")
    _git(repo, "add", "app.py", "unrelated.py")
    _commit(repo, "add app + unrelated")

    results = detect_affected_files(_modified_diff("app.py"), repo, depth=1)
    assert _importers(results) == set()


def test_reverse_edge_excludes_non_code_files(tmp_path: Path) -> None:
    # A markdown/doc file cannot import a code module; it must never be an importer.
    repo = _make_repo_with_main(tmp_path)
    (repo / "widget.py").write_text("x = 1\ny = 2\n")
    (repo / "notes.md").write_text("The widget is documented here.\n")
    (repo / "caller.py").write_text("import widget\n")
    _git(repo, "add", "widget.py", "notes.md", "caller.py")
    _commit(repo, "add code + docs")

    importers = _importers(detect_affected_files(_modified_diff("widget.py"), repo, depth=1))
    assert "caller.py" in importers
    assert "notes.md" not in importers


def test_reverse_edge_capped_at_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo_with_main(tmp_path)
    for stem in ("widget", "gadget"):
        (repo / f"{stem}.py").write_text("x = 1\ny = 2\n")
    overshoot = _MAX_IMPORTERS + 5
    for stem in ("widget", "gadget"):
        for i in range(overshoot):
            (repo / f"{stem}_importer_{i}.py").write_text(f"import {stem}\n")
    _git(repo, "add", "widget.py", "gadget.py",
         *[f"{stem}_importer_{i}.py" for stem in ("widget", "gadget") for i in range(overshoot)])
    _commit(repo, "many importers")

    real_grep_fixed = git_ops.grep_fixed_matches
    calls = 0

    def one_batch(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("grep_fixed_matches called more than once")
        return real_grep_fixed(*args, **kwargs)

    def no_legacy_grep(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy git_ops.grep must not be called")

    monkeypatch.setattr(git_ops, "grep_fixed_matches", one_batch)
    monkeypatch.setattr(git_ops, "grep", no_legacy_grep)

    results = detect_affected_files(
        _modified_diff("widget.py") + _modified_diff("gadget.py"), repo, depth=1
    )
    importers = _importers(results)
    widget = {p for p in importers if p.startswith("widget_importer_")}
    gadget = {p for p in importers if p.startswith("gadget_importer_")}
    assert len(widget) == _MAX_IMPORTERS
    assert len(gadget) == _MAX_IMPORTERS
    assert len(importers) == _MAX_IMPORTERS * 2
    assert {r.path for r in results if r.role == "modified"} == {"widget.py", "gadget.py"}


# --- Symbol index (definitions + line numbers) ------------------------------


def test_symbol_index_python_records_file_and_line_range(tmp_path: Path) -> None:
    from daydream.tree_sitter_index import build_symbol_index

    (tmp_path / "widget.py").write_text(
        "def compute_total(x):\n    return x + 1\n\nclass Box:\n    pass\n"
    )
    (tmp_path / "config.py").write_text("SETTINGS = {}\ndef load():\n    return SETTINGS\n")
    idx = build_symbol_index(tmp_path, ["widget.py", "config.py"])
    assert idx["compute_total"] == [
        {"path": "widget.py", "line": 1, "end_line": 2, "kind": "function"}
    ]
    assert idx["Box"] == [
        # tree-sitter's ``class_definition`` node spans the body, so the
        # definition-range end is line 5 (the ``pass`` line), not the header.
        {"path": "widget.py", "line": 4, "end_line": 5, "kind": "class"}
    ]
    # config.py's generic stem does not matter to the index itself; it is
    # recorded like any other module that defines symbols.
    assert idx["load"] == [
        {"path": "config.py", "line": 2, "end_line": 3, "kind": "function"}
    ]


def test_symbol_index_rust_records_file_and_line_range(tmp_path: Path) -> None:
    from daydream.tree_sitter_index import build_symbol_index

    (tmp_path / "lib.rs").write_text(
        (Path(__file__).parent / "fixtures" / "symbols" / "lib.rs").read_text()
    )
    idx = build_symbol_index(tmp_path, ["lib.rs"])
    assert idx["total"] == [
        {"path": "lib.rs", "line": 1, "end_line": 3, "kind": "function"}
    ]
    assert idx["Widget"] == [
        {"path": "lib.rs", "line": 5, "end_line": 7, "kind": "class"}
    ]


def test_config_py_with_definition_receives_reverse_edges(tmp_path: Path) -> None:
    """A generic-stem file that actually defines a symbol must not be skipped
    by the reverse-import lookup (config.py -> app.py ``imported_by`` edge)."""
    (tmp_path / "config.py").write_text("def load_config():\n    return {}\n")
    (tmp_path / "app.py").write_text("import config\n")
    diff = (
        "diff --git a/config.py b/config.py\n--- a/config.py\n+++ b/config.py\n"
        "@@ -1 +1,2 @@\n x\n+y\n"
    )
    _git(tmp_path, "init", "-q")
    _configure_identity(tmp_path)
    _git(tmp_path, "add", ".")
    _commit(tmp_path, "init")
    results = detect_affected_files(diff, tmp_path, depth=1)
    assert any(r.path == "app.py" and r.role == "imported_by" for r in results)


# --- Shared version guard (issue #1087, M6) --------------------------------


def test_detect_affected_files_refuses_known_bad_tree_sitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1087 (M6): the shared guard covers every native-analysis entry point,
    not just the quality analyzer — index consumers refuse bad installs too."""
    from daydream import _tree_sitter_safety as safety

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.26.0")
    with pytest.raises(safety.TreeSitterBadVersionError):
        detect_affected_files(
            repo_root=tmp_path,
            diff_text=(
                "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
                "@@ -1 +1 @@\n-x\n+y\n"
            ),
        )


def test_get_parser_refuses_known_bad_tree_sitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1087 (M6): get_parser is the true Parser construction site, so the
    shared guard must fire there too — otherwise deep-sharding's
    build_import_graph can still construct a native parser on a bad install."""
    from daydream import _tree_sitter_safety as safety
    from daydream.tree_sitter_index import _PARSER_CACHE, get_parser

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.26.0")
    _PARSER_CACHE.clear()
    try:
        with pytest.raises(safety.TreeSitterBadVersionError):
            get_parser("python")
        assert "python" not in _PARSER_CACHE
    finally:
        _PARSER_CACHE.clear()


# --- Diagram grounding primitives (issue #1113) ----------------------------
#
# Every node type in ``BRANCH_NODE_TYPES``/``TERMINAL_NODE_TYPES`` and every name
# in ``TERMINAL_CALL_NAMES``/``TERMINAL_MACRO_NAMES`` is exercised below against a
# real parse of an inline source, with the asserted line numbers read off that
# source. ``test_*_table_entries_all_occur_in_probe_sources`` closes the loop by
# proving no table entry is a name the grammars never produce.

PYTHON_CF = b'''import os
import sys


def f(x, items):
    if x > 1:
        pass
    elif x > 0:
        pass
    else:
        pass
    for i in items:
        pass
    while x:
        break
    try:
        pass
    except ValueError:
        pass
    finally:
        pass
    with open("f") as fh:
        fh.close()
    match x:
        case 1:
            pass
        case _:
            pass
    y = 1 if x else 2
    z = x and y
    assert z
    if x:
        raise ValueError("no")
    if y:
        sys.exit(1)
    if z:
        os._exit(1)
    if items:
        exit(1)
    if not items:
        quit()
    return 0
'''

TYPESCRIPT_CF = b'''function f(a: boolean, b: boolean, items: number[], o?: { p?: number }) {
  if (a) {
    return 1;
  } else if (b) {
    return 2;
  } else {
    throw new Error("x");
  }
  for (let i = 0; i < 3; i++) {
    continue;
  }
  for (const it of items) {
    break;
  }
  while (a) {
    break;
  }
  do {
    break;
  } while (a);
  try {
    process.exit(1);
  } catch (e) {
    throw e;
  } finally {
    console.log("done");
  }
  const t = a ? 1 : 2;
  const q = o?.p;
  switch (t) {
    case 1:
      switch (q) {
        case 2:
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
  return t;
}
'''

TSX_CF = b'''export const Panel = ({ n }: { n: number }) => {
  if (n > 1) {
    return <div>big</div>;
  }
  return n > 0 ? <span>small</span> : null;
};
'''

JAVASCRIPT_CF = b'''function f(a, items) {
  if (a) {
    return 1;
  }
  for (const i of items) {
    if (i) continue;
  }
  process.exit(0);
}
'''

GO_CF = b'''package main

import (
\t"log"
\t"os"
\t"testing"
)

func f(a bool, b bool, ch chan int, v interface{}) int {
\tif a {
\t\treturn 1
\t} else if b {
\t\treturn 2
\t} else {
\t\treturn 3
\t}
\tfor i := 0; i < 3; i++ {
\t\tcontinue
\t}
\tswitch a {
\tcase true:
\t\tswitch b {
\t\tcase true:
\t\t\tos.Exit(1)
\t\tdefault:
\t\t\tlog.Fatal("x")
\t\t}
\tdefault:
\t\tpanic("x")
\t}
\tswitch t := v.(type) {
\tcase int:
\t\t_ = t
\t}
\tselect {
\tcase <-ch:
\t\treturn 4
\t}
\treturn 0
}

func g(t *testing.T) {
\tlog.Fatalf("%v", 1)
\tlog.Panic("x")
\tt.Fatal("x")
\tt.Fatalf("%v", 1)
}
'''

RUST_CF = b'''use std::fs;
use std::process;

fn f(a: bool, b: bool, v: Option<i32>, items: &[i32]) -> Result<i32, std::io::Error> {
    if a {
        return Ok(1);
    } else if b {
        return Ok(2);
    } else {
        std::process::exit(1);
    }
    while a {
        break;
    }
    for i in items {
        continue;
    }
    loop {
        break;
    }
    match v {
        Some(1) => panic!("x"),
        Some(_) => unreachable!(),
        None => todo!(),
    }
    if let Some(x) = v {
        println!("{}", x);
    }
    let _s = fs::read_to_string("f")?;
    if a {
        process::exit(1);
    }
    if b {
        std::process::abort();
    }
    process::abort();
    Ok(0)
}
'''

# ``BRANCH_NODE_TYPES`` coverage per language id. ``tsx``/``javascript`` reuse the
# TypeScript source: all three ids are served by the tree-sitter-typescript
# grammars, which is exactly the property under test.
_BRANCH_PROBE_SOURCES: dict[str, tuple[bytes, ...]] = {
    "python": (PYTHON_CF,),
    "typescript": (TYPESCRIPT_CF,),
    "tsx": (TYPESCRIPT_CF, TSX_CF),
    "javascript": (TYPESCRIPT_CF, JAVASCRIPT_CF),
    "go": (GO_CF,),
    "rust": (RUST_CF,),
}


def _node_types(language_id: str, source: bytes) -> set[str]:
    parser = get_parser(language_id)
    assert parser is not None
    return {node.type for node in _walk(parser.parse(source).root_node)}


def _terminal_lines(language_id: str | None, source: bytes) -> list[int]:
    rows = source.split(b"\n")
    return [n + 1 for n in range(len(rows)) if is_terminal_line(language_id, source, n + 1)]


def _defs(repo_root: Path, path: str) -> set[tuple[str, int, int, str]]:
    return {
        (str(d["name"]), int(str(d["line"])), int(str(d["end_line"])), str(d["kind"]))
        for d in definitions_in_file(repo_root, path)
    }


# --- language_for_path -------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/mod.py", "python"),
        ("src/a.ts", "typescript"),
        ("src/a.tsx", "tsx"),
        # .js is parsed by the TypeScript grammar, .jsx by the TSX grammar.
        ("src/a.js", "javascript"),
        ("src/a.jsx", "tsx"),
        ("cmd/main.go", "go"),
        ("src/lib.rs", "rust"),
        ("README.md", None),
        ("Makefile", None),
        ("src/a.PY", None),  # suffix matching is case-sensitive
    ],
)
def test_language_for_path_maps_supported_suffixes(path: str, expected: str | None) -> None:
    assert language_for_path(path) == expected


# --- definitions_in_file -----------------------------------------------------


def test_definitions_in_file_typescript_covers_every_diagram_pattern(tmp_path: Path) -> None:
    (tmp_path / "defs.ts").write_text(
        "export function alpha(): void {}\n"
        "export function* gen() {\n"
        "  yield 1;\n"
        "}\n"
        "export const beta = (x: number) => x + 1;\n"
        "const gamma = function inner() {\n"
        "  return 1;\n"
        "};\n"
        "export class Widget {\n"
        "  constructor() {}\n"
        "  static make(): Widget {\n"
        "    return new Widget();\n"
        "  }\n"
        "}\n"
        "export abstract class Base {\n"
        "  abstract go(): void;\n"
        "}\n"
        "export interface Shape {\n"
        "  area(): number;\n"
        "}\n"
        "export type Alias = string;\n"
        "export enum Color {\n"
        "  Red,\n"
        "}\n"
        "declare function sig(a: string): void;\n"
    )
    assert _defs(tmp_path, "defs.ts") == {
        ("alpha", 1, 1, "function"),
        ("gen", 2, 4, "function"),
        # The arrow/function-expression capture lands on the variable_declarator,
        # so `const beta = ...` and `const gamma = function inner() {}` both index.
        ("beta", 5, 5, "function"),
        ("gamma", 6, 8, "function"),
        ("Widget", 9, 14, "class"),
        ("constructor", 10, 10, "function"),
        ("make", 11, 13, "function"),
        ("Base", 15, 17, "class"),
        ("Shape", 18, 20, "class"),
        ("area", 19, 19, "function"),
        ("Alias", 21, 21, "type"),
        ("Color", 22, 24, "class"),
        ("sig", 25, 25, "function"),
    }


def test_definitions_in_file_tsx_indexes_jsx_component_and_method(tmp_path: Path) -> None:
    (tmp_path / "defs.tsx").write_text(
        "export const Panel = ({ title }: { title: string }) => <div>{title}</div>;\n"
        "\n"
        "export function App() {\n"
        '  return <Panel title="x" />;\n'
        "}\n"
        "\n"
        "export class Box {\n"
        "  render() {\n"
        "    return <span />;\n"
        "  }\n"
        "}\n"
    )
    assert _defs(tmp_path, "defs.tsx") == {
        ("Panel", 1, 1, "function"),
        ("App", 3, 5, "function"),
        ("Box", 7, 11, "class"),
        ("render", 8, 10, "function"),
    }


def test_definitions_in_file_javascript_indexes_arrow_and_class(tmp_path: Path) -> None:
    (tmp_path / "defs.js").write_text(
        "export function alpha() {}\n"
        "export const beta = (x) => x + 1;\n"
        "export class Widget {\n"
        "  go() {}\n"
        "}\n"
    )
    assert _defs(tmp_path, "defs.js") == {
        ("alpha", 1, 1, "function"),
        ("beta", 2, 2, "function"),
        ("Widget", 3, 5, "class"),
        ("go", 4, 4, "function"),
    }


def test_definitions_in_file_go_covers_func_method_type_spec_and_alias(tmp_path: Path) -> None:
    (tmp_path / "defs.go").write_text(
        "package main\n"
        "\n"
        "type Server struct {\n"
        "\tName string\n"
        "}\n"
        "\n"
        "type Alias = int\n"
        "\n"
        "type Stringer interface {\n"
        "\tString() string\n"
        "}\n"
        "\n"
        "func New() *Server {\n"
        "\treturn &Server{}\n"
        "}\n"
        "\n"
        "func (s *Server) Handle(x int) int {\n"
        "\treturn x\n"
        "}\n"
    )
    # `type Alias = int` is a type_alias, not a type_spec -- both are captured, and
    # neither is stamped "class" the way the shared _definition_kind would.
    assert _defs(tmp_path, "defs.go") == {
        ("Server", 3, 5, "type"),
        ("Alias", 7, 7, "type"),
        ("Stringer", 9, 11, "type"),
        ("New", 13, 15, "function"),
        ("Handle", 17, 19, "function"),
    }


def test_definitions_in_file_rust_covers_every_item_kind(tmp_path: Path) -> None:
    (tmp_path / "defs.rs").write_text(
        "pub struct Widget {\n"
        "\tpub n: u32,\n"
        "}\n"
        "\n"
        "pub enum Color {\n"
        "\tRed,\n"
        "}\n"
        "\n"
        "pub trait Draw {\n"
        "\tfn draw(&self);\n"
        "}\n"
        "\n"
        "pub type Alias = u32;\n"
        "\n"
        "pub mod inner {\n"
        "\tpub fn deep() {}\n"
        "}\n"
        "\n"
        "pub union U {\n"
        "\ta: u32,\n"
        "}\n"
        "\n"
        "pub fn top() -> u32 {\n"
        "\t1\n"
        "}\n"
    )
    assert _defs(tmp_path, "defs.rs") == {
        ("Widget", 1, 3, "class"),
        ("Color", 5, 7, "class"),
        ("Draw", 9, 11, "class"),
        ("draw", 10, 10, "function"),  # function_signature_item inside the trait
        ("Alias", 13, 13, "type"),
        ("inner", 15, 17, "module"),
        ("deep", 16, 16, "function"),
        ("U", 19, 21, "class"),
        ("top", 23, 25, "function"),
    }


def test_definitions_in_file_python_range_excludes_the_decorator(tmp_path: Path) -> None:
    (tmp_path / "defs.py").write_text(
        "import functools\n"
        "\n"
        "\n"
        "@functools.cache\n"
        "def alpha(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    def go(self):\n"
        "        return 1\n"
    )
    # decorated_definition has no `name` field, so the inner function_definition is
    # captured and the decorator line (4) sits outside the reported range. Widget
    # and go also share end_line 11: a line number alone is not a unique key.
    assert _defs(tmp_path, "defs.py") == {
        ("alpha", 5, 6, "function"),
        ("Widget", 9, 11, "class"),
        ("go", 10, 11, "function"),
    }


def test_definitions_in_file_missing_file_returns_empty(tmp_path: Path) -> None:
    assert definitions_in_file(tmp_path, "nope.ts") == []


def test_definitions_in_file_unknown_language_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# not code\n")
    assert definitions_in_file(tmp_path, "notes.md") == []


def test_definitions_in_file_degrades_on_syntax_error(tmp_path: Path) -> None:
    # python recovery keeps the enclosing definition; typescript collapses to a
    # top-level ERROR and loses it. Neither may raise.
    (tmp_path / "broken.py").write_text("def alpha(:\n    return 1\n")
    (tmp_path / "broken.ts").write_text("function alpha( {\n  return 1;\n")
    assert ("alpha", 1, 2, "function") in _defs(tmp_path, "broken.py")
    assert definitions_in_file(tmp_path, "broken.ts") == []


def test_shared_definition_query_still_excludes_typescript_and_go() -> None:
    """The diagram query must not widen the reverse-import-edge gate (issue #1113).

    ``detect_affected_files``' ``defining_paths`` gate reads
    ``_def_query_for_language``; admitting TypeScript/Go there would start adding
    reverse edges for every generic-stem ``index.ts``/``main.go`` in every repo.
    """
    for language_id in ("typescript", "tsx", "javascript", "go"):
        assert _def_query_for_language(language_id) is None
        assert _diagram_def_query_for_language(language_id) is not None
    assert _def_query_for_language("python") is PYTHON_DEF_QUERY
    assert _diagram_def_query_for_language("python") is PYTHON_DEF_QUERY


# --- branch_statement_lines --------------------------------------------------


def test_branch_lines_python_flat_elif_chain_and_match_cases() -> None:
    # The if/elif/else chain is flat in python: `elif` (8) counts, `else` (10)
    # does not, so the chain yields one line per condition, not two per branch.
    # `match x:` (24) is dropped in favour of its case_clause lines (25, 27).
    assert branch_statement_lines("python", PYTHON_CF) == [
        6, 8, 12, 14, 16, 18, 20, 22, 25, 27, 29, 30, 31, 32, 34, 36, 38, 40
    ]


def test_branch_lines_typescript_nested_else_chain_and_nested_switch() -> None:
    # else_clause NESTS in TypeScript: the second `if` (4) lives inside the first
    # `else_clause`, and both share their physical line with their `else`. Counting
    # if_statement only therefore still yields one line per condition (2, 4) rather
    # than 2N. Both switch containers (30, 32) are dropped in favour of their own
    # cases (31/39 outer, 33/35 inner) -- resolved independently per nesting level.
    assert branch_statement_lines("typescript", TYPESCRIPT_CF) == [
        2, 4, 9, 12, 15, 18, 21, 23, 25, 28, 29, 31, 33, 35, 39
    ]


def test_branch_lines_tsx_grammar_matches_typescript_on_the_same_source() -> None:
    assert branch_statement_lines("tsx", TYPESCRIPT_CF) == branch_statement_lines(
        "typescript", TYPESCRIPT_CF
    )


def test_branch_lines_javascript_grammar_matches_typescript_on_the_same_source() -> None:
    assert branch_statement_lines("javascript", TYPESCRIPT_CF) == branch_statement_lines(
        "typescript", TYPESCRIPT_CF
    )


def test_branch_lines_tsx_jsx_component() -> None:
    # 2 = if_statement, 5 = ternary_expression in a JSX return.
    assert branch_statement_lines("tsx", TSX_CF) == [2, 5]


def test_branch_lines_javascript_source() -> None:
    assert branch_statement_lines("javascript", JAVASCRIPT_CF) == [2, 5, 6]


def test_branch_lines_go_has_no_else_node_and_four_switch_kinds() -> None:
    # Go has no else_clause at all: `} else if b {` (12) is an if_statement whose
    # parent is the outer if_statement, so the chain needs no exclusion rule.
    # All four container kinds (20, 22 expression_switch, 31 type_switch, 35
    # select) are dropped in favour of their cases (21, 23, 25, 28, 32, 36).
    assert branch_statement_lines("go", GO_CF) == [10, 12, 17, 21, 23, 25, 28, 32, 36]


def test_branch_lines_rust_nested_else_chain_and_match_arms() -> None:
    # Rust nests else_clause exactly like TypeScript (5, 7 are the two heads).
    # 26 is an if_expression whose condition is a let_condition (both on that
    # line); 29 is a `?` try_expression; `match v {` (21) is dropped in favour of
    # its match_arm lines (22, 23, 24).
    assert branch_statement_lines("rust", RUST_CF) == [
        5, 7, 12, 15, 18, 22, 23, 24, 26, 29, 30, 33
    ]


def test_branch_lines_rust_let_condition_on_its_own_line() -> None:
    # Isolates let_condition from the if_expression head, which they normally
    # share a line with.
    source = b"fn g(v: Option<i32>) {\n    if\n        let Some(x) = v\n    {\n        1;\n    }\n}\n"
    assert branch_statement_lines("rust", source) == [2, 3]


def test_branch_lines_unknown_language_is_empty() -> None:
    assert branch_statement_lines("elixir", b"if x do\n  1\nend\n") == []


def test_branch_lines_degrade_on_malformed_source() -> None:
    # python's error recovery keeps the inner if; typescript collapses to a
    # top-level ERROR. Neither may raise.
    assert branch_statement_lines("python", b"def f(:\n    if x:\n        pass\n") == [2]
    assert branch_statement_lines("typescript", b"function f( {\n  if (a) {\n") == []


@pytest.mark.parametrize("language_id", sorted(BRANCH_NODE_TYPES))
def test_branch_table_entries_all_occur_in_probe_sources(language_id: str) -> None:
    """No BRANCH_NODE_TYPES entry may be a name the grammars never produce."""
    present: set[str] = set()
    for source in _BRANCH_PROBE_SOURCES[language_id]:
        present |= _node_types(language_id, source)
    assert BRANCH_NODE_TYPES[language_id] - present == set()


# --- is_branch_line ----------------------------------------------------------


@pytest.mark.parametrize(
    ("language_id", "line"),
    [
        ("python", 24),  # match x:
        ("typescript", 30),  # switch (t) {
        ("typescript", 32),  # nested switch (q) {
        ("go", 20),  # switch a {
        ("go", 22),  # nested switch b {
        ("go", 31),  # switch t := v.(type) {
        ("go", 35),  # select {
        ("rust", 21),  # match v {
    ],
)
def test_is_branch_line_reports_switch_container_lines(language_id: str, line: int) -> None:
    """A container line is a branch line even though branch_statement_lines,
    which must not double-count, reports its cases instead."""
    source = {"python": PYTHON_CF, "typescript": TYPESCRIPT_CF, "go": GO_CF, "rust": RUST_CF}[
        language_id
    ]
    assert is_branch_line(language_id, source, line) is True
    assert line not in branch_statement_lines(language_id, source)


def test_is_branch_line_false_for_a_plain_statement() -> None:
    assert is_branch_line("python", PYTHON_CF, 1) is False
    assert is_branch_line("python", PYTHON_CF, 42) is False


def test_is_branch_line_falls_back_to_keyword_regex() -> None:
    elixir = b"defmodule M do\n  if x do\n    1\n  end\nend\n"
    for language_id in (None, "elixir"):
        assert is_branch_line(language_id, elixir, 2) is True
        assert is_branch_line(language_id, elixir, 3) is False
    # Out-of-range and non-positive lines are answers, not errors.
    assert is_branch_line(None, elixir, 999) is False
    assert is_branch_line(None, elixir, 0) is False


# --- is_terminal_line --------------------------------------------------------


def test_terminal_lines_python_return_raise_and_exit_calls() -> None:
    # 33 raise, 35 sys.exit, 37 os._exit, 39 exit, 41 quit, 42 return.
    assert _terminal_lines("python", PYTHON_CF) == [33, 35, 37, 39, 41, 42]


def test_terminal_lines_typescript_return_throw_and_process_exit() -> None:
    # 3/5 return, 7 throw new Error, 22 process.exit, 24 throw e, 42 return.
    assert _terminal_lines("typescript", TYPESCRIPT_CF) == [3, 5, 7, 22, 24, 42]


def test_terminal_lines_tsx_and_javascript() -> None:
    assert _terminal_lines("tsx", TSX_CF) == [3, 5]
    assert _terminal_lines("javascript", JAVASCRIPT_CF) == [3, 8]  # 8 = process.exit


def test_terminal_lines_go_return_os_exit_panic_and_log_fatal() -> None:
    # 11/13/15/37/39 return, 24 os.Exit, 26 log.Fatal, 29 panic, 43 log.Fatalf,
    # 44 log.Panic, 45 t.Fatal, 46 t.Fatalf.
    assert _terminal_lines("go", GO_CF) == [11, 13, 15, 24, 26, 29, 37, 39, 43, 44, 45, 46]


def test_terminal_lines_rust_return_exit_abort_and_panic_macros() -> None:
    # 6/8 return, 10 std::process::exit, 22 panic!, 23 unreachable!, 24 todo!,
    # 31 process::exit, 34 std::process::abort, 36 process::abort.
    assert _terminal_lines("rust", RUST_CF) == [6, 8, 10, 22, 23, 24, 31, 34, 36]


def test_terminal_lines_rust_ignores_non_panic_macros() -> None:
    # macro_invocation alone is far too broad to be a terminal: `println!` (27)
    # must not match, which is what makes the TERMINAL_MACRO_NAMES check load-bearing.
    assert is_terminal_line("rust", RUST_CF, 27) is False
    assert "println" not in TERMINAL_MACRO_NAMES
    assert TERMINAL_MACRO_NAMES == frozenset({"panic", "unreachable", "todo", "unimplemented"})


def test_is_terminal_line_falls_back_to_keyword_regex() -> None:
    elixir = b"defmodule M do\n  raise \"no\"\n  x = 1\nend\n"
    for language_id in (None, "elixir"):
        assert is_terminal_line(language_id, elixir, 2) is True
        assert is_terminal_line(language_id, elixir, 3) is False
    assert is_terminal_line(None, elixir, 999) is False


@pytest.mark.parametrize("language_id", sorted(TERMINAL_NODE_TYPES))
def test_terminal_table_entries_all_occur_in_probe_sources(language_id: str) -> None:
    """No TERMINAL_NODE_TYPES/TERMINAL_CALL_NAMES entry may be unobservable."""
    source = {
        "python": PYTHON_CF,
        "typescript": TYPESCRIPT_CF,
        "tsx": TYPESCRIPT_CF,
        "javascript": TYPESCRIPT_CF,
        "go": GO_CF,
        "rust": RUST_CF,
    }[language_id]
    parser = get_parser(language_id)
    assert parser is not None
    root = parser.parse(source).root_node
    assert not root.has_error, "probe source must parse cleanly"
    present: set[str] = set()
    callees: set[str] = set()
    for node in _walk(root):
        present.add(node.type)
        if node.type in ("call", "call_expression"):
            callee = node.child_by_field_name("function")
            if callee is not None and callee.text is not None:
                callees.add(callee.text.decode())
    assert TERMINAL_NODE_TYPES[language_id] - present == set()
    assert TERMINAL_CALL_NAMES[language_id] - callees == set()
