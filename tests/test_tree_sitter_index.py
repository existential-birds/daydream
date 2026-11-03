"""Unit tests for daydream.tree_sitter_index.detect_affected_files()."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import pytest

from daydream import _tree_sitter_safety as safety, tree_sitter_index
from daydream.tree_sitter_index import (
    TERMINAL_MACRO_NAMES,
    branch_statement_lines,
    definitions_in_file,
    detect_affected_files,
    get_parser,
    is_branch_line,
    is_terminal_line,
    statements,
)
from daydream.tree_sitter_index.runtime import _PARSER_CACHE


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


def _materialize(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create files under tmp_path with the given relative paths and contents."""
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


def test_go_imports_reuse_one_package_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _materialize(
        tmp_path,
        {
            "cmd/alpha.go": 'package main\n\nimport "example.com/models"\n',
            "cmd/beta.go": 'package main\n\nimport "example.com/helpers"\n',
            "models/user.go": "package models\n",
            "models/order.go": "package models\n",
            "helpers/format.go": "package helpers\n",
            "lib/foo.rb": "class Foo\n  def bar; end\nend\n",
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

    unsupported_diff = (
        "diff --git a/lib/foo.rb b/lib/foo.rb\n"
        "index 1111111..2222222 100644\n"
        "--- a/lib/foo.rb\n"
        "+++ b/lib/foo.rb\n"
        "@@ -1,1 +1,2 @@\n"
        " class Foo\n"
        "+  def bar; end\n"
    )
    deleted_diff = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-print('bye')\n"
    )
    results = detect_affected_files(
        _modified_diff("cmd/alpha.go") + _modified_diff("cmd/beta.go") + unsupported_diff + deleted_diff
        # A header without a new-side path names no file rather than failing the parse.
        + "diff --git malformed-header\n", repo,
    )
    assert [(r.path, r.role) for r in results if r.path == "lib/foo.rb"] == [("lib/foo.rb", "modified")]
    assert [(r.path, r.role) for r in results if r.path == "gone.py"] == [("gone.py", "modified")]
    imports_paths = {r.path for r in results if r.role == "imports"}
    assert imports_paths == {"models/user.go", "models/order.go", "helpers/format.go"}
    assert {r.path for r in results if r.role == "modified"} == {
        "cmd/alpha.go", "cmd/beta.go", "lib/foo.rb", "gone.py",
    }


def test_known_bad_tree_sitter_is_refused_at_detection_and_parser_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Guard actual Parser construction, including import-graph callers; an absent install is the degrade path."""
    def _not_installed() -> None:
        raise PackageNotFoundError("tree-sitter")

    monkeypatch.setattr(safety, "installed_tree_sitter_version", _not_installed)
    safety.assert_tree_sitter_safe()  # not installed is the existing degrade path, not bad
    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.26.0")
    with pytest.raises(safety.TreeSitterBadVersionError):
        detect_affected_files(
            repo_root=tmp_path,
            diff_text=("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n" "@@ -1 +1 @@\n-x\n+y\n"),
        )
    _PARSER_CACHE.clear()
    try:
        with pytest.raises(safety.TreeSitterBadVersionError):
            get_parser("python")
        assert "python" not in _PARSER_CACHE
    finally:
        _PARSER_CACHE.clear()


# Literal line outcomes exercise branch and terminal classification through real grammars.

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
    if
        let Some(x) = v
    {
        process::exit(1);
    }
    if b {
        std::process::abort();
    }
    process::abort();
    Ok(0)
}
'''

# TypeScript, TSX and JavaScript share tree-sitter-typescript grammars and probe source.


def _terminal_lines(language_id: str | None, source: bytes) -> list[int]:
    rows = source.split(b"\n")
    return [n + 1 for n in range(len(rows)) if is_terminal_line(language_id, source, n + 1)]


def _defs(repo_root: Path, path: str) -> set[tuple[str, int, int, str]]:
    return {
        (str(d["name"]), int(str(d["line"])), int(str(d["end_line"])), str(d["kind"]))
        for d in definitions_in_file(repo_root, path)
    }


def test_definitions_in_file_rust_covers_every_item_kind_and_skips_unreadable_files(tmp_path: Path) -> None:
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
        ("Widget", 1, 3, "class"), ("Color", 5, 7, "class"), ("Draw", 9, 11, "class"),
        ("draw", 10, 10, "function"),  # function_signature_item inside the trait
        ("Alias", 13, 13, "type"), ("inner", 15, 17, "module"), ("deep", 16, 16, "function"), ("U", 19, 21, "class"),
        ("top", 23, 25, "function"),
    }
    assert definitions_in_file(tmp_path, "nope.ts") == []
    (tmp_path / "notes.md").write_text("# not code\n")
    assert definitions_in_file(tmp_path, "notes.md") == []


def test_statement_classification_without_a_grammar_uses_keywords_or_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert branch_statement_lines("elixir", b"if x do\n  1\nend\n") == []
    elixir = b"defmodule M do\n  if x do\n    1\n  end\nend\n"
    for language_id in (None, "elixir"):
        assert is_branch_line(language_id, elixir, 2) is True
        assert is_branch_line(language_id, elixir, 3) is False
    # Out-of-range and non-positive lines are answers, not errors.
    assert is_branch_line(None, elixir, 999) is False
    assert is_branch_line(None, elixir, 0) is False
    raising = b"defmodule M do\n  raise \"no\"\n  x = 1\nend\n"
    for language_id in (None, "elixir"):
        assert is_terminal_line(language_id, raising, 2) is True
        assert is_terminal_line(language_id, raising, 3) is False
    assert is_terminal_line(None, raising, 999) is False

    monkeypatch.setattr(statements, "get_parser", lambda _language_id: None)
    source = b"def resolve():\n    result = compute()\n"

    assert not tree_sitter_index.is_executable_statement_line("python", source, 2)
    assert not tree_sitter_index.is_executable_statement_line("elixir", b"  work()\n", 1)
    assert not tree_sitter_index.is_executable_statement_line(None, b"  work()\n", 1)
    assert not tree_sitter_index.is_executable_statement_line(None, source, 0)
    assert not tree_sitter_index.is_executable_statement_line(None, source, 999)

def test_terminal_lines_rust_return_exit_abort_and_panic_macros() -> None:
    # 6/8 return, 10 std::process::exit, 22 panic!, 23 unreachable!, 24 todo!,
    # 33 process::exit, 36 std::process::abort, 38 process::abort.
    assert _terminal_lines("rust", RUST_CF) == [6, 8, 10, 22, 23, 24, 33, 36, 38]
    assert is_terminal_line("rust", RUST_CF, 27) is False
    assert "println" not in TERMINAL_MACRO_NAMES
    assert TERMINAL_MACRO_NAMES == frozenset({"panic", "unreachable", "todo", "unimplemented"})
