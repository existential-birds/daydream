"""Definition kind policies stay bound to their native query or caller."""
from pathlib import Path
from typing import Callable, cast

import pytest

from daydream.tree_sitter_index import runtime


def test_generic_definition_query_keeps_default_kind_policy() -> None:
    parser = runtime.get_parser("typescript")
    assert parser is not None
    records = runtime.extract_definitions(
        parser,
        b"function f() {}\nclass C {}\ntype T = string;\n",
        runtime.TYPESCRIPT_DIAGRAM_DEF_QUERY,
    )
    assert {r["name"]: r["kind"] for r in records} == {
        "f": "class", "C": "class", "T": "class",
    }


def test_generic_definition_query_honors_explicit_callback() -> None:
    parser = runtime.get_parser("python")
    assert parser is not None
    visited: list[str] = []

    def kind_for(node_type: str) -> str:
        visited.append(node_type)
        return f"custom:{node_type}"

    records = runtime.extract_definitions(
        parser, b"class C:\n def m(self): pass\ndef f(): pass\n",
        runtime.PYTHON_DEF_QUERY, kind_for=kind_for,
    )
    assert [r["kind"] for r in records] == [f"custom:{kind}" for kind in visited]
    assert len(visited) == 3


@pytest.mark.parametrize("label", ["function", "class", "type", "module"])
def test_unknown_queries_do_not_override_generic_kind_policy(label: str) -> None:
    parser = runtime.get_parser("python")
    assert parser is not None
    assert runtime.extract_definitions(
        parser, b"class C: pass\n", f"(class_definition) @def @{label}",
    ) == [{"name": "C", "line": 1, "end_line": 1, "kind": "class"}]


def test_native_file_capture_preserves_all_definition_kinds(tmp_path: Path) -> None:
    source = tmp_path / "module.rs"
    source.write_text(
        "fn f() {}\nstruct C {}\nenum E { A }\ntrait T {}\n"
        "union U { x: u32 }\ntype A = u32;\nmod M {}\n",
        encoding="utf-8",
    )
    assert {r["name"]: r["kind"] for r in runtime.definitions_in_file(tmp_path, source.name)} == {
        "f": "function", "C": "class", "E": "class", "T": "class",
        "U": "class", "A": "type", "M": "module",
    }


def test_generic_capture_does_not_admit_missing_kind_callback() -> None:
    parser = runtime.get_parser("python")
    assert parser is not None
    assert runtime.extract_definitions(
        parser, b"def f(): pass\n", runtime.PYTHON_DEF_QUERY,
        kind_for=cast(Callable[[str], str], None),
    ) == []
