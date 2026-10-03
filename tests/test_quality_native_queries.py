"""Native quality rule boundaries retain complete source syntax."""
from pathlib import Path

import pytest
from tree_sitter import Language, Query

from daydream.eval import quality
from tests.test_analyzer import _quality


@pytest.mark.parametrize(
    ("source", "verbosity"),
    [
        ("def f(x: int, y: str):\n    return g(x, y)\n", 1.0),
        ("def f(x, y):\n    return g(y, x)\n", 0.0),
        ("def f(x):\n    return g((x))\n", 0.0),
        ("def f(x, /):\n    return g(x)\n", 0.0),
        ("def f(*, x):\n    return g(x)\n", 0.0),
        ("def f(x):\n    return g(x);\n", 0.0),
        ("def f(x):\n    f\"{x}\"\n    return g(x)\n", 1.0),
        ("def f(x):\n    \"a\" \"b\"\n    return g(x)\n", 0.0),
        ("def f(x):\n    return g(x)\n    return g(x)\n", 0.0),
    ],
)
def test_complete_native_wrapper_syntax_preserves_quality_metrics(
    tmp_path: Path, source: str, verbosity: float,
) -> None:
    result = _quality(tmp_path, {"app.py": source})
    assert result["per_file"]["app.py"]["verbosity"] == verbosity


@pytest.mark.parametrize(
    ("expression", "verbosity"),
    [
        ("[x for x in xs]", 0.5),
        ("{x for x in xs}", 0.5),
        ("(x for x in xs)", 0.5),
        ("[y for x in xs]", 0.0),
        ("[x for x in xs if x]", 0.0),
        ("[x for x in xs for y in ys]", 0.0),
        ("[x for x, y in xs]", 0.0),
        ("[(x) for x in xs]", 0.0),
        ("{x: x for x in xs}", 0.0),
    ],
)
def test_complete_identity_comprehension_syntax_preserves_quality_metrics(
    tmp_path: Path, expression: str, verbosity: float,
) -> None:
    result = _quality(tmp_path, {"app.py": f"def f(xs):\n    return {expression}\n"})
    assert result["per_file"]["app.py"]["verbosity"] == verbosity


def test_native_quality_compiles_each_fixed_grammar_once_for_a_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality._syntactic_quality_query.cache_clear()
    quality._empty_guard_query.cache_clear()
    native_query = Query
    compilations: list[str] = []

    def capture_query(language: Language, source: str) -> Query:
        compilations.append(source)
        return native_query(language, source)

    monkeypatch.setattr(quality, "Query", capture_query)
    result = _quality(tmp_path, {
        "a.py": "def a(x):\n    return g(x)\n",
        "b.py": "def b(y):\n    return h(y)\n",
    })
    assert list(result["per_file"]) == ["a.py", "b.py"]
    assert compilations == [quality._SYNTACTIC_QUALITY_QUERY, quality._EMPTY_GUARD_QUERY]
    quality._syntactic_quality_query.cache_clear()
    quality._empty_guard_query.cache_clear()


@pytest.mark.parametrize(
    ("loop", "guard", "verbosity"),
    [
        ("for item in items", "not items", 0.5),
        ("while items", "len(items) == 0", 0.5),
        ("while len(items)", "not items", 0.5),
        ("while len(items) > 0", "not items", 0.5),
        ("while len(items) >= 0", "not items", 0.0),
        ("while 0 < len(items)", "not items", 0.0),
        ("while len(items, 3) > 0", "len(items) > 0", 0.5),
        ("while len(items, other) > 0", "not items", 0.0),
        ("for item in items", "len(items, 3) != 0", 0.5),
        ("for item in items", "len(items, other) == 0", 0.0),
        ("for item in items", "len(items) < len(other) < 0", 0.0),
        ("for item in items", "len(other) < len(items) < 0", 0.5),
    ],
)
def test_native_nonempty_guard_syntax_preserves_quality_metrics(
    tmp_path: Path, loop: str, guard: str, verbosity: float,
) -> None:
    source = f"def f(items, other):\n    {loop}:\n        if {guard}:\n            pass\n"
    result = _quality(tmp_path, {"app.py": source})
    assert result["per_file"]["app.py"]["verbosity"] == verbosity


@pytest.mark.parametrize(
    ("statement", "before", "verbosity"),
    [
        ("items.clear()", True, 0.0),
        ("items.clear()", False, 0.4),
        ("if other: items.clear()", True, 0.0),
        ("lambda: items.clear()", True, 0.0),
        ("other.append(items)", True, 0.4),
        ("items.method.attr()", True, 0.4),
        ("items[0] = other", True, 0.4),
        ("del items[0]", True, 0.4),
    ],
)
def test_native_guard_mutation_proof_retains_sequential_scope(
    tmp_path: Path, statement: str, before: bool, verbosity: float,
) -> None:
    guard = "        if not items:\n            pass\n"
    mutation = f"        {statement}\n"
    source = "def f(items, other):\n    for item in items:\n" + (
        mutation + guard if before else guard + mutation
    )
    result = _quality(tmp_path, {"app.py": source})
    assert result["per_file"]["app.py"]["verbosity"] == verbosity
