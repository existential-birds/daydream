"""#1215: no corpus or benchmark module may hand-roll the temp+rename write primitive.

The scan is AST-based because `.replace(` alone is dominated by str.replace noise;
a rename is `os.replace(a, b)` or a single-argument `X.replace(target)` call, while
str.replace always takes two arguments.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCOPE_ROOTS = ("daydream/training", "daydream/benchmark")

#: Files allowed to call os.replace(): the primitive lives outside the scanned roots.
_PERMITTED_OS_REPLACE_FILES = frozenset({
    "daydream/training/adjudication/publish.py",                  # directory promotion (own fsync + identity re-check)
    "daydream/benchmark/harbor/build.py",                         # staged harbor tree promotion
    "daydream/benchmark/storage.py",                              # transactional commit + crash-recovery restore
    "daydream/benchmark/harbor/templates/metric.py",              # stdlib-only, shipped into the container
    "daydream/benchmark/harbor/templates/tests/score_review.py",  # stdlib-only, shipped into the container
})

#: The private writers deleted by #1215; none may be redefined.
_DELETED_PRIVATE_WRITERS = {
    "daydream/training/adjudication/materialize.py": {"_write_atomic"},
    "daydream/training/adjudication/final_bundle.py": {"_write_atomic", "_write_atomic_bytes"},
    "daydream/training/corpus_projection/projector.py": {"_atomic_write"},
    "daydream/training/adjudication/cli.py": {"_write_queue"},
}


def _os_replace_lines(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]


def _single_arg_replace_lines(source: str) -> list[int]:
    """`X.replace(target)` — a Path rename. A str.replace always carries two args."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "os")
        and len(node.args) == 1
        and not node.keywords
    ]


def _scope_files() -> list[Path]:
    return sorted(
        path
        for root in _SCOPE_ROOTS
        for path in (_REPO_ROOT / root).rglob("*.py")
    )


def test_no_hand_rolled_temp_rename_writer_remains() -> None:
    offenders: list[str] = []
    for path in _scope_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        offenders += [f"{rel}:{line} os.replace()" for line in _os_replace_lines(source)
                      if rel not in _PERMITTED_OS_REPLACE_FILES]
        offenders += [f"{rel}:{line} Path.replace()" for line in _single_arg_replace_lines(source)]
    assert offenders == []


def test_deleted_private_writer_helpers_stay_deleted() -> None:
    surviving: list[str] = []
    for rel, names in _DELETED_PRIVATE_WRITERS.items():
        defined = {
            node.name for node in ast.walk(ast.parse((_REPO_ROOT / rel).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
        }
        surviving += [f"{rel}::{name}" for name in sorted(names & defined)]
    assert surviving == []


def test_scanner_reports_a_reintroduced_hand_rolled_writer() -> None:
    """The guard's own discrimination check: feed it the shape it exists to catch."""
    reintroduced = (
        "def _write_atomic(out_path, payload):\n"
        "    tmp_path = out_path.with_name(out_path.name + '.tmp')\n"
        "    tmp_path.write_text(payload, encoding='utf-8')\n"
        "    os.replace(tmp_path, out_path)\n"
    )
    assert _os_replace_lines(reintroduced) == [4]
    assert _single_arg_replace_lines("def f(target):\n    Path('x.tmp').replace(target)\n") == [2]
    # ...and that it stays quiet on the two legitimate spellings it must not flag:
    assert _os_replace_lines("value = 'a'.replace('a', 'b')\n") == []
    assert _single_arg_replace_lines("value = 'a-b'.replace('-', '') if False else None\n") == []
