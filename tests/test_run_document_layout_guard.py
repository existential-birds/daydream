"""#1220: the run-document layout literals live in the owner, plus a frozen allowlist.

The scan is AST-based (a plain text grep cannot tell a path composition from a
docstring, a data key, or prose) and is modelled on
``tests/test_atomic_write_consolidation.py``: declared scope roots, a frozenset of
reasoned allowances, and a liveness assertion so no allowance can rot into a
blanket exemption.

A literal counts only when it is *path-shaped*:

* a string that carries ``trajectory.json`` / ``trajectories`` / ``runs`` as a path
  segment (bare, or embedded in a longer path such as ``runs/*/trajectory.json``), and
* for the bare names, only in a path context — a ``/`` composition, a comparison
  against a document name, or an argument to ``Path``/``glob``/``rglob``/
  ``joinpath``/``with_name``/``with_suffix``/``relative_to``.

Docstrings are skipped: the layout is restated in prose in several modules and
requirement 6 is about path composition, not message wording.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCOPE_ROOTS = ("daydream", "rl")
_OWNER = "daydream/trajectory.py"

#: Directories that are not first-party source. ``.venv``/``venv``/``site-packages``
#: keep the scan from walking an installed dependency tree (the ``rl`` project's own
#: environment lives at ``rl/daydream_review/.venv`` once ``make deadcode`` syncs it).
_EXCLUDED_PARTS = frozenset({"tests", "__pycache__", ".venv", "venv", "site-packages"})

#: The in-scope callers. Each must import the layout surface (requirement 10).
_IN_SCOPE_CONSUMERS = frozenset({
    "daydream/archive/__init__.py",
    "daydream/archive/hydrate.py",
    "daydream/archive/index.py",
    "daydream/archive/license_enrich.py",
    "daydream/archive/sanitize.py",
    "daydream/eval/analyzer.py",
    "daydream/artifact_visibility.py",
    "daydream/phases.py",
    "daydream/training/adjudication/materialize.py",
    "daydream/training/adjudication/cli.py",
    "daydream/training/coordinator.py",
    "daydream/runner.py",
})

#: Symbols that count as "consuming the layout surface".
_LAYOUT_SYMBOLS = frozenset({
    "RUNS_DIRNAME",
    "RUN_DOCUMENT_NAME",
    "SIBLINGS_DIRNAME",
    "PARTIAL_SUFFIX",
    "run_directory",
    "run_document_path",
    "siblings_directory",
    "sibling_document_path",
    "partial_document_path",
    "default_trajectory_path",
})

#: (file, literal kind) allowances. Every entry names why that file legitimately
#: carries the name in a *different* layout or cannot import the owner.
_PERMITTED = {
    # A download temp file's suffix, unrelated to trajectory documents.
    "daydream/archive/hydrate.py": frozenset({"partial"}),
    # Harbor container logs: `agent/trajectory.json` inside a benchmark job tree.
    "daydream/benchmark/harbor/agent.py": frozenset({"root-document"}),
    "daydream/benchmark/harbor/clean.py": frozenset({"root-document"}),
    "daydream/benchmark/harbor/entrypoint.py": frozenset({"root-document"}),
    # Training corpus batches: `batches/<sid>/trajectory.json`, its own producer.
    "daydream/training/corpus_projection/projector.py": frozenset({"root-document"}),
    # A standalone package that cannot import `daydream` (reads via a runtime seam).
    "rl/daydream_review/daydream_review/rundir.py": frozenset({"root-document", "runs-dir"}),
    # The `runs` SQLite table of the archive index (`archive/_schema.py`), not a path.
    "daydream/training/adjudication/cli.py": frozenset({"runs-dir"}),
}

_PATH_CALLS = frozenset({
    "Path",
    "glob",
    "rglob",
    "joinpath",
    "with_name",
    "with_suffix",
    "relative_to",
})
_BARE_NAMES = frozenset({"trajectory.json", "trajectories", "runs"})
_ROOT_DOCUMENT = re.compile(r"(?:^|/)trajectory\.json$|(?:^|/)trajectory\.json/")
_SIBLINGS_DIR = re.compile(r"(?:^|/)trajectories$|(?:^|/)trajectories/")
_RUNS_DIR = re.compile(r"(?:^|/)runs$|(?:^|/)runs/")


def _production_modules() -> list[tuple[str, Path]]:
    modules: list[tuple[str, Path]] = []
    for root_name in _SCOPE_ROOTS:
        for path in sorted((_REPO_ROOT / root_name).rglob("*.py")):
            if _EXCLUDED_PARTS & set(path.parts):
                continue
            modules.append((path.relative_to(_REPO_ROOT).as_posix(), path))
    return modules


def _docstring_constants(tree: ast.AST) -> set[int]:
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return docstrings


def _is_path_context(parents: dict[int, ast.AST], node: ast.AST) -> bool:
    current: ast.AST = node
    for _ in range(4):
        parent = parents.get(id(current))
        if parent is None:
            return False
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
            return True
        if isinstance(parent, ast.Compare):
            return True
        if isinstance(parent, ast.Call):
            func = parent.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            return name in _PATH_CALLS
        if isinstance(parent, (ast.JoinedStr, ast.FormattedValue)):
            current = parent
            continue
        return False
    return False


def _layout_literals(module: str, path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_constants(tree)
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstrings:
            continue
        value = node.value
        kinds: set[str] = set()
        if _ROOT_DOCUMENT.search(value):
            kinds.add("root-document")
        if _SIBLINGS_DIR.search(value):
            kinds.add("siblings-dir")
        if _RUNS_DIR.search(value):
            kinds.add("runs-dir")
        if ".partial" in value:
            kinds.add("partial")
        if not kinds:
            continue
        if value in _BARE_NAMES and not _is_path_context(parents, node):
            continue
        found.append(("+".join(sorted(kinds)), node.lineno))
    return found


def test_no_layout_literal_outside_the_owner_and_its_allowlist() -> None:
    offenders = {
        module: [hit for hit in _layout_literals(module, path) if not _permitted(module, hit[0])]
        for module, path in _production_modules()
        if module != _OWNER
    }
    offenders = {module: hits for module, hits in offenders.items() if hits}
    assert not offenders, (
        "run-document layout literals outside "
        f"{_OWNER} (use daydream.trajectory's layout surface): {offenders}"
    )


def _permitted(module: str, kinds: str) -> bool:
    return set(kinds.split("+")) <= _PERMITTED.get(module, frozenset())


def test_every_allowlisted_file_still_carries_its_permitted_literal() -> None:
    """An allowance that no longer matches anything fails, so it cannot rot."""
    stale: list[str] = []
    for module, permitted in _PERMITTED.items():
        carried: set[str] = set()
        for kinds, _line in _layout_literals(module, _REPO_ROOT / module):
            carried.update(kinds.split("+"))
        stale.extend(f"{module}:{kind}" for kind in sorted(permitted - carried))
    assert not stale, f"allowlist entries no longer match anything: {stale}"


def test_in_scope_consumers_import_the_layout_surface() -> None:
    missing: list[str] = []
    for module in sorted(_IN_SCOPE_CONSUMERS):
        tree = ast.parse((_REPO_ROOT / module).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "daydream.trajectory":
                imported.update(alias.name for alias in node.names)
        if not imported & _LAYOUT_SYMBOLS:
            missing.append(module)
    assert not missing, f"in-scope callers that do not import the layout surface: {missing}"
