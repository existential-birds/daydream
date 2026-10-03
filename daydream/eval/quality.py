"""Deterministic Python structural quality and clone analysis."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from daydream._tree_sitter_safety import TreeSitterBadVersionError, assert_tree_sitter_safe
from daydream.generated_files import is_generated_file

# Quality metrics (issue #316) ----------------------------------------------

_QUALITY_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".daydream",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".worktrees",
        "dist",
        "build",
        "vendor",
        "third_party",
        "migrations",
        # ``atif`` is daydream/atif, explicitly vendored from Harbor
        # (see daydream/atif/NOTICE) — out of metric scope (Finding #5).
        "atif",
    }
)

# Node types that add cyclomatic complexity (tree-sitter-python). ``else_clause``
# is intentionally absent: an else adds no new path.
_CC_DECISION_TYPES = frozenset(
    {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "except_clause",
        "with_statement",
        "assert_statement",
        "conditional_expression",
        "boolean_operator",
        "case_clause",
        "for_in_clause",
    }
)

_COMPREHENSION_TYPES = frozenset(
    {
        "list_comprehension",
        "set_comprehension",
        "dictionary_comprehension",
        "generator_expression",
    }
)

_MIN_CLONE_BLOCK = 3
_MAX_CLONE_BLOCK = 20

_QUALITY_CALIBRATION = {
    "human_verbosity": 0.19,
    "human_erosion": 0.34,
    "paper": "arXiv:2603.24755",
}


@lru_cache(maxsize=1)
def _quality_python_parser() -> Any | None:
    """Cache the Python parser; unavailable bindings yield None.

    Known unsafe tree-sitter versions raise before parser construction.
    """
    try:
        assert_tree_sitter_safe()
        import tree_sitter_python
        from tree_sitter import Language, Parser

        return Parser(Language(tree_sitter_python.language()))
    except TreeSitterBadVersionError:
        raise
    except Exception:
        return None


def _iter_tree(node: Any) -> Iterator[Any]:
    """Yield *node* and all descendants in a deterministic depth-first order."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _iter_function(func: Any) -> Iterator[Any]:
    """Walk a function, yielding nested definitions without traversing their bodies."""
    stack = [func]
    while stack:
        node = stack.pop()
        yield node
        if node is not func and node.type == "function_definition":
            continue
        stack.extend(reversed(node.children))


def _is_comprehension_filter(node: Any) -> bool:
    """An ``if_clause`` attached to a comprehension, not a match-case guard."""
    parent = node.parent
    if parent is None:
        return False
    if parent.type in _COMPREHENSION_TYPES:
        return True
    grandparent = parent.parent
    return grandparent is not None and grandparent.type in _COMPREHENSION_TYPES


def _is_wildcard_case(node: Any) -> bool:
    """A ``case _:`` clause (no guard) — matches any value, adds no path."""
    if node.type != "case_clause":
        return False
    if node.child_by_field_name("guard") is not None:
        return False
    pattern = next((c for c in node.children if c.type == "case_pattern"), None)
    return pattern is not None and pattern.text.decode().strip() == "_"


def _count_decision_nodes(node: Any) -> int:
    """Cyclomatic decision count of a subtree, skipping nested functions."""
    return sum(
        (child.type in _CC_DECISION_TYPES and not _is_wildcard_case(child))
        or (child.type == "if_clause" and _is_comprehension_filter(child))
        for child in _iter_function(node)
    )


def _scoped_python_files(workspace: Path) -> list[tuple[Path, str]]:
    """Enumerate sorted Python paths, excluding exact directory components and generated files."""
    files: list[tuple[Path, str]] = []
    for path in workspace.rglob("*.py"):
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        if any(part in _QUALITY_EXCLUDED_DIRS for part in rel.parts):
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if is_generated_file(str(rel), content):
            continue
        files.append((path, str(rel)))
    return sorted(files, key=lambda item: item[1])


def _parse_python_file(path: Path) -> tuple[Any, list[str]] | None:
    """Read/parse once; return None for unreadable, missing, partial, or invalid trees.

    Failed candidates remain in scoped_files but contribute no metrics or clone evidence.
    """
    parser = _quality_python_parser()
    if parser is None:
        return None
    try:
        source = path.read_bytes()
    except OSError:
        return None
    try:
        tree = parser.parse(source)
    except Exception:
        return None
    if tree is None:
        return None
    root = tree.root_node
    if root.has_error:
        return None
    lines = source.decode("utf-8", errors="replace").splitlines()
    return root, lines


def _file_quality_from_tree(
    root: Any,
    lines: list[str],
    cross_file_flagged: set[int] | None = None,
) -> dict[str, Any]:
    """Pool cyclomatic mass and union local/cross-file verbosity flags.

    Zero denominators produce None ratios; counts remain available for aggregation.
    """
    # Erosion: pooled cyclomatic mass of functions with CC > 10.
    functions = [node for node in _iter_tree(root) if node.type == "function_definition"]
    metrics: list[tuple[int, int, float]] = []  # (cc, sloc, mass)
    for func in functions:
        body = func.child_by_field_name("body")
        cc = 1 + _count_decision_nodes(body) if body is not None else 1
        sloc = func.end_point.row - func.start_point.row + 1
        if sloc < 1:
            continue
        metrics.append((cc, sloc, cc * math.sqrt(sloc)))

    total_mass = sum(mass for _, _, mass in metrics)
    high_mass = sum(mass for cc, _, mass in metrics if cc > 10)
    erosion = high_mass / total_mass if total_mass > 0 else None

    # Verbosity: deterministic rule subset + clone detection over lines.
    sloc_file = sum(1 for line in lines if line.strip())
    flagged = _verbosity_flagged_lines(root)
    flagged |= _clone_flagged_lines(lines)
    if cross_file_flagged:
        flagged |= cross_file_flagged
    flagged &= {i for i, line in enumerate(lines) if line.strip()}
    verbosity = len(flagged) / sloc_file if sloc_file > 0 else None

    return {
        "entry": {
            "erosion": round(erosion, 4) if erosion is not None else None,
            "verbosity": round(verbosity, 4) if verbosity is not None else None,
            "sloc": sloc_file,
            "functions": len(metrics),
            "high_cc_functions": sum(1 for cc, _, _ in metrics if cc > 10),
        },
        "mass": total_mass,
        "high_mass": high_mass,
        "flagged": len(flagged),
        "loc": sloc_file,
    }


def _verbosity_flagged_lines(root: Any) -> set[int]:
    """Line rows (0-based) flagged by the deterministic taxonomy subset."""
    flagged: set[int] = set()
    flagged |= _identity_comprehension_lines(root)
    flagged |= _empty_list_guard_lines(root)
    flagged |= _single_use_variable_lines(root)
    flagged |= _trivial_wrapper_lines(root)
    flagged |= _nested_ladder_lines(root)
    return flagged


def _comprehension_body(node: Any) -> Any | None:
    """Find a comprehension's output expression, skipping container delimiters."""
    for child in node.children:
        if child.type not in ("[", "]", "{", "}", "(", ")"):
            return child
    return None


def _identity_comprehension_lines(root: Any) -> set[int]:
    """Flag a single unfiltered generator whose output is exactly its loop variable."""
    flagged: set[int] = set()
    for node in _iter_tree(root):
        if node.type not in _COMPREHENSION_TYPES:
            continue
        for_clauses = [child for child in node.children if child.type == "for_in_clause"]
        if len(for_clauses) != 1:
            continue
        target = for_clauses[0].child_by_field_name("left")
        if target is None or target.type != "identifier":
            continue
        has_filter = any(
            child.type == "if_clause"
            or (child.type == "for_in_clause" and any(grandchild.type == "if_clause" for grandchild in child.children))
            for child in node.children
        )
        if has_filter:
            continue
        body = _comprehension_body(node)
        if body is not None and body.text == target.text:
            flagged.update(range(node.start_point.row, node.end_point.row + 1))
    return flagged


def _empty_guard_variable(if_node: Any) -> str | None:
    """The variable tested by ``len(x) == 0`` or ``not x``, else ``None``."""
    cond = if_node.child_by_field_name("condition")
    if cond is None:
        return None
    if cond.type == "not_operator":
        arg = cond.child_by_field_name("argument")
        if arg is not None and arg.type == "identifier":
            return str(arg.text.decode())
        return None
    if cond.type == "comparison_operator":
        call = None
        zero = None
        for child in cond.children:
            if child.type == "call":
                call = child
            elif child.type == "integer":
                zero = child
        if call is None or zero is None or zero.text.decode().strip() != "0":
            return None
        return _len_call_argument(call)
    return None


def _len_call_argument(node: Any) -> str | None:
    """The single argument name of a ``len(x)`` call, else ``None``."""
    if node is None or node.type != "call":
        return None
    fn = node.child_by_field_name("function")
    args = node.child_by_field_name("arguments")
    if fn is None or fn.type != "identifier" or fn.text.decode() != "len" or args is None:
        return None
    arg_ids = [child for child in args.children if child.type == "identifier"]
    if len(arg_ids) == 1:
        return str(arg_ids[0].text.decode())
    return None


def _empty_guard_collection(condition: Any) -> str | None:
    """Recognize while x, while len(x), or while len(x) > 0 as nonempty proofs.

    Passing x to another predicate proves nothing.
    """
    if condition is None:
        return None
    if condition.type == "identifier":
        return str(condition.text.decode())
    name = _len_call_argument(condition)
    if name is not None:
        return name
    if condition.type == "comparison_operator":
        operands = list(condition.children)
        if len(operands) == 3 and operands[1].type == ">":
            left, _, right = operands
            if right.type == "integer" and right.text.decode().strip() == "0":
                return _len_call_argument(left)
    return None


def _statement_mutates(node: Any, name: str) -> bool:
    """Treat collection method calls or reassignment as invalidating nonempty proofs."""
    for sub in _iter_tree(node):
        if sub.type in ("assignment", "augmented_assignment"):
            target = sub.child_by_field_name("left")
            if target is not None and target.type == "identifier" and target.text.decode() == name:
                return True
        if sub.type == "call":
            fn = sub.child_by_field_name("function")
            if fn is not None and fn.type == "attribute":
                obj = fn.child_by_field_name("object")
                if obj is not None and obj.type == "identifier" and obj.text.decode() == name:
                    return True
    return False


def _empty_list_guard_lines(root: Any) -> set[int]:
    """Flag empty guards only while the enclosing loop's nonempty proof still holds.

    Any intervening possible mutation makes the guard meaningful.
    """
    flagged: set[int] = set()
    for node in _iter_tree(root):
        if node.type not in ("for_statement", "while_statement"):
            continue
        body = node.child_by_field_name("body")
        if body is None:
            continue
        if node.type == "for_statement":
            iterable = node.child_by_field_name("right")
            if iterable is None or iterable.type != "identifier":
                continue
            guarded = iterable.text.decode()
        else:
            guarded = _empty_guard_collection(node.child_by_field_name("condition"))
            if guarded is None:
                continue
        mutated = False
        for child in body.children:
            if child.type == "if_statement":
                guard_var = _empty_guard_variable(child)
                if guard_var == guarded and not mutated:
                    flagged.update(range(child.start_point.row, child.end_point.row + 1))
            if _statement_mutates(child, guarded):
                mutated = True
    return flagged


def _count_later_references(func: Any, name: str, after_row: int) -> int:
    """Identifier occurrences of *name* after *after_row* inside *func*."""
    return sum(
        1
        for node in _iter_function(func)
        if node.type == "identifier" and node.text.decode() == name and node.start_point.row > after_row
    )


def _single_use_variable_lines(root: Any) -> set[int]:
    """Flag plain non-private names assigned then referenced once later in their function."""
    flagged: set[int] = set()
    for func in _iter_tree(root):
        if func.type != "function_definition":
            continue
        for node in _iter_function(func):
            if node.type != "assignment":
                continue
            target = node.child_by_field_name("left")
            if target is None or target.type != "identifier":
                continue
            name = target.text.decode()
            if name.startswith("_"):
                continue
            if _count_later_references(func, name, node.start_point.row) == 1:
                flagged.add(node.start_point.row)
    return flagged


def _return_value(return_node: Any) -> Any | None:
    """The expression a ``return`` yields (the ``return`` keyword is a token)."""
    for child in return_node.children:
        if child.type != "return":
            return child
    return None


def _param_definitions(params: Any) -> list[tuple[str, bool]] | None:
    """Return ordered names/default flags, or None for splats and positional/keyword separators."""
    definitions: list[tuple[str, bool]] = []
    for child in params.children:
        if child.type in (",", "(", ")"):
            continue
        if child.type == "identifier":
            definitions.append((child.text.decode(), False))
        elif child.type == "typed_parameter":
            name = next((c for c in child.children if c.type == "identifier"), None)
            if name is None:
                return None
            definitions.append((name.text.decode(), False))
        elif child.type in ("default_parameter", "typed_default_parameter"):
            name = child.child_by_field_name("name")
            if name is None or name.type != "identifier":
                return None
            definitions.append((name.text.decode(), True))
        else:
            return None
    return definitions


def _forwarded_argument_names(args: Any) -> list[str] | None:
    """Accept only plain positional identifiers; other argument forms are nontrivial."""
    names: list[str] = []
    for child in args.children:
        if child.type in (",", "(", ")"):
            continue
        if child.type == "identifier":
            names.append(child.text.decode())
        else:
            return None
    return names


def _is_docstring_statement(node: Any) -> bool:
    """An ``expression_statement`` whose whole content is a string literal."""
    if node.type != "expression_statement":
        return False
    contents = [child for child in node.children if child.type != ";"]
    return len(contents) == 1 and contents[0].type == "string"


def _trivial_wrapper(func: Any) -> set[int] | None:
    """Flag a body consisting of return other(same args), ignoring its leading docstring."""
    body = func.child_by_field_name("body")
    if body is None:
        return None
    statements = list(body.children)
    if statements and _is_docstring_statement(statements[0]):
        statements = statements[1:]
    if len(statements) != 1 or statements[0].type != "return_statement":
        return None
    value = _return_value(statements[0])
    if value is None or value.type != "call":
        return None
    fn_name = value.child_by_field_name("function")
    args = value.child_by_field_name("arguments")
    params = func.child_by_field_name("parameters")
    if fn_name is None or fn_name.type != "identifier" or args is None or params is None:
        return None
    param_defs = _param_definitions(params)
    arg_names = _forwarded_argument_names(args)
    if param_defs is None or arg_names is None:
        return None
    if any(has_default for _, has_default in param_defs):
        return None
    if [name for name, _ in param_defs] != arg_names:
        return None
    return set(range(func.start_point.row, func.end_point.row + 1))


def _trivial_wrapper_lines(root: Any) -> set[int]:
    flagged: set[int] = set()
    for func in _iter_tree(root):
        if func.type != "function_definition":
            continue
        lines = _trivial_wrapper(func)
        if lines is not None:
            flagged.update(lines)
    return flagged


def _direct_nested_ifs(if_node: Any) -> list[Any]:
    """Direct ``if`` children of *if_node*'s consequence/alternative blocks."""
    nested: list[Any] = []
    consequence = if_node.child_by_field_name("consequence")
    if consequence is not None:
        nested.extend(child for child in consequence.children if child.type == "if_statement")
    alternative = if_node.child_by_field_name("alternative")
    if alternative is not None and alternative.type in ("elif_clause", "else_clause"):
        block = alternative.child_by_field_name("consequence") or alternative.child_by_field_name("body")
        if block is not None:
            nested.extend(child for child in block.children if child.type == "if_statement")
    return nested


def _nested_ladder_lines(root: Any) -> set[int]:
    """Innermost ``if`` of any ≥3-deep directly-nested if ladder."""
    flagged: set[int] = set()

    def visit(if_node: Any, depth: int) -> None:
        nested = _direct_nested_ifs(if_node)
        if nested:
            for child in nested:
                visit(child, depth + 1)
        elif depth >= 3:
            flagged.update(range(if_node.start_point.row, if_node.end_point.row + 1))

    for node in _iter_tree(root):
        if node.type == "if_statement":
            visit(node, 1)
    return flagged


def _repeated_block_rows(
    stripped: dict[Path, list[str]],
    *,
    require_distinct_sources: bool,
) -> dict[Path, set[int]]:
    """Flag repeated stripped blocks of 3..20 nonblank lines.

    With require_distinct_sources, occurrences must span at least two paths.
    """
    flagged: dict[Path, set[int]] = {path: set() for path in stripped}
    for length in range(_MIN_CLONE_BLOCK, _MAX_CLONE_BLOCK + 1):
        by_block: dict[tuple[str, ...], list[tuple[Path, int]]] = {}
        for path, lines in stripped.items():
            for i in range(len(lines) - length + 1):
                block = tuple(lines[i : i + length])
                if all(block):
                    by_block.setdefault(block, []).append((path, i))
        for occurrences in by_block.values():
            sources = {path for path, _ in occurrences} if require_distinct_sources else occurrences
            if len(sources) < 2:
                continue
            for path, i in occurrences:
                flagged[path].update(range(i, i + length))
    return flagged


def _clone_flagged_lines(lines: list[str]) -> set[int]:
    """Line rows in ≥2 occurrences of an identical contiguous block (3..20 lines)."""
    return _repeated_block_rows({Path(): [line.strip() for line in lines]}, require_distinct_sources=False)[Path()]


def _cross_file_clone_flagged_lines(
    file_lines: list[tuple[Path, list[str]]],
    target_paths: set[Path] | None = None,
) -> dict[Path, set[int]]:
    """Flag exact peer blocks; restrict reporting, retain all peers, and union local clones without double counting."""
    flagged = _repeated_block_rows(
        {path: [line.strip() for line in lines] for path, lines in file_lines},
        require_distinct_sources=True,
    )
    target_set = target_paths if target_paths is not None else set(flagged)
    for path in flagged:
        if path not in target_set:
            flagged[path].clear()
    return flagged


def _aggregate_per_file(
    candidates: list[tuple[Path, str]],
    parsed: dict[Path, Any],
    parsed_lines: dict[Path, list[str]],
    cross_file_flagged: dict[Path, set[int]],
) -> tuple[dict[str, dict[str, Any]], float, float, int, int]:
    """Aggregate parsed candidates; failed parses retain scope membership but add no metrics."""
    per_file: dict[str, dict[str, Any]] = {}
    total_mass = 0.0
    high_mass = 0.0
    total_flagged = 0
    total_loc = 0
    for path, rel in candidates:
        root = parsed.get(path)
        if root is None:
            continue
        quality = _file_quality_from_tree(
            root,
            parsed_lines[path],
            cross_file_flagged=cross_file_flagged.get(path),
        )
        per_file[rel] = quality["entry"]
        total_mass += quality["mass"]
        high_mass += quality["high_mass"]
        total_flagged += quality["flagged"]
        total_loc += quality["loc"]
    return per_file, total_mass, high_mass, total_flagged, total_loc


def analyze_quality(
    daydream_dir: str | Path,
    candidate_paths: set[str] | None = None,
    *,
    code_workspace: Path | None = None,
) -> dict[str, Any]:
    """Measure scoped Python quality in code_workspace (default daydream_dir.parent).

    Empty candidate sets skip enumeration; other sets limit reports, retaining all
    parseable peers for clones. Parse failures count in scope but not metrics. Empty
    denominators yield None. Unsafe tree-sitter is refused even for empty scopes.
    """
    daydream_dir = Path(daydream_dir)
    workspace = daydream_dir.parent if code_workspace is None else code_workspace

    # Reject unsafe tree-sitter before parser construction or any empty-scope shortcut.
    assert_tree_sitter_safe()

    # An explicitly empty candidate set is a zero-count empty report: never
    # walk (or parse) the workspace (issue #457).
    if candidate_paths is not None and not candidate_paths:
        return {
            "erosion": None,
            "verbosity": None,
            "per_file": {},
            "calibration": dict(_QUALITY_CALIBRATION),
            "scoped_files": 0,
        }

    scoped_files = _scoped_python_files(workspace)
    if candidate_paths is None:
        candidates = scoped_files
        candidate_path_set: set[Path] | None = None
    else:
        # Exact workspace-relative match: a candidate that fails eligibility is
        # absent from ``scoped_files`` and so is not a candidate (issue #457).
        candidates = [t for t in scoped_files if t[1] in candidate_paths]
        candidate_path_set = {path for path, _rel in candidates}
    scoped = len(candidates)

    # Parse once. Failed parses remain scoped but unmeasured; eligible peers still provide clone evidence.
    parsed: dict[Path, Any] = {}
    parsed_lines: dict[Path, list[str]] = {}
    file_lines: list[tuple[Path, list[str]]] = []
    for path, _rel in scoped_files:
        result = _parse_python_file(path)
        if result is None:
            continue
        root, lines = result
        file_lines.append((path, lines))
        if candidate_path_set is None or path in candidate_path_set:
            parsed[path] = root
            parsed_lines[path] = lines

    # Cross-file clones flag only parsed targets; peers provide evidence without metrics. Local
    # clones are unioned per file.
    cross_file_flagged = _cross_file_clone_flagged_lines(
        file_lines,
        target_paths=set(parsed) if candidate_paths is not None else None,
    )

    per_file, total_mass, high_mass, total_flagged, total_loc = _aggregate_per_file(
        candidates, parsed, parsed_lines, cross_file_flagged
    )

    return {
        "erosion": round(high_mass / total_mass, 4) if total_mass > 0 else None,
        "verbosity": round(total_flagged / total_loc, 4) if total_loc > 0 else None,
        "per_file": per_file,
        "calibration": dict(_QUALITY_CALIBRATION),
        "scoped_files": scoped,
    }
