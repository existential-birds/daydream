"""Deterministic Python structural quality and clone analysis."""

from __future__ import annotations

import math
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, cast

from tree_sitter import Language, Query, QueryCursor

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

_QUALITY_CALIBRATION = {
    "human_verbosity": 0.19,
    "human_erosion": 0.34,
    "paper": "arXiv:2603.24755",
}


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


def _function_quality(func: Any) -> tuple[int, set[int]]:
    """Analyze one function's scoped decisions and later-name evidence once.

    Definitions include their signatures, while cyclomatic decisions belong
    only to the body. Nested function bodies remain independently owned.
    Identifier rows follow source order; same-line uses never count as later.
    """
    body = func.child_by_field_name("body")
    decisions = 0
    references: dict[str, list[int]] = {}
    assignments: list[tuple[str, int]] = []
    for node in _iter_function(func):
        if body is not None and node.start_byte >= body.start_byte:
            decisions += (
                node.type in _CC_DECISION_TYPES and not _is_wildcard_case(node)
                or node.type == "if_clause" and _is_comprehension_filter(node)
            )
        if node.type == "identifier":
            references.setdefault(node.text.decode(), []).append(node.start_point.row)
        elif node.type == "assignment":
            target = node.child_by_field_name("left")
            if target is not None and target.type == "identifier":
                name = target.text.decode()
                if not name.startswith("_"):
                    assignments.append((name, node.start_point.row))
    flagged: set[int] = set()
    flagged.update(
        row for name, row in assignments
        if len(references[name]) - bisect_right(references[name], row) == 1
    )
    return 1 + decisions, flagged


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
    try:
        from daydream.tree_sitter_index import get_parser

        parser = get_parser("python")
    except TreeSitterBadVersionError:
        raise
    except Exception:
        return None
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
    flagged = _syntactic_quality_lines(root) | _empty_list_guard_lines(root) | _nested_ladder_lines(root)
    for func in functions:
        cc, function_flags = _function_quality(func)
        flagged |= function_flags
        sloc = func.end_point.row - func.start_point.row + 1
        if sloc < 1:
            continue
        metrics.append((cc, sloc, cc * math.sqrt(sloc)))

    total_mass = sum(mass for _, _, mass in metrics)
    high_mass = sum(mass for cc, _, mass in metrics if cc > 10)
    erosion = high_mass / total_mass if total_mass > 0 else None

    # Verbosity: deterministic rule subset + clone detection over lines.
    sloc_file = sum(1 for line in lines if line.strip())
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


# Native grammar owns the complete wrapper and identity-comprehension shapes.
# Match counts reject any unmatched child rather than admitting a partial query.
_SYNTACTIC_QUALITY_QUERY = """
(function_definition
  parameters: [
    (parameters "(" ")")
    (parameters "("
      [(identifier) @parameter (typed_parameter . (identifier) @parameter)]
      ("," [(identifier) @parameter (typed_parameter . (identifier) @parameter)])*
      ","? ")")
  ] @parameters
  body: (block
    . (expression_statement (string))? @docstring .
    (return_statement (call
      function: (identifier)
      arguments: [
        (argument_list "(" ")")
        (argument_list "(" (identifier) @argument
          ("," (identifier) @argument)* ","? ")")
      ] @arguments)) .) @body) @wrapper

((list_comprehension
  . (identifier) @value .
  (for_in_clause . left: (identifier) @variable . right: (_) .) .) @identity
  (#eq? @value @variable))
((set_comprehension
  . (identifier) @value .
  (for_in_clause . left: (identifier) @variable . right: (_) .) .) @identity
  (#eq? @value @variable))
((generator_expression
  . (identifier) @value .
  (for_in_clause . left: (identifier) @variable . right: (_) .) .) @identity
  (#eq? @value @variable))
"""


@lru_cache(maxsize=1)
def _syntactic_quality_query(language: Language) -> Query:
    """Compile the fixed native grammar once for the admitted Python language."""
    return Query(language, _SYNTACTIC_QUALITY_QUERY)


def _syntactic_quality_lines(root: Any) -> set[int]:
    """Flag complete native wrappers and unfiltered identity comprehensions."""
    from daydream.tree_sitter_index import get_parser

    parser = get_parser("python")
    if parser is None:
        return set()
    flagged: set[int] = set()
    query = _syntactic_quality_query(parser.language)
    for _pattern, captures in QueryCursor(query).matches(root):
        if "wrapper" in captures:
            parameters = captures.get("parameter", [])
            arguments = captures.get("argument", [])
            if len(parameters) != captures["parameters"][0].named_child_count:
                continue
            if len(arguments) != captures["arguments"][0].named_child_count:
                continue
            if len(captures["body"][0].children) != 1 + len(captures.get("docstring", [])):
                continue
            if [node.text for node in parameters] != [node.text for node in arguments]:
                continue
            node = captures["wrapper"][0]
        else:
            node = captures["identity"][0]
        flagged.update(range(node.start_point.row, node.end_point.row + 1))
    return flagged


_EMPTY_GUARD_QUERY = r"""
(for_statement right: (identifier) @collection body: (block) @body) @loop
(while_statement condition: (_) @condition body: (block) @body) @loop
(if_statement condition: (not_operator argument: (identifier) @collection)) @guard
(if_statement condition: (comparison_operator) @condition) @guard
((call function: (identifier) @function arguments: (argument_list) @arguments) @length
  (#eq? @function "len"))
(assignment left: (identifier) @collection) @mutation
(augmented_assignment left: (identifier) @collection) @mutation
(call function: (attribute object: (identifier) @collection)) @mutation
"""


@lru_cache(maxsize=1)
def _empty_guard_query(language: Language) -> Query:
    """Compile the fixed nonempty-proof grammar for the admitted Python language."""
    return Query(language, _EMPTY_GUARD_QUERY)


def _empty_list_guard_lines(root: Any) -> set[int]:
    """Flag direct empty guards until a preceding statement may mutate the collection.

    Native capture owns loop, guard, call and mutation shapes. The sequential
    reducer still decides when a loop's nonempty proof has been invalidated.
    """
    from daydream.tree_sitter_index import get_parser

    parser = get_parser("python")
    if parser is None:
        return set()
    matches = QueryCursor(_empty_guard_query(parser.language)).matches(root)
    lengths: dict[int, str] = {}
    guards: dict[int, str] = {}
    loops: list[dict[str, list[Any]]] = []
    mutations: dict[str, list[int]] = {}
    comparisons: list[tuple[Any, Any]] = []
    for _pattern, captures in matches:
        if "length" in captures:
            arguments = captures["arguments"][0]
            identifiers = [node for node in arguments.children if node.type == "identifier"]
            if len(identifiers) == 1:
                lengths[captures["length"][0].id] = cast(bytes, identifiers[0].text).decode()
        elif "loop" in captures:
            loops.append(captures)
        elif "guard" in captures:
            guard = captures["guard"][0]
            if "collection" in captures:
                guards[guard.id] = cast(bytes, captures["collection"][0].text).decode()
            else:
                comparisons.append((guard, captures["condition"][0]))
        else:
            mutations.setdefault(cast(bytes, captures["collection"][0].text).decode(), []).append(
                captures["mutation"][0].start_byte
            )
    for guard, comparison in comparisons:
        calls = [node for node in comparison.children if node.type == "call"]
        integers = [node for node in comparison.children if node.type == "integer"]
        if calls and integers and cast(bytes, integers[-1].text).decode().strip() == "0":
            name = lengths.get(calls[-1].id)
            if name is not None:
                guards[guard.id] = name
    flagged: set[int] = set()
    for captures in loops:
        body = captures["body"][0]
        if "collection" in captures:
            name = cast(bytes, captures["collection"][0].text).decode()
        else:
            condition = captures["condition"][0]
            name = cast(bytes, condition.text).decode() if condition.type == "identifier" else lengths.get(condition.id)
            if name is None and condition.type == "comparison_operator":
                operands = list(condition.children)
                if (
                    len(operands) == 3
                    and operands[1].type == ">"
                    and cast(bytes, operands[2].text).decode().strip() == "0"
                ):
                    name = lengths.get(operands[0].id) if operands[2].type == "integer" else None
        if name is None:
            continue
        mutated = False
        for child in body.children:
            if guards.get(child.id) == name and not mutated:
                flagged.update(range(child.start_point.row, child.end_point.row + 1))
            if any(child.start_byte <= start < child.end_byte for start in mutations.get(name, [])):
                mutated = True
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
    """Flag repeated nonblank blocks using only their minimum three-line windows.

    Every longer clone is a union of repeated three-line windows, so scanning
    additional lengths cannot add a flagged row. Distinct-source filtering
    applies to each window before its rows enter the result.
    """
    flagged: dict[Path, set[int]] = {path: set() for path in stripped}
    by_block: dict[tuple[str, ...], list[tuple[Path, int]]] = {}
    for path, lines in stripped.items():
        for i in range(len(lines) - _MIN_CLONE_BLOCK + 1):
            block = tuple(lines[i : i + _MIN_CLONE_BLOCK])
            if all(block):
                by_block.setdefault(block, []).append((path, i))
    for occurrences in by_block.values():
        sources = {path for path, _ in occurrences} if require_distinct_sources else occurrences
        if len(sources) < 2:
            continue
        for path, i in occurrences:
            flagged[path].update(range(i, i + _MIN_CLONE_BLOCK))
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
    parsed = {
        path: result for path, _rel in scoped_files
        if (result := _parse_python_file(path)) is not None
    }
    cross_file_flagged = _cross_file_clone_flagged_lines(
        [(path, lines) for path, (_root, lines) in parsed.items()],
        target_paths=candidate_path_set,
    )
    qualities = {
        rel: _file_quality_from_tree(*parsed[path], cross_file_flagged=cross_file_flagged.get(path))
        for path, rel in candidates if path in parsed
    }
    total_mass = sum(quality["mass"] for quality in qualities.values())
    high_mass = sum(quality["high_mass"] for quality in qualities.values())
    total_flagged = sum(quality["flagged"] for quality in qualities.values())
    total_loc = sum(quality["loc"] for quality in qualities.values())

    return {
        "erosion": round(high_mass / total_mass, 4) if total_mass > 0 else None,
        "verbosity": round(total_flagged / total_loc, 4) if total_loc > 0 else None,
        "per_file": {rel: quality["entry"] for rel, quality in qualities.items()},
        "calibration": dict(_QUALITY_CALIBRATION),
        "scoped_files": scoped,
    }
