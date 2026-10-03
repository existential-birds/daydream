"""Language-specific branch, terminal, and executable-line classification."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from tree_sitter import Node

from .runtime import get_parser

# Validate node names against real grammars before adding them: an unknown name
# invalidates the whole query. JS uses the TS grammar and JSX uses TSX.
_TYPESCRIPT_BRANCH_NODES = frozenset(
    {
        "if_statement",
        "switch_statement",
        "switch_case",
        "switch_default",
        "try_statement",
        "catch_clause",
        "finally_clause",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "ternary_expression",
        "optional_chain",
    }
)

# Node types that open a control-flow branch. Head nodes only: ``else_clause`` is
# deliberately absent (an ``else`` adds no new condition, matching
# ``eval/quality._CC_DECISION_TYPES``), which is also what keeps an
# if/else-if/else chain at N decisions instead of 2N -- see
# :func:`branch_statement_lines`.
BRANCH_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {
            "if_statement",
            "elif_clause",
            "for_statement",
            "while_statement",
            "try_statement",
            "except_clause",
            "finally_clause",
            "with_statement",
            "match_statement",
            "case_clause",
            "conditional_expression",
            "boolean_operator",
            "assert_statement",
        }
    ),
    "typescript": _TYPESCRIPT_BRANCH_NODES,
    "tsx": _TYPESCRIPT_BRANCH_NODES,
    "javascript": _TYPESCRIPT_BRANCH_NODES,
    "go": frozenset(
        {
            "if_statement",
            "for_statement",
            "expression_switch_statement",
            "type_switch_statement",
            "select_statement",
            "expression_case",
            "default_case",
            "type_case",
            "communication_case",
        }
    ),
    "rust": frozenset(
        {
            "if_expression",
            "match_expression",
            "match_arm",
            "while_expression",
            "for_expression",
            "loop_expression",
            "let_condition",
            "try_expression",
        }
    ),
}

# Statements that end a control-flow path. ``break``/``continue``/``goto``/
# ``fallthrough`` are non-local *jumps*, not terminals, and are excluded.
TERMINAL_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"return_statement", "raise_statement"}),
    "typescript": frozenset({"return_statement", "throw_statement"}),
    "tsx": frozenset({"return_statement", "throw_statement"}),
    "javascript": frozenset({"return_statement", "throw_statement"}),
    "go": frozenset({"return_statement"}),
    "rust": frozenset({"return_expression"}),
}

# Exit-like calls have no dedicated node kind, so they are matched on the text of
# the call's ``function:`` child (``attribute`` for python, ``member_expression``
# for TS, ``selector_expression`` for go, ``scoped_identifier`` for rust).
TERMINAL_CALL_NAMES: dict[str, frozenset[str]] = {
    "python": frozenset({"sys.exit", "os._exit", "exit", "quit"}),
    "typescript": frozenset({"process.exit"}),
    "tsx": frozenset({"process.exit"}),
    "javascript": frozenset({"process.exit"}),
    "go": frozenset(
        {"os.Exit", "panic", "log.Fatal", "log.Fatalf", "log.Panic", "t.Fatal", "t.Fatalf"}
    ),
    "rust": frozenset(
        {"std::process::exit", "process::exit", "std::process::abort", "process::abort"}
    ),
}

# Rust's panic family is a ``macro_invocation``, which on its own is far too broad
# to be a terminal (``assert!``, ``println!`` and ``vec!`` are all
# ``macro_invocation``), so the ``macro:`` name check below is mandatory.
TERMINAL_MACRO_NAMES: frozenset[str] = frozenset(
    {"panic", "unreachable", "todo", "unimplemented"}
)

_CALL_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"call"}),
    "typescript": frozenset({"call_expression"}),
    "tsx": frozenset({"call_expression"}),
    "javascript": frozenset({"call_expression"}),
    "go": frozenset({"call_expression"}),
    "rust": frozenset({"call_expression"}),
}

_PYTHON_EXECUTABLE_STATEMENT_NODES = frozenset(
    {
        "assert_statement",
        "break_statement",
        "class_definition",
        "continue_statement",
        "delete_statement",
        "exec_statement",
        "expression_statement",
        "function_definition",
        "future_import_statement",
        "global_statement",
        "import_from_statement",
        "import_statement",
        "nonlocal_statement",
        "pass_statement",
        "print_statement",
        "type_alias_statement",
    }
) | BRANCH_NODE_TYPES["python"] | TERMINAL_NODE_TYPES["python"]

_TYPESCRIPT_EXECUTABLE_STATEMENT_NODES = frozenset(
    {
        "break_statement",
        "class_declaration",
        "continue_statement",
        "debugger_statement",
        "expression_statement",
        "function_declaration",
        "function_signature",
        "generator_function_declaration",
        "labeled_statement",
        "lexical_declaration",
        "method_definition",
        "method_signature",
        "variable_declaration",
        "with_statement",
    }
) | _TYPESCRIPT_BRANCH_NODES | TERMINAL_NODE_TYPES["typescript"]

EXECUTABLE_STATEMENT_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": _PYTHON_EXECUTABLE_STATEMENT_NODES,
    "typescript": _TYPESCRIPT_EXECUTABLE_STATEMENT_NODES,
    "tsx": _TYPESCRIPT_EXECUTABLE_STATEMENT_NODES,
    "javascript": _TYPESCRIPT_EXECUTABLE_STATEMENT_NODES,
    "go": frozenset(
        {
            "assignment_statement",
            "break_statement",
            "const_declaration",
            "continue_statement",
            "dec_statement",
            "defer_statement",
            "expression_statement",
            "function_declaration",
            "go_statement",
            "goto_statement",
            "inc_statement",
            "labeled_statement",
            "method_declaration",
            "send_statement",
            "short_var_declaration",
            "var_declaration",
        }
    )
    | BRANCH_NODE_TYPES["go"]
    | TERMINAL_NODE_TYPES["go"],
    "rust": frozenset(
        {
            "expression_statement",
            "function_item",
            "function_signature_item",
            "let_declaration",
        }
    )
    | BRANCH_NODE_TYPES["rust"]
    | TERMINAL_NODE_TYPES["rust"],
}

# Switch-like containers and the case nodes they own. Counting both levels would
# report one decision twice, so :func:`branch_statement_lines` reports the case
# level and drops the container whenever the container actually has cases. Go has
# four container kinds and shares ``default_case`` across three of them, which is
# why the relation is expressed as two flat per-language sets rather than pairs.
_SWITCH_CONTAINER_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"match_statement"}),
    "typescript": frozenset({"switch_statement"}),
    "tsx": frozenset({"switch_statement"}),
    "javascript": frozenset({"switch_statement"}),
    "go": frozenset(
        {"expression_switch_statement", "type_switch_statement", "select_statement"}
    ),
    "rust": frozenset({"match_expression"}),
}
_SWITCH_CASE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"case_clause"}),
    "typescript": frozenset({"switch_case", "switch_default"}),
    "tsx": frozenset({"switch_case", "switch_default"}),
    "javascript": frozenset({"switch_case", "switch_default"}),
    "go": frozenset({"expression_case", "default_case", "type_case", "communication_case"}),
    "rust": frozenset({"match_arm"}),
}
# Body nodes that sit between a switch-like container and its cases: ``block``
# (python ``match``), ``switch_body`` (TS), ``match_block`` (rust). Go's cases are
# direct children of their container.
_SWITCH_BODY_TYPES: frozenset[str] = frozenset({"block", "switch_body", "match_block"})

# Fallback line classifiers for a file whose language has no grammar here. Pure
# text, no parse -- deliberately crude, and only ever reached when
# ``language_id`` is None/unknown or the parser is unavailable.
_BRANCH_KEYWORD_RE = re.compile(
    r"^\s*(if|elif|else|for|while|match|case|try|except|switch|catch|loop)\b"
)
_TERMINAL_KEYWORD_RE = re.compile(
    r"^\s*(return|raise|throw|panic|os\.Exit|std::process::exit|sys\.exit)\b"
)


def _walk(node: Node) -> Iterator[Node]:
    """Yield ``node`` and every descendant (order unspecified)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.children)


def _container_owns_cases(node: Node, case_types: frozenset[str]) -> bool:
    """Find direct cases or cases through one body wrapper; exclude nested switches."""
    for child in node.children:
        if child.type in case_types:
            return True
        if child.type in _SWITCH_BODY_TYPES:
            for grandchild in child.children:
                if grandchild.type in case_types:
                    return True
    return False


def branch_statement_lines(language_id: str, source: bytes) -> list[int]:
    """Return sorted unique 1-based branch lines; unavailable parsing yields [].

    Count condition-bearing heads, excluding else clauses. Switch-like containers
    count only when they own no cases; otherwise count their cases. Nested
    containers apply this rule independently. Unsafe parser versions still raise."""
    branch_types = BRANCH_NODE_TYPES.get(language_id)
    if branch_types is None:
        return []
    parser = get_parser(language_id)
    if parser is None:
        return []
    containers = _SWITCH_CONTAINER_TYPES.get(language_id, frozenset())
    case_types = _SWITCH_CASE_TYPES.get(language_id, frozenset())
    lines: set[int] = set()
    try:
        tree = parser.parse(source)
        for node in _walk(tree.root_node):
            if node.type not in branch_types:
                continue
            if node.type in containers and _container_owns_cases(node, case_types):
                continue
            lines.add(node.start_point[0] + 1)
    except Exception:
        return []
    return sorted(lines)


def _terminal_call_name(node: Node) -> str | None:
    """Return the callee text of a call ``node``, or None when unreadable."""
    callee = node.child_by_field_name("function")
    if callee is None or callee.text is None:
        return None
    return callee.text.decode("utf-8", errors="replace").strip()


def _is_bare_string_statement(node: Node) -> bool:
    if node.type != "expression_statement":
        return False
    contents = [child for child in node.named_children]
    return len(contents) == 1 and contents[0].type in {"string", "string_literal"}


@dataclass(frozen=True)
class StatementLines:
    """Grounding memberships captured by one native source traversal."""

    executable: frozenset[int]
    branches: frozenset[int]
    terminals: frozenset[int]


def statement_lines(language_id: str | None, source: bytes) -> StatementLines:
    """Capture executable, branch-head and terminal lines once.

    Unavailable/failed parsing retains keyword fallback only for branches and
    terminals; executable grounding remains fail-closed. Unsafe versions raise.
    Branch membership includes switch containers, unlike branch-count policy.
    """
    executable: set[int] = set()
    branches: set[int] = set()
    terminals: set[int] = set()
    language = language_id or ""
    parser = get_parser(language) if language else None
    if parser is not None and language in EXECUTABLE_STATEMENT_NODE_TYPES:
        statement_types = EXECUTABLE_STATEMENT_NODE_TYPES.get(language, frozenset())
        branch_types = BRANCH_NODE_TYPES.get(language, frozenset())
        terminal_types = TERMINAL_NODE_TYPES.get(language, frozenset())
        call_types = _CALL_NODE_TYPES.get(language, frozenset())
        call_names = TERMINAL_CALL_NAMES.get(language, frozenset())
        try:
            tree = parser.parse(source)
            for node in _walk(tree.root_node):
                line = node.start_point[0] + 1
                if node.type in statement_types and not _is_bare_string_statement(node):
                    executable.add(line)
                if node.type in branch_types:
                    branches.add(line)
                if node.type in terminal_types:
                    terminals.add(line)
                elif node.type in call_types and _terminal_call_name(node) in call_names:
                    terminals.add(line)
                elif node.type == "macro_invocation":
                    macro = node.child_by_field_name("macro")
                    if macro is not None and macro.text is not None:
                        name = macro.text.decode("utf-8", errors="replace").strip()
                        if name in TERMINAL_MACRO_NAMES:
                            terminals.add(line)
            return StatementLines(
                frozenset(executable), frozenset(branches), frozenset(terminals),
            )
        except Exception:
            pass
    rows = [row.decode("utf-8", errors="replace") for row in source.split(b"\n")]
    return StatementLines(
        frozenset(),
        frozenset(
            line for line, row in enumerate(rows, 1) if _BRANCH_KEYWORD_RE.match(row)
        ),
        frozenset(
            line for line, row in enumerate(rows, 1) if _TERMINAL_KEYWORD_RE.match(row)
        ),
    )
