"""CUT-08 AST sweep: verify the legacy _log_debug system is fully removed.

Walks the AST of every .py file under daydream/ and tests/ and rejects:

1. Name nodes referencing forbidden symbols (catches direct calls and
   references)
2. Attribute nodes accessing .debug_log (catches AgentState.debug_log
   and similar attribute lookups even on aliased objects)
3. ImportFrom nodes importing forbidden names (catches the canonical
   Pitfall 13 lazy import in codex.py:_raw_log that grep alone misses)
4. String-literal Constant nodes containing forbidden log prefixes
   (catches accidental re-introduction of the bracketed log prefixes,
   such as a raw print that uses the old format).

Self-excludes this test file via __file__ comparison so its own
forbidden-literal constants don't trigger a failure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# AST-level forbidden symbols (Name, Attribute, ImportFrom)
FORBIDDEN_NAMES: set[str] = {
    "_log_debug",
    "_raw_log",
    "_ui_debug",
    "set_debug_log",
    "get_debug_log",
}
FORBIDDEN_ATTRS: set[str] = {
    "debug_log",  # AgentState.debug_log
}

# String-literal forbidden prefixes — re-introduction of any of these
# via raw print() / logger.info() / etc. would resurrect the legacy
# debug-log format.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "[REVERT]",
    "[PARSE_FAIL]",
    "[STAGE]",
    "[TTT_REVIEW]",
    "[TTT_PLAN]",
    "[PRE_SCAN]",
    "[PROMPT]",
    "[TEXT]",
    "[TOOL_USE]",
    "[TOOL_RESULT]",
    "[TOOL_RESULT_PANEL]",
    "[COST]",
    "[TOKENS]",
    "[CODEX_RAW]",
    "[CODEX_WARN]",
    "[CODEX_UNHANDLED]",
    "[SCHEMA]",
    "[SCHEMA_OK]",
    "[SCHEMA_MISS]",
    "[SCHEMA_FALLBACK]",
    "[STRUCTURED_OUTPUT]",
    "[EXECUTE_INIT_ERROR]",
    "[EXECUTE_ERROR]",
    "[PHASE2_ERROR]",
    "[PARSE_FALLBACK]",
    "[THINKING]",
    "[UI_HEADER]",
)

SOURCE_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / "daydream",
    PROJECT_ROOT / "tests",
)


def _all_py_files() -> list[Path]:
    files: list[Path] = []
    for d in SOURCE_DIRS:
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            files.append(p)
    return sorted(files)


@pytest.mark.parametrize(
    "py_file",
    _all_py_files(),
    ids=lambda p: str(p.relative_to(PROJECT_ROOT)),
)
def test_no_legacy_debug_logging_references(py_file: Path) -> None:
    """CUT-08: every .py file is free of forbidden debug-logging symbols and prefixes."""
    # Self-exclude: this file references the forbidden literals in
    # constants and docstrings; scanning itself would always fail.
    if py_file.resolve() == Path(__file__).resolve():
        return

    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    rel = py_file.relative_to(PROJECT_ROOT)

    for node in ast.walk(tree):
        # 1. Name (catches direct references, e.g. _log_debug, even when
        # the import was top-level: every actual call is a Name node).
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            pytest.fail(f"{rel}:{node.lineno}: forbidden Name reference '{node.id}'")

        # 2. Attribute (catches state.debug_log, _state.debug_log, any
        # .debug_log access regardless of base object).
        if isinstance(node, ast.Attribute) and (
            node.attr in FORBIDDEN_NAMES or node.attr in FORBIDDEN_ATTRS
        ):
            pytest.fail(f"{rel}:{node.lineno}: forbidden Attribute '.{node.attr}'")

        # 3. ImportFrom (catches lazy imports inside function bodies —
        # the canonical Pitfall 13 case).
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    pytest.fail(
                        f"{rel}:{node.lineno}: forbidden ImportFrom alias '{alias.name}'"
                    )

        # 4. String literal constants — catch re-introduction of legacy
        # log prefixes via raw print(), logger.info(), etc.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for prefix in FORBIDDEN_PREFIXES:
                if prefix in node.value:
                    pytest.fail(
                        f"{rel}:{node.lineno}: forbidden literal prefix {prefix!r} "
                        f"in string constant"
                    )
