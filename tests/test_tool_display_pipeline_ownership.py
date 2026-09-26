"""#1227: the Codex display-variant strip has exactly one render-path owner.

AST-based because the owner imports the helper inside its function body
(`daydream/ui/tools.py:129`) — a top-level-import grep would miss a copied
function-local call site (the shape `tests/test_cutover_ast.py` CUT-08 documents).
Scan scope is the production package: `tests/` references the helper legitimately.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "daydream"
_FORBIDDEN_NAME = "display_shell_command"

#: The only modules permitted to reference the Codex display-variant strip step.
_PERMITTED_CALLERS = frozenset(
    {
        "daydream/ui/tools.py",  # the display-pipeline owner: _redacted_bash_command
        "daydream/backends/codex.py",  # the helper's definition + the strip-only supervisor entry point
    }
)


def _strip_reference_lines(source: str) -> list[int]:
    """Line numbers referencing the strip helper, via any import or call form."""
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == _FORBIDDEN_NAME:
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == _FORBIDDEN_NAME:
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            lines += [node.lineno for alias in node.names if alias.name == _FORBIDDEN_NAME]
    return sorted(lines)


def test_only_permitted_modules_reference_the_codex_display_variant() -> None:
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _PERMITTED_CALLERS:
            continue
        offenders += [f"{rel}:{line}" for line in _strip_reference_lines(path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_scanner_reports_a_reintroduced_strip_call() -> None:
    """The guard's own discrimination check: feed it the shape it exists to catch."""
    reintroduced = (
        "def render(name, command):\n"
        "    from daydream.backends.codex import display_shell_command\n"
        "    if name == 'shell':\n"
        "        command = display_shell_command(command)\n"
        "    return command\n"
    )
    assert _strip_reference_lines(reintroduced) == [2, 4]
    # ...and neither an aliased import nor an attribute call bypasses the scan:
    assert _strip_reference_lines("from daydream.backends.codex import display_shell_command as dsc\n") == [1]
    assert _strip_reference_lines("from daydream.backends import codex\n\ncodex.display_shell_command('x')\n") == [3]
    # ...and it stays quiet on a mere string mention:
    assert _strip_reference_lines("value = 'display_shell_command'\n") == []


def test_permitted_callers_still_reference_the_helper() -> None:
    """The allowlist cannot rot into a blanket exemption."""
    for rel in sorted(_PERMITTED_CALLERS):
        source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert _strip_reference_lines(source), f"{rel} is allowlisted but no longer references {_FORBIDDEN_NAME}"
