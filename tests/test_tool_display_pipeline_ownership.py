"""Keep identifiable command rendering on the shared display pipeline.

The AST guard tracks command keys, local aliases, and key == command branches;
Codex stripping belongs only to _redacted_bash_command and its backend definition.
This bounded source-pattern check supplements real rendering tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "daydream"
_FORBIDDEN_NAME = "display_shell_command"
_DISPLAY_OWNER_MODULE = "daydream/ui/tools.py"
_DISPLAY_OWNER_FUNCTION = "_redacted_bash_command"
_BACKEND_MODULE = "daydream/backends/_codex_events.py"
_BACKEND_FACADE = "daydream/backends/codex.py"

#: The only modules permitted to reference the Codex display-variant strip step.
_PERMITTED_CALLERS = frozenset({
        "daydream/ui/tools.py",  # the display-pipeline owner: _redacted_bash_command
        "daydream/backends/codex.py",  # compatibility re-export
        "daydream/backends/_codex_events.py",  # strip definition + supervisor entry point
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


def _is_command_value(expr: ast.expr | None, aliases: set[str]) -> bool:
    """Recognize literal command lookups and local aliases of their values."""
    if expr is None:
        return False
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "command":
            return True
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "command"
        ):
            return True
    return False


def _redaction_lines(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[int]:
    """Find direct command redaction, including aliases through a local cap."""
    lines: list[int] = []

    def check(expr: ast.expr | None, aliases: set[str]) -> None:
        if expr is None:
            return
        for node in ast.walk(expr):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            if name in {"redact_structured_text", "redact_text"} and _is_command_value(node.args[0], aliases):
                lines.append(node.lineno)

    def scan(statements: list[ast.stmt], aliases: set[str]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = statement.value
                check(value, aliases)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        if _is_command_value(value, aliases):
                            aliases.add(target.id)
                        else:
                            aliases.discard(target.id)
            elif isinstance(statement, ast.If):
                check(statement.test, aliases)
                branch_aliases = aliases.copy()
                if any(isinstance(node, ast.Compare)
                    and any(isinstance(item, ast.Constant) and item.value == "command" for item in node.comparators)
                    and isinstance(node.left, ast.Name)
                    and node.left.id == "key"
                    for node in ast.walk(statement.test)
                ):
                    branch_aliases.add("value")
                scan(statement.body, branch_aliases)
                scan(statement.orelse, aliases.copy())
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                scan(statement.body, aliases.copy())
                scan(statement.orelse, aliases.copy())
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            else:
                for field in ast.iter_fields(statement):
                    if isinstance(field[1], ast.expr):
                        check(field[1], aliases)

    scan(function.body, set())
    return lines


def _command_pipeline_violations(source: str, module: str) -> list[int]:
    """Report recognized command shortcuts outside the one display owner."""
    tree = ast.parse(source)
    owner = next((node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _DISPLAY_OWNER_FUNCTION
        ), None,
    ) if module == _DISPLAY_OWNER_MODULE else None

    def within_owner(line: int) -> bool:
        return owner is not None and owner.end_lineno is not None and owner.lineno <= line <= owner.end_lineno

    if module == _BACKEND_MODULE:
        strip_lines: list[int] = []
    else:
        reexports = {
            node.lineno for node in tree.body
            if module == _BACKEND_FACADE and isinstance(node, ast.ImportFrom)
            and node.module == "daydream.backends._codex_events"
        }
        strip_lines = [
            line for line in _strip_reference_lines(source)
            if not within_owner(line) and line not in reexports
        ]
        strip_lines += [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "_CD_PREFIX_RE"
            and not within_owner(node.lineno)
        ]
    redaction_lines = [line
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not owner
        for line in _redaction_lines(node)
    ]
    return sorted(set(strip_lines + redaction_lines))

def test_command_display_pipeline_has_one_render_owner() -> None:
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        offenders += [f"{rel}:{line}" for line in _command_pipeline_violations(path.read_text(encoding="utf-8"), rel)]
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

@pytest.mark.parametrize("module", ["daydream/ui/new_surface.py", "daydream/ui/tools.py"])
def test_new_command_renderer_cannot_redact_a_capped_command(module: str) -> None:
    source = ("def render(args):\n" "    return redact_structured_text(args['command'][:200])\n")
    assert _strip_reference_lines(source) == []  # the previous guard missed this exact shortcut
    assert _command_pipeline_violations(source, module) == [2]

@pytest.mark.parametrize("source, expected_lines",
    [
        ("def render(args):\n" "    return redact_structured_text(args['command'])\n", [2],),
        (
            "def render(args):\n"
            "    command = args.get('command', '')[:200]\n"
            "    return redact_structured_text(command)\n",
            [3],
        ),
        (
            "def render(args):\n"
            "    command = args['command']\n"
            "    command = display_shell_command(command)\n"
            "    return command\n",
            [3],
        ),
        (
            "def render(args):\n"
            "    command = _CD_PREFIX_RE.sub('', args['command'], count=1)\n"
            "    return redact_structured_text(command[:200])\n",
            [2, 3],
        ),
    ], ids=["direct-redaction", "cap-before-redact-alias", "copied-strip-call", "copied-strip-regex"],
)
def test_command_pipeline_shortcuts_are_reported(source: str, expected_lines: list[int]) -> None:
    assert _command_pipeline_violations(source, "daydream/ui/tools.py") == expected_lines

def test_unrelated_redaction_and_strip_only_supervisor_are_outside_guard() -> None:
    source = (
        "def render(args):\n"
        "    command = args['command']\n"
        "    return redact_structured_text(args['description'])\n"
        "\n"
        "def supervise(event):\n"
        "    command = event.input.get('command')\n"
        "    return supervisor_shell_command(command)\n"
    )
    assert _command_pipeline_violations(source, "daydream/agent.py") == []


def test_backend_facade_allows_only_strip_reexport() -> None:
    source = (
        "from daydream.backends._codex_events import display_shell_command\n"
        "def render(command):\n"
        "    return display_shell_command(command)\n"
    )
    assert _command_pipeline_violations(source, _BACKEND_FACADE) == [3]
