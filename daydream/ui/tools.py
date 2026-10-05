"""Tool-call rendering helpers.

Task-id harvesting/labeling, callback-progress formatting, tool-arg
colorization, and the shared header/body/result builders consumed by the
live tool-call panels.
"""

import re

from rich.console import Group
from rich.markdown import Markdown
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from daydream.ui.colorize import _colorize_line, _detect_shell_syntax
from daydream.ui.console import _get_gradient_color, _interpolate_color
from daydream.ui.theme import (
    _BACKGROUND_TASK_TOOLS,
    _EDIT_PREVIEW_MAX_LINES,
    _LAUNCH_TASK_TOOLS,
    _MECHANICAL_TOOL_ARGS,
    _RESULT_MAX_LINES,
    _TASK_PROMPT_MAX_LINES,
    _TODO_TASK_TOOLS,
    NEON_COLORS,
    STATUS_CONFIG,
    STYLE_BOLD_CYAN,
    STYLE_BOLD_PINK,
    STYLE_CYAN,
    STYLE_DIM,
    STYLE_FG,
    STYLE_GREEN,
    STYLE_ORANGE,
    STYLE_PURPLE,
    STYLE_RED,
    STYLE_YELLOW,
    mystical_term,
)

# Patterns for harvesting the assigned task id out of an originating tool's result string.
_LAUNCH_TASK_ID_PATTERN = re.compile(r"\bCommand running in background with ID:\s*([A-Za-z0-9]+)\b")
_TASKCREATE_ID_PATTERN = re.compile(r"Task #(\d+)")

# Single-line icon + primary-arg spec for the callback/parallel render path,
# which cannot open Rich panels (concurrent agents would each fight for the
# shared console's Live context). ``_PRIMARY_TOOL_ARG`` is the shared source of
# truth for primary argument selection across single-line render surfaces.
_CALLBACK_TOOL_ICONS = {
    "Read": "📜",
    "Write": "⛏️",
    "Edit": "⚕",
    "Glob": "🔮",
    "Grep": "🧙",
    "Bash": "🔨",
    "shell": "🔨",
    "Skill": "✨",
    "TodoWrite": "🔧",
    **{name: "🎠" for name in (*_BACKGROUND_TASK_TOOLS, *_TODO_TASK_TOOLS)},
}
# Shared by Bash display and agent_stream._summarize_input.
_BASH_COMMAND_MAX_CHARS = 200
_PRIMARY_TOOL_ARG = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "NotebookEdit": ("notebook_path", "file_path"),
    "Glob": ("pattern",),
    "Grep": ("pattern",),
    "Bash": ("command", "description"),
    "shell": ("command", "description"),
    "Skill": ("skill",),
}


def _primary_tool_value(name: str, args: dict[str, object]) -> tuple[str, str | None]:
    """Return (value, key) for the preferred nonempty string argument.

    Unknown tools fall back to the first non-mechanical string. The key lets
    callers color paths separately from patterns; Bash prefers command."""
    for key in _PRIMARY_TOOL_ARG.get(name, ()):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    for key, value in args.items():
        if key in _MECHANICAL_TOOL_ARGS or isinstance(value, bool):
            continue
        if isinstance(value, str) and value.strip():
            return value, key
    return "", None


def _primary_value_style(key: str | None) -> Style:
    """Color path roles cyan; patterns and other arguments stay orange."""
    if key is not None and key.endswith("path"):
        return STYLE_CYAN
    return STYLE_ORANGE


def _redacted_bash_command(
    name: str,
    command: str,
    *,
    ellipsis: bool = False,
    max_chars: int = _BASH_COMMAND_MAX_CHARS,
) -> str:
    """Redact (and Codex 'shell'-strip) a Bash command for single-line display.

    Codex ('shell') commands carry the replayable cd-prefixed payload; only the
    display variant strips it. The complete value is redacted before truncation
    so a credential cannot be shortened into an unmatchable fragment.
    """
    from daydream.backends.codex import display_shell_command
    from daydream.redaction import redact_structured_text

    if name == "shell":
        command = display_shell_command(command)
    command = redact_structured_text(command)
    if len(command) > max_chars:
        command = command[:max_chars] + ("..." if ellipsis else "")
    return command


def _parse_assigned_task_id(name: str, output: str) -> str | None:
    """Extract launch-tool background ids or TaskCreate numeric ids; else None."""
    if name in _LAUNCH_TASK_TOOLS:
        match = _LAUNCH_TASK_ID_PATTERN.search(output)
        return match.group(1) if match else None
    if name == "TaskCreate":
        match = _TASKCREATE_ID_PATTERN.search(output)
        return match.group(1) if match else None
    return None


def _task_label_ns_key(name: str, task_id: str) -> str:
    """Separate background and todo id namespaces so identical ids cannot collide."""
    if name in _LAUNCH_TASK_TOOLS:
        return f"bg:{task_id}"
    return f"tc:{task_id}"


def _derive_task_label(args: dict[str, object], task_id: str) -> str:
    """Choose subject, description, subagent, first command/prompt line, or task id.

    Labels are capped at 80 characters; task-id fallbacks are unchanged."""
    for key in ("subject", "description", "subagent_type"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value[:80]
    for key in ("command", "prompt"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.splitlines()[0][:80]
    return task_id


def _task_id_key(name: str) -> str:
    """Background tools use task_id; todo tools use taskId."""
    return "task_id" if name in _BACKGROUND_TASK_TOOLS else "taskId"


def _label_source_name(name: str) -> str:
    """Map task consumers to the originating label namespace."""
    return "Bash" if name in _BACKGROUND_TASK_TOOLS else "TaskCreate"


def format_callback_progress(
    name: str,
    args: dict[str, object],
    label: str | None,
    max_len: int = _BASH_COMMAND_MAX_CHARS,
) -> Text:
    """Render one indented tool status without opening a competing Live panel.

    Task tools lead with a human label and demote ids; mechanical flags are hidden.
    Other tools use the same primary argument and role colors as panel headers."""
    line = Text("    ")
    icon = _CALLBACK_TOOL_ICONS.get(name)
    if icon:
        line.append(f"{icon} ", style=STYLE_ORANGE)
    line.append(name, style=STYLE_BOLD_PINK)

    if name in _BACKGROUND_TASK_TOOLS or name in _TODO_TASK_TOOLS:
        task_id = str(args.get(_task_id_key(name), "")).strip()
        lead = label or _derive_task_label(args, task_id)
        suffix = _format_label_and_id_str(lead, task_id if task_id != lead else "")
        if suffix:
            line.append(" ")
            line.append(suffix, style=STYLE_CYAN)
        return line

    value, key = _primary_tool_value(name, args)
    if value:
        if name in ("Bash", "shell") and key == "command":
            value = _redacted_bash_command(name, value, max_chars=max_len)
        else:
            value = value[:max_len]
        line.append(" ")
        line.append(value, style=_primary_value_style(key))
    return line


def format_callback_text(text: str) -> Text:
    """Render interleaved agent narration dimmed beneath progress headers."""
    return Text(f"    {text}", style=STYLE_DIM)


def _colorize_tool_args(args: dict[str, object]) -> Text:
    """Color visible arguments by type, distinguishing paths from other strings."""
    result = Text()

    items = [(key, value) for key, value in args.items() if key not in _MECHANICAL_TOOL_ARGS]

    for i, (key, value) in enumerate(items):
        if i > 0:
            result.append(", ", style=STYLE_FG)

        result.append(str(key), style=STYLE_CYAN)
        result.append("=", style=STYLE_PURPLE)

        if isinstance(value, bool):
            result.append(str(value), style=STYLE_PURPLE)
        elif isinstance(value, (int, float)):
            result.append(str(value), style=STYLE_YELLOW)
        elif isinstance(value, str):
            if "/" in value or value.endswith((".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml")):
                result.append(value, style=STYLE_CYAN)
            else:
                result.append(value, style=STYLE_ORANGE)
        elif value is None:
            result.append("None", style=STYLE_PURPLE)
        else:
            result.append(str(value), style=STYLE_FG)

    return result


def _format_label_and_id_str(label: str | None, task_id: str, *, id_prefix: str = "") -> str:
    """Join an optional label and parenthesized id, with an optional id prefix."""
    parts: list[str] = []
    if label is not None:
        parts.append(label)
    if task_id.strip():
        parts.append(f"({id_prefix}{task_id})")
    return " ".join(parts)


def _append_label_and_id(header_line: Text, label: str | None, task_id: str, *, id_prefix: str = "") -> None:
    """Append a cyan label and dim id; omit each absent component."""
    if label is not None:
        header_line.append(' → "')
        header_line.append(label, style=STYLE_CYAN)
        header_line.append('"')
    if task_id.strip():
        header_line.append(f" ({id_prefix}{task_id})", style=STYLE_DIM)


def _append_gradient_preview(content: Text, string: str, start_hex: str, end_hex: str, *, bold: bool) -> None:
    """Append an Edit old/new preview with a per-char color gradient.

    Truncates to ``_EDIT_PREVIEW_MAX_LINES`` lines and interpolates each
    character's color from ``start_hex`` to ``end_hex``.
    """
    lines = string.split("\n")
    preview = "\n".join(lines[:_EDIT_PREVIEW_MAX_LINES])
    if len(lines) > _EDIT_PREVIEW_MAX_LINES:
        preview += "\n..."
    preview_len = max(len(preview) - 1, 1)
    for i, char in enumerate(preview):
        if char == "\n":
            content.append("\n  ")
        else:
            t = i / preview_len
            color = _interpolate_color(start_hex, end_hex, t)
            content.append(char, style=Style(color=color, bold=bold or None))


def _append_arg_field(
    content: Text,
    key: str,
    value: str,
    style: str | Style,
    *,
    indent: str = "",
) -> None:
    """Append one ``key=value`` argument row on a fresh indented line."""
    content.append(f"\n{indent}")
    content.append(f"{key}=", style=STYLE_PURPLE)
    content.append(value, style=style)


def _build_tool_header(
    name: str,
    args: dict[str, object],
    quiet_mode: bool = False,
    *,
    label: str | None = None,
) -> Text:
    """Render a tool header, demoting task ids below the resolved human label."""
    content = Text()

    if name == "Skill":
        header_line = Text()
        header_line.append("✨ ", style=STYLE_PURPLE)
        header_line.append(f"{mystical_term('Skill')} ", style=Style(color=NEON_COLORS["pink"], italic=True))
        header_line.append("Skill", style=STYLE_BOLD_CYAN)
        content.append_text(header_line)
        content.append("\n  ")

        skill_name = str(args.get("skill", ""))
        for i, char in enumerate(skill_name):
            position = i / max(len(skill_name) - 1, 1)
            color = _get_gradient_color(position)
            content.append(char, style=Style(color=color, bold=True))

        if not quiet_mode:
            skill_args = args.get("args")
            if skill_args:
                _append_arg_field(content, "args", str(skill_args), STYLE_ORANGE, indent="  ")

        return content

    icon = _CALLBACK_TOOL_ICONS.get(name, "🎠")
    icon_style = {"Glob": STYLE_PURPLE, "Grep": STYLE_PURPLE, "Edit": STYLE_CYAN}.get(name, STYLE_ORANGE)
    content.append(f"{icon} ", style=icon_style)
    content.append("Bash" if name == "shell" else name, style=STYLE_BOLD_PINK)

    # Background-task tools: lead with the resolved label, demote the opaque
    # task_id to a dim suffix, and never surface block/timeout plumbing.
    if name in _BACKGROUND_TASK_TOOLS:
        task_id = str(args.get(_task_id_key(name), ""))
        _append_label_and_id(content, label, task_id)
        return content

    # Todo-list tools lead with the todo subject and demote the numeric id;
    # TaskUpdate appends its status change. Never surface plumbing.
    if name in _TODO_TASK_TOOLS:
        if name == "TaskCreate":
            subject = str(args.get("subject", "")).strip()
            if subject:
                content.append("  ")
                content.append(subject, style=STYLE_CYAN)
        else:
            task_id = str(args.get(_task_id_key(name), ""))
            _append_label_and_id(content, label, task_id, id_prefix="#")
            if name == "TaskUpdate":
                status = str(args.get("status", "")).strip()
                if status:
                    content.append(" → ")
                    content.append(status, style=STYLE_CYAN)
        return content

    if name == "TodoWrite":

        todos = args.get("todos", [])
        if isinstance(todos, list):
            for todo in todos:
                if isinstance(todo, dict):
                    todo_content = todo.get("content", "")
                    if not todo_content:
                        continue
                    status = todo.get("status", "pending")
                    config = STATUS_CONFIG.get(status, STATUS_CONFIG["pending"])
                    content.append("\n")
                    content.append(f"{config['icon']} ", style=Style(color=config["color"]))
                    content.append(todo_content, style=Style(color=config["color"]))
        else:
            content.append("\n")
            content.append_text(_colorize_tool_args(args))

        return content

    if name in ("Bash", "shell"):

        description = str(args.get("description", ""))
        if description:
            content.append("\n")
            content.append(description, style=STYLE_CYAN)

        raw_command = str(args.get("command", ""))
        command = _redacted_bash_command(name, raw_command, ellipsis=True)
        if command.strip():
            content.append("\n")
            content.append("$ ", style=STYLE_DIM)
            content.append(command, style=STYLE_DIM)

        return content

    if name == "Write":
        file_path = str(args.get("file_path", ""))

        content.append(f" {mystical_term('Write')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(file_path, style=STYLE_CYAN)

        return content

    if name in ("Glob", "Grep"):
        # Glob and Grep share one header shape (pattern, then an optional path);
        # Grep only appends the two optional filter fields, in that order.
        pattern = str(args.get("pattern", ""))
        search_path = str(args.get("path", ""))

        content.append(f" {mystical_term(name)}... ", style=f"{STYLE_PURPLE} italic")
        content.append(pattern, style=STYLE_ORANGE)

        if search_path:
            _append_arg_field(content, "path", search_path, STYLE_CYAN)

        if name == "Grep":
            for field in ("glob", "type"):
                value = str(args.get(field, ""))
                if value:
                    _append_arg_field(content, field, value, STYLE_YELLOW)

        return content

    if name == "Read":
        file_path = str(args.get("file_path", ""))
        offset = args.get("offset")
        limit = args.get("limit")

        content.append(f" {mystical_term('Read')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(file_path, style=STYLE_CYAN)

        if offset is not None or limit is not None:
            content.append("\n")
            if offset is not None:
                content.append("offset=", style=STYLE_PURPLE)
                content.append(str(offset), style=STYLE_YELLOW)
            if limit is not None:
                if offset is not None:
                    content.append(", ", style=STYLE_FG)
                content.append("limit=", style=STYLE_PURPLE)
                content.append(str(limit), style=STYLE_YELLOW)

        return content

    if name == "Edit":
        file_path = str(args.get("file_path", ""))
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        replace_all = args.get("replace_all", False)

        content.append(f" {mystical_term('Edit')}... ", style=Style(color=NEON_COLORS["purple"], italic=True))
        content.append(file_path, style=STYLE_CYAN)

        if replace_all:
            _append_arg_field(content, "replace_all", "True", STYLE_PURPLE)

        content.append("\n")

        content.append("\n")
        content.append("  ⊗ ", style=STYLE_RED)
        content.append("BLOCKED", style=Style(color=NEON_COLORS["red"], bold=True))
        content.append("\n")
        if old_string:
            _append_gradient_preview(content, old_string, NEON_COLORS["red"], NEON_COLORS["orange"], bold=False)

        content.append("\n\n")
        content.append("  ✓ ", style=STYLE_GREEN)
        content.append("FLOWING", style=Style(color=NEON_COLORS["green"], bold=True))
        content.append("\n")
        if new_string:
            _append_gradient_preview(content, new_string, NEON_COLORS["cyan"], NEON_COLORS["green"], bold=True)

        return content

    # Task description/prompt render as markdown below the header (see
    # _build_tool_body_extras).
    if name == "Task":
        subagent = str(args.get("subagent_type", ""))
        if subagent:
            content.append("  ")
            content.append(subagent, style=STYLE_CYAN)
        return content


    if args:
        content.append("\n")
        args_text = _colorize_tool_args(args)
        content.append_text(args_text)

    return content


def _build_tool_body_extras(name: str, args: dict[str, object]) -> list[Markdown]:
    """Render Task description/prompt and TaskCreate description as Markdown."""
    if name == "TaskCreate":
        description = str(args.get("description", "")).strip()
        return [Markdown(description)] if description else []
    if name != "Task":
        return []
    extras: list[Markdown] = []
    description = str(args.get("description", "")).strip()
    prompt = str(args.get("prompt", "")).strip()
    if description:
        extras.append(Markdown(f"**{description}**"))
    if prompt:
        prompt_lines = prompt.split("\n")
        max_prompt_lines = _TASK_PROMPT_MAX_LINES
        if len(prompt_lines) > max_prompt_lines:
            remaining = len(prompt_lines) - max_prompt_lines
            truncated_prompt = "\n".join(prompt_lines[:max_prompt_lines])
            truncated_prompt += f"\n\n_... ({remaining} more lines)_"
            extras.append(Markdown(truncated_prompt))
        else:
            extras.append(Markdown(prompt))
    return extras


def _build_result_content(
    content: str,
    is_error: bool = False,
    max_lines: int = _RESULT_MAX_LINES,
) -> Text | Syntax | Group:
    """Truncate and color result lines, highlighting recognized shell output."""
    lines = content.split("\n")
    truncated = False
    total_lines = len(lines)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    if not is_error and _detect_shell_syntax(content):
        display_content = "\n".join(lines)
        syntax = Syntax(
            display_content,
            "bash",
            theme="dracula",
            line_numbers=False,
            word_wrap=True,
        )

        if truncated:
            truncation_text = Text()
            truncation_text.append(
                f"\n... ({total_lines - max_lines} more lines)",
                style=Style(color=NEON_COLORS["yellow"], italic=True),
            )
            return Group(syntax, truncation_text)
        return syntax

    result_text = Text()
    for i, line in enumerate(lines):
        result_text.append_text(_colorize_line(line, is_error))
        if i < len(lines) - 1:
            result_text.append("\n")

    if truncated:
        result_text.append("\n")
        result_text.append(
            f"... ({total_lines - max_lines} more lines)",
            style=Style(color=NEON_COLORS["yellow"], italic=True),
        )

    return result_text
