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
_LAUNCH_TASK_TOOLS = frozenset({"Bash", "Agent", "Task"})

# Single-line icon + primary-arg spec for the callback/parallel render path,
# which cannot open Rich panels (concurrent agents would each fight for the
# shared console's Live context). These mirror the per-tool choices baked into
# ``_build_tool_header`` so a parallel-fix line names the same icon and primary
# argument the panel header would lead with.
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
_PRIMARY_TOOL_ARG = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "NotebookEdit": ("notebook_path", "file_path"),
    "Glob": ("pattern",),
    "Grep": ("pattern",),
    "Bash": ("description", "command"),
    "shell": ("description", "command"),
    "Skill": ("skill",),
}


def _primary_tool_value(name: str, args: dict[str, object]) -> tuple[str, str | None]:
    """Return the meaningful primary-argument value for a tool's progress line.

    Mirrors the per-tool choice in ``_build_tool_header`` (Read/Edit/Write →
    ``file_path``, Grep/Glob → ``pattern``, Bash → ``description``/``command``).
    Falls back to the first non-mechanical, non-boolean value so an unknown tool
    still shows something meaningful rather than a stray flag — the old blind
    ``next(iter(args.values()))`` surfaced ``replace_all=False`` as ``"False"``.

    Args:
        name: The tool's name.
        args: The tool's input args.

    Returns:
        ``(value, key)`` — the primary value as a string and the arg key it came
        from (``None`` when no key matched). ``("", None)`` when nothing
        meaningful exists. The key lets the caller color by argument *role*
        (path vs pattern) the way the panel header does.

    """
    for key in _PRIMARY_TOOL_ARG.get(name, ()):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    for key, value in args.items():
        if key in _MECHANICAL_TOOL_ARGS or isinstance(value, bool):
            continue
        if isinstance(value, str) and value.strip():
            return value, key
        if isinstance(value, (int, float)):
            return str(value), key
    return "", None


def _primary_value_style(key: str | None) -> Style:
    """Color the primary arg by role, matching ``_build_tool_header``.

    Path-bearing keys render cyan (Read/Edit/Write file_path); ``pattern`` and
    everything else render orange (Grep/Glob), the same split the panel header
    uses — so a glob pattern that happens to end in ``.py`` stays orange.
    """
    if key is not None and key.endswith("path"):
        return STYLE_CYAN
    return STYLE_ORANGE


def _parse_assigned_task_id(name: str, output: str) -> str | None:
    """Extract the assigned task id from an originating tool's result string.

    Backgrounded launch tools (Bash/Agent/Task) report their id in a
    ``Command running in background with ID: <id>. …`` prose string; TaskCreate
    reports it as ``Task #<N> created successfully: …``.

    Args:
        name: The originating tool's name.
        output: The tool's result string.

    Returns:
        The captured id, or None if the input does not match the expected shape.
    """
    if name in _LAUNCH_TASK_TOOLS:
        match = _LAUNCH_TASK_ID_PATTERN.search(output)
        return match.group(1) if match else None
    if name == "TaskCreate":
        match = _TASKCREATE_ID_PATTERN.search(output)
        return match.group(1) if match else None
    return None


def _task_label_ns_key(name: str, task_id: str) -> str:
    """Return a namespaced ``_task_labels`` dict key for *task_id*.

    Background launch tools (Bash/Agent/Task) use opaque alphanumeric ids;
    TaskCreate uses small integer ids.  Storing both in one flat dict risks
    collision (e.g. a background id of ``'1'`` colliding with TaskCreate id
    ``1``).  This function prefixes each id with a short namespace tag so the
    two id spaces never share keys.

    Args:
        name: The originating tool's name.
        task_id: The raw assigned id string.

    Returns:
        ``"bg:<task_id>"`` for launch tools, ``"tc:<task_id>"`` for TaskCreate.
    """
    if name in _LAUNCH_TASK_TOOLS:
        return f"bg:{task_id}"
    return f"tc:{task_id}"


def _derive_task_label(args: dict[str, object], task_id: str) -> str:
    """Derive a human label from an originating Task-family call's input args.

    Prefers ``subject`` (todo-list ``TaskCreate``), then ``description``,
    then ``subagent_type``, then the first line of ``command``/``prompt``,
    falling back to the bare id.

    Args:
        args: The originating tool call's input args.
        task_id: The assigned id, used as the fallback when no label key is set.

    Returns:
        The derived label, or ``task_id`` if no meaningful key is present.

    """
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
    """Return the arg key that carries a task id for a Task-family tool.

    Background-task tools (``TaskOutput``/``TaskStop``) use ``"task_id"``;
    todo-list tools (``TaskGet``/``TaskUpdate``/…) use ``"taskId"``.  Keeping
    this mapping in one place eliminates the three-way duplication that existed
    across ``resolve_call_label``, ``format_callback_progress``, and
    ``_build_tool_header``.
    """
    return "task_id" if name in _BACKGROUND_TASK_TOOLS else "taskId"


def format_callback_progress(
    name: str,
    args: dict[str, object],
    label: str | None,
    max_len: int = 80,
) -> Text:
    """Build a styled one-line progress entry for the callback render path.

    The callback (parallel-fix/quiet) render path streams a single status line
    per tool call instead of a Rich panel — concurrent agents cannot each open
    their own Rich ``Live`` on the shared console. The line still carries theme
    styling (tool icon, pink tool name, cyan path / orange pattern) so it matches
    the rest of the UI, and it names the same primary argument the panel header
    would lead with rather than a blind first-value dump.

    Task-family tools must never surface the opaque ``name + task_id`` dump or
    mechanical ``block``/``timeout`` args here; they lead with the resolved label
    (when known) and otherwise fall back to a non-opaque derivation
    (``subject``/``description``/first command line), demoting the bare id to a
    parenthesized suffix.

    Args:
        name: The tool's name.
        args: The tool's input args.
        label: The resolved label for the call's id, or None when unknown.
        max_len: Maximum length of the primary-argument value.

    Returns:
        A styled, indented Rich Text line that nests under the fix-progress header.

    """
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
        line.append(" ")
        line.append(value[:max_len], style=_primary_value_style(key))
    return line


def format_callback_text(text: str) -> Text:
    """Style a line of agent narration for the callback/parallel render path.

    Narration interleaves across concurrent fix agents, so it renders dim and
    indented — secondary to the colored ``[N/total] Fixing:`` headers, but still
    visible as a liveness signal during long parallel fixes.

    Args:
        text: A single line of agent narration.

    Returns:
        A dim, indented Rich Text line.

    """
    return Text(f"    {text}", style=STYLE_DIM)


def _colorize_tool_args(args: dict[str, object]) -> Text:
    """Apply neon syntax highlighting to tool call arguments.

    Colorizes parameter names and values based on their types:
    - Keys in cyan
    - Strings in orange (paths in cyan)
    - Numbers in yellow
    - Booleans in purple

    Args:
        args: Dictionary of argument key-value pairs.

    Returns:
        Rich Text with neon styling applied.

    """
    result = Text()

    # Drop mechanical/plumbing keys entirely so the line leads with meaningful args.
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
    """Return the plain-string ``lead (id)`` suffix shared by all task-tool paths.

    This is the single source of truth for the lead-label / id-suffix pattern
    (R13). Both the Rich panel path (``_append_label_and_id``) and the plain
    callback/quiet path (``format_callback_progress``) delegate here so the
    logic is never duplicated.

    Args:
        label: Resolved human-readable label, or None when unresolved.
        task_id: The opaque/numeric id to demote to a parenthesized suffix.
        id_prefix: Optional prefix for the id (e.g. ``"#"`` for todo ids).

    Returns:
        A plain string such as ``'some label (42)'`` or ``'(42)'``.

    """
    parts: list[str] = []
    if label is not None:
        parts.append(label)
    if task_id.strip():
        parts.append(f"({id_prefix}{task_id})")
    return " ".join(parts)


def _append_label_and_id(header_line: Text, label: str | None, task_id: str, *, id_prefix: str = "") -> None:
    """Append the shared ``→ "label" (id)`` suffix to a task-tool header line.

    Used by both the background-task (``TaskOutput``/``TaskStop``) and todo-list
    (``TaskGet``/``TaskUpdate``/``TaskList``) header branches so the label/id
    formatting lives in one place (R13). When ``label`` is ``None`` only the
    dimmed id is shown; when ``task_id`` is empty (e.g. id-less ``TaskList``)
    no id suffix is appended at all.

    Args:
        header_line: The header Text to append to (mutated in place).
        label: Resolved human-readable label, or None when unresolved.
        task_id: The opaque/numeric id to demote to a dim suffix.
        id_prefix: Optional prefix for the id (e.g. ``"#"`` for todo ids).

    """
    if label is not None:
        header_line.append(' → "')
        header_line.append(label, style=STYLE_CYAN)
        header_line.append('"')
    if task_id.strip():
        header_line.append(f" ({id_prefix}{task_id})", style=STYLE_DIM)


def _build_tool_header(
    name: str,
    args: dict[str, object],
    quiet_mode: bool = False,
    *,
    label: str | None = None,
) -> Text:
    """Build styled tool header content.

    Used by LiveToolPanel._build_tool_header().

    Args:
        name: Name of the tool being called.
        args: Dictionary of arguments passed to the tool.
        quiet_mode: If True, hide command details for Bash tools.
        label: Resolved human-readable label for background-task tools
            (``TaskOutput``/``TaskStop``); when provided it leads the header
            and the opaque ``task_id`` is demoted to a dim suffix.

    Returns:
        Rich Text containing the styled tool header.

    """
    content = Text()

    # Background-task tools: lead with the resolved label, demote the opaque
    # task_id to a dim suffix, and never surface block/timeout plumbing.
    if name in _BACKGROUND_TASK_TOOLS:
        header_line = Text()
        header_line.append("🎠 ", style=STYLE_ORANGE)
        header_line.append(name, style=STYLE_BOLD_PINK)
        task_id = str(args.get(_task_id_key(name), ""))
        _append_label_and_id(header_line, label, task_id)
        content.append_text(header_line)
        return content

    # Todo-list tools lead with the todo subject and demote the numeric id;
    # TaskUpdate appends its status change. Never surface plumbing.
    if name in _TODO_TASK_TOOLS:
        header_line = Text()
        header_line.append("🎠 ", style=STYLE_ORANGE)
        header_line.append(name, style=STYLE_BOLD_PINK)
        if name == "TaskCreate":
            subject = str(args.get("subject", "")).strip()
            if subject:
                header_line.append("  ")
                header_line.append(subject, style=STYLE_CYAN)
        else:
            task_id = str(args.get(_task_id_key(name), ""))
            _append_label_and_id(header_line, label, task_id, id_prefix="#")
            if name == "TaskUpdate":
                status = str(args.get("status", "")).strip()
                if status:
                    header_line.append(" → ")
                    header_line.append(status, style=STYLE_CYAN)
        content.append_text(header_line)
        return content

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
                content.append("\n  ")
                content.append("args=", style=STYLE_PURPLE)
                content.append(str(skill_args), style=STYLE_ORANGE)

        return content

    if name == "TodoWrite":
        header_line = Text()
        header_line.append("🔧 ", style=STYLE_ORANGE)
        header_line.append("TodoWrite", style=STYLE_BOLD_PINK)
        content.append_text(header_line)

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
        header_line = Text()
        header_line.append("🔨 ", style=STYLE_ORANGE)
        header_line.append("Bash", style=STYLE_BOLD_PINK)
        content.append_text(header_line)

        description = str(args.get("description", ""))
        if description:
            content.append("\n")
            content.append(description, style=STYLE_CYAN)

        if not quiet_mode:
            command = str(args.get("command", ""))
            if command:
                content.append("\n")
                content.append("$ ", style=STYLE_DIM)
                content.append(command, style=STYLE_DIM)

        return content

    if name == "Write":
        file_path = str(args.get("file_path", ""))

        header_line = Text()
        header_line.append("⛏️ ", style=STYLE_ORANGE)
        header_line.append("Write", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Write')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(file_path, style=STYLE_CYAN)

        return content

    if name == "Glob":
        pattern = str(args.get("pattern", ""))
        search_path = str(args.get("path", ""))

        header_line = Text()
        header_line.append("🔮 ", style=STYLE_PURPLE)
        header_line.append("Glob", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Glob')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(pattern, style=STYLE_ORANGE)

        if search_path:
            content.append("\n")
            content.append("path=", style=STYLE_PURPLE)
            content.append(search_path, style=STYLE_CYAN)

        return content

    if name == "Grep":
        pattern = str(args.get("pattern", ""))
        search_path = str(args.get("path", ""))
        glob_filter = str(args.get("glob", ""))
        file_type = str(args.get("type", ""))

        header_line = Text()
        header_line.append("🧙 ", style=STYLE_PURPLE)
        header_line.append("Grep", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Grep')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(pattern, style=STYLE_ORANGE)

        if search_path:
            content.append("\n")
            content.append("path=", style=STYLE_PURPLE)
            content.append(search_path, style=STYLE_CYAN)

        if glob_filter:
            content.append("\n")
            content.append("glob=", style=STYLE_PURPLE)
            content.append(glob_filter, style=STYLE_YELLOW)

        if file_type:
            content.append("\n")
            content.append("type=", style=STYLE_PURPLE)
            content.append(file_type, style=STYLE_YELLOW)

        return content

    if name == "Read":
        file_path = str(args.get("file_path", ""))
        offset = args.get("offset")
        limit = args.get("limit")

        header_line = Text()
        header_line.append("📜 ", style=STYLE_ORANGE)
        header_line.append("Read", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
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

        header_line = Text()
        header_line.append("⚕ ", style=STYLE_CYAN)
        header_line.append("Edit", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Edit')}... ", style=Style(color=NEON_COLORS["purple"], italic=True))
        content.append(file_path, style=STYLE_CYAN)

        if replace_all:
            content.append("\n")
            content.append("replace_all=", style=STYLE_PURPLE)
            content.append("True", style=STYLE_PURPLE)

        content.append("\n")

        content.append("\n")
        content.append("  ⊗ ", style=STYLE_RED)
        content.append("BLOCKED", style=Style(color=NEON_COLORS["red"], bold=True))
        content.append("\n")
        if old_string:
            lines = old_string.split("\n")[:_EDIT_PREVIEW_MAX_LINES]
            preview = "\n".join(lines)
            if len(old_string.split("\n")) > _EDIT_PREVIEW_MAX_LINES:
                preview += "\n..."
            preview_len = max(len(preview) - 1, 1)
            for i, char in enumerate(preview):
                if char == "\n":
                    content.append("\n  ")
                else:
                    t = i / preview_len
                    color = _interpolate_color(NEON_COLORS["red"], NEON_COLORS["orange"], t)
                    content.append(char, style=Style(color=color))

        content.append("\n\n")
        content.append("  ✓ ", style=STYLE_GREEN)
        content.append("FLOWING", style=Style(color=NEON_COLORS["green"], bold=True))
        content.append("\n")
        if new_string:
            lines = new_string.split("\n")[:_EDIT_PREVIEW_MAX_LINES]
            preview = "\n".join(lines)
            if len(new_string.split("\n")) > _EDIT_PREVIEW_MAX_LINES:
                preview += "\n..."
            preview_len = max(len(preview) - 1, 1)
            for i, char in enumerate(preview):
                if char == "\n":
                    content.append("\n  ")
                else:
                    t = i / preview_len
                    color = _interpolate_color(NEON_COLORS["cyan"], NEON_COLORS["green"], t)
                    content.append(char, style=Style(color=color, bold=True))

        return content

    # Task description/prompt render as markdown below the header (see
    # _build_tool_body_extras).
    if name == "Task":
        header_line = Text()
        header_line.append("🎠 ", style=STYLE_ORANGE)
        header_line.append("Task", style=STYLE_BOLD_PINK)
        subagent = str(args.get("subagent_type", ""))
        if subagent:
            header_line.append("  ")
            header_line.append(subagent, style=STYLE_CYAN)
        content.append_text(header_line)
        return content

    header_line = Text()
    header_line.append("🎠 ", style=STYLE_ORANGE)
    header_line.append(name, style=STYLE_BOLD_PINK)
    content.append_text(header_line)

    if args:
        content.append("\n")
        args_text = _colorize_tool_args(args)
        content.append_text(args_text)

    return content


def _build_tool_body_extras(name: str, args: dict[str, object]) -> list:
    """Return extra renderables to display between header and result.

    Used for the Task tool, whose `description` and `prompt` arguments are
    rendered as Markdown, and for TaskCreate, whose `description` renders as the
    Markdown body below its `subject` header — so bold/italic/code/lists display
    properly instead of as a flat key=value dump.
    """
    if name == "TaskCreate":
        description = str(args.get("description", "")).strip()
        return [Markdown(description)] if description else []
    if name != "Task":
        return []
    extras: list = []
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
) -> tuple[Text | Syntax | Group, bool]:
    """Build styled result content with syntax highlighting.

    Used by LiveToolPanel._build_result_content().

    Args:
        content: The result content to display.
        is_error: Whether this is an error result.
        max_lines: Maximum number of lines to display.

    Returns:
        Tuple of (renderable content, was_truncated).

    """
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
            return Group(syntax, truncation_text), truncated
        return syntax, truncated

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

    return result_text, truncated
