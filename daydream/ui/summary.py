"""Fix progress, issue/verdict/exploration summaries, and deep-stage notices."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group
from rich.style import Style
from rich.table import Table
from rich.text import Text

from daydream.severity import normalize_severity
from daydream.ui.messages import print_dim
from daydream.ui.theme import (
    NEON_COLORS,
    STYLE_BOLD_CYAN,
    STYLE_BOLD_GREEN,
    STYLE_BOLD_PINK,
    STYLE_CYAN,
    STYLE_DIM,
    STYLE_FG,
    STYLE_GREEN,
    STYLE_ORANGE,
    STYLE_PINK,
    STYLE_PURPLE,
    STYLE_RED,
    STYLE_YELLOW,
    pill,
)

if TYPE_CHECKING:
    from daydream.exploration import ExplorationContext


def print_fix_progress(
    console: Console, item_num: int, total: int, description: str
) -> None:
    """Print progress for the 1-based item number."""
    text = Text()
    text.append("  ", style=Style())
    text.append(f"[{item_num}/{total}] ", style=STYLE_BOLD_CYAN)
    text.append("Fixing: ", style=STYLE_PINK)
    desc = description[:60] + "..." if len(description) > 60 else description
    text.append(desc, style=STYLE_FG)
    console.print(text)


def print_fix_complete(
    console: Console,
    item_num: int,
    total: int,
    outcome: str | None = None,
) -> None:
    """Only a resolved verifier outcome means applied; other outcomes remain attempted, not fixed."""
    text = Text()
    text.append("  ", style=Style())
    text.append(f"[{item_num}/{total}] ", style=STYLE_BOLD_CYAN)
    if outcome == "resolved":
        text.append("✔ Fix applied", style=STYLE_GREEN)
    elif outcome is None:
        text.append("Fix attempted", style=STYLE_BOLD_CYAN)
    else:
        text.append("Attempted, not fixed", style=STYLE_ORANGE)
    console.print(text)


def print_issues_table(console: Console, issues: list[dict[str, Any]]) -> None:
    """Render an issue table followed by each issue’s details."""
    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style=STYLE_PURPLE,
        show_header=True,
        header_style=STYLE_BOLD_CYAN,
    )
    table.add_column("#", style=STYLE_YELLOW, width=4)
    table.add_column("Severity", style=STYLE_ORANGE, width=8)
    table.add_column("Issue", style=STYLE_FG)

    severity_style = {"high": STYLE_RED, "medium": STYLE_YELLOW, "low": STYLE_GREEN}

    for issue in issues:
        sev = normalize_severity(issue.get("severity"))
        table.add_row(
            str(issue.get("id", "?")),
            Text(sev or "--", style=severity_style.get(sev or "", STYLE_FG)),
            issue.get("title", issue.get("description", "No title")),
        )

    console.print()
    console.print(table)

    for issue in issues:
        console.print()
        issue_id = issue.get("id", "?")
        title = issue.get("title", "No title")
        console.print(Text(f"  #{issue_id}: {title}", style=STYLE_BOLD_PINK))
        if "description" in issue:
            console.print(Text(f"  {issue['description']}", style=STYLE_FG))
        if "recommendation" in issue:
            console.print(Text(f"  Recommendation: {issue['recommendation']}", style=STYLE_CYAN))
        if "files" in issue and issue["files"]:
            files_str = ", ".join(issue["files"])
            console.print(Text(f"  Files: {files_str}", style=STYLE_DIM))


def format_verdict_join(
    *,
    matched: list[int | None],
    unmatched: list[int | None],
    skipped: list[int | None],
    structural: list[int | None],
    other: list[int | None],
    total: int,
) -> Table:
    """Show category ids/counts and an independent total. Matched/Structural always appear;
    empty other categories are omitted. Skipped findings differ from missing unmatched verdicts.
    """
    table = Table(
        title="Verdict Join",
        title_style=STYLE_BOLD_GREEN,
        box=box.ROUNDED,
        border_style=STYLE_PURPLE,
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Category", style=STYLE_CYAN)
    table.add_column("Count", style=STYLE_FG)
    table.add_column("IDs", style=STYLE_DIM)

    categories = (
        ("Matched", matched),
        ("Unmatched", unmatched),
        ("Skipped", skipped),
        ("Structural", structural),
        ("Other", other),
    )
    computed_total = sum(len(values) for _, values in categories)
    for label, values in categories:
        if values or label in {"Matched", "Structural"}:
            table.add_row(label, str(len(values)), ", ".join(str(value) for value in values))
    mismatch = f"  [expected {total}]" if computed_total != total else ""
    table.add_row("Total", str(computed_total) + mismatch, "")

    return table


_EXPLORATION_LIST_CAP = 8


def render_exploration_summary(ctx: "ExplorationContext") -> "Group | Text":
    """Show context counts and capped lists; render a quiet message when empty."""
    if not (ctx.affected_files or ctx.conventions or ctx.dependencies or ctx.guidelines):
        return Text("Exploration: no codebase context gathered", style=STYLE_DIM)

    def _count(n: int, singular: str, plural: str) -> str:
        return f"{n} {singular}" if n == 1 else f"{n} {plural}"

    parts: list[str] = []
    if ctx.affected_files:
        parts.append(_count(len(ctx.affected_files), "file", "files"))
    if ctx.conventions:
        parts.append(_count(len(ctx.conventions), "convention", "conventions"))
    if ctx.dependencies:
        parts.append(_count(len(ctx.dependencies), "dependency", "dependencies"))
    if ctx.guidelines:
        parts.append(_count(len(ctx.guidelines), "guideline", "guidelines"))

    table = Table(
        title="🔍 Exploration",
        title_style=STYLE_BOLD_CYAN,
        box=box.ROUNDED,
        border_style=STYLE_PURPLE,
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Section", style=STYLE_CYAN, no_wrap=True)
    table.add_column("Detail", style=STYLE_FG)

    table.add_row("", pill(f" {' · '.join(parts)} ", NEON_COLORS["purple"], NEON_COLORS["background"]))

    def _add_items(label: str, items: list[str]) -> None:
        shown = items[:_EXPLORATION_LIST_CAP]
        body = Text()
        for i, line in enumerate(shown):
            if i:
                body.append("\n")
            body.append(line, style=STYLE_FG)
        extra = len(items) - len(shown)
        if extra > 0:
            body.append(f"\n+{extra} more", style=STYLE_DIM)
        table.add_row(Text(label, style=STYLE_BOLD_PINK), body)

    if ctx.conventions:
        _add_items(
            "Conventions",
            [f"{c.name} — {c.description}" + (f" ({c.source})" if c.source else "") for c in ctx.conventions],
        )
    if ctx.dependencies:
        _add_items(
            "Dependencies",
            [f"{d.source} {d.relationship} {d.target}" for d in ctx.dependencies],
        )
    if ctx.affected_files:
        _add_items(
            "Affected Files",
            [f"{f.path} ({f.role})" for f in ctx.affected_files],
        )
    if ctx.guidelines:
        _add_items("Project Guidelines", list(ctx.guidelines))

    return Group(Text(""), table)


def print_stage_progress(console: Console, current: int, total: int, name: str) -> None:
    """Print the 1-based stage boundary banner."""
    console.print(f"[neon.cyan]▶[/] [neon.fg][stage {current}/{total}: {name}][/]")


def print_verification_summary(console: Console, verdicts_path: Path) -> None:
    """Summarize verifier flags/counts; missing or malformed artifacts cannot block fixing."""
    try:
        data = json.loads(verdicts_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    verdicts = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(verdicts, list):
        return
    contradicts = sum(1 for v in verdicts if isinstance(v, dict) and v.get("verdict") == "contradicts")
    uncertain = sum(1 for v in verdicts if isinstance(v, dict) and v.get("verdict") == "uncertain")
    flagged = contradicts + uncertain
    selection = data.get("selection") if isinstance(data, dict) else None
    selected = selection.get("selected") if isinstance(selection, dict) else None
    skipped = selection.get("skipped") if isinstance(selection, dict) else None
    suffix = ""
    if _is_count(selected) and _is_count(skipped):
        suffix = f" · {selected} selected / {skipped} skipped"
    print_dim(
        console,
        f"Recommendation verification: {len(verdicts)} findings · {flagged} flagged "
        f"({contradicts} contradicts, {uncertain} uncertain){suffix}",
    )


def _is_count(value: object) -> bool:
    """Whether *value* is a real integer count (``bool`` is not)."""
    return isinstance(value, int) and not isinstance(value, bool)


def print_preflight_notice(
    console: Console,
    *,
    stages: list[str],
    stack_lines: list[str],
    agent_count: int,
    exploration_available: bool,
) -> None:
    """Show selected stages, stack skills, and total agent count before review."""
    console.print("[neon.cyan]▶[/] [neon.fg]Deep-review pipeline pre-flight[/]")
    if exploration_available:
        console.print("Exploration pre-scan: enabled (runs before stage 1)", style="dim")
    else:
        console.print(
            "Exploration pre-scan: unavailable (deep pipeline runs without grounding)",
            style="dim yellow",
        )
    console.print("[neon.fg]  Stages:[/]")
    for idx, stage in enumerate(stages, start=1):
        console.print(f"    {idx}. {stage}")
    console.print("[neon.fg]  Detected stacks:[/]")
    for line in stack_lines:
        console.print(f"    - {line}")
    console.print(f"[neon.fg]  Total agents: {agent_count}[/]")
