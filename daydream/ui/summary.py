"""Summary, table, and deep-review status components.

Fix-progress indicators, the iteration divider, the run summary table, the
issues table, the verdict-join table, the exploration-context summary, the TTT
plan renderer, and the deep-mode stage/verification/preflight notices.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

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
    """Print fix progress indicator.

    Args:
        console: Rich Console instance for output.
        item_num: Current item number (1-indexed).
        total: Total number of items.
        description: Description of the fix.

    Returns:
        None

    """
    text = Text()
    text.append("  ", style=Style())
    text.append(f"[{item_num}/{total}] ", style=STYLE_BOLD_CYAN)
    text.append("Fixing: ", style=STYLE_PINK)
    desc = description[:60] + "..." if len(description) > 60 else description
    text.append(desc, style=STYLE_FG)
    console.print(text)


def print_fix_complete(console: Console, item_num: int, total: int) -> None:
    """Print fix completion indicator.

    Args:
        console: Rich Console instance for output.
        item_num: Current item number (1-indexed).
        total: Total number of items.

    Returns:
        None

    """
    text = Text()
    text.append("  ", style=Style())
    text.append(f"[{item_num}/{total}] ", style=STYLE_BOLD_CYAN)
    text.append("✔ Fix applied", style=STYLE_GREEN)
    console.print(text)


def print_iteration_divider(console: Console, iteration: int, max_iterations: int) -> None:
    """Print an iteration divider for loop mode."""
    console.print()
    label = f" Iteration {iteration} of {max_iterations} "
    console.print(Rule(label, style=STYLE_PURPLE, characters="━"))
    console.print()


@dataclass
class SummaryData:
    """Data class for summary information.

    Attributes:
        skill: Name of the skill that was executed.
        target: Target file or directory that was reviewed.
        feedback_count: Number of issues found during review.
        fixes_applied: Number of fixes that were applied.
        test_retries: Number of times tests were retried.
        tests_passed: Whether all tests passed after fixes.
        loop_mode: If True, loop mode was enabled for iterative fixes.
        iterations_used: Number of loop iterations used (1 if not in loop mode).

    """

    skill: str
    target: str
    feedback_count: int
    fixes_applied: int
    test_retries: int
    tests_passed: bool
    loop_mode: bool = False
    iterations_used: int = 1


def print_summary(console: Console, data: SummaryData) -> None:
    """Print a summary table with neon styling.

    Displays a comprehensive summary of the review/fix session
    with status badges for pass/fail.

    Args:
        console: Rich Console instance for output.
        data: SummaryData containing all summary fields.

    """
    table = Table(
        title="✨ Review Summary",
        title_style=STYLE_BOLD_GREEN,
        box=box.ROUNDED,
        border_style=STYLE_PURPLE,
        show_header=False,
        padding=(0, 1),
    )

    table.add_column("Field", style=STYLE_CYAN)
    table.add_column("Value", style=STYLE_FG)

    table.add_row("Skill", data.skill)
    table.add_row("Target", data.target)
    table.add_row("Issues Found", str(data.feedback_count))

    if data.loop_mode:
        table.add_row("Iterations", str(data.iterations_used))
    table.add_row("Fixes Applied", str(data.fixes_applied))
    table.add_row("Test Retries", str(data.test_retries))

    if data.tests_passed:
        status_badge = pill(" PASSED ", NEON_COLORS["green"], NEON_COLORS["background"])
    else:
        status_badge = pill(" FAILED ", NEON_COLORS["red"], NEON_COLORS["background"])
    table.add_row("Tests", status_badge)

    console.print(table)


def print_issues_table(console: Console, issues: list[dict]) -> None:
    """Display issues as a numbered Rich table.

    Args:
        console: Rich Console instance.
        issues: List of issue dicts with id, title, severity, description,
                recommendation, files keys.

    """
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
        sev = issue.get("severity", "medium")
        table.add_row(
            str(issue.get("id", "?")),
            Text(sev, style=severity_style.get(sev, STYLE_FG)),
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
    structural: list[int | None],
    other: list[int | None],
    total: int,
) -> Table:
    """Build a table summarizing how merged items joined to verifier verdicts.

    One row per category (Matched / Unmatched / Structural / Other) showing the
    count and, dimly, the ids; a final Total row. The Other row is omitted when
    empty. Mirrors the print_summary table style.

    Args:
        matched: Ids of items that matched a verifier verdict.
        unmatched: Ids of verdict-eligible items with no verifier verdict.
        structural: Ids of structural (verdict-exempt) items.
        other: Leftover ids that fit no other bucket.
        total: Total number of items to fix (len(items)).

    Returns:
        A rich Table ready to pass to console.print.

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

    def _ids(values: list[int | None]) -> str:
        return ", ".join(str(v) for v in values)

    n_matched = len(matched)
    n_unmatched = len(unmatched)
    n_structural = len(structural)
    n_other = len(other)
    computed_total = n_matched + n_unmatched + n_structural + n_other

    table.add_row("Matched", str(n_matched), _ids(matched))
    table.add_row("Unmatched", str(n_unmatched), _ids(unmatched))
    table.add_row("Structural", str(n_structural), _ids(structural))
    if other:
        table.add_row("Other", str(n_other), _ids(other))
    mismatch = f"  [expected {total}]" if computed_total != total else ""
    table.add_row("Total", str(computed_total) + mismatch, "")

    return table


_EXPLORATION_LIST_CAP = 8


def render_exploration_summary(ctx: "ExplorationContext") -> "Group | Text":
    """Build a renderable summarising the pre-scan exploration context.

    Replaces the raw structured-output JSON dump with a counts header plus
    compact lists of conventions, dependencies, and affected files. Mirrors
    the field shapes and label wording of ExplorationContext.to_prompt_section.

    Args:
        ctx: Aggregated exploration results (read-only).

    Returns:
        A rich renderable (Table wrapped in a Group, or a plain Text for the
        empty case) ready to pass to console.print.

    """
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


def render_ttt_plan(console: Console, plan: dict) -> None:
    """Render a TTT plan, visually distinguishing ungrounded steps.

    Plan steps with a non-empty ``references`` list render with default style
    and their references inline beneath the change line. Steps with an empty
    ``references`` list render dimmed with an ``(ungrounded)`` marker so the
    user can spot LOW-grounded recommendations at a glance (D-08).

    Args:
        console: Rich console to render into.
        plan: Plan dict. May be the flat shape ``{"changes": [...]}`` or the
            nested shape ``{"plan": {"issues": [{"changes": [...]}, ...]}}``.

    """
    changes: list[dict] = []
    if isinstance(plan.get("changes"), list):
        changes = list(plan["changes"])
    else:
        nested = plan.get("plan", {}) if isinstance(plan.get("plan"), dict) else {}
        for issue in nested.get("issues", []) or []:
            if isinstance(issue, dict):
                changes.extend(issue.get("changes", []) or [])

    for change in changes:
        if not isinstance(change, dict):
            continue
        file_path = change.get("file", "")
        description = change.get("description", "")
        references = change.get("references") or []
        line = f"{file_path}: {description}" if file_path else description

        if references:
            console.print(Text(line))
            for ref in references:
                if not isinstance(ref, dict):
                    continue
                ref_file = ref.get("file", "")
                ref_symbol = ref.get("symbol", "")
                console.print(Text.assemble(("    → ", STYLE_DIM), (f"{ref_file}::{ref_symbol}", STYLE_DIM)))
        else:
            console.print(Text.assemble((line, STYLE_DIM), (" (ungrounded)", "yellow")))


def print_stage_progress(console: Console, current: int, total: int, name: str) -> None:
    """Print a ``[stage N/M: name]`` banner at deep-mode stage boundaries (D-44).

    Args:
        console: Rich Console instance for output.
        current: Current stage number (1-indexed).
        total: Total number of stages.
        name: Human-readable stage name.
    """
    console.print(f"[neon.cyan]▶[/] [neon.fg][stage {current}/{total}: {name}][/]")


def print_verification_summary(console: Console, verdicts_path: Path) -> None:
    """Print a one-line summary of recommendation-verifier verdicts.

    Reads the verdicts JSON written by ``phase_verify_recommendations`` and
    emits a single dim line of the form ``Recommendation verification: N
    findings · M flagged (X contradicts, Y uncertain)``. Missing,
    empty, or malformed files are treated as a no-op so the fix gate is
    never blocked by verifier output.

    Args:
        console: Rich Console instance for output.
        verdicts_path: Path to the ``recommendation-verdicts.json`` file.
    """
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
    print_dim(
        console,
        f"Recommendation verification: {len(verdicts)} findings · {flagged} flagged "
        f"({contradicts} contradicts, {uncertain} uncertain)",
    )


def print_preflight_notice(
    console: Console,
    *,
    stages: list[str],
    stack_lines: list[str],
    agent_count: int,
    exploration_available: bool,
) -> None:
    """Print the deep-mode pre-flight notice (D-30).

    Lists the stages, detected stacks, skill per stack, and total agent
    count. The D-31 ``cost_usd=None`` caveat was removed when #194 reversed
    D-16 (Codex now synthesizes cost at the backend layer); per-model
    unpriceable cases surface via the renderer's "cost unavailable" marker
    (#156) at render time instead.

    Args:
        console: Rich Console instance for output.
        stages: Ordered list of stage display names.
        stack_lines: Per-stack human-readable summary lines.
        agent_count: Total agent invocation count (D-30 formula).
        exploration_available: True when the exploration infrastructure
            (Phases 1-4) is installed and the pre-scan is wired in.
    """
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
