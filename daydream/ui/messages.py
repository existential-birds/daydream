"""Simple message and prompt components.

The error/warning/success/cost/info/skipped/dim print helpers and the
selection menu. Interactive input is owned by ``RunContext.choice`` in
``daydream.run_context``; the raw reader ``_read_user_input`` below is that
gateway's stdin-only leaf.
"""

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from daydream.ui.theme import (
    NEON_COLORS,
    STYLE_BOLD_CYAN,
    STYLE_CYAN,
    STYLE_FG,
    STYLE_PINK,
    STYLE_RED,
    STYLE_YELLOW,
)


def print_error(console: Console, title: str, message: str) -> None:
    """Print an error in a red double-border panel."""
    panel = Panel(
        Text(message, style=STYLE_RED),
        title=f"⚠️  {title}",
        title_align="left",
        box=box.DOUBLE_EDGE,
        border_style=STYLE_RED,
        padding=(0, 1),
    )
    console.print(panel)


def print_warning(console: Console, message: str) -> None:
    """Print a warning panel with yellow styling."""
    panel = Panel(
        Text(message, style=STYLE_YELLOW),
        box=box.ROUNDED,
        border_style=STYLE_YELLOW,
        padding=(0, 1),
    )
    console.print(panel)


def print_success(console: Console, message: str) -> None:
    """Print a success message with green styling."""
    console.print(f"[neon.success]✔[/] [neon.green]{message}[/]")


def print_cost(console: Console, cost_usd: float) -> None:
    """Print a cost indicator with cyan styling."""
    console.print(f"[neon.cyan]💰[/] [neon.dim]${cost_usd:.4f}[/]")


def print_info(console: Console, message: str) -> None:
    """Print an info message with cyan styling."""
    console.print(f"[neon.cyan]ℹ[/] [neon.fg]{message}[/]")


def print_dim(console: Console, message: str) -> None:
    """Print a dimmed message for secondary information."""
    console.print(f"[neon.dim]{message}[/]")


def print_menu(console: Console, title: str, options: list[tuple[str, str]]) -> None:
    """Print (key, description) choices in a selection panel."""
    menu_text = Text()
    for key, description in options:
        menu_text.append(f"  [{key}] ", style=STYLE_BOLD_CYAN)
        menu_text.append(f"{description}\n", style=STYLE_FG)

    panel = Panel(
        menu_text,
        title=title,
        title_align="left",
        box=box.ROUNDED,
        border_style=STYLE_PINK,
        padding=(0, 1),
    )
    console.print(panel)


def print_intent_summary(console: Console, text: str) -> None:
    """Show the full intent at the confirmation gate, or an empty-summary placeholder."""
    body: Markdown | Text
    if text.strip():
        body = Markdown(text)
    else:
        body = Text("(the agent produced no intent summary)", style=STYLE_YELLOW)
    panel = Panel(
        body,
        title="🎧 Understanding",
        title_align="left",
        box=box.ROUNDED,
        border_style=STYLE_CYAN,
        padding=(0, 1),
    )
    console.print(panel)


def _read_user_input(console: Console, message: str, default: str) -> str:
    """Read stdin without policy lookup; use the default for empty input or EOF."""
    prompt_text = Text()
    prompt_text.append("▶ ", style=STYLE_CYAN)
    prompt_text.append(message, style=STYLE_CYAN)
    if default:
        prompt_text.append(f" [{default}]", style=Style(color=NEON_COLORS["foreground"], dim=True))
    prompt_text.append(": ", style=STYLE_CYAN)

    console.print(prompt_text, end="")
    try:
        user_input = input()
    except EOFError:
        console.print(
            f"[prompt] stdin closed (EOF) — using default: {default!r}",
            style=STYLE_YELLOW,
        )
        return default
    return user_input if user_input else default
