"""Neon terminal UI components for review_fix_loop.py.

Implements a 1980s neon terminal aesthetic using the Rich library,
with a Dracula-based color theme and animated elements.

This package is a re-exporting facade over focused submodules; callers
continue to ``from daydream.ui import X`` exactly as they did when this was
a single ``ui.py`` module.
"""

from daydream.ui.agent_text import AgentTextRenderer
from daydream.ui.console import (
    create_console,
    print_phase_hero,
)
from daydream.ui.messages import (
    print_cost,
    print_dim,
    print_error,
    print_info,
    print_intent_summary,
    print_menu,
    print_success,
    print_warning,
    prompt_user,
)
from daydream.ui.panels import (
    LiveToolPanelRegistry,
    ShutdownPanel,
    get_shutdown_panel,
    print_thinking,
    set_shutdown_panel,
)
from daydream.ui.summary import (
    format_verdict_join,
    print_fix_complete,
    print_fix_progress,
    print_issues_table,
    print_preflight_notice,
    print_stage_progress,
    print_verification_summary,
    render_exploration_summary,
)
from daydream.ui.theme import (
    NEON_THEME,
    PHASE_SUBTITLES,
    phase_subtitle,
)
from daydream.ui.tools import (
    format_callback_progress,
    format_callback_text,
)

__all__ = [
    "AgentTextRenderer",
    "LiveToolPanelRegistry",
    "NEON_THEME",
    "PHASE_SUBTITLES",
    "ShutdownPanel",
    "create_console",
    "format_callback_progress",
    "format_callback_text",
    "format_verdict_join",
    "get_shutdown_panel",
    "phase_subtitle",
    "print_cost",
    "print_dim",
    "print_error",
    "print_fix_complete",
    "print_fix_progress",
    "print_info",
    "print_intent_summary",
    "print_issues_table",
    "print_menu",
    "print_phase_hero",
    "print_preflight_notice",
    "print_stage_progress",
    "print_success",
    "print_thinking",
    "print_verification_summary",
    "print_warning",
    "prompt_user",
    "render_exploration_summary",
    "set_shutdown_panel",
]
