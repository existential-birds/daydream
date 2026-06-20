"""Neon terminal UI components for review_fix_loop.py.

Implements a 1980s neon terminal aesthetic using the Rich library,
with a Dracula-based color theme and animated elements.

This package is a re-exporting facade over focused submodules; callers
continue to ``from daydream.ui import X`` exactly as they did when this was
a single ``ui.py`` module.
"""

from daydream.ui.agent_text import (
    AgentTextRenderer,
)
from daydream.ui.console import (
    create_console,
    print_phase_hero,
)
from daydream.ui.messages import (
    print_cost,
    print_dim,
    print_error,
    print_feedback_table,
    print_info,
    print_menu,
    print_skipped_phases,
    print_success,
    print_warning,
    prompt_user,
)
from daydream.ui.panels import (
    CrazySpinner,
    LiveThinkingPanel,
    LiveToolPanel,
    LiveToolPanelRegistry,
    ShutdownPanel,
    ShutdownStep,
    get_shutdown_panel,
    print_thinking,
    set_shutdown_panel,
)
from daydream.ui.summary import (
    SummaryData,
    format_verdict_join,
    print_fix_complete,
    print_fix_progress,
    print_issues_table,
    print_iteration_divider,
    print_preflight_notice,
    print_stage_progress,
    print_summary,
    print_verification_summary,
    render_exploration_summary,
    render_ttt_plan,
)
from daydream.ui.theme import (
    ASCII_GRADIENT_COLORS,
    GRADIENT_COLORS,
    MYSTICAL_TERMS,
    NEON_COLORS,
    NEON_THEME,
    PHASE_SUBTITLES,
    STATUS_CONFIG,
    STYLE_AGENT_BG,
    STYLE_BOLD_CYAN,
    STYLE_BOLD_GREEN,
    STYLE_BOLD_PINK,
    STYLE_BOLD_PURPLE,
    STYLE_BOLD_RED,
    STYLE_BOLD_YELLOW,
    STYLE_CYAN,
    STYLE_DIM,
    STYLE_FG,
    STYLE_GREEN,
    STYLE_ORANGE,
    STYLE_PANEL_BG,
    STYLE_PINK,
    STYLE_PURPLE,
    STYLE_RED,
    STYLE_YELLOW,
    SURGERY_CHAKRA_SYMBOLS,
    SURGERY_ENERGY_FLOW,
    SURGERY_PHASES,
    mystical_term,
    phase_subtitle,
    pill,
)
from daydream.ui.tools import (
    format_callback_progress,
)
