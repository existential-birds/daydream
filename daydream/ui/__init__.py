"""Neon terminal UI components for review_fix_loop.py.

Implements a 1980s neon terminal aesthetic using the Rich library,
with a Dracula-based color theme and animated elements.

This package is a re-exporting facade over focused submodules; callers
continue to ``from daydream.ui import X`` exactly as they did when this was
a single ``ui.py`` module.
"""

from daydream.ui.agent_text import AgentTextRenderer as AgentTextRenderer
from daydream.ui.console import (
    create_console as create_console,
)
from daydream.ui.console import (
    print_phase_hero as print_phase_hero,
)
from daydream.ui.messages import (
    print_cost as print_cost,
)
from daydream.ui.messages import (
    print_dim as print_dim,
)
from daydream.ui.messages import (
    print_error as print_error,
)
from daydream.ui.messages import (
    print_feedback_table as print_feedback_table,
)
from daydream.ui.messages import (
    print_info as print_info,
)
from daydream.ui.messages import (
    print_intent_summary as print_intent_summary,
)
from daydream.ui.messages import (
    print_menu as print_menu,
)
from daydream.ui.messages import (
    print_skipped_phases as print_skipped_phases,
)
from daydream.ui.messages import (
    print_success as print_success,
)
from daydream.ui.messages import (
    print_warning as print_warning,
)
from daydream.ui.messages import (
    prompt_user as prompt_user,
)
from daydream.ui.panels import (
    CrazySpinner as CrazySpinner,
)
from daydream.ui.panels import (
    LiveThinkingPanel as LiveThinkingPanel,
)
from daydream.ui.panels import (
    LiveToolPanel as LiveToolPanel,
)
from daydream.ui.panels import (
    LiveToolPanelRegistry as LiveToolPanelRegistry,
)
from daydream.ui.panels import (
    ShutdownPanel as ShutdownPanel,
)
from daydream.ui.panels import (
    ShutdownStep as ShutdownStep,
)
from daydream.ui.panels import (
    get_shutdown_panel as get_shutdown_panel,
)
from daydream.ui.panels import (
    print_thinking as print_thinking,
)
from daydream.ui.panels import (
    set_shutdown_panel as set_shutdown_panel,
)
from daydream.ui.summary import (
    SummaryData as SummaryData,
)
from daydream.ui.summary import (
    format_verdict_join as format_verdict_join,
)
from daydream.ui.summary import (
    print_fix_complete as print_fix_complete,
)
from daydream.ui.summary import (
    print_fix_progress as print_fix_progress,
)
from daydream.ui.summary import (
    print_issues_table as print_issues_table,
)
from daydream.ui.summary import (
    print_preflight_notice as print_preflight_notice,
)
from daydream.ui.summary import (
    print_stage_progress as print_stage_progress,
)
from daydream.ui.summary import (
    print_summary as print_summary,
)
from daydream.ui.summary import (
    print_verification_summary as print_verification_summary,
)
from daydream.ui.summary import (
    render_exploration_summary as render_exploration_summary,
)
from daydream.ui.theme import (
    ASCII_GRADIENT_COLORS as ASCII_GRADIENT_COLORS,
)
from daydream.ui.theme import (
    GRADIENT_COLORS as GRADIENT_COLORS,
)
from daydream.ui.theme import (
    MYSTICAL_TERMS as MYSTICAL_TERMS,
)
from daydream.ui.theme import (
    NEON_COLORS as NEON_COLORS,
)
from daydream.ui.theme import (
    NEON_THEME as NEON_THEME,
)
from daydream.ui.theme import (
    PHASE_SUBTITLES as PHASE_SUBTITLES,
)
from daydream.ui.theme import (
    STATUS_CONFIG as STATUS_CONFIG,
)
from daydream.ui.theme import (
    STYLE_AGENT_BG as STYLE_AGENT_BG,
)
from daydream.ui.theme import (
    STYLE_BOLD_CYAN as STYLE_BOLD_CYAN,
)
from daydream.ui.theme import (
    STYLE_BOLD_GREEN as STYLE_BOLD_GREEN,
)
from daydream.ui.theme import (
    STYLE_BOLD_PINK as STYLE_BOLD_PINK,
)
from daydream.ui.theme import (
    STYLE_BOLD_PURPLE as STYLE_BOLD_PURPLE,
)
from daydream.ui.theme import (
    STYLE_BOLD_RED as STYLE_BOLD_RED,
)
from daydream.ui.theme import (
    STYLE_BOLD_YELLOW as STYLE_BOLD_YELLOW,
)
from daydream.ui.theme import (
    STYLE_CYAN as STYLE_CYAN,
)
from daydream.ui.theme import (
    STYLE_DIM as STYLE_DIM,
)
from daydream.ui.theme import (
    STYLE_FG as STYLE_FG,
)
from daydream.ui.theme import (
    STYLE_GREEN as STYLE_GREEN,
)
from daydream.ui.theme import (
    STYLE_ORANGE as STYLE_ORANGE,
)
from daydream.ui.theme import (
    STYLE_PANEL_BG as STYLE_PANEL_BG,
)
from daydream.ui.theme import (
    STYLE_PINK as STYLE_PINK,
)
from daydream.ui.theme import (
    STYLE_PURPLE as STYLE_PURPLE,
)
from daydream.ui.theme import (
    STYLE_RED as STYLE_RED,
)
from daydream.ui.theme import (
    STYLE_YELLOW as STYLE_YELLOW,
)
from daydream.ui.theme import (
    SURGERY_CHAKRA_SYMBOLS as SURGERY_CHAKRA_SYMBOLS,
)
from daydream.ui.theme import (
    SURGERY_ENERGY_FLOW as SURGERY_ENERGY_FLOW,
)
from daydream.ui.theme import (
    SURGERY_PHASES as SURGERY_PHASES,
)
from daydream.ui.theme import (
    mystical_term as mystical_term,
)
from daydream.ui.theme import (
    phase_subtitle as phase_subtitle,
)
from daydream.ui.theme import (
    pill as pill,
)
from daydream.ui.tools import (
    format_callback_progress as format_callback_progress,
)
from daydream.ui.tools import (
    format_callback_text as format_callback_text,
)
