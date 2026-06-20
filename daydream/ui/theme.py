"""Theme primitives for the neon terminal UI.

Dracula-based color theme, reusable Style constants, size caps, tool-arg
classification frozensets, mystical action terms, phase subtitles, gradient
palettes, and the status configuration — plus the ``pill`` primitive shared
across UI clusters.
"""

import random

from rich.style import Style
from rich.text import Text
from rich.theme import Theme

NEON_COLORS = {
    "background": "#282A36",
    "foreground": "#F8F8F2",
    "red": "#FF5555",
    "green": "#50FA7B",
    "yellow": "#F1FA8C",
    "purple": "#BD93F9",
    "pink": "#FF79C6",
    "cyan": "#8BE9FD",
    "orange": "#FFB86C",
}

NEON_THEME = Theme({
    "neon.bg": NEON_COLORS["background"],
    "neon.fg": NEON_COLORS["foreground"],
    "neon.red": NEON_COLORS["red"],
    "neon.green": NEON_COLORS["green"],
    "neon.yellow": NEON_COLORS["yellow"],
    "neon.purple": NEON_COLORS["purple"],
    "neon.pink": NEON_COLORS["pink"],
    "neon.cyan": NEON_COLORS["cyan"],
    "neon.orange": NEON_COLORS["orange"],
    "neon.error": f"bold {NEON_COLORS['red']}",
    "neon.success": f"bold {NEON_COLORS['green']}",
    "neon.warning": f"bold {NEON_COLORS['yellow']}",
    "neon.info": NEON_COLORS["cyan"],
    "neon.dim": f"dim {NEON_COLORS['foreground']}",
    "neon.tool": f"bold {NEON_COLORS['pink']}",
    "neon.path": NEON_COLORS["cyan"],
    "neon.number": NEON_COLORS["yellow"],
    "neon.string": NEON_COLORS["orange"],
})

# Pre-defined Style objects for Text.append() and other Rich components that
# require Style objects rather than theme strings.
STYLE_CYAN = Style(color=NEON_COLORS["cyan"])
STYLE_PURPLE = Style(color=NEON_COLORS["purple"])
STYLE_PINK = Style(color=NEON_COLORS["pink"])
STYLE_GREEN = Style(color=NEON_COLORS["green"])
STYLE_YELLOW = Style(color=NEON_COLORS["yellow"])
STYLE_ORANGE = Style(color=NEON_COLORS["orange"])
STYLE_RED = Style(color=NEON_COLORS["red"])
STYLE_FG = Style(color=NEON_COLORS["foreground"])

STYLE_BOLD_PINK = Style(color=NEON_COLORS["pink"], bold=True)
STYLE_BOLD_CYAN = Style(color=NEON_COLORS["cyan"], bold=True)
STYLE_BOLD_PURPLE = Style(color=NEON_COLORS["purple"], bold=True)
STYLE_BOLD_GREEN = Style(color=NEON_COLORS["green"], bold=True)
STYLE_BOLD_YELLOW = Style(color=NEON_COLORS["yellow"], bold=True)
STYLE_BOLD_RED = Style(color=NEON_COLORS["red"], bold=True)

STYLE_PANEL_BG = Style(bgcolor="#1e1e2e")
STYLE_AGENT_BG = Style(bgcolor="#051208")

STYLE_DIM = Style(dim=True)

# Truncation limits shared across the tool renderers. Named so the scattered
# magic line-counts/char-caps have one source of meaning instead of bare literals.
_TASK_PROMPT_MAX_LINES = 10
_RESULT_MAX_LINES = 20
_EDIT_PREVIEW_MAX_LINES = 30
_GLOB_MAX_LINES = 15

# Mechanical/plumbing tool args dropped from the generic key=value fallback so the
# line leads with meaningful keys instead of noise.
_MECHANICAL_TOOL_ARGS = frozenset({"block", "timeout"})

# Background-task tools that reference an opaque background ``task_id``; rendered
# with the resolved label leading and the id demoted to a dim suffix.
_BACKGROUND_TASK_TOOLS = frozenset({"TaskOutput", "TaskStop"})

# Todo-list tools that reference a small-integer ``taskId``; the human label is
# the todo's ``subject`` (in ``TaskCreate``'s input; later tools resolve by id).
_TODO_TASK_TOOLS = frozenset({"TaskCreate", "TaskGet", "TaskUpdate", "TaskList"})

# Launch tools whose background-task result strings carry the assigned task id
# in the ``Command running in background with ID: <id>`` prose pattern.
_LAUNCH_TASK_TOOLS = frozenset({"Bash", "Agent", "Task"})

MYSTICAL_TERMS = {
    "Glob": ["scrying", "divining", "seeking", "wandering"],
    "Grep": ["channeling", "attuning", "resonating", "listening"],
    "Read": ["beholding", "absorbing", "dreaming into", "perceiving"],
    "Edit": ["healing", "realigning", "restoring flow to", "mending"],
    "Write": ["inscribing", "etching", "conjuring", "forging"],
    "Skill": ["invoking", "channeling", "summoning", "awakening"],
}

SURGERY_CHAKRA_SYMBOLS = ["◉", "◎", "●", "○", "◐", "◑", "◒"]
SURGERY_ENERGY_FLOW = ["~", "≈", "∿", "〜", "⌇", "⌁", "⚡"]
SURGERY_PHASES = [
    ("LOCATING MERIDIAN", "yellow"),
    ("CLEARING BLOCKAGE", "red"),
    ("CHANNELING FLOW", "purple"),
    ("HARMONY RESTORED", "green"),
]


def mystical_term(tool: str) -> str:
    """Return a random mystical action term for the given tool."""
    return random.choice(MYSTICAL_TERMS.get(tool, ["processing"]))


PHASE_SUBTITLES = {
    "DAYDREAM": [
        "the mind wanders, the code improves",
        "where code dreams itself awake",
        "the quiet work of becoming",
        "drift into clarity",
    ],
    "REFLECT": [
        "gathering scattered thoughts",
        "seeing what surfaces",
        "listening to the echoes",
        "patterns emerge from stillness",
    ],
    "HEAL": [
        "mending what was found",
        "gentle corrections",
        "restoring harmony",
        "the wounds close softly",
    ],
    "AWAKEN": [
        "returning to waking life",
        "does the dream hold?",
        "testing the vision",
        "reality reasserts itself",
    ],
    "LISTEN": [
        "hearing what was spoken",
        "the echoes return",
        "voices from the waking world",
        "attending to the murmurs",
    ],
    "WONDER": [
        "imagining what could be",
        "seeing with different eyes",
        "questioning the obvious",
        "the road not taken",
    ],
    "ARBITRATE": [
        "weighing the contested",
        "a second, sharper look",
        "settling the divergence",
        "the deciding voice",
    ],
    "ENVISION": [
        "shaping the path forward",
        "drawing the map",
        "from thought to intention",
        "the blueprint emerges",
    ],
}


def phase_subtitle(phase: str) -> str:
    """Return a random dreamlike subtitle for the given phase."""
    return random.choice(PHASE_SUBTITLES.get(phase, ["..."]))


GRADIENT_COLORS = [
    "#881177",
    "#aa3355",
    "#cc6666",
    "#ee9944",
    "#eedd00",
    "#99dd55",
    "#44dd88",
    "#22ccbb",
    "#00bbcc",
    "#0099cc",
    "#3366bb",
    "#663399",
]

STATUS_CONFIG = {
    "pending": {"icon": "⏲", "color": NEON_COLORS["yellow"]},
    "in_progress": {"icon": "⮕", "color": NEON_COLORS["cyan"]},
    "completed": {"icon": "✔", "color": NEON_COLORS["green"]},
    "failed": {"icon": "✗", "color": NEON_COLORS["red"]},
}

# Gradient colors for ASCII art header (cyan -> pink -> purple)
ASCII_GRADIENT_COLORS = [
    "#8BE9FD",  # Cyan (Dracula)
    "#7DDFFA",
    "#6FD5F7",
    "#61CBF4",
    "#53C1F1",
    "#45B7EE",
    "#5AABE5",
    "#6F9FDC",
    "#8493D3",
    "#9987CA",
    "#AE7BC1",
    "#C36FB8",
    "#D863AF",
    "#ED57A6",
    "#FF79C6",  # Pink (Dracula)
    "#F07DC1",
    "#E181BC",
    "#D285B7",
    "#C389B2",
    "#B48DAD",
    "#A591A8",
    "#9695A3",
    "#87999E",
    "#A08FAB",
    "#B985B8",
    "#BD93F9",  # Purple (Dracula)
]


def pill(text: str, bg_color: str, fg_color: str) -> Text:
    """Create a pill-shaped badge with the given colors.

    Uses the pattern: [bg]foreground[/bg] to create a badge effect
    with half-block characters on the edges.

    Args:
        text: The text to display inside the pill.
        bg_color: Background color hex code.
        fg_color: Foreground (text) color hex code.

    Returns:
        Rich Text object containing the styled pill.

    """
    result = Text()
    bg_style = Style(color=bg_color)
    result.append("▌", style=bg_style)
    result.append(text, style=Style(color=fg_color, bgcolor=bg_color, bold=True))
    result.append("▐", style=bg_style)
    return result
