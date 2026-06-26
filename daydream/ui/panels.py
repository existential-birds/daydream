"""Live-updating animated panels.

The thinking panel, the spinner animation, the consolidated tool-call
Live panel and its multi-panel registry, and the shutdown progress panel.
"""

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from daydream.ui.console import _interpolate_color
from daydream.ui.theme import (
    _BACKGROUND_TASK_TOOLS,
    _GLOB_MAX_LINES,
    _LAUNCH_TASK_TOOLS,
    _RESULT_MAX_LINES,
    _TODO_TASK_TOOLS,
    GRADIENT_COLORS,
    NEON_COLORS,
    STATUS_CONFIG,
    STYLE_BOLD_CYAN,
    STYLE_BOLD_GREEN,
    STYLE_BOLD_RED,
    STYLE_CYAN,
    STYLE_DIM,
    STYLE_FG,
    STYLE_PANEL_BG,
    STYLE_PURPLE,
    STYLE_YELLOW,
    SURGERY_CHAKRA_SYMBOLS,
    SURGERY_ENERGY_FLOW,
    SURGERY_PHASES,
)
from daydream.ui.tools import (
    _build_result_content,
    _build_tool_body_extras,
    _build_tool_header,
    _derive_task_label,
    _label_source_name,
    _parse_assigned_task_id,
    _task_id_key,
    _task_label_ns_key,
)


class LiveThinkingPanel:
    """Animated thinking panel with crazy spinners in title.

    Displays the AI's thought process in a purple-styled panel
    with animated spinners alongside the thought bubble icon.

    Args:
        console: Rich Console instance for output.
        content: The thinking content to display.
        max_length: Maximum content length before truncation.

    """

    def __init__(self, console: Console, content: str, max_length: int = 300) -> None:
        """Initialize the panel.

        Args:
            console: Rich Console instance for output.
            content: The thinking content to display.
            max_length: Maximum content length before truncation.

        """
        self._console = console
        self._content = content if len(content) <= max_length else content[:max_length] + "..."
        self._spinner = CrazySpinner(num_spinners=3)

    def __rich__(self) -> Panel:
        """Render panel with animated title.

        Returns:
            Rich Panel with animated spinners in title.

        """
        title = Text()
        title.append("💭 Thinking")
        title.append_text(self._spinner.render())

        return Panel(
            Markdown(self._content),
            title=title,
            title_align="left",
            box=box.ROUNDED,
            border_style=STYLE_PURPLE,
            style=Style(color=NEON_COLORS["purple"], italic=True),
            padding=(0, 1),
        )

    def show(self, duration: float = 1.0) -> None:
        """Show animated panel for duration, then persist final state.

        Args:
            duration: How long to show animation before settling.

        """
        self._console.print()
        with Live(self, console=self._console, refresh_per_second=10, transient=True):
            time.sleep(duration)
        self._console.print(
            Panel(
                Markdown(self._content),
                title="💭 Thinking",
                title_align="left",
                box=box.ROUNDED,
                border_style=STYLE_PURPLE,
                style=Style(color=NEON_COLORS["purple"], italic=True),
                padding=(0, 1),
            )
        )


def print_thinking(console: Console, content: str, max_length: int = 300) -> None:
    """Print a thinking panel with brief animation.

    Displays the AI's thought process in a purple-styled panel
    with animated spinners in the title that settle after a brief duration.

    Args:
        console: Rich Console instance for output.
        content: The thinking content to display.
        max_length: Maximum content length before truncation.

    """
    panel = LiveThinkingPanel(console, content, max_length)
    panel.show(duration=0.5)


class CrazySpinner:
    """Wild multi-pattern spinner with gradient colors.

    Displays multiple spinner characters simultaneously, each with its own
    animation pattern and gradient color cycling. Creates a chaotic but
    visually striking loading indicator.

    """

    SPINNERS = [
        # Braille dots - vertical bounce
        ["⠁", "⠂", "⠄", "⡀", "⡀", "⠄", "⠂", "⠁"],
        # Braille dots - horizontal sweep
        ["⠈", "⠐", "⠠", "⢀", "⢀", "⠠", "⠐", "⠈"],
        # Quarter blocks - rotation
        ["◴", "◷", "◶", "◵"],
        # Arrows - spinning
        ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        # Box drawing - morphing
        ["┤", "┘", "┴", "└", "├", "┌", "┬", "┐"],
        # Stars - twinkling
        ["✶", "✷", "✸", "✹", "✺", "✹", "✸", "✷"],
        # Geometric - pulsing
        ["◯", "◎", "◉", "●", "◉", "◎"],
        # Dice - rolling
        ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"],
    ]

    def __init__(self, num_spinners: int = 3) -> None:
        """Initialize with multiple independent spinner states.

        Args:
            num_spinners: Number of spinner characters to display.

        """
        self._num_spinners = num_spinners
        self._frame = 0
        self._spinner_indices = [i % len(self.SPINNERS) for i in range(num_spinners)]
        self._offsets = [i * 2 for i in range(num_spinners)]

    def render(self) -> Text:
        """Render the current frame of all spinners with gradient colors.

        Returns:
            Rich Text with multiple animated spinner characters.

        """
        result = Text()
        result.append(" ")

        num_colors = len(GRADIENT_COLORS)

        for i in range(self._num_spinners):
            pattern_idx = (self._spinner_indices[i] + self._frame // 8) % len(self.SPINNERS)
            pattern = self.SPINNERS[pattern_idx]

            char_idx = (self._frame + self._offsets[i]) % len(pattern)
            char = pattern[char_idx]

            color_idx = (self._frame + i * 3) % num_colors
            color = GRADIENT_COLORS[color_idx]

            result.append(char, style=Style(color=color, bold=True))

        self._frame += 1

        return result

    def reset(self) -> None:
        """Reset animation to beginning."""
        self._frame = 0


class LiveToolPanel:
    """Manage a single tool call panel with Live updates.

    Consolidates tool call display and result into a single live-updating panel.
    Shows an animated throbber while waiting for the result, then replaces it
    with the actual result content.

    In quiet mode, renders the tool header only and skips result display.

    Args:
        console: Rich Console instance for output.
        tool_use_id: Unique identifier for the tool use.
        name: Name of the tool being called.
        args: Dictionary of arguments passed to the tool.
        quiet_mode: If True, use static display instead of Live updates.

    Usage:
        panel = LiveToolPanel(console, "tool-123", "Bash", {"command": "ls"})
        panel.start()
        # ... tool executes ...
        panel.set_result("file1.txt\nfile2.txt", is_error=False)
        panel.finish()

    """

    def __init__(
        self,
        console: Console,
        tool_use_id: str,
        name: str,
        args: dict[str, object],
        quiet_mode: bool = False,
        label: str | None = None,
    ) -> None:
        """Initialize the LiveToolPanel.

        Args:
            console: Rich Console instance for output.
            tool_use_id: Unique identifier for the tool use.
            name: Name of the tool being called.
            args: Dictionary of arguments passed to the tool.
            quiet_mode: If True, use static display instead of Live updates.
            label: Resolved human label for background-task tools, threaded
                through to the header by the registry.

        """
        self._console = console
        self._tool_use_id = tool_use_id
        self._name = name
        self._args = args
        self._label = label
        self._result: str | None = None
        self._is_error: bool = False
        self._live: Live | None = None
        self._spinner = CrazySpinner(num_spinners=3)
        self._quiet_mode = quiet_mode
        self._frame = 0  # Animation frame counter for Edit surgery visualization

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self._name

    def _build_tool_header_content(self) -> Text:
        """Build the tool call header content.

        Delegates to shared _build_tool_header() helper for consistent styling.

        Returns:
            Rich Text containing the styled tool header.

        """
        return _build_tool_header(self._name, self._args, self._quiet_mode, label=self._label)

    def _build_result_content_internal(self, max_lines: int = _RESULT_MAX_LINES) -> Text | Syntax | Group:
        """Build the result content with syntax highlighting.

        Delegates to shared _build_result_content() helper for consistent styling.
        Special handling for Glob and Grep tools to show file counts and formatted lists.

        Args:
            max_lines: Maximum number of lines to display.

        Returns:
            Rich renderable containing the styled result.

        """
        if self._result is None:
            return Text()
        if not self._result.strip():
            return Text()

        if self._name == "Glob" and not self._is_error:
            return self._build_glob_result(max_lines)

        if self._name == "Grep" and not self._is_error:
            return self._build_grep_result(max_lines)

        if self._name == "Edit" and not self._is_error:
            return self._build_edit_result()

        # Surface the <output> snippet, stripping the XML-ish tag plumbing
        # (retrieval_status, task_id, status, ...).
        if self._name == "TaskOutput" and not self._is_error:
            match = re.search(r"<output>(.*?)</output>", self._result, re.DOTALL)
            snippet = match.group(1).strip() if match else self._result
            result, _ = _build_result_content(snippet, self._is_error, max_lines)
            return result

        result, _ = _build_result_content(self._result, self._is_error, max_lines)
        return result

    def _build_glob_result(self, max_lines: int = _GLOB_MAX_LINES) -> Text:
        """Build formatted Glob result showing file count and paths.

        Args:
            max_lines: Maximum number of files to display.

        Returns:
            Rich Text with formatted file list.

        """
        assert self._result is not None  # Caller ensures this
        lines = [line for line in self._result.strip().split("\n") if line.strip()]
        total_files = len(lines)

        result = Text()

        result.append("✨ ", style=STYLE_YELLOW)
        result.append(f"Found {total_files} file{'s' if total_files != 1 else ''}", style=STYLE_BOLD_CYAN)
        result.append("\n")

        display_lines = lines[:max_lines]
        for i, filepath in enumerate(display_lines):
            filename = filepath.rsplit("/", 1)[-1] if "/" in filepath else filepath
            parent = filepath.rsplit("/", 1)[0] if "/" in filepath else ""

            result.append("  ")
            result.append(filename, style=STYLE_CYAN)
            if parent:
                if len(parent) > 40:
                    parent = "..." + parent[-37:]
                result.append(f"  {parent}", style=STYLE_DIM)
            if i < len(display_lines) - 1:
                result.append("\n")

        if total_files > max_lines:
            result.append("\n")
            result.append(
                f"  ... and {total_files - max_lines} more",
                style=Style(color=NEON_COLORS["yellow"], italic=True),
            )

        return result

    def _build_grep_result(self, max_lines: int = _RESULT_MAX_LINES) -> Text | Syntax | Group:
        """Build formatted Grep result showing match count.

        Args:
            max_lines: Maximum number of lines to display.

        Returns:
            Rich renderable with formatted grep output.

        """
        assert self._result is not None  # Caller ensures this
        lines = [line for line in self._result.strip().split("\n") if line.strip()]
        total_matches = len(lines)

        result = Text()

        result.append("✨ ", style=STYLE_YELLOW)
        result.append(f"Found {total_matches} match{'es' if total_matches != 1 else ''}", style=STYLE_BOLD_CYAN)
        result.append("\n")

        content, _ = _build_result_content(self._result, self._is_error, max_lines)

        return Group(result, content)

    def _build_edit_result(self) -> Text:
        """Build formatted Edit result with surgery completion message.

        Returns:
            Rich Text with harmony restored visualization.

        """
        result = Text()

        result.append("  ")
        result.append("✓ ", style=STYLE_BOLD_GREEN)
        result.append("HARMONY RESTORED", style=Style(color=NEON_COLORS["green"], bold=True))
        result.append("\n\n")

        flow_width = 30
        for i in range(flow_width):
            t = i / flow_width
            color = _interpolate_color(NEON_COLORS["cyan"], NEON_COLORS["green"], t)
            char = SURGERY_ENERGY_FLOW[i % len(SURGERY_ENERGY_FLOW)]
            result.append(char, style=Style(color=color))

        result.append("\n\n")

        if self._result and self._result.strip():
            result.append("  ", style=STYLE_DIM)
            result.append(self._result.strip(), style=STYLE_DIM)

        return result

    def _build_surgery_phase_indicator(self) -> Text:
        """Build animated surgery phase indicator for Edit tool.

        Returns:
            Rich Text with animated energy flow and phase status.

        """
        self._frame += 1

        result = Text()

        # At 10fps refresh, 4 frames per phase = 0.4s per phase, full cycle in 1.6s.
        phase_duration = 4
        phase_idx = (self._frame // phase_duration) % len(SURGERY_PHASES)
        phase_name, phase_color_key = SURGERY_PHASES[phase_idx]
        phase_color = NEON_COLORS[phase_color_key]

        energy_idx = self._frame % len(SURGERY_ENERGY_FLOW)
        energy_char = SURGERY_ENERGY_FLOW[energy_idx]

        chakra_idx = (self._frame // 2) % len(SURGERY_CHAKRA_SYMBOLS)
        chakra_char = SURGERY_CHAKRA_SYMBOLS[chakra_idx]

        result.append("\n\n  ")
        result.append(energy_char, style=Style(color=phase_color))
        result.append(" ", style=Style(color=phase_color))
        result.append(chakra_char, style=Style(color=phase_color, bold=True))
        result.append(f" {phase_name} ", style=Style(color=phase_color, bold=True))
        result.append(chakra_char, style=Style(color=phase_color, bold=True))
        result.append(" ", style=Style(color=phase_color))
        result.append(energy_char, style=Style(color=phase_color))

        return result

    def _render_panel(self) -> Panel:
        """Render the current state as a Panel.

        Shows tool call header + either throbber (if waiting) or result.

        Returns:
            Rich Panel containing the consolidated tool call display.

        """
        header = self._build_tool_header_content()

        if self._name == "Skill":
            border_color = NEON_COLORS["cyan"]
        elif self._is_error:
            border_color = NEON_COLORS["red"]
        elif self._name == "Edit" and self._result is None:
            # At 10fps, modulo 10 = 1 second full pulse cycle.
            pulse = abs((self._frame % 10) - 5) / 5
            border_color = _interpolate_color(NEON_COLORS["purple"], NEON_COLORS["cyan"], pulse)
        else:
            border_color = NEON_COLORS["purple"]

        body_extras = _build_tool_body_extras(self._name, self._args)

        if self._result is None:
            if self._name == "Edit":
                surgery_indicator = self._build_surgery_phase_indicator()
                header_with_surgery = Text()
                header_with_surgery.append_text(header)
                header_with_surgery.append_text(surgery_indicator)
                header_with_surgery.append_text(self._spinner.render())
                content = Group(header_with_surgery)
            else:
                header_with_spinner = Text()
                header_with_spinner.append_text(header)
                header_with_spinner.append_text(self._spinner.render())
                content = Group(header_with_spinner, *body_extras)
        elif self._name == "Skill":
            # Skill calls skip the output section; the header already shows the skill name.
            content = Group(header)
        elif self._quiet_mode:
            content = Group(header, *body_extras)
        else:
            result_content = self._build_result_content_internal()

            if self._is_error:
                result_title = Text()
                result_title.append("❌ Error", style=STYLE_BOLD_RED)
            else:
                result_title = Text()
                result_title.append("Output", style=STYLE_BOLD_CYAN)

            if isinstance(result_content, Text) and not result_content.plain.strip():
                # Wrap the lone header in Group for return-type consistency.
                content = Group(header, *body_extras)
            else:
                content = Group(
                    header,
                    *body_extras,
                    result_title,
                    result_content,
                )

        return Panel(
            content,
            box=box.ROUNDED,
            border_style=Style(color=border_color),
            style=STYLE_PANEL_BG,
            padding=(0, 1),
        )

    def __rich__(self) -> Panel:
        """Return renderable for Rich Live refresh."""
        return self._render_panel()

    def start(self) -> None:
        """Start Live context and show tool call with animated throbber."""
        self._console.print()

        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def set_result(self, content: str, is_error: bool = False) -> None:
        """Store result and update the display.

        Args:
            content: The result content from the tool.
            is_error: Whether this is an error result.

        """
        self._result = content
        self._is_error = is_error

        if self._live is not None:
            self._live.update(self._render_panel())

    def finish(self) -> None:
        """Stop Live context. Final panel state persists on screen."""
        if self._live is not None:
            self._live.stop()
            self._live = None


class _ActivePanelsGroup:
    """Renderable that dynamically renders all active panels in a registry.

    Used as the target for a single shared ``Live`` instance so that
    multiple concurrent tool panels animate without conflicting.
    """

    def __init__(self, registry: "LiveToolPanelRegistry") -> None:
        self._registry = registry

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Yield each active panel's rendered output."""
        for panel in self._registry.iter_active_panels():
            yield panel._render_panel()


class LiveToolPanelRegistry:
    """Registry for tracking multiple concurrent tool panels.

    Manages a **single** ``Rich.Live`` instance that renders all active
    panels together, preventing the display corruption that occurs when
    multiple ``Live`` contexts compete for the terminal (as happens with
    the Codex backend's parallel command execution).

    When a panel is finalized (result received), the shared ``Live`` is
    stopped, the finished panel is printed statically, and the ``Live``
    is restarted for any remaining active panels.

    Args:
        console: Rich Console instance for creating panels.
        quiet_mode: If True, panels use static display instead of Live updates.

    Usage:
        registry = LiveToolPanelRegistry(console, quiet_mode=False)
        panel = registry.create("tool-123", "Bash", {"command": "ls"})
        # ... tool executes ...
        panel.set_result("file1.txt", is_error=False)
        panel.finish()
        registry.remove("tool-123")

    """

    def __init__(self, console: Console, quiet_mode: bool = False) -> None:
        """Initialize the registry.

        Args:
            console: Rich Console instance for creating panels.
            quiet_mode: If True, panels use static display instead of Live updates.

        """
        self._console = console
        self._quiet_mode = quiet_mode
        self._panels: dict[str, LiveToolPanel] = {}
        self._active_order: list[str] = []
        self._static_ids: set[str] = set()
        self._live: Live | None = None
        self._group = _ActivePanelsGroup(self)
        self._task_labels: dict[str, str] = {}  # keyed by _task_label_ns_key(name, id)
        # Originating-call args by tool_use_id, populated on every ToolStartEvent so
        # result→label correlation works even when no panel was created (callback path).
        self._call_args: dict[str, tuple[str, dict[str, object]]] = {}

    # Tool names whose panels should scroll inline rather than pin via Live.
    _STATIC_TOOL_NAMES: frozenset[str] = frozenset({"Task"})

    # Tool names whose result strings assign an id we can label.
    _LABEL_SOURCE_TOOL_NAMES: frozenset[str] = _LAUNCH_TASK_TOOLS | frozenset({"TaskCreate"})

    def note_call(self, tool_use_id: str, name: str, args: dict[str, object]) -> None:
        """Record an originating call's name + args for later label correlation.

        Called on **every** ``ToolStartEvent`` (both the Live-panel and the
        callback render paths) so that ``observe_result`` can resolve a
        background ``task_id``'s label without a panel having been created.

        Args:
            tool_use_id: The id of the tool call.
            name: The tool's name.
            args: The tool's input args.

        """
        self._call_args[tool_use_id] = (name, args)

    def observe_result(self, tool_use_id: str, output: str) -> None:
        """Harvest a ``task_id → label`` mapping from an originating tool's result.

        When a backgrounded launch tool (``Bash``/``Agent``/``Task``) or a
        ``TaskCreate`` call returns, its result string assigns an id. This
        records that id mapped to a human-readable label derived from the
        originating call's input args. Reads the originating call from the
        ``note_call`` store, so it works in both render modes (panel creation
        is not a prerequisite).

        Args:
            tool_use_id: The id of the originating tool call.
            output: The originating tool's result string.

        """
        call = self._call_args.pop(tool_use_id, None)
        if call is None or call[0] not in self._LABEL_SOURCE_TOOL_NAMES:
            return
        name, args = call
        task_id = _parse_assigned_task_id(name, output)
        if task_id is None:
            return
        self._task_labels[_task_label_ns_key(name, task_id)] = _derive_task_label(args, task_id)

    def resolve_label(self, name: str, task_id: str) -> str | None:
        """Return the harvested label for a task id, or None if unknown.

        Args:
            name: The originating tool's name (used to select the correct
                namespace prefix — launch tools vs. TaskCreate).
            task_id: The assigned task id to look up.

        Returns:
            The mapped label, or None if no mapping was recorded.

        """
        return self._task_labels.get(_task_label_ns_key(name, task_id))

    def resolve_call_label(self, name: str, args: dict[str, object]) -> str | None:
        """Resolve a Task-family call's human label from its args.

        Background-task tools (``TaskOutput``/``TaskStop``) key off ``task_id``;
        todo-list tools (``TaskGet``/``TaskUpdate``/…) key off ``taskId``. Returns
        the harvested label for that id, or None when no mapping was recorded (the
        caller falls back to a non-opaque rendering).

        Args:
            name: The tool's name.
            args: The tool's input args.

        Returns:
            The resolved label, or None if the tool is not Task-family or the id
            is unknown.

        """
        if name in _BACKGROUND_TASK_TOOLS or name in _TODO_TASK_TOOLS:
            task_id = str(args.get(_task_id_key(name), ""))
            return self.resolve_label(_label_source_name(name), task_id)
        return None

    def create(
        self,
        tool_use_id: str,
        name: str,
        args: dict[str, object],
    ) -> LiveToolPanel:
        """Create and register a new panel.

        The panel is added to the shared ``Live`` context which renders
        all active panels together.  Individual panels do **not** own
        their own ``Live`` instance.

        Args:
            tool_use_id: Unique identifier for the tool use.
            name: Name of the tool being called.
            args: Dictionary of arguments passed to the tool.

        Returns:
            The created LiveToolPanel.

        """
        if tool_use_id in self._panels:
            self._finalize_panel(tool_use_id)

        # Store the originating call's args for later result→label correlation.
        self.note_call(tool_use_id, name, args)

        # Resolve a human label so the panel header leads with it instead of the
        # opaque/numeric id.
        label = self.resolve_call_label(name, args)

        panel = LiveToolPanel(
            console=self._console,
            tool_use_id=tool_use_id,
            name=name,
            args=args,
            quiet_mode=self._quiet_mode,
            label=label,
        )
        self._panels[tool_use_id] = panel

        # Static tools (e.g. Task) scroll inline instead of joining Live.
        if name in self._STATIC_TOOL_NAMES:
            self._static_ids.add(tool_use_id)
            self._stop_live()
            self._console.print()
            self._console.print(panel._render_panel())
            self._ensure_live()
            return panel

        self._active_order.append(tool_use_id)

        if self._live is None:
            self._console.print()

        self._ensure_live()
        return panel

    def get(self, tool_use_id: str) -> LiveToolPanel | None:
        """Get a panel by its tool_use_id.

        Args:
            tool_use_id: The unique identifier of the tool use.

        Returns:
            The LiveToolPanel if found, None otherwise.

        """
        return self._panels.get(tool_use_id)

    def iter_active_panels(self) -> Iterator[LiveToolPanel]:
        """Iterate over active panels in order.

        Yields:
            LiveToolPanel instances in the order they were created.

        """
        for tid in self._active_order:
            panel = self._panels.get(tid)
            if panel:
                yield panel

    def remove(self, tool_use_id: str) -> None:
        """Remove a panel from the registry and print its final state.

        Stops the shared ``Live``, prints the finalized panel statically,
        then restarts ``Live`` for any remaining active panels.

        Args:
            tool_use_id: The unique identifier of the tool use to remove.

        Returns:
            None

        """
        self._finalize_panel(tool_use_id)

    def finish_all(self) -> None:
        """Finalize all remaining panels.

        Stops the shared ``Live`` and prints each panel's current state
        statically.  Use this for cleanup when a response ends
        unexpectedly.

        Returns:
            None

        """
        self._stop_live()
        for tid in list(self._active_order):
            panel = self._panels.get(tid)
            if panel:
                self._console.print(panel._render_panel())
        self._active_order.clear()
        self._panels.clear()
        self._call_args.clear()
        self._task_labels.clear()

    def discard_all(self) -> None:
        """Stop rendering and clear tracked panels without printing them."""
        self._stop_live()
        self._active_order.clear()
        self._panels.clear()
        self._call_args.clear()
        self._task_labels.clear()

    def _ensure_live(self) -> None:
        """Start the shared ``Live`` if there are active panels."""
        if self._active_order:
            if self._live is None:
                self._live = Live(
                    self._group,
                    console=self._console,
                    refresh_per_second=10,
                    transient=True,
                )
                self._live.start()
        else:
            self._stop_live()

    def _stop_live(self) -> None:
        """Stop the shared ``Live`` instance (if running)."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _finalize_panel(self, tool_use_id: str) -> None:
        """Remove a panel from the active set and print its final state."""
        # Prune the originating-call record; by finalization time observe_result
        # has already harvested any task_id → label mapping from it.
        self._call_args.pop(tool_use_id, None)

        if tool_use_id in self._static_ids:
            self._static_ids.discard(tool_use_id)
            panel = self._panels.pop(tool_use_id, None)
            if panel is not None:
                self._stop_live()
                self._console.print(panel._render_panel())
                self._ensure_live()
            return

        if tool_use_id not in self._active_order:
            self._panels.pop(tool_use_id, None)
            return

        panel = self._panels.pop(tool_use_id, None)
        self._active_order.remove(tool_use_id)

        # Stop Live so the finalized panel can be printed statically.
        self._stop_live()

        if panel:
            self._console.print(panel._render_panel())

        self._ensure_live()


@dataclass
class ShutdownStep:
    """A step in the shutdown process.

    Attributes:
        message: Description of the shutdown step.
        status: Current status of the step ("pending", "in_progress", or "completed").

    """

    message: str
    status: str = "pending"  # pending, in_progress, completed


class ShutdownPanel:
    """Live-updating panel for showing shutdown progress.

    Consolidates all shutdown messages into a single panel that
    updates in place, showing the progression of shutdown steps.

    Args:
        console: Rich Console instance for output.

    Usage:
        panel = ShutdownPanel(console)
        panel.start("Received SIGINT, shutting down")
        panel.add_step("Terminating running agent...")
        panel.complete_step(0)  # Mark first step as completed
        panel.finish()

    """

    def __init__(self, console: Console) -> None:
        """Initialize the ShutdownPanel.

        Args:
            console: Rich Console instance for output.

        """
        self._console = console
        self._steps: list[ShutdownStep] = []
        self._live: Live | None = None

    def _render_panel(self) -> Panel:
        """Render the current state as a Panel.

        Returns:
            Rich Panel containing all shutdown steps with status icons.

        """
        content = Text()

        for i, step in enumerate(self._steps):
            config = STATUS_CONFIG.get(step.status, STATUS_CONFIG["pending"])
            icon = config["icon"]
            color = config["color"]

            content.append(f"{icon} ", style=Style(color=color))
            content.append(step.message, style=STYLE_FG)

            if i < len(self._steps) - 1:
                content.append("\n")

        return Panel(
            content,
            box=box.ROUNDED,
            border_style=STYLE_YELLOW,
            padding=(0, 1),
        )

    def start(self, initial_message: str) -> None:
        """Start the live panel with an initial message.

        Args:
            initial_message: The first message to display (e.g., "Received SIGINT").

        Returns:
            None

        """
        self._steps.append(ShutdownStep(message=initial_message, status="completed"))

        self._console.print()
        self._live = Live(
            self._render_panel(),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    def add_step(self, message: str, status: str = "in_progress") -> int:
        """Add a new step to the shutdown sequence.

        Args:
            message: The step message to display.
            status: Initial status ("pending", "in_progress", "completed").

        Returns:
            Index of the added step for later updates.

        """
        self._steps.append(ShutdownStep(message=message, status=status))
        if self._live is not None:
            self._live.update(self._render_panel())
        return len(self._steps) - 1

    def complete_step(self, index: int) -> None:
        """Mark a step as completed.

        Args:
            index: Index of the step to mark as completed.

        Returns:
            None

        """
        if 0 <= index < len(self._steps):
            self._steps[index].status = "completed"
            if self._live is not None:
                self._live.update(self._render_panel())

    def complete_last_step(self) -> None:
        """Mark the last step as completed.

        Returns:
            None

        """
        if self._steps:
            self.complete_step(len(self._steps) - 1)

    def finish(self) -> None:
        """Stop the live context and print the final panel.

        Call this when all shutdown steps are complete.

        Returns:
            None

        """
        if self._live is not None:
            self._live.stop()
            self._live = None

        if self._steps:
            self._console.print(self._render_panel())


# Global shutdown panel instance for cross-module access
_shutdown_panel: ShutdownPanel | None = None


def get_shutdown_panel() -> ShutdownPanel | None:
    """Get the current shutdown panel instance.

    Returns:
        ShutdownPanel | None: The current shutdown panel, or None if not set.

    """
    return _shutdown_panel


def set_shutdown_panel(panel: ShutdownPanel | None) -> None:
    """Set the global shutdown panel instance.

    Args:
        panel: The ShutdownPanel instance to set, or None to clear.

    Returns:
        None

    """
    global _shutdown_panel
    _shutdown_panel = panel
