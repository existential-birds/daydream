"""Per-attempt presentation of agent events in log, progress, and live modes."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.text import Text

from daydream.backends import (
    AgentEvent,
    CostEvent,
    DiagnosticEvent,
    MetricsEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.redaction import redact_structured_text, redact_value
from daydream.run_context import InteractionPolicy
from daydream.ui.agent_text import AgentTextRenderer
from daydream.ui.messages import print_cost, print_warning
from daydream.ui.panels import LiveToolPanelRegistry, print_thinking
from daydream.ui.tools import (
    _BASH_COMMAND_MAX_CHARS,
    _PRIMARY_TOOL_ARG,
    _redacted_bash_command,
    format_callback_progress,
    format_callback_text,
)


def _summarize_input(input_data: dict[str, Any], name: str) -> str:
    """One-line summary of tool input for log output."""
    if not input_data:
        return ""
    # The COMPLETE selected string is redacted before any [:_BASH_COMMAND_MAX_CHARS]
    # slice — redact-after-slice would truncate a credential into an unmatchable fragment.
    # Key the shared primary table by tool name the way ui.tools._primary_tool_value
    # does instead of hard-applying the Bash-only (command, description) preference:
    # TaskCreate/Agent inputs also carry "description", and letting the Bash pair
    # shadow it would replace their short subject with the long field.
    for key in _PRIMARY_TOOL_ARG.get(name, ()):
        value = input_data.get(key)
        if isinstance(value, str) and value:
            # S1 parity with the live render surfaces (ui.tools): the stored
            # input keeps the replayable cd-prefixed payload, but the --verbose
            # surface shows the cd-stripped display variant. Codex-only
            # ('shell'): Claude/Pi Bash commands never pass through the Codex
            # wrapper, so their operator-authored cd prefix must render.
            if key == "command":
                return _redacted_bash_command(name, value)
            return redact_structured_text(value)[:_BASH_COMMAND_MAX_CHARS]
    if "path" in input_data:
        complete = f"{input_data['path']}" + (
            f" -> {input_data.get('new_path', '')}" if "new_path" in input_data else ""
        )
        return redact_structured_text(complete)
    # Generic: first value that's a string
    for v in input_data.values():
        if isinstance(v, str):
            return redact_structured_text(v)[:_BASH_COMMAND_MAX_CHARS]
    return redact_structured_text(str(input_data))[:_BASH_COMMAND_MAX_CHARS]

def _summarize_output(output: str) -> str:
    """One-line summary of tool output for log output."""
    if not output:
        return "(empty)"
    # Redact the COMPLETE output before strip/first-line/[:200] — a credential
    # straddling the summary boundary must be caught before the slice.
    redacted = redact_structured_text(output)
    # Take first non-empty line or first 200 chars
    first_line = redacted.strip().split("\n")[0]
    return first_line[:200]

def _print_log(value: str) -> None:
    """Redact a complete verbose payload before printing it."""
    print(redact_structured_text(value), flush=True)

class AgentDisplay:
    """Own streamed text and tool panels for one attempt; retries get fresh state."""

    def __init__(
        self, console: Console, policy: InteractionPolicy,
        callback: Callable[[Text], Any] | None, *, structured: bool,
    ) -> None:
        self.console = console
        self.log_mode = policy.log_mode
        self.callback = callback
        self.structured = structured
        self.tools = LiveToolPanelRegistry(console, policy.quiet)
        self.text = AgentTextRenderer(console)
        self.tool_names: dict[str, str] = {}
        self.callback_parts: list[str] = []

    async def _notify(self, text: Text) -> None:
        if self.callback is not None:
            result = self.callback(text)
            if inspect.isawaitable(result):
                await result

    async def flush(self) -> None:
        if self.callback is not None and self.callback_parts:
            text = "".join(self.callback_parts)
            self.callback_parts.clear()
            last_line = text.strip().split("\n")[-1]
            if last_line:
                await self._notify(format_callback_text(last_line))

    def _finish_text(self) -> None:
        if self.text.has_content:
            self.text.finish()

    async def observe(self, event: AgentEvent) -> None:
        if not isinstance(event, (TextEvent, DiagnosticEvent)):
            await self.flush()
        if self.log_mode:
            self._log(event)
        elif self.callback is not None:
            if isinstance(event, TextEvent):
                self.callback_parts.append(event.text)
            elif isinstance(event, ToolStartEvent):
                self.tools.note_call(event.id, event.name, event.input)
                label = self.tools.resolve_call_label(event.name, event.input)
                await self._notify(format_callback_progress(event.name, event.input, label))
            elif isinstance(event, ToolResultEvent):
                self.tools.observe_result(event.id, event.output)
        else:
            self._live(event)

    def _log(self, event: AgentEvent) -> None:
        if isinstance(event, TextEvent):
            _print_log(event.text)
        elif isinstance(event, ThinkingEvent):
            _print_log(f"[thinking] {event.text}")
        elif isinstance(event, ToolStartEvent):
            self.tool_names[event.id] = event.name
            _print_log(f"[tool:{event.name}] {_summarize_input(event.input, event.name)}")
        elif isinstance(event, ToolResultEvent):
            name = self.tool_names.get(event.id, "unknown")
            status = "ERROR" if event.is_error else "result"
            _print_log(f"[tool:{name} {status}] {_summarize_output(event.output)}")
        elif isinstance(event, MetricsEvent):
            _print_log(f"[metrics] prompt={event.prompt_tokens} completion={event.completion_tokens}")
        elif isinstance(event, CostEvent):
            cost = f"${event.cost_usd:.4f}" if event.cost_usd is not None else "unknown"
            _print_log(f"[cost] {cost}")

    def _live(self, event: AgentEvent) -> None:
        if isinstance(event, TextEvent):
            if not self.structured:
                self.text.append(event.text)
        elif isinstance(event, ThinkingEvent):
            self._finish_text()
            print_thinking(self.console, event.text)
        elif isinstance(event, ToolStartEvent):
            self._finish_text()
            self.tools.create(event.id, event.name, event.input)
        elif isinstance(event, ToolResultEvent):
            self.tools.observe_result(event.id, event.output)
            panel = self.tools.get(event.id)
            if panel:
                panel.set_result(event.output, event.is_error)
                self.tools.remove(event.id)
        elif isinstance(event, CostEvent) and event.cost_usd:
            self._finish_text()
            self.console.print()
            print_cost(self.console, event.cost_usd)

    async def _status(self, log: str, callback: str, live: str) -> None:
        if self.log_mode:
            _print_log(log)
        elif self.callback is not None:
            await self._notify(format_callback_text(callback))
        else:
            print_warning(self.console, live)

    async def aborted(self, reason: str) -> None:
        await self._status(f"[aborted] {reason}", f"[budget] aborted: {reason}", f"Turn aborted: {reason}")

    async def retry(self, message: str) -> None:
        await self._status(f"[retry] {message}", f"[retry] {message}", message)

    def finish(self) -> None:
        if self.callback is None and not self.log_mode:
            self._finish_text()
            self.tools.finish_all()
            self.console.print()


def _issue_line(issue: dict[str, Any]) -> str:
    issue_id = issue.get("id", "?")
    if "file" in issue and "line" in issue:
        return f"[{issue_id}] {issue['file']}:{issue['line']} - {issue.get('description', '')}"
    return f"[{issue_id}] {issue.get('title', issue.get('description', ''))}"


def present_result(console: Console, policy: InteractionPolicy, callback: Callable[[Text], Any] | None,
                   value: Any) -> None:
    """Present the host-resolved result after validation and any staged admission."""
    if policy.log_mode:
        _print_log(f"[result] {json.dumps(redact_value(value))[:500]}")
    elif callback is None and isinstance(value, dict):
        issues = value.get("issues", [])
        if issues:
            renderer = AgentTextRenderer(console)
            renderer.append(redact_structured_text("\n".join(_issue_line(issue) for issue in issues)))
            renderer.finish()
