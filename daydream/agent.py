"""Agent interaction and backend management."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from daydream.backends import (
    Backend,
    ContinuationToken,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.config import UNKNOWN_SKILL_PATTERN
from daydream.ui import (
    AgentTextRenderer,
    LiveToolPanelRegistry,
    create_console,
    print_cost,
    print_thinking,
)


class MissingSkillError(Exception):
    """Raised when a required skill is not available.

    Args:
        skill_name: The name of the skill that was not found.

    """

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        super().__init__(f"Skill '{skill_name}' is not available")


@dataclass
class AgentState:
    """Consolidated state for agent module.

    Attributes:
        debug_log: File handle for debug logging, or None to disable.
        quiet_mode: True to hide tool calls and results, False to show them.
        model: Model name to use for agent interactions.
        shutdown_requested: True if shutdown has been requested.
        current_backends: List of active backend instances.

    """

    debug_log: TextIO | None = None
    quiet_mode: bool = False
    model: str = "opus"
    shutdown_requested: bool = False
    current_backends: list[Backend] = field(default_factory=list)


# Module-level Singletons
# =======================
# This module uses a singleton pattern for global state management. The module
# is imported once, creating these instances which persist for the process lifetime.
# Access and modify state through the getter/setter functions below (get_state,
# set_debug_log, set_quiet_mode, etc.) rather than accessing _state directly.
# Use reset_state() to restore defaults between test runs or CLI invocations.

_state = AgentState()
console = create_console()


def get_state() -> AgentState:
    """Get the global agent state singleton.

    Returns:
        The global AgentState instance.

    """
    return _state


def reset_state() -> None:
    """Reset the global agent state to defaults.

    Creates a new AgentState instance with default values.

    Returns:
        None

    """
    global _state
    _state = AgentState()


def set_debug_log(log_file: TextIO | None) -> None:
    """Set the debug log file handle.

    Args:
        log_file: File handle for debug logging, or None to disable.

    Returns:
        None

    """
    _state.debug_log = log_file


def get_debug_log() -> TextIO | None:
    """Get the current debug log file handle.

    Returns:
        The current debug log file handle, or None if not set.

    """
    return _state.debug_log


def set_quiet_mode(quiet: bool) -> None:
    """Set quiet mode for agent output.

    Args:
        quiet: True to hide tool calls and results, False to show them.

    Returns:
        None

    """
    _state.quiet_mode = quiet


def get_quiet_mode() -> bool:
    """Get current quiet mode setting.

    Returns:
        True if quiet mode is enabled, False otherwise.

    """
    return _state.quiet_mode


def set_model(model: str) -> None:
    """Set the model to use for agent interactions.

    Args:
        model: Model name ("opus", "sonnet", or "haiku").

    Returns:
        None

    """
    _state.model = model


def get_model() -> str:
    """Get the current model setting.

    Returns:
        The current model name.

    """
    return _state.model


def set_shutdown_requested(requested: bool) -> None:
    """Set shutdown requested flag.

    Args:
        requested: True to indicate shutdown has been requested, False otherwise.

    Returns:
        None

    """
    _state.shutdown_requested = requested


def get_shutdown_requested() -> bool:
    """Get shutdown requested flag.

    Returns:
        True if shutdown has been requested, False otherwise.

    """
    return _state.shutdown_requested


def get_current_backends() -> list[Backend]:
    """Get all currently running backends.

    Returns:
        List of active backend instances.

    """
    return list(_state.current_backends)


def _log_debug(message: str) -> None:
    """Write a message to the debug log if enabled."""
    if _state.debug_log is not None:
        _state.debug_log.write(message)
        _state.debug_log.flush()


def detect_test_success(output: str) -> bool:
    """Detect if tests passed using pattern matching.

    Uses regex patterns to handle negation (e.g., "no failures" should not
    trigger failure detection).

    Args:
        output: Agent output containing test results

    Returns:
        True if tests clearly passed, False otherwise

    """
    output_lower = output.lower()

    # Strong success patterns (explicit pass statements)
    success_patterns = [
        r"all \d+ tests? passed",
        r"tests? passed successfully",
        r"test suite passed",
        r"\d+ passed,? 0 failed",
        r"0 failed,? \d+ passed",
        r"passed:? \d+.*failed:? 0",
        r"no (?:test )?failures",
        r"0 failures",
        r"all tests pass",
    ]

    # Check for strong success patterns
    for pattern in success_patterns:
        if re.search(pattern, output_lower):
            return True

    # Failure patterns that indicate actual failures (not negated)
    failure_patterns = [
        r"(?<!no )(?<!0 )(?<!\d )failed",
        r"\d+ failed",
        r"tests? failing",
        r"test failure",
        r"assertion error",
        r"traceback",
    ]

    # Check for failure patterns
    for pattern in failure_patterns:
        if pattern == r"\d+ failed":
            match = re.search(r"(\d+) failed", output_lower)
            if match and int(match.group(1)) > 0:
                return False
        elif re.search(pattern, output_lower):
            return False

    return "passed" in output_lower


async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    continuation: ContinuationToken | None = None,
) -> tuple[str | Any, ContinuationToken | None]:
    """Run agent with the given prompt and return output plus continuation token.

    Streams verbose output to stdout as it's received. When progress_callback
    is provided, runs in quiet mode and routes status updates through the
    callback instead of printing to the console.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the agent.
        prompt: The prompt to send to the agent.
        output_schema: Optional JSON schema for structured output.
        progress_callback: Optional callback for status updates (quiet mode).
        continuation: Optional continuation token for multi-turn.

    Returns:
        Tuple of (output, continuation_token). Output is text or structured data.

    Raises:
        MissingSkillError: If a required skill is not available.

    """
    _log_debug(f"\n{'='*80}\n")
    _log_debug(f"[PROMPT] cwd={cwd}\n{prompt}\n")
    _log_debug(f"{'='*80}\n\n")

    output_parts: list[str] = []
    structured_result: Any = None
    result_continuation: ContinuationToken | None = None
    use_callback = progress_callback is not None

    if output_schema is not None:
        _log_debug(f"[SCHEMA] {output_schema}\n")

    _state.current_backends.append(backend)
    try:
        if not use_callback:
            agent_renderer = AgentTextRenderer(console)
            tool_registry = LiveToolPanelRegistry(console, _state.quiet_mode)

        try:
            event_iter = backend.execute(cwd, prompt, output_schema, continuation)
        except Exception as exc:
            _log_debug(f"[EXECUTE_INIT_ERROR] {type(exc).__name__}: {exc}\n")
            raise

        async for event in event_iter:
            if isinstance(event, TextEvent):
                output_parts.append(event.text)
                _log_debug(f"[TEXT] {event.text}\n")

                # Check for missing skill error
                skill_match = re.search(UNKNOWN_SKILL_PATTERN, event.text)
                if skill_match:
                    if not use_callback:
                        agent_renderer.finish()
                        tool_registry.finish_all()
                    raise MissingSkillError(skill_match.group(1))

                if use_callback and progress_callback is not None:
                    last_line = event.text.strip().split("\n")[-1]
                    if last_line:
                        progress_callback(last_line)
                else:
                    agent_renderer.append(event.text)

            elif isinstance(event, ThinkingEvent):
                _log_debug(f"[THINKING] {event.text}\n")
                if not use_callback:
                    if agent_renderer.has_content:
                        agent_renderer.finish()
                    print_thinking(console, event.text)

            elif isinstance(event, ToolStartEvent):
                _log_debug(
                    f"[TOOL_USE] {event.name}({event.input}) "
                    f"id={event.id} use_callback={use_callback}\n"
                )
                if progress_callback is not None:
                    first_arg = next(iter(event.input.values()), "") if event.input else ""
                    progress_callback(f"{event.name} {first_arg}"[:80])
                else:
                    if agent_renderer.has_content:
                        agent_renderer.finish()
                    tool_registry.create(event.id, event.name, event.input)

            elif isinstance(event, ToolResultEvent):
                _log_debug(
                    f"[TOOL_RESULT{' [ERROR]' if event.is_error else ''}] "
                    f"id={event.id} output_len={len(event.output)} "
                    f"output_preview={event.output[:200]!r} "
                    f"use_callback={use_callback}\n"
                )
                if not use_callback:
                    panel = tool_registry.get(event.id)
                    _log_debug(
                        f"[TOOL_RESULT_PANEL] id={event.id} "
                        f"panel_found={panel is not None} "
                        f"panel_name={panel.name if panel else 'N/A'}\n"
                    )
                    if panel:
                        panel.set_result(event.output, event.is_error)
                        panel.finish()
                        tool_registry.remove(event.id)
                    else:
                        _log_debug(f"[WARN] No panel for tool_use_id={event.id}\n")

            elif isinstance(event, CostEvent):
                if event.cost_usd:
                    _log_debug(f"[COST] ${event.cost_usd:.4f}\n")
                    if not use_callback:
                        if agent_renderer.has_content:
                            agent_renderer.finish()
                        console.print()
                        print_cost(console, event.cost_usd)
                elif event.input_tokens is not None:
                    _log_debug(f"[TOKENS] in={event.input_tokens} out={event.output_tokens}\n")

            elif isinstance(event, ResultEvent):
                if event.structured_output is not None:
                    structured_result = event.structured_output
                    _log_debug(f"[STRUCTURED_OUTPUT] {structured_result}\n")
                    if not use_callback:
                        issues = structured_result.get("issues", []) if isinstance(structured_result, dict) else []
                        if issues:
                            formatted = []
                            for i in issues:
                                if "file" in i and "line" in i:
                                    desc = i.get("description", "")
                                    formatted.append(
                                        f"[{i['id']}] {i['file']}:{i['line']} - {desc}"
                                    )
                                else:
                                    label = i.get("title", i.get("description", ""))
                                    formatted.append(f"[{i.get('id', '?')}] {label}")
                            agent_renderer.append("\n".join(formatted))
                result_continuation = event.continuation

        if not use_callback:
            if agent_renderer.has_content:
                agent_renderer.finish()
            tool_registry.finish_all()
            console.print()
    except Exception as exc:
        _log_debug(f"[EXECUTE_ERROR] {type(exc).__name__}: {exc}\n")
        raise
    finally:
        _state.current_backends.remove(backend)

    if output_schema is not None and structured_result is not None:
        _log_debug(f"[SCHEMA_OK] structured_result={structured_result!r}\n")
        return structured_result, result_continuation
    if output_schema is not None:
        raw = "".join(output_parts)
        _log_debug(
            f"[SCHEMA_MISS] output_schema set but structured_result is None; "
            f"raw text ({len(raw)} chars): {raw[:500]!r}\n"
        )
        # Fallback: try to JSON-parse the raw text when structured output failed
        if raw.strip():
            try:
                parsed = json.loads(raw)
                _log_debug(f"[SCHEMA_FALLBACK] parsed raw text as JSON: {parsed!r}\n")
                return parsed, result_continuation
            except (json.JSONDecodeError, ValueError):
                _log_debug("[SCHEMA_FALLBACK] raw text is not valid JSON\n")
    return "".join(output_parts), result_continuation
