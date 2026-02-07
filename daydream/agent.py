"""Agent interaction and SDK client management."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
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

    """

    debug_log: TextIO | None = None
    quiet_mode: bool = False
    model: str = "opus"
    shutdown_requested: bool = False


# Module-level Singletons
# =======================
# This module uses a singleton pattern for global state management. The module
# is imported once, creating these instances which persist for the process lifetime.
# Access and modify state through the getter/setter functions below (get_state,
# set_debug_log, set_quiet_mode, etc.) rather than accessing _state directly.
# Use reset_state() to restore defaults between test runs or CLI invocations.

_state = AgentState()
_current_clients: list[ClaudeSDKClient] = []
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


def get_current_clients() -> list[ClaudeSDKClient]:
    """Get all currently running clients.

    Returns:
        List of active ClaudeSDKClient instances.

    """
    return list(_current_clients)


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
    cwd: Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> str | Any:
    """Run agent with the given prompt and return full text output.

    Streams verbose output to stdout as it's received. When progress_callback
    is provided, runs in quiet mode and routes status updates through the
    callback instead of printing to the console.

    Args:
        cwd: Working directory for the agent
        prompt: The prompt to send to the agent
        output_schema: Optional JSON schema for structured output
        progress_callback: Optional callback for status updates (enables quiet mode)

    Returns:
        The full text output from the agent session, or structured data if
        output_schema is provided

    Raises:
        MissingSkillError: If a required skill is not available
        SystemExit: If the agent is cancelled due to script termination

    """
    _log_debug(f"\n{'='*80}\n")
    _log_debug(f"[PROMPT] cwd={cwd}\n{prompt}\n")
    _log_debug(f"{'='*80}\n\n")

    output_format = (
        {"type": "json_schema", "schema": output_schema}
        if output_schema
        else None
    )

    options = ClaudeAgentOptions(
        cwd=str(cwd),
        permission_mode="bypassPermissions",
        setting_sources=["user", "project", "local"],
        model=_state.model,
        output_format=output_format,
    )

    output_parts: list[str] = []
    structured_result: Any = None
    use_callback = progress_callback is not None

    async with ClaudeSDKClient(options=options) as client:
        _current_clients.append(client)
        try:
            if not use_callback:
                agent_renderer = AgentTextRenderer(console)
                tool_registry = LiveToolPanelRegistry(console, _state.quiet_mode)

            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            output_parts.append(block.text)
                            _log_debug(f"[TEXT] {block.text}\n")
                            skill_match = re.search(UNKNOWN_SKILL_PATTERN, block.text)
                            if skill_match:
                                if not use_callback:
                                    agent_renderer.finish()
                                    tool_registry.finish_all()
                                raise MissingSkillError(skill_match.group(1))
                            if use_callback and progress_callback is not None:
                                # Send last non-empty line to callback
                                last_line = block.text.strip().split("\n")[-1]
                                if last_line:
                                    progress_callback(last_line)
                            else:
                                agent_renderer.append(block.text)
                        elif isinstance(block, ThinkingBlock) and block.thinking:
                            _log_debug(f"[THINKING] {block.thinking}\n")
                            if not use_callback:
                                if agent_renderer.has_content:
                                    agent_renderer.finish()
                                print_thinking(console, block.thinking)
                        elif isinstance(block, ToolUseBlock):
                            _log_debug(f"[TOOL_USE] {block.name}({block.input})\n")
                            if block.name == "StructuredOutput":
                                continue
                            if use_callback and progress_callback is not None:
                                first_arg = next(iter(block.input.values()), "") if block.input else ""
                                progress_callback(f"{block.name} {first_arg}"[:80])
                            else:
                                if agent_renderer.has_content:
                                    agent_renderer.finish()
                                tool_registry.create(block.id, block.name, block.input or {})

                elif isinstance(msg, UserMessage):
                    for user_block in msg.content:
                        if isinstance(user_block, ToolResultBlock):
                            content_str = str(user_block.content) if user_block.content else ""
                            error_marker = " [ERROR]" if user_block.is_error else ""
                            _log_debug(f"[TOOL_RESULT{error_marker}] {content_str}\n")
                            if not use_callback:
                                panel = tool_registry.get(user_block.tool_use_id)
                                if panel:
                                    panel.set_result(content_str, user_block.is_error or False)
                                    panel.finish()
                                    tool_registry.remove(user_block.tool_use_id)
                                else:
                                    _log_debug(f"[WARN] No panel for tool_use_id={user_block.tool_use_id}\n")

                elif isinstance(msg, ResultMessage):
                    if msg.structured_output is not None:
                        structured_result = msg.structured_output
                        _log_debug(f"[STRUCTURED_OUTPUT] {structured_result}\n")
                        if not use_callback:
                            issues = structured_result.get("issues", [])
                            if issues:
                                lines = [f"[{i['id']}] {i['file']}:{i['line']} - {i['description']}" for i in issues]
                                agent_renderer.append("\n".join(lines))
                    if msg.total_cost_usd:
                        _log_debug(f"[COST] ${msg.total_cost_usd:.4f}\n")
                        if not use_callback:
                            if agent_renderer.has_content:
                                agent_renderer.finish()
                            console.print()
                            print_cost(console, msg.total_cost_usd)

            if not use_callback:
                if agent_renderer.has_content:
                    agent_renderer.finish()
                tool_registry.finish_all()
                console.print()
        finally:
            _current_clients.remove(client)

    if output_schema is not None and structured_result is not None:
        return structured_result
    return "".join(output_parts)
