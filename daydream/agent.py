"""Agent interaction and SDK client management."""

import re
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


# Module-level singleton state
_state = AgentState()

# Track the currently running client for cleanup on termination
_current_client: ClaudeSDKClient | None = None

# Global console instance
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


async def query_llm_simple(
    cwd: Path,
    prompt: str,
    model: str | None = None,
    label: str | None = None,
) -> str:
    """Simple LLM query returning text response only.

    Unlike run_agent(), this function:
    - Does not render Rich UI output
    - Does not support structured output
    - Returns only text content
    - Logs all fields needed for Rich UI upgrade

    Args:
        cwd: Working directory for the agent.
        prompt: The prompt to send to the agent.
        model: Optional model override (defaults to global model setting).
        label: Optional label for logging (e.g., "root", "sub-query").

    Returns:
        The text response from the agent.
    """
    global _current_client

    log_prefix = f"[{label}] " if label else ""
    effective_model = model if model is not None else _state.model

    _log_debug(f"\n{'='*80}\n")
    _log_debug(f"{log_prefix}[QUERY_LLM_SIMPLE] model={effective_model} cwd={cwd}\n")
    _log_debug(f"{log_prefix}[PROMPT]\n{prompt}\n")
    _log_debug(f"{'='*80}\n\n")

    options = ClaudeAgentOptions(
        cwd=str(cwd),
        permission_mode="bypassPermissions",
        model=effective_model,
    )

    output_parts: list[str] = []

    async with ClaudeSDKClient(options=options) as client:
        _current_client = client
        try:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            output_parts.append(block.text)
                            _log_debug(f"{log_prefix}[TEXT] {block.text}\n")
                        elif isinstance(block, ThinkingBlock) and block.thinking:
                            _log_debug(f"{log_prefix}[THINKING] {block.thinking}\n")
                        elif isinstance(block, ToolUseBlock):
                            _log_debug(f"{log_prefix}[TOOL_USE] {block.name}({block.input})\n")

                elif isinstance(msg, UserMessage):
                    for user_block in msg.content:
                        if isinstance(user_block, ToolResultBlock):
                            content_str = str(user_block.content) if user_block.content else ""
                            error_marker = " [ERROR]" if user_block.is_error else ""
                            _log_debug(f"{log_prefix}[TOOL_RESULT{error_marker}] {content_str}\n")

                elif isinstance(msg, ResultMessage):
                    if msg.total_cost_usd:
                        _log_debug(f"{log_prefix}[COST] ${msg.total_cost_usd:.4f}\n")
        finally:
            _current_client = None

    response = "".join(output_parts)
    _log_debug(f"{log_prefix}[RESPONSE_LENGTH] {len(response)} chars\n")

    return response


def get_current_client() -> ClaudeSDKClient | None:
    """Get the currently running client.

    Returns:
        The currently running ClaudeSDKClient instance, or None if no client is active.

    """
    return _current_client


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
    model: str | None = None,
) -> str | Any:
    """Run agent with the given prompt and return full text output.

    Streams verbose output to stdout as it's received.

    Args:
        cwd: Working directory for the agent
        prompt: The prompt to send to the agent
        output_schema: Optional JSON schema for structured output
        model: Optional model override (defaults to global model setting)

    Returns:
        The full text output from the agent session, or structured data if
        output_schema is provided

    Raises:
        MissingSkillError: If a required skill is not available
        SystemExit: If the agent is cancelled due to script termination

    """
    global _current_client

    _log_debug(f"\n{'='*80}\n")
    _log_debug(f"[PROMPT] cwd={cwd}\n{prompt}\n")
    _log_debug(f"{'='*80}\n\n")

    output_format = (
        {"type": "json_schema", "schema": output_schema}
        if output_schema
        else None
    )

    effective_model = model if model is not None else _state.model
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        permission_mode="bypassPermissions",
        setting_sources=["user", "project", "local"],
        model=effective_model,
        output_format=output_format,
    )

    output_parts: list[str] = []
    structured_result: Any = None

    async with ClaudeSDKClient(options=options) as client:
        _current_client = client
        try:
            agent_renderer = AgentTextRenderer(console)
            tool_registry = LiveToolPanelRegistry(console, _state.quiet_mode)

            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            agent_renderer.append(block.text)
                            output_parts.append(block.text)
                            _log_debug(f"[TEXT] {block.text}\n")
                            skill_match = re.search(UNKNOWN_SKILL_PATTERN, block.text)
                            if skill_match:
                                agent_renderer.finish()
                                tool_registry.finish_all()
                                raise MissingSkillError(skill_match.group(1))
                        elif isinstance(block, ThinkingBlock) and block.thinking:
                            if agent_renderer.has_content:
                                agent_renderer.finish()
                            print_thinking(console, block.thinking)
                            _log_debug(f"[THINKING] {block.thinking}\n")
                        elif isinstance(block, ToolUseBlock):
                            if block.name == "StructuredOutput":
                                # Skip panel - handled in ResultMessage
                                _log_debug(f"[TOOL_USE] {block.name}({block.input})\n")
                                continue
                            if agent_renderer.has_content:
                                agent_renderer.finish()
                            tool_registry.create(block.id, block.name, block.input or {})
                            _log_debug(f"[TOOL_USE] {block.name}({block.input})\n")

                elif isinstance(msg, UserMessage):
                    for user_block in msg.content:
                        if isinstance(user_block, ToolResultBlock):
                            content_str = str(user_block.content) if user_block.content else ""
                            panel = tool_registry.get(user_block.tool_use_id)
                            if panel:
                                panel.set_result(content_str, user_block.is_error or False)
                                panel.finish()
                                tool_registry.remove(user_block.tool_use_id)
                            else:
                                _log_debug(f"[WARN] No panel for tool_use_id={user_block.tool_use_id}\n")
                            error_marker = " [ERROR]" if user_block.is_error else ""
                            _log_debug(f"[TOOL_RESULT{error_marker}] {content_str}\n")

                elif isinstance(msg, ResultMessage):
                    if msg.structured_output is not None:
                        structured_result = msg.structured_output
                        # Format issues as agent text
                        issues = structured_result.get("issues", [])
                        if issues:
                            lines = [f"[{i['id']}] {i['file']}:{i['line']} - {i['description']}" for i in issues]
                            agent_renderer.append("\n".join(lines))
                        _log_debug(f"[STRUCTURED_OUTPUT] {structured_result}\n")
                    if msg.total_cost_usd:
                        if agent_renderer.has_content:
                            agent_renderer.finish()
                        console.print()
                        print_cost(console, msg.total_cost_usd)
                        _log_debug(f"[COST] ${msg.total_cost_usd:.4f}\n")

            if agent_renderer.has_content:
                agent_renderer.finish()
            tool_registry.finish_all()
            console.print()
        finally:
            _current_client = None

    if output_schema is not None and structured_result is not None:
        return structured_result
    return "".join(output_parts)
