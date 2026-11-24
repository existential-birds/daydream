"""Per-invocation private directory for the Antigravity (``agy``) adapter.

``agy`` loads a custom agent, hooks and MCP servers from ``<dir>/.agents`` of a
directory passed with ``--add-dir``. Each invocation gets its own: a Daydream-owned
agent with an explicit tool list, the deny-by-default guard hook with its policy,
and, only with host tools, the loopback MCP server's URL and bearer. The bearer is
written nowhere else, and the guard denies every tool path inside this directory.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from daydream.backends._command_guards import BACKGROUND_COMMAND_PATTERN, DANGEROUS_COMMAND_PATTERNS
from daydream.backends._mcp_host import HOST_SERVER

_MCP_TOOL = "call_mcp_tool"
_FINISH_TOOL = "finish"
READ_ONLY_TOOLS: tuple[str, ...] = ("view_file", "grep_search", "find_by_name", "list_dir")
MUTATING_TOOLS: tuple[str, ...] = READ_ONLY_TOOLS + (
    "write_to_file", "replace_file_content", "multi_replace_file_content", "run_command", "command_status",
)
# The only permission rule an invocation adds, and only when it carries host tools.
MCP_PERMISSION_SETTINGS = '{"permissions":{"allow":["mcp(daydream/*)"]}}'
# agy 1.5.15 reports these registry tools under another name in events and hook input.
# Names recorded for no alias are listed as registered: an unrecorded alias fails closed.
_HOOK_ALIASES = {"view_file": "read_file", "replace_file_content": "edit_file", "write_to_file": "write_file"}
_DEFAULT_INSTRUCTIONS = "Complete the user's request using only the tools this agent provides."
# H1 lines delimit the agent file's sections, so one inside the instructions is escaped to stay text.
# Stable policy text: models otherwise nest the payload and spend several rejected calls recovering.
_MCP_CALL_SHAPE = (
    f'Host tools are served by the MCP server "{HOST_SERVER}". Call one as {_MCP_TOOL} with ServerName '
    f'"{HOST_SERVER}", ToolName set to the bare tool name (for example structured_output), and Arguments set to '
    "that tool's argument object directly, never wrapped in another Arguments object."
)
_H1_LINE = re.compile(r"(?m)^( {0,3})#(?=\s)")


@dataclass(frozen=True)
class RunDirectory:
    """One invocation's private directory, the agent it defines and that agent's tools."""

    path: Path
    agent: str
    tools: tuple[str, ...]

    def remove(self) -> None:
        """Delete the directory and the bearer with it; safe to repeat."""
        shutil.rmtree(self.path, ignore_errors=True)


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as handle:
        handle.write(text)


def create_run_directory(cwd: Path, *, read_only: bool, review_instructions: str | None,
                         mcp_url: str | None = None, mcp_token: str | None = None,
                         native_schema: bool = False) -> RunDirectory:
    """Write the agent, guard hook and policy (and MCP config, given a URL and bearer) to a fresh 0700 directory."""
    host_tools = mcp_url is not None and mcp_token is not None
    agent = "daydream-review" if read_only else "daydream-edit"
    # MCP access comes from ``inheritMcp``: agy's registry has no ``call_mcp_tool`` to list and refuses to start.
    # agy returns ``--json-schema`` output through its ``finish`` tool: without it the turn loops and
    # the result carries no structured output.
    tools = (READ_ONLY_TOOLS if read_only else MUTATING_TOOLS) + ((_FINISH_TOOL,) if native_schema else ())
    instructions = _H1_LINE.sub(r"\1\#", review_instructions or _DEFAULT_INSTRUCTIONS)
    if host_tools:
        instructions += "\n\n" + _MCP_CALL_SHAPE
    root = Path(tempfile.mkdtemp(prefix="daydream-antigravity-")).resolve()
    try:
        agents = root / ".agents"
        frontmatter = [
            "---", f"name: {agent}", "description: Daydream-owned agent for one invocation.",
            "mainAgent: true", "inheritMcp: true", "tools:", *(f"  - {tool}" for tool in tools), "---",
        ]
        _write_private(agents / "agents" / f"{agent}.md", "\n".join(
            [*frontmatter, "", "# System Prompt", "", instructions, ""]))

        guard, policy = root / "guard.py", root / "guard-policy.json"
        _write_private(guard, resources.files("daydream.backends").joinpath("antigravity_guard.py").read_text(
            encoding="utf-8"))
        denied_commands = [] if read_only else [
            *({"pattern": pattern.pattern, "reason": "dangerous command blocked (always-on guard)"}
              for pattern in DANGEROUS_COMMAND_PATTERNS),
            {"pattern": BACKGROUND_COMMAND_PATTERN.pattern,
             "reason": "background command blocked (always-on guard): daydream reads this turn's final text as "
                       "the result, so a command left running with '&' never reports. Run it in the foreground "
                       "and wait for it to finish."},
        ]
        _write_private(policy, json.dumps({
            "allowed_tools": [*tools, *(_HOOK_ALIASES[tool] for tool in tools if tool in _HOOK_ALIASES),
                              *((_MCP_TOOL,) if host_tools else ())],
            "protected_root": str(root),
            "working_directory": str(cwd.resolve()),
            "mcp_server": HOST_SERVER if host_tools else None,
            "denied_commands": denied_commands,
        }, indent=2))
        _write_private(agents / "hooks.json", json.dumps({HOST_SERVER: {"PreToolUse": [{"matcher": ".*", "hooks": [{
            "type": "command", "command": shlex.join([sys.executable, "-I", str(guard), str(policy)]), "timeout": 10,
        }]}]}}, indent=2))
        if host_tools:
            _write_private(agents / "mcp_config.json", json.dumps({"mcpServers": {HOST_SERVER: {
                "url": mcp_url, "headers": {"Authorization": f"Bearer {mcp_token}"}}}}, indent=2))
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return RunDirectory(root, agent, tools)
