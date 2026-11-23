"""Deny-by-default PreToolUse hook for one Antigravity (``agy``) invocation.

Packaged data: the adapter copies this file into the invocation's private run
directory and ``agy`` runs it there as ``<python> guard.py <policy.json>`` with the
hook payload on stdin. It uses the standard library only and never imports
``daydream``, so the hook costs one bare interpreter start.

``agy`` allows a tool when the hook exits 0 with empty stdout, denies it on a
``{"decision": "deny"}`` object, and fails closed on a non-zero exit. Only a positive
allow decision exits 0 silently; every internal failure prints a denial and exits non-zero.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

_MCP_TOOL = "call_mcp_tool"
_SHELL_TOOL = "run_command"
_REASON_MAX_CHARS = 512


def _strings(value: object) -> Iterator[str]:
    """Every string anywhere in a JSON argument value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _decide(payload: object, policy: object) -> str | None:
    if not isinstance(policy, dict):
        return "the guard policy is malformed"
    allowed, root_text = policy.get("allowed_tools"), policy.get("protected_root")
    mcp_server, denied_commands = policy.get("mcp_server"), policy.get("denied_commands")
    if (not isinstance(allowed, list) or not isinstance(root_text, str) or not root_text
            or not isinstance(denied_commands, list) or not (mcp_server is None or isinstance(mcp_server, str))):
        return "the guard policy is malformed"
    call = payload.get("toolCall") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(call, dict):
        return "the hook payload is malformed"
    name, args, cwd = call.get("name"), call.get("args", {}), payload.get("cwd")
    if not isinstance(name, str) or not isinstance(args, dict) or not isinstance(cwd, str):
        return "the hook payload is malformed"
    if name not in allowed:
        return f"tool {name!r} is not allowed in this invocation"
    if name == _MCP_TOOL and (mcp_server is None or args.get("ServerName") != mcp_server):
        return "only this invocation's own MCP server may be called"

    root = Path(root_text).resolve()
    for text in _strings(args):
        candidate = Path(cwd, os.path.expanduser(text)).resolve()
        if candidate == root or root in candidate.parents:
            return "the invocation's private run directory is not accessible"

    command = args.get("CommandLine")
    if name == _SHELL_TOOL and not isinstance(command, str):
        return "the command line is malformed"
    if isinstance(command, str):
        if root_text in command or str(root) in command:
            return "the invocation's private run directory is not accessible"
        for entry in denied_commands:
            if re.compile(entry["pattern"]).search(command):
                return str(entry["reason"])
    return None


def decide(payload: object, policy: object) -> str | None:
    """Return the denial reason for one hook payload, or None to allow; never raises."""
    try:
        return _decide(payload, policy)
    except Exception as exc:  # noqa: BLE001 - an unexpected shape or path must deny, not allow
        return f"the tool call could not be checked ({type(exc).__name__})"


def _deny(reason: str, exit_code: int) -> int:
    """Print one denial; a denial that cannot be written still blocks through its exit code."""
    try:
        decision = {"decision": "deny", "reason": f"daydream guard: {reason}"[:_REASON_MAX_CHARS]}
        sys.stdout.write(json.dumps(decision) + "\n")
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        return exit_code or 2
    return exit_code


def main(argv: list[str]) -> int:
    """Decide the hook payload on stdin under the policy file named by ``argv[1]``."""
    try:
        policy = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        payload = json.loads(sys.stdin.read())
    except Exception as exc:  # noqa: BLE001 - unreadable input fails closed
        return _deny(f"the guard could not read its input ({type(exc).__name__})", 2)
    reason = decide(payload, policy)
    return 0 if reason is None else _deny(reason, 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
