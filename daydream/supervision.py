"""Runtime findings and tool supervision helpers."""

from __future__ import annotations

import re
from typing import Any

from daydream.extensions import ToolDecision
from daydream.generated_files import match_suffix_glob
from daydream.trajectory import DaydreamPhase


class RuleBasedToolSupervisor:
    """Veto denied file writes/edits and Bash commands."""

    def __init__(self, *, deny_globs: list[str], bash_deny: list[str]) -> None:
        self._deny_globs = tuple(deny_globs)
        self._bash_deny = tuple(re.compile(pattern) for pattern in bash_deny)

    def __call__(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        phase: DaydreamPhase,
    ) -> ToolDecision:
        del phase
        if tool_name in {"Write", "Edit"}:
            path = tool_input.get("file_path") or tool_input.get("path")
            if isinstance(path, str):
                matched = match_suffix_glob(path, self._deny_globs)
                if matched is not None:
                    return ToolDecision(veto=True, reason=f"denied by glob '{matched}'")
        elif tool_name == "Bash":
            command = tool_input.get("command")
            if isinstance(command, str):
                for pattern in self._bash_deny:
                    if pattern.search(command):
                        return ToolDecision(
                            veto=True,
                            reason=f"Bash command denied by regex '{pattern.pattern}'",
                        )
        return ToolDecision(veto=False)
