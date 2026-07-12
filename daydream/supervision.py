"""Runtime findings and tool supervision helpers."""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from daydream.extensions import ToolDecision
from daydream.trajectory import DaydreamPhase

_REVISABLE_FINDING_FIELDS = (
    "severity",
    "confidence",
    "description",
    "rationale",
    "evidence",
)

FindingVerdictEvent = tuple[int, str, str]


def revise_finding_fields(record: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Apply only verdict fields that may revise a finding in place."""
    for field in _REVISABLE_FINDING_FIELDS:
        if field in verdict:
            record[field] = verdict[field]


def apply_findings_verdicts(
    items: list[dict[str, Any]],
    verdicts: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[FindingVerdictEvent]]:
    """Apply id-keyed findings verdicts in one fail-open pass."""
    kept: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    events: list[FindingVerdictEvent] = []

    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, int):
            kept.append(item)
            continue
        verdict = verdicts.get(item_id)
        if verdict is None or verdict.get("id") != item_id:
            kept.append(item)
            continue

        action = verdict.get("action")
        if not isinstance(action, str):
            kept.append(item)
            continue
        if action == "drop":
            events.append((item_id, action, str(verdict.get("reason", ""))))
        elif action == "edit":
            revise_finding_fields(item, verdict)
            kept.append(item)
            events.append((item_id, action, str(verdict.get("reason", ""))))
        elif action == "hold":
            held.append(item)
            events.append((item_id, action, str(verdict.get("reason", ""))))
        else:
            kept.append(item)

    return kept, held, events


def _matched_glob(path: str, deny_globs: tuple[str, ...]) -> str | None:
    normalized = path.replace("\\", "/").lstrip("/")
    parts = normalized.split("/")
    for pattern in deny_globs:
        candidates = ("/".join(parts[index:]) for index in range(len(parts)))
        if any(fnmatch.fnmatchcase(candidate, pattern) for candidate in candidates):
            return pattern
    return None


class RuleBasedSupervisor:
    """Apply deny-glob rules to canonical findings."""

    def __init__(self, *, deny_globs: list[str]) -> None:
        self._deny_globs = tuple(deny_globs)

    def review_findings(self, items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """Return drop verdicts for findings whose files match a deny glob."""
        verdicts: dict[int, dict[str, Any]] = {}
        for item in items:
            item_id = item.get("id")
            file_path = item.get("file")
            if not isinstance(item_id, int) or not isinstance(file_path, str):
                continue
            matched = _matched_glob(file_path, self._deny_globs)
            if matched is not None:
                verdicts[item_id] = {
                    "id": item_id,
                    "action": "drop",
                    "reason": f"denied by glob '{matched}'",
                }
        return verdicts


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
                matched = _matched_glob(path, self._deny_globs)
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
