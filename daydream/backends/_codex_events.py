"""Normalize Codex wire commands, file changes, and bounded parser diagnostics."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import uuid
from collections import Counter
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.backends import DiagnosticEvent, ToolResultEvent, ToolStartEvent
from daydream.trajectory import redact_structured_text

_DIAGNOSTIC_LABEL_MAX_CHARS = 64
_DIAGNOSTIC_LABEL_MAX_DISTINCT = 32
_NON_JSON_EXCERPT_MAX_CHARS_PER_LINE = 256
# Only observed public error items activate coverage gaps; absence cannot justify invented tool events.
# This literal identifies the external Codex release/wire-mode contract.
_TRANSPORT_COVERAGE_CONTRACT = "codex-cli-0.153.4-json-code-mode"

_SHELL_LC_PREFIX_RE = re.compile(r"^/bin/(?:zsh|bash|sh)\s+-lc\s+")


def _unwrap_shell_command(command: str) -> str:
    """Decode /bin/{zsh,bash,sh} -lc while preserving replayable payload bytes and leading cd.
    Single quoted payloads are unwrapped; bare multi-word payloads retain embedded quotes.
    Unknown/malformed wrappers, missing payloads, and quoted payloads with trailing args pass through.
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return command
    if len(argv) >= 2 and argv[0] in ("/bin/zsh", "/bin/bash", "/bin/sh") and argv[1] == "-lc":
        if len(argv) == 3:
            return argv[2]
        # Quoted payloads with trailing argv pass through; bare multi-word -lc payloads preserve raw embedded quotes.
        wrapper = _SHELL_LC_PREFIX_RE.match(command)
        if wrapper is not None:
            payload = command[wrapper.end() :]
            if payload and payload[0] not in ("'", '"'):
                return payload
    return command


_CD_PREFIX_RE = re.compile(r"^cd\s+\S+\s*&&\s*")


def display_shell_command(command: str) -> str:
    """Decode shell wrappers and strip a matching leading cd <dir> && for display only.
    Stored inputs preserve replay; malformed wrappers and unmatched cd prefixes pass through.
    """
    decoded = _unwrap_shell_command(command)
    return _CD_PREFIX_RE.sub("", decoded, count=1)


# Supervisors use this unredacted, unbounded strip for start-anchored command matching.
supervisor_shell_command = display_shell_command


def _bounded_diagnostic_label(value: Any) -> str:
    """Return one redacted, bounded scalar label for diagnostic aggregation."""
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        return "<non-scalar>"
    return redact_structured_text(str(value))[:_DIAGNOSTIC_LABEL_MAX_CHARS]


def _bounded_process_excerpt(value: str) -> str:
    """Redact a complete non-JSON line before applying the exception cap."""
    return redact_structured_text(value)[:_NON_JSON_EXCERPT_MAX_CHARS_PER_LINE]


def _record_unknown(
    counter: Counter[str],
    value: Any,
    *,
    overflow: list[int],
) -> None:
    """Count an unknown label with bounded retained cardinality."""
    label = _bounded_diagnostic_label(value)
    if label in counter:
        counter[label] += 1
    elif len(counter) < _DIAGNOSTIC_LABEL_MAX_DISTINCT:
        counter[label] = 1
    else:
        overflow[0] += 1


def _counter_summary(counter: Counter[str], overflow: list[int]) -> dict[str, Any]:
    return {
        "total": sum(counter.values()) + overflow[0],
        "labels": dict(counter),
        "overflow": overflow[0],
    }


@dataclass
class _CodexDiagnostics:
    """Bounded native parser evidence and first/final emission history for one invocation."""

    parse_warnings: Counter[str] = field(default_factory=Counter)
    unknown_event_types: Counter[str] = field(default_factory=Counter)
    unknown_event_overflow: list[int] = field(default_factory=lambda: [0])
    unknown_item_types: Counter[str] = field(default_factory=Counter)
    unknown_item_overflow: list[int] = field(default_factory=lambda: [0])
    malformed_shapes: Counter[str] = field(default_factory=Counter)
    non_json_count: int = 0
    error_sentinel_count: int = 0
    _emitted: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)

    def warning(self, reason: str) -> None:
        self.parse_warnings[reason] += 1
        logging.getLogger("daydream.backends.codex").warning("codex parser warning: %s", reason.replace("_", " "))

    def unknown_item(self, value: Any) -> None:
        _record_unknown(self.unknown_item_types, value, overflow=self.unknown_item_overflow)

    def unknown_event(self, value: Any) -> None:
        _record_unknown(self.unknown_event_types, value, overflow=self.unknown_event_overflow)

    def events(self, *, final: bool = False) -> Iterator[DiagnosticEvent]:
        """Emit initial code markers immediately; final polls also emit changed aggregates."""
        for event in self._current():
            signature = (event.message, event.metadata)
            if event.code not in self._emitted or final and self._emitted[event.code] != signature:
                self._emitted[event.code] = (event.message, deepcopy(event.metadata))
                yield event

    def _current(self) -> list[DiagnosticEvent]:
        """Build deterministic conditional diagnostics from bounded parser state."""
        diagnostics: list[DiagnosticEvent] = []
        if self.error_sentinel_count:
            diagnostics.append(
                DiagnosticEvent(
                    code="codex_transport_coverage",
                    message=(
                        "The current Codex public stream contains uncorrelated error items; "
                        "tool coverage is incomplete."
                    ),
                    metadata={
                        "coverage": "incomplete",
                        "reason": "uncorrelated_public_error_item",
                        "occurrences": self.error_sentinel_count,
                        "contract": _TRANSPORT_COVERAGE_CONTRACT,
                    },
                )
            )
        if (
            self.unknown_event_types
            or self.unknown_event_overflow[0]
            or self.unknown_item_types
            or self.unknown_item_overflow[0]
            or self.malformed_shapes
            or self.non_json_count
            or self.parse_warnings
        ):
            diagnostics.append(
                DiagnosticEvent(
                    code="codex_parser_coverage",
                    message="The Codex public stream contained parser coverage gaps.",
                    metadata={
                        "unknown_event_types": _counter_summary(self.unknown_event_types, self.unknown_event_overflow),
                        "unknown_item_types": _counter_summary(self.unknown_item_types, self.unknown_item_overflow),
                        "malformed_shapes": dict(self.malformed_shapes),
                        "non_json_lines": self.non_json_count,
                        "warnings": {
                            "total": sum(self.parse_warnings.values()),
                            "reasons": dict(self.parse_warnings),
                        },
                    },
                )
            )
        return diagnostics


def _file_change_events(
    item: dict[str, Any],
    execution_cwd: Path,
) -> Iterator[ToolStartEvent | ToolResultEvent]:
    """Synthesize patch start/result pairs while retaining legacy scalar supervisor inputs.
    Modern changes include path/kind lists with in-checkout absolute paths made relative.
    Missing status means success; pathless input errors; each result section is capped at 500 characters.
    """
    item_id = item.get("id", str(uuid.uuid4()))
    changes = item.get("changes")
    if isinstance(changes, (dict, list)):
        if isinstance(changes, list):
            changes = {
                str(change.get("path") or change.get("file_path")): change
                for change in changes
                if isinstance(change, dict) and (change.get("path") or change.get("file_path"))
            }
        parsed = []
        for raw_path, entry in changes.items():
            kind = entry.get("type", "unknown") if isinstance(entry, dict) else "unknown"
            path = str(raw_path)
            try:
                if os.path.isabs(path) and os.path.commonpath([path, str(execution_cwd)]) == str(execution_cwd):
                    path = os.path.relpath(path, execution_cwd)
            except ValueError:
                pass  # Keep absolute paths on disjoint drives.
            parsed.append({"path": path, "kind": kind})
        status = item.get("status") or "completed"
        output = ", ".join(f"{change['kind']}: {change['path']}" for change in parsed)[:500]
        if status == "declined":
            output = f"File change declined by sandbox: {output}"
        if status in ("failed", "declined"):
            for stream in ("stdout", "stderr"):
                excerpt = item.get(stream, "")
                if excerpt:
                    output += f"\n{stream}: {excerpt[:500]}"
        start_input: dict[str, Any] = {"changes": parsed, "file": "unknown", "action": "modified"}
        if len(parsed) == 1:
            start_input.update(file=parsed[0]["path"], action=parsed[0]["kind"])
        is_error = status != "completed"
    elif "file_path" in item:
        file_path = item.get("file_path", "unknown")
        action = item.get("action", "modified")
        start_input = {"file": file_path, "action": action}
        output = f"{action}: {file_path}"
        is_error = False
        status = None
    else:
        fields = {key: value for key, value in item.items() if key != "type" and value != "unknown"}
        start_input = {"file_change": fields}
        output = f"unparseable file_change item: {json.dumps(fields)[:500]}"
        is_error = True
        status = item.get("status") or None
    yield ToolStartEvent(
        id=item_id,
        name="patch",
        input=start_input,
    )
    yield ToolResultEvent(
        id=item_id,
        output=output,
        is_error=is_error,
        status=status,
    )
