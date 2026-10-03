"""Shared diagram values keep eligibility, grounding, and rendering independent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, TypedDict


def as_list(value: Any) -> list[Any]:
    """Return ``value`` when it is a list, else the empty list."""
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a dict, else the empty dict."""
    return value if isinstance(value, dict) else {}


def as_int(value: Any) -> int:
    """Return ``value`` when it is a real int (not a bool), else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def as_optional_str(value: Any) -> str | None:
    """Return ``value`` when it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


class SequenceSpec(TypedDict):
    """Schema-shaped, admitted sequence proposal or grounded final graph."""

    participants: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]


class FlowchartSpec(TypedDict):
    """Schema-shaped flowchart; None marks an unusable author root."""

    root: dict[str, Any] | None
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


# Ordered kind vocabularies shared by schema enums, grounding, and rendering.
PARTICIPANT_KINDS = ("internal", "external")
MESSAGE_KINDS = ("call", "reply", "self")
BLOCK_KINDS = ("alt", "opt", "loop")
NODE_KINDS = ("start", "end", "process", "decision", "subroutine", "io")

# Per-kind persisted outcome: status (rendered/omitted/skipped/failed), reason,
# spec_final (pruned, capped, schema-valid), omit_reasons, mermaid, and advisory.
# reason, spec_final, mermaid, and advisory may be None.
# Findings omit mermaid. Optional advisory records transport, allowance_bytes,
# admitted_bytes, admitted [{label, bytes}], and omitted [{label, bytes, reason}].
# Optional fields depend on status; consumers document the subset they read.
DiagramResult: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class DiagramThresholds:
    """Per-run thresholds resolved from file configuration over host defaults.

    Sequence eligibility counts changed non-test code files and distinct modules;
    flowcharts count changed branch points within one function.
    """

    min_code_files: int = 3
    min_modules: int = 2
    min_branch_points: int = 3


@dataclass(frozen=True)
class CandidateRoot:
    """One tree-sitter function/method root, with a repository-relative POSIX file path.

    line/end_line are 1-based and inclusive, spanning the definition and body for
    NODE_OUTSIDE_ROOT checks. branch_points counts branches inside both that range
    and head-side changed hunks; it drives eligibility thresholds and candidate order.
    """

    file: str
    name: str
    line: int
    end_line: int
    branch_points: int


__all__ = [
    "BLOCK_KINDS",
    "CandidateRoot",
    "DiagramResult",
    "DiagramThresholds",
    "MESSAGE_KINDS",
    "NODE_KINDS",
    "PARTICIPANT_KINDS",
]
