"""Findings for review and fix phases."""

from typing import Any

from daydream.severity import SEVERITY_RANK

# Placeholder evidence strings that are structurally present but carry no
# grounding -- treated as "no evidence" by the gate (issue #227).
_PLACEHOLDER_EVIDENCE: frozenset[str] = frozenset({"n/a", "none", "-"})


def severity_sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable-sort canonical items by severity (high < medium < low)."""
    return sorted(items, key=lambda it: SEVERITY_RANK.get(it.get("severity") or "", 1))
