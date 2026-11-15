"""Findings for review and fix phases."""

from typing import Any

from daydream.severity import SEVERITY_RANK


def severity_sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable-sort canonical items by severity (high < medium < low)."""
    return sorted(items, key=lambda it: SEVERITY_RANK.get(it.get("severity") or "", 1))
