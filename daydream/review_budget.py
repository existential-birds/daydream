"""Persist incomplete review coverage without turning budget stops into run failures."""

from __future__ import annotations

import json
from pathlib import Path


class ReviewBudgetExceeded(RuntimeError):
    """A review phase stopped before producing its final result."""

    def __init__(self, phase: str, reason: str) -> None:
        self.phase = phase
        self.reason = reason
        super().__init__(f"{phase} hit its budget: {reason}")


def review_budget_path(deep_dir: Path) -> Path:
    """Phase budget stops retained across a merge/fix resume."""
    return deep_dir / "review-budget-stops.json"


def record_review_budget_stop(deep_dir: Path, phase: str, reason: str) -> None:
    """Record a non-stack budget stop; stack failures have their own artifact."""
    path = review_budget_path(deep_dir)
    stops = json.loads(path.read_text()) if path.exists() else {}
    stops[phase] = reason
    path.write_text(json.dumps(stops, indent=2, sort_keys=True))


def clear_review_budget_stop(deep_dir: Path, phase: str) -> None:
    """A completed rerun supersedes the earlier timeout for that phase."""
    path = review_budget_path(deep_dir)
    if path.exists():
        stops = json.loads(path.read_text())
        stops.pop(phase, None)
        path.write_text(json.dumps(stops, indent=2, sort_keys=True))


def review_warnings(deep_dir: Path) -> tuple[str, ...]:
    """Collect budget-limited phases and stacks for reports and posting."""
    from daydream.deep.artifacts import _load_failures, per_stack_failures_path

    path = review_budget_path(deep_dir)
    stops = json.loads(path.read_text()) if path.exists() else {}
    warnings = [f"{phase}: {reason}" for phase, reason in sorted(stops.items())]
    warnings.extend(
        f"{stack}: {reason}"
        for stack, reason in sorted(_load_failures(per_stack_failures_path(deep_dir)).items())
        if isinstance(reason, str) and reason.startswith("budget exhausted:")
    )
    return tuple(warnings)


def render_review_warnings(warnings: tuple[str, ...]) -> str:
    """An incomplete review must never look like a clean bill of health."""
    if not warnings:
        return ""
    return (
        "⚠️ **Review incomplete.** Findings from completed reviewers are included; "
        "additional issues may remain in the unfinished review.\n\n"
        + "\n".join(f"- {warning}" for warning in warnings)
    )
