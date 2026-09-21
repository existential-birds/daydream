"""Persist incomplete review coverage without turning budget stops into run failures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream import clock


@dataclass(frozen=True)
class ReviewLimits:
    """Investigation bounds plus a separate, bounded evidence finalization turn."""

    investigation_s: float = 480
    finalization_s: float = 120
    tool_calls: int = 48
    discovery: bool = True


_review_deadline: ContextVar[tuple[float, float] | None] = ContextVar("review_deadline", default=None)


@contextmanager
def review_deadline_scope(seconds: float) -> Iterator[None]:
    """A fresh/resumed review shares one deadline, including queue and retry time.

    Only review-limited calls consume this scope; fixing and publication do not.
    Context-local state isolates concurrent runs and is restored on cancellation.
    """
    finish = clock.monotonic() + seconds
    token = _review_deadline.set((finish - min(300, seconds / 5), finish))
    try:
        yield
    finally:
        _review_deadline.reset(token)


def review_deadline(*, discovery: bool) -> float | None:
    """Reserve five minutes of the model pipeline for synthesis/adjudication."""
    deadlines = _review_deadline.get()
    return deadlines[0 if discovery else 1] if deadlines is not None else None


class ReviewBudgetExceeded(RuntimeError):
    """A review phase stopped before producing its final result."""

    def __init__(self, phase: str, reason: str, partial_result: Any = None) -> None:
        self.phase = phase
        self.reason = reason
        self.partial_result = partial_result
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
    """Collect incomplete phases and stacks for reports and posting."""
    from daydream.deep.artifacts import _load_failures, per_stack_failures_path

    path = review_budget_path(deep_dir)
    stops = json.loads(path.read_text()) if path.exists() else {}
    warnings = [f"{phase}: {reason}" for phase, reason in sorted(stops.items())]
    warnings.extend(
        f"{stack}: {reason}"
        for stack, reason in sorted(_load_failures(per_stack_failures_path(deep_dir)).items())
        if isinstance(reason, str)
        and reason.startswith(("budget exhausted:", "evidence incomplete:"))
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
