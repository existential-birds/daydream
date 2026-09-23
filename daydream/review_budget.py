"""Persist incomplete review coverage without turning budget stops into run failures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
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
_review_scale: ContextVar[int] = ContextVar("review_scale", default=1)


def review_scale_for_diff(diff: str) -> int:
    """Give large changes room to finish without slowing modest reviews."""
    lines = diff.count("\n") + bool(diff and not diff.endswith("\n"))
    size = len(diff.encode("utf-8"))
    if lines > 10_000 or size > 512 * 1024:
        return 6
    if lines > 5_000 or size > 256 * 1024:
        return 4
    if lines > 1_000 or size > 64 * 1024:
        return 2
    return 1


def review_limits_for_scope(limits: ReviewLimits) -> ReviewLimits:
    """Scale each review role's existing time and call allowances together."""
    scale = _review_scale.get()
    return replace(
        limits,
        investigation_s=limits.investigation_s * scale,
        finalization_s=limits.finalization_s * scale,
        tool_calls=limits.tool_calls * scale,
    )


def review_scale_for_scope() -> int:
    """Return the current review's workload multiplier for outer phase guards."""
    return _review_scale.get()


@contextmanager
def review_deadline_scope(
    seconds: float, *, diff: str = "", scale_deadline: bool = False
) -> Iterator[None]:
    """A fresh/resumed review shares one deadline, including queue and retry time.

    Only review-limited calls consume this scope; fixing and publication do not.
    Context-local state isolates concurrent runs and is restored on cancellation.
    """
    scale = review_scale_for_diff(diff)
    effective_seconds = seconds * scale if scale_deadline else seconds
    finish = clock.monotonic() + effective_seconds
    token = _review_deadline.set(
        (finish - min(300 * scale, effective_seconds / 5), finish)
    )
    scale_token = _review_scale.set(scale)
    try:
        yield
    finally:
        _review_scale.reset(scale_token)
        _review_deadline.reset(token)


def review_deadline(*, discovery: bool) -> float | None:
    """Return the discovery or total deadline, with scaled synthesis reserve."""
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
