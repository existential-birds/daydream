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


@dataclass
class ReviewInvestigationBudget:
    """Observed spend and an absolute deadline for one staged reviewer.

    The event stream exposes starts after native execution may have begun. This
    tracks received observations, including a start that exceeds the allowance;
    it does not promise prospective tool admission or unseen buffered events.
    """

    limits: ReviewLimits
    deadline: float
    observed_tool_starts: int = 0
    shared_deadline: float | None = None

    @classmethod
    def from_limits(
        cls, limits: ReviewLimits, *, deadline: float | None = None,
    ) -> ReviewInvestigationBudget:
        """Resolve scaling once; serialization time is never investigation time."""
        scaled = review_limits_for_scope(limits)
        bounds = [clock.monotonic() + scaled.investigation_s]
        shared = review_deadline(discovery=scaled.discovery)
        if shared is not None:
            bounds.append(shared)
        if deadline is not None:
            bounds.append(deadline)
        return cls(scaled, min(bounds), shared_deadline=shared)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.limits.tool_calls - self.observed_tool_starts)

    def observe_tool_start(self) -> None:
        """Charge before deadline and policy handling, without retry refunds."""
        self.observed_tool_starts += 1


class ReviewBudgetExceeded(RuntimeError):
    """A review phase stopped before producing its final result."""

    def __init__(self, phase: str, reason: str, partial_result: Any = None) -> None:
        self.phase = phase
        self.reason = reason
        self.partial_result = partial_result
        super().__init__(f"{phase} hit its budget: {reason}")


def review_warnings(deep_dir: Path) -> tuple[str, ...]:
    """Render checked unfinished coverage; freeform diagnostics cannot establish status."""
    from daydream.deep.artifacts import DeepArtifact
    from daydream.review_result import ReviewCoverage

    path = DeepArtifact.REVIEW_COVERAGE.at(deep_dir)
    if not path.exists():
        return ()
    coverage = ReviewCoverage.from_dict(json.loads(path.read_text()))
    return tuple(f"{key}: {coverage.diagnostics[kind].get(key, ', '.join(outcome['reason_codes']))}"
                 for kind, outcomes in [('phases', coverage.phases), ('scopes', coverage.scopes)]
                 for key, outcome in sorted(outcomes.items()) if outcome['status'] != 'complete')


def render_review_warnings(warnings: tuple[str, ...]) -> str:
    """An incomplete review must never look like a clean bill of health."""
    if not warnings:
        return ""
    return (
        "⚠️ **Review incomplete.** Findings from completed reviewers are included; "
        "additional issues may remain in the unfinished review.\n\n"
        + "\n".join(f"- {warning}" for warning in warnings)
    )
