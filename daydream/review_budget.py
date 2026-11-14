"""Persist incomplete review coverage without turning budget stops into run failures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from daydream import clock
from daydream.config import REVIEWER_TOOL_CALL_BUDGET


@dataclass(frozen=True)
class ReviewLimits:
    """Investigation bounds plus a separate, bounded evidence finalization turn."""

    investigation_s: float = 480
    finalization_s: float = 120
    tool_calls: int = REVIEWER_TOOL_CALL_BUDGET
    discovery: bool = True


_review_deadline: ContextVar[tuple[float, float] | None] = ContextVar("review_deadline", default=None)
_review_scale: ContextVar[int] = ContextVar("review_scale", default=1)


def review_scale_for_diff(diff: str) -> int:
    """Give large changes room to finish without slowing modest reviews."""
    lines = diff.count("\n") + bool(diff and not diff.endswith("\n"))
    size = len(diff.encode("utf-8"))
    if lines > 10_000 or size > 512 * 1024:
        return 9
    if lines > 5_000 or size > 256 * 1024:
        return 6
    if lines > 1_000 or size > 64 * 1024:
        return 3
    return 1


def review_limits_for_scope(limits: ReviewLimits) -> ReviewLimits:
    """Scale a role's time and call allowances together, once, by the run's diff-size tier."""
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
    """A review shares one deadline, including queue and retry time.

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


BudgetDimension = Literal["time", "tool_calls"]


@dataclass(frozen=True)
class BudgetStop:
    """Facts of one budget stop: durations and counts only, never monotonic values."""

    owner: str
    dimension: BudgetDimension
    limit: float
    spent: float
    remaining_at_dispatch: float
    reserved: int

    def attributes(self) -> dict[str, Any]:
        """The ``daydream.budget.*`` trace attributes (agent span or ``daydream.budget.stop`` event)."""
        return {f"daydream.budget.{key}": value for key, value in asdict(self).items()}


@dataclass(eq=False)
class BudgetLedger:
    """One budget owner's absolute deadline and charged tool-call count.

    ``deadline`` is absolute on ``clock.monotonic()``. A relative ``wall_s`` is anchored
    once, when ``run_agent`` starts after admission, and is never re-based. ``charged``
    counts authorized tool starts against ``limit``; ``observed`` also counts refused
    and over-limit starts. A retry after a retryable provider failure refunds that
    attempt's charges; elapsed time and compaction never renew anything. A carved child
    charges and refunds its ancestors too, so parallel siblings never share a count.
    ``reserved`` records calls already withheld from ``limit`` for later work.
    ``start_floor`` refuses a dispatch whose remaining calls cannot fit minimum work.
    ``investigation`` marks a reviewer's cumulative ledger, whose dispatch-time
    snapshots are rendered to the model.
    """

    owner: str
    deadline: float | None = None
    limit: int | None = None
    wall_s: float | None = None
    start_floor: int = 0
    reserved: int = 0
    investigation: bool = False
    shared_deadline: float | None = None
    parent: BudgetLedger | None = field(default=None, repr=False)
    charged: int = 0
    observed: int = 0
    refunded: int = 0
    started: float | None = None
    stopped: BudgetStop | None = field(default=None, init=False)
    _dispatched: tuple[float, int] | None = field(default=None, init=False, repr=False)


    def _chain(self) -> Iterator[BudgetLedger]:
        ledger: BudgetLedger | None = self
        while ledger is not None:
            yield ledger
            ledger = ledger.parent

    def anchor(self) -> BudgetLedger:
        """Fix a relative wall allowance to an absolute deadline, once."""
        now = clock.monotonic()
        if self.started is None:
            self.started = now
        if self.wall_s is not None:
            bound = now + self.wall_s
            self.deadline = bound if self.deadline is None else min(self.deadline, bound)
            self.wall_s = None
        return self

    def begin(self) -> BudgetLedger:
        """Start one invocation: a reused ledger forgets the previous dispatch and stop."""
        self.stopped = None
        self._dispatched = None
        return self

    def carve(self, owner: str, *, limit: int | None = None, deadline: float | None = None,
              reserved: int = 0, start_floor: int | None = None) -> BudgetLedger:
        """Allocate a child within this ledger's remaining calls and deadline."""
        self.anchor()
        remaining = self.remaining_tool_calls
        if remaining is not None:
            limit = remaining if limit is None else min(limit, remaining)
        bounds = [bound for bound in (self.deadline, deadline) if bound is not None]
        return BudgetLedger(
            owner, deadline=min(bounds) if bounds else None, limit=limit, reserved=reserved,
            start_floor=self.start_floor if start_floor is None else start_floor,
            investigation=self.investigation, shared_deadline=self.shared_deadline, parent=self,
            started=clock.monotonic(),
        )

    @property
    def remaining_tool_calls(self) -> int | None:
        remaining: int | None = None
        for ledger in self._chain():
            if ledger.limit is not None:
                own = max(0, ledger.limit - ledger.charged)
                remaining = own if remaining is None else min(remaining, own)
        return remaining

    def remaining_s(self) -> float | None:
        return None if self.deadline is None else max(0.0, self.deadline - clock.monotonic())

    def expired(self) -> bool:
        return self.deadline is not None and clock.monotonic() >= self.deadline

    def charge(self) -> bool:
        """Observe one tool start and charge it, unless the allocation is spent."""
        exhausted = self.remaining_tool_calls == 0
        for ledger in self._chain():
            ledger.observed += 1
            ledger.charged += not exhausted
        return not exhausted

    def deny(self) -> None:
        """Observe one start refused without charge."""
        for ledger in self._chain():
            ledger.observed += 1

    def dispatch(self) -> int:
        """Snapshot one attempt's dispatch; the returned count is its refund point."""
        self._dispatched = (clock.monotonic(), self.charged)
        return self.charged

    def refund(self, since: int) -> None:
        """Refund charges made after the ``since`` dispatch snapshot; time is never refunded."""
        amount = max(0, self.charged - since)
        for ledger in self._chain():
            ledger.charged -= amount
            ledger.refunded += amount

    def stop(self, dimension: BudgetDimension) -> BudgetStop:
        """Record the facts of the limit that stopped this owner; a dispatch-free stop reports now."""
        now = clock.monotonic()
        dispatched_at, charged_at = self._dispatched or (now, self.charged)
        if dimension == "time":
            started = self.started if self.started is not None else dispatched_at
            deadline = self.deadline if self.deadline is not None else now
            self.stopped = BudgetStop(
                self.owner, dimension, round(max(0.0, deadline - started), 3), round(max(0.0, now - started), 3),
                round(max(0.0, deadline - dispatched_at), 3), self.reserved)
        else:
            limit = self.limit if self.limit is not None else self.charged
            self.stopped = BudgetStop(self.owner, dimension, limit, self.charged, max(0, limit - charged_at),
                                      self.reserved)
        return self.stopped


class BudgetExhausted(Exception):
    """The ledger deadline, not a liveness failure, ended a backend read."""

    retryable = False


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
