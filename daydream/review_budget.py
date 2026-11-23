"""Persist incomplete review coverage without turning budget stops into run failures."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import clock
from daydream.config import REVIEWER_TOOL_CALL_BUDGET
from daydream.output_schema import strict_object
from daydream.retry_policy import FailureClass, classify_failure

if TYPE_CHECKING:
    from daydream.backends.lifecycle import BackendLifecycleSnapshot


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


class ReasonCode(StrEnum):
    """The one public cause vocabulary: every stop's ``code`` and every outcome's reasons."""

    BACKEND_FAILURE = 'backend_failure'
    AUTHENTICATION_FAILURE = 'authentication_failure'
    MALFORMED_OUTPUT = 'malformed_output'
    MISSING_OUTPUT = 'missing_output'
    MALFORMED_ARTIFACT = 'malformed_artifact'
    MISSING_ARTIFACT = 'missing_artifact'
    EVIDENCE_INCOMPLETE = 'evidence_incomplete'
    HOST_WALL_BUDGET_EXHAUSTION = 'host_wall_budget_exhaustion'
    HOST_TOOL_BUDGET_EXHAUSTION = 'host_tool_budget_exhaustion'
    SYNTHESIS_FAILURE = 'synthesis_failure'
    INTERRUPTION = 'interruption'
    UNEXPECTED_ANALYSIS_FAILURE = 'unexpected_analysis_failure'
    COVERAGE_UNKNOWN = 'coverage_unknown'
    POLICY_VETO = 'policy_veto'
    DIRTY_SNAPSHOT = 'dirty_snapshot'


#: The codes a ledger limit assigns; their stops name the ledger whose limit applied.
BUDGET_REASON_CODES = frozenset({ReasonCode.HOST_WALL_BUDGET_EXHAUSTION, ReasonCode.HOST_TOOL_BUDGET_EXHAUSTION})


def reason_for_exception(exc: BaseException) -> ReasonCode:
    """The one exception-to-code mapping: a declared code, else the retry policy's failure class."""
    try:
        return ReasonCode(str(getattr(exc, 'reason_code', None)))
    except ValueError:
        pass
    return {
        FailureClass.AUTH_CONFIG: ReasonCode.AUTHENTICATION_FAILURE,
        FailureClass.SCHEMA: ReasonCode.MALFORMED_OUTPUT,
        FailureClass.TOOL_POLICY: ReasonCode.POLICY_VETO,
    }.get(classify_failure(exc).failure_class, ReasonCode.BACKEND_FAILURE)


# Host-chosen identifiers only: ledger owners, exception type names and backend categories.
_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_.:-]{0,127}')
_CATEGORY = re.compile(r'[A-Z][A-Z0-9_]{0,63}')
_NAME_SCHEMA = {'type': 'string', 'pattern': f'^{_NAME.pattern}$'}
_SECONDS = {'type': ['number', 'null'], 'minimum': 0}
STOP_SCHEMA = strict_object({
    'code': {'enum': [code.value for code in ReasonCode]},
    'owner': _NAME_SCHEMA,
    'bound': {**_NAME_SCHEMA, 'type': ['string', 'null']},
    'limit': _SECONDS, 'spent': _SECONDS,
    'attempts': {'type': 'integer', 'minimum': 0},
    'error': {'anyOf': [{'type': 'null'}, strict_object({
        'type': _NAME_SCHEMA,
        'failure_class': {'enum': [member.value for member in FailureClass]},
        'category': {'type': ['string', 'null'], 'pattern': f'^{_CATEGORY.pattern}$'},
        'status': {'type': ['integer', 'null'], 'minimum': 100, 'maximum': 599},
        'retryable': {'type': 'boolean'},
    })]},
    'progress': {'anyOf': [{'type': 'null'}, strict_object({
        'idle_s': _SECONDS, 'active_tools': {'type': 'integer', 'minimum': 0}, 'generation_open': {'type': 'boolean'},
    })]},
})


def _error_facts(exc: BaseException) -> dict[str, Any]:
    """A backend failure's content-free facts; a value that is not a bounded identifier is not recorded."""
    decision = classify_failure(exc)
    category, status = getattr(exc, 'category', None), getattr(exc, 'status', None)
    return {
        'type': name if _NAME.fullmatch(name := type(exc).__name__) else 'Exception',
        'failure_class': decision.failure_class.value,
        'category': category if isinstance(category, str) and _CATEGORY.fullmatch(category) else None,
        'status': status if type(status) is int and 100 <= status <= 599 else None,
        'retryable': decision.retries_allowed,
    }


@dataclass(frozen=True)
class StopRecord:
    """One stop, described where it is observed and carried unchanged to the sealed result.

    Content-free and observed only: ``owner`` is the work that stopped and ``bound`` the ledger
    whose limit applied, with that ledger's ``limit`` and ``spent`` (seconds or tool calls).
    ``attempts`` counts dispatched attempts. ``error`` keeps the last backend failure even when a
    deadline ended the call, and ``progress`` the last attempt's native progress, when its
    backend observes it. Reason codes, span status and every rendered text are projections of it.
    """

    code: ReasonCode
    owner: str
    bound: str | None = None
    limit: float | None = None
    spent: float | None = None
    attempts: int = 0
    error: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None

    @classmethod
    def of(cls, exc: BaseException, owner: str) -> StopRecord:
        """A host failure that no invocation ledger observed."""
        return cls(reason_for_exception(exc), owner, error=_error_facts(exc))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), 'code': self.code.value}

    def attributes(self) -> dict[str, Any]:
        """The ``daydream.stop.*`` trace attributes: nested facts flattened one level, absent facts omitted."""
        flat: dict[str, Any] = {}
        for key, value in self.to_dict().items():
            for name, item in (value.items() if isinstance(value, dict) else [(None, value)]):
                if item is not None:
                    flat[f"daydream.stop.{key}" + (f".{name}" if name else "")] = item
        return flat


@dataclass(eq=False)
class BudgetLedger:
    """One budget owner's absolute deadline and charged tool-call count.

    ``deadline`` is absolute on ``clock.monotonic()``. A relative ``wall_s`` is anchored
    once, when ``run_agent`` starts after admission, and is never re-based. ``charged``
    counts authorized tool starts against ``limit``; ``observed`` also counts refused
    and over-limit starts. A retry after a retryable provider failure refunds that
    attempt's charges; elapsed time and compaction never renew anything. A carved child
    charges and refunds its ancestors too, so parallel siblings never share a count, and
    remembers which ledger set its deadline, so a stop names the limit that applied.
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
    parent: BudgetLedger | None = field(default=None, repr=False)
    charged: int = 0
    observed: int = 0
    refunded: int = 0
    started: float | None = None
    attempts: int = field(default=0, init=False)
    stopped: StopRecord | None = field(default=None, init=False)
    _deadline_owner: BudgetLedger | None = field(default=None, init=False, repr=False)
    _limit_owner: BudgetLedger | None = field(default=None, init=False, repr=False)

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
            if self.deadline is None or bound <= self.deadline:
                self.deadline, self._deadline_owner = bound, None
            self.wall_s = None
        return self

    def begin(self) -> BudgetLedger:
        """Start one invocation: a reused ledger forgets the previous attempts and stop."""
        self.stopped = None
        self.attempts = 0
        return self

    def carve(self, owner: str, *, limit: int | None = None, deadline: float | None = None,
              reserved: int = 0, start_floor: int | None = None) -> BudgetLedger:
        """Allocate a child within this ledger's remaining calls and deadline."""
        self.anchor()
        remaining = self.remaining_tool_calls
        # The child's own limit or deadline applies when it is at least as tight as the inherited one.
        inherits_limit = remaining is not None and (limit is None or remaining < limit)
        if remaining is not None:
            limit = remaining if limit is None else min(limit, remaining)
        child = BudgetLedger(
            owner, deadline=self.deadline, limit=limit, reserved=reserved,
            start_floor=self.start_floor if start_floor is None else start_floor,
            investigation=self.investigation, parent=self, started=clock.monotonic(),
        )
        if deadline is not None and (self.deadline is None or deadline <= self.deadline):
            child.deadline = deadline
        else:
            child._deadline_owner = self._deadline_owner or self
        if inherits_limit:
            child._limit_owner = self._limit_owner or self
        return child

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
        """Count one dispatched attempt; the returned count is its refund point."""
        self.attempts += 1
        return self.charged

    def refund(self, since: int) -> None:
        """Refund charges made after the ``since`` dispatch snapshot; time is never refunded."""
        amount = max(0, self.charged - since)
        for ledger in self._chain():
            ledger.charged -= amount
            ledger.refunded += amount

    def stop(self, code: ReasonCode, *, error: BaseException | None = None,
             progress: BackendLifecycleSnapshot | None = None) -> StopRecord:
        """Record this owner's one stop; a ledger limit names the ledger whose limit applied."""
        bound: BudgetLedger | None = None
        limit = spent = None
        if code is ReasonCode.HOST_WALL_BUDGET_EXHAUSTION and self.deadline is not None:
            bound = self._deadline_owner or self
            started = bound.started if bound.started is not None else clock.monotonic()
            limit = round(max(0.0, bound.deadline - started), 3) if bound.deadline is not None else None
            spent = round(max(0.0, clock.monotonic() - started), 3)
        elif code is ReasonCode.HOST_TOOL_BUDGET_EXHAUSTION:
            capped = [ledger for ledger in self._chain() if ledger.limit is not None]
            if capped:
                tightest = min(capped, key=lambda ledger: max(0, (ledger.limit or 0) - ledger.charged))
                bound = tightest._limit_owner or tightest
                limit, spent = bound.limit, bound.charged
        self.stopped = StopRecord(code, self.owner, bound.owner if bound is not None else None, limit, spent,
                            self.attempts, _error_facts(error) if error is not None else None,
                            progress.progress_facts() if progress is not None else None)
        return self.stopped

    def fail(self, exc: BaseException, *, progress: BackendLifecycleSnapshot | None = None) -> StopRecord:
        """Record a failure that ended this owner's work: the one exception-to-code mapping and its facts."""
        return self.stop(reason_for_exception(exc), error=exc, progress=progress)


class BudgetExhausted(Exception):
    """The ledger deadline, not a liveness failure, ended a backend read."""

    retryable = False
    reason_code = ReasonCode.HOST_WALL_BUDGET_EXHAUSTION.value


def review_warnings(deep_dir: Path) -> tuple[str, ...]:
    """Render checked unfinished coverage by its reason codes."""
    from daydream.deep.artifacts import DeepArtifact
    from daydream.review_result import ReviewCoverage

    path = DeepArtifact.REVIEW_COVERAGE.at(deep_dir)
    if not path.exists():
        return ()
    coverage = ReviewCoverage.from_dict(json.loads(path.read_text()))
    return tuple(f"{key}: {', '.join(outcome['reason_codes'])}"
                 for outcomes in (coverage.phases, coverage.scopes)
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
