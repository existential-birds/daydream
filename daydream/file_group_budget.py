"""Aggregate guard over all fix calls for one file group (#201)."""

from dataclasses import dataclass, field

from daydream import clock


def _wall_start() -> float:
    """Read the shared clock seam at construction.

    A function default factory (rather than a direct call) keeps the read late
    so a fake clock installed after class definition still applies.
    """
    return clock.monotonic()


@dataclass
class FileGroupBudget:
    """Aggregate guard over all fix ``run_agent`` calls for one file group (#201).

    The per-invocation guard (``DEFAULT_WALL_BUDGET_S``) bounds each individual
    turn. This bounds their *sum* within a single file
    group so one file with many findings cannot silently dominate a
    review-fix-test run.
    Enforced two ways: :meth:`check` is a pure between-calls guard consulted
    before each fix call, and :attr:`deadline` is the same absolute wall limit
    threaded into a fix call so it can abort mid-call.

    Two axes bound the group: cumulative wall-clock (starts at construction via
    the shared clock seam) and the serial-item count (bumped once per completed
    fix call via :meth:`record_item`). Output tokens are deliberately not an
    axis — they are collinear with wall-time and call-count on the only
    population this guard can reach, so they add no independent signal.

    Attributes:
        max_wall_seconds: Total wall-clock ceiling for the group.
        max_serial_items: Max number of per-finding fix calls in the group.
    """

    max_wall_seconds: float
    max_serial_items: int
    _wall_start: float = field(init=False, default_factory=_wall_start)
    _items_processed: int = field(init=False, default=0)

    @property
    def deadline(self) -> float:
        """The absolute monotonic instant the group's wall ceiling is reached."""
        return self._wall_start + self.max_wall_seconds

    def remaining(self) -> float:
        """Wall-clock seconds left before the deadline, clamped at zero."""
        return max(0.0, self.deadline - clock.monotonic())

    def check(self) -> str | None:
        """Return a budget-reason string if any ceiling is reached, else None.

        Pure read (no side effects): safe to call before every fix call. The
        checks are ordered items → wall so the reason is deterministic when both
        ceilings are simultaneously breached.
        """
        if self._items_processed >= self.max_serial_items:
            return "group_serial_item_limit"
        if clock.monotonic() - self._wall_start >= self.max_wall_seconds:
            return "group_wall_budget_exceeded"
        return None

    def record_item(self) -> None:
        """Mark one completed fix call. Bumps the serial-item counter."""
        self._items_processed += 1

    @property
    def items_processed(self) -> int:
        """Number of fix calls completed in this group so far."""
        return self._items_processed
