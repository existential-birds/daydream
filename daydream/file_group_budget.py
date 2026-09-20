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

    Where the per-invocation guard bounds each turn,
    ``DEFAULT_WALL_BUDGET_S``/this bounds their *sum* in one group so a file
    with many findings cannot dominate a run. Two axes bound the group:
    cumulative wall-clock (started at construction via the shared clock seam)
    and serial-item count (bumped per completed call by :meth:`record_item`).
    Output tokens are deliberately not an axis, being collinear with wall-time
    and call-count here. Enforced by the pure between-calls :meth:`check` and
    the absolute :attr:`deadline` threaded into a fix call for mid-call abort.
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

    def elapsed_s(self) -> float:
        """Wall-clock seconds consumed by the group so far (unclamped).

        The recorded duration, not the remaining budget: a stop event states how
        much of the ceiling the group actually consumed.
        """
        return clock.monotonic() - self._wall_start

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
