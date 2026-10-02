"""Aggregate guard over all fix calls for one file group (#201)."""

from dataclasses import dataclass, field

from daydream import clock


def _wall_start() -> float:
    """Read the shared clock at instance construction so late-installed fake clocks apply."""
    return clock.monotonic()


@dataclass
class FileGroupBudget:
    """Bound aggregate wall time and completed fix calls for one file group.

    The clock starts at construction; record_item increments the serial count.
    check enforces limits between calls; deadline enables mid-call cancellation.
    Tokens add no independent bound beyond wall time and call count."""

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
        """Unclamped wall time consumed, for stop-event reporting."""
        return clock.monotonic() - self._wall_start

    def check(self) -> str | None:
        """Return the exceeded ceiling's reason without mutation, or None.
        Check items before wall time to resolve simultaneous breaches deterministically."""
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
