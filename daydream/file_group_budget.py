"""Aggregate guard over all fix calls for one file group (#201)."""

import time
from dataclasses import dataclass, field


@dataclass
class FileGroupBudget:
    """Aggregate guard over all fix ``run_agent`` calls for one file group (#201).

    The per-invocation guards (``FIX_MAX_TURNS``, ``DEFAULT_TOOL_CALL_BUDGET``,
    ``DEFAULT_WALL_BUDGET_S``) bound each individual turn. This bounds their
    *sum* within a single file group so one file with many findings cannot
    silently dominate a review-fix-test run (root cause of the 62-min pi/GLM run
    in #186). Enforced between calls in ``phase_fix_parallel`` (Approach B —
    :meth:`check` is a pure read consulted before each fix call; there is no
    mid-call abort).

    Two axes bound the group: cumulative wall-clock (starts at construction via
    ``time.monotonic``) and the serial-item count (bumped once per completed fix
    call via :meth:`record_item`). Output tokens are deliberately not an axis —
    they are collinear with wall-time and call-count on the only population this
    between-calls guard can reach, so they add no independent signal (see the
    calibration in ``.beagle/concepts/file-group-budget/``).

    Attributes:
        max_wall_seconds: Total wall-clock ceiling for the group.
        max_serial_items: Max number of per-finding fix calls in the group.
    """

    max_wall_seconds: float
    max_serial_items: int
    _wall_start: float = field(init=False, default_factory=time.monotonic)
    _items_processed: int = field(init=False, default=0)

    def check(self) -> str | None:
        """Return a budget-reason string if any ceiling is reached, else None.

        Pure read (no side effects): safe to call before every fix call. The
        checks are ordered items → wall so the reason is deterministic when both
        ceilings are simultaneously breached.
        """
        if self._items_processed >= self.max_serial_items:
            return "group_serial_item_limit"
        if time.monotonic() - self._wall_start >= self.max_wall_seconds:
            return "group_wall_budget_exceeded"
        return None

    def record_item(self) -> None:
        """Mark one completed fix call. Bumps the serial-item counter."""
        self._items_processed += 1

    @property
    def items_processed(self) -> int:
        """Number of fix calls completed in this group so far."""
        return self._items_processed
