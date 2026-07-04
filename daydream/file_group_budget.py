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

    Wall-clock starts at construction (``time.monotonic``). Token accounting is
    fed by the ``on_metrics`` callback wired through ``run_agent`` (accumulates
    ``prompt_tokens + completion_tokens`` per ``MetricsEvent``). The serial-item
    counter is bumped once per completed fix call via :meth:`record_item`.

    Attributes:
        max_wall_seconds: Total wall-clock ceiling for the group.
        max_serial_items: Max number of per-finding fix calls in the group.
        max_cumulative_tokens: Ceiling on input + output tokens across the group.
    """

    max_wall_seconds: float
    max_serial_items: int
    max_cumulative_tokens: int
    _wall_start: float = field(default_factory=time.monotonic)
    _items_processed: int = 0
    _cumulative_tokens: int = 0

    def check(self) -> str | None:
        """Return a budget-reason string if any ceiling is reached, else None.

        Pure read (no side effects): safe to call before every fix call. The
        checks are ordered items → wall → tokens so the reason is deterministic
        when more than one ceiling is simultaneously breached.
        """
        if self._items_processed >= self.max_serial_items:
            return "group_serial_item_limit"
        if time.monotonic() - self._wall_start >= self.max_wall_seconds:
            return "group_wall_budget_exceeded"
        if self._cumulative_tokens >= self.max_cumulative_tokens:
            return "group_token_budget_exceeded"
        return None

    def add_tokens(self, tokens: int) -> None:
        """Accumulate per-turn tokens. Wired as ``run_agent``'s ``on_metrics``."""
        self._cumulative_tokens += tokens

    def record_item(self) -> None:
        """Mark one completed fix call. Bumps the serial-item counter."""
        self._items_processed += 1

    @property
    def items_processed(self) -> int:
        """Number of fix calls completed in this group so far."""
        return self._items_processed
