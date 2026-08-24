"""Read-only resolution of exact completed Harbor benchmark runs (issue #888).

A future optimizer can consume an attributable, machine-readable Harbor
objective for any exact completed run. This module is the read-only surface:
it resolves a ledgered run by explicit ``run_id``, validates it reached a
terminal ``complete`` state, and binds its identity/metrics in later tasks.

Everything here is strictly read-only and immutable. State is modelled with
frozen dataclasses; ``_load_ledger`` is reused from ``run`` so the reader never
re-implements ledger parsing or validation — it only filters on ``state``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.benchmark.harbor import run as run_mod


class ObjectiveError(Exception):
    """The single typed error for objective resolution.

    Raised when a run is absent from the ledger, is not in a terminal
    ``complete`` state, or the ledger itself fails to parse/validate. The
    message always names the offending artifact path and run id.
    """


@dataclass(frozen=True)
class CompletedRun:
    """Immutable projection of a ledgered, completed benchmark run."""

    run_id: str
    mode: str
    state: str
    # Bound in Task 3 (full compatibility identity).
    identity: Any | None = None
    # Per-task reward rows (flattened); populated in Task 2.
    task_rows: list[dict[str, object] | None] = field(default_factory=list)
    # Bound in Task 4 (count-derived objective).
    objective: Any | None = None


def read_completed_run(
    workspace: Path, run_id: str, *, env: dict[str, Any] | None = None
) -> CompletedRun:
    """Resolve a ledgered run by explicit ``run_id``.

    Loads the ledger through ``run_mod._load_ledger`` and admits only a run
    whose ``state == "complete"``. Missing runs, running/cleanup-pending/
    cleaned runs, and any ledger parse/validation failure all fail closed with
    an ``ObjectiveError`` naming the run id and workspace.
    """
    del env  # reserved for provenance binding in later tasks.
    try:
        doc = run_mod._load_ledger(workspace)
    except run_mod.RunError as exc:
        raise ObjectiveError(f"ledger failure at {workspace}: {exc}") from exc

    entry = None
    for run in doc["runs"]:
        if run.get("run_id") == run_id:
            entry = run
            break

    if entry is None:
        raise ObjectiveError(
            f"run {run_id!r} not found in the harbor ledger at {workspace}"
        )
    if entry.get("state") != "complete":
        raise ObjectiveError(
            f"run {run_id!r} at {workspace} is not complete "
            f"(state {entry.get('state')!r})"
        )

    return CompletedRun(
        run_id=entry["run_id"],
        mode=entry["mode"],
        state=entry["state"],
    )
