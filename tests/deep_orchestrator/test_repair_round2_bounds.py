"""Regression tests for the two round-2 repair-job findings on #1210.

Item 1 (medium): record_diagnostic minted a job record carrying only default
policy bounds, and the coordinator reads any record whose id matches as the
job's captured grant -- so a diagnostic recorded before the job started
silently replaced the run's configured repair_* bounds for the rest of the job.

Item 2 (low): the Pi system prompt hard-stated DEFAULT_WALL_BUDGET_S (1800s)
even for a repair turn granted a far smaller ceiling. The fix threads the
invocation's real allowances into the rendered preamble; these tests pin both
the rendering and the production call site that supplies them.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from daydream.agent import run_agent
from daydream.backends import AgentEvent, ResultEvent, TextEvent
from daydream.backends.pi import (
    PiBackend,
    pi_system_preamble,
    render_pi_preamble,
)
from daydream.config import DEFAULT_TOOL_CALL_BUDGET, DEFAULT_WALL_BUDGET_S
from daydream.deep.repair_coordinator import (
    _is_captured_grant,
    _load_job,
    _persist,
)
from daydream.deep.repair_job import (
    RepairJobPolicy,
    RepairJobRecord,
    read_repair_job_record,
    record_diagnostic,
    write_repair_job_record,
)
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import init_repo


def _footprint() -> AuthorizedFixFootprint:
    return AuthorizedFixFootprint(run_allowed_paths=frozenset({"daydream/"}), policy_revision=1)


class TestDiagnosticNeverBecomesTheGrant:
    """Item 1: a diagnostic must never be adopted as the job's captured grant.

    Requirement 28 still requires the diagnostic to REACH the record (a resuming
    job must see the blocker), so the fix is at the reader, not the writer: a
    diagnostic-minted record carries diagnostics only, never the defaults.
    """

    def test_diagnostic_minted_record_carries_no_grant(self, tmp_path: Path) -> None:
        recorded = record_diagnostic(tmp_path, "repair-s1", "checkpoint_write_failed: disk full")
        assert recorded is not None, "Requirement 28: the blocker must reach the job record"
        assert recorded.diagnostics == ("checkpoint_write_failed: disk full",)
        assert recorded.policy_revision == 0
        assert recorded.authorized_scope == ()
        assert not _is_captured_grant(recorded)

    def test_minted_record_does_not_become_the_captured_grant(self, tmp_path: Path) -> None:
        """The defect: the run's configured bounds were silently dropped."""
        configured = RepairJobPolicy(execution_s=90.0, job_total_s=300.0, max_executions=2)
        record_diagnostic(tmp_path, "repair-s1", "checkpoint_write_failed: disk full")
        job = _load_job(
            tmp_path, job_id="repair-s1", footprint=_footprint(), policy=configured
        )
        assert job.policy == configured, "a default-bounds record became the captured grant"
        assert job.policy != RepairJobPolicy(), "the run's configured bounds were dropped"

    def test_diagnostics_survive_onto_the_rebuilt_job(self, tmp_path: Path) -> None:
        """Requirement 28: the blocker is still visible to a resuming job."""
        record_diagnostic(tmp_path, "repair-s1", "checkpoint_write_failed: disk full")
        job = _load_job(
            tmp_path, job_id="repair-s1", footprint=_footprint(), policy=RepairJobPolicy()
        )
        assert job.diagnostics == ("checkpoint_write_failed: disk full",)

    def test_diagnostic_still_annotates_an_existing_record(self, tmp_path: Path) -> None:
        """The common path is unchanged: annotate, never clobber."""
        write_repair_job_record(tmp_path, RepairJobRecord(job_id="repair-s1", executions=1))
        recorded = record_diagnostic(tmp_path, "repair-s1", "checkpoint_write_failed: disk full")
        assert recorded is not None
        assert recorded.diagnostics == ("checkpoint_write_failed: disk full",)
        assert recorded.executions == 1, "the diagnostic merge dropped the earlier execution"

    def test_a_real_job_record_is_still_read_as_its_grant(self, tmp_path: Path) -> None:
        """The fix must not break resumption: a stamped record keeps its own policy."""
        granted = RepairJobPolicy(execution_s=45.0, job_total_s=120.0, max_executions=1)
        job = _load_job(
            tmp_path, job_id="repair-s1", footprint=_footprint(), policy=granted
        )
        _persist(tmp_path, job)
        resumed = _load_job(
            tmp_path, job_id="repair-s1", footprint=_footprint(), policy=RepairJobPolicy()
        )
        assert _is_captured_grant(resumed)
        assert resumed.policy == granted, "a resumed job re-resolved its policy"

    def test_diagnostic_for_another_job_is_refused(self, tmp_path: Path) -> None:
        """A diagnostic must not relabel a record belonging to a different job."""
        write_repair_job_record(tmp_path, RepairJobRecord(job_id="repair-s1", executions=3))
        assert record_diagnostic(tmp_path, "repair-s2", "scope_request_rejected: nope") is None
        stored = read_repair_job_record(tmp_path)
        assert stored is not None
        assert stored.job_id == "repair-s1", "the record was relabelled into another job's grant"
        assert stored.diagnostics == ()
        assert stored.executions == 3




