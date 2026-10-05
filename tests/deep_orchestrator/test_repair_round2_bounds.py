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
from pathlib import Path

import pytest

from daydream.agent import run_agent
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
    merge_repair_job_record,
    read_repair_job_record,
    record_diagnostic,
)
from daydream.fix_footprint import AuthorizedFixFootprint


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
        merge_repair_job_record(tmp_path, {"job_id": "repair-s1", "executions": 1})
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
        merge_repair_job_record(tmp_path, {"job_id": "repair-s1", "executions": 3})
        assert record_diagnostic(tmp_path, "repair-s2", "scope_request_rejected: nope") is None
        stored = read_repair_job_record(tmp_path)
        assert stored is not None
        assert stored.job_id == "repair-s1", "the record was relabelled into another job's grant"
        assert stored.diagnostics == ()
        assert stored.executions == 3


class TestPreambleStatesTheRealCeiling:
    """Item 2: the prompt must state the bound the host actually imposes."""

    def test_a_smaller_ceiling_is_stated_in_the_prompt(self) -> None:
        rendered = pi_system_preamble(wall_budget_s=90.0, tool_call_budget=12)
        assert "90" in rendered
        assert "1800" not in rendered, (
            "the prompt still hard-states DEFAULT_WALL_BUDGET_S for a 90s turn"
        )
        assert rendered == render_pi_preamble(90.0, 12)

    def test_defaults_are_used_when_the_caller_supplies_nothing(self) -> None:
        assert (
            pi_system_preamble()
            == render_pi_preamble(DEFAULT_WALL_BUDGET_S, DEFAULT_TOOL_CALL_BUDGET)
        )

    def test_the_module_preamble_is_not_mutated(self) -> None:
        """Rendering a small turn must not corrupt the shared default string."""
        pi_system_preamble(wall_budget_s=90.0, tool_call_budget=12)
        assert pi_system_preamble() == render_pi_preamble(
            DEFAULT_WALL_BUDGET_S, DEFAULT_TOOL_CALL_BUDGET
        )

    def test_only_the_wall_budget_can_differ(self) -> None:
        """An uncapped tool-call budget is the honest default, not a mismatch."""
        rendered = pi_system_preamble(wall_budget_s=DEFAULT_WALL_BUDGET_S, tool_call_budget=None)
        assert rendered == render_pi_preamble(DEFAULT_WALL_BUDGET_S, None)


class TestProductionPathSuppliesTheAllowances:
    """The renderer is only honest if the real dispatcher passes the values in.

    A renderer nothing calls with the real numbers reproduces the original
    defect exactly, so these assert the wiring, not just the helper.
    """

    def test_pi_backend_declares_the_capability(self) -> None:
        assert getattr(PiBackend, "supports_budget_preamble", False) is True

    def test_pi_execute_accepts_both_allowances(self) -> None:
        params = inspect.signature(PiBackend.execute).parameters
        assert "wall_budget_s" in params, "PiBackend.execute cannot receive the ceiling"
        assert "tool_call_budget" in params

    def test_run_agent_forwards_the_enforced_allowances(self) -> None:
        """agent.run_agent must pass its own wall_budget_s/tool_call_budget on."""
        source = (Path(__file__).resolve().parents[2] / "daydream/agent.py").read_text()
        assert "supports_budget_preamble" in source, (
            "no capability gate: the kwargs would reach every backend"
        )
        assert 'execute_kwargs["wall_budget_s"] = wall_budget_s' in source, (
            "run_agent does not forward the ceiling it enforces, so the prompt "
            "still states the module default"
        )
        assert 'execute_kwargs["tool_call_budget"] = tool_call_budget' in source

    def test_run_agent_exposes_the_allowances_as_parameters(self) -> None:
        params = inspect.signature(run_agent).parameters
        assert "wall_budget_s" in params
        assert "tool_call_budget" in params


def test_module_exports_remain_importable() -> None:
    """Guard against a partial import refactor shadowing the public surface."""
    assert callable(pi_system_preamble)
    assert callable(render_pi_preamble)
    assert DEFAULT_WALL_BUDGET_S > 0
    # None is the honest "uncapped" default for tool calls, not a bug.
    assert DEFAULT_TOOL_CALL_BUDGET is None or isinstance(DEFAULT_TOOL_CALL_BUDGET, int)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

