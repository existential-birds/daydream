"""Group deadlines across retries, batches, serial fallback, and sibling work."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest

from tests.deep_orchestrator.support import (
    _batched_group_size,
    _scan_phase_events,
    _scan_trajectory_extra,
    _single_fix_calls_for,
)
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _install_stub_backend,
    _merge_item,
    _silence,
)


async def test_run_retry_ladder_is_bounded_by_the_group_deadline(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """(15a) retry backoff + backend time for one invocation cannot exceed the budget."""
    from daydream.backends.pi import PiError  # same import as tests/test_agent_retry.py:22
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock

    _silence(monkeypatch)
    fake = FakeClock(monotonic_value=10_000.0).install(monkeypatch)
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_WALL_S", 600.0)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "App.tsx", "high")]
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 250.0
    stub.fix_retryable_failures = 99  # a failing ladder that must be cut, not exhausted
    stub.fix_retryable_error = PiError("429 rate limit", retryable=True)
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    assert isinstance(exit_code, int)
    assert len([c for c in stub.calls if c["prompt"].lower().startswith("fix this issue")]) == 3
    recorded = json.loads((multi_stack_target / ".daydream/deep/fix-failures.json").read_text())
    assert recorded["App.tsx"].startswith("file_group_budget_exceeded: group_wall_budget_exceeded")


async def test_run_cuts_a_single_item_group_at_the_group_deadline(  # (15b)
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """(15b) A one-call group is cut at the GROUP deadline, and the turn is not progress."""
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock

    _silence(monkeypatch)
    fake = FakeClock(monotonic_value=10_000.0).install(monkeypatch)
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_WALL_S", 600.0)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "App.tsx", "high")]  # exactly one finding -> one call
    stub.runaway_single_fix_file = "App.tsx"
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 200.0
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert isinstance(exit_code, int)

    singles = _single_fix_calls_for(stub, "App.tsx")
    assert len(singles) == 1  # exactly one call, no fallback loop
    recorded = json.loads((multi_stack_target / ".daydream/deep/fix-failures.json").read_text())
    assert recorded["App.tsx"].startswith("file_group_budget_exceeded: group_wall_budget_exceeded")
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "file_group_budget_exceeded")
    meta = events[0]["metadata"]
    assert meta["file"] == "App.tsx" and meta["reason"] == "group_wall_budget_exceeded"
    assert meta["items_processed"] == 0 and meta["items_skipped"] == 1
    assert meta["elapsed_s"] >= 600.0


async def test_run_cuts_a_batched_group_at_the_group_deadline(  # (15c/15f)
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """(15c/15f) remaining < scaled call budget: the batch dies at the group deadline, zero fallback."""
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock

    _silence(monkeypatch)
    fake = FakeClock(monotonic_value=10_000.0).install(monkeypatch)
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_WALL_S", 600.0)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)]
    stub.runaway_batched_fix_file = "api.py"
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 200.0
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert isinstance(exit_code, int)

    assert _single_fix_calls_for(stub, "api.py") == []  # zero fallback fixes
    recorded = json.loads((multi_stack_target / ".daydream/deep/fix-failures.json").read_text())
    assert recorded["api.py"].startswith("file_group_budget_exceeded: group_wall_budget_exceeded")
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "file_group_budget_exceeded")
    assert events[0]["metadata"]["reason"] == "group_wall_budget_exceeded"
    assert events[0]["metadata"]["items_processed"] == 0
    assert events[0]["metadata"]["elapsed_s"] >= 600.0
    stop_reasons = _scan_trajectory_extra(multi_stack_target / ".daydream", traj, "stop_reason")
    assert "wall_budget_exceeded" in stop_reasons  # the real budget path, not a stub raise


async def test_run_serial_fallback_runs_under_the_same_group_deadline(  # (15d)
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A batched failure (not a deadline) falls back per-finding, cut at the SAME deadline."""
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock

    _silence(monkeypatch)
    fake = FakeClock(monotonic_value=10_000.0).install(monkeypatch)
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_WALL_S", 600.0)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)]
    stub.fail_batched_fix_file = "api.py"  # batched raises instantly: fallback DOES run
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 200.0  # only the per-finding serial turns burn time
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    assert isinstance(exit_code, int)
    api_singles = _single_fix_calls_for(stub, "api.py")
    group_size = _batched_group_size(stub, "api.py")
    assert 0 < len(api_singles) < group_size  # cut mid-group, not all N
    recorded = json.loads((multi_stack_target / ".daydream/deep/fix-failures.json").read_text())
    assert recorded["api.py"].startswith("file_group_budget_exceeded: group_wall_budget_exceeded")
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "file_group_budget_exceeded")
    meta = events[0]["metadata"]
    # The final fallback turn was aborted mid-stream, so it is NOT progress: the
    # completed count is one less than the dispatched count, and every remaining
    # group member (including the aborted one) is skipped.
    assert meta["items_processed"] == len(api_singles) - 1
    assert meta["items_skipped"] == group_size - meta["items_processed"]
    assert meta["elapsed_s"] >= 600.0


async def test_run_expired_group_does_not_cancel_a_healthy_sibling(  # (15e)
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """One group's deadline stop leaves the sibling group's work intact."""
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock

    _silence(monkeypatch)
    fake = FakeClock(monotonic_value=10_000.0).install(monkeypatch)
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_WALL_S", 600.0)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "App.tsx", "high"), _merge_item(2, "api.py", "high")]
    # Keep api.py a SINGLE-item group: the structural meta-stack's finding would
    # otherwise fold into api.py and make it a two-item batched group, which by
    # design does not take the single-turn runaway branch.
    stub.parse_by_stack = {
        "structure": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "web.ts",
            "line": 1,
            "description": "Structural maintainability concern",
        },
    }
    stub.runaway_single_fix_file = "api.py"  # the group that must expire
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 200.0
    # Hold the expiring group's burst until the sibling's fix turn has fully
    # returned, so the shared clock cannot be burned while the healthy group is
    # still working (the fix-footprint guard removes the stub's sentinels, so the
    # completion list -- not a sentinel file -- is the durable signal).
    stub.runaway_gate = lambda: "App.tsx" in stub.completed_fix_files
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    assert isinstance(exit_code, int)
    assert "App.tsx" in stub.completed_fix_files  # sibling's fix turn returned
    assert len(_single_fix_calls_for(stub, "App.tsx")) == 1  # and its turn completed
    recorded = json.loads((multi_stack_target / ".daydream/deep/fix-failures.json").read_text())
    assert list(recorded) == ["api.py"]  # sibling not blamed
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "file_group_budget_exceeded")
    # The scan sees the workspace trajectory and the top-level copy, but every
    # budget event names ONLY the expired api.py group -- never the sibling.
    assert {e["metadata"]["file"] for e in events} == {"api.py"}
    assert all(e["metadata"]["reason"] == "group_wall_budget_exceeded" for e in events)


async def test_the_configured_allowance_bounds_a_group_s_retry_ladder(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """The file-config value reaches every run_agent call the fix group owns."""
    from daydream.backends.pi import PiError
    from daydream.config_file import load_file_config
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock

    _silence(monkeypatch)
    fake = FakeClock(monotonic_value=100_000.0).install(monkeypatch)
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    (multi_stack_target / ".daydream.toml").write_text(
        "retry_recovery_allowance_s = 25\n", encoding="utf-8"
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    # Keep api.py a SINGLE-item group: the structural meta-stack's finding would
    # otherwise fold into api.py and make it a two-item batched group, whose fix
    # fan-out runs one batched ladder plus per-finding serial fallback ladders
    # (up to three) instead of the single ladder the comments below describe.
    stub.parse_by_stack = {
        "structure": {
            "severity": "medium", "confidence": "MEDIUM",
            "file": "web.ts", "line": 1,
            "description": "Structural maintainability concern",
        },
    }
    stub.fix_retryable_file = "api.py"  # only api.py's ladder may fail: web.ts must succeed, or its
    # retryable failures would share the run-scoped circuit and cut both
    # ladders before the allowance binds
    stub.fix_retryable_failures = 20
    stub.fix_retryable_error = PiError("503 Service Unavailable", retryable=True, category="SERVER_ERROR")
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 20.0     # each retryable attempt burns 20 s of the allowance
    mute_side_effects()
    traj = tmp_path / "trajectory.json"

    with anyio.fail_after(30):
        exit_code = await run(make_config(
            multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        ))

    assert isinstance(exit_code, int)
    stops = _scan_phase_events(multi_stack_target / ".daydream", traj, "agent_budget_stop")
    assert stops, "the ladder stop was not recorded"
    assert any(e["metadata"]["retry_stop_reason"] == "retry_recovery_allowance_exhausted" for e in stops)
    # The original attempt is useful work and is not charged (Task 3's design);
    # each retry is charged 20 s, so the ladder is cut at four dispatches after
    # ~40 s of retries (the refusal fires check-then-charge inside the
    # just-failed attempt's handler). The 300 s default (a missed threading
    # site) would not stretch that to ~17 attempts: the run-scoped circuit
    # (threshold 3) bounds any failing ladder at ~4 dispatches, so the
    # configured value shows up in the stop reason asserted above, not in this
    # attempt count.
    assert max(e["metadata"]["attempts"] for e in stops) <= 4
    assert stub.completed_fix_files.count("api.py") == 0     # the group never applied a fix


async def test_an_outage_circuit_bounds_the_group_fan_out_and_restarts_no_completed_work(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """A run-scoped circuit bounds the fix fan-out's ladders; a completed sibling is not restarted."""
    from daydream.backends.pi import PiError
    from daydream.runner import run
    from tests.harness.fake_clock import FakeClock

    _silence(monkeypatch)
    fake = FakeClock(monotonic_value=100_000.0).install(monkeypatch)
    monkeypatch.setenv("DAYDREAM_PI_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_MAX_DELAY_S", "0")
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # api.py's fix ladder always fails retryably; App.tsx's fix succeeds. Distinct
    # file groups, so both reach the fix fan-out.
    stub.merge_items = [_merge_item(1, "api.py", "high"), _merge_item(2, "App.tsx", "medium")]
    # Keep api.py a SINGLE-item group: the structural meta-stack's finding would
    # otherwise fold into api.py and make it a batched group, whose fallback
    # ladder would multiply the attempts the circuit is meant to bound.
    stub.parse_by_stack = {
        "structure": {
            "severity": "medium", "confidence": "MEDIUM",
            "file": "web.ts", "line": 1,
            "description": "Structural maintainability concern",
        },
    }
    stub.fix_retryable_file = "api.py"
    stub.fix_retryable_failures = 20
    stub.fix_retryable_error = PiError("503 Service Unavailable", retryable=True, category="SERVER_ERROR")
    # Serialize the file groups. A concurrent healthy sibling's success can call
    # record_success and reset the circuit mid-ladder, making the attempt total
    # interleaving-dependent; the integration proof here is that both groups reach
    # the SAME run-scoped circuit and that the circuit bounds the ladder. Task 8's
    # _ending_backend unit test owns the concurrent-share proof.
    stub.fanout_concurrency = 1
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 1.0
    traj = tmp_path / "trajectory.json"
    mute_side_effects()

    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes",
                                         output_mode="loop"))

    assert isinstance(exit_code, int)
    stops = _scan_phase_events(multi_stack_target / ".daydream", traj, "agent_budget_stop")
    # One coordinated ladder, not one per sibling: total dispatches stay at the
    # circuit threshold plus the single probe.
    assert sum(e["metadata"]["attempts"] for e in stops) <= 3 + 1
    assert {e["metadata"]["circuit_state"] for e in stops} <= {"open", "half_open"}
    # The healthy group completed exactly once and was never re-dispatched when the
    # circuit closed or the probe ran.
    assert stub.completed_fix_files.count("App.tsx") == 1
    assert len(_single_fix_calls_for(stub, "App.tsx")) <= 1
