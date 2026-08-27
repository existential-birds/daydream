"""Deep non-wonder budget/truncation integration tests."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

from daydream.runner import RunConfig
from tests.harness.stub_backend import install_stub_backend, silence


def _test_step_stop_reasons(run_root: Path, traj: Path) -> list[str]:
    """stop_reason values on TEST-phase trajectory steps (empty list if none)."""
    values: list[str] = []
    for path in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for step in payload.get("steps", []):
            extra = step.get("extra") or {}
            if extra.get("daydream_phase") == "test":
                reason = extra.get("stop_reason")
                if reason:
                    values.append(str(reason))
    return values


async def test_budget_truncated_stack_lands_in_failed_stacks(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., 'RunConfig'],
    mute_side_effects: Callable[..., None],
) -> None:
    """A truncated per-stack review is recorded as a failure, not a success."""
    from daydream.runner import run

    silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_stack = "python"
    mute_side_effects()
    with anyio.fail_after(30):
        await run(
            make_config(
                multi_stack_target,
                trajectory_path=tmp_path / "trajectory.json",
                assume="yes",
                output_mode="loop",
            )
        )
    failures = json.loads((multi_stack_target / ".daydream" / "deep" / "per-stack-failures.json").read_text())
    assert "python" in failures, failures
    assert "budget" in failures["python"].lower()


async def test_runaway_test_turn_is_bounded_and_reaches_abort(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., 'RunConfig'],
    mute_side_effects: Callable[..., None],
) -> None:
    """A hung test turn is capped, so the run reaches the heal/abort path."""
    from daydream.runner import run

    silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_test = True
    mute_side_effects(heal=False)
    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert exit_code != 0
    assert [c for c in stub.calls if "run the project's test suite" in c["prompt"].lower()]
    test_stop_reasons = _test_step_stop_reasons(multi_stack_target / ".daydream", traj)
    assert any("budget" in r for r in test_stop_reasons), test_stop_reasons
