"""Deep wonder concurrency and budget/truncation integration tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from tests.harness.stub_backend import StubBackend, install_stub_backend, silence


async def _run_deep(target: Path) -> int:
    from daydream.runner import RunConfig, run

    return await run(RunConfig(target=str(target), cleanup=False))


def _install_raw(monkeypatch: pytest.MonkeyPatch, stub: StubBackend) -> None:
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kw: stub)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)


def _scan_trajectory_extra(run_root: Path, traj: Path, key: str) -> list[str]:
    values: list[str] = []
    for path in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for step in payload.get("steps", []):
            value = (step.get("extra") or {}).get(key)
            if value:
                values.append(value)
    return values


async def test_tool_heavy_wonder_completes_under_default_budget(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config, mute_side_effects,
) -> None:
    """The default tool-call budget is unlimited, so a tool-heavy wonder pass lands."""
    from daydream.runner import run

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.alternatives_tool_calls = 60
    mute_side_effects()
    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(60):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    assert isinstance(exit_code, int)
    alts = json.loads((multi_stack_target / ".daydream" / "deep" / "alternatives.json").read_text())
    assert [i["title"] for i in alts] == ["Inconsistent greeting wording"]
    assert not any("tool_call_budget" in str(v)
                   for v in _scan_trajectory_extra(multi_stack_target / ".daydream", traj, "stop_reason"))


async def test_root_trajectory_step_ids_survive_concurrent_wonder(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config, mute_side_effects,
) -> None:
    """A wonder turn outliving the per-stack fan-out still writes a valid trajectory."""
    from daydream.runner import run

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.alternatives_tool_calls = 5
    mute_side_effects()
    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(60):
        await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    payload = json.loads(traj.read_text())
    ids = [s["step_id"] for s in payload["steps"]]
    assert ids == list(range(1, len(ids) + 1)), ids
    phases = [(s.get("extra") or {}).get("daydream_phase") for s in payload["steps"]]
    assert "alternatives" in phases, phases


async def test_budget_truncated_wonder_fails_loudly(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config, mute_side_effects,
) -> None:
    """A budget-truncated wonder pass fails the run instead of degrading to []."""
    from daydream.runner import run

    silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_alternatives = True
    mute_side_effects()
    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        with pytest.raises(RuntimeError, match="budget"):
            await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    run_root = multi_stack_target / ".daydream"
    assert any("budget" in str(v) for v in _scan_trajectory_extra(run_root, traj, "stop_reason"))
    alts = run_root / "deep" / "alternatives.json"
    assert not alts.exists() or json.loads(alts.read_text()) != []


class _WonderRendezvousStub(StubBackend):
    """The wonder turn blocks until a per-stack review has started."""

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.per_stack_started = anyio.Event()

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None,
        continuation: Any = None, agents: Any = None, max_turns: Any = None,
        read_only: bool = False,
    ):
        pl = prompt.lower()
        if "you are reviewing the" in pl:
            self.per_stack_started.set()
        elif "would you have done this differently" in pl or "evaluate the implementation" in pl:
            await self.per_stack_started.wait()
        async for event in super().execute(
            cwd, prompt, output_schema=output_schema, continuation=continuation,
            agents=agents, max_turns=max_turns, read_only=read_only,
        ):
            yield event


async def test_wonder_runs_concurrently_with_per_stack(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wonder and the fan-out overlap; alternatives.json lands before parse reads it."""
    silence(monkeypatch)
    _install_raw(monkeypatch, _WonderRendezvousStub(multi_stack_target))
    with anyio.fail_after(15):
        assert await _run_deep(multi_stack_target) == 0
    alts = json.loads((multi_stack_target / ".daydream" / "deep" / "alternatives.json").read_text())
    assert alts, "wonder's artifact must be written before parse consumes it"


async def test_concurrent_per_stack_prompts_omit_alternatives_pointer(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-stack reviewers drop the pointer; adjudication prompts keep it."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    assert await _run_deep(multi_stack_target) == 0
    per_stack = [c["prompt"] for c in stub.calls if "you are reviewing the" in c["prompt"].lower()]
    assert per_stack
    for prompt in per_stack:
        assert "alternatives.json" not in prompt
        assert "intent.md" in prompt
    for phrase in ("you are the arbiter", "cross-stack merge agent"):
        matching = [c["prompt"] for c in stub.calls if phrase in c["prompt"].lower()]
        assert matching, phrase
        assert all("alternatives.json" in p for p in matching), phrase


async def test_single_stack_keeps_serial_order_and_pointer(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-stack mode has no merge agent, so the reviewer pointer must survive."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, tiny_diff_target)
    assert await _run_deep(tiny_diff_target) == 0
    order = [c["prompt"].lower() for c in stub.calls]
    wonder_idx = next(i for i, p in enumerate(order)
                      if "would you have done this differently" in p or "evaluate the implementation" in p)
    per_stack_idx = next(i for i, p in enumerate(order) if "you are reviewing the" in p)
    assert wonder_idx < per_stack_idx, "single-stack mode must stay serial"
    per_stack = [c["prompt"] for c in stub.calls if "you are reviewing the" in c["prompt"].lower()]
    assert per_stack
    assert all("alternatives.json" in p for p in per_stack)


async def test_wonder_failure_fails_run_with_fanout_outputs_on_disk(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held wonder exception is re-raised after the join, original type intact."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.fail_alternatives = True
    with pytest.raises(RuntimeError, match="alternatives blew up"):
        await _run_deep(multi_stack_target)
    reviews = sorted(p.name for p in (multi_stack_target / ".daydream" / "deep").glob("stack-*-review.md"))
    assert reviews, "fan-out outputs must survive for a --start-at resume"
