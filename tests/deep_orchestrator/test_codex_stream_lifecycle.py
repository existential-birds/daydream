"""Real runner/Git isolation acceptance for synthetic external Codex stalls."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import anyio
import pytest
from rich.text import Text

from daydream import runner
from daydream.agent import run_agent
from daydream.backends import Backend, create_backend
from daydream.backends.codex import CodexBackend
from daydream.config_file import DaydreamFileConfig
from daydream.trajectory import DaydreamPhase, DispatchHandle, dispatch_scope, maybe_fork
from tests.harness.codex_lifecycle_cli import FIX_MARKER, LifecycleMode, install_codex_lifecycle_cli
from tests.harness.processes import wait_for_process_group_exit
from tests.harness.trajectory import make_recorder, read_trajectory, root_trajectory
from tests.test_deep_orchestrator import MakeConfig, _silence


@pytest.mark.parametrize("drain_mode", ["terminal_open_stdout", "descendant_stdout"])
async def test_run_isolates_three_codex_fixers_and_retains_terminal_evidence(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    drain_mode: LifecycleMode,
) -> None:
    """A real drain/reap stall restores its group and cannot discard healthy edits.

    The executable barrier requires all three fix processes to be live before
    any answers complete. The actual runner factory creates one cached backend.
    """
    target = multi_stack_target
    _silence(monkeypatch)
    fixture = install_codex_lifecycle_cli(tmp_path / "codex", fix_modes={
        "App.tsx": "healthy", "api.py": drain_mode, "README.md": "terminal_eof_live_child",
    })
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_CODEX_TERMINAL_DRAIN_TIMEOUT_S", "0.2")
    monkeypatch.setenv("DAYDREAM_CODEX_PROCESS_REAP_TIMEOUT_S", "0.2")
    monkeypatch.setenv("DAYDREAM_CODEX_CLEANUP_TIMEOUT_S", "1")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "0")
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    created: list[Backend] = []
    factory = create_backend

    def observe_factory(*args: Any, **kwargs: Any) -> Backend:
        backend = factory(*args, **kwargs)
        created.append(backend)
        return backend

    monkeypatch.setattr(runner, "create_backend", observe_factory)
    before = {file: (target / file).read_bytes() for file in ("App.tsx", "api.py", "README.md")}
    with anyio.fail_after(25):
        code = await runner.run(make_config(
            target, backend="codex", model="fixture-model", reasoning_effort="high", assume="yes",
            shallow_fanout_threshold=0, review_cache_enabled=False,
            file_config=DaydreamFileConfig(quality_gate_enabled=False),
        ))
    assert code == 1
    assert len(created) == 1 and isinstance(created[0], CodexBackend)
    assert (target / "App.tsx").read_bytes() == before["App.tsx"] + FIX_MARKER.encode()
    assert (target / "api.py").read_bytes() == before["api.py"]
    assert (target / "README.md").read_bytes() == before["README.md"]
    failures = json.loads((target / ".daydream/deep/fix-failures.json").read_text())
    assert set(failures) == {"api.py", "README.md"}
    assert "terminal_drain" in failures["api.py"]
    assert "process_reap" in failures["README.md"]
    assert all("wall_budget" not in reason for reason in failures.values())
    observations = [row for row in fixture.observations() if row["file"]]
    assert {row["file"] for row in observations} == set(before)
    assert len(observations) == 3  # No replay of a completed mutating turn.
    assert len({row["cwd"] for row in observations}) == 3
    assert all(Path(row["cwd"]) != target and not Path(row["cwd"]).exists() for row in observations)
    root = root_trajectory(target)
    refs = [ref for ref in root["extra"]["subtrajectories"]
            if ref.get("descriptor", "").startswith("fix-")]
    assert len(refs) == 3
    assert len({ref["trajectory_id"] for ref in refs}) == 3
    children = {ref["descriptor"].removeprefix("fix-"):
                read_trajectory(target / ".daydream" / ref["sibling_trajectory_ref"]) for ref in refs}
    for file, child in children.items():
        serialized = json.dumps(child)
        assert "Synthetic initial commentary" in serialized
        assert f"Synthetic final answer for {file}" in serialized
        assert any(step.get("tool_calls") for step in child["steps"])
        assert child["final_metrics"]["total_prompt_tokens"] == 10
        assert child["final_metrics"]["total_completion_tokens"] == 5
        invocations = child["extra"]["subtrajectories"]
        assert len(invocations) == 1
        lifecycle = invocations[0]["backend_lifecycle"]
        assert lifecycle["terminal_received_s"] is not None
        assert lifecycle["recorder_event_count"] > 0
        assert lifecycle["consumer_event_count"] > 0
        assert lifecycle["teardown_completed_s"] is not None
        if file == "App.tsx":
            assert lifecycle["process_returncode"] == 0
            assert lifecycle["expiry_stage"] is None
            assert "failed" not in invocations[0].get("status", "")
        else:
            assert lifecycle["expiry_stage"] == ("terminal_drain" if file == "api.py" else "process_reap")
        if file == "README.md":
            assert lifecycle["stdout_eof_s"] is not None
        if file == "api.py" and drain_mode == "descendant_stdout":
            assert lifecycle["process_returncode"] == 0
            assert lifecycle["process_exit_s"] is not None
    for row in observations:
        await wait_for_process_group_exit(row["pid"])


async def test_suspended_host_consumers_reproduce_missing_later_public_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly suspended public callback explains an evidence pattern.

    All synthetic children complete and exit zero. This control establishes a
    possible boundary, not the cause of the historical October 5 incident.
    """
    fixture = install_codex_lifecycle_cli(tmp_path / "codex")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    backend = CodexBackend(model="fixture-model")
    recorder = make_recorder(tmp_path)
    results: dict[int, tuple[Any, Any, Any]] = {}

    async def worker(ordinal: int, dispatch: DispatchHandle | None) -> None:
        suspended = False

        async def suspend_initial(text: Text) -> None:
            nonlocal suspended
            if "Synthetic initial commentary" in text.plain and not suspended:
                suspended = True
                await asyncio.Event().wait()

        async with maybe_fork(recorder, f"consumer-{ordinal}", dispatch=dispatch):
            results[ordinal] = await run_agent(
                backend, tmp_path, "synthetic lifecycle request", phase=DaydreamPhase.FIX,
                wall_budget_s=2.0, progress_callback=suspend_initial,
            )

    async with recorder:
        async with dispatch_scope(recorder, phase=DaydreamPhase.FIX,
                                  descriptors=[f"consumer-{ordinal}" for ordinal in range(3)]) as dispatch, \
                anyio.create_task_group() as group:
            for ordinal in range(3):
                group.start_soon(worker, ordinal, dispatch)
    assert len(results) == 3
    assert all(result[2] == "wall_budget_exceeded" for result in results.values())
    assert all(result[0] == "Synthetic initial commentary" for result in results.values())
    root = read_trajectory(recorder.path)
    assert len(root["extra"]["subtrajectories"]) == 3
    for ref in root["extra"]["subtrajectories"]:
        child = read_trajectory(recorder.path.parent / ref["sibling_trajectory_ref"])
        assert "Synthetic initial commentary" in json.dumps(child)
        assert "Synthetic lifecycle response" not in json.dumps(child)
        assert not any(step.get("tool_calls") for step in child["steps"])
        assert child["final_metrics"].get("total_prompt_tokens") in (None, 0)
        lifecycle = child["extra"]["subtrajectories"][0]["backend_lifecycle"]
        assert lifecycle["terminal_received_s"] is None
        assert lifecycle["process_returncode"] == 0
        assert lifecycle["process_exit_s"] is not None
        assert lifecycle["loop_sample_count"] > 0
        assert lifecycle["last_loop_sample_s"] > lifecycle["last_consumer_event_s"]
        assert lifecycle["last_native_event_kind"] == "item_completed"
    observations = fixture.observations()
    assert len(observations) == 3 and all(row["stage"] == "terminal" for row in observations)
    for row in observations:
        await wait_for_process_group_exit(row["pid"])
