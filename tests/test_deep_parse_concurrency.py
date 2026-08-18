"""Deep parse fan-out concurrency and truncation integration tests."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from tests.harness.stub_backend import StubBackend, install_stub_backend, silence


async def _run_deep(target: Path) -> int:
    from daydream.runner import RunConfig, run

    return await run(RunConfig(target=str(target), cleanup=False))


def _parse_stack_name(prompt: str) -> str:
    """Stack name a parse prompt points at, via its ``stack-<name>-review.md``."""
    m = re.search(r"stack-(\S+?)-review\.md", prompt)
    return m.group(1) if m else ""


_FIRST_PARSE_STACK = "generic"
_LAST_PARSE_STACK = "structure"


class _RendezvousParseStub(StubBackend):
    """The first and last stack's parse both block until the other has started."""

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.first_parse_started = anyio.Event()
        self.other_parse_started = anyio.Event()

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None,
        continuation: Any = None, agents: Any = None, max_turns: Any = None,
        read_only: bool = False,
    ):
        if "extract only actionable issues" in prompt.lower():
            name = _parse_stack_name(prompt)
            if name == _LAST_PARSE_STACK:
                self.other_parse_started.set()
                await self.first_parse_started.wait()
            elif name == _FIRST_PARSE_STACK:
                self.first_parse_started.set()
                await self.other_parse_started.wait()
        async for event in super().execute(
            cwd, prompt, output_schema=output_schema, continuation=continuation,
            agents=agents, max_turns=max_turns, read_only=read_only,
        ):
            yield event


def _install_raw(
    monkeypatch: pytest.MonkeyPatch, stub: StubBackend,
    install_backend: Callable[[object], object],
) -> None:
    """Install a *caller-supplied custom* stub backend via the shared
    ``install_backend`` fixture, then pin skill availability.

    Unlike ``install_stub_backend`` (which builds its own ``StubBackend`` from a
    target), this installs a custom stub (e.g. a rendezvous or failing stub) at
    the ``create_backend`` seam so tests can observe parse concurrency / failure
    behavior. The two skill-availability pins are identical to
    ``install_stub_backend``'s defaults and keep the test off the local Beagle
    plugin registry.
    """
    install_backend(stub)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)


async def test_parse_runs_concurrently(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
    install_backend: Callable[[object], object],
) -> None:
    """The N per-stack parse calls overlap; every stack's records land on disk."""
    stub = _RendezvousParseStub(multi_stack_target)
    _install_raw(monkeypatch, stub, install_backend)

    with anyio.fail_after(15):
        exit_code = await _run_deep(multi_stack_target)

    assert exit_code == 0
    deep = multi_stack_target / ".daydream" / "deep"
    assert sorted(p.name for p in deep.glob("stack-*-records.json")) == [
        "stack-generic-records.json", "stack-python-records.json",
        "stack-react-records.json", "stack-structure-records.json",
    ]
    parse_order = [_parse_stack_name(c["prompt"]) for c in stub.calls
                   if "extract only actionable issues" in c["prompt"].lower()]
    assert {_FIRST_PARSE_STACK, _LAST_PARSE_STACK} <= set(parse_order), parse_order


class _FailingParseStub(StubBackend):
    """Raises a distinctive error for one stack's parse; siblings succeed."""

    def __init__(self, target: Path, failing_stack: str) -> None:
        super().__init__(target)
        self._failing_stack = failing_stack

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None,
        continuation: Any = None, agents: Any = None, max_turns: Any = None,
        read_only: bool = False,
    ):
        if ("extract only actionable issues" in prompt.lower()
                and _parse_stack_name(prompt) == self._failing_stack):
            raise ZeroDivisionError("parse blew up")
        async for event in super().execute(
            cwd, prompt, output_schema=output_schema, continuation=continuation,
            agents=agents, max_turns=max_turns, read_only=read_only,
        ):
            yield event


async def test_parse_failure_propagates_original_exception_type(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config, install_backend: Callable[[object], object],
) -> None:
    """One stack's parse failure fails the run with its own exception type."""
    stub = _FailingParseStub(multi_stack_target, failing_stack="python")
    _install_raw(monkeypatch, stub, install_backend)
    from daydream.runner import run

    trajectory_path = tmp_path / "trajectory.json"
    with pytest.raises(ZeroDivisionError, match="parse blew up"):
        await run(make_config(multi_stack_target, trajectory_path=trajectory_path))

    deep = multi_stack_target / ".daydream" / "deep"
    survivors = sorted(p.name for p in deep.glob("stack-*-records.json"))
    assert "stack-python-records.json" not in survivors
    assert survivors, "sibling parse results must survive one stack's failure"
    trajectory = json.loads(trajectory_path.read_text())
    dispatch_results = next(
        step["observation"]["results"] for step in trajectory["steps"]
        if step.get("extra", {}).get("daydream_phase") == "parse"
        and step["message"].startswith("Dispatching ")
    )
    linked_stacks = {result["content"].removeprefix("Dispatched to parse-") for result in dispatch_results}
    successful_stacks = {path.removeprefix("stack-").removesuffix("-records.json") for path in survivors}
    assert successful_stacks <= linked_stacks
    for result in dispatch_results:
        reference = result["subagent_trajectory_ref"][0]
        assert (multi_stack_target / ".daydream" / reference["trajectory_path"]).is_file()


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


async def test_budget_truncated_parse_fails_loudly(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config, mute_side_effects,
) -> None:
    """A budget-truncated parse fails the run instead of dropping a stack's findings."""
    from daydream.runner import run

    silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_parse = "python"
    mute_side_effects()
    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        with pytest.raises(RuntimeError, match="budget"):
            await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    run_root = multi_stack_target / ".daydream"
    assert any("budget" in str(v) for v in _scan_trajectory_extra(run_root, traj, "stop_reason"))
    assert not (run_root / "deep" / "stack-python-records.json").exists()


async def test_shard_names_flow_through_parse_and_sort_deterministically(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch,
    make_config,
) -> None:
    """Issue #731 (P2): synthetic ``#`` shard names ride artifact paths and the
    sorted parse/merge ordering deterministically."""
    from daydream.runner import run

    install_stub_backend(monkeypatch, shard_many_python_target)
    deep = shard_many_python_target / ".daydream" / "deep"
    # Shard record files sort stably (python#0 < python#1 ... < structure), so
    # parse fan-out and merge consume them in a fixed order regardless of
    # completion order (sorted iteration at orchestrator.py:1322).
    def shard_names() -> list[str]:
        return sorted(p.name for p in deep.glob("stack-*-records.json"))

    seen: list[list[str]] = []
    for _ in range(2):  # two identical runs -> identical shard set, stable order
        rc = await run(make_config(shard_many_python_target, deep_shard_enabled=True,
                                   deep_shard_max_files=1, deep_shard_max_bytes=10**9))
        assert rc == 0
        names = shard_names()
        assert any(n.startswith("stack-python#") for n in names)
        assert names == sorted(names)   # deterministic by construction
        seen.append(names)
    # The stated determinism claim: both runs must produce the same shard set
    # in the same order. A shard-naming or record-path nondeterminism
    # regression flips this comparison.
    assert seen[0] == seen[1]
