"""Diagram scheduling contracts through the production runner and real Git.

Existing grounded-diagram tests protect rendering and repair. These cases own
entrypoint defaults, optional-work scheduling, and the lifetime of concurrent
writers; only the model-provider boundary is replaced.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio
import pytest

from daydream import cli, runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends import AgentEvent
from daydream.config_file import DaydreamFileConfig
from daydream.run_config import RunConfig
from tests.harness import diagram_repos as dr
from tests.harness.git_helpers import bare_remote, git
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import StubBackend, review_stage_state


class DiagramBarrierBackend(StubBackend):
    """Hold real provider streams at independently observable phase boundaries."""

    def __init__(self, target: Path, *, capacity: int = 4, primary_count: int = 2, block: bool = False) -> None:
        super().__init__(target)
        self.fanout_concurrency = capacity
        self.diagram_specs = {"sequence": [dr.sequence_spec()]}
        self.merge_echo_records = True
        self.primary_count = primary_count
        self.block = block
        self.block_synthesis = False
        self.required_marker_root: Path | None = None
        self.synthesis_entered = anyio.Event()
        self.release_synthesis = anyio.Event()
        self.primary_entered = anyio.Event()
        self.diagram_entered = anyio.Event()
        self.release_primary = anyio.Event()
        self.release_diagram = anyio.Event()
        self.diagram_closed = anyio.Event()
        self.primary_keys: set[str] = set()
        self.active: set[str] = set()
        self.maximum_active = 0
        self.active_count = 0
        self.service_delay_s = 0.0
        self.intervals: list[tuple[str, float, float]] = []
        self.events: list[tuple[str, str]] = []

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        state = review_stage_state(prompt)
        lower = prompt.lower()
        diagram = self._diagram_dispatch(lower) is not None
        wonder = "would you have done this differently" in lower or "evaluate the implementation" in lower
        synthesis = "cross-stack merge agent" in lower
        key = "diagram" if diagram else "wonder" if wonder else "synthesis" if synthesis else (
            f"review:{state['scope_id']}" if state is not None else "other"
        )
        primary = wonder or state is not None
        initial = primary and key not in self.primary_keys
        started = perf_counter()
        self.events.append(("start", key))
        self.active.add(key)
        self.active_count += 1
        self.maximum_active = max(self.maximum_active, self.active_count)
        try:
            if initial:
                self.primary_keys.add(key)
                if len(self.primary_keys) == self.primary_count:
                    self.primary_entered.set()
            if diagram:
                if self.required_marker_root is not None:
                    assert list(self.required_marker_root.rglob("custom-diagram.txt"))
                self.diagram_entered.set()
                if self.block:
                    await self.release_diagram.wait()
            elif primary and self.block:
                await self.release_primary.wait()
            elif synthesis:
                self.synthesis_entered.set()
                if self.block_synthesis:
                    await self.release_synthesis.wait()
            if self.service_delay_s and (primary or diagram):
                await anyio.sleep(self.service_delay_s)
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
        finally:
            self.active.discard(key)
            self.active_count -= 1
            self.intervals.append((key, started, perf_counter()))
            self.events.append(("end", key))
            if diagram:
                self.diagram_closed.set()


def config_for(target: Path, **overrides: Any) -> RunConfig:
    fields: dict[str, Any] = {
        "target": str(target), "base": "main", "output_mode": "review", "non_interactive": True,
        "cleanup": False, "archive": False, "run_eval": False, "review_cache_enabled": False,
        "shallow_fanout_threshold": 0,
    }
    fields.update(overrides)
    return RunConfig(**fields)


def install_provider(monkeypatch: pytest.MonkeyPatch, backend: DiagramBarrierBackend, capacity: int) -> None:
    # Synthetic provider streams consume no model/CLI resources. Keep the global
    # resource restriction intact outside this test's monkeypatch scope.
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", str(capacity))
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)


def assert_attached(target: Path) -> None:
    artifact = dr.load_diagram_artifact(target)
    assert artifact["results"]["sequence"]["status"] == "rendered"
    for path in (target / ".review-output.md", target / ".daydream/deep/review-output.md"):
        report = path.read_text()
        assert report.count("## Diagrams") == 1
        assert report.count("sequenceDiagram") == 1
        assert "## Issues" in report
    coverage = json.loads((target / ".daydream/deep/review-coverage.json").read_text())
    assert all(scope["status"] == "complete" for scope in coverage["stack_outcomes"])


@pytest.mark.parametrize("opt_in", [False, True])
def test_cli_diagram_opt_in_reaches_real_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opt_in: bool,
) -> None:
    """CLI defaults survive detached checkout selection and source publication."""
    target = dr.build_cross_module_repo(tmp_path)
    origin = bare_remote(tmp_path / "origin.git")
    git(target, "remote", "add", "origin", str(origin))
    git(target, "push", "origin", "main", "feature")
    source_head = git(target, "rev-parse", "HEAD")
    backend = DiagramBarrierBackend(target)
    install_provider(monkeypatch, backend, 1)
    argv = ["daydream", "--review", "--worktree", "--base", "main", "--non-interactive", "--no-archive"]
    if opt_in:
        argv.extend(["--diagram", "auto"])
    monkeypatch.setattr(sys, "argv", [*argv, str(target)])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 0
    assert git(target, "rev-parse", "HEAD") == source_head
    assert backend.calls
    provider_cwds = {Path(call["cwd"]).resolve() for call in backend.calls}
    assert len(provider_cwds) == 1
    provider_cwd = provider_cwds.pop()
    assert provider_cwd != target.resolve()
    assert not provider_cwd.exists(), "the detached operational checkout must be cleaned up"
    assert backend.diagram_entered.is_set() is opt_in
    assert (target / ".daydream/deep/diagram.json").exists() is opt_in


async def test_diagram_overlaps_independent_primary_review_then_attaches_after_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serial diagram dispatch deadlocks at this barrier on the pre-fix runner."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target, primary_count=3, block=True)
    install_provider(monkeypatch, backend, 4)
    result: list[int] = []

    async def invoke() -> None:
        result.append(await runner.run(config_for(
            target, diagram="auto", review_profile=independent_alternatives_profile(),
        )))

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await backend.primary_entered.wait()
            await backend.diagram_entered.wait()
            assert {"review:python", "review:structure", "wonder", "diagram"} <= backend.active
            # Complete the diagram before synthesis is allowed to create reports.
            backend.release_diagram.set()
            await backend.diagram_closed.wait()
            backend.release_primary.set()
    assert result == [0]
    assert backend.maximum_active <= 4
    assert not backend.active
    assert_attached(target)


@pytest.mark.parametrize("capacity", [1, 2])
async def test_constrained_capacity_preserves_primary_review_and_final_diagram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capacity: int,
) -> None:
    """Optional generation must not add a hidden provider slot under saturation."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target, capacity=capacity, block=capacity == 2)
    install_provider(monkeypatch, backend, capacity)
    if capacity == 2:
        backend.release_diagram.set()
        result: list[int] = []

        async def invoke() -> None:
            result.append(await runner.run(config_for(target, diagram="auto")))

        with anyio.fail_after(30):
            async with anyio.create_task_group() as group:
                group.start_soon(invoke)
                await backend.primary_entered.wait()
                assert backend.active == {"review:python", "review:structure"}
                assert not backend.diagram_entered.is_set()
                backend.release_primary.set()
        assert result == [0]
    else:
        assert await runner.run(config_for(target, diagram="auto")) == 0
    assert backend.maximum_active <= capacity
    diagram_start = backend.events.index(("start", "diagram"))
    primary_ends = [index for index, (event, key) in enumerate(backend.events)
                    if event == "end" and key.startswith("review:")]
    assert primary_ends and max(primary_ends) < diagram_start
    assert_attached(target)


async def test_cancellation_joins_diagram_writer_and_preserves_resumable_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optional writer cannot outlive cancellation or corrupt a later resume."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target, block=True)
    install_provider(monkeypatch, backend, 4)
    config = config_for(target, diagram="auto")
    caught: list[BaseException] = []
    private_base = tmp_path / "private"
    roots = private_root_locations(base=private_base)

    async def invoke() -> None:
        try:
            await runner.run(config, private_roots=roots)
        except BaseException as exc:
            caught.append(exc)
            raise

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await backend.primary_entered.wait()
            await backend.diagram_entered.wait()
            backend.release_primary.set()
            # Provider completion is not durable completion: wait for the real
            # records that the merge resume consumes, while yielding the loop.
            while not list(private_base.rglob("review-coverage.json")):
                await anyio.sleep(0)
            group.cancel_scope.cancel()
    assert len(caught) == 1 and isinstance(caught[0], anyio.get_cancelled_exc_class())
    assert backend.diagram_closed.is_set() and not backend.active
    assert backend.diagram_turns == {}
    resumed = DiagramBarrierBackend(target)
    install_provider(monkeypatch, resumed, 4)
    assert await runner.run(replace(config, start_at="per-stack"), private_roots=roots) == 0
    assert_attached(target)


async def test_diagram_finishing_after_synthesis_is_attached_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final join must attach a diagram that was absent during synthesis."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target, block=True)
    backend.release_primary.set()
    install_provider(monkeypatch, backend, 4)
    private_base = tmp_path / "private"
    roots = private_root_locations(base=private_base)
    result: list[int] = []

    async def invoke() -> None:
        result.append(await runner.run(config_for(target, diagram="auto"), private_roots=roots))

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await backend.diagram_entered.wait()
            await backend.synthesis_entered.wait()
            reports: list[Path] = []
            while not reports:
                reports = list(private_base.rglob("deep/review-output.md"))
                await anyio.sleep(0)
            assert "sequenceDiagram" not in reports[0].read_text()
            backend.release_diagram.set()
    assert result == [0]
    assert_attached(target)


@pytest.mark.parametrize("corrupt_payload", [False, True])
async def test_completed_diagram_survives_cancellation_with_validated_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_payload: bool,
) -> None:
    """Persisted generation is independent of an interrupted synthesis turn."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target, block=True)
    backend.block_synthesis = True
    install_provider(monkeypatch, backend, 4)
    config = config_for(target, diagram="auto")
    private_base = tmp_path / "private"
    roots = private_root_locations(base=private_base)
    caught: list[BaseException] = []

    async def invoke() -> None:
        try:
            await runner.run(config, private_roots=roots)
        except BaseException as exc:
            caught.append(exc)
            raise

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await backend.primary_entered.wait()
            await backend.diagram_entered.wait()
            backend.release_diagram.set()
            while not list(private_base.rglob("diagram-complete.json")):
                await anyio.sleep(0)
            backend.release_primary.set()
            await backend.synthesis_entered.wait()
            if corrupt_payload:
                # Change actual session-private bytes after completion, keeping
                # valid JSON so only the checkpoint digest can reject reuse.
                markers = list(private_base.rglob("diagram-complete.json"))
                assert len(markers) == 1
                payload = markers[0].with_name("diagram.json")
                payload.write_bytes(payload.read_bytes() + b"\n")
            group.cancel_scope.cancel()
    assert len(caught) == 1 and isinstance(caught[0], anyio.get_cancelled_exc_class())
    assert not backend.active
    resumed = DiagramBarrierBackend(target)
    install_provider(monkeypatch, resumed, 4)
    assert await runner.run(replace(config, start_at="merge"), private_roots=roots) == 0
    assert resumed.diagram_turns == ({"sequence": 1} if corrupt_payload else {})
    assert_attached(target)


@pytest.mark.parametrize("changed", ["model", "threshold"])
async def test_diagram_resume_regenerates_when_generation_inputs_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    """A completed marker cannot authorize output from another model or policy."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target)
    install_provider(monkeypatch, backend, 4)
    config = config_for(target, diagram="auto")
    assert await runner.run(config) == 0
    resumed = DiagramBarrierBackend(target)
    install_provider(monkeypatch, resumed, 4)
    options: dict[str, Any]
    if changed == "model":
        resumed.model = "changed-model"
        options = {"model": "changed-model"}
    else:
        options = {"file_config": DaydreamFileConfig(diagram_min_modules=1)}
    assert await runner.run(replace(config, start_at="per-stack", **options)) == 0
    assert resumed.diagram_turns == {"sequence": 1}
    assert_attached(target)


@pytest.mark.parametrize("placement", ["before", "after"])
async def test_custom_deep_flow_retains_its_explicit_diagram_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, placement: str,
) -> None:
    """Extensions keep the named diagram phase's input and completion boundary."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target)
    private_base = tmp_path / "private"
    roots = private_root_locations(base=private_base)
    if placement == "before":
        backend.required_marker_root = private_base
    install_provider(monkeypatch, backend, 4)
    observation = (
        "    report = (dd / 'review-output.md').read_text()\n"
        "    assert ctx.data['diagrams']['results']['sequence']['status'] == 'rendered'\n"
        "    assert report.count('sequenceDiagram') == 1\n" if placement == "after" else ""
    )
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "async def observe(ctx):\n"
        "    dd = ctx.data['dd']\n"
        + observation
        + "    (dd / 'custom-diagram.txt').write_text('explicit ordering retained')\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='observe-diagram', run=observe))\n"
        f"    registry.insert_{placement}('deep', anchor='diagram', step='observe-diagram')\n"
    )
    assert await runner.run(config_for(target, diagram="auto"), private_roots=roots) == 0
    assert (target / ".daydream/deep/custom-diagram.txt").read_text() == "explicit ordering retained"
    assert_attached(target)


@pytest.mark.parametrize("capacity", [1, 4])
async def test_matched_diagram_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capacity: int,
) -> None:
    """Matched provider service permits baseline/fix latency measurement.

    Sleep models provider service, never a rendezvous or pass/fail threshold.
    Run this exact node on both revisions with the same pytest/coverage options.
    """
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target, capacity=capacity)
    invalid = dr.sequence_spec()
    invalid["messages"][0]["evidence"]["file"] = "pkg_b/ghost.py"
    backend.diagram_specs = {"sequence": [invalid, dr.sequence_spec()]}
    backend.diagram_session_id = "matched-diagram-session"
    backend.service_delay_s = 0.05
    install_provider(monkeypatch, backend, capacity)
    started = perf_counter()
    assert await runner.run(config_for(target, diagram="auto")) == 0
    published = perf_counter()
    primary_ends = [end for key, _start, end in backend.intervals
                    if key.startswith("review:") or key == "wonder"]
    diagram_calls = [call for call in backend.calls if backend._diagram_dispatch(call["prompt"].lower())]
    repairs = [call for call in diagram_calls if "diagram repair turn" in call["prompt"].lower()]
    assert backend.diagram_turns == {"sequence": 2}
    assert len(diagram_calls) == 2 and len(repairs) == 1
    assert backend.maximum_active <= capacity
    assert_attached(target)
    print(json.dumps({
        "fixture": "matched-diagram-workload-v1", "capacity": capacity,
        "service_delay_s": backend.service_delay_s, "review_completed_s": max(primary_ends) - started,
        "final_publication_s": published - started, "maximum_active": backend.maximum_active,
        "diagram_calls": len(diagram_calls), "repair_calls": len(repairs), "provider_fixture_tokens": 0,
        "provider_intervals": [{"role": key, "start_s": start - started, "end_s": end - started}
                               for key, start, end in backend.intervals if key != "other"],
    }, sort_keys=True))
