"""Diagram scheduling contracts through the production runner and real Git.

Existing grounded-diagram tests protect rendering and repair. These cases own
entrypoint defaults, optional-work scheduling, and the lifetime of concurrent
writers. Provider fixtures retain real filesystem behavior; checkpoint fault
cases fail only their exact marker operations.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio
import pytest

from daydream import cli, git_ops, runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends import AgentEvent, BackendExecutionInput, ResultEvent, create_backend as real_create_backend
from daydream.config_file import DaydreamFileConfig
from daydream.github_app import GitHubExecutionInput
from daydream.run_config import RunConfig
from tests.harness import diagram_repos as dr
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git, seed_feature_branch
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import StubBackend, review_stage_state


@pytest.fixture(autouse=True)
def _no_open_pr(fake_gh: FakeGh) -> None:
    fake_gh.set_response("pr-list", value=[])


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
    config = config_for(target, diagram="auto", output_mode="loop", assume="no")
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
    config = config_for(target, diagram="auto", output_mode="loop", assume="no")
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


async def test_checkpoint_publication_failure_keeps_review_complete_and_resume_regenerates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional reuse metadata cannot discard a successfully grounded diagram.

    Existing checkpoint tests cover valid, stale and corrupt persisted inputs;
    this fault targets the actual marker publication after successful generation.
    """
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target)
    install_provider(monkeypatch, backend, 4)
    config = config_for(target, diagram="auto", output_mode="loop", assume="no")
    private_base = tmp_path / "private"
    roots = private_root_locations(base=private_base)
    real_replace = os.replace
    attempted: list[Path] = []

    def fail_marker_rename(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        path = Path(destination)
        if path.name == "diagram-complete.json" and path.is_relative_to(private_base):
            attempted.append(path)
            raise OSError("fixture: checkpoint marker rename failed")
        real_replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr("daydream.json_utils.os.replace", fail_marker_rename)
        assert await runner.run(config, private_roots=roots) == 0
    assert attempted
    assert backend.diagram_turns == {"sequence": 1}
    assert not list(private_base.rglob("diagram-complete.json"))
    assert not (target / ".daydream/deep/diagram-complete.json").exists()
    assert_attached(target)

    resumed = DiagramBarrierBackend(target)
    install_provider(monkeypatch, resumed, 4)
    assert await runner.run(replace(config, start_at="merge"), private_roots=roots) == 0
    assert resumed.diagram_turns == {"sequence": 1}
    assert (target / ".daydream/deep/diagram-complete.json").is_file()
    assert_attached(target)


@pytest.mark.parametrize("eligible", [True, False], ids=["eligible", "ineligible"])
async def test_checkpoint_invalidation_failure_preserves_review_and_diagram_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eligible: bool,
) -> None:
    """A denied recovery-marker unlink cannot fail generation or omission.

    The rename-fault keeper covers publication after generation. This fault
    reaches the earlier invalidation operation, including the no-eligible path.
    """
    if eligible:
        target = dr.build_cross_module_repo(tmp_path)
    else:
        target = tmp_path / "docs-only"
        seed_feature_branch(target, base={"README.md": "# Before\n"}, feature={"README.md": "# After\n"})
    backend = DiagramBarrierBackend(target)
    install_provider(monkeypatch, backend, 4)
    private_base = tmp_path / "private"
    roots = private_root_locations(base=private_base)
    real_unlink = Path.unlink
    attempted: list[Path] = []

    def fail_marker_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "diagram-complete.json" and path.is_relative_to(private_base):
            attempted.append(path)
            raise PermissionError("fixture: checkpoint marker unlink denied")
        real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", fail_marker_unlink)
        assert await runner.run(config_for(target, diagram="auto"), private_roots=roots) == 0
    assert attempted
    assert not list(private_base.rglob("diagram-complete.json"))
    assert not (target / ".daydream/deep/diagram-complete.json").exists()
    coverage = json.loads((target / ".daydream/deep/review-coverage.json").read_text())
    assert coverage["schema_version"] == 2
    assert all(scope["status"] == "complete" for scope in coverage["stack_outcomes"])
    if eligible:
        assert backend.diagram_turns == {"sequence": 1}
        assert_attached(target)
    else:
        assert backend.diagram_turns == {}
        artifact = dr.load_diagram_artifact(target)
        assert all(result["status"] == "skipped" for result in artifact["results"].values())
        for path in (target / ".review-output.md", target / ".daydream/deep/review-output.md"):
            assert "sequenceDiagram" not in path.read_text()


@pytest.mark.parametrize("changed", ["model", "threshold"])
async def test_diagram_resume_regenerates_when_generation_inputs_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    """A completed marker cannot authorize output from another model or policy."""
    target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target)
    install_provider(monkeypatch, backend, 4)
    config = config_for(target, diagram="auto", output_mode="loop", assume="no")
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


@pytest.mark.parametrize("phase", ["supervise", "diagram", "arbiter"])
async def test_embedded_review_preserves_backend_dispatch_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    """Optional planning preserves lazy dispatch and primary failure evidence."""
    if phase == "diagram":
        target = tmp_path / "docs-only"
        seed_feature_branch(target, base={"README.md": "# Before\n"}, feature={"README.md": "# After\n"})
    else:
        target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target)
    backend.diagram_specs = {"sequence": [dr.sequence_spec()]}
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", "4")
    environment = {
        "PATH": os.environ["PATH"], "HOME": str(tmp_path / "embedded-home"),
        "GH_CONFIG_DIR": str(tmp_path / "embedded-gh"), "DAYDREAM_FANOUT_CONCURRENCY": "4",
    }
    backend_input = BackendExecutionInput.from_environment(environment, backend="codex")
    execution = runner.RunnerExecutionInput(
        backend_input, GitHubExecutionInput(git_ops.StaticGitHubAuth(environment)),
    )
    created: list[str] = []

    def create_backend(name: str, *args: Any, **kwargs: Any) -> Any:
        created.append(name)
        if name == "codex":
            assert kwargs["execution_input"] is backend_input
            return backend
        # Keep unsupported/unused backend construction real: the regression
        # fails at its actual execution-input guard rather than a test sentinel.
        return real_create_backend(name, *args, **kwargs)

    monkeypatch.setattr(runner, "create_backend", create_backend)
    config = config_for(target, diagram="auto", file_config=DaydreamFileConfig(
        backend="codex", supervisor="rules", phases={phase: {"backend": "osprey"}},
    ))
    roots = private_root_locations(base=tmp_path / "private")
    if phase == "arbiter":
        # This phase really needs its provider. Fail at ordinary dispatch after
        # preserving primary evidence, rather than during optional preflight.
        with pytest.raises(ValueError, match="explicit BackendExecutionInput is not supported for osprey"):
            await runner.run(config, execution=execution, private_roots=roots)
        deep = target / ".daydream/deep"
        for scope in ("python", "structure"):
            records = json.loads((deep / f"stack-{scope}-records.json").read_text())
            assert records["issues"]
        coverage = json.loads((deep / "review-coverage.json").read_text())
        assert coverage["schema_version"] == 2
        assert coverage["analyzed_revision"]["head_sha"] == git(target, "rev-parse", "HEAD")
        assert {scope["scope_id"]: scope["status"] for scope in coverage["stack_outcomes"]} == {
            "python": "complete", "structure": "complete",
        }
        assert backend.diagram_turns == {}
    else:
        assert await runner.run(config, execution=execution, private_roots=roots) == 0
        assert backend.calls
        assert set(created) == {"codex"}
        if phase == "diagram":
            artifact = dr.load_diagram_artifact(target)
            assert all(result["status"] == "skipped" for result in artifact["results"].values())
            assert backend.diagram_turns == {}
        else:
            assert_attached(target)


async def test_post_diagram_custom_phase_regains_its_own_provider_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any,
) -> None:
    """The review's smaller admission pool must end at the final diagram join."""
    target = dr.build_cross_module_repo(tmp_path)
    review_backend = DiagramBarrierBackend(target, capacity=3)
    entered = anyio.Event()
    release = anyio.Event()
    active = 0
    maximum = 0

    async def respond(*_args: Any, **_kwargs: Any) -> list[ResultEvent]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 4:
            entered.set()
        try:
            await release.wait()
            return [ResultEvent(structured_output=None, continuation=None)]
        finally:
            active -= 1

    later_backend = ScriptedBackend(responder=respond, fanout_concurrency=4)
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", "4")
    monkeypatch.setattr(runner, "create_backend", lambda _name, model=None, **_kwargs:
                        later_backend if model == "later-fixture" else review_backend)
    ext_dir.write_module(
        "import anyio\n"
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.trajectory import DaydreamPhase, get_current_recorder, maybe_fork\n"
        "async def fanout(ctx):\n"
        "    async def child(index):\n"
        "        async with maybe_fork(get_current_recorder(), f'post-diagram-{index}'):\n"
        "            await run_agent(ctx.backend_for('later'), ctx.work.repo, 'POST_DIAGRAM_FANOUT',\n"
        "                            phase=DaydreamPhase.REVIEW, read_only=True, run_context=ctx.run_context)\n"
        "    async with anyio.create_task_group() as group:\n"
        "        for index in range(4):\n"
        "            group.start_soon(child, index)\n"
        "    (ctx.data['dd'] / 'post-diagram-fanout.txt').write_text('four provider streams completed')\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='post-diagram-fanout', run=fanout))\n"
        "    registry.insert_after('deep', anchor='diagram', step='post-diagram-fanout')\n"
    )
    config = config_for(target, diagram="auto", file_config=DaydreamFileConfig(
        phases={"later": {"model": "later-fixture"}},
    ))
    result: list[int] = []

    async def invoke() -> None:
        result.append(await runner.run(config))

    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await entered.wait()
            assert active == 4
            release.set()
    assert result == [0] and maximum == 4 and active == 0
    assert len(later_backend.calls) == 4
    assert (target / ".daydream/deep/post-diagram-fanout.txt").read_text() == "four provider streams completed"
    assert_attached(target)
