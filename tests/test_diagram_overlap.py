"""Diagram scheduling contracts through the production runner and real Git.

Existing grounded-diagram tests protect rendering and repair. These cases own
entrypoint defaults, optional-work scheduling, and the lifetime of concurrent
writers. Provider fixtures retain real filesystem behavior; checkpoint fault
cases fail only their exact marker operations.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import anyio
import pytest

from daydream import git_ops, runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends import AgentEvent, BackendExecutionInput, create_backend as real_create_backend
from daydream.config_file import DaydreamFileConfig
from daydream.github_app import GitHubExecutionInput
from daydream.run_config import RunConfig
from tests.harness import diagram_repos as dr
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git, seed_feature_branch
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
    assert await runner.run(replace(config, start_at="merge"), private_roots=roots) == 0
    assert resumed.diagram_turns == {"sequence": 1}
    assert_attached(target)


async def test_diagram_checkpoint_accepts_only_a_matching_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching publication restores without a provider; changed bytes miss; unavailable metadata only warns."""
    from daydream.deep import diagram_steps
    from daydream.deep.artifacts import DeepArtifact
    from daydream.deep.diagram_checkpoint import complete_diagram_checkpoint, restore_diagram_checkpoint
    from daydream.deep.diagram_trigger import Eligibility, KindDecision
    from daydream.deep.diagram_types import DiagramThresholds
    from daydream.deep.state import DeepState
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.workspace import WorkContext

    work = WorkContext(repo=tmp_path, source=tmp_path, base_branch="main", base_sha="", head_branch=None,
                       head_sha="head", is_ephemeral=False, run_id="checkpoint")
    ctx = FlowContext(config=config_for(tmp_path, start_at="merge"), work=work, registry=Registry(),
                      data={"dd": tmp_path})
    payload = DeepArtifact.DIAGRAM.at(tmp_path)
    complete_diagram_checkpoint(ctx, "key")
    assert "Diagram recovery checkpoint unavailable (FileNotFoundError)" in capsys.readouterr().out
    assert not DeepArtifact.DIAGRAM_COMPLETE.at(tmp_path).exists()
    payload.write_text('{"results": {}}')
    complete_diagram_checkpoint(ctx, "key")
    assert restore_diagram_checkpoint(ctx, "key") == {"results": {}}
    assert restore_diagram_checkpoint(ctx, "other-key") is None
    # The durable step restores that publication on resume instead of constructing a backend.
    monkeypatch.setattr(diagram_steps, "diagram_checkpoint_key", lambda *_a, **_k: "key")
    eligible = Eligibility(code_files=["a.py"], modules={"a.py": "a"}, services={}, cross_module_edges=1,
        function_branch_counts=[], candidate_roots=[], thresholds=DiagramThresholds(), force="auto",
        sequence=KindDecision(eligible=True, rule="cross-module", reason="test"),
        flowchart=KindDecision(eligible=False, rule=None, reason="test"))
    inputs = diagram_steps.DiagramInputs(diagram_steps.DiagramSettings("auto", DiagramThresholds(), []), eligible, {})
    assert await diagram_steps._run_diagram_step(ctx, phase=cast(Any, None), inputs=inputs) is None
    restored = DeepState(ctx.data).diagrams
    assert restored is not None and restored["results"] == {}
    assert "restored completed generation" in capsys.readouterr().out
    payload.write_text('{"results": {}}\n')
    assert restore_diagram_checkpoint(ctx, "key") is None


@pytest.mark.parametrize("phase", ["diagram", "arbiter"])
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
