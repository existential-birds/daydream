"""Diagram scheduling contracts through the production runner and real Git.

Existing grounded-diagram tests protect rendering and repair. These cases own
entrypoint defaults, optional-work scheduling, and the lifetime of concurrent
writers. Provider fixtures retain real filesystem behavior.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import git_ops, runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends import AgentEvent, BackendExecutionInput, create_backend as real_create_backend
from daydream.config_file import DaydreamFileConfig
from daydream.github_app import GitHubExecutionInput
from daydream.review_result import FinalizedReview
from daydream.run_config import FrozenAuthorDescription, RunConfig
from tests.harness import diagram_repos as dr
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.stub_backend import StubBackend, force_interactive, review_stage_state, silence


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
        self.primary_entered = anyio.Event()
        self.diagram_entered = anyio.Event()
        self.release_primary = anyio.Event()
        self.release_diagram = anyio.Event()
        self.diagram_closed = anyio.Event()
        self.primary_keys: set[str] = set()
        self.active: set[str] = set()

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        state = review_stage_state(prompt)
        lower = prompt.lower()
        diagram = self._diagram_dispatch(lower) is not None
        wonder = "would you have done this differently" in lower or "evaluate the implementation" in lower
        key = "diagram" if diagram else "wonder" if wonder else (
            f"review:{state['scope_id']}" if state is not None else "other"
        )
        primary = wonder or state is not None
        initial = primary and key not in self.primary_keys
        self.active.add(key)
        try:
            if initial:
                self.primary_keys.add(key)
                if len(self.primary_keys) == self.primary_count:
                    self.primary_entered.set()
            if diagram:
                self.diagram_entered.set()
                if self.block:
                    await self.release_diagram.wait()
            elif primary and self.block:
                await self.release_primary.wait()
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
        finally:
            self.active.discard(key)
            if diagram:
                self.diagram_closed.set()


def config_for(target: Path, **overrides: Any) -> RunConfig:
    fields: dict[str, Any] = {
        "target": str(target), "base": "main", "output_mode": "review", "non_interactive": True,
        "cleanup": False, "archive": False, "run_eval": False, "shallow_fanout_threshold": 0,
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
    silence(monkeypatch)
    force_interactive(monkeypatch)
    config = config_for(target, diagram="auto", output_mode="loop", non_interactive=False)
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


@pytest.mark.parametrize("phase", ["diagram", "arbiter"])
async def test_embedded_review_preserves_backend_dispatch_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], phase: str,
) -> None:
    """Optional planning preserves lazy dispatch and primary failure evidence."""
    if phase == "diagram":
        target = tmp_path / "docs-only"
        seed_feature_branch(target, base={"README.md": "# Before\n"}, feature={"README.md": "# After\n"})
    else:
        target = dr.build_cross_module_repo(tmp_path)
    backend = DiagramBarrierBackend(target)
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
        backend="codex", phases={phase: {"backend": "osprey"}},
    ))
    roots = private_root_locations(base=tmp_path / "private")
    delivered: list[FinalizedReview] = []
    exit_code = await runner.run(config, execution=execution, private_roots=roots,
                                 review_result_handler=delivered.append)
    if phase == "arbiter":
        # The fresh challenge needs native host tools: the review is refused before any backend,
        # workspace or artifact exists, and nothing is delivered.
        assert exit_code == 1
        assert "'osprey' does not serve them" in capsys.readouterr().out
        # So is embedder-supplied author text over its bound, or frozen for another head than the approved one.
        head = git(target, "rev-parse", "HEAD")
        supported = replace(config, file_config=DaydreamFileConfig(backend="codex"), approved_head_sha=head)
        for description, refusal in ((FrozenAuthorDescription(head, "!" * 65_531), "exceeds its encoded UTF-8 bound"),
                                     (FrozenAuthorDescription("0" * 40, "intent"), "does not match analyzed HEAD")):
            assert await runner.run(replace(supported, frozen_author_description=description), execution=execution,
                                    private_roots=roots, review_result_handler=delivered.append) == 1
            assert refusal in " ".join(capsys.readouterr().out.split())
        assert created == [] and delivered == [] and backend.calls == []
        assert not (target / ".daydream").exists() and not (tmp_path / "private").exists()
    else:
        assert exit_code == 0
        assert set(created) == {"codex"}
        artifact = dr.load_diagram_artifact(target)
        assert all(result["status"] == "skipped" for result in artifact["results"].values())
        assert backend.diagram_turns == {}
        # The embedded callback receives the one sealed result and the projected canonical items.
        (review,) = delivered
        assert review.terminal_result["analysis_state"] == "complete"
        assert review.terminal_result["analyzed_revision"]["head_sha"] == git(target, "rev-parse", "HEAD")
        assert [scope["scope_id"] for scope in review.terminal_result["planned_scopes"]] == ["committed:00000"]
        (item,) = review.items
        assert {key: item[key] for key in ("id", "file", "line", "description")} == {
            "id": 1, "file": "README.md", "line": 1, "description": "Sample issue"}
        assert item["item_uid"] and item["rationale"].startswith("The changed behavior is incorrect.")
        assert not {"severity", "confidence", "lens"} & item.keys()
        assert json.loads((target / ".daydream/deep/merged-items.json").read_text())["items"] == review.items
