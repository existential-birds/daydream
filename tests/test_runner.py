"""Tests for daydream.runner.RunConfig and the unified ``run`` dispatch."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import anyio
import pytest

from daydream import git_ops, runner
from daydream.archive.git_context import GitContext
from daydream.archive.manifest import (
    Manifest,
    archive_recorder_provenance_from_snapshot,
    build_manifest_from_snapshot,
)
from daydream.backends import (
    AUDIT_ROOT_ISOLATION_V1,
    AgentEvent,
    AuditIsolationError,
    Backend,
    ResultEvent,
    TextEvent,
)
from daydream.exploration import ExplorationContext
from daydream.extensions.loader import build_registry
from daydream.flows.engine import FlowContext
from daydream.runner import RunConfig
from daydream.trajectory import (
    DaydreamRunFlow,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
)
from daydream.workspace import AuditWorkspace, WorkContext
from tests.harness.backend import ScriptedBackend, Turn
from tests.harness.git_helpers import bare_remote
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
from tests.harness.remote_ci import NoCIRemote
from tests.test_deep_pr_comment_integration import (
    FakeAssistantMessage,
    FakeResultMessage,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeToolResultBlock,
    FakeToolUseBlock,
    FakeUserMessage,
    _answer_prompts,
    _FakeSDKClient,
    _silence_ui,
)


@pytest.fixture
def deep_target(tmp_path: Path) -> Path:
    """Real git repo on a feature branch with one Python file changed.

    Mirrors ``tests/test_deep_pr_comment_integration.py``'s fixture so the
    real-path App-identity test drives the identical single-file deep path
    (tier ``"skip"``) with the shared fake SDK.
    """
    repo = tmp_path / "deep_repo"
    _init_repo(repo)
    (repo / "foo.py").write_text("def foo():\n    return 1\n")
    _git(repo, "add", ".")
    _commit(repo, "init")
    _git(repo, "checkout", "-b", "feature")
    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    _git(repo, "add", ".")
    _commit(repo, "tweak foo")
    return repo


@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every SDK symbol that ``ClaudeBackend.execute`` does isinstance on."""
    for symbol, fake in (
        ("ClaudeSDKClient", _FakeSDKClient),
        ("AssistantMessage", FakeAssistantMessage),
        ("UserMessage", FakeUserMessage),
        ("ResultMessage", FakeResultMessage),
        ("TextBlock", FakeTextBlock),
        ("ThinkingBlock", FakeThinkingBlock),
        ("ToolUseBlock", FakeToolUseBlock),
        ("ToolResultBlock", FakeToolResultBlock),
    ):
        monkeypatch.setattr(f"daydream.backends.claude.{symbol}", fake)

_RESULT = ResultEvent(structured_output=None, continuation=None)


@pytest.mark.parametrize("flow_name", [None, "deep", "shallow"])
async def test_unborn_non_improve_runner_fails_before_backend(
    tmp_path: Path, make_config: Callable[..., RunConfig], flow_name: str | None,
) -> None:
    repo = tmp_path / "unborn"
    _init_repo(repo)
    (repo / "staged.py").write_text("value = 1\n")
    _git(repo, "add", "staged.py")
    before = git_ops.staged_patch(repo)
    config = make_config(repo, flow_name=flow_name)
    assert await runner.run(config) == 1
    assert git_ops.is_unborn_head(repo)
    assert git_ops.staged_patch(repo) == before


async def test_unborn_improve_approved_head_rejected_before_backend(
    tmp_path: Path, make_config: Callable[..., RunConfig],
) -> None:
    repo = tmp_path / "unborn"
    _init_repo(repo)
    (repo / "staged.py").write_text("value = 1\n")
    _git(repo, "add", "staged.py")
    before = git_ops.staged_patch(repo)
    config = make_config(
        repo, flow_name="improve", approved_head_sha="a" * 40,
    )
    assert await runner.run(config) == 1
    assert git_ops.is_unborn_head(repo)
    assert git_ops.staged_patch(repo) == before


# A failing test run, then the heal fix agent's turn.
_FAIL_TURN: tuple[AgentEvent, ...] = (TextEvent(text="1 failed, 0 passed"), _RESULT)
_FIX_TURN: tuple[AgentEvent, ...] = (TextEvent(text="Applied fix attempt"), _RESULT)
# Raised if the heal loop calls the backend past its script -- the bounded-loop guard.
_BEYOND_SCRIPT: Turn = (AssertionError("backend invoked beyond scripted call count"),)


def _handoff_turn(body: str) -> Turn:
    """The read-only failure-summarizer's structured handoff response."""
    return (ResultEvent(structured_output={"handoff_prompt": body}, continuation=None),)


def test_run_config_rejects_unsupported_exploration_depth() -> None:
    with pytest.raises(TypeError, match="exploration_depth"):
        RunConfig(exploration_depth=2)  # type: ignore[call-arg]


def test_run_config_diagram_defaults_to_unset_not_auto() -> None:
    """#1113 (D2): ``None`` is the unset marker, so a file-config
    ``[tool.daydream.diagram] mode = "off"`` can win over the built-in default
    while an explicit CLI value still overrides the file. A ``"auto"`` default
    would make the file-level ``off`` unreachable.
    """
    assert RunConfig().diagram is None
    assert RunConfig(diagram="both").diagram == "both"


def test_run_config_has_no_skill_availability_field() -> None:
    """M1: RunConfig no longer carries installed-skill availability."""
    assert not hasattr(RunConfig(), "skill_availability")


def test_run_config_has_no_bot_field() -> None:
    """M2: RunConfig no longer carries the feedback-mode ``bot`` field."""
    cfg = RunConfig(target="/tmp")
    assert not hasattr(cfg, "bot")


def test_feedback_routes_to_review_shim_not_feedback() -> None:
    """M2: numeric targets no longer select feedback; no feedback entry point exists."""
    assert not hasattr(runner, "run_feedback")


def test_run_config_exploration_context_defaults_to_none() -> None:
    cfg = RunConfig()
    assert cfg.exploration_context is None
    explicit = ExplorationContext()
    cfg2 = RunConfig(exploration_context=explicit)
    assert cfg2.exploration_context is explicit


_VALID_DOCUMENT_BYTES = json.dumps(
    {"session_id": "session", "trajectory_id": "session", "steps": [], "final_metrics": {}, "extra": {}}
).encode()


@pytest.fixture
def capture_recorder(tmp_path: Path) -> TrajectoryRecorder:
    """A bare recorder identifying the ``session`` run for capture tests."""
    return TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test",
        session_id="session",
    )


def _capture_snapshot(
    tmp_path: Path,
    status: Literal["complete", "partial"],
    *,
    json_bytes: bytes = _VALID_DOCUMENT_BYTES,
    root: str = "session",
    cutoff_at: str = "2026-09-06T00:00:00Z",
) -> RunWriteSnapshot:
    """One single-document run-write snapshot whose path is never written."""
    return RunWriteSnapshot(
        status=status,
        cutoff_at=cutoff_at,
        root_trajectory_id=root,
        documents=(
            TrajectoryDocumentSnapshot(
                trajectory_id="session",
                path=tmp_path / f"missing.{'json' if status == 'complete' else 'partial'}",
                json_bytes=json_bytes,
            ),
        ),
    )


def test_run_write_capture_retains_valid_final_without_io_and_records_invalid(
    tmp_path: Path, capture_recorder: TrajectoryRecorder
) -> None:
    """The recorder callback is a non-raising immutable handoff, not finalization."""
    from daydream.runner import _RunSnapshotCaptureError, _RunWriteCapture

    final = _capture_snapshot(tmp_path, "complete")
    capture = _RunWriteCapture(session_id="session")

    capture.retain(capture_recorder, final)
    assert capture.final is final
    assert capture.partial is None
    assert capture.validation_error is None
    assert not final.documents[0].path.exists()

    capture.retain(capture_recorder, _capture_snapshot(tmp_path, "partial", root="other"))
    assert capture.final is final
    assert isinstance(capture.validation_error, _RunSnapshotCaptureError)


def test_run_write_capture_closes_ordinary_json_validation_failure(
    tmp_path: Path, capture_recorder: TrajectoryRecorder
) -> None:
    """The synchronous recorder callback never leaks an ordinary parser error."""
    from daydream.runner import _RunSnapshotCaptureError, _RunWriteCapture

    partial = _capture_snapshot(tmp_path, "partial")
    capture = _RunWriteCapture(session_id="session")
    capture.retain(capture_recorder, partial)

    # Ordinary parser failure, interpreter-independent: a truncated document
    # raises json.JSONDecodeError on every supported Python. (Deep nesting is
    # NOT usable here — CPython 3.14's rewritten JSON scanner no longer
    # recurses in C, so 10k-deep but well-formed JSON parses cleanly there and
    # only 3.12/3.13 raise RecursionError.)
    capture.retain(
        capture_recorder,
        _capture_snapshot(
            tmp_path,
            "complete",
            json_bytes=b'{"session_id":"session","trajectory_id":"session","value":[0',
            cutoff_at="2026-09-06T00:00:01Z",
        ),
    )

    assert capture.partial is partial
    assert capture.final is None
    assert isinstance(capture.validation_error, _RunSnapshotCaptureError)
    assert "JSONDecodeError" in str(capture.validation_error)


def test_run_write_capture_does_not_swallow_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_recorder: TrajectoryRecorder
) -> None:
    """Cancellation-class failures remain authoritative at the callback boundary."""
    from daydream.runner import _RunWriteCapture

    capture = _RunWriteCapture(session_id="session")

    def interrupt(_payload: bytes) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr("daydream.runner.json.loads", interrupt)

    with pytest.raises(KeyboardInterrupt):
        capture.retain(capture_recorder, _capture_snapshot(tmp_path, "partial", json_bytes=b"{}"))
    assert capture.partial is None
    assert capture.final is None
    assert capture.validation_error is None


def test_flow_context_exposes_typed_artifact_session_without_data_fallback(
    make_work: Callable[[Path], WorkContext],
    tmp_path: Path,
) -> None:
    """Task 4 receives the host session explicitly, never through ctx.data."""
    from daydream.artifact_visibility import ArtifactSession
    from daydream.extensions import get_registry

    sentinel = cast(ArtifactSession, object())
    ctx = FlowContext(
        config=RunConfig(target=str(tmp_path)),
        work=make_work(tmp_path),
        registry=get_registry(),
        artifacts=sentinel,
    )

    assert ctx.artifacts is sentinel
    assert "artifacts" not in ctx.data


def test_findings_preparation_diagnostic_does_not_expose_private_write_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The producer acknowledges preparation without disclosing host storage."""
    from daydream.pr_review import PRInfo

    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n")
    _git(repo, "add", "app.py")
    _commit(repo, "base")
    private_path = tmp_path / "private" / "runtime" / "secret" / "findings.json"
    monkeypatch.setattr(
        "daydream.pr_review.find_pr_by_number",
        lambda *_args, **_kwargs: PRInfo(
            number=7, head_sha="a" * 40, base_sha="b" * 40, base_ref="main", head_ref="feature",
            owner="owner", repo="repo", url="https://example.invalid/owner/repo/pull/7",
        ),
    )

    result = runner._write_findings_for_parsed(
        repo, RunConfig(pr_number=7, findings_out=str(private_path)), []
    )

    output = capsys.readouterr().out
    assert result == 0
    assert private_path.is_file()
    assert "Findings artifact prepared." in output
    assert str(private_path) not in output


def _feature_repo(tmp_path: Path, *, remote: bool = False) -> Path:
    """Real git repo with one committed change on ``feature`` over ``main``."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n")
    _git(repo, "add", "app.py")
    _commit(repo, "base")
    if remote:
        _git(repo, "remote", "add", "origin", str(bare_remote(tmp_path / "origin.git")))
        _git(repo, "push", "-u", "origin", "main")
    _git(repo, "checkout", "-b", "feature")
    (repo / "app.py").write_text("VALUE = 2\n")
    _git(repo, "add", "app.py")
    _commit(repo, "feature")
    return repo


def _write_probe_flow(ext_dir: Any, flow_name: str) -> None:
    """Register a one-step flow whose step makes one real ``run_agent`` call."""
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def _probe(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        "    assert ctx.private_workspace_owner is not None\n"
        "    await run_agent(ctx.backend_for('probe'), ctx.work.repo, 'PROBE',"
        " phase=DaydreamPhase.REVIEW)\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='probe', run=_probe))\n"
        f"    registry.set_flow({flow_name!r}, ['probe'])\n"
    )


class _ControlledBackend:
    """In-process backend with the stream behaviors the host boundary needs.

    ``mode`` picks what the stream does once entered: ``yield`` streams a normal
    turn, ``block`` waits for ``release``/``cancel``, ``hang`` sleeps forever,
    and ``raise`` raises *error* mid-iteration (the trailing ``yield`` keeps this
    an async generator, so a real backend's failure shape is preserved).
    """

    model = "controlled-model"

    def __init__(
        self,
        mode: Literal["yield", "block", "hang", "raise"] = "yield",
        *,
        error: BaseException | None = None,
    ) -> None:
        self.mode = mode
        self.error = error
        self.entered = anyio.Event()
        self.release = anyio.Event()
        self.cancelled = False
        self.cwd: Path | None = None

    async def execute(self, cwd: Path, *_args: Any, **_kwargs: Any) -> AsyncIterator[AgentEvent]:
        self.cwd = cwd
        self.entered.set()
        if self.mode == "raise":
            assert self.error is not None
            raise self.error
        if self.mode == "block":
            await self.release.wait()
        elif self.mode == "hang":
            await anyio.sleep_forever()
        yield TextEvent(text="controlled output")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        self.cancelled = True
        self.release.set()


async def _run_private(config: RunConfig, tmp_path: Path) -> int:
    """Drive the real ``runner.run`` against a per-test private artifact root."""
    from daydream.artifact_visibility import private_root_locations

    return await runner.run(
        config, private_roots=private_root_locations(base=tmp_path / "private")
    )


def _assert_one_published_run(repo: Path, archive_dir: Path) -> tuple[Path, Path]:
    """Exactly one public run and one archived run; return both directories."""
    public_runs = list((repo / ".daydream" / "runs").iterdir())
    archived_runs = list((archive_dir / "runs").iterdir())
    assert len(public_runs) == len(archived_runs) == 1
    return public_runs[0], archived_runs[0]


def _assert_partial_evidence_published(repo: Path, archive_dir: Path) -> None:
    """A joined-but-failed run publishes partial evidence under both roots."""
    public_run, archived_run = _assert_one_published_run(repo, archive_dir)
    assert json.loads((public_run / "trajectory.json").read_text())["extra"]["partial"] is True
    assert json.loads((archived_run / "manifest.json").read_text())["archive_status"] == "partial"


@pytest.mark.parametrize("failure_mode", ["none", "destination", "archive"])
async def test_artifact_session_runner_controlled_custom_flow_publishes_after_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    archive_dir: Path,
    failure_mode: str,
) -> None:
    """A real custom flow stays private until its real agent invocation joins."""
    repo = _feature_repo(tmp_path)
    _write_probe_flow(ext_dir, "artifact-probe")
    backend = _ControlledBackend("block")
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    if failure_mode == "archive":
        (repo / ".review-output.md").write_bytes(b"operator baseline\x00")
        from daydream.archive import ArchiveFinalizationError

        def fail_archive(**_kwargs: Any) -> None:
            raise ArchiveFinalizationError("injected strict archive failure")

        monkeypatch.setattr("daydream.archive.finalize_archive_run", fail_archive)
    external_trajectory = tmp_path / "external trajectory.json"
    external_trajectory.write_text("operator baseline\n", encoding="utf-8")
    config = RunConfig(
        target=str(repo),
        base="main",
        flow_name="artifact-probe",
        trajectory_path=external_trajectory,
        run_eval=False,
        archive=True,
        non_interactive=True,
    )
    result: list[int] = []

    async def invoke() -> None:
        result.append(await _run_private(config, tmp_path))

    with anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await backend.entered.wait()
            assert not (repo / ".daydream").exists()
            assert external_trajectory.read_text(encoding="utf-8") == "operator baseline\n"
            assert list((tmp_path / "private" / "runtime").glob("*/runs/*/live/.daydream/diff.patch"))
            if failure_mode == "destination":
                replacement = external_trajectory.with_name("replacement.tmp")
                replacement.write_text("concurrent replacement\n", encoding="utf-8")
                os.replace(replacement, external_trajectory)
            backend.release.set()

    if failure_mode != "none":
        assert result == [1]
        assert not list((archive_dir / "runs").glob("*"))
        if failure_mode == "destination":
            assert external_trajectory.read_text(encoding="utf-8") == "concurrent replacement\n"
            assert not (repo / ".daydream").exists()
        else:
            assert external_trajectory.read_text(encoding="utf-8") == "operator baseline\n"
            assert (repo / ".review-output.md").read_bytes() == b"operator baseline\x00"
            assert not (repo / ".daydream" / "runs").exists()
        return

    assert result == [0]
    public_run, archived_run = _assert_one_published_run(repo, archive_dir)
    archived_bytes = (archived_run / "trajectory.json").read_bytes()
    assert (public_run / "trajectory.json").read_bytes() == archived_bytes
    assert external_trajectory.read_bytes() == archived_bytes
    assert json.loads((archived_run / "manifest.json").read_text())["session_id"] == public_run.name


async def test_forced_ephemeral_runner_records_source_while_backend_uses_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, archive_dir: Path
) -> None:
    """Root/fork provenance is stable source identity, not the deleted model cwd."""
    repo = _feature_repo(tmp_path, remote=True)
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.trajectory import DaydreamPhase, get_current_recorder\n"
        "async def _probe(ctx):\n"
        "    recorder = get_current_recorder()\n"
        "    assert recorder is not None\n"
        "    await run_agent(ctx.backend_for('probe'), ctx.work.repo, 'ROOT', "
        "phase=DaydreamPhase.REVIEW)\n"
        "    async with recorder.fork('source-probe'):\n"
        "        await run_agent(ctx.backend_for('probe'), ctx.work.repo, 'PROBE', "
        "phase=DaydreamPhase.REVIEW)\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='probe', run=_probe))\n"
        "    registry.set_flow('source-probe', ['probe'])\n"
    )
    backend = _ControlledBackend()
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)

    result = await _run_private(
        RunConfig(
            target=str(repo),
            base="main",
            flow_name="source-probe",
            force_worktree=True,
            run_eval=True,
            archive=True,
            non_interactive=True,
        ),
        tmp_path,
    )

    assert result == 0
    assert backend.cwd is not None
    assert backend.cwd != repo.resolve()
    assert not backend.cwd.exists()
    public_run_dir, run_dir = _assert_one_published_run(repo, archive_dir)
    payloads = [
        json.loads(path.read_text())
        for path in (run_dir / "trajectory.json", *sorted((run_dir / "trajectories").glob("*.json")))
    ]
    assert len(payloads) == 2
    assert {payload["extra"]["target_dir"] for payload in payloads} == {str(repo.resolve())}
    assert str(backend.cwd) not in json.dumps(payloads)
    assert (public_run_dir / "trajectory.json").read_bytes() == (
        run_dir / "trajectory.json"
    ).read_bytes()
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["quality"]["scoped_files"] == 1
    assert list(evaluation["quality"]["per_file"]) == ["app.py"]
    assert evaluation["daydream_dir"] == str(repo.resolve() / ".daydream")
    assert str(backend.cwd) not in json.dumps(evaluation)


@pytest.mark.parametrize("finalizer_interrupt", [False, True])
async def test_artifact_session_runner_preserves_primary_and_publishes_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    archive_dir: Path,
    finalizer_interrupt: bool,
) -> None:
    """A complete recorder write cannot turn a failed body into host success."""
    repo = _feature_repo(tmp_path)
    _write_probe_flow(ext_dir, "artifact-probe-error")
    primary = RuntimeError("model boundary failed")
    monkeypatch.setattr(
        runner, "create_backend", lambda *_a, **_k: _ControlledBackend("raise", error=primary)
    )
    if finalizer_interrupt:
        monkeypatch.setattr(
            "daydream.archive.finalize_archive_run",
            lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    with pytest.raises(RuntimeError) as raised:
        await _run_private(
            RunConfig(
                target=str(repo),
                base="main",
                flow_name="artifact-probe-error",
                run_eval=False,
                archive=True,
                non_interactive=True,
            ),
            tmp_path,
        )

    assert raised.value is primary
    if finalizer_interrupt:
        assert not (repo / ".daydream").exists()
        assert not list((archive_dir / "runs").glob("*"))
        assert any("secondary base failure" in note for note in primary.__notes__)
        return
    _assert_partial_evidence_published(repo, archive_dir)


async def test_artifact_session_runner_cancellation_finalizes_then_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, archive_dir: Path
) -> None:
    """Cancellation joins the backend, durably publishes evidence, then escapes."""
    repo = _feature_repo(tmp_path)
    _write_probe_flow(ext_dir, "artifact-probe-cancel")
    backend = _ControlledBackend("hang")
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    config = RunConfig(
        target=str(repo),
        base="main",
        flow_name="artifact-probe-cancel",
        run_eval=False,
        archive=True,
        non_interactive=True,
    )
    caught: list[BaseException] = []

    async def invoke() -> None:
        try:
            await _run_private(config, tmp_path)
        except BaseException as exc:
            caught.append(exc)
            raise

    async with anyio.create_task_group() as group:
        group.start_soon(invoke)
        await backend.entered.wait()
        assert not (repo / ".daydream").exists()
        group.cancel_scope.cancel()

    assert len(caught) == 1
    assert isinstance(caught[0], anyio.get_cancelled_exc_class())
    assert backend.cancelled is True
    _assert_partial_evidence_published(repo, archive_dir)


async def test_signal_flush_immutable_cutoff_before_first_root_step(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    make_config: Callable[..., RunConfig],
) -> None:
    """Initial exploration fan-out archives a rooted immutable T1 snapshot."""
    from daydream.artifact_visibility import private_root_locations
    from daydream.atif import validate as atif_validate
    from daydream.cli import _signal_handler
    from daydream.ui import get_shutdown_panel, set_shutdown_panel
    from tests.harness.stub_backend import StubBackend, silence

    class InitialExplorationBarrierBackend(StubBackend):
        fanout_concurrency = 2

        def __init__(self, target: Path) -> None:
            super().__init__(target)
            self.entered = anyio.Event()
            self.release = anyio.Event()
            self.entered_count = 0
            self.active_count = 0

        async def execute(
            self,
            cwd: Path,
            prompt: str,
            output_schema: Any = None,
            continuation: Any = None,
            agents: Any = None,
            max_turns: Any = None,
            read_only: bool = False,
        ) -> AsyncIterator[AgentEvent]:
            is_initial_specialist = "specialist" in prompt.lower() and self.entered_count < 2
            if is_initial_specialist:
                self.entered_count += 1
                self.active_count += 1
                if self.entered_count == 2:
                    self.entered.set()
                try:
                    await self.release.wait()
                finally:
                    self.active_count -= 1
            async for event in super().execute(
                cwd,
                prompt,
                output_schema=output_schema,
                continuation=continuation,
                agents=agents,
                max_turns=max_turns,
                read_only=read_only,
            ):
                yield event

    # Four changed files select pre_scan's parallel tier; the backend's real
    # fan-out capacity admits exactly two children while the third waits.
    (multi_stack_target / "extra.py").write_text("EXTRA = 1\n", encoding="utf-8")
    _git(multi_stack_target, "add", "extra.py")
    _commit(multi_stack_target, "add fourth changed file")

    fake_bin = multi_stack_target.parent / "signal-bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/bin/sh\n"
        "if [ \"$*\" = \"api /user\" ]; then\n"
        "  printf '%s\\n' '{\"login\":\"signal-runner\"}'\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"unexpected gh call: $*\" >&2\n"
        "exit 91\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(fake_bin), os.environ.get("PATH", ""))))

    backend = InitialExplorationBarrierBackend(multi_stack_target)
    silence(monkeypatch)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", True)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    clock_tick = 0

    def deterministic_now() -> str:
        nonlocal clock_tick
        clock_tick += 1
        return f"2026-01-01T00:00:00.{clock_tick:06d}Z"

    monkeypatch.setattr("daydream.trajectory.now_iso", deterministic_now)
    outcome: dict[str, int] = {}
    finished = anyio.Event()
    private_base = multi_stack_target.parent / "signal-private"

    async def run_review() -> None:
        try:
            outcome["exit_code"] = await runner.run(
                make_config(
                    multi_stack_target,
                    flow_name="review",
                    archive=True,
                    run_eval=True,
                    diagram="off",
                ),
                private_roots=private_root_locations(base=private_base),
            )
        finally:
            finished.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_review)
        with anyio.fail_after(10):
            await backend.entered.wait()
        assert backend.active_count == 2

        with pytest.raises(KeyboardInterrupt):
            _signal_handler(signal.SIGINT, None)
        panel = get_shutdown_panel()
        if panel is not None:
            panel.finish()
            set_shutdown_panel(None)

        # Task 3 routes the recorder privately. Deep's pre-recorder sidecars
        # remain Task 4 producer work, so only the public run directory is
        # forbidden at this checkpoint.
        assert not (multi_stack_target / ".daydream" / "runs").exists()
        live_runs = [
            path
            for path in (private_base / "runtime").glob("*/runs/*/live/.daydream/runs/*")
            if path.is_dir()
        ]
        assert len(live_runs) == 1
        live_run = live_runs[0]
        partial_paths = sorted(live_run.rglob("*.partial"))
        assert len(partial_paths) == 3
        partial_bytes = {path.relative_to(live_run): path.read_bytes() for path in partial_paths}
        partial_payloads = [json.loads(value) for value in partial_bytes.values()]
        assert all(atif_validate(payload, validate_images=False) for payload in partial_payloads)
        t1_values = {payload["extra"]["snapshot_at"] for payload in partial_payloads}
        assert len(t1_values) == 1
        t1 = t1_values.pop()

        root_partial = json.loads(partial_bytes[Path("trajectory.json.partial")])
        assert root_partial["trajectory_id"] == live_run.name
        assert root_partial["steps"] == [
            {
                "step_id": 1,
                "timestamp": t1,
                "source": "system",
                "message": "Daydream run snapshot",
                "extra": {
                    "daydream_run_flow": "ttt",
                    "host_event": "partial_snapshot",
                },
            }
        ]
        assert all("run_ended_at" not in payload["extra"] for payload in partial_payloads)
        partial_merge_events = [
            event
            for event in root_partial["extra"]["phase_events"]
            if event["phase"] == "exploration"
        ]
        assert [event["event"] for event in partial_merge_events] == ["phase_start"]

        archived_run = archive_dir / "runs" / live_run.name
        assert not archived_run.exists()

        backend.release.set()
        with anyio.fail_after(20):
            await finished.wait()

    assert outcome == {"exit_code": 0}
    assert backend.active_count == 0
    assert all((live_run / relative).read_bytes() == value for relative, value in partial_bytes.items())
    final_root = json.loads((live_run / "trajectory.json").read_text())
    assert final_root["extra"]["run_ended_at"] > t1
    assert all(step["message"] != "Daydream run snapshot" for step in final_root["steps"])
    exploration_events = [
        event for event in final_root["extra"]["phase_events"] if event["phase"] == "exploration"
    ]
    assert [event["event"] for event in exploration_events] == ["phase_start", "phase_end"]
    assert exploration_events[0]["scope_id"] == exploration_events[1]["scope_id"]
    assert exploration_events[1]["status"] == "succeeded"
    assert exploration_events[1]["timestamp"] > t1
    assert any(
        step.get("extra", {}).get("dispatch_status") == "succeeded"
        and step.get("extra", {}).get("daydream_phase") == "exploration"
        for step in final_root["steps"]
    )
    final_manifest = json.loads((archive_dir / "runs" / live_run.name / "manifest.json").read_text())
    assert final_manifest["archive_status"] == "complete"


# --- Stage 4.1b dispatch tests ---------------------------------------------


@pytest.fixture
def patch_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_work: Callable[..., WorkContext]
) -> Any:
    """Stub ``open_workspace`` and the in-place fallback so dispatch tests
    keep their synthetic ``WorkContext`` while exercising the real artifact
    lease around dispatch.
    """
    from daydream.artifact_visibility import private_root_locations

    _init_repo(tmp_path)
    work = make_work(tmp_path)
    locations = private_root_locations(base=tmp_path.parent / f"{tmp_path.name}-private")

    @asynccontextmanager
    async def _fake_open_workspace(*_args: Any, **_kwargs: Any) -> AsyncIterator[WorkContext]:
        yield work

    monkeypatch.setattr("daydream.runner.open_workspace", _fake_open_workspace)
    monkeypatch.setattr("daydream.runner.private_root_locations", lambda: locations)
    # Force the in-place fallback off so every call goes through the fake CM.
    monkeypatch.setattr("daydream.runner.git_ops.is_inside_worktree", lambda _p: True)
    return work


@pytest.fixture
def silence_runner_ui(silence_console: Callable[..., None]) -> None:
    """Drop ``daydream.runner``'s UI helpers (notably the ``print_phase_hero``
    banner). ``daydream.deep.orchestrator``'s own ``print_phase_hero`` /
    ``print_dim`` bindings are deliberately left live: the AWAKEN-hero ordering
    test spies on them via ``daydream.phases``.
    """
    silence_console("daydream.runner")


_DISPATCH_TARGETS = (
    "_run_loop_deep",
    "_run_improve",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_target", "config_kwargs", "expected_attr", "expected_value"),
    [
        # A PR number is metadata and does not select a separate dispatch path.
        ("_run_loop_deep", {"pr_number": 42}, "pr_number", 42),
        ("_run_loop_deep", {"output_mode": "comment"}, "output_mode", "comment"),
        ("_run_loop_deep", {"output_mode": "review"}, "output_mode", "review"),
        # Issue #1113: diagram-only joins comment/review on the branch that
        # skips ``_require_reviewable_branch`` -- it neither fixes nor commits.
        (
            "_run_loop_deep",
            {"output_mode": "diagram", "diagram": "sequence"},
            "output_mode",
            "diagram",
        ),
        ("_run_loop_deep", {"output_mode": "loop", "shallow": True}, "shallow", True),
        # Stage 4.2: deep is the default. No flags required to route here.
        ("_run_loop_deep", {"output_mode": "loop"}, "shallow", False),
        ("_run_improve", {"flow_name": "improve"}, "flow_name", "improve"),
    ],
    ids=[
        "pr_number_metadata_goes_deep",
        "comment_mode",
        "review_mode",
        "diagram_only_mode",
        "shallow_mode",
        "deep_loop_by_default",
        "improve_flow",
    ],
)
async def test_run_dispatches_to_expected_flow(
    expected_target: Any,
    config_kwargs: Any,
    expected_attr: Any,
    expected_value: Any,
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace: Any,
    silence_runner_ui: None,  # noqa: F841
    tmp_path: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """``run()`` routes each flag combination to exactly one flow entrypoint.

    Every dispatch function is stubbed, so the recorded call list also proves
    exclusivity: every PR-process mode (comment / review / shallow / deep loop)
    lands on the single deep flow, while ``pr_number`` remains metadata only.
    """
    called: list[tuple[str, WorkContext, RunConfig]] = []

    def _record(name: str) -> Any:
        async def stub(work: Any, config: Any, _run_artifacts: Any = None) -> int:
            called.append((name, work, config))
            return 0

        return stub

    for name in _DISPATCH_TARGETS:
        monkeypatch.setattr(f"daydream.runner.{name}", _record(name))

    config = make_config(tmp_path, **config_kwargs)

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert [name for name, _work, _config in called] == [expected_target]
    _name, work, seen_config = called[0]
    assert work is patch_workspace
    observed = getattr(seen_config, expected_attr)
    assert observed == expected_value and type(observed) is type(expected_value)


@pytest.mark.asyncio
async def test_run_rejects_head_mismatch_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace: Any,
    silence_runner_ui: None,  # noqa: F841
    tmp_path: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """Head drift: run() returns 1 and no flow is dispatched."""
    called: list[str] = []

    def _record(name: str) -> Any:
        async def stub(work: Any, config: Any, _run_artifacts: Any = None) -> int:
            called.append(name)
            return 0

        return stub

    for name in _DISPATCH_TARGETS:
        monkeypatch.setattr(f"daydream.runner.{name}", _record(name))

    config = make_config(tmp_path, approved_head_sha="DEADBEEF")
    exit_code = await runner.run(config)
    assert exit_code == 1
    assert called == []


@pytest.mark.asyncio
async def test_run_allows_matching_approved_head(
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace: Any,
    silence_runner_ui: None,  # noqa: F841
    tmp_path: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """Matching approved head: run() proceeds to the expected flow."""
    called: list[str] = []

    def _record(name: str) -> Any:
        async def stub(work: Any, config: Any, _run_artifacts: Any = None) -> int:
            called.append(name)
            return 0

        return stub

    for name in _DISPATCH_TARGETS:
        monkeypatch.setattr(f"daydream.runner.{name}", _record(name))

    config = make_config(tmp_path, approved_head_sha="CAFEBABE")
    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called == ["_run_loop_deep"]


@pytest.mark.asyncio
async def test_run_rejects_head_mismatch_on_real_worktree(
    monkeypatch: pytest.MonkeyPatch,
    silence_runner_ui: None,  # noqa: F841
    deep_target: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """Head drift on a real checkout: run() returns 1 and no flow is dispatched.

    Unlike the stub-based gate tests above (``patch_workspace`` yields a
    synthetic ``WorkContext`` with a hardcoded fake ``head_sha``), this drives
    ``open_workspace`` -> ``git_ops.head_sha`` for real, so a regression in
    the workspace plumbing surfaces as a failing gate instead of a silent
    no-op on real checkouts.
    """
    called: list[str] = []

    def _record(name: str) -> Any:
        async def stub(work: Any, config: Any, _run_artifacts: Any = None) -> int:
            called.append(name)
            return 0

        return stub

    for name in _DISPATCH_TARGETS:
        monkeypatch.setattr(f"daydream.runner.{name}", _record(name))

    config = make_config(deep_target, approved_head_sha="DEADBEEF")
    exit_code = await runner.run(config)
    assert exit_code == 1
    assert called == []


@pytest.mark.asyncio
async def test_run_allows_matching_approved_head_on_real_worktree(
    monkeypatch: pytest.MonkeyPatch,
    silence_runner_ui: None,  # noqa: F841
    deep_target: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """Matching approved head on a real checkout: run() proceeds to the flow.

    The dispatch stub records the ``WorkContext`` the real ``open_workspace``
    built, proving ``work.head_sha`` came from ``git rev-parse HEAD`` on the
    actual repo and not from a synthetic context.
    """
    called: list[str] = []
    head_shas: list[str] = []

    def _record(name: str) -> Any:
        async def stub(work: Any, config: Any, _run_artifacts: Any = None) -> int:
            called.append(name)
            head_shas.append(work.head_sha)
            return 0

        return stub

    for name in _DISPATCH_TARGETS:
        monkeypatch.setattr(f"daydream.runner.{name}", _record(name))

    real_head = _git(deep_target, "rev-parse", "HEAD").strip()
    config = make_config(deep_target, approved_head_sha=real_head)
    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called == ["_run_loop_deep"]
    assert head_shas == [real_head]


@pytest.mark.asyncio
@pytest.mark.parametrize("flow_name", [None, "deep"], ids=["default_deep", "explicit_deep"])
async def test_deep_run_mints_app_identity_before_posting_path(
    flow_name: str | None,
    monkeypatch: pytest.MonkeyPatch,
    deep_target: Path,
    patch_sdk: None,
) -> None:
    """Real-path: deep runs mint the App token before their PR-posting path.

    Drives ``daydream.runner.run`` end-to-end in deep mode — both the default
    dispatch (``flow_name=None``) and an explicit ``--flow deep`` — on a real
    temp git worktree with GitHub App credentials set. Only the external
    network/API seams are mocked: the App installation-token mint, the Claude
    SDK transport (``patch_sdk``), and the final ``gh`` PR-posting transport
    (``find_open_pr`` + ``_submit_review``). The real deep orchestrator,
    ``ClaudeBackend.execute``, every phase, ``_post``, ``classify``, and
    ``build_payload`` run unmodified.

    Asserts the observable outcome rather than a stubbed loop: the App token
    is minted before the posting path is reached, ``config.identity`` resolves
    to the App bot identity, and the minted token is injected as ``GH_TOKEN``
    into every ``gh`` subprocess for the duration of the run.
    """
    from daydream import pr_review
    from daydream.runner import RunConfig

    _silence_ui(monkeypatch)
    _answer_prompts(monkeypatch)

    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "test-private-key")

    events: list[str] = []
    payloads: list[dict[str, Any]] = []

    def fake_mint(*_args: object) -> SimpleNamespace:
        events.append("mint")
        return SimpleNamespace(
            token="installation-token",
            identity="daydream-review[bot]",
            expires_at=4_102_444_800.0,
        )

    fake_pr = pr_review.PRInfo(
        number=123,
        head_sha="0" * 40,
        base_sha="1" * 40,
        base_ref="main",
        head_ref="feature",
        owner="test-owner",
        repo="test-repo",
        url="https://example/pr/123",
    )

    def fake_find_open_pr(_target_dir: object) -> pr_review.PRInfo:
        events.append("find-open-pr")
        return fake_pr

    def fake_submit_review(
        _target_dir: object, _pr: object, payload: dict[str, Any]
    ) -> tuple[str, None]:
        events.append("post")
        payloads.append(payload)
        return "https://example/pr/123#review-1", None

    monkeypatch.setattr("daydream.github_app._mint_installation_token", fake_mint)
    monkeypatch.setattr("daydream.pr_review.find_open_pr", fake_find_open_pr)
    monkeypatch.setattr("daydream.pr_review._submit_review", fake_submit_review)

    config = RunConfig(
        target=str(deep_target),
        flow_name=flow_name,
        pr_repo="acme/widgets",
        cleanup=False,
        archive=False,
    )

    rc = await runner.run(config)

    assert rc == 0, f"run() returned {rc}"
    # Mint strictly precedes the posting path: find_open_pr is _post()'s first
    # action, and the review is submitted only after classify + build_payload.
    assert events == ["mint", "find-open-pr", "post"], events
    assert config.identity == "daydream-review[bot]"
    assert git_ops.get_gh_token_env() == {"GH_TOKEN": "installation-token"}
    assert payloads, "deep flow never reached _submit_review"


@pytest.mark.asyncio
async def test_review_run_does_not_mint_app_identity(
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace: WorkContext,
    silence_runner_ui: None,  # noqa: F841  # noqa
    tmp_path: Path,
    make_config: Callable[..., RunConfig],
) -> None:
    """``--review`` remains report-only when App credentials are configured."""
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "test-private-key")
    monkeypatch.setattr("daydream.github_app.resolve_user_identity", lambda _target: "operator")

    def mint_forbidden(*_args: object) -> SimpleNamespace:
        pytest.fail("report-only --review must not mint an App installation token")

    async def post_forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("report-only --review must not post a PR review")

    async def fake_review(
        _work: WorkContext,
        config: RunConfig,
        _run_artifacts: Any = None,
    ) -> int:
        assert config.identity == "operator"
        return 0

    monkeypatch.setattr("daydream.github_app._mint_installation_token", mint_forbidden)
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", post_forbidden)
    monkeypatch.setattr("daydream.runner._run_loop_deep", fake_review)

    rc = await runner.run(make_config(tmp_path, output_mode="review", pr_repo="acme/widgets"))

    assert rc == 0


@pytest.mark.asyncio
async def test_owner_preflight_failure_never_reaches_identity_or_backend(
    monkeypatch: pytest.MonkeyPatch,
    silence_runner_ui: None,  # noqa: F841
    tmp_path: Path,
) -> None:
    """Real-path: owner preflight failure stays inside the trace boundary.

    A valid non-Git target fails ``resolve_private_workspace_owner`` after the
    run-root span opens. Both assertions sit on external seams — the App
    installation-token mint and ``create_backend`` — so the proof is that the
    outside world was never touched. App credentials are configured and the
    default deep flow with ``pr_repo`` posts, so a token would be minted here
    if identity resolution ran at all.
    """
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "test-private-key")

    def mint_forbidden(*_args: object) -> None:
        pytest.fail("owner preflight failure must not mint an App installation token")

    def backend_forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("owner preflight failure must not construct a backend")

    monkeypatch.setattr("daydream.github_app._mint_installation_token", mint_forbidden)
    monkeypatch.setattr("daydream.runner.create_backend", backend_forbidden)

    config = RunConfig(
        target=str(tmp_path),  # valid directory, no Git repository
        pr_repo="acme/widgets",
        cleanup=False,
        archive=False,
    )

    assert await runner.run(config) == 1


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (RunConfig(output_mode="comment"), True),
        # Issue #1113: the diagram-only run's deliverable IS a GitHub write.
        (RunConfig(output_mode="diagram", diagram="sequence"), True),
        (RunConfig(output_mode="loop"), True),
        (RunConfig(flow_name="deep"), True),
        (RunConfig(output_mode="review"), False),
        (RunConfig(flow_name="review"), False),
        (RunConfig(output_mode="loop", shallow=True), False),
        (RunConfig(flow_name="shallow"), False),
        (RunConfig(flow_name="custom-audit"), False),
    ],
    ids=[
        "comment",
        "diagram_only",
        "default_deep",
        "explicit_deep",
        "review",
        "explicit_review",
        "shallow_loop",
        "explicit_shallow",
        "custom_flow",
    ],
)
def test_run_posts_to_github_matches_dispatch(config: RunConfig, expected: bool) -> None:
    """The identity classifier follows the runner's known write-capable paths."""
    assert runner._run_posts_to_github(config) is expected


@pytest.mark.asyncio
async def test_comment_mode_without_open_pr_dispatches_to_deep_flow(
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace: Any,
    silence_runner_ui: None,  # noqa: F841
    tmp_path: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """``--comment --branch X`` with no open PR for X runs the deep flow.

    The old review flow refused up front ("No Open PR" error, exit 1); the
    collapsed deep flow dropped that pre-flight — ``_step_post_review`` warns and
    skips the post when no PR is resolvable (covered at the seam by
    ``test_post_skips_when_no_pr``). The runner-level contract is now: comment
    mode reaches the deep flow regardless of PR existence.
    """
    monkeypatch.setattr(
        "daydream.runner.git_ops.gh_pr_list_for_branch", lambda _repo, _branch: []
    )
    seen: dict[str, Any] = {}

    async def fake_deep(
        work: WorkContext,
        config: RunConfig,
        _run_artifacts: Any = None,
    ) -> int:
        seen["output_mode"] = config.output_mode
        seen["branch"] = config.branch
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_deep", fake_deep)

    config = make_config(tmp_path, output_mode="comment", branch="feat/missing")
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert seen == {"output_mode": "comment", "branch": "feat/missing"}, seen


# --- Per-phase model resolution tests (Task 2) -----------------------------


class TestResolveBackendPhaseModel:
    def test_explicit_phase_flag_wins_over_table(self) -> None:
        config = RunConfig(backend="claude", review_model="claude-haiku-4-5")
        backend = runner._resolve_backend(config, "review")
        assert backend.model == "claude-haiku-4-5"

    def test_table_default_used_when_no_flag(self) -> None:
        config = RunConfig(backend="claude")  # no review_model override
        backend = runner._resolve_backend(config, "review")
        assert backend.model == "claude-opus-5"  # claude REVIEW default

    def test_table_default_for_phase_without_flag(self) -> None:
        # WONDER has no override flag but should still get the table default.
        config = RunConfig(backend="claude")
        backend = runner._resolve_backend(config, "wonder")
        assert backend.model == "claude-opus-5"

    def test_codex_table_default(self) -> None:
        config = RunConfig(backend="codex")
        backend = runner._resolve_backend(config, "parse")
        assert backend.model == "gpt-5.6-luna"  # codex PARSE default (cheap tier)

    def test_backend_override_uses_overridden_backends_table(self) -> None:
        # review_backend=codex while default is claude: resolver must use the codex table.
        config = RunConfig(backend="claude", review_backend="codex")
        backend = runner._resolve_backend(config, "review")
        assert backend.model == "gpt-5.6-sol"  # codex REVIEW default (heavy tier)

    def test_cache_returns_same_instance_for_same_phase_and_backend(self) -> None:
        cache: dict[tuple[str, str | None, str | None, Path | None], Backend] = {}
        config = RunConfig(backend="claude")
        b1 = runner._resolve_backend(config, "review", cache)
        b2 = runner._resolve_backend(config, "review", cache)
        assert b1 is b2

    def test_cache_returns_distinct_instances_for_different_phases(self) -> None:
        # Different models -> different backends, even on the same backend kind.
        cache: dict[tuple[str, str | None, str | None, Path | None], Backend] = {}
        config = RunConfig(backend="claude")
        review_backend = runner._resolve_backend(config, "review", cache)
        parse_backend = runner._resolve_backend(config, "parse", cache)
        assert review_backend is not parse_backend

    def test_codex_backend_receives_resolved_reasoning_effort_and_cache_splits_on_it(self) -> None:
        cache: dict[tuple[str, str | None, str | None, Path | None], Backend] = {}
        config = RunConfig(backend="codex", reasoning_effort="low")
        low_backend: Any = runner._resolve_backend(config, "review", cache)
        assert low_backend.reasoning_effort == "low"
        config.reasoning_effort = "high"
        high_backend: Any = runner._resolve_backend(config, "review", cache)
        assert high_backend.reasoning_effort == "high"
        assert low_backend is not high_backend  # different effort -> distinct cached instance

    def test_audit_workspace_is_forwarded_and_splits_backend_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        first = tmp_path / "audit-one"
        first.mkdir()
        second = tmp_path / "audit-two"
        second.mkdir()
        created: list[tuple[object, dict[str, Any]]] = []

        def fake_create_backend(name: str, **kwargs: Any) -> object:
            backend = SimpleNamespace(
                model=kwargs.get("model") or "mock",
                audit_root=kwargs.get("audit_root"),
            )
            created.append((backend, kwargs))
            return backend

        monkeypatch.setattr(runner, "create_backend", fake_create_backend)
        config = RunConfig(target=str(source), backend="claude")
        cache: dict[
            tuple[str, str | None, str | None, Path | None], Backend
        ] = {}

        def boundary(repo: Path) -> AuditWorkspace:
            return AuditWorkspace(
                repo=repo,
                source=source,
                repo_git_common_dir=repo / ".git",
                source_git_common_dir=source / ".git",
                outward_symlinks=frozenset({repo / "outward"}),
            )

        first_boundary = boundary(first)
        first_backend = runner._resolve_backend(
            config,
            "recon",
            cache,
            cwd=source,
            audit_workspace=first_boundary,
        )
        assert runner._resolve_backend(
            config,
            "recon",
            cache,
            cwd=source,
            audit_workspace=first_boundary,
        ) is first_backend
        second_backend = runner._resolve_backend(
            config,
            "recon",
            cache,
            cwd=source,
            audit_workspace=boundary(second),
        )

        assert second_backend is not first_backend
        assert len(created) == 2
        assert created[0][1]["audit_root"] == first.resolve(strict=True)
        assert created[0][1]["audit_outward_symlinks"] == frozenset(
            {first / "outward"}
        )


@pytest.mark.parametrize(
    ("capability", "bound_root", "reason"),
    [
        (None, "expected", "missing_capability"),
        ("wrong-token", "expected", "wrong_capability"),
        (AUDIT_ROOT_ISOLATION_V1, "other", "wrong_root"),
    ],
)
def test_improve_backend_preflight_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: str | None,
    bound_root: str,
    reason: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    work = WorkContext(
        repo=source,
        source=source,
        base_branch="main",
        base_sha="1" * 40,
        head_branch="feature",
        head_sha="2" * 40,
        is_ephemeral=False,
        run_id="run-1",
    )
    boundary = AuditWorkspace(
        repo=audit,
        source=source,
        repo_git_common_dir=audit / ".git",
        source_git_common_dir=source / ".git",
        outward_symlinks=frozenset(),
    )
    context = FlowContext(
        config=RunConfig(target=str(source), flow_name="improve"),
        work=work,
        registry=build_registry(),
        audit_workspace=boundary,
    )
    backend = SimpleNamespace(model="mock", audit_root=audit if bound_root == "expected" else other)
    if capability is not None:
        backend.audit_root_isolation = capability
    monkeypatch.setattr(FlowContext, "backend_for", lambda _self, _phase: backend)

    with pytest.raises(AuditIsolationError) as exc_info:
        runner._preflight_improve_backends(context)

    assert exc_info.value.backend_name == "claude"
    assert exc_info.value.phase == "recon"
    assert exc_info.value.reason == reason


@pytest.mark.anyio
@pytest.mark.parametrize("unborn", [False, True])
async def test_improve_inherited_storage_override_stops_before_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: Callable[..., RunConfig],
    unborn: bool,
) -> None:
    repo = tmp_path / "source"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n")
    _git(repo, "add", "app.py")
    if not unborn:
        _commit(repo, "initial")
    (repo / "app.py").write_text("VALUE = 2\n")
    before_index = (repo / ".git" / "index").read_bytes()
    before_refs = _git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
    backend_calls: list[str] = []

    def unexpected_backend(*args: Any, **kwargs: Any) -> Backend:
        backend_calls.append("created")
        raise AssertionError("snapshot refusal must precede model construction")

    monkeypatch.setattr("daydream.runner.create_backend", unexpected_backend)
    alternate = str(repo / ".git" / "objects")
    with monkeypatch.context() as poison:
        poison.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", alternate)
        result = await runner.run(make_config(repo, flow_name="improve"))
        assert os.environ["GIT_ALTERNATE_OBJECT_DIRECTORIES"] == alternate
    output = capsys.readouterr().out
    assert result == 1
    assert backend_calls == []
    assert "snapshot refuses inherited Git" in output
    assert len(output) < 5_000
    assert (repo / "app.py").read_text() == "VALUE = 2\n"
    assert (repo / ".git" / "index").read_bytes() == before_index
    assert _git(repo, "for-each-ref", "--format=%(refname) %(objectname)") == before_refs


# --- Task 6: deep fix-cycle hero is followed by Model: dim line -------------


def _seed_fix_resume(target: Path, items: list[dict[str, Any]]) -> Path:
    """Prime the deep artifacts a ``--start-at fix`` resume reads.

    ``merged-items.json`` is the canonical items source the fix gate reads, and
    ``diff-key`` must match the current diff so ``check_deep_artifacts`` does not
    treat the primed set as stale. Key written first so every prerequisite is
    strictly newer than the key (the resume freshness gate requires it).

    Returns:
        The ``.daydream/deep`` directory.
    """
    from daydream import git_ops
    from daydream.deep.artifacts import diff_key, diff_key_path, merged_items_path
    from daydream.workspace import _resolve_base

    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    base = _resolve_base(target, None, None)
    diff = git_ops.diff(target, base)
    diff_key_path(deep).write_text(diff_key(diff or ""), encoding="utf-8")
    merged_items_path(deep).write_text(json.dumps({"items": items}))
    return deep


def _fix_item(item_id: int = 1, *, severity: str = "medium") -> dict[str, Any]:
    """Build a validated merged item targeting the fixture's tracked file."""
    return {
        "id": item_id,
        "lens": "per-stack",
        "file": "main.py",
        "line": 1,
        "severity": severity,
        "description": f"{severity} issue in main.py",
        "confidence": "MEDIUM",
        "rationale": "rationale",
        "evidence": "main.py:1",
    }


def _silence_fix_cycle_ui(silence_console: Callable[..., None]) -> None:
    """Silence the runner / deep orchestrator / phases noise; keeps phase hero
    and dim bindings alive so the hero-ordering test can spy on them."""
    silence_console("daydream.runner")
    silence_console("daydream.deep.orchestrator")
    silence_console("daydream.phases", keep=("print_phase_hero", "print_dim"))


async def _stub_verify(*_a: Any, **_k: Any) -> tuple[Path, dict[str, Any]]:
    """Empty-verdicts stand-in for ``phase_verify_recommendations``."""
    return Path("/nonexistent"), {"verdicts": []}


async def _stub_fix_verify(
    _backend: Any,
    _work: Any,
    items: list[dict[str, Any]],
    *_args: Any,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Resolve every canonical item for fix-cycle harnesses."""
    return [
        {"issue_id": item["id"], "verdict": "resolved", "reason": "harness"}
        for item in items
    ]


@pytest.mark.asyncio
async def test_fix_cycle_awaken_hero_followed_by_model_line(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
    make_config: Callable[..., 'RunConfig'],
    silence_console: Callable[..., None],
) -> None:
    """The AWAKEN test-phase hero must be followed by a dim ``Model: <name>``
    line scoped to the test backend.

    The shallow flow's HEAL hero was dropped in the single-flow collapse (#330);
    the surviving phase-hero + Model-line pair in the deep fix cycle is
    ``phase_test_and_heal``'s AWAKEN hero followed by the dim Model line. Drives
    the deep fix cycle (``shallow=True, start_at="fix"``) real-path with the REAL
    ``phase_test_and_heal`` over a passing scripted suite, and asserts hero + dim
    call ordering through the module bindings that actually render them.
    """
    _seed_fix_resume(feature_branch_repo, [_fix_item()])
    _silence_fix_cycle_ui(silence_console)

    test_backend = ScriptedBackend(
        script=[(TextEvent(text="All 1 tests passed. 0 failed."), _RESULT)],
        model="test-model-xyz",
    )
    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, cache=None, **_kwargs: (
            test_backend if phase == "test" else ScriptedBackend(model="stub-model")
        ),
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)
    monkeypatch.setattr("daydream.phases.phase_fix_verify", _stub_fix_verify)

    async def _noop_fix(*_a: Any, **_k: Any) -> dict[str, str]:
        return {}

    async def _noop_commit(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _noop_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

    # Capture hero + dim calls in order.
    calls: list[tuple[str, str]] = []  # (kind, payload)

    def _hero_spy(_console: Any, title: Any, _description: Any) -> None:
        calls.append(("hero", title))

    def _dim_spy(_console: Any, message: Any) -> None:
        calls.append(("dim", message))

    monkeypatch.setattr("daydream.phases.print_phase_hero", _hero_spy)
    monkeypatch.setattr("daydream.phases.print_dim", _dim_spy)

    exit_code = await runner.run(
        make_config(feature_branch_repo, start_at="fix", shallow=True, assume="yes")
    )
    assert exit_code == 0

    # Find the AWAKEN hero in call order.
    awaken_idx = next(
        (
            i
            for i, (kind, payload) in enumerate(calls)
            if kind == "hero" and payload == "AWAKEN"
        ),
        None,
    )
    assert awaken_idx is not None, f"AWAKEN hero never rendered; calls={calls!r}"
    # The very next call must be a dim Model: line carrying the test backend's id.
    assert awaken_idx + 1 < len(calls), "AWAKEN hero has no following call"
    next_kind, next_payload = calls[awaken_idx + 1]
    assert next_kind == "dim", (
        f"Call after AWAKEN hero was not a dim line; got {(next_kind, next_payload)!r}"
    )
    assert next_payload == "Model: test-model-xyz", (
        f"Dim line after AWAKEN hero did not echo the test backend's model; got {next_payload!r}"
    )


@pytest.mark.asyncio
async def test_fix_cycle_items_severity_ordered(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
    make_config: Callable[..., 'RunConfig'],
    silence_console: Callable[..., None],
) -> None:
    """Merged items are severity-sorted (high before low) before ``phase_fix_parallel``.

    The fix gate seeds ``ctx.data["items"]`` with canonical merged items in an
    out-of-order shape [low, high]; after ``severity_sorted`` the HIGH item must
    be fixed first. Asserts on the severity ``phase_fix_parallel`` actually
    receives (observable consequence), never on dispatch bookkeeping.
    """
    _seed_fix_resume(
        feature_branch_repo,
        [_fix_item(1, severity="low"), _fix_item(2, severity="high")],
    )
    _silence_fix_cycle_ui(silence_console)

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: ScriptedBackend(model="stub-model"),
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)
    monkeypatch.setattr("daydream.phases.phase_fix_verify", _stub_fix_verify)

    order: list[list[str]] = []

    async def _spy_fix_parallel(_b: Any, _w: Any, items: Any, **_k: Any) -> dict[str, Any]:
        order.append([item["severity"] for item in items])
        return {}

    async def _noop_test(*_a: Any, **kwargs: Any) -> Any:
        from daydream.phases import TestAndHealResult, TestAttemptEvidence

        key = kwargs["capture_tree_key"]()
        attempt = TestAttemptEvidence(
            session_id=kwargs["session_id"],
            kind="host",
            command=("true",),
            passed=True,
            input_tree_key=key,
            output_tree_key=key,
        )
        return TestAndHealResult(True, 0, True, False, (attempt,))

    async def _noop_commit(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _spy_fix_parallel)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _noop_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

    exit_code = await runner.run(
        make_config(feature_branch_repo, start_at="fix", shallow=True, assume="yes")
    )
    assert exit_code == 0
    assert order, "phase_fix_parallel was never called"
    # The fix gate is a round-aware loop (issue #744): round 1 dispatches the
    # full severity-sorted canonical list, so the HIGH item is fixed first.
    assert order[0] == ["high", "low"], (
        f"phase_fix_parallel did not receive severity-ordered items on round 1; got {order[0]!r}"
    )
    # Every round (round 1 and any re-dispatch) preserves the canonical
    # severity ordering derived from the severity-sorted item list.
    for round_items in order:
        assert round_items == sorted(
            round_items, key=lambda s: {"high": 0, "low": 1}[s]
        ), (
            f"phase_fix_parallel received out-of-order severities {round_items!r}"
        )


# --- Task 4: non_interactive threading -------------------------------------


def test_runconfig_defaults_non_interactive_false() -> None:
    assert RunConfig().non_interactive is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dispatch_target", "config_kwargs"),
    [
        ("daydream.runner._run_loop_deep", {"output_mode": "loop"}),
        ("daydream.runner._run_loop_deep", {"output_mode": "loop", "shallow": True}),
        ("daydream.runner._run_loop_deep", {"output_mode": "comment"}),
        ("daydream.runner._run_improve", {"flow_name": "improve"}),
    ],
    ids=["deep_loop", "shallow", "comment", "improve"],
)
async def test_run_threads_non_interactive_into_agent_state(
    dispatch_target: Any,
    config_kwargs: Any,
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace: Any,
    silence_runner_ui: None,  # noqa: F841
    tmp_path: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """``config.non_interactive=True`` flips the agent singleton flag before any
    promptable phase, on every dispatch branch ``run()`` can take. Each case
    patches one dispatch fn so ``run()`` reaches the run-start setup (where
    ``set_non_interactive`` fires) without executing real phases.
    """
    from daydream.agent import get_non_interactive, reset_state

    reset_state()
    try:

        async def stub(work: Any, config: Any, _run_artifacts: Any = None) -> int:
            return 0

        monkeypatch.setattr(dispatch_target, stub)
        config = make_config(tmp_path, non_interactive=True, **config_kwargs)

        exit_code = await runner.run(config)
        assert exit_code == 0
        assert get_non_interactive() is True
    finally:
        reset_state()


# --- Deep fix-cycle commit gate semantics ----------------------------------


class _CommitWritingBackend:
    """Scripted fake whose test-suite and commit turns really touch the worktree.

    The test-suite turn reports a green run; the commit turn runs a REAL ``git
    commit`` carrying the Daydream trailers, so ``_do_commit``'s post-commit
    trailer verification sees a new HEAD. The backend is the only mocked seam —
    exactly the shape an agent with tools would take.
    """

    model = "mock-model"

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.commit_prompts: list[str] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        pl = prompt.lower()
        if "run the project's test suite" in pl:
            yield TextEvent(text="All 1 tests passed. 0 failed.")
            yield ResultEvent(structured_output=None, continuation=None)
            return
        if "the daydream changes are already staged" in pl:
            self.commit_prompts.append(prompt)
            run_id = re.search(r"Daydream-Run: (\S+)", prompt)
            version = re.search(r"Daydream-Version: (\S+)", prompt)
            message = (
                "fix: apply daydream fix\n\n"
                f"Daydream-Run: {run_id.group(1) if run_id else 'unknown'}\n"
                f"Daydream-Version: {version.group(1) if version else 'unknown'}\n"
            )
            _git(cwd, "commit", "-m", message)
            yield TextEvent(text="Committed.")
            yield ResultEvent(structured_output=None, continuation=None)
            return
        yield TextEvent(text="ok")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass


@pytest.mark.asyncio
async def test_fix_cycle_yes_commits_fixes(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
    tmp_path: Path,
    make_config: Callable[..., 'RunConfig'],
    silence_console: Callable[..., None],
    no_ci_remote: NoCIRemote,
) -> None:
    """Real-path: a --yes shallow deep run whose tests pass commits its fixes.

    The deep fix cycle's commit step is ``phase_commit_push`` (interactive gate);
    under ``assume="yes"`` the gate auto-approves and the commit agent runs. The
    observable outcome is a NEW git commit carrying the Daydream trailer.
    """
    from daydream.config import REVIEW_OUTPUT_FILE

    _seed_fix_resume(feature_branch_repo, [_fix_item()])
    _silence_fix_cycle_ui(silence_console)
    # Host-native commit/push (issue #726) pushes to 'origin' for real; give
    # the repo a bare remote so the push + ls-remote verification succeeds.
    no_ci_remote.connect(feature_branch_repo, bare_remote(tmp_path / "origin.git"))

    commit_backend = _CommitWritingBackend(feature_branch_repo)
    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: commit_backend,
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)
    monkeypatch.setattr("daydream.phases.phase_fix_verify", _stub_fix_verify)

    async def _fix_writes(*_a: Any, **_k: Any) -> dict[str, str]:
        main_py = feature_branch_repo / "main.py"
        main_py.write_text(main_py.read_text() + "\n# daydream fix\n")
        return {}

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _fix_writes)

    # Pre-create the review output so the fix-gate decline path never triggers.
    (feature_branch_repo / REVIEW_OUTPUT_FILE).write_text("# Review\n")
    head_before = _git(feature_branch_repo, "rev-parse", "HEAD")

    exit_code = await runner.run(
        make_config(
            feature_branch_repo,
            start_at="fix",
            shallow=True,
            assume="yes",
            pr_number=no_ci_remote.pr_number,
            pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0

    head_after = _git(feature_branch_repo, "rev-parse", "HEAD")
    assert head_after != head_before, "the --yes run never committed"
    # Issue #726: the commit is host-native — no agent commit turn runs.
    assert commit_backend.commit_prompts == []
    assert "Daydream-Run:" in _git(feature_branch_repo, "log", "-1", "--format=%B")
    assert "# daydream fix" in _git(feature_branch_repo, "show", "HEAD:main.py")


@pytest.mark.asyncio
async def test_fix_cycle_non_interactive_declines_fix_and_commit(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
    make_config: Callable[..., 'RunConfig'],
    silence_console: Callable[..., None],
) -> None:
    """Real-path: a non-interactive shallow deep run with no ``--yes`` declines at
    the apply-fixes gate — no fix, no test, no commit (observable: HEAD unchanged).
    """
    _seed_fix_resume(feature_branch_repo, [_fix_item()])
    _silence_fix_cycle_ui(silence_console)

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: ScriptedBackend(model="stub-model"),
    )
    head_before = _git(feature_branch_repo, "rev-parse", "HEAD")

    exit_code = await runner.run(
        make_config(feature_branch_repo, start_at="fix", shallow=True)
    )

    assert exit_code == 0
    assert _git(feature_branch_repo, "rev-parse", "HEAD") == head_before, (
        "a non-interactive run without --yes committed"
    )
    assert not (feature_branch_repo / ".daydream-fix-applied").exists(), (
        "a non-interactive run without --yes applied a fix"
    )


async def _drive_fix_cycle_failing(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    config: RunConfig,
    *,
    script: list[Turn],
    stdin_guard_message: str | None = None,
    stdin_answers: list[str] | None = None,
    clipboard_is_available: bool = False,
) -> tuple[int, ScriptedBackend, list[bool]]:
    """Drive the deep fix-cycle (``shallow=True, start_at="fix"``) with the REAL
    ``phase_test_and_heal`` over a scripted failing test run.

    Holds everything the two failing-fix-cycle real-path tests share: the
    fix-resume artifact seed, the scripted backend bound to the ``test`` phase,
    stubbed verify/fix so only the test phase touches the backend, one commit
    spy, and either a stdin trap (unattended) or a fed stdin queue (interactive
    gate answers). ``_BEYOND_SCRIPT`` is appended so a heal loop that runs past
    its script raises instead of silently replaying the last turn forever.

    Returns:
        ``(exit_code, test_backend, commit_calls)``.
    """
    from daydream.agent import reset_state

    _seed_fix_resume(target, [_fix_item()])

    # Silence the flow's terminal noise; the test phase is observed through the
    # backend script, not scraped rendering.
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **k: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **k: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )
    monkeypatch.setattr(
        "daydream.runner.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )

    test_backend = ScriptedBackend(script=[*script, _BEYOND_SCRIPT], model="test-model-xyz")
    stub_backend = ScriptedBackend(model="stub-model")

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, cache=None, **_kwargs: (
            test_backend if phase == "test" else stub_backend
        ),
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)
    monkeypatch.setattr("daydream.phases.phase_fix_verify", _stub_fix_verify)

    async def _noop_fix(*_a: Any, **_k: Any) -> dict[str, str]:
        return {}

    commit_calls: list[bool] = []

    async def _spy_commit(*_a: Any, **_k: Any) -> None:
        commit_calls.append(True)

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _noop_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _spy_commit)
    monkeypatch.setattr("daydream.phases.clipboard_available", lambda: clipboard_is_available)

    if stdin_answers is not None:
        answers = iter(stdin_answers)

        def _feed_input(*_a: Any, **_k: Any) -> str:
            return next(answers)

        monkeypatch.setattr("builtins.input", _feed_input)
        monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
        monkeypatch.delenv("CI", raising=False)
    else:

        def _forbidden_input(*_a: Any, **_k: Any) -> str:
            raise AssertionError(stdin_guard_message or "stdin must not be touched")

        monkeypatch.setattr("builtins.input", _forbidden_input)

    reset_state()
    try:
        exit_code = await runner.run(config)
    finally:
        reset_state()

    return exit_code, test_backend, commit_calls


def _assert_single_handoff(repo: Path, body: str) -> None:
    handoffs = list(repo.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8") == body


@pytest.mark.asyncio
async def test_fix_cycle_failing_tests_abort_writes_handoff(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """Real-path: an interactive deep shallow run whose tests FAIL and the operator
    aborts (heal-menu "4") writes a handoff and exits 1 without committing.

    Drives the deep fix cycle with the REAL ``phase_test_and_heal`` over a
    scripted failing test run. Fix-gate answered "y"; the heal menu answered "4"
    (abort): the read-only failure-summarizer runs, ``handoff.md`` lands in the
    live run directory, and the run exits 1. The mutating heal fix agent is never
    launched ("Analyze the failures and fix them" absent).
    """
    exit_code, test_backend, commit_calls = await _drive_fix_cycle_failing(
        monkeypatch,
        feature_branch_repo,
        make_config(feature_branch_repo, start_at="fix", shallow=True, non_interactive=False),
        script=[_FAIL_TURN, _handoff_turn("# Handoff\n\ninteractive abort")],
        stdin_answers=["y", "4"],
    )

    assert exit_code == 1

    _assert_single_handoff(feature_branch_repo, "# Handoff\n\ninteractive abort")

    assert commit_calls == [], "a commit ran despite tests failing"

    # Exactly two test-backend calls: the failing test run + the read-only
    # summarizer. The mutating heal fix agent was never launched.
    assert len(test_backend.prompts) == 2
    assert "read-only failure-summarizer" in test_backend.prompts[1]
    assert all(
        "Analyze the failures and fix them" not in p for p in test_backend.prompts
    ), test_backend.prompts


@pytest.mark.asyncio
async def test_fix_cycle_clipboard_timeout_keeps_event_loop_responsive_and_shows_manual_copy_guidance(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """Real-path: a hung clipboard utility during the confirmed failure handoff is bounded to
    5s, runs off the event loop, degrades to the manual-copy warning, and the run still exits 1.

    Drives the deep fix cycle with the REAL phase_test_and_heal over a scripted failing test
    run. The operator confirms the fix gate, aborts the heal menu, and confirms the clipboard
    copy. The clipboard subprocess fake blocks 0.3s (a compressed stand-in for the 5s bound),
    raises TimeoutExpired, and the real copy_to_clipboard returns False -> the manual-copy
    warning fires. An event-loop ticker must keep ticking during the worker-thread block —
    proving the copy is offloaded, not run on the loop.
    """
    from daydream import clipboard

    warnings: list[str] = []
    monkeypatch.setattr(
        "daydream.phases.print_warning",
        lambda console_arg, message: warnings.append(message),  # noqa
    )
    successes: list[str] = []
    monkeypatch.setattr(
        "daydream.phases.print_success",
        lambda console_arg, message: successes.append(message),  # noqa
    )

    observed_timeouts: list[Any] = []
    state = {"ticks": 0, "at_entry": -1, "at_release": -1}
    stop_tick = threading.Event()

    def _blocking_run(argv: list[str], **kwargs: Any) -> None:
        observed_timeouts.append(kwargs.get("timeout"))
        state["at_entry"] = state["ticks"]
        time.sleep(0.3)  # simulated hung clipboard utility, bounded (stands in for 5s)
        state["at_release"] = state["ticks"]
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 5.0))

    monkeypatch.setattr(
        clipboard,
        "subprocess",
        SimpleNamespace(run=_blocking_run, SubprocessError=subprocess.SubprocessError),
    )
    monkeypatch.setattr("daydream.clipboard._detect_clipboard_command", lambda: ["pbcopy"])

    async def _ticker() -> None:
        while not stop_tick.is_set():
            state["ticks"] += 1
            await anyio.sleep(0.001)

    ticker_task = asyncio.create_task(_ticker())

    exit_code, _backend, commit_calls = await _drive_fix_cycle_failing(
        monkeypatch,
        feature_branch_repo,
        make_config(feature_branch_repo, start_at="fix", shallow=True, non_interactive=False),
        script=[_FAIL_TURN, _handoff_turn("# Handoff\n\nclipboard timeout")],
        stdin_answers=["y", "4", "y"],
        clipboard_is_available=True,
    )
    stop_tick.set()
    await ticker_task

    assert exit_code == 1
    assert observed_timeouts == [5], f"expected timeout exactly 5, got {observed_timeouts}"
    assert state["at_release"] >= state["at_entry"], (
        "event loop did not tick during the blocked clipboard copy — the copy is running "
        "synchronously on the loop, not offloaded"
    )
    assert any(
        "Clipboard copy failed; copy manually from path above" in m for m in warnings
    ), f"manual-copy warning missing; got {warnings!r}"
    assert successes == [], f"expected no success message, got {successes!r}"
    assert commit_calls == [], "a commit ran despite tests failing"

    _assert_single_handoff(feature_branch_repo, "# Handoff\n\nclipboard timeout")


@pytest.mark.asyncio
async def test_fix_cycle_failing_tests_bounded_fix_then_handoff(
    monkeypatch: pytest.MonkeyPatch,
    feature_branch_repo: Path,
    make_config: Callable[..., 'RunConfig'],
) -> None:
    """Real-path: a --yes shallow deep run whose tests FAIL runs ONE fix attempt
    then aborts.

    Drives the deep fix cycle with the REAL ``phase_test_and_heal`` against a
    scripted backend: fail → fix → fail → handoff (summarizer). With
    ``assume="yes"`` the bounded-loop guard (``decision is True and
    retries_used > 0``) fires after the first auto fix attempt, writing
    ``handoff.md`` and returning exit code 1. The fix prompt text ("Analyze the
    failures and fix them") appears in exactly one call (the fix agent), proving
    the fix ran once and only once. stdin must never be touched.
    """
    exit_code, test_backend, commit_calls = await _drive_fix_cycle_failing(
        monkeypatch,
        feature_branch_repo,
        make_config(feature_branch_repo, start_at="fix", shallow=True, assume="yes"),
        # Script: fail → fix (one bounded attempt) → fail → handoff (summarizer).
        # Any 5th call raises via the beyond-script turn, proving the loop ends.
        script=[
            _FAIL_TURN,
            _FIX_TURN,
            _FAIL_TURN,
            _handoff_turn("# Handoff\n\n--yes bounded fix failure"),
        ],
        stdin_guard_message="input() must not be called under --yes",
    )

    assert exit_code == 1

    _assert_single_handoff(feature_branch_repo, "# Handoff\n\n--yes bounded fix failure")

    assert commit_calls == [], "a commit ran despite tests failing"

    # Exactly four test-backend calls: fail → fix → fail → summarizer. No 5th
    # call (bounded-loop guard fired); call 4 ran read-only.
    assert len(test_backend.prompts) == 4, test_backend.prompts
    assert "Analyze the failures and fix them" in test_backend.prompts[1]
    assert "read-only failure-summarizer" in test_backend.prompts[3]
    assert test_backend.read_only_calls == [False, False, False, True], (
        test_backend.read_only_calls
    )


def test_open_recorder_resolves_backend_identity(tmp_path: Path) -> None:
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(target=str(target_dir), backend="codex", fix_backend="pi", run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.NORMAL,
    )
    assert recorder.backend_name == "codex"
    assert recorder.review_backend_name == "codex"
    assert recorder.fix_backend_name == "pi"
    assert recorder.test_backend_name == "codex"


def test_open_recorder_resolves_backend_via_per_stack_review(tmp_path: Path) -> None:
    """The representative backend follows per_stack_review, not the review phase.

    The deep flow's actual review fan-out runs on ``per_stack_review``; a
    per-phase override on ``per_stack_review`` must win for the run's backend
    identity.
    """
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(
        target=str(target_dir),
        run_eval=False,
        review_backend="codex",
        file_config=DaydreamFileConfig(phases={"per_stack_review": {"backend": "pi"}}),
    )
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.NORMAL,
    )
    assert recorder.backend_name == "pi"
    assert recorder.review_backend_name == "pi"
    assert recorder.fix_backend_name == "claude"
    assert recorder.test_backend_name == "claude"


def test_open_recorder_improve_omits_fix_test_backend(tmp_path: Path) -> None:
    """Improve trajectories carry no fix/test backend identity: the improve flow
    never runs those phases, so serializing them would mislabel the run."""
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(target=str(target_dir), backend="codex", run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.IMPROVE,
    )
    assert recorder.backend_name == "codex"
    assert recorder.review_backend_name == "codex"
    assert recorder.fix_backend_name == ""
    assert recorder.test_backend_name == ""


def test_open_recorder_review_only_omits_fix_test_backend(tmp_path: Path) -> None:
    """Review-only (TTT) trajectories carry no fix/test backend identity.

    ``--review``/``--comment`` stop the deep spine after post-review, so their
    flow never runs the fix cycle; emitting fix/test labels would mislabel them.
    """
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(target=str(target_dir), backend="codex", run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.TTT,
    )
    assert recorder.backend_name == "codex"
    assert recorder.review_backend_name == "codex"
    assert recorder.fix_backend_name == ""
    assert recorder.test_backend_name == ""


def test_open_recorder_custom_omits_fix_test_backend(tmp_path: Path) -> None:
    """Custom (fork) flows carry no fix/test backend identity.

    A fork-defined CUSTOM flow's composition is unknowable at recorder-open
    time — review-only forks are explicitly supported — so labeling it
    unconditionally would mislabel runs that never run the fix/test phases.
    Even an explicit ``fix_backend`` must not leak onto a CUSTOM run.
    """
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(target=str(target_dir), backend="codex", fix_backend="pi", run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.CUSTOM,
    )
    assert recorder.backend_name == "codex"
    assert recorder.review_backend_name == "codex"
    assert recorder.fix_backend_name == ""
    assert recorder.test_backend_name == ""


def test_open_recorder_diagram_omits_fix_test_backend(tmp_path: Path) -> None:
    """#1113: diagram-only trajectories carry no fix/test backend identity.

    The ``diagram`` flow is exploration -> diagram -> post-diagram; it never
    runs the fix cycle, so emitting fix/test labels would mislabel it. Even an
    explicit ``--fix-backend`` must not leak onto a diagram-only run.
    """
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(target=str(target_dir), backend="codex", fix_backend="pi", run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.DIAGRAM,
    )
    assert recorder.backend_name == "codex"
    assert recorder.review_backend_name == "codex"
    assert recorder.fix_backend_name == ""
    assert recorder.test_backend_name == ""


def test_open_recorder_diagram_resolves_backend_via_the_diagram_phase(tmp_path: Path) -> None:
    """#1113: the diagram flow's representative backend follows its only agent
    phase, so a ``[tool.daydream.phases.diagram]`` backend override wins over
    the never-run ``review`` phase."""
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(
        target=str(target_dir),
        run_eval=False,
        review_backend="codex",
        file_config=DaydreamFileConfig(phases={"diagram": {"backend": "pi"}}),
    )
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.DIAGRAM,
    )
    assert recorder.backend_name == "pi"
    assert recorder.review_backend_name == "pi"
    assert recorder.fix_backend_name == ""
    assert recorder.test_backend_name == ""


def test_open_recorder_improve_resolves_backend_via_recon(tmp_path: Path) -> None:
    """The improve flow's representative backend follows its advisory phases.

    The improve flow runs exclusively on recon/audit/vet/plan_write steps (no
    review step), so a file-config override on ``recon`` — its first advisory
    phase — must win for the run's backend identity instead of the never-run
    ``review`` phase.
    """
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(
        target=str(target_dir),
        run_eval=False,
        review_backend="codex",
        file_config=DaydreamFileConfig(phases={"recon": {"backend": "pi"}}),
    )
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.IMPROVE,
    )
    assert recorder.backend_name == "pi"
    assert recorder.review_backend_name == "pi"
    assert recorder.fix_backend_name == ""
    assert recorder.test_backend_name == ""


def test_open_recorder_pr_flow_resolves_fix_omits_test(tmp_path: Path) -> None:
    """PR trajectories retain their historical fix-without-test backend metadata."""
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(target=str(target_dir), review_backend="codex", run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.PR,
    )
    assert recorder.backend_name == "codex"
    assert recorder.review_backend_name == "codex"
    assert recorder.fix_backend_name == "claude"
    assert recorder.test_backend_name == ""


def test_open_recorder_backend_falls_back_to_claude(tmp_path: Path) -> None:
    from daydream.runner import _open_recorder
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    config = RunConfig(target=str(target_dir), run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.NORMAL,
    )
    assert recorder.backend_name == "claude"
    assert recorder.review_backend_name == "claude"
    assert recorder.fix_backend_name == "claude"
    assert recorder.test_backend_name == "claude"


def _build_manifest(config: RunConfig, flow: DaydreamRunFlow, tmp_path: Path) -> Manifest:
    """Build a manifest for ``config``/``flow`` from one real frozen snapshot."""
    snapshot = _capture_snapshot(tmp_path, "complete")
    return build_manifest_from_snapshot(
        recorder_provenance=archive_recorder_provenance_from_snapshot(
            write_snapshot=snapshot, run_flow=flow
        ),
        write_snapshot=snapshot,
        config=config,
        git_ctx=GitContext(),
        status="complete",
        archive_path=tmp_path,
    )


def test_manifest_backend_is_general_default_not_per_stack_review(tmp_path: Path) -> None:
    """#647: the manifest records the phase-agnostic general default backend.

    The archive manifest must NOT resolve its general ``backend`` through a
    per-phase key (e.g. ``per_stack_review``) — that mirrored the trajectory's
    mislabeled review identity. ``backend`` stays the general default
    (``config.backend`` → file-config global → ``"claude"``) even when a
    per-phase review override is set, and ``review_backend`` records only the
    review-specific override marker.
    """
    from daydream.config_file import DaydreamFileConfig

    config = RunConfig(
        target=str(tmp_path / "project"),
        run_eval=False,
        review_backend="codex",
        file_config=DaydreamFileConfig(phases={"per_stack_review": {"backend": "pi"}}),
    )
    m = _build_manifest(config, DaydreamRunFlow.NORMAL, tmp_path)
    assert m.backend == "claude"
    assert m.review_backend == "codex"


def test_manifest_normal_records_fix_and_test_backend(tmp_path: Path) -> None:
    """A deep-flow run records per-phase fix/test backends on the manifest."""
    config = RunConfig(
        target=str(tmp_path / "project"),
        run_eval=False,
        backend="codex",
        fix_backend="pi",
        test_backend="osprey",
    )
    m = _build_manifest(config, DaydreamRunFlow.NORMAL, tmp_path)
    assert m.backend == "codex"
    assert m.review_backend is None
    assert m.fix_backend == "pi"
    assert m.test_backend == "osprey"
    run = m.to_dict()["run"]
    assert run["fix_backend"] == "pi"
    assert run["test_backend"] == "osprey"


def test_manifest_review_only_omits_fix_test_backend(tmp_path: Path) -> None:
    """Review-only (TTT) manifests omit fix/test backend keys entirely.

    ``--review``/``--comment`` never run the fix or test phases, so
    ``fix_backend``/``test_backend`` resolve empty and the ``or None`` omission
    in ``build_manifest`` drops the keys rather than mislabeling the run.
    """
    config = RunConfig(target=str(tmp_path / "project"), run_eval=False, backend="codex")
    m = _build_manifest(config, DaydreamRunFlow.TTT, tmp_path)
    assert m.backend == "codex"
    assert m.review_backend is None
    assert m.fix_backend is None
    assert m.test_backend is None
    run = m.to_dict()["run"]
    assert "fix_backend" not in run
    assert "test_backend" not in run


def test_manifest_improve_omits_fix_test_backend(tmp_path: Path) -> None:
    """Improve manifests omit fix/test backend keys (those phases never run)."""
    config = RunConfig(target=str(tmp_path / "project"), run_eval=False, backend="codex")
    m = _build_manifest(config, DaydreamRunFlow.IMPROVE, tmp_path)
    assert m.backend == "codex"
    assert m.review_backend is None
    assert m.fix_backend is None
    assert m.test_backend is None
    run = m.to_dict()["run"]
    assert "fix_backend" not in run
    assert "test_backend" not in run


def test_manifest_pr_flow_records_fix_omits_test_backend(tmp_path: Path) -> None:
    """PR manifests retain their historical fix-without-test backend metadata."""
    config = RunConfig(
        target=str(tmp_path / "project"), run_eval=False, review_backend="codex",
    )
    m = _build_manifest(config, DaydreamRunFlow.PR, tmp_path)
    assert m.backend == "claude"
    assert m.review_backend == "codex"
    assert m.fix_backend == "claude"
    assert m.test_backend is None
    run = m.to_dict()["run"]
    assert "test_backend" not in run


def test_manifest_backend_falls_back_to_claude(tmp_path: Path) -> None:
    """With no backend configured anywhere, the manifest defaults to claude."""
    config = RunConfig(target=str(tmp_path / "project"), run_eval=False)
    m = _build_manifest(config, DaydreamRunFlow.NORMAL, tmp_path)
    assert m.backend == "claude"
    assert m.review_backend is None
    assert m.fix_backend == "claude"
    assert m.test_backend == "claude"
