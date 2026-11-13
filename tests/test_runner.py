"""RunConfig and unified runner dispatch tests."""

from __future__ import annotations

import json
import re
import subprocess
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import anyio
import pytest
from rich.console import Console

from daydream import git_ops, pr_review, runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends import (
    AUDIT_ROOT_ISOLATION,
    AgentEvent,
    AuditIsolationError,
    ResultEvent,
    TextEvent,
)
from daydream.dataset import LocalRecordStore
from daydream.deep.artifacts import (
    DeepArtifact,
    diff_key,
)
from daydream.extensions.loader import build_registry
from daydream.flows.engine import BackendFactory, FlowContext
from daydream.github_app import GitHubExecutionInput
from daydream.observability.config import ObservabilityConfig
from daydream.pr_review import PRInfo
from daydream.run_artifacts import (
    _RunSnapshotCaptureError,
    _RunWriteCapture,
    capture_manifest_run_identity,
)
from daydream.run_config import RunConfig
from daydream.run_context import RunContext, current_run_context
from daydream.trajectory import (
    DaydreamRunFlow,
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
)
from daydream.workspace import AuditWorkspace, WorkContext, _resolve_base
from tests.harness.backend import ScriptedBackend, Turn
from tests.harness.dataset import read_records
from tests.harness.git_helpers import bare_remote, commit as _commit, git as _git, init_repo as _init_repo
from tests.harness.otlp import attributes, kind_of
from tests.harness.trajectory import make_recorder

_RESULT = ResultEvent(structured_output=None, continuation=None)

_ARTIFACT_PR = PRInfo(
    number=7, head_sha="a" * 40, base_sha="b" * 40, base_ref="main", head_ref="feature", owner="owner", repo="repo",
    url="https://example.invalid/owner/repo/pull/7",
)

async def test_unborn_improve_approved_head_rejected_before_backend(
    tmp_path: Path, make_config: Callable[..., RunConfig],
) -> None:
    repo = tmp_path / "unborn"
    _init_repo(repo)
    (repo / "staged.py").write_text("value = 1\n")
    _git(repo, "add", "staged.py")
    before = git_ops.staged_patch(repo)
    config = make_config(repo, flow_name="improve", approved_head_sha="a" * 40)
    assert await runner.run(config) == 1
    assert git_ops.is_unborn_head(repo)
    assert git_ops.staged_patch(repo) == before

_FIX_TURN: tuple[AgentEvent, ...] = (TextEvent(text="Applied fix attempt"), _RESULT)
# Raised if the heal loop calls the backend past its script -- the bounded-loop guard.
_BEYOND_SCRIPT: Turn = (AssertionError("backend invoked beyond scripted call count"),)

def _handoff_turn(body: str) -> Turn:
    """The read-only failure-summarizer's structured handoff response."""
    return (ResultEvent(structured_output={"handoff_prompt": body}, continuation=None),)

_VALID_DOCUMENT_BYTES = json.dumps(
    {"session_id": "session", "trajectory_id": "session", "steps": [], "final_metrics": {}, "extra": {}}
).encode()

@pytest.fixture
def capture_recorder(tmp_path: Path) -> TrajectoryRecorder:
    """A bare recorder identifying the ``session`` run for capture tests."""
    return make_recorder(tmp_path, path=tmp_path / "trajectory.json", agent_model_name="test", session_id="session")

def _capture_snapshot(
    tmp_path: Path, status: Literal["complete", "partial"], *, json_bytes: bytes = _VALID_DOCUMENT_BYTES,
    root: str = "session", cutoff_at: str = "2026-09-06T00:00:00Z",
) -> RunWriteSnapshot:
    """One single-document run-write snapshot whose path is never written."""
    return RunWriteSnapshot(
        status=status, cutoff_at=cutoff_at, root_trajectory_id=root,
        documents=(
            TrajectoryDocumentSnapshot(
                trajectory_id="session", path=tmp_path / f"missing.{'json' if status == 'complete' else 'partial'}",
                json_bytes=json_bytes,
            ),
        ),
    )

def test_run_write_capture_retains_valid_final_without_io_and_records_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_recorder: TrajectoryRecorder
) -> None:
    """The recorder callback is a non-raising immutable handoff, not finalization."""
    # Completed children are references: retention checks their metadata, never their files.
    child = TrajectoryDocumentReference("session:child", tmp_path / "child.json", tmp_path / "child.json", 2, "0" * 64)
    prepared = _capture_snapshot(tmp_path, "complete")
    final = replace(prepared, documents=(*prepared.documents, child))
    capture = _RunWriteCapture(session_id="session")
    capture.retain(capture_recorder, final)
    assert capture.final is final
    assert capture.partial is None
    assert capture.validation_error is None
    assert not final.documents[0].path.exists() and not child.private_path.exists()
    for invalid in (_capture_snapshot(tmp_path, "partial", root="other"),
                    replace(final, documents=(*prepared.documents, replace(child, sha256="unchecked")))):
        capture.retain(capture_recorder, invalid)
        assert capture.final is final
        assert isinstance(capture.validation_error, _RunSnapshotCaptureError)
    # A BaseException during validation propagates instead of being recorded.
    capture = _RunWriteCapture(session_id="session")
    def interrupt(_payload: bytes) -> Any:
        raise KeyboardInterrupt
    monkeypatch.setattr("daydream.trajectory.types.json.loads", interrupt)
    with pytest.raises(KeyboardInterrupt):
        capture.retain(capture_recorder, _capture_snapshot(tmp_path, "partial", json_bytes=b"{}"))
    assert capture.partial is None
    assert capture.final is None
    assert capture.validation_error is None

def test_diagram_artifact_uses_the_captured_pr_and_the_run_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A captured identity replaces target resolution; classification keeps the run auth.

    The prepared-artifact notice never exposes the private write path.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n")
    _git(repo, "add", "app.py")
    _commit(repo, "base")
    head = git_ops.head_sha(repo)
    auth = git_ops.StaticGitHubAuth({"PATH": "/usr/bin", "GH_TOKEN": "artifact-owner"})
    seen: list[git_ops.GitHubAuth] = []
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number",
                        lambda *_a, **_k: pytest.fail("no live PR lookup at export"))

    def read_diff(
        target_dir: Path, number: int, *, auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
    ) -> str:
        assert target_dir == repo
        assert number == 7
        seen.append(auth)
        return ""
    monkeypatch.setattr(git_ops, "gh_pr_diff", read_diff)
    destination = tmp_path / "private" / "runtime" / "secret" / "findings.json"

    assert runner._write_findings_for_parsed(
        repo, RunConfig(pr_number=7, findings_out=str(destination)), [], auth=auth, kind="diagram",
        renderers=pr_review.ReviewRenderers(pr_review.default_render_finding, pr_review.default_render_summary),
        run_info="Fixture run info", captured_pr=replace(_ARTIFACT_PR, head_sha=head, base_sha=head),
    ) == 0

    # base == head leaves the local changed-file diff empty, so the GitHub-side
    # read is reached and must carry the owning session's auth.
    assert seen == [auth]
    assert json.loads(destination.read_text())["head_sha"] == head
    assert "artifact-owner" not in destination.read_text()
    output = capsys.readouterr().out
    assert "Findings artifact prepared." in output and str(destination) not in output

    # The captured identity is required: the writer has no live-lookup fallback left.
    with pytest.raises(TypeError):
        runner._write_findings_for_parsed(
            repo, RunConfig(pr_number=7, findings_out=str(destination)), [], auth=auth, kind="diagram",
            renderers=pr_review.ReviewRenderers(pr_review.default_render_finding, pr_review.default_render_summary),
            run_info="Fixture run info",
        )  # type: ignore[call-arg]

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

def _write_probe_flow(ext_dir: Any, flow_name: str, *, exit_code: int = 0) -> None:
    """Register a one-step flow whose step makes one real ``run_agent`` call."""
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep, Stop\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def _probe(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        "    assert ctx.private_workspace_owner is not None\n"
        "    await run_agent(ctx.backend_for('probe'), ctx.work.repo, 'PROBE',"
        " phase=DaydreamPhase.REVIEW)\n"
        "    ctx.artifacts.review_output.write_text('Completed review\\n')\n"
        f"    return Stop({exit_code})\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='probe', run=_probe))\n"
        f"    registry.set_flow({flow_name!r}, ['probe'])\n"
    )

class _ControlledBackend:
    """Model normal, blocked, permanently silent, and mid-stream failing backend invocations.

    Keep failures inside an async generator to preserve the real iteration boundary.
    """

    model = "controlled-model"

    def __init__(
        self, mode: Literal["yield", "block", "hang", "raise"] = "yield", *, error: BaseException | None = None,
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
    return await runner.run(config, private_roots=private_root_locations(base=tmp_path / "private"))

def _assert_one_published_run(repo: Path, archive_dir: Path) -> tuple[Path, Path]:
    public_runs = list((repo / ".daydream" / "runs").iterdir())
    archived_runs = list((archive_dir / "runs").iterdir())
    assert len(public_runs) == len(archived_runs) == 1
    return public_runs[0], archived_runs[0]


@pytest.mark.parametrize("failure_mode", [
    "none", "source-replaced", "source-missing-worktree", "git-redirect-capture-worktree",
])
async def test_artifact_session_runner_controlled_custom_flow_publishes_after_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, archive_dir: Path, failure_mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Finalization publishes frozen evidence after the model; lost or redirected sources are never touched.

    ``none`` is the only runner proof that the public run, archive, external trajectory, and
    dump are byte-equal to the frozen trajectory, and that the selected local trace stays only
    in the private run directory; the other rows prove a replacement checkout or foreign Git
    metadata at the original path is left byte-identical.
    """
    from daydream.eval.analyzer import analyze_session
    from tests.test_artifact_visibility import _Entry, _manifest, _walk_entry

    force_worktree = failure_mode.endswith("-worktree")
    repo = _feature_repo(tmp_path, remote=force_worktree)
    source_loss = failure_mode.startswith("source-")
    git_redirect = failure_mode == "git-redirect-capture-worktree"
    original_head = _git(repo, "rev-parse", "HEAD")
    original_diff = git_ops.diff(repo, "main")
    _write_probe_flow(ext_dir, "artifact-probe")
    backend = _ControlledBackend("block")
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    evaluation_entered = anyio.Event()
    evaluation_release = threading.Event()

    def pause_evaluation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        anyio.from_thread.run_sync(evaluation_entered.set)
        assert evaluation_release.wait(timeout=10), "test did not release evaluation"
        return analyze_session(*args, **kwargs)

    if failure_mode == "none":
        monkeypatch.setattr("daydream.eval.analyzer.analyze_session", pause_evaluation)
    store_path = tmp_path / "records"
    external_trajectory = tmp_path / "external trajectory.json"
    external_trajectory.write_text("operator baseline\n", encoding="utf-8")
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "prior.txt").write_text("prior dump")
    (repo / ".review-output.md").write_text("prior review")
    config = RunConfig(
        target=str(repo), base="main", flow_name="artifact-probe", trajectory_path=external_trajectory,
        run_eval=True, dump_artifacts=str(dump), archive=True, non_interactive=True,
        dataset_capture=source_loss or git_redirect,
        dataset_store_path=store_path,
        force_worktree=force_worktree,
        observability=ObservabilityConfig(destinations=("local",)),
    )
    result: list[int] = []
    frozen_trajectory: bytes | None = None
    active_progress = ""
    async def invoke() -> None:
        result.append(await _run_private(config, tmp_path))
    replacement_root: Path
    replacement_before: tuple[_Entry, ...]
    replacement_git: tuple[str, str]
    replacement_git_entries: list[_Entry]
    capture_dispatched = False
    real_run = subprocess.run
    foreign_repo, foreign_worktree = tmp_path / "foreign", tmp_path / "foreign-worktree"
    foreign_entries: list[_Entry] = []
    foreign_git_dir = foreign_head = ""
    if git_redirect:
        _init_repo(foreign_repo)
        _git(foreign_repo, "config", "maintenance.auto", "false")
        (foreign_repo / "foreign.py").write_bytes(b"FOREIGN_SENTINEL = 9\n")
        _git(foreign_repo, "add", "foreign.py")
        _commit(foreign_repo, "foreign head")
        _git(foreign_repo, "worktree", "add", "--detach", str(foreign_worktree), "HEAD")
        foreign_head = _git(foreign_repo, "rev-parse", "HEAD")
        assert foreign_head != original_head
        foreign_git_dir = _git(foreign_worktree, "rev-parse", "--absolute-git-dir")
        _walk_entry(tmp_path, foreign_repo, foreign_entries)
        _walk_entry(tmp_path, foreign_worktree, foreign_entries)

    def lose_source() -> None:
        nonlocal replacement_root, replacement_before, replacement_git, replacement_git_entries
        repo.rename(tmp_path / "removed-source")
        if failure_mode == "source-replaced":
            _init_repo(repo)
            (repo / "replacement.txt").write_bytes(b"replacement checkout\x00\xff")
            _git(repo, "add", "replacement.txt")
            _commit(repo, "replacement")
        replacement_root = repo if failure_mode == "source-replaced" else tmp_path / "removed-source"
        replacement_before = _manifest(replacement_root)
        replacement_git = (_git(replacement_root, "rev-parse", "HEAD"),
                           _git(replacement_root, "status", "--porcelain"))
        replacement_git_entries = []
        _walk_entry(replacement_root, replacement_root / ".git", replacement_git_entries)

    def replace_before_capture(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        nonlocal capture_dispatched
        if (not capture_dispatched and args == ["git", "rev-parse", "HEAD"] and kwargs.get("cwd") == backend.cwd
                and list((tmp_path / "private" / "runtime").glob("*/runs/*/frozen/.daydream/diff.patch"))):
            capture_dispatched = True
            assert backend.cwd is not None
            (backend.cwd / ".git").write_text(f"gitdir: {foreign_git_dir}\n", encoding="utf-8")
            assert _git(backend.cwd, "rev-parse", "HEAD") == foreign_head
        return real_run(args, **kwargs)

    with anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(invoke)
            await backend.entered.wait()
            assert not (repo / ".daydream").exists()
            assert external_trajectory.read_text(encoding="utf-8") == "operator baseline\n"
            assert list((tmp_path / "private" / "runtime").glob("*/runs/*/live/.daydream/diff.patch"))
            if git_redirect:
                monkeypatch.setattr("daydream.git_ops.process.subprocess.run", replace_before_capture)
            elif source_loss:
                lose_source()
            backend.release.set()
            if failure_mode == "none":
                try:
                    await evaluation_entered.wait()
                    active = capsys.readouterr()
                    active_progress = active.err
                    assert "Finalization" not in active.out
                    assert "Finalization started." in active_progress
                    assert "Finalization: Evaluating run." in active_progress
                    assert "Finalization complete" not in active_progress
                    assert result == []
                    assert not (repo / ".daydream").exists()
                    assert not list((archive_dir / "runs").glob("[!.]*"))
                    live_trajectories = list(
                        (tmp_path / "private" / "runtime").glob("*/runs/*/live/.daydream/runs/*/trajectory.json")
                    )
                    assert len(live_trajectories) == 1
                    frozen_trajectory = live_trajectories[0].read_bytes()
                finally:
                    evaluation_release.set()
    assert capture_dispatched is git_redirect
    captured = capsys.readouterr()
    progress = active_progress + captured.err
    assert not re.search(r"Finalization(?: started|:| complete| failed| finished)", captured.out)
    assert "\x1b[" not in progress and "\r" not in progress
    assert str(tmp_path / "private") not in progress
    if git_redirect:
        assert result == [0]
        raw = read_records(LocalRecordStore(store_path)).runs
        assert len(raw) == 1 and raw[0]["final_state"]["value"]["head_sha"] is None
        assert _git(repo, "rev-parse", "HEAD") == original_head
        frozen_diff, = (tmp_path / "private" / "runtime").glob("*/runs/*/frozen/.daydream/diff.patch")
        assert frozen_diff.read_text() == original_diff
        assert all(document["session_id"] == raw[0]["run_id"]
                   for document in raw[0]["trajectories"]["value"]["documents"])
        retained_foreign: list[_Entry] = []
        _walk_entry(tmp_path, foreign_repo, retained_foreign)
        _walk_entry(tmp_path, foreign_worktree, retained_foreign)
        assert retained_foreign == foreign_entries
        assert not list((archive_dir / "runs").glob("[!.]*"))
        assert "Finalization complete" not in progress
        assert "Local JSONL capture: retained" in progress and "Archive/evaluation: failed" in progress
        assert "Public publication: installed" in progress
        assert str(foreign_repo) not in progress and foreign_head not in progress
        assert (repo / ".review-output.md").read_text() == "Completed review\n"
        return
    if source_loss:
        assert result == [1]
        assert "Finalization complete" not in progress
        raw = read_records(LocalRecordStore(store_path)).runs
        assert len(raw) == 1
        assert raw[0]["final_state"]["value"]["head_sha"] is None
        documents = raw[0]["trajectories"]["value"]["documents"]
        assert all(document["session_id"] == raw[0]["run_id"] for document in documents)
        frozen = list((tmp_path / "private" / "runtime").glob("*/runs/*/frozen/.daydream/diff.patch"))
        assert len(frozen) == 1 and frozen[0].read_text() == original_diff
        assert _manifest(replacement_root) == replacement_before
        retained_git_entries: list[_Entry] = []
        _walk_entry(replacement_root, replacement_root / ".git", retained_git_entries)
        assert retained_git_entries == replacement_git_entries
        if force_worktree:
            assert backend.cwd is not None and backend.cwd.is_dir()
            assert "Worktree cleanup skipped: original source is unavailable or changed." in captured.out
        assert (_git(replacement_root, "rev-parse", "HEAD"),
                _git(replacement_root, "status", "--porcelain")) == replacement_git
        if failure_mode.startswith("source-missing"):
            assert not repo.exists() and not repo.is_symlink()
            archived = list((archive_dir / "runs").glob("[!.]*"))
            assert len(archived) == 1
            evaluation = json.loads((archived[0] / "evaluation.json").read_text())
            assert evaluation["quality"]["unavailable"] is True
            assert evaluation["quality"]["erosion"] is evaluation["quality"]["verbosity"] is None
            assert json.loads((archived[0] / "manifest.json").read_text())["git"]["head_sha"] == original_head
            assert "Archive/evaluation: retained" in progress
            assert "artifact source is unavailable" in captured.out.lower()
        else:
            assert (repo / "replacement.txt").read_bytes() == b"replacement checkout\x00\xff"
        assert "Local JSONL capture: retained" in progress
        assert "Public publication: failed" in progress or "Public publication: not attempted" in progress
        assert "Run evidence could not be persisted" not in captured.out
        assert json.loads(external_trajectory.read_text()) == documents[0]
        assert (dump / "prior.txt").read_text() == "prior dump"
        assert list((tmp_path / "private" / "runtime").glob("*/transactions/*/journal.json"))
        return
    assert re.search(r"Finalization complete after \d+\.\ds", progress)
    assert result == [0] and "Data Collection" not in captured.out
    assert (repo / ".review-output.md").read_text() == "Completed review\n"
    public_run, archived_run = _assert_one_published_run(repo, archive_dir)
    local_trace, = (tmp_path / "private" / "runtime").glob(f"*/runs/{public_run.name}/otel-trace.jsonl")
    assert list(tmp_path.rglob("otel-trace.jsonl")) == [local_trace]
    assert local_trace.stat().st_mode & 0o777 == 0o600
    spans = [span for line in local_trace.read_text().splitlines() for resource in json.loads(line)["resourceSpans"]
             for scope in resource["scopeSpans"] for span in scope["spans"]]
    assert kind_of(spans, "run") == [spans[-1]] and "parentSpanId" not in spans[-1]
    assert spans[-1]["status"] == {"code": 1}
    (agent,), (attempt,) = kind_of(spans, "agent"), kind_of(spans, "attempt")
    assert "PROBE" in attributes(agent)["traceloop.entity.input"]
    assert json.loads(attributes(attempt)["traceloop.entity.output"]) == "controlled output"
    archived_bytes = (archived_run / "trajectory.json").read_bytes()
    assert archived_bytes == frozen_trajectory
    evaluation = json.loads((archived_run / "evaluation.json").read_text())
    assert evaluation["quality"]["scoped_files"] == 1
    assert list(evaluation["quality"]["per_file"]) == ["app.py"]
    assert "Finalization: Publishing run archive." in progress
    assert "Finalization: Publishing run outputs." in progress
    assert (public_run / "trajectory.json").read_bytes() == archived_bytes
    assert external_trajectory.read_bytes() == archived_bytes
    assert (dump / "trajectory.json").read_bytes() == archived_bytes
    manifest = json.loads((archived_run / "manifest.json").read_text())
    assert manifest["session_id"] == public_run.name
    assert manifest["archive_status"] == "complete"

# --- Stage 4.1b dispatch tests ---------------------------------------------

@pytest.fixture
def silence_runner_ui(silence_console: Callable[..., None]) -> None:
    """Mute runner UI only; phase hero/model bindings stay live for the AWAKEN ordering test."""
    silence_console("daydream.runner")

_DISPATCH_TARGETS = ("_run_loop_deep", "_run_improve")

def _make_recording_dispatch(
    called: list[str], on_call: Callable[[str, Any, Any, RunContext], None] | None = None,
) -> Any:
    def _record(name: str) -> Any:
        async def stub(
            work: Any, config: Any, _run_artifacts: Any = None, *,
            run_context: RunContext, github_execution: GitHubExecutionInput, backend_factory: BackendFactory | None,
        ) -> int:
            assert run_context is current_run_context()
            called.append(name)
            if on_call is not None:
                on_call(name, work, config, run_context)
            return 0
        return stub
    return _record

async def test_artifact_session_runner_cancellation_finalizes_then_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, archive_dir: Path,
) -> None:
    """Cancellation joins the backend, durably publishes partial evidence, then escapes."""
    repo = _feature_repo(tmp_path)
    _write_probe_flow(ext_dir, "artifact-probe-cancel")
    backend = _ControlledBackend("hang")
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    config = RunConfig(
        target=str(repo), base="main", flow_name="artifact-probe-cancel", run_eval=True, archive=True,
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
    assert len(caught) == 1 and isinstance(caught[0], anyio.get_cancelled_exc_class())
    assert backend.cancelled is True
    public_runs = list((repo / ".daydream" / "runs").iterdir())
    assert len(public_runs) == 1
    assert json.loads((public_runs[0] / "trajectory.json").read_text())["extra"]["partial"] is True
    _, archived_run = _assert_one_published_run(repo, archive_dir)
    assert json.loads((archived_run / "manifest.json").read_text())["archive_status"] == "partial"


async def test_run_rejects_head_mismatch_on_real_worktree(
    monkeypatch: pytest.MonkeyPatch,
    silence_runner_ui: None,  # noqa: F841
    deep_target: Path, make_config: Callable[..., 'RunConfig'],
) -> None:
    called: list[str] = []
    _record = _make_recording_dispatch(called)
    for name in _DISPATCH_TARGETS:
        monkeypatch.setattr(f"daydream.runner.{name}", _record(name))
    config = make_config(deep_target, approved_head_sha="DEADBEEF")
    exit_code = await runner.run(config)
    assert exit_code == 1
    assert called == []

async def test_owner_preflight_failure_never_reaches_identity_or_backend(
    monkeypatch: pytest.MonkeyPatch,
    silence_runner_ui: None,  # noqa: F841
    tmp_path: Path,
) -> None:
    """Invalid ownership fails inside the run trace before token minting or backend creation.

    Configured App credentials make premature identity resolution observable.
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
        pr_repo="acme/widgets", cleanup=False, archive=False,
    )
    assert await runner.run(config) == 1

# --- Per-phase model resolution tests --------------------------------------

async def test_capture_review_workspace_refuses_resume_and_an_unpinned_comment_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured reviews never resume retained judgments, and comment mode requires a pinned PR base."""
    work = WorkContext(repo=tmp_path, source=tmp_path, base_branch="main", base_sha="1" * 40, head_branch="feature",
                       head_sha="2" * 40, is_ephemeral=False, run_id="run-1")
    owner: Any = None
    execution: Any = SimpleNamespace(auth=git_ops.INHERIT_GITHUB_AUTH)
    with pytest.raises(git_ops.GitError, match="cannot resume retained judgments"):
        async with runner._capture_review_workspace(
                work, RunConfig(findings_out="f.json", start_at="merge"), owner, execution):
            pytest.fail("resume yielded a captured workspace")
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", lambda *_a, **_k: _ARTIFACT_PR)
    monkeypatch.setattr("daydream.pr_review.capture_pr_base_tip", lambda *_a, **_k: None)
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda *_a, **_k: _ARTIFACT_PR)
    for selection in ({"pr_number": 7}, {"branch": "feature"}, {}):
        with pytest.raises(git_ops.GitError, match="base could not be pinned"):
            async with runner._capture_review_workspace(
                    work, RunConfig(output_mode="comment", **selection), owner, execution):
                pytest.fail("an unpinned base yielded a captured workspace")


def test_improve_backend_preflight_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or wrong audit capability, or a backend bound to another root, rejects recon."""
    source = tmp_path / "source"
    source.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    work = WorkContext(
        repo=source, source=source, base_branch="main", base_sha="1" * 40, head_branch="feature", head_sha="2" * 40,
        is_ephemeral=False, run_id="run-1",
    )
    boundary = AuditWorkspace(repo=audit, outward_symlinks=frozenset())
    context = FlowContext(
        config=RunConfig(target=str(source), flow_name="improve"), work=work, registry=build_registry(),
        audit_workspace=boundary,
    )
    for capability, bound_root, reason in ((None, audit, "missing_capability"),
                                           ("wrong-token", audit, "wrong_capability"),
                                           (AUDIT_ROOT_ISOLATION, other, "wrong_root")):
        backend = SimpleNamespace(model="mock", audit_root=bound_root)
        if capability is not None:
            backend.audit_root_isolation = capability
        monkeypatch.setattr(FlowContext, "backend_for", lambda _self, _phase, backend=backend: backend)
        with pytest.raises(AuditIsolationError) as exc_info:
            runner._preflight_improve_backends(context)
        assert (exc_info.value.backend_name, exc_info.value.phase, exc_info.value.reason) == (
            "claude", "recon", reason)

# --- Deep fix-cycle hero is followed by Model: dim line --------------------

def _seed_fix_resume(target: Path, items: list[dict[str, Any]]) -> Path:
    """Seed identity-bound completed shallow review evidence for fix-only tests."""
    from daydream.deep.artifacts import persist_review_coverage
    from daydream.deep.diff import _diff_changed_files
    from daydream.deep.orchestrator import _prepare_review_stacks
    from daydream.deep.records import stamp_item_uids
    from daydream.review_result import AnalyzedRevision, PlannedScope, ReviewCoverage
    from tests.harness.review_result import frozen_target_decisions

    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    base = _resolve_base(target, None, None)
    diff = git_ops.diff(target, base) or ""
    head = git_ops.head_sha(target)
    merge_base = git_ops.resolve_diff_merge_base(target, base, head)
    config = RunConfig(target=str(target), shallow=True)
    stacks, _, _ = _prepare_review_stacks(config, _diff_changed_files(diff), diff, target, "deep")
    coverage = ReviewCoverage("fix-resume-fixture", AnalyzedRevision(head, merge_base, diff_key(diff)),
                              [PlannedScope(stack.stack_name, stack.stack_name.split("#", 1)[0],
                                            files=tuple(sorted(stack.files))) for stack in stacks],
                              ("intent", "alternatives", "merge"))
    for stack in stacks:
        coverage.record_scope(stack.stack_name, "complete", target_decisions=frozen_target_decisions(
            target, stack, coverage.revision, diff))
    coverage.record_phase("intent", "complete")
    coverage.record_phase("alternatives", "complete", noop=True)
    coverage.record_phase("merge", "complete")
    DeepArtifact.DIFF_KEY.at(deep).write_text(diff_key(diff), encoding="utf-8")
    stamp_item_uids(items)
    DeepArtifact.MERGED_ITEMS.at(deep).write_text(json.dumps({"items": items}))
    persist_review_coverage(deep, coverage)
    return deep

def _fix_item(item_id: int = 1, *, severity: str = "medium") -> dict[str, Any]:
    """Build a validated merged item targeting the fixture's tracked file."""
    return {
        "id": item_id, "lens": "per-stack", "file": "main.py", "line": 1, "severity": severity,
        "description": f"{severity} issue in main.py", "confidence": "MEDIUM", "rationale": "rationale",
        "evidence": "main.py:1", "source_uids": [f"python:{item_id}"], "related_files": None,
    }

async def _stub_verify(*_a: Any, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
    items = json.loads(kwargs["merged_items_path"].read_text())["items"]
    return Path("/nonexistent"), {"verdicts": [], "selection": {"decisions": [
        {"item_uid": item["item_uid"], "selected": True, "reason_code": "verify_all"} for item in items
    ]}}

async def _stub_fix_verify(
    _backend: Any, _work: Any, items: list[dict[str, Any]], *_args: Any, **_kwargs: Any,
) -> list[dict[str, Any]]:
    """Resolve every canonical item for fix-cycle harnesses."""
    return [
        {"issue_id": item["id"], "verdict": "resolved", "reason": "harness"}
        for item in items
    ]

async def _noop_fix(*_a: Any, **_k: Any) -> dict[str, str]:
    return {}

# --- non_interactive threading ----------------------------------------------

# --- Deep fix-cycle commit gate semantics ----------------------------------

async def _drive_fix_cycle_failing(
    monkeypatch: pytest.MonkeyPatch, target: Path, config: RunConfig, *, script: list[Turn],
    repair_turn: Turn | None = None,
    stdin_guard_message: str | None = None, stdin_answers: list[str] | None = None,
    clipboard_is_available: bool = False,
) -> tuple[int, ScriptedBackend, ScriptedBackend, list[bool]]:
    """Drive real test-and-heal with seeded resume artifacts. Stub verify/fix, spy on commits, control gate
    answers, and reject excess healing via _BEYOND_SCRIPT. Return exit code, backends, and commit calls.

    Test execution runs as a real host process; the read-only summarizer runs on the
    TEST phase's backend. The heal turn runs on the FIX phase's backend (issue #1210) and
    is scripted through ``repair_turn`` rather than through ``script``.
    """
    _seed_fix_resume(target, [_fix_item()])
    config.test_command = "false"
    # The backend script observes phases; mute only terminal output.
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **k: None)
    monkeypatch.setattr("daydream.deep.review_steps.print_stage_progress", lambda *a, **k: None)
    monkeypatch.setattr("daydream.deep.merge_steps.print_stage_progress", lambda *a, **k: None)
    monkeypatch.setattr("daydream.deep.fix_steps.print_verification_summary", lambda *a, **k: None)
    monkeypatch.setattr("daydream.agent.console", Console(file=StringIO()))
    monkeypatch.setattr("daydream.runner.console", type("C", (), {"print": lambda *a, **kw: None})())
    test_backend = ScriptedBackend(script=[*script, _BEYOND_SCRIPT], model="test-model-xyz")
    repair_backend = ScriptedBackend(
        script=[repair_turn, _BEYOND_SCRIPT] if repair_turn is not None else [_BEYOND_SCRIPT], model="stub-model",
    )
    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, cache=None, **_kwargs: (
            test_backend if phase == "test" else repair_backend
        ),
    )
    monkeypatch.setattr("daydream.deep.fix_steps.phase_verify_recommendations", _stub_verify)
    monkeypatch.setattr("daydream.phases.phase_fix_verify", _stub_fix_verify)
    commit_calls: list[bool] = []
    async def _spy_commit(*_a: Any, **_k: Any) -> None:
        commit_calls.append(True)
    monkeypatch.setattr("daydream.deep.fix_steps.phase_fix_parallel", _noop_fix)
    monkeypatch.setattr("daydream.deep.fix_steps.phase_commit_push", _spy_commit)
    monkeypatch.setattr("daydream.phases.handoff.clipboard_available", lambda: clipboard_is_available)
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
    exit_code = await runner.run(config)
    return exit_code, test_backend, repair_backend, commit_calls

def _assert_single_handoff(repo: Path, body: str) -> None:
    """Assert exactly one handoff, whose authored text is ``body``.

    The host appends its own sections after the turn's text (the repair-attempt
    record names a run-scoped execution id, so it can never be part of a
    fixed expected body), so the authored text is matched as a prefix.
    """
    handoffs = list(repo.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8").startswith(body)

async def test_fix_cycle_failing_tests_bounded_fix_then_handoff(
    monkeypatch: pytest.MonkeyPatch, feature_branch_repo: Path, make_config: Callable[..., 'RunConfig'],
) -> None:
    """Unattended failure runs one heal attempt, then writes a handoff and exits 1.

    The real cycle follows fail/fix/fail/summary, never reads stdin, and emits one fix prompt.
    The fix prompt is served by the FIX backend; only the summary stays on TEST.
    """
    calls: list[tuple[str, str]] = []  # (kind, payload)
    def _hero_spy(_console: Any, title: Any, _description: Any) -> None:
        calls.append(("hero", title))
    def _dim_spy(_console: Any, message: Any) -> None:
        calls.append(("dim", message))
    monkeypatch.setattr("daydream.ui.print_phase_hero", _hero_spy)
    monkeypatch.setattr("daydream.ui.print_dim", _dim_spy)
    exit_code, test_backend, repair_backend, commit_calls = await _drive_fix_cycle_failing(
        monkeypatch, feature_branch_repo,
        make_config(feature_branch_repo, output_mode="loop", start_at="fix", shallow=True, assume="yes"),
        # Suite executions are host processes; reject a second TEST backend turn after the handoff.
        script=[_handoff_turn("# Handoff\n\n--yes bounded fix failure")],
        repair_turn=_FIX_TURN,
        stdin_guard_message="input() must not be called under --yes",
    )
    assert exit_code == 1
    _assert_single_handoff(feature_branch_repo, "# Handoff\n\n--yes bounded fix failure")
    assert commit_calls == [], "a commit ran despite tests failing"
    assert len(test_backend.prompts) == 1, test_backend.prompts
    assert "read-only failure-summarizer" in test_backend.prompts[0]
    assert test_backend.read_only_calls == [True]
    verdict = json.loads((feature_branch_repo / ".daydream" / "deep" / "test-verdict.json").read_text())
    assert len(verdict["attempts"]) == 2
    assert all(attempt["kind"] == "host" and not attempt["passed"] for attempt in verdict["attempts"])
    assert len(verdict["repairs"]) == 1
    # The single bounded repair turn ran on the FIX-configured backend, never on TEST.
    repair = repair_backend.prompts[0]
    assert "Analyze the failures and fix them" in repair and "CONCISE MODE" not in repair
    assert all(value in repair.lower() for value in ("generated", "migration", "package manifests", "lockfile update"))
    assert repair_backend.read_only_calls == [False]
    assert all("Analyze the failures and fix them" not in p for p in test_backend.prompts), test_backend.prompts

    awaken_idx = next(( i for i, (kind, payload) in enumerate(calls) if kind == "hero" and payload == "AWAKEN" ), None)
    assert awaken_idx is not None, f"AWAKEN hero never rendered; calls={calls!r}"
    assert awaken_idx + 1 < len(calls), "AWAKEN hero has no following call"
    next_kind, next_payload = calls[awaken_idx + 1]
    assert next_kind == "dim", (
        f"Call after AWAKEN hero was not a dim line; got {(next_kind, next_payload)!r}"
    )
    assert next_payload == "Model: test-model-xyz", (
        f"Dim line after AWAKEN hero did not echo the test backend's model; got {next_payload!r}"
    )

@pytest.mark.parametrize(
    'flow', [DaydreamRunFlow.PR, ]
)
def test_manifest_identity_preserves_mode_capabilities(tmp_path: Path, flow: DaydreamRunFlow) -> None:
    identity = capture_manifest_run_identity(
        RunConfig(fix_backend="codex", test_backend="pi"), flow, build_registry(), tmp_path,
    )
    runs_fix = flow in (DaydreamRunFlow.NORMAL, DaydreamRunFlow.DEEP, DaydreamRunFlow.PR)
    runs_test = flow in (DaydreamRunFlow.NORMAL, DaydreamRunFlow.DEEP)
    assert identity.fix_backend == ("codex" if runs_fix else None)
    assert identity.test_backend == ("pi" if runs_test else None)
    assert identity.phases.fix is runs_fix
    assert identity.phases.test is runs_test
    assert identity.phases.merge is (flow in (DaydreamRunFlow.NORMAL, DaydreamRunFlow.DEEP, DaydreamRunFlow.TTT))
    assert identity.phases.push is runs_test
    assert identity.phases.remote_ci is runs_test
