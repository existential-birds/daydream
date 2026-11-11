"""Real-runner acceptance coverage for external adapter artifact visibility."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest

from daydream import runner
from daydream.artifact_visibility import ArtifactSession, private_root_locations
from daydream.dataset import LocalRecordStore
from daydream.run_config import RunConfig
from daydream.trajectory import RunWriteSnapshot, TrajectoryDocumentReference
from tests.harness.blocked_mode import wait_for_entered
from tests.harness.claude_sdk import (
    MockResultMessage,
    patch_claude_sdk,
)
from tests.harness.dataset import read_records
from tests.harness.git_helpers import commit, git
from tests.harness.protocol_cli import ProtocolCli, install_protocol_cli

SOURCE_CANARY = "SOURCE_CANARY"
PRIOR_REASONING_CANARY = "PRIOR_REASONING_CANARY"
CURRENT_REASONING_CANARY = "CURRENT_REASONING_CANARY"
SIBLING_REASONING_CANARY = "SIBLING_REASONING_CANARY"
RESUME_CACHE_CANARY = "RESUME_CACHE_CANARY"
SANCTIONED_INPUT_CANARY = "SANCTIONED_INPUT_CANARY"
PRIVATE_ROOT_CANARY = "PRIVATE_ROOT_CANARY"
RUNTIME_STATE_CANARY = "RUNTIME_STATE_CANARY"
_OBSERVED_CANARIES = (
    SOURCE_CANARY, PRIOR_REASONING_CANARY, CURRENT_REASONING_CANARY, SIBLING_REASONING_CANARY, RESUME_CACHE_CANARY,
    SANCTIONED_INPUT_CANARY, PRIVATE_ROOT_CANARY, RUNTIME_STATE_CANARY,
)

MakeConfig = Callable[..., RunConfig]

# In-process sink for backend instances resolved by the real runner inside the
# extension flow (imported by the generated extension module, same process).
BACKEND_SINK: list[Any] = []


def _write_probe_extension(
    ext_dir: Any, sanctioned_path: Path, *, sink_backend: bool = False, read_only: bool = False, oversize: bool = False,
) -> None:
    """Register the ``artifact-visibility-probe`` extension flow.

    Two real ``run_agent`` calls against the resolved backend: the first
    response's canary lands in the live trajectory, then the second external
    invocation searches its actual model cwd. With ``oversize`` the sanctioned
    input is one byte beyond the 12,288 inline budget, so
    ``prepare_sanctioned_inputs`` must fail closed before anything spawns.
    """
    read_only_lines = "        read_only=True,\n" if read_only else ""
    prompt_kind = "READ-ONLY" if read_only else "VISIBILITY"
    sink_lines = (
        "    from tests.test_artifact_visibility_integration import BACKEND_SINK\n"
        "    BACKEND_SINK.append(backend)\n"
        if sink_backend
        else ""
    )
    run_lines = (
        # Oversize row: preparation itself must raise; nothing below runs.
        "    raise AssertionError('oversize sanctioned input must not reach run_agent')\n"
        if oversize
        else
        "    first, _discarded_continuation, _reason = await run_agent(\n"
        f"        backend, ctx.work.repo, 'FIRST {prompt_kind} PROBE',\n"
        "        phase=DaydreamPhase.REVIEW, sanctioned_inputs=prepared,\n"
        + read_only_lines
        + "    )\n"
        "    assert 'CURRENT_REASONING_CANARY' in first\n"
        "    await run_agent(\n"
        f"        backend, ctx.work.repo, 'SECOND {prompt_kind} PROBE',\n"
        "        phase=DaydreamPhase.REVIEW, sanctioned_inputs=prepared,\n"
        + read_only_lines
        + "    )\n"
    )
    ext_dir.write_module(
        "from pathlib import Path\n"
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.prompt_budget import prepare_sanctioned_inputs\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def _probe(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        "    backend = ctx.backend_for('review')\n"
        + sink_lines
        + f"    sanctioned_path = Path({str(sanctioned_path)!r})\n"
        "    prepared = prepare_sanctioned_inputs(\n"
        "        backend, ctx.work.repo, {'external evidence': sanctioned_path},\n"
        f"        read_only={read_only!r},\n"
        "    )\n"
        + run_lines
        + "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='artifact-visibility-probe', run=_probe))\n"
        "    registry.set_flow('artifact-visibility-probe', ['artifact-visibility-probe'])\n"
    )


def _seed_private_canaries(private_base: Path) -> None:
    private_base.mkdir(mode=0o700)
    (private_base / "private sentinel.txt").write_text(PRIVATE_ROOT_CANARY, encoding="utf-8")
    runtime_root = private_base / "runtime"
    runtime_root.mkdir(mode=0o700)
    sibling = runtime_root / "sibling-owner" / "runs" / "sibling-run"
    sibling.mkdir(parents=True)
    (sibling / "reasoning.txt").write_text(SIBLING_REASONING_CANARY, encoding="utf-8")
    (runtime_root / "runtime state.txt").write_text(RUNTIME_STATE_CANARY, encoding="utf-8")


def _seed_visibility_canaries(repo: Path, private_base: Path) -> None:
    source = repo / "visibility_source.py"
    source.write_text(f"VALUE = {SOURCE_CANARY!r}\n", encoding="utf-8")
    git(repo, "add", source.name)
    commit(repo, "seed artifact visibility source canary")

    prior = repo / ".daydream" / "runs" / "prior-public-run"
    prior.mkdir(parents=True)
    (prior / "trajectory.json").write_text(json.dumps({"reasoning": PRIOR_REASONING_CANARY}), encoding="utf-8")
    legacy = repo / ".daydream" / "exploration" / "legacy cache.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(RESUME_CACHE_CANARY, encoding="utf-8")
    _seed_private_canaries(private_base)


def _scan_cwd(cwd: Path) -> tuple[list[str], dict[str, bool]]:
    entries: list[str] = []
    hits = dict.fromkeys(_OBSERVED_CANARIES, False)
    pending = [cwd]
    visited: set[Path] = set()
    while pending:
        with os.scandir(pending.pop()) as children:
            for entry in children:
                path = Path(entry.path)
                entries.append(path.relative_to(cwd).as_posix())
                resolved = _scan_resolve(path)
                if resolved is None:
                    # Unresolvable symlink (dangling, loop, depth cap):
                    # still visible in the cwd listing, no content to scan.
                    continue
                if resolved.is_dir():
                    if resolved not in visited:
                        visited.add(resolved)
                        pending.append(resolved)
                elif resolved.is_file():
                    with open(resolved, "rb") as handle:
                        payload = handle.read(_SCAN_CWD_READ_CAP)
                    for canary in hits:
                        hits[canary] = hits[canary] or canary.encode() in payload
    return sorted(entries), hits


_SCAN_CWD_READ_CAP = 1 << 20
_SCAN_CWD_LINK_DEPTH = 8


def _scan_resolve(path: Path) -> Path | None:
    """Resolve a scan entry to a real file or directory, following links.

    The model-cwd privacy scan must see through symlinks: a link pointing at
    private content is still a read path from the model cwd and would trip
    the canary. Depth is bounded so a link cycle cannot loop forever.
    """
    current = path
    for _ in range(_SCAN_CWD_LINK_DEPTH):
        if not current.is_symlink():
            return current
        try:
            target_text = os.readlink(current)
        except OSError:
            return None
        target = Path(target_text)
        current = target if target.is_absolute() else current.parent / target
    return None


def _configure_cli_backend(
    backend: str, fixture_root: Path, sanctioned_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
    response_mode: Literal["success", "model_error", "process_error", "block"] = "success",
    forbidden_paths: tuple[Path, ...] = (),
) -> ProtocolCli:
    fixture = install_protocol_cli(fixture_root,
        backend,  # type: ignore[arg-type]
        response_mode=response_mode, sanctioned_files=(sanctioned_path,), forbidden_paths=forbidden_paths,
    )
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    if backend == "osprey":
        monkeypatch.setenv("OSPREY_BINARY", str(fixture.executable))
    if backend == "pi":
        settings = tmp_path / "empty pi coding agent dir"
        settings.mkdir()
        monkeypatch.setenv("PI_PROVIDER", "nous")
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(settings))
        for name in ("PI_THINKING", "PI_API_KEY", "NOUS_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    return fixture


_HIDDEN_CWD_CANARIES = (PRIOR_REASONING_CANARY, CURRENT_REASONING_CANARY, SIBLING_REASONING_CANARY, RESUME_CACHE_CANARY,
    SANCTIONED_INPUT_CANARY,
)


def _seed_world(repo: Path, tmp_path: Path) -> tuple[Path, Path, Path]:
    """Seed the source/private canaries plus one external sanctioned input.

    Returns ``(private_base, sanctioned_dir, sanctioned_path)``. The spaces in
    every name are deliberate: they exercise quoting on each transport.
    """
    private_base = (tmp_path / "test private base").resolve()
    _seed_visibility_canaries(repo, private_base)
    sanctioned_dir = tmp_path / "external sanctioned artifacts"
    sanctioned_dir.mkdir()
    sanctioned_path = (sanctioned_dir / "evidence artifact with spaces.txt").resolve()
    sanctioned_path.write_text(SANCTIONED_INPUT_CANARY, encoding="utf-8")
    return private_base, sanctioned_dir, sanctioned_path


def _assert_cwd_isolated(observation: dict[str, Any], *, extra: tuple[str, ...] = ()) -> None:
    """The model saw the source tree and nothing private: no prior/sibling/runtime bytes."""
    assert observation["cwd_canaries"][SOURCE_CANARY] is True
    for canary in _HIDDEN_CWD_CANARIES + extra:
        assert observation["cwd_canaries"][canary] is False
    assert "private sentinel.txt" not in observation["cwd_entries"]
    assert "runtime state.txt" not in observation["cwd_entries"]


def _probe_config(
    make_config: MakeConfig, repo: Path, *, flow_name: str = "artifact-visibility-probe", **overrides: Any,
) -> RunConfig:
    return make_config(repo, flow_name=flow_name, model="fixture-model", archive=True, run_eval=True, **overrides,)


async def _run_probe(config: RunConfig, private_base: Path) -> int:
    return await runner.run(config, private_roots=private_root_locations(base=private_base))


def _assert_archived_run(archive_dir: Path, trajectory: Path, *, backend: str, status: str) -> dict[str, Any]:
    """The archived bundle mirrors the explicit trajectory byte-for-byte."""
    explicit: dict[str, Any] = json.loads(trajectory.read_bytes())
    session_id = explicit["session_id"]
    assert explicit["trajectory_id"] == session_id
    assert explicit["extra"]["backend"] == backend
    archived_run = archive_dir / "runs" / session_id
    assert (archived_run / "trajectory.json").read_bytes() == trajectory.read_bytes()
    manifest = json.loads((archived_run / "manifest.json").read_bytes())
    assert manifest["session_id"] == session_id
    assert manifest["archive_status"] == status
    assert manifest["run"]["backend"] == backend
    assert json.loads((archived_run / "evaluation.json").read_bytes())["session_id"] == session_id
    return explicit


def _assert_inline_sanctioned_prompt(observation: dict[str, Any]) -> None:
    """The sanctioned bytes arrived inline: bounded, captured, never a path."""
    assert observation["prompt_canaries"][SANCTIONED_INPUT_CANARY] is True
    assert observation["prompt_bytes"] <= 12_288
    assert observation["sanctioned_reads"] == {}


def _write_improve_probe_extension(ext_dir: Any) -> None:
    """Insert one bounded probe step into the real built-in improve flow.

    The probe runs after ``recon`` on the real audit backend: it produces a
    sanctioned artifact through the source-bound artifact session and hands its
    captured bytes inline to the strict audit-root backend.
    """
    ext_dir.write_module(
        "from pathlib import Path\n"
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.prompt_budget import prepare_sanctioned_inputs\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def _audit_probe(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        "    assert ctx.audit_workspace is not None\n"
        "    backend = ctx.backend_for('audit')\n"
        "    canary_path = (\n"
        "        ctx.artifacts.daydream_dir / 'audit probe evidence with spaces.txt'\n"
        "    )\n"
        "    canary_path.parent.mkdir(parents=True, exist_ok=True)\n"
        f"    canary_path.write_text({SANCTIONED_INPUT_CANARY!r}, encoding='utf-8')\n"
        "    prepared = prepare_sanctioned_inputs(\n"
        "        backend, ctx.audit_workspace.repo, {'audit evidence': canary_path},\n"
        "        read_only=True,\n"
        "    )\n"
        "    await run_agent(\n"
        "        backend, ctx.audit_workspace.repo, 'AUDIT VISIBILITY PROBE',\n"
        "        phase=DaydreamPhase.AUDIT, read_only=True, persist_session=False,\n"
        "        sanctioned_inputs=prepared,\n"
        "    )\n"
        "def register(registry):\n"
        "    registry.register_phase(\n"
        "        FlowStep(name='artifact-visibility-audit-probe', run=_audit_probe)\n"
        "    )\n"
        "    flow = registry.flow('improve')\n"
        "    flow.insert(flow.index('recon') + 1, 'artifact-visibility-audit-probe')\n"
        "    registry.set_flow('improve', flow)\n"
    )


def _install_improve_strict_boundary(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Fake only the SDK client boundary; route the real improve prompts."""
    observations: list[dict[str, Any]] = []

    class ImproveRouterClient:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self.prompt = ""

        async def __aenter__(self) -> ImproveRouterClient:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def _response(self, prompt: str) -> MockResultMessage:
            lowered = prompt.lower()
            if "you are the **repo-survey** specialist" in lowered:
                structured: Any = {"conventions": [], "guidelines": []}
            elif "IMPROVE_RECON" in prompt:
                structured = {"languages": ["python"], "commands": [], "conventions": [],
                    "intent_docs": [], "guidelines": [],
                }
            elif "read-only improve audit specialist" in prompt:
                structured = {"findings": []}
            elif "AUDIT VISIBILITY PROBE" in prompt:
                structured = None
            else:
                raise AssertionError(f"unexpected improve prompt: {prompt[:160]}")
            return MockResultMessage(
                total_cost_usd=None, structured_output=structured, is_error=False, subtype="success",
                session_id=f"fixture-improve-session-{len(observations)}",
            )

        async def query(self, prompt: str) -> None:
            self.prompt = prompt
            cwd = Path(self.options.cwd)
            entries, cwd_canaries = _scan_cwd(cwd)
            observations.append({"backend": "claude", "effective_cwd": str(cwd.resolve()), "prompt": prompt,
                    "prompt_bytes": len(prompt.encode()),
                    "prompt_canaries": {canary: canary in prompt for canary in _OBSERVED_CANARIES},
                    "cwd_entries": entries, "cwd_canaries": cwd_canaries, "sanctioned_reads": {},
                    "options": {"cwd": self.options.cwd, "model": self.options.model,
                        "permission_mode": self.options.permission_mode,
                        "allowed_tools": list(self.options.allowed_tools or []),
                        "tools": list(getattr(self.options, "tools", None) or []),
                        "mcp_servers": dict(self.options.mcp_servers or {}),
                        "strict_mcp_config": self.options.strict_mcp_config,
                        "setting_sources": list(self.options.setting_sources or []),
                        "skills": list(getattr(self.options, "skills", None) or []),
                        "plugins": list(getattr(self.options, "plugins", None) or []), "agents": self.options.agents,
                        "resume": self.options.resume, "extra_args": dict(self.options.extra_args or {}),
                        "hook_matchers": [matcher.matcher
                            for matcher in (self.options.hooks or {}).get("PreToolUse", [])
                        ], "env_values": list((self.options.env or {}).values()),
                    },
                }
            )

        async def receive_response(self) -> Any:
            message = self._response(self.prompt)
            setattr(message, "model", self.options.model)
            yield message

    patch_claude_sdk(monkeypatch, ImproveRouterClient)
    return observations

@pytest.mark.asyncio
async def test_runner_improve_claude_strict_uses_audit_root_and_inline_sanctioned_input(
    tiny_diff_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, make_config: MakeConfig,
    archive_dir: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tiny_diff_target
    private_base = (tmp_path / "improve private base").resolve()
    _seed_visibility_canaries(repo, private_base)
    _write_improve_probe_extension(ext_dir)
    observations = _install_improve_strict_boundary(monkeypatch)

    model = "fixture-model"
    explicit_trajectory = tmp_path / "improve trajectory output.json"
    store = LocalRecordStore(tmp_path / "records")
    config = _probe_config(make_config, repo, flow_name="improve", backend="claude",
        trajectory_path=explicit_trajectory, dataset_capture=True, dataset_store_path=store.root,
    )
    # A completed child file lost before freeze leaves archived evidence and the dataset, never the run's outputs.
    freeze, dropped = ArtifactSession.freeze, []

    def freeze_without_child(session: ArtifactSession, snapshot: RunWriteSnapshot) -> Any:
        child = next(document for document in snapshot.documents if isinstance(document, TrajectoryDocumentReference))
        child.private_path.unlink()
        dropped.append(child.path.name)
        return freeze(session, snapshot)

    monkeypatch.setattr(ArtifactSession, "freeze", freeze_without_child)
    assert await _run_probe(config, private_base) == 0
    assert observations

    audit_roots = {observation["effective_cwd"] for observation in observations}
    assert len(audit_roots) == 1
    audit_root = Path(audit_roots.pop())
    assert audit_root != repo.resolve()
    assert not audit_root.is_relative_to(private_base)
    assert not audit_root.is_relative_to(repo.resolve())
    # The standalone audit snapshot is process-owned and gone after the run.
    assert not audit_root.exists()

    classified: dict[str, int] = {"survey": 0, "recon": 0, "probe": 0, "audit": 0}
    for observation in observations:
        prompt = observation["prompt"]
        if "you are the **repo-survey** specialist" in prompt.lower():
            classified["survey"] += 1
        elif "IMPROVE_RECON" in prompt:
            assert "Combined repository discovery" in prompt
            classified["recon"] += 1
        elif "AUDIT VISIBILITY PROBE" in prompt:
            classified["probe"] += 1
        else:
            assert "read-only improve audit specialist" in prompt
            classified["audit"] += 1
        _assert_cwd_isolated(observation, extra=(PRIVATE_ROOT_CANARY, RUNTIME_STATE_CANARY))
        # No runtime or source path may be named in any prompt.
        assert str(private_base) not in prompt
        assert str(repo.resolve()) not in prompt
        # No external file open: the strict-audit transport has no sanctioned
        # path pointer, only inline bytes on the probe turn.
        assert observation["sanctioned_reads"] == {}
        options = observation["options"]
        assert options["cwd"] == str(audit_root)
        assert options["model"] == model
        assert options["permission_mode"] == "bypassPermissions"
        assert options["allowed_tools"] == ["Read", "Grep", "Glob", "StructuredOutput"]
        assert options["tools"] == ["Read", "Grep", "Glob", "StructuredOutput"]
        assert options["mcp_servers"] == {}
        assert options["strict_mcp_config"] is True
        assert options["setting_sources"] == []
        assert options["skills"] == []
        assert options["plugins"] == []
        assert options["agents"] is None
        assert options["resume"] is None
        assert options["extra_args"] == {"no-session-persistence": None}
        assert options["hook_matchers"] == [".*"]
        for value in options["env_values"]:
            assert str(private_base) not in value
            assert str(repo.resolve()) not in value

    # The no-finding route exercises one combined discovery and every audit
    # category plus the bounded probe turn.
    assert classified["survey"] == 0
    assert classified["recon"] == 1
    assert classified["probe"] == 1
    assert classified["audit"] >= 1

    probe_observations = [o for o in observations if "AUDIT VISIBILITY PROBE" in o["prompt"]]
    assert len(probe_observations) == 1
    _assert_inline_sanctioned_prompt(probe_observations[0])
    assert "Sanctioned phase inputs (captured verbatim):" in probe_observations[0]["prompt"]

    explicit = _assert_archived_run(archive_dir, explicit_trajectory, backend="claude", status="complete")
    assert explicit["extra"].get("partial") is not True
    archived_children = archive_dir / "runs" / explicit["session_id"] / "trajectories"
    assert dropped and any(archived_children.iterdir()) and not (archived_children / dropped[0]).exists()
    # Incomplete trajectory evidence is never captured as a record.
    assert "lost a completed child trajectory" in capsys.readouterr().out and not read_records(store).runs


def _assert_pid_reaped(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"external process {pid} is still live after cancellation")


def _assert_single_partial_snapshot(repo: Path, archive_dir: Path, explicit_trajectory: Path, *, backend: str
) -> dict[str, Any]:
    """Exactly one honest partial snapshot is frozen across all destinations."""
    explicit_bytes = explicit_trajectory.read_bytes()
    explicit: dict[str, Any] = json.loads(explicit_bytes)
    session_id = explicit["session_id"]
    assert explicit["trajectory_id"] == session_id
    assert explicit["extra"]["partial"] is True
    # Honest: no model content was fabricated for the cancelled turn.
    assert CURRENT_REASONING_CANARY not in explicit_bytes.decode("utf-8")
    assert not explicit_trajectory.with_suffix(".json.partial").exists()
    assert not list(explicit_trajectory.parent.glob("*.tmp"))

    public_runs = [path for path in (repo / ".daydream" / "runs").iterdir() if path.name != "prior-public-run"]
    assert [path.name for path in public_runs] == [session_id]
    assert [path.name for path in (archive_dir / "runs").iterdir()] == [session_id]
    assert (public_runs[0] / "trajectory.json").read_bytes() == explicit_bytes
    _assert_archived_run(archive_dir, explicit_trajectory, backend=backend, status="partial")
    return explicit


def _assert_archive_runs_empty(archive_dir: Path) -> None:
    runs = archive_dir / "runs"
    assert not runs.exists() or not list(runs.iterdir())

@pytest.mark.asyncio
async def test_runner_external_adapter_cancellation_reaps_process_and_freezes_once(
    tiny_diff_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any,
    make_config: MakeConfig, archive_dir: Path,
) -> None:
    backend = "osprey"
    repo = tiny_diff_target
    private_base, _, sanctioned_path = _seed_world(repo, tmp_path)
    BACKEND_SINK.clear()
    _write_probe_extension(ext_dir, sanctioned_path, sink_backend=True)

    fixture = _configure_cli_backend(
        backend, tmp_path / f"{backend} cancel protocol fixture", sanctioned_path, monkeypatch, tmp_path,
        response_mode="block",
    )

    explicit_trajectory = tmp_path / f"{backend} cancelled trajectory.json"
    config = _probe_config(make_config, repo, backend=backend, trajectory_path=explicit_trajectory)
    run_task = asyncio.create_task(runner.run(config, private_roots=private_root_locations(base=private_base)))
    try:
        pid = await wait_for_entered(fixture)
        # The external process is blocked mid-turn: nothing may be frozen yet.
        assert not explicit_trajectory.exists()
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        if not run_task.done():
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await run_task

    _assert_pid_reaped(pid)
    assert len(BACKEND_SINK) == 1
    resolved_backend = BACKEND_SINK.pop()
    assert resolved_backend._transports == []
    observation = fixture.read_observations()
    assert len(observation) == 1
    assert observation[0]["response_mode"] == "block"
    assert observation[0]["process_outcome"] == "entered"
    assert observation[0]["pid"] == pid
    _assert_single_partial_snapshot(repo, archive_dir, explicit_trajectory, backend=backend)

@pytest.mark.asyncio
async def test_runner_claude_sdk_cancellation_completes_disconnect_before_freeze(
    tiny_diff_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    private_base, _, sanctioned_path = _seed_world(repo, tmp_path)
    BACKEND_SINK.clear()
    _write_probe_extension(ext_dir, sanctioned_path, sink_backend=True)

    state: dict[str, Any] = {
        "entered": asyncio.Event(), "disconnect_started_event": asyncio.Event(), "release_disconnect": asyncio.Event(),
        "queries": [], "post_disconnect_queries": 0, "response_cancelled": False, "disconnect_started": False,
        "disconnect_complete": False, "shutdown_interrupted": None,
    }

    class BlockingCancelClient:
        """SDK boundary stand-in exposing disconnect_started/complete."""

        def __init__(self, options: Any = None) -> None:
            self.options = options
            self._pump: asyncio.Task[None] | None = None

        async def __aenter__(self) -> BlockingCancelClient:
            # The SDK owns query/transport work beyond the caller's stream.
            self._pump = asyncio.create_task(self._pump_loop())
            return self

        async def _pump_loop(self) -> None:
            while not state["release_disconnect"].is_set():
                await asyncio.sleep(0.01)

        async def __aexit__(self, *_args: Any) -> None:
            state["disconnect_started"] = True
            state["disconnect_started_event"].set()
            try:
                # Emulate the SDK's shielded close: the owned shutdown (waiting
                # for the host release, then joining the owned pump task) must
                # complete even while the enclosing scope is being cancelled.
                with anyio.CancelScope(shield=True):
                    await state["release_disconnect"].wait()
                    assert self._pump is not None
                    await self._pump
            except BaseException as exc:
                state["shutdown_interrupted"] = f"{type(exc).__name__}: {exc}"
                raise
            state["disconnect_complete"] = True

        async def query(self, prompt: str) -> None:
            if state["disconnect_started"]:
                state["post_disconnect_queries"] += 1
            state["queries"].append(prompt)
            state["entered"].set()

        async def receive_response(self) -> Any:
            try:
                await state["release_disconnect"].wait()
            except asyncio.CancelledError:
                state["response_cancelled"] = True
                raise
            yield MockResultMessage(is_error=False, subtype="success")  # pragma: no cover

    patch_claude_sdk(monkeypatch, BlockingCancelClient)

    explicit_trajectory = tmp_path / "claude cancelled trajectory.json"
    config = _probe_config(make_config, repo, backend="claude", trajectory_path=explicit_trajectory)
    run_task = asyncio.create_task(runner.run(config, private_roots=private_root_locations(base=private_base)))
    try:
        await asyncio.wait_for(state["entered"].wait(), timeout=30)
        assert len(state["queries"]) == 1
        assert not explicit_trajectory.exists()
        _assert_archive_runs_empty(archive_dir)
        run_task.cancel()
        # The runner may not freeze anything while the SDK client is still
        # awaiting its owned shutdown.
        await asyncio.wait_for(state["disconnect_started_event"].wait(), timeout=30)
        assert state["response_cancelled"] is True
        assert not explicit_trajectory.exists()
        _assert_archive_runs_empty(archive_dir)
        # Release the owned shutdown; only then may the runner freeze/publish.
        state["release_disconnect"].set()
        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        state["release_disconnect"].set()
        if not run_task.done():
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await run_task

    assert state["shutdown_interrupted"] is None
    assert state["disconnect_complete"] is True
    assert state["post_disconnect_queries"] == 0
    assert len(BACKEND_SINK) == 1
    resolved_backend = BACKEND_SINK.pop()
    assert resolved_backend._active_clients == set()
    _assert_single_partial_snapshot(repo, archive_dir, explicit_trajectory, backend="claude")
