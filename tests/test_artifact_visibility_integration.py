"""Real-runner acceptance coverage for external adapter artifact visibility."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest

from daydream import runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends.claude import ClaudeAgentError
from daydream.backends.codex import CodexError
from daydream.backends.osprey import OspreyTerminalError
from daydream.backends.pi import PiError
from daydream.runner import RunConfig
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    patch_claude_sdk,
)
from tests.harness.git_helpers import bare_remote, commit, git
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
    SOURCE_CANARY,
    PRIOR_REASONING_CANARY,
    CURRENT_REASONING_CANARY,
    SIBLING_REASONING_CANARY,
    RESUME_CACHE_CANARY,
    SANCTIONED_INPUT_CANARY,
    PRIVATE_ROOT_CANARY,
    RUNTIME_STATE_CANARY,
)

MakeConfig = Callable[..., RunConfig]

# In-process sink for backend instances resolved by the real runner inside the
# extension flow (imported by the generated extension module, same process).
BACKEND_SINK: list[Any] = []


def _write_probe_extension(
    ext_dir: Any,
    sanctioned_path: Path,
    *,
    sink_backend: bool = False,
    read_only: bool = False,
) -> None:
    read_only_lines = "        read_only=True,\n" if read_only else ""
    prompt_kind = "READ-ONLY" if read_only else "VISIBILITY"
    sink_lines = (
        "    from tests.test_artifact_visibility_integration import BACKEND_SINK\n"
        "    BACKEND_SINK.append(backend)\n"
        if sink_backend
        else ""
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
        "def register(registry):\n"
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
    legacy = repo / ".daydream" / "resume" / "legacy cache.json"
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


def _install_claude_boundary(
    monkeypatch: pytest.MonkeyPatch,
    sanctioned_path: Path,
    *,
    model_error: bool = False,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []

    class ObservingClient:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self.prompt = ""

        async def __aenter__(self) -> ObservingClient:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            self.prompt = prompt
            cwd = Path(self.options.cwd)
            entries, cwd_canaries = _scan_cwd(cwd)
            sanctioned_reads: dict[str, str] = {}
            if str(sanctioned_path) in prompt:
                metadata = sanctioned_path.lstat()
                assert stat.S_ISREG(metadata.st_mode)
                sanctioned_reads[str(sanctioned_path)] = hashlib.sha256(sanctioned_path.read_bytes()).hexdigest()
            observations.append(
                {
                    "backend": "claude",
                    "response_mode": "model_error" if model_error else "success",
                    "effective_cwd": str(cwd.resolve()),
                    "prompt_bytes": len(prompt.encode()),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_canaries": {
                        canary: canary in prompt for canary in _OBSERVED_CANARIES
                    },
                    "cwd_entries": entries,
                    "cwd_canaries": cwd_canaries,
                    "sanctioned_reads": sanctioned_reads,
                    "model": self.options.model,
                    "permission_mode": self.options.permission_mode,
                    "allowed_tools": self.options.allowed_tools,
                    "setting_sources": self.options.setting_sources,
                    "resume": self.options.resume,
                }
            )

        async def receive_response(self) -> Any:
            message = MockAssistantMessage(content=[MockTextBlock(text=CURRENT_REASONING_CANARY)])
            setattr(message, "model", self.options.model)
            yield message
            yield MockResultMessage(
                total_cost_usd=None,
                is_error=model_error,
                result="fixture failure" if model_error else None,
                subtype="error" if model_error else "success",
                session_id=f"fixture-claude-session-{len(observations)}",
            )

    patch_claude_sdk(monkeypatch, ObservingClient)
    return observations


def _configure_cli_backend(
    backend: str,
    fixture_root: Path,
    sanctioned_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    response_mode: Literal["success", "model_error", "process_error", "block"] = "success",
    forbidden_paths: tuple[Path, ...] = (),
) -> ProtocolCli:
    fixture = install_protocol_cli(
        fixture_root,
        backend,  # type: ignore[arg-type]
        response_mode=response_mode,
        sanctioned_files=(sanctioned_path,),
        forbidden_paths=forbidden_paths,
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


_HIDDEN_CWD_CANARIES = (
    PRIOR_REASONING_CANARY,
    CURRENT_REASONING_CANARY,
    SIBLING_REASONING_CANARY,
    RESUME_CACHE_CANARY,
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
    make_config: MakeConfig,
    repo: Path,
    *,
    flow_name: str = "artifact-visibility-probe",
    **overrides: Any,
) -> RunConfig:
    return make_config(
        repo,
        flow_name=flow_name,
        model="fixture-model",
        archive=True,
        run_eval=True,
        **overrides,
    )


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


def _assert_external_observations(
    observations: list[dict[str, Any]],
    *,
    backend: str,
    repo: Path,
    sanctioned_path: Path,
) -> None:
    expected_digest = hashlib.sha256(sanctioned_path.read_bytes()).hexdigest()
    assert len(observations) == 2
    for observation in observations:
        assert observation["backend"] == backend
        assert observation["effective_cwd"] == str(repo.resolve())
        _assert_cwd_isolated(observation)
        assert observation["prompt_canaries"][CURRENT_REASONING_CANARY] is False
        assert observation["prompt_canaries"][SANCTIONED_INPUT_CANARY] is False
        assert observation["sanctioned_reads"] == {str(sanctioned_path): expected_digest}
        assert not any(entry == ".daydream" or entry.startswith(".daydream/") for entry in observation["cwd_entries"])


def _assert_frozen_outputs(
    repo: Path,
    archive_dir: Path,
    explicit_trajectory: Path,
    dump_dir: Path,
    *,
    backend: str,
    model: str,
) -> None:
    explicit_bytes = explicit_trajectory.read_bytes()
    explicit = json.loads(explicit_bytes)
    session_id = explicit["session_id"]
    assert explicit["trajectory_id"] == session_id
    assert CURRENT_REASONING_CANARY in explicit_trajectory.read_text(encoding="utf-8")

    public_run = repo / ".daydream" / "runs" / session_id
    archived_run = archive_dir / "runs" / session_id
    assert (public_run / "trajectory.json").read_bytes() == explicit_bytes
    assert (archived_run / "trajectory.json").read_bytes() == explicit_bytes
    assert (dump_dir / "trajectory.json").read_bytes() == explicit_bytes

    archive_manifest = json.loads((archived_run / "manifest.json").read_bytes())
    dump_manifest = json.loads((dump_dir / "manifest.json").read_bytes())
    archive_evaluation = json.loads((archived_run / "evaluation.json").read_bytes())
    dump_evaluation = json.loads((dump_dir / "evaluation.json").read_bytes())
    assert archive_manifest == dump_manifest
    assert archive_evaluation == dump_evaluation
    assert archive_manifest["session_id"] == session_id
    assert archive_manifest["run"]["backend"] == backend
    assert archive_manifest["archive_status"] == "complete"
    assert archive_evaluation["session_id"] == session_id
    assert archive_manifest["git"]["source_path"] == str(repo.resolve())
    assert explicit["extra"]["backend"] == backend
    assert explicit["agent"]["model_name"] == model

    agent_steps = [step for step in explicit["steps"] if step["source"] == "agent"]
    assert len(agent_steps) == 2
    assert {step["model_name"] for step in agent_steps} == {model}


@pytest.mark.parametrize("backend", ["claude", "codex", "pi", "osprey"])
@pytest.mark.asyncio
async def test_runner_ordinary_adapter_two_turn_visibility(
    backend: str,
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    private_base, _, sanctioned_path = _seed_world(repo, tmp_path)
    _write_probe_extension(ext_dir, sanctioned_path)

    model = "fixture-model"
    cli_fixture: ProtocolCli | None = None
    claude_observations: list[dict[str, Any]] = []
    if backend == "claude":
        claude_observations = _install_claude_boundary(monkeypatch, sanctioned_path)
    else:
        cli_fixture = _configure_cli_backend(
            backend,
            tmp_path / f"{backend} external protocol fixture",
            sanctioned_path,
            monkeypatch,
            tmp_path,
        )

    explicit_trajectory = tmp_path / f"{backend} explicit trajectory output.json"
    dump_dir = tmp_path / f"{backend} explicit artifact dump"
    result = await _run_probe(
        _probe_config(
            make_config,
            repo,
            backend=backend,
            trajectory_path=explicit_trajectory,
            dump_artifacts=str(dump_dir),
        ),
        private_base,
    )

    assert result == 0
    observations = claude_observations if cli_fixture is None else cli_fixture.read_observations()
    _assert_external_observations(observations, backend=backend, repo=repo, sanctioned_path=sanctioned_path)
    _assert_frozen_outputs(repo, archive_dir, explicit_trajectory, dump_dir, backend=backend, model=model)


_MODEL_FAILURES: dict[str, type[Exception]] = {
    "codex": CodexError,
    "pi": PiError,
    "osprey": OspreyTerminalError,
    "claude": ClaudeAgentError,
}
_MODEL_FAILURE_MESSAGES = {
    "codex": "fixture failure",
    "pi": "fixture failure",
    "osprey": "outcome 'failed': exit_code=0",
    "claude": "fixture failure",
}


@pytest.mark.parametrize("backend", ["codex", "pi", "osprey", "claude"])
@pytest.mark.asyncio
async def test_runner_external_adapter_model_failure_preserves_partial_evidence(
    backend: str,
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    private_base, _, sanctioned_path = _seed_world(repo, tmp_path)
    _write_probe_extension(ext_dir, sanctioned_path)

    model = "fixture-model"
    cli_fixture: ProtocolCli | None = None
    claude_observations: list[dict[str, Any]] = []
    if backend == "claude":
        claude_observations = _install_claude_boundary(monkeypatch, sanctioned_path, model_error=True)
    else:
        cli_fixture = _configure_cli_backend(
            backend,
            tmp_path / f"{backend} model failure protocol fixture",
            sanctioned_path,
            monkeypatch,
            tmp_path,
            response_mode="model_error",
        )
    if backend == "pi":
        monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "1")

    explicit_trajectory = tmp_path / f"{backend} failed trajectory output.json"
    with pytest.raises(_MODEL_FAILURES[backend], match=_MODEL_FAILURE_MESSAGES[backend]) as raised:
        await _run_probe(
            _probe_config(make_config, repo, backend=backend, trajectory_path=explicit_trajectory),
            private_base,
        )

    assert type(raised.value) is _MODEL_FAILURES[backend]
    observations = claude_observations if cli_fixture is None else cli_fixture.read_observations()
    assert len(observations) == 1
    observation = observations[0]
    assert observation["response_mode"] == "model_error"
    assert observation["effective_cwd"] == str(repo.resolve())
    assert observation["cwd_canaries"][SOURCE_CANARY] is True
    assert observation["cwd_canaries"][PRIOR_REASONING_CANARY] is False
    assert observation["cwd_canaries"][RESUME_CACHE_CANARY] is False
    if backend == "claude":
        assert observation["model"] == model
    else:
        argv = observation["argv"]
        assert argv[argv.index("--model") + 1] == model

    explicit = _assert_archived_run(archive_dir, explicit_trajectory, backend=backend, status="partial")
    assert explicit["extra"]["partial"] is True
    assert not explicit_trajectory.with_suffix(".json.partial").exists()
    public_run = repo / ".daydream" / "runs" / explicit["session_id"]
    assert (public_run / "trajectory.json").read_bytes() == explicit_trajectory.read_bytes()


def _release_fifo_invocations(
    fixture: ProtocolCli,
    expected: int,
    stop: threading.Event,
    failures: list[BaseException],
) -> None:
    seen_pids: set[int] = set()
    deadline = time.monotonic() + 15
    try:
        while len(seen_pids) < expected:
            if stop.is_set():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("blocked protocol invocation did not enter")
            try:
                entered = json.loads(fixture.entered.read_text(encoding="utf-8"))
                pid = entered["pid"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                stop.wait(0.01)
                continue
            if not isinstance(pid, int) or pid in seen_pids:
                stop.wait(0.01)
                continue
            with fixture.release.open("wb", buffering=0) as release:
                release.write(b"release")
            seen_pids.add(pid)
    except BaseException as exc:
        failures.append(exc)


async def _run_blocked_codex(*, fixture: ProtocolCli, config: RunConfig, private_base: Path) -> int:
    stop = threading.Event()
    failures: list[BaseException] = []
    releaser = threading.Thread(
        target=_release_fifo_invocations,
        args=(fixture, 2, stop, failures),
        name="artifact-visibility-fifo-releaser",
        daemon=True,
    )
    releaser.start()
    try:
        with anyio.fail_after(20):
            return await runner.run(config, private_roots=private_root_locations(base=private_base))
    finally:
        stop.set()
        releaser.join(timeout=5)
        assert not releaser.is_alive()
        assert failures == []


def _tracked_source_state(repo: Path) -> dict[str, Any]:
    tracked = git(repo, "ls-files").splitlines()
    return {
        "head": git(repo, "rev-parse", "HEAD"),
        "refs": git(repo, "show-ref"),
        "index": git(repo, "ls-files", "--stage"),
        "diff": git(repo, "diff", "--binary", "HEAD"),
        "bytes": {name: (repo / name).read_bytes() for name in tracked},
    }


@pytest.mark.asyncio
async def test_two_ephemeral_runs_for_one_source_cannot_observe_sibling_runtime(
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    first_private = (tmp_path / "first private owner").resolve()
    second_private = (tmp_path / "second private owner").resolve()
    _seed_visibility_canaries(repo, first_private)
    _seed_private_canaries(second_private)
    origin = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-u", "origin", "main")
    git(repo, "push", "-u", "origin", "feature")
    git(repo, "fetch", "origin")
    source_before = _tracked_source_state(repo)

    sanctioned_dir = tmp_path / "ephemeral sanctioned artifacts"
    sanctioned_dir.mkdir()
    sanctioned_path = (sanctioned_dir / "ephemeral evidence with spaces.txt").resolve()
    sanctioned_path.write_text(SANCTIONED_INPUT_CANARY, encoding="utf-8")
    _write_probe_extension(ext_dir, sanctioned_path)

    trajectories: list[Path] = []
    fixtures: list[ProtocolCli] = []
    private_bases = first_private, second_private
    for index, private_base in enumerate(private_bases, start=1):
        fixture = _configure_cli_backend(
            "codex",
            tmp_path / f"blocked codex fixture {index}",
            sanctioned_path,
            monkeypatch,
            tmp_path,
            response_mode="block",
        )
        trajectory = tmp_path / f"ephemeral run {index} trajectory.json"
        result = await _run_blocked_codex(
            fixture=fixture,
            config=_probe_config(
                make_config, repo, backend="codex", force_worktree=True, trajectory_path=trajectory
            ),
            private_base=private_base,
        )
        assert result == 0
        trajectories.append(trajectory)
        fixtures.append(fixture)

    source_after = _tracked_source_state(repo)
    assert source_after == source_before

    observations_by_run = [fixture.read_observations() for fixture in fixtures]
    assert [len(observations) for observations in observations_by_run] == [2, 2]
    model_cwds: list[Path] = []
    expected_digest = hashlib.sha256(sanctioned_path.read_bytes()).hexdigest()
    for index, observations in enumerate(observations_by_run):
        run_cwds = {Path(observation["effective_cwd"]) for observation in observations}
        assert len(run_cwds) == 1
        model_cwd = run_cwds.pop()
        model_cwds.append(model_cwd)
        assert model_cwd != repo.resolve()
        assert model_cwd.is_relative_to(private_bases[index] / "workspaces")
        assert not model_cwd.exists()
        for observation in observations:
            assert observation["response_mode"] == "block"
            assert observation["process_outcome"] == "block"
            _assert_cwd_isolated(observation)
            assert observation["sanctioned_reads"] == {str(sanctioned_path): expected_digest}
            serialized = json.dumps(observation, sort_keys=True)
            assert str(private_bases[1 - index] / "runtime") not in serialized

    assert model_cwds[0] != model_cwds[1]
    session_ids = {json.loads(path.read_bytes())["session_id"] for path in trajectories}
    assert len(session_ids) == 2
    for trajectory in trajectories:
        payload = json.loads(trajectory.read_bytes())
        session_id = payload["session_id"]
        assert payload["trajectory_id"] == session_id
        assert (repo / ".daydream" / "runs" / session_id / "trajectory.json").read_bytes() == trajectory.read_bytes()
        archived = archive_dir / "runs" / session_id
        assert (archived / "trajectory.json").read_bytes() == trajectory.read_bytes()
        assert json.loads((archived / "evaluation.json").read_bytes())["session_id"] == session_id


def _assert_inline_sanctioned_prompt(observation: dict[str, Any]) -> None:
    """The sanctioned bytes arrived inline: bounded, captured, never a path."""
    assert observation["prompt_canaries"][SANCTIONED_INPUT_CANARY] is True
    assert observation["prompt_bytes"] <= 12_288
    assert observation["sanctioned_reads"] == {}


def _assert_no_forbidden_path_observed(observation: dict[str, Any]) -> None:
    """No source/runtime path reached the child's stdin, argv, or environment."""
    for hits in observation["forbidden_path_hits"].values():
        assert hits == {"argv": False, "stdin": False, "env": False}


@pytest.mark.asyncio
async def test_runner_codex_read_only_uses_clone_and_inline_sanctioned_input(
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    private_base, _, sanctioned_path = _seed_world(repo, tmp_path)
    _write_probe_extension(ext_dir, sanctioned_path, read_only=True)
    fixture = _configure_cli_backend(
        "codex",
        tmp_path / "codex read-only protocol fixture",
        sanctioned_path,
        monkeypatch,
        tmp_path,
        forbidden_paths=(repo.resolve(), private_base),
    )

    source_before = _tracked_source_state(repo)
    explicit_trajectory = tmp_path / "read-only trajectory output.json"
    result = await _run_probe(
        _probe_config(make_config, repo, backend="codex", trajectory_path=explicit_trajectory),
        private_base,
    )
    assert result == 0

    source_after = _tracked_source_state(repo)
    assert source_after == source_before

    observations = fixture.read_observations()
    assert len(observations) == 2
    clone_paths: list[Path] = []
    for observation in observations:
        assert observation["backend"] == "codex"
        assert observation["response_mode"] == "success"
        argv = observation["argv"]
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        # Disposable clone: both the child's own cwd and the --cd target, never
        # the source, and never the private runtime root.
        clone = Path(observation["effective_cwd"])
        assert Path(observation["inherited_cwd"]) == clone
        assert argv[argv.index("--cd") + 1] == str(clone)
        clone_paths.append(clone)
        assert clone != repo.resolve()
        assert not clone.is_relative_to(repo.resolve())
        assert not clone.is_relative_to(private_base)
        _assert_cwd_isolated(observation, extra=(RUNTIME_STATE_CANARY,))
        # The disposable clone has no source remote.
        assert observation["git_remote_count"] == 0
        # Sanctioned bytes inline; source/runtime paths absent from stdin/argv/env.
        assert observation["stdin_bytes"] == observation["prompt_bytes"]
        _assert_inline_sanctioned_prompt(observation)
        _assert_no_forbidden_path_observed(observation)

    # The two sequential read-only calls each rebuilt (and removed) their clone.
    assert clone_paths[0] != clone_paths[1]
    for clone in clone_paths:
        assert not clone.exists()
    # No disposable checkout directories survive anywhere.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("daydream-codex-read-only-")]

    explicit = _assert_archived_run(archive_dir, explicit_trajectory, backend="codex", status="complete")
    assert explicit["extra"].get("partial") is not True


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
        "    registry.insert_after(\n"
        "        'improve', anchor='recon', step='artifact-visibility-audit-probe'\n"
        "    )\n"
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
                structured = {"languages": ["python"], "commands": [], "conventions": [], "intent_docs": []}
            elif "read-only improve audit specialist" in prompt:
                structured = {"findings": []}
            elif "AUDIT VISIBILITY PROBE" in prompt:
                structured = None
            else:
                raise AssertionError(f"unexpected improve prompt: {prompt[:160]}")
            return MockResultMessage(
                total_cost_usd=None,
                structured_output=structured,
                is_error=False,
                subtype="success",
                session_id=f"fixture-improve-session-{len(observations)}",
            )

        async def query(self, prompt: str) -> None:
            self.prompt = prompt
            cwd = Path(self.options.cwd)
            entries, cwd_canaries = _scan_cwd(cwd)
            observations.append(
                {
                    "backend": "claude",
                    "effective_cwd": str(cwd.resolve()),
                    "prompt": prompt,
                    "prompt_bytes": len(prompt.encode()),
                    "prompt_canaries": {
                        canary: canary in prompt for canary in _OBSERVED_CANARIES
                    },
                    "cwd_entries": entries,
                    "cwd_canaries": cwd_canaries,
                    "sanctioned_reads": {},
                    "options": {
                        "cwd": self.options.cwd,
                        "model": self.options.model,
                        "permission_mode": self.options.permission_mode,
                        "allowed_tools": list(self.options.allowed_tools or []),
                        "tools": list(getattr(self.options, "tools", None) or []),
                        "mcp_servers": dict(self.options.mcp_servers or {}),
                        "strict_mcp_config": self.options.strict_mcp_config,
                        "setting_sources": list(self.options.setting_sources or []),
                        "skills": list(getattr(self.options, "skills", None) or []),
                        "plugins": list(getattr(self.options, "plugins", None) or []),
                        "agents": self.options.agents,
                        "resume": self.options.resume,
                        "extra_args": dict(self.options.extra_args or {}),
                        "hook_matchers": [
                            matcher.matcher
                            for matcher in (self.options.hooks or {}).get(
                                "PreToolUse", []
                            )
                        ],
                        "env_values": list((self.options.env or {}).values()),
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
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    private_base = (tmp_path / "improve private base").resolve()
    _seed_visibility_canaries(repo, private_base)
    _write_improve_probe_extension(ext_dir)
    observations = _install_improve_strict_boundary(monkeypatch)

    model = "fixture-model"
    explicit_trajectory = tmp_path / "improve trajectory output.json"
    config = _probe_config(
        make_config, repo, flow_name="improve", backend="claude", trajectory_path=explicit_trajectory
    )
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

    # The no-finding route still exercised the real recon and every audit
    # category plus the bounded probe turn.
    assert classified["survey"] == 1
    assert classified["recon"] == 1
    assert classified["probe"] == 1
    assert classified["audit"] >= 1

    probe_observations = [o for o in observations if "AUDIT VISIBILITY PROBE" in o["prompt"]]
    assert len(probe_observations) == 1
    _assert_inline_sanctioned_prompt(probe_observations[0])
    assert "Sanctioned phase inputs (captured verbatim):" in probe_observations[0]["prompt"]

    explicit = _assert_archived_run(archive_dir, explicit_trajectory, backend="claude", status="complete")
    assert explicit["extra"].get("partial") is not True


def _write_osprey_sandbox_extension(
    ext_dir: Any,
    sanctioned_path: Path,
    allowed_roots: tuple[Path, ...],
    osprey_binary: Path,
) -> None:
    """Extension constructs the real sandboxed OspreyBackend — no factory patch."""
    roots_literal = ", ".join(f"Path({str(root)!r})" for root in allowed_roots)
    ext_dir.write_module(
        "from pathlib import Path\n"
        "from daydream.agent import run_agent\n"
        "from daydream.backends.osprey import OspreyBackend\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.prompt_budget import prepare_sanctioned_inputs\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "from tests.test_artifact_visibility_integration import BACKEND_SINK\n"
        "async def _probe(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        f"    backend = OspreyBackend(\n"
        "        'fixture-model',\n"
        "        sandbox=True,\n"
        f"        allowed_roots=[{roots_literal}],\n"
        f"        osprey_binary={str(osprey_binary)!r},\n"
        "    )\n"
        "    BACKEND_SINK.append(backend)\n"
        f"    sanctioned_path = Path({str(sanctioned_path)!r})\n"
        "    prepared = prepare_sanctioned_inputs(\n"
        "        backend, ctx.work.repo, {'external evidence': sanctioned_path},\n"
        "        read_only=False,\n"
        "    )\n"
        "    first, _discarded_continuation, _reason = await run_agent(\n"
        "        backend, ctx.work.repo, 'FIRST SANDBOX PROBE',\n"
        "        phase=DaydreamPhase.REVIEW, sanctioned_inputs=prepared,\n"
        "    )\n"
        "    assert 'CURRENT_REASONING_CANARY' in first\n"
        "    await run_agent(\n"
        "        backend, ctx.work.repo, 'SECOND SANDBOX PROBE',\n"
        "        phase=DaydreamPhase.REVIEW, sanctioned_inputs=prepared,\n"
        "    )\n"
        "def register(registry):\n"
        "    registry.register_phase(\n"
        "        FlowStep(name='artifact-visibility-osprey-sandbox', run=_probe)\n"
        "    )\n"
        "    registry.set_flow(\n"
        "        'artifact-visibility-osprey-sandbox',\n"
        "        ['artifact-visibility-osprey-sandbox'],\n"
        "    )\n"
    )


@pytest.mark.asyncio
async def test_runner_extension_osprey_sandbox_preserves_roots_and_inlines_input(
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    private_base, sanctioned_dir, sanctioned_path = _seed_world(repo, tmp_path)
    BACKEND_SINK.clear()
    osprey_fixture = install_protocol_cli(
        tmp_path / "osprey sandbox protocol fixture",
        "osprey",
        response_mode="success",
        sanctioned_files=(sanctioned_path,),
        forbidden_paths=(repo.resolve(), private_base),
    )
    # Only pre-approved non-runtime roots are handed to the constructor.
    _write_osprey_sandbox_extension(
        ext_dir,
        sanctioned_path,
        allowed_roots=(sanctioned_dir,),
        osprey_binary=osprey_fixture.executable,
    )

    explicit_trajectory = tmp_path / "sandbox trajectory output.json"
    dump_dir = tmp_path / "sandbox explicit artifact dump"
    config = _probe_config(
        make_config,
        repo,
        flow_name="artifact-visibility-osprey-sandbox",
        backend="osprey",
        trajectory_path=explicit_trajectory,
        dump_artifacts=str(dump_dir),
    )
    assert await _run_probe(config, private_base) == 0

    observations = osprey_fixture.read_observations()
    assert len(observations) == 2
    for observation in observations:
        assert observation["backend"] == "osprey"
        assert observation["response_mode"] == "success"
        argv = observation["argv"]
        assert "--sandbox" in argv
        allowed_roots = [argv[index + 1] for index, flag in enumerate(argv) if flag == "--allowed-root"]
        # Explicit non-runtime roots preserved byte-for-byte; nothing else.
        assert allowed_roots == [str(sanctioned_dir)]
        assert observation["effective_cwd"] == str(repo.resolve())
        _assert_cwd_isolated(observation)
        # Sandbox row: sanctioned bytes inline, no sanctioned file open.
        assert observation["stdin_bytes"] == 0
        _assert_inline_sanctioned_prompt(observation)

    # The extension-constructed backend is the real class, and the run's ATIF
    # identity matches it.
    assert len(BACKEND_SINK) == 1
    backend = BACKEND_SINK.pop()
    assert type(backend).__name__ == "OspreyBackend"
    assert backend.sandbox is True
    assert backend.allowed_roots == (str(sanctioned_dir),)

    _assert_frozen_outputs(repo, archive_dir, explicit_trajectory, dump_dir, backend="osprey", model="fixture-model")


async def _wait_for_entered(fixture: ProtocolCli, timeout_s: float = 30.0) -> int:
    """Wait for the blocked fixture's atomic entered marker; return its pid."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            entered = json.loads(fixture.entered.read_text(encoding="utf-8"))
            pid = entered["pid"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            await anyio.sleep(0.02)
            continue
        if isinstance(pid, int):
            return pid
        await anyio.sleep(0.02)
    raise AssertionError("blocked protocol invocation never published entered")


def _assert_pid_reaped(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"external process {pid} is still live after cancellation")


def _assert_single_partial_snapshot(
    repo: Path, archive_dir: Path, explicit_trajectory: Path, *, backend: str
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


@pytest.mark.parametrize("backend", ["codex", "pi", "osprey"])
@pytest.mark.asyncio
async def test_runner_external_adapter_cancellation_reaps_process_and_freezes_once(
    backend: str,
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    repo = tiny_diff_target
    private_base, _, sanctioned_path = _seed_world(repo, tmp_path)
    BACKEND_SINK.clear()
    _write_probe_extension(ext_dir, sanctioned_path, sink_backend=True)

    fixture = _configure_cli_backend(
        backend,
        tmp_path / f"{backend} cancel protocol fixture",
        sanctioned_path,
        monkeypatch,
        tmp_path,
        response_mode="block",
    )
    if backend == "pi":
        monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "1")

    explicit_trajectory = tmp_path / f"{backend} cancelled trajectory.json"
    config = _probe_config(make_config, repo, backend=backend, trajectory_path=explicit_trajectory)
    run_task = asyncio.create_task(runner.run(config, private_roots=private_root_locations(base=private_base)))
    try:
        pid = await _wait_for_entered(fixture)
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
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: MakeConfig,
    archive_dir: Path,
) -> None:
    """Adjudicated Claude cancellation: disconnect completes, then freeze."""
    repo = tiny_diff_target
    private_base, _, sanctioned_path = _seed_world(repo, tmp_path)
    BACKEND_SINK.clear()
    _write_probe_extension(ext_dir, sanctioned_path, sink_backend=True)

    state: dict[str, Any] = {
        "entered": asyncio.Event(),
        "disconnect_started_event": asyncio.Event(),
        "release_disconnect": asyncio.Event(),
        "queries": [],
        "post_disconnect_queries": 0,
        "response_cancelled": False,
        "disconnect_started": False,
        "disconnect_complete": False,
        "shutdown_interrupted": None,
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
