"""Pin launch/exit contracts; test_harness_e2e covers complete intercepted rollouts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import verifiers.v1 as vf
from conftest import PROJECT_ROOT, FakeRuntime, passed_gate_report
from verifiers.v1.graph import MessageNode
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntimeInfo

from daydream_review.backends import STRATEGIES
from daydream_review.harness import DaydreamReviewHarness, DaydreamReviewHarnessConfig
from daydream_review.taskset import (
    DaydreamReviewConfig,
    DaydreamReviewTask,
    DaydreamReviewTaskset,
)

ENDPOINT = "http://127.0.0.1:54321/v1"
SECRET = "rollout-secret"
MODEL = "some-org/some-policy-model"


def _task(fixture_manifest_path: Path) -> DaydreamReviewTask:
    # Supply passed gate evidence so these tests reach launch and sealing.
    with passed_gate_report() as gate_path:
        taskset = DaydreamReviewTaskset(
            DaydreamReviewConfig(id="daydream-review", manifest_path=fixture_manifest_path, gate_report_path=gate_path)
        )
        return list(taskset.load())[0]


def _trace(task: DaydreamReviewTask, *, turns: int = 1) -> vf.Trace:
    """Build a node-based trace compatible with vendored verifiers; zero turns exercises capture loss."""
    trace: vf.Trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), agent=vf.AgentInfo(model=MODEL),
    )
    for index in range(turns):
        parent = None if index == 0 else len(trace.nodes) - 1
        trace.nodes.append(MessageNode(parent=parent, message={"role": "user", "content": "go"}, sampled=False))
        trace.nodes.append(
            MessageNode(parent=len(trace.nodes) - 1, message={"role": "assistant", "content": "ok"}, sampled=True)
        )
    return trace


def _ctx() -> vf.ModelContext:
    return vf.ModelContext(model=MODEL, client=None, sampling=vf.Sampling())


def _archive_with_trajectory(archive_root: str, *, final_metrics: object) -> dict[str, bytes]:
    session = f"{archive_root}/runs/session-1"
    return {f"{session}/trajectory.json": json.dumps({"final_metrics": final_metrics}).encode()}


class _SessionsListingRuntime(FakeRuntime):
    """A FakeRuntime whose `ls` of the archive reports ``self.sessions``."""

    sessions: list[str]

    async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        await super().run(argv, env)
        if argv[:2] == ["sh", "-c"] and argv[2].startswith("ls -1 "):
            return vf.ProgramResult(exit_code=0, stdout="\n".join(self.sessions), stderr="")
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


class _ArchiveRuntime(_SessionsListingRuntime):
    """A FakeRuntime whose `ls` of the archive reports one session dir."""

    def __init__(self, *, exit_code: int, sessions: list[str], files: dict[str, bytes] | None = None) -> None:
        super().__init__(exit_code=exit_code, files=files)
        self.sessions = sessions


@pytest.mark.parametrize("backend", sorted(STRATEGIES))
async def test_launch_passes_the_selected_backend_to_the_cli(backend: str, fixture_manifest_path: Path) -> None:
    task = _task(fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig(backend=backend, fanout_concurrency=3))
    runtime = FakeRuntime(exit_code=0)

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    (argv, env), = runtime.programs
    assert "daydream" in argv
    assert argv[argv.index("--backend") + 1] == backend
    assert argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--base") + 1] == task.data.base_sha
    assert argv[-1] == "/work/repo"
    # Fix rollouts require --yes; review/comment modes conflict with that path.
    assert "--yes" in argv and "--non-interactive" in argv
    assert "--review" not in argv and "--comment" not in argv
    assert env["DAYDREAM_ARCHIVE_DIR"] == "/rollout/archive"
    assert env["HOME"] == "/rollout"
    assert trace.info["daydream_backend"] == backend
    assert trace.info["daydream_exit_code"] == 0

async def test_launch_carries_extra_args_before_the_target(fixture_manifest_path: Path) -> None:
    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(
        DaydreamReviewHarnessConfig(backend="codex", extra_args=["--reasoning-effort", "high"])
    )
    runtime = FakeRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    (argv, _), = runtime.programs
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
    assert argv.index("--reasoning-effort") < argv.index("/work/repo")

async def test_launch_unsets_ambient_github_credentials(fixture_manifest_path: Path) -> None:
    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = FakeRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    (argv, _), = runtime.programs
    assert argv[:10] == [
        "env", "-u", "DAYDREAM_APP_ID", "-u", "DAYDREAM_APP_PRIVATE_KEY", "-u", "GH_TOKEN", "-u", "GITHUB_TOKEN",
        "daydream",
    ]

async def test_launch_clears_operator_observability_env(fixture_manifest_path: Path) -> None:
    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = FakeRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    (_, env), = runtime.programs
    assert env["DAYDREAM_TRACE_TO"] == ""
    assert env["DAYDREAM_TRAJECTORY_HUB_REPO"] == ""

async def test_launch_stops_the_trace_when_a_completed_run_exits_nonzero(fixture_manifest_path: Path) -> None:
    """Tests still red after the fix pass is an outcome to score, not a crash."""
    task = _task(fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchiveRuntime(exit_code=1, sessions=["session-1"],
        files=_archive_with_trajectory("/rollout/archive", final_metrics={"cost_usd": 1.0}),
    )

    result = await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert result.exit_code == 1
    assert trace.stop_condition == "daydream_completed_nonzero"

async def test_launch_leaves_a_crash_to_raise(fixture_manifest_path: Path) -> None:
    """No artifacts means infrastructure failure: let HarnessError fire."""
    task = _task(fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchiveRuntime(exit_code=1, sessions=[])

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert trace.stop_condition is None

async def test_launch_does_not_stop_on_a_half_written_archive(fixture_manifest_path: Path) -> None:
    """final_metrics is written last; without it the pipeline did not finish."""
    task = _task(fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchiveRuntime(
        exit_code=1, sessions=["session-1"], files=_archive_with_trajectory("/rollout/archive", final_metrics=None),
    )

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert trace.stop_condition is None

async def test_setup_names_the_missing_binaries(fixture_manifest_path: Path) -> None:
    class MissingBinaries(_DockerLikeRuntime):
        """Docker-shaped runtime whose image is missing every required binary."""

        async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
            await super().run(argv, env)
            return vf.ProgramResult(exit_code=127, stdout="", stderr="not found")

    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig(backend="pi"))
    with pytest.raises(RuntimeError) as excinfo:
        await harness.setup(MissingBinaries())
    message = str(excinfo.value)
    assert "daydream" in message and "pi" in message
    assert "run-as-agent" in message, "a wrapper-less image must fail setup"
    assert "build_images.py" in message, "the error must say how to fix it"

async def test_launch_refuses_a_rollout_that_captured_no_model_calls(fixture_manifest_path: Path) -> None:
    """Capture loss must be loud: a bypassed interception server would otherwise
    produce a normal-looking archive and a positive reward."""
    task = _task(fixture_manifest_path)
    trace = _trace(task, turns=0)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())

    with pytest.raises(RuntimeError) as excinfo:
        await harness.launch(_ctx(), trace, FakeRuntime(exit_code=0), ENDPOINT, SECRET, {})
    assert "no model calls" in str(excinfo.value)
    assert ENDPOINT in str(excinfo.value)


class _DockerLikeRuntime(FakeRuntime):
    """A FakeRuntime shaped like the docker runtime (wrapper-prefix contract)."""

    def __init__(self, *, exit_code: int = 0) -> None:
        super().__init__(exit_code=exit_code)
        self.config = DockerConfig()
        self.info = DockerRuntimeInfo(**self.config.model_dump())


class _OrderingDockerRuntime(_DockerLikeRuntime):
    """Record run and run_program together to prove handoff-before-launch ordering."""

    def __init__(self, *, exit_code: int = 0, failed_argv: list[str] | None = None) -> None:
        super().__init__(exit_code=exit_code)
        self.sequence: list[list[str]] = []
        self.failed_argv = failed_argv

    async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        self.sequence.append(argv)
        if self.failed_argv is not None and argv[: len(self.failed_argv)] == self.failed_argv:
            return vf.ProgramResult(exit_code=1, stdout="", stderr="failed")
        return await super().run(argv, env)

    async def run_program(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        self.sequence.append(argv)
        return await super().run_program(argv, env)


async def test_launch_uses_run_as_agent_wrapper_under_docker(fixture_manifest_path: Path) -> None:
    """Docker launches through the root-owned wrapper and actually drops to the agent uid.

    Local subprocess runs need no privilege boundary. Later sealing makes artifacts
    root-owned/read-only, and suite verification uses its separate non-root identity.
    """
    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _DockerLikeRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    (argv, _), = runtime.programs
    assert argv[0] == "run-as-agent"
    assert argv[1] == "env"
    assert "daydream" in argv
    assert argv[argv.index("--backend") + 1] == "claude"

async def test_docker_launch_preflights_writability_before_run_as_agent(fixture_manifest_path: Path) -> None:
    """Preflight both baked agent-owned trees before dropping privileges; never repair ownership.

    Failures name the paths and image rebuild entrypoint.
    """
    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _OrderingDockerRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    for argv in runtime.commands:
        assert "chown" not in argv, ("a docker rollout must never issue a recursive ownership command; "
            "the image bakes ownership at build time"
        )
    preflight = next(argv for argv in runtime.commands if f"test -w {harness.config.repo_path}" in " ".join(argv))
    joined = " ".join(preflight)
    assert f"test -w {harness.config.repo_path}" in joined
    assert "test -w /srv/mirror.git" in joined
    # One probe chains every surface: the tree roots and the per-file write surfaces.
    assert f"test -w {harness.config.repo_path}/.git" in joined
    assert "test -w /srv/mirror.git/refs" in joined
    (argv, _), = runtime.programs
    assert argv[0] == "run-as-agent"
    assert runtime.sequence.index(preflight) < runtime.sequence.index(argv), (
        "the writability preflight must run before the run-as-agent launch"
    )

async def test_preflight_quotes_repo_path(fixture_manifest_path: Path) -> None:
    """#705 fold-in: the binary-check preflight passes repo_path as one shell
    argument, never an unquoted interpolation."""
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig(repo_path="/data/repo with spaces & $dollar"))
    runtime = FakeRuntime(exit_code=0)

    await harness.setup(runtime)

    (argv,) = [argv for argv in runtime.commands if argv[:2] == ["sh", "-c"]]
    assert "test -d '/data/repo with spaces & $dollar'" in argv[2]

async def test_docker_launch_fails_closed_when_trees_not_agent_writable(fixture_manifest_path: Path) -> None:
    """Unwritable agent trees require an image rebuild; never repair them or launch run-as-agent."""
    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _OrderingDockerRuntime(exit_code=1,
        failed_argv=["run-as-agent", "sh", "-c"],  # the single writability preflight
    )

    with pytest.raises(RuntimeError) as excinfo:
        await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    message = str(excinfo.value)
    assert str(harness.config.repo_path) in message
    assert "/srv/mirror.git" in message
    assert "images/build_images.py" in message
    assert runtime.programs == [], "no launch attempt may follow a failed preflight"

def test_run_as_agent_wrapper_executes_and_enforces_root_only() -> None:
    """Execute the wrapper: it must reject non-root callers or successfully leave root.

    Missing setpriv/agent prerequisites fail closed; merely naming a wrapper is insufficient.
    """
    wrapper = PROJECT_ROOT / "images" / "run-as-agent"
    result = subprocess.run([str(wrapper), "id", "-u"], capture_output=True, text=True)
    assert os.access(wrapper, os.X_OK), "run-as-agent wrapper must be executable"
    if os.geteuid() != 0:
        assert result.returncode != 0, "non-root callers must be refused"
        assert "must be run as root" in result.stderr
    elif result.returncode == 0:
        assert result.stdout.strip() != "0", "run-as-agent must drop off root"
    else:
        # A failed wrapper must never execute the payload as root, even when agent prerequisites are
        # missing.
        assert result.stdout.strip() != "0", "a failing wrapper must not run the payload as root"


class _ArchivingDockerRuntime(_SessionsListingRuntime, _DockerLikeRuntime):
    """A docker-shaped runtime whose `ls` of the archive reports one session dir."""

    def __init__(self, *, exit_code: int = 0) -> None:
        super().__init__(exit_code=exit_code)
        self.sessions = ["session-1"]


async def test_seal_re_chowns_the_run_dir_root_owned_under_docker(fixture_manifest_path: Path) -> None:
    """Docker sealing reclaims artifacts as root-owned/read-only without touching the local path."""
    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchivingDockerRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    hardened = [cmd for cmd in runtime.commands if "chown -R root:root" in " ".join(cmd)]
    assert hardened, "seal_archived_run must re-chown the sealed run dir root-owned"
    assert "chmod -R a-w" in hardened[-1][2]
    assert any(path.endswith("/seal.json") for path in runtime.writes), "seal.json must be written"

async def test_seal_failure_is_fail_closed_and_recorded(fixture_manifest_path: Path) -> None:
    """A sealing exception leaves an invalid marker and a recorded failure on the trace.

    Scoring must yield seal_verified=0 and zero reward, never legacy unsealed trust.
    """

    class ExplodingGit(_ArchivingDockerRuntime):
        async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
            result = await super().run(argv, env)
            if argv[:1] == ["git"]:
                raise RuntimeError("git exploded")
            return result

    task = _task(fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = ExplodingGit(exit_code=0)
    trace = _trace(task)

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert trace.info["daydream_seal_ok"] is False
    marker = runtime.writes.get("/rollout/archive/runs/session-1/seal.json")
    assert marker == b'{"seal_failed": true}', "a failed seal must carry the fail-closed marker"
