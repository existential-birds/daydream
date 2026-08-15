"""Phase 4: what the harness actually launches, and how it classifies the exit.

The full rollout (real interception server, real CLI, real scoring) lives in
``test_harness_e2e.py``; these tests pin the contract that rollout depends on.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
import verifiers.v1 as vf
from conftest import PROJECT_ROOT, FakeRuntime
from verifiers.v1.graph import MessageNode

from daydream_review_v1.backends import STRATEGIES
from daydream_review_v1.harness import DaydreamReviewHarness, DaydreamReviewHarnessConfig
from daydream_review_v1.taskset import DaydreamReviewConfig, DaydreamReviewTaskset

ENDPOINT = "http://127.0.0.1:54321/v1"
SECRET = "rollout-secret"
MODEL = "some-org/some-policy-model"


def _task(corpus_mini_dir, fixture_manifest_path):
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
        )
    )
    return list(taskset.load())[0]


def _trace(task, *, turns: int = 1) -> vf.Trace:
    """A trace carrying *turns* captured model turns.

    The harness refuses a rollout that captured none, so the default is one: a
    real rollout always makes model calls, and a zero-turn one is the capture
    loss the harness exists to catch. Nodes rather than `trace.calls`, so the
    helper works under prime-rl's vendored verifiers too, which has no per-call
    list.
    """
    trace: vf.Trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
    for index in range(turns):
        parent = None if index == 0 else len(trace.nodes) - 1
        trace.nodes.append(MessageNode(parent=parent, message={"role": "user", "content": "go"}, sampled=False))
        trace.nodes.append(
            MessageNode(parent=len(trace.nodes) - 1, message={"role": "assistant", "content": "ok"}, sampled=True)
        )
    return trace


def _ctx() -> vf.ModelContext:
    return vf.ModelContext(model=MODEL, client=None, sampling=vf.SamplingConfig())


def _archive_with_trajectory(archive_root: str, *, final_metrics: object) -> dict[str, bytes]:
    session = f"{archive_root}/runs/session-1"
    return {f"{session}/trajectory.json": json.dumps({"final_metrics": final_metrics}).encode()}


class _ArchiveRuntime(FakeRuntime):
    """A FakeRuntime whose `ls` of the archive reports one session dir."""

    def __init__(self, *, exit_code: int, sessions: list[str], files: dict[str, bytes] | None = None) -> None:
        super().__init__(exit_code=exit_code, files=files)
        self.sessions = sessions

    async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        await super().run(argv, env)
        if argv[:2] == ["sh", "-c"] and argv[2].startswith("ls -1 "):
            return vf.ProgramResult(exit_code=0, stdout="\n".join(self.sessions), stderr="")
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


@pytest.mark.parametrize("backend", sorted(STRATEGIES))
async def test_launch_passes_the_selected_backend_to_the_cli(
    backend: str, corpus_mini_dir, fixture_manifest_path
) -> None:
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig(backend=backend, fanout_concurrency=3))
    runtime = FakeRuntime(exit_code=0)

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    (argv, env), = runtime.programs
    assert argv[0] == "daydream"
    assert argv[argv.index("--backend") + 1] == backend
    assert argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--base") + 1] == task.data.base_sha
    assert argv[-1] == "/work/repo"
    # --yes is what makes this a fix rollout rather than a review-only one, and
    # --review with --yes is a parse error, so it must never appear.
    assert "--yes" in argv and "--non-interactive" in argv
    assert "--review" not in argv and "--comment" not in argv
    assert env["DAYDREAM_ARCHIVE_DIR"] == "/rollout/archive"
    assert env["HOME"] == "/rollout"
    assert trace.info["daydream_backend"] == backend
    assert trace.info["daydream_exit_code"] == 0


async def test_launch_carries_extra_args_before_the_target(corpus_mini_dir, fixture_manifest_path) -> None:
    task = _task(corpus_mini_dir, fixture_manifest_path)
    harness = DaydreamReviewHarness(
        DaydreamReviewHarnessConfig(backend="codex", extra_args=["--reasoning-effort", "high"])
    )
    runtime = FakeRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    (argv, _), = runtime.programs
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
    assert argv.index("--reasoning-effort") < argv.index("/work/repo")


async def test_launch_stops_the_trace_when_a_completed_run_exits_nonzero(
    corpus_mini_dir, fixture_manifest_path
) -> None:
    """Tests still red after the fix pass is an outcome to score, not a crash."""
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchiveRuntime(
        exit_code=1,
        sessions=["session-1"],
        files=_archive_with_trajectory("/rollout/archive", final_metrics={"cost_usd": 1.0}),
    )

    result = await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert result.exit_code == 1
    assert trace.stop_condition == "daydream_completed_nonzero"


async def test_launch_leaves_a_crash_to_raise(corpus_mini_dir, fixture_manifest_path) -> None:
    """No artifacts means infrastructure failure: let HarnessError fire."""
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchiveRuntime(exit_code=1, sessions=[])

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert trace.stop_condition is None


async def test_launch_does_not_stop_on_a_half_written_archive(corpus_mini_dir, fixture_manifest_path) -> None:
    """final_metrics is written last; without it the pipeline did not finish."""
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchiveRuntime(
        exit_code=1,
        sessions=["session-1"],
        files=_archive_with_trajectory("/rollout/archive", final_metrics=None),
    )

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert trace.stop_condition is None


async def test_setup_names_the_missing_binaries(corpus_mini_dir, fixture_manifest_path) -> None:
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


async def test_launch_refuses_a_rollout_that_captured_no_model_calls(
    corpus_mini_dir, fixture_manifest_path
) -> None:
    """Capture loss must be loud: a bypassed interception server would otherwise
    produce a normal-looking archive and a positive reward."""
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, turns=0)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())

    with pytest.raises(RuntimeError) as excinfo:
        await harness.launch(_ctx(), trace, FakeRuntime(exit_code=0), ENDPOINT, SECRET, {})
    assert "no model calls" in str(excinfo.value)
    assert ENDPOINT in str(excinfo.value)


class _DockerLikeRuntime(FakeRuntime):
    """A FakeRuntime shaped like the docker runtime (wrapper-prefix contract)."""

    def __init__(self, *, exit_code: int = 0) -> None:
        from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntimeInfo

        super().__init__(exit_code=exit_code)
        self.config = DockerConfig()
        self.info = DockerRuntimeInfo(**self.config.model_dump())


async def test_launch_uses_run_as_agent_wrapper_under_docker(
    corpus_mini_dir, fixture_manifest_path
) -> None:
    """Container launches drop to the non-root agent identity via run-as-agent.

    The image's default user is root and the root-owned run-as-agent wrapper is
    the agent-launch privilege-drop seam, so a container rollout must launch
    daydream through it — every backend CLI subprocess then inherits the agent
    uid. The sealed run dir is re-chowned root-owned read-only at seal time
    (rundir.seal_archived_run), so no rollout process can rewrite the sealed
    artifacts once the agent's write window closes; and the suite re-run runs
    under the distinct non-root verifier identity against a separate
    root-owned read-only checkout — never the agent-mutable tree. The local
    subprocess smoke path has no wrapper (no root boundary to cross).

    Beyond argv, the wrapper itself must actually deliver the drop — the
    property it exists for, not just the prefix it is named with — so a
    wrapper that cannot leave root (or a harness that merely names one) fails
    this container contract.
    """
    task = _task(corpus_mini_dir, fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _DockerLikeRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    (argv, _), = runtime.programs
    assert argv[0] == "run-as-agent"
    assert argv[1] == "daydream"
    assert argv[argv.index("--backend") + 1] == "claude"

    # Pin the terminal property of the seam by executing it: root must drop
    # off root, and a non-root caller must be refused.
    wrapper = PROJECT_ROOT / "images" / "run-as-agent"
    dropped = subprocess.run([str(wrapper), "id", "-u"], capture_output=True, text=True)
    if os.geteuid() == 0:
        assert dropped.returncode == 0, dropped.stderr
        assert dropped.stdout.strip() != "0", "run-as-agent must drop off root"
    else:
        assert dropped.returncode != 0, "non-root callers must be refused"
        assert "must be run as root" in dropped.stderr


def test_run_as_agent_wrapper_executes_and_enforces_root_only() -> None:
    """The privilege-drop seam must actually run, not just be argv[0].

    The docker-path test above pins that the harness prefixes the wrapper; this
    pins the wrapper itself by executing it. On a non-root host (the normal
    dev/CI case) the wrapper must refuse — it exists to leave root, never to
    run the payload as root. On a root host that can satisfy setpriv/agent
    (i.e. the built image) a successful run must land off root; anywhere else
    it must fail closed. A missing, non-executable, or silently-passing wrapper
    is exactly the rollout-time failure this pins at build time.
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
        # Root host where the wrapper failed (broken wrapper or missing agent
        # user): fail-closed means the payload must never run as root, even on
        # the failure path — a non-zero exit that still emitted the root uid
        # would be a silent fail-open.
        assert result.stdout.strip() != "0", "a failing wrapper must not run the payload as root"


class _ArchivingDockerRuntime(_DockerLikeRuntime):
    """A docker-shaped runtime whose `ls` of the archive reports one session dir."""

    def __init__(self, *, exit_code: int = 0) -> None:
        super().__init__(exit_code=exit_code)
        self.sessions = ["session-1"]

    async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        await super().run(argv, env)
        if argv[:2] == ["sh", "-c"] and argv[2].startswith("ls -1 "):
            return vf.ProgramResult(exit_code=0, stdout="\n".join(self.sessions), stderr="")
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


async def test_seal_re_chowns_the_run_dir_root_owned_under_docker(
    corpus_mini_dir, fixture_manifest_path
) -> None:
    """The sealed run dir is re-chowned root-owned read-only at seal time.

    base.Dockerfile documents that the supervisor re-chowns the sealed run dir
    root-owned read-only at seal time (rundir.seal_archived_run) — the
    mechanism that keeps the sealed artifacts agent-inaccessible once the
    agent's write window closes. The docker runtime execs as the container
    root, so the chown lands there and the local path is untouched.
    """
    task = _task(corpus_mini_dir, fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _ArchivingDockerRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    hardened = [cmd for cmd in runtime.commands if "chown -R root:root" in " ".join(cmd)]
    assert hardened, "seal_archived_run must re-chown the sealed run dir root-owned"
    assert "chmod -R a-w" in hardened[-1][2]
    assert any(path.endswith("/seal.json") for path in runtime.writes), "seal.json must be written"


async def test_seal_failure_is_fail_closed_and_recorded(
    corpus_mini_dir, fixture_manifest_path
) -> None:
    """A seal-production failure is fail-closed, never silently unsealed.

    seal_archived_run swallows every exception, but not fail-open: it overwrites
    seal.json with an unvalidatable marker so verify_seal reads a failed seal
    (seal_verified 0.0, zero reward) instead of a missing-seal full trust, and
    the harness records the outcome on the trace instead of discarding the
    bool — an operator can tell "sealed and verified" from "sealing failed".
    """

    class ExplodingGit(_ArchivingDockerRuntime):
        async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
            result = await super().run(argv, env)
            if argv[:1] == ["git"]:
                raise RuntimeError("git exploded")
            return result

    task = _task(corpus_mini_dir, fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = ExplodingGit(exit_code=0)
    trace = _trace(task)

    await harness.launch(_ctx(), trace, runtime, ENDPOINT, SECRET, {})

    assert trace.info["daydream_seal_ok"] is False
    marker = runtime.writes.get("/rollout/archive/runs/session-1/seal.json")
    assert marker == b'{"seal_failed": true}', "a failed seal must carry the fail-closed marker"
