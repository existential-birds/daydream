"""Phase 4: what the harness actually launches, and how it classifies the exit.

The full rollout (real interception server, real CLI, real scoring) lives in
``test_harness_e2e.py``; these tests pin the contract that rollout depends on.
"""

from __future__ import annotations

import json

import pytest
import verifiers.v1 as vf
from conftest import FakeRuntime
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
    class MissingBinaries(FakeRuntime):
        async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
            await super().run(argv, env)
            return vf.ProgramResult(exit_code=127, stdout="", stderr="not found")

    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig(backend="pi"))
    with pytest.raises(RuntimeError) as excinfo:
        await harness.setup(MissingBinaries())
    message = str(excinfo.value)
    assert "daydream" in message and "pi" in message
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
    the single privilege-drop seam, so a container rollout must launch daydream
    through it — every backend CLI subprocess then inherits the agent uid. The
    local subprocess smoke path has no wrapper (no root boundary to cross).
    """
    task = _task(corpus_mini_dir, fixture_manifest_path)
    harness = DaydreamReviewHarness(DaydreamReviewHarnessConfig())
    runtime = _DockerLikeRuntime(exit_code=0)

    await harness.launch(_ctx(), _trace(task), runtime, ENDPOINT, SECRET, {})

    (argv, _), = runtime.programs
    assert argv[0] == "run-as-agent"
    assert argv[1] == "daydream"
    assert argv[argv.index("--backend") + 1] == "claude"
