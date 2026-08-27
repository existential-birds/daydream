"""Harness: one rollout is one headless daydream deep run inside the sandbox.

The harness starts exactly one program and returns; every model turn it makes —
including the parallel exploration and per-stack fan-outs — flows through the
interception server on the way out, so the whole run lands in one trace as a DAG
of branches. Branches are training samples; nothing here flattens them.

Deliberately NOT a new daydream backend. Every existing backend delegates tool
execution to its own CLI runtime; a raw HTTP backend would mean reimplementing
daydream's entire tool loop, which is a different project (osprey). Swapping the
rollout agent is one config key: ``backend``.
"""

from __future__ import annotations

import logging
from typing import Any

import verifiers.v1 as vf
from pydantic import field_validator

from daydream_review_v1.backends import (
    DEFAULT_PI_CONTEXT_WINDOW,
    DEFAULT_PI_MAX_TOKENS,
    ROLLOUT_HOME,
    STRATEGIES,
    BackendStrategy,
    PiStrategy,
)
from daydream_review_v1.rundir import DEFAULT_ARCHIVE_ROOT, daydream_completed, seal_archived_run
from daydream_review_v1.taskset import DEFAULT_REPO_PATH, DaydreamReviewData

logger = logging.getLogger(__name__)


class DaydreamReviewHarnessConfig(vf.HarnessConfig):
    backend: str = "claude"
    """Which daydream backend drives the rollout — any key of ``STRATEGIES``."""

    fanout_concurrency: int = 4
    """Parallel-``execute()`` hint handed to the backend.

    Effective upstream concurrency is roughly ``pool.max_workers ×
    fanout_concurrency``: each rollout runs its own fan-out.
    """

    home: str = ROLLOUT_HOME
    """``HOME`` for the daydream process; where per-backend config is planted."""

    repo_path: str = DEFAULT_REPO_PATH
    """The repository under review, baked into the image at the task's head SHA."""

    archive_root: str = DEFAULT_ARCHIVE_ROOT
    """``DAYDREAM_ARCHIVE_DIR``. The reward reads the run dir back out of it."""

    extra_args: list[str] = []
    """Escape hatch, e.g. ``["--reasoning-effort", "high"]`` for codex."""

    pi_context_window: int = DEFAULT_PI_CONTEXT_WINDOW
    pi_max_tokens: int = DEFAULT_PI_MAX_TOKENS
    """pi only: the catalogue entry it needs for the policy model.

    Capabilities of the endpoint, which nothing here can infer, so they are
    config (SPEC C1). Match them to the run's ``seq_len`` and
    ``max_completion_tokens``; a window declared larger than the endpoint serves
    fails at rollout time."""

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(f"unknown backend {value!r}; valid: {', '.join(sorted(STRATEGIES))}")
        return value


class DaydreamReviewHarness(vf.Harness[DaydreamReviewHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False
    SUPPORTS_MESSAGE_PROMPT = False

    @property
    def strategy(self) -> BackendStrategy:
        if self.config.backend == PiStrategy.name:
            return PiStrategy(
                self.config.home,
                context_window=self.config.pi_context_window,
                max_tokens=self.config.pi_max_tokens,
            )
        return STRATEGIES[self.config.backend](self.config.home)

    async def setup(self, runtime: vf.Runtime) -> None:
        """Fail fast with remediation text; the image bakes everything else (D6)."""
        strategy = self.strategy
        binaries = ("daydream", *strategy.required_binaries)
        if runtime.type == "docker":
            # The image's root-owned run-as-agent wrapper is the single
            # privilege-drop seam; a wrapper-less image must fail here, not
            # opaquely at docker exec.
            binaries = (*binaries, "run-as-agent")
        checks = " && ".join(f"command -v {binary} >/dev/null" for binary in binaries)
        result = await runtime.run(
            ["sh", "-c", f"{checks} && test -d {self.config.repo_path}"], self.config.resolved_env
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"rollout image is not usable for backend={strategy.name}: it must carry "
                f"{', '.join(binaries)} on PATH and a repository at {self.config.repo_path}. "
                f"Build it with images/build_images.py. {result.stdout}{result.stderr}"
            )

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        _mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        data: DaydreamReviewData = trace.task.data
        strategy = self.strategy
        await strategy.provision(runtime, endpoint, secret, ctx.model)

        env: dict[str, str] = {
            **self.config.resolved_env,
            **strategy.env(endpoint, secret, fanout_concurrency=self.config.fanout_concurrency),
            "HOME": self.config.home,
            "DAYDREAM_ARCHIVE_DIR": self.config.archive_root,
        }
        # --yes is load-bearing: --non-interactive alone takes the fix gate's safe
        # default and exits 0 having applied nothing (deep/orchestrator.py:1187-1196),
        # which would make every rollout a review-only rollout. --review is never
        # set: with --yes it is a parse error (cli.py:978-981). Deep is the default
        # flow, so there is no --deep to pass (runner.py:812).
        argv = [
            "daydream",
            "--non-interactive",
            "--yes",
            "--backend",
            strategy.name,
            "--model",
            ctx.model,
            "--base",
            data.base_sha,
            *self.config.extra_args,
            self.config.repo_path,
        ]
        if runtime.type == "docker":
            # The repo image clones the checkout as root (repo.Dockerfile), so
            # hand the workspace — and the in-container origin mirror the deep
            # flow pushes its fix to — to the agent identity before the
            # privilege drop. Otherwise every deep-flow write (.daydream/,
            # worktrees, fix edits, git add/commit, push) EACCESes as the
            # agent uid and the rollout dies before any model turn.
            handoff = await runtime.run(
                ["chown", "-R", "agent:agent", self.config.repo_path, "/srv/mirror.git"], env
            )
            if handoff.exit_code != 0:
                raise RuntimeError(
                    "could not hand the checkout to the agent identity: "
                    f"{handoff.stdout}{handoff.stderr}"
                )
            # Container launches drop from the container default user (root) to
            # the non-root agent identity through the image's root-owned
            # run-as-agent wrapper, so every daydream process and backend CLI
            # subprocess it spawns runs as the agent uid — never root, and never
            # able to write the sealed surfaces. The local subprocess smoke path
            # has no wrapper (there is no root boundary to cross).
            argv = ["run-as-agent", *argv]
        result = await runtime.run_program(argv, env)

        # A deep run always talks to a model. Zero captured turns means the CLI
        # reached a provider directly and the whole rollout is untrainable — and
        # it would otherwise SCORE, because the reward reads daydream's artifacts
        # and those look perfectly normal. A live codex rollout did exactly this:
        # real tokens billed, zero turns in the trace, reward 1.0. Silent capture
        # loss is the one failure mode this harness must never absorb.
        #
        # `num_turns`, not `trace.calls`: the per-call list is a 0.2.1 field that
        # prime-rl's vendored verifiers submodule does not have, while the turn
        # count is derived from the node graph and exists in both. The guard has
        # to survive the version this actually trains under.
        if not trace.num_turns:
            raise RuntimeError(
                f"backend={strategy.name} made no model calls through the interception server at "
                f"{endpoint}: the rollout produced no trainable turns. The CLI is reaching a "
                "provider directly — check this backend's endpoint injection before trusting any "
                "reward from it."
            )

        info: dict[str, Any] = trace.info
        info["daydream_exit_code"] = result.exit_code
        info["daydream_backend"] = strategy.name
        info["daydream_repo_path"] = self.config.repo_path
        info["daydream_archive_root"] = self.config.archive_root

        # Exit 1 with complete artifacts is a legitimate outcome, not a crash: the
        # deep flow stops non-zero when the suite is still red after the fix pass
        # (orchestrator.py:1389). Setting a stop condition makes the framework
        # return quietly instead of raising HarnessError (harness.py:99-118), so the
        # rollout SCORES — a red suite is exactly the signal suite_non_regression wants.
        # A non-zero exit with no artifacts is left to raise: that is infrastructure
        # failure and belongs to the retry budget, not to the gradient.
        if result.exit_code != 0 and await daydream_completed(runtime, self.config.archive_root):
            trace.stop("daydream_completed_nonzero")
        # Produce the integrity seal over the archived run dir now that the
        # agent's write window has closed: the reward verifies the staged copy
        # against this seal before trusting any value, so an attempted tamper
        # with the archived artifacts zeroes the reward instead of recording
        # honest telemetry. The candidate diff is the rollout's own committed
        # diff against the baked head (b"" when it cannot be re-derived).
        sealed = await seal_archived_run(
            runtime,
            self.config.archive_root,
            repo=self.config.repo_path,
            head_sha=data.head_sha,
        )
        # A seal-production failure must never be silent or fail-open:
        # seal_archived_run overwrites seal.json with an unvalidatable marker
        # (verify_seal -> False -> zero reward), and the trace records the
        # outcome so an operator can distinguish "sealed and verified" from
        # "sealing failed, zero protection" — a completed run with no valid
        # seal is never scored at full trust as if the protection existed.
        trace.info["daydream_seal_ok"] = sealed
        if not sealed:
            logger.warning(
                "seal_archived_run failed for %s (repo=%s, head=%s): the rollout "
                "will score seal_verified 0.0 and zero intrinsic reward",
                self.config.archive_root,
                self.config.repo_path,
                data.head_sha,
            )
        return result
