"""Run one headless Daydream deep program per rollout. All model turns, including parallel fan-outs,
pass through interception as a DAG of training branches, never a flattened trace. The backend config
key selects the existing CLI runtime and its tool execution.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

# external contract: verifiers.v1 is the vendored third-party API module name, not a project-owned name
import verifiers.v1 as vf
from pydantic import field_validator

from daydream_review.backends import (
    DEFAULT_PI_CONTEXT_WINDOW,
    DEFAULT_PI_MAX_TOKENS,
    ROLLOUT_HOME,
    STRATEGIES,
    BackendStrategy,
    PiStrategy,
)
from daydream_review.rundir import DEFAULT_ARCHIVE_ROOT, daydream_completed, seal_archived_run
from daydream_review.taskset import DEFAULT_REPO_PATH, DaydreamReviewData

logger = logging.getLogger(__name__)

_ROLLOUT_GITHUB_ENV_TO_UNSET = (
    "DAYDREAM_APP_ID",
    "DAYDREAM_APP_PRIVATE_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)

_ROLLOUT_ENV_TO_CLEAR = (
    "DAYDREAM_TRACE_TO",
    "DAYDREAM_TRAJECTORY_HUB_REPO",
)
"""Operator-only settings that must never leak into a rollout.

Observability is activated only by operator CLI/environment settings; a rollout
process inherits ambient host variables, so a host with DAYDREAM_TRACE_TO set
would make every rollout try to initialize that trace destination. The
subprocess runtime inherits non-API-key variables only, so the destination's
credential never arrives and tracing initialization fails the run before its
first model call. Clear both in the launch env — empty means disabled for
tracing, and an empty hub repo means no upload — keeping rollouts hermetic."""


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
    # The smoke path (configs/eval-stub.toml) runs on the subprocess runtime; the
    # harness itself stages daydream into whatever runtime it is given.
    NEEDS_CONTAINER = False

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
        """Fail fast with remediation text; all remaining dependencies are baked into the image."""
        strategy = self.strategy
        binaries = ("daydream", *strategy.required_binaries)
        if runtime.type == "docker":
            # The image's root-owned run-as-agent wrapper is the single
            # privilege-drop seam; a wrapper-less image must fail here, not
            # opaquely at docker exec.
            binaries = (*binaries, "run-as-agent")
        checks = " && ".join(f"command -v {binary} >/dev/null" for binary in binaries)
        result = await runtime.run(
            ["sh", "-c", f"{checks} && test -d {shlex.quote(self.config.repo_path)}"], self.config.resolved_env
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
        mcp_urls: dict[str, str],  # noqa: F841 — tool-server wiring lands in a later verifiers
    ) -> vf.ProgramResult:
        data: DaydreamReviewData = trace.task.data
        strategy = self.strategy
        await strategy.provision(runtime, endpoint, secret, ctx.model)

        env: dict[str, str] = {
            **self.config.resolved_env,
            **strategy.env(endpoint, secret, fanout_concurrency=self.config.fanout_concurrency),
            "HOME": self.config.home,
            "DAYDREAM_ARCHIVE_DIR": self.config.archive_root,
            # Operator-only destinations are cleared, never inherited (see
            # _ROLLOUT_ENV_TO_CLEAR).
            **{name: "" for name in _ROLLOUT_ENV_TO_CLEAR},
        }
        # --yes authorizes fixes; --non-interactive alone accepts the no-fix default. Never combine
        # --review with --yes. Deep is the default flow. Clear inherited GitHub credentials so host
        # App settings cannot divert or abort a hermetic rollout before its first model call.
        github_env_unsets = [
            argument
            for name in _ROLLOUT_GITHUB_ENV_TO_UNSET
            for argument in ("-u", name)
        ]
        argv = [
            "env",
            *github_env_unsets,
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
            # The image bakes agent-owned repository/mirror trees. Probe writability through
            # run-as-agent, since a root probe would pass via CAP_DAC_OVERRIDE and hide missing
            # ownership. One sh -c chains every surface so any unwritable one short-circuits to a
            # non-zero exit: the two tree roots, plus the per-file surfaces (.git, refs) whose
            # ownership writable roots alone do not guarantee, since commits and pushes must be
            # able to write their files. Failure requires rebuilding the image, never runtime
            # ownership repair. Quote configured paths in the privileged sh -c command to prevent
            # whitespace splitting or shell metacharacter execution under the agent UID.
            writability = await runtime.run(
                [
                    "run-as-agent",
                    "sh",
                    "-c",
                    f"test -w {shlex.quote(self.config.repo_path)}"
                    f" && test -w {shlex.quote(self.config.repo_path)}/.git"
                    " && test -w /srv/mirror.git && test -w /srv/mirror.git/refs",
                ],
                env,
            )
            if writability.exit_code != 0:
                raise RuntimeError(
                    "repo and mirror are not agent-writable (tree roots plus the per-file "
                    "write surfaces: the checkout's .git and the mirror's refs); the image "
                    "was likely built without the ownership layer. Rebuild it with "
                    "images/build_images.py, which bakes agent ownership for "
                    f"{self.config.repo_path} and /srv/mirror.git: "
                    f"{writability.stdout}{writability.stderr}"
                )
            # Use the root-owned wrapper to drop Docker launches and all backend subprocesses to the
            # agent UID, which cannot write sealed surfaces. The local smoke runtime has no root
            # boundary or wrapper.
            argv = ["run-as-agent", *argv]
        result = await runtime.run_program(argv, env)

        # Zero captured turns means model calls may have bypassed interception: artifacts can still
        # score normally, but the rollout is untrainable. Never absorb capture loss. Use num_turns
        # from the node graph; trace.calls exists in verifiers 0.2.1 but not prime-rl's vendored
        # version.
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

        # Nonzero exit with completed artifacts remains scoreable (for example, a red post-fix
        # suite); set the framework stop condition to avoid HarnessError. Without completed
        # artifacts, let infrastructure failure raise into the retry budget.
        if result.exit_code != 0 and await daydream_completed(runtime, self.config.archive_root):
            trace.stop("daydream_completed_nonzero")
        sealed = await seal_archived_run(
            runtime,
            self.config.archive_root,
            repo=self.config.repo_path,
            head_sha=data.head_sha,
        )
        # Record sealing failure for operators and scoring. seal_archived_run marks invalid seals; a
        # completed run without valid protection must never receive legacy full trust.
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
