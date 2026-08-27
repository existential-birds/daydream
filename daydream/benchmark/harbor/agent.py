"""Harbor ``BaseAgent`` that reviews a frozen private-PR snapshot in-container (issue #780).

``DaydreamReviewAgent`` runs host-side: it guarantees the Pi/OpenRouter backend, builds
a fail-closed allowlist child environment, and invokes the controlled
in-container entrypoint (``daydream.benchmark.harbor.entrypoint``) via
``environment.exec``. The entrypoint runs the real Daydream runner in-process
against the frozen snapshot, then publishes the candidate artifact and ATIF
trajectory. Harbor is an optional extra: this module imports it lazily so a host
without the ``benchmark`` extra can still import the package.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from harbor.agents.base import BaseAgent
    from harbor.models.agent.context import AgentContext

    _HARBOR = True
except ImportError:  # Harbor is an optional extra; degrade to plain bases.
    BaseAgent = object
    AgentContext = object
    _HARBOR = False


class AgentError(Exception):
    """Typed failure carrier for the Daydream Harbor agent lifecycle."""


_REQUIRED_PROCESS_VARS = ("PATH", "HOME", "LANG")
_BANNED_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "DAYDREAM_APP_ID",
    "DAYDREAM_APP_PRIVATE_KEY",
    "HF_TOKEN",
    "DAYDREAM_TRAJECTORY_HUB_REPO",
    "DAYDREAM_ARCHIVE_DIR",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_API_KEY",
    "PI_API_KEY",
)
# Judge vars and raw provider credentials must never leak into the child env.
_BANNED_PREFIXES = (
    "DAYDREAM_JUDGE_",
    "ANTHROPIC_",
    "OPENAI_",
    "OPENROUTER_",
    "PI_",
)


class DaydreamReviewAgent(BaseAgent):  # type: ignore[misc]
    """Harbor agent driving the in-container, privacy-safe Daydream reviewer.

    Attributes:
        SUPPORTS_ATIF: This agent writes an ATIF trajectory and backfills
            ``AgentContext`` metrics from it.
    """

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        """Agent name reported to Harbor."""
        return "daydream"

    @classmethod
    def version(cls) -> str:
        """The packaged Daydream release this agent runs."""
        from daydream import __version__

        return __version__

    async def setup(self, environment: Any) -> None:
        """Network-free setup: confirm the container installs this exact Daydream
        release and the Pi CLI.

        A single ``environment.exec`` runs an in-container Python probe; a
        non-zero exec return (missing exact version or missing backend SDK)
        raises :class:`AgentError` -- never a silent pass.
        """
        probe = (
            "import importlib.metadata, shutil;"
            f"assert importlib.metadata.version('daydream') == {self.version()!r};"
            "assert shutil.which('pi') is not None;"
        )
        command = 'python -X utf8 -c "' + probe + '"'
        result = await environment.exec(command)
        if result.return_code != 0:
            raise AgentError(
                f"container setup probe failed (rc={result.return_code}): "
                f"{result.stdout or ''}{result.stderr or ''}"
            )

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> None:
        """Review the frozen snapshot in-container.

        Fail-closed: refuses any backend other than ``pi`` *before* any
        reviewing (never installs tools or widens network access), maps the
        allowlist child environment, and invokes the controlled entrypoint. A
        non-zero entrypoint return raises :class:`AgentError`.
        """
        if not _HARBOR:
            raise AgentError("Harbor is not installed; install 'daydream[benchmark]'")
        backend = (self.extra_env.get("DAYDREAM_REVIEW_BACKEND") or "pi").strip().lower()
        if backend != "pi":
            raise AgentError(
                f"unsupported DAYDREAM_REVIEW_BACKEND={backend!r}; only 'pi' is supported"
            )
        parent = {**os.environ, **self.extra_env}
        child_env = build_child_env(parent)
        result = await environment.exec(
            "python -m daydream.benchmark.harbor.entrypoint",
            cwd=child_env.get("DAYDREAM_REVIEW_REPO_DIR", "/workspace/repo"),
            env=child_env,
            timeout_sec=1800,
        )
        if result.return_code != 0:
            raise AgentError(
                f"entrypoint review failed (rc={result.return_code}): "
                f"{result.stdout or ''}{result.stderr or ''}"
            )

    def populate_context_post_run(self, context: Any) -> None:
        """Backfill ``AgentContext`` cost/token metrics from the ATIF trajectory.

        Reads ``<logs_dir>/agent/trajectory.json`` (Harbor syncs the container's
        ``/logs/agent/`` there after a trial). When present with ``final_metrics``,
        fills the corresponding ``AgentContext`` fields. A missing or malformed
        trajectory leaves metrics unset -- no fabricated zeros.
        """
        traj = Path(self.logs_dir) / "agent" / "trajectory.json"
        try:
            data = json.loads(traj.read_text())
        except (OSError, json.JSONDecodeError):
            return
        final_metrics = data.get("final_metrics") if isinstance(data, dict) else None
        if not isinstance(final_metrics, dict):
            return
        mapping = {
            "n_input_tokens": "total_prompt_tokens",
            "n_cache_tokens": "total_cached_tokens",
            "n_output_tokens": "total_completion_tokens",
            "cost_usd": "total_cost_usd",
        }
        for attr, key in mapping.items():
            value = final_metrics.get(key)
            if value is not None and hasattr(context, attr):
                setattr(context, attr, value)


def build_child_env(parent_env: Mapping[str, str]) -> dict[str, str]:
    """Build the fail-closed allowlist child environment.

    Keeps only ``DAYDREAM_REVIEW_*`` reviewer config/credential plus the required
    process variables, then explicitly drops the banned variables (GitHub/HF/judge/
    archive and raw provider vars) so any future secret-holding variable not in
    the keep-set still cannot leak by default. Never passes the parent env wholesale.

    The review-profile candidate (``DAYDREAM_REVIEW_PROFILE_CANDIDATE``, issue
    #885/R11) rides the ``DAYDREAM_REVIEW_*`` allowlist to the entrypoint; the
    verifier env is isolated to ``DAYDREAM_JUDGE_*`` (render_job_config), so the
    candidate never reaches the judge.
    """
    child = {
        key: value
        for key, value in dict(parent_env).items()
        if key.startswith("DAYDREAM_REVIEW_") or key in _REQUIRED_PROCESS_VARS
    }
    for banned in _BANNED_VARS:
        child.pop(banned, None)
    for prefix in _BANNED_PREFIXES:
        for key in [k for k in child if k.startswith(prefix)]:
            child.pop(key, None)
    return child
