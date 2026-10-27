"""Host-side Harbor agent that launches the controlled reviewer entrypoint in-container. An
allowlisted child environment carries reviewer configuration. The entrypoint runs
Daydream against the frozen snapshot and publishes candidates/trajectory. Harbor imports
stay lazy so the base package works without the optional extra.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from daydream.benchmark.harbor import env_policy

try:
    from harbor.agents.base import BaseAgent

    _HARBOR = True
except ImportError:  # Harbor is an optional extra; degrade to plain bases.
    BaseAgent = object
    _HARBOR = False


class AgentError(Exception):
    """Typed failure carrier for the Daydream Harbor agent lifecycle."""


def _supported_backend(extra_env: Mapping[str, str]) -> str:
    """Return the validated ``DAYDREAM_REVIEW_BACKEND`` or raise :class:`AgentError`."""
    from daydream.benchmark.harbor.entrypoint import EntrypointError, require_supported_backend

    try:
        return require_supported_backend({env_policy.BACKEND_ENV: extra_env.get(env_policy.BACKEND_ENV) or "pi"})
    except EntrypointError as exc:
        raise AgentError(str(exc)) from exc


def _require_exec_ok(result: Any, label: str) -> None:
    """Raise :class:`AgentError` when an in-container exec returned non-zero."""
    if result.return_code != 0:
        raise AgentError(f"{label} (rc={result.return_code}): {result.stdout or ''}{result.stderr or ''}")


class DaydreamReviewAgent(BaseAgent):  # type: ignore[misc]
    """Harbor reviewer writing ATIF trajectories and backfilling AgentContext metrics."""

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
        """Without network access, probe the exact Daydream release and selected backend.

        Validate the shared backend allowlist before probing the native baseline.
        A missing probe or nonzero container probe result raises ``AgentError``.
        """
        _supported_backend(self.extra_env)
        probe = (
            "import importlib.metadata, shutil, subprocess;"
            f"assert importlib.metadata.version('daydream') == {self.version()!r};"
            "assert shutil.which('pi') is not None;"
            "assert subprocess.check_output(['node','--version'],text=True).strip() == 'v24.18.1';"
            "assert subprocess.check_output(['pi','--version'],text=True).strip() == '1.4.0';"
        )
        command = 'python -X utf8 -c "' + probe + '"'
        result = await environment.exec(command)
        _require_exec_ok(result, "container setup probe failed")

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> None:
        """Review the frozen snapshot through the controlled container entrypoint.

        Validate the backend before exec and pass only the allowlisted child environment.
        Do not install tools or widen network access; nonzero exit raises ``AgentError``.
        """
        if not _HARBOR:
            raise AgentError("Harbor is not installed; install 'daydream[benchmark]'")
        _supported_backend(self.extra_env)
        parent = {**os.environ, **self.extra_env}
        child_env = build_child_env(parent)
        result = await environment.exec(
            "python -m daydream.benchmark.harbor.entrypoint",
            cwd=child_env.get(env_policy.REPO_DIR_ENV, "/workspace/repo"),
            env=child_env,
            timeout_sec=1800,
        )
        _require_exec_ok(result, "entrypoint review failed")

    def populate_context_post_run(self, context: Any) -> None:
        """Read synced ATIF final metrics into AgentContext; absent/malformed trajectories
        leave values unset.
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
    """Carry only declared reviewer controls and process configuration into Harbor.

    Judge, GitHub, Hub, archive and unselected native credentials stay excluded.
    """
    host = env_policy.HOST
    return {
        key: value
        for key, value in dict(parent_env).items()
        if (key.startswith(host.keep_prefixes) or key in host.required_process_vars)
        and key not in host.banned_vars
        and not key.startswith(tuple(host.banned_prefixes))
    }
