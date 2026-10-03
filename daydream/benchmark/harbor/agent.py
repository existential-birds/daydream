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
    backend = (extra_env.get(env_policy.BACKEND_ENV) or "pi").strip().lower()
    from daydream.benchmark.harbor.entrypoint import _SUPPORTED_BACKENDS

    if backend not in _SUPPORTED_BACKENDS:
        supported = ", ".join(repr(b) for b in _SUPPORTED_BACKENDS)
        raise AgentError(
            f"unsupported DAYDREAM_REVIEW_BACKEND={backend!r}; supported backends: {supported}"
        )
    return backend


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
        """Without network access, probe the exact Daydream release and selected backend."""
        backend = _supported_backend(self.extra_env)
        # An allowlisted backend still needs its own probe; reject incomplete additions.
        backend_probe = {
            "pi": "assert shutil.which('pi') is not None;",
            "claude": "import claude_agent_sdk;",
        }.get(backend)
        if backend_probe is None:
            raise AgentError(
                f"no setup probe is defined for DAYDREAM_REVIEW_BACKEND={backend!r}; "
                "extend the probe map in DaydreamReviewAgent.setup()"
            )
        probe = (
            "import importlib.metadata, shutil;"
            f"assert importlib.metadata.version('daydream') == {self.version()!r};"
            + backend_probe
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
        """Review the frozen snapshot through the controlled container entrypoint."""
        if not _HARBOR:
            raise AgentError("Harbor is not installed; install 'daydream[benchmark]'")
        backend = _supported_backend(self.extra_env)
        parent = {**os.environ, **self.extra_env}
        child_env = build_child_env(parent, backend=backend)
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


def build_child_env(parent_env: Mapping[str, str], *, backend: str = "pi") -> dict[str, str]:
    """Pass only declared reviewer/process configuration, never the parent environment
    wholesale. Claude additionally receives the exact HOST policy Anthropic credential
    allowlist; other backends scrub those credentials. Judge, GitHub, Hub, and archive
    secrets stay excluded. Review-profile candidates reach the reviewer entrypoint but
    never the independently configured judge.
    """
    host = env_policy.HOST
    keep_anthropic = backend == "claude"
    child = {
        key: value
        for key, value in dict(parent_env).items()
        if key.startswith(env_policy.REVIEW_CHANNEL_PREFIX)
        or key in host.required_process_vars
        or (keep_anthropic and key in host.claude_keep_vars)
    }
    banned_vars = host.banned_vars if not keep_anthropic else host.banned_vars - host.claude_exempt_vars
    banned_prefixes = (
        host.banned_prefixes if not keep_anthropic else host.banned_prefixes - {host.claude_exempt_prefix}
    )
    for banned in banned_vars:
        child.pop(banned, None)
    for prefix in banned_prefixes:
        for key in [k for k in child if k.startswith(prefix)]:
            child.pop(key, None)
    return child
