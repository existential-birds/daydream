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
        release and the backend SDK for the selected reviewer backend (the Pi
        CLI for ``pi``, ``claude_agent_sdk`` for ``claude``). A backend outside
        the shared ``_SUPPORTED_BACKENDS`` allowlist is refused here, before
        any probe (an unsupported value must never probe a wrong SDK).

        A single ``environment.exec`` runs an in-container Python probe; a
        non-zero exec return (missing exact version or missing backend SDK)
        raises :class:`AgentError` -- never a silent pass.
        """
        backend = _supported_backend(self.extra_env)
        # The allowlist can grow before a probe exists; never KeyError on an
        # allowlisted-but-unprobed backend (and never probe a wrong SDK) --
        # refuse with a typed error instead.
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
        """Review the frozen snapshot in-container.

        Fail-closed: refuses any backend outside the shared
        ``_SUPPORTED_BACKENDS`` allowlist *before* any reviewing (never
        installs tools or widens network access), maps the allowlist child
        environment, and invokes the controlled entrypoint. A non-zero
        entrypoint return raises :class:`AgentError`.
        """
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


def build_child_env(parent_env: Mapping[str, str], *, backend: str = "pi") -> dict[str, str]:
    """Build the fail-closed allowlist child environment.

    Keeps only ``DAYDREAM_REVIEW_*`` reviewer config/credential plus the required
    process variables, then explicitly drops the banned variables (GitHub/HF/judge/
    archive and raw provider vars) so any future secret-holding variable not in
    the keep-set still cannot leak by default. Never passes the parent env wholesale.

    Backend-conditional credential handling: for ``backend="claude"`` exactly
    the control-plane ``ANTHROPIC_*`` keep-set declared in ``env_policy.HOST``
    survives so the Claude Agent SDK / claude CLI in the container has
    credentials; any other host-ambient ``ANTHROPIC_*`` var is scrubbed like
    any other raw credential. For ``pi`` (default) and any other value, the
    ``ANTHROPIC_*`` scrub is exactly today's fail-closed behavior.

    The review-profile candidate (``DAYDREAM_REVIEW_PROFILE_CANDIDATE``, issue
    #885/R11) rides the ``DAYDREAM_REVIEW_*`` allowlist to the entrypoint; the
    verifier env is isolated to ``DAYDREAM_JUDGE_*`` (render_job_config), so the
    candidate never reaches the judge.
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
