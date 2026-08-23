# daydream/benchmark/harbor/agent.py
"""Harbor 0.21 agent module backing the compiled ``harbor-job.yaml``.

``render_job_config`` emits ``import_path: daydream.benchmark.harbor.agent:
DaydreamReviewAgent``; Harbor's :func:`~harbor.utils.import_path.import_class`
imports this module (with the bundled Harbor 0.21 harness) on the host and
instantiates :class:`DaydreamReviewAgent` through ``AgentFactory``. The
compiled case checkout inside the agent image carries ``base``/``head`` refs
(isolated from ``origin``) and a ``daydream`` wheel install, so the agent runs
the standalone ``daydream`` CLI review against that checkout. The job config's
``DAYDREAM_REVIEW_*`` variables select the backend/model and the provider
credential the review process reaches its model provider with.
"""

from __future__ import annotations

import importlib.metadata
import shlex
from typing import Mapping

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: Where the compile path clones the repository checkout inside the agent image.
_REPO_DIR = "/workspace/repo"
#: Ref the compile path bakes into the checkout as the diff base.
_BASE_REF = "base"
#: Backend used when the job config carries no DAYDREAM_REVIEW_BACKEND.
_DEFAULT_BACKEND = "claude"

#: DAYDREAM_REVIEW_* -> provider-native environment name for the backends whose
#: CLI reads its credential/base-url from the environment (the claude-agent-sdk
#: spawns its CLI with the full parent environment). Other backends resolve
#: their own provider config and are left untouched here.
_BACKEND_CREDENTIAL_ENVARS: Mapping[str, Mapping[str, str]] = {
    "claude": {
        "DAYDREAM_REVIEW_API_KEY": "ANTHROPIC_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL": "ANTHROPIC_BASE_URL",
    },
}


class DaydreamReviewAgent(BaseAgent):
    """Run a daydream review against the compiled case's checked-out history."""

    @staticmethod
    def name() -> str:
        return "daydream-review"

    def version(self) -> str:
        try:
            return importlib.metadata.version("daydream")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to provision: the wheel and checkout are baked by the image."""
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the standalone ``daydream`` review CLI against the case checkout."""
        backend = (self._get_env("DAYDREAM_REVIEW_BACKEND") or _DEFAULT_BACKEND).strip()
        if not backend:
            backend = _DEFAULT_BACKEND
        model = self._get_env("DAYDREAM_REVIEW_MODEL")
        workdir = str(environment.task_env_config.workdir or _REPO_DIR)

        argv = ["daydream", "--non-interactive", "--review", "--backend", backend]
        if model:
            argv += ["--model", model]
        argv += ["--base", _BASE_REF, workdir]

        env: dict[str, str] = {}
        for review_key, native_key in _BACKEND_CREDENTIAL_ENVARS.get(backend, {}).items():
            value = self._get_env(review_key)
            if value is not None:
                env[native_key] = value

        self.logger.info("running daydream review: %s", " ".join(shlex.quote(p) for p in argv))
        result = await environment.exec(
            " ".join(shlex.quote(part) for part in argv),
            cwd=workdir,
            env=env or None,
            timeout_sec=None,
        )
        if result.return_code != 0:
            detail = (result.stdout or result.stderr or "").strip()[:500]
            raise RuntimeError(
                f"daydream review exited {result.return_code} for {workdir}: {detail}"
            )
