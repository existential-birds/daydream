"""Public Harbor adapter for Daydream's selected native reviewer.

The custom exported image installs the frozen candidate wheel and selected native CLI.
Authentication is a mounted role-scoped capability, never an agent kwarg value.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

try:
    from harbor.agents.base import BaseAgent
except ImportError:
    BaseAgent = object


class DaydreamPublicReviewAgent(BaseAgent):  # type: ignore[misc]
    """Run Daydream without exposing independent verifier assets or credentials."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return 'daydream-public-reviewbench'

    @classmethod
    def version(cls) -> str:
        from daydream import __version__

        return __version__

    async def setup(self, environment: Any) -> None:
        probe = ('import importlib.metadata,os,shutil;from pathlib import Path;'
                 f'assert importlib.metadata.version("daydream")=={self.version()!r};'
                 'from daydream.benchmark.reviewbench.harbor_entrypoint import load_public_task;'
                 'p=Path(os.environ.get("DAYDREAM_RB_INPUTS","/work/pr"));'
                 '_,profile=load_public_task(p/"daydream-profile.json",p/"pr.json",p/"diff.patch");'
                 'assert shutil.which(profile["reviewer"]["backend"])')
        result = await environment.exec('python -I -c ' + shlex.quote(probe))
        if result.return_code != 0:
            raise RuntimeError('public ReviewBench candidate image setup failed')

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        # The upstream instruction is embodied by the exact frozen task metadata.
        # Do not forward arbitrary agent env (including API keys) to this lane.
        if self.extra_env:
            raise ValueError('public ReviewBench agent does not accept extra_env; use the reviewer credential mount')
        result = await environment.exec('python -I -m daydream.benchmark.reviewbench.harbor_entrypoint',
                                        cwd='/work/repo', timeout_sec=3600)
        from daydream.redaction import redact_structured_text

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log = Path(self.logs_dir) / 'reviewer-entrypoint.log'
        with log.open('x') as stream:
            stream.write(redact_structured_text((result.stdout or '') + (result.stderr or '')))
        if result.return_code != 0:
            raise RuntimeError('public ReviewBench native reviewer failed; inspect retained execution evidence')

    def populate_context_post_run(self, context: Any) -> None:
        """Preserve measured token totals; monetary estimates remain in evidence."""
        path = Path(self.logs_dir) / 'daydream' / 'execution.json'
        try:
            execution = json.loads(path.read_text())
            usage = execution.get('usage', {})
            for attribute, name in (('n_input_tokens', 'prompt_tokens'), ('n_output_tokens', 'completion_tokens'),
                                    ('n_cache_tokens', 'cached_tokens')):
                value = usage.get(name)
                if value is not None and hasattr(context, attribute):
                    setattr(context, attribute, value)
        except (OSError, ValueError, TypeError, AttributeError):
            return
