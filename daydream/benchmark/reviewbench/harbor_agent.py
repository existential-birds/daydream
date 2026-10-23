"""Public Harbor adapter for Daydream's selected native reviewer.

The custom exported image installs the frozen candidate wheel and selected native CLI.
Authentication is a mounted role-scoped capability, never an agent kwarg value.
"""
from __future__ import annotations

import asyncio
import json
import math
import shlex
import time
import tomllib
import uuid
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
        probe = ('import os,shutil;from pathlib import Path;'
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
        from daydream.benchmark.reviewbench.harbor_process import (
            FORCE_REAP_SECONDS,
            SHUTDOWN_GRACE_SECONDS,
            cancellation_marker,
            warn_diagnostics_unavailable,
        )

        invocation = str(uuid.uuid4())
        started = time.monotonic()
        declared_timeout = _declared_outer_timeout(Path(self.logs_dir))
        command = ('python -I -m daydream.benchmark.reviewbench.harbor_process --invocation ' + invocation)
        if declared_timeout is not None:
            command += ' --outer-timeout ' + str(declared_timeout)
        execution = asyncio.create_task(environment.exec(command, cwd='/work/repo', timeout_sec=3600))
        try:
            result = await asyncio.shield(execution)
        except (asyncio.CancelledError, Exception) as failure:
            interrupted = isinstance(failure, asyncio.CancelledError)
            reason = 'host_task_cancelled' if interrupted else 'host_execution_failed'
            channels: dict[str, str] = {}
            try:
                from daydream.benchmark.reviewbench.harbor_process import publish_diagnostic_bytes

                marker = {'invocation_id': invocation, 'interruption': reason}
                publish_diagnostic_bytes(cancellation_marker(Path(self.logs_dir), invocation),
                    (json.dumps(marker, sort_keys=True) + '\n').encode())
                channels['shared_marker'] = 'published'
            except Exception as exc:
                channels['shared_marker'] = 'unavailable:' + type(exc).__name__
                warn_diagnostics_unavailable('host cancellation marker', exc)
            # The in-container supervisor observes the shared marker even when
            # Harbor's collector is cancelled or fails. Join a cooperative
            # collector before returning; a final receipt covers failed collectors.
            # One absolute bound also applies across repeated cancellation.
            deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS + FORCE_REAP_SECONDS + 10

            async def stop_and_collect() -> None:
                while time.monotonic() < deadline:
                    if execution.done() and (not execution.cancelled() and execution.exception() is None
                            or _process_completion_observed(Path(self.logs_dir), invocation)):
                        return
                    await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

            cleanup = asyncio.create_task(stop_and_collect())
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            # Detached noncooperative collectors must not hold the agent's
            # cancellation open indefinitely; the supervisor has its own bound.
            execution.add_done_callback(_consume_task_result)
            cleanup.add_done_callback(_consume_task_result)
            try:
                from daydream.benchmark.reviewbench.harbor_process import publish_diagnostic_bytes

                publish_diagnostic_bytes(Path(self.logs_dir) / 'reviewer-host.json', (json.dumps({
                    'schema': 1, 'invocation_id': invocation, 'interruption': reason,
                    'execution_error': None if interrupted else type(failure).__name__[:80],
                    'cancellation_channels': channels,
                    'declared_outer_timeout_seconds': declared_timeout,
                    'elapsed_seconds': time.monotonic() - started,
                    'output_collection': 'returned' if execution.done() and not execution.cancelled()
                        and execution.exception() is None else 'unavailable',
                    'process_completion': 'observed' if _process_completion_observed(Path(self.logs_dir), invocation)
                        else 'unavailable',
                }, indent=2) + '\n').encode())
            except Exception as exc:
                warn_diagnostics_unavailable('host receipt', exc)
            raise
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


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Observe detached collector errors without changing Harbor's cancellation."""
    if not task.cancelled():
        task.exception()


def _process_completion_observed(logs: Path, invocation: str) -> bool:
    """Observe this invocation's bounded final receipt without following a leaf."""
    from daydream.benchmark.reviewbench.harbor_process import read_diagnostic_bytes

    try:
        receipt = json.loads(read_diagnostic_bytes(logs / 'reviewer-process.json'))
        return (isinstance(receipt, dict) and receipt.get('schema') == 1
                and receipt.get('invocation_id') == invocation
                and (type(receipt.get('child_returncode')) is int
                     or receipt.get('cleanup_outcome') in ('cancelled_before_start', 'startup_failed')))
    except (OSError, ValueError):
        return False


def _declared_outer_timeout(logs: Path) -> float | None:
    """Read the pinned Harbor trial's declared timeout without changing policy."""
    try:
        path = logs.parent / 'config.json'
        if path.stat().st_size > 1048576:
            return None
        trial = json.loads(path.read_bytes())
        agent = trial['agent']
        base = agent.get('override_timeout_sec')
        if not base:
            task_path = Path(trial['task']['path']) / 'task.toml'
            if task_path.stat().st_size > 1048576:
                return None
            base = tomllib.loads(task_path.read_text())['agent']['timeout_sec']
        ceiling = agent.get('max_timeout_sec') or float('inf')
        multiplier = trial.get('agent_timeout_multiplier')
        multiplier = trial.get('timeout_multiplier', 1.0) if multiplier is None else multiplier
        if any(type(value) not in {int, float} or value <= 0 for value in (base, ceiling, multiplier)):
            return None
        resolved = min(base, ceiling) * multiplier
        return float(resolved) if math.isfinite(resolved) else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
