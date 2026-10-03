# daydream/backends/claude.py
"""Run-owned Claude SDK transport and startup."""

from __future__ import annotations

import json
import platform
import re
import shutil
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from subprocess import PIPE
from typing import Any

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk._errors import CLIConnectionError, CLINotFoundError
from claude_agent_sdk._internal._task_compat import spawn_detached
from claude_agent_sdk._internal.query import Query
from claude_agent_sdk._internal.transport import subprocess_cli as _sdk_subprocess
from claude_agent_sdk._version import __version__ as _sdk_version
from claude_agent_sdk.types import (
    _hooks_to_internal_format,
)


class _RunLocalSubprocessCLITransport(_sdk_subprocess.SubprocessCLITransport):
    """SDK subprocess transport that never merges the host process environment."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        environment: dict[str, str],
    ) -> None:
        self._run_environment = dict(environment)

        async def _empty_prompt() -> AsyncGenerator[dict[str, Any], None]:
            messages: tuple[dict[str, Any], ...] = ()
            for message in messages:
                yield message

        prompt: AsyncIterable[dict[str, Any]] = _empty_prompt()
        super().__init__(prompt=prompt, options=options)

    def _find_cli(self) -> str:
        bundled_cli = self._find_bundled_cli()
        if bundled_cli:
            return bundled_cli

        search_path = self._run_environment.get("PATH", "")
        which_hit = shutil.which("claude", path=search_path)
        if which_hit and (platform.system() != "Windows" or self._is_windows_native_exe(which_hit)):
            return which_hit
        if platform.system() == "Windows":
            exe = shutil.which("claude.exe", path=search_path)
            if exe and self._is_windows_native_exe(exe):
                return exe

        home_raw = self._run_environment.get("HOME") or self._run_environment.get("USERPROFILE")
        if home_raw:
            home = Path(home_raw)
            locations = (
                [home / ".local/bin/claude.exe"]
                if platform.system() == "Windows"
                else [
                    home / ".npm-global/bin/claude",
                    home / ".local/bin/claude",
                    home / "node_modules/.bin/claude",
                    home / ".yarn/bin/claude",
                    home / ".claude/local/claude",
                ]
            )
            for path in locations:
                if path.exists() and path.is_file():
                    return str(path)
        if platform.system() != "Windows":
            for path in (Path("/usr/local/bin/claude"),):
                if path.exists() and path.is_file():
                    return str(path)
        if which_hit is not None:
            return which_hit
        raise CLINotFoundError("Claude Code not found in the run-owned execution environment")

    def _native_environment(self) -> dict[str, str]:
        process_env = {key: value for key, value in self._run_environment.items() if key != "CLAUDECODE"}
        process_env.setdefault("CLAUDE_CODE_ENTRYPOINT", "sdk-py")
        process_env["CLAUDE_AGENT_SDK_VERSION"] = _sdk_version

        try:
            from opentelemetry import propagate

            carrier: dict[str, str] = {}
            propagate.inject(carrier)
            if "traceparent" in carrier:
                for key in ("TRACEPARENT", "TRACESTATE"):
                    if key not in self._run_environment:
                        process_env.pop(key, None)
                for key, value in carrier.items():
                    upper = key.upper()
                    if upper not in self._run_environment:
                        process_env[upper] = value
        except Exception:  # noqa: BLE001 - tracing remains best effort
            _sdk_subprocess.logger.debug("OTEL trace context injection failed", exc_info=True)

        if self._options.enable_file_checkpointing:
            process_env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"
        if self._cwd:
            process_env["PWD"] = self._cwd
        return process_env

    async def connect(self) -> None:
        """Start the SDK CLI with only the run-owned complete environment."""
        if self._process:
            return
        if self._cli_path is None:
            self._cli_path = await anyio.to_thread.run_sync(self._find_cli)
        self._reject_windows_batch_cli(self._cli_path)
        if not self._run_environment.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
            await self._check_claude_version()

        cmd = self._build_command()
        try:
            stderr_dest = PIPE if self._options.stderr is not None else None
            self._process = await anyio.open_process(
                cmd,
                stdin=PIPE,
                stdout=PIPE,
                stderr=stderr_dest,
                cwd=self._cwd,
                env=self._native_environment(),
                user=self._options.user,
            )
            _sdk_subprocess._ACTIVE_CHILDREN.add(self._process)
            if self._process.stdout:
                self._stdout_stream = TextReceiveStream(self._process.stdout)
            if stderr_dest is PIPE and self._process.stderr:
                self._stderr_stream = TextReceiveStream(self._process.stderr)
                self._stderr_task = spawn_detached(self._handle_stderr())
            if self._process.stdin:
                self._stdin_stream = TextSendStream(self._process.stdin)
            self._ready = True
        except FileNotFoundError as exc:
            if self._cwd and not Path(self._cwd).exists():
                error = CLIConnectionError(f"Working directory does not exist: {self._cwd}")
                self._exit_error = error
                raise error from exc
            error = CLINotFoundError(f"Claude Code not found at: {self._cli_path}")
            self._exit_error = error
            raise error from exc
        except Exception as exc:
            error = CLIConnectionError(f"Failed to start Claude Code: {exc}")
            self._exit_error = error
            raise error from exc

    async def _check_claude_version(self) -> None:
        """Run the SDK version probe with the same run-owned environment."""
        if self._cli_path is None:
            raise CLINotFoundError("CLI path not resolved. Call connect() first.")
        version_process = None
        try:
            with anyio.fail_after(2):
                version_process = await anyio.open_process(
                    [self._cli_path, "-v"],
                    stdout=PIPE,
                    stderr=PIPE,
                    env=self._native_environment(),
                )
                if version_process.stdout:
                    stdout_bytes = await version_process.stdout.receive()
                    version_output = stdout_bytes.decode().strip()
                    match = re.match(r"([0-9]+\.[0-9]+\.[0-9]+)", version_output)
                    if match:
                        version_parts = [int(value) for value in match.group(1).split(".")]
                        minimum_parts = [int(value) for value in _sdk_subprocess.MINIMUM_CLAUDE_CODE_VERSION.split(".")]
                        if version_parts < minimum_parts:
                            _sdk_subprocess.logger.warning(
                                "Claude Code version %s at %s is unsupported; minimum is %s",
                                match.group(1),
                                self._cli_path,
                                _sdk_subprocess.MINIMUM_CLAUDE_CODE_VERSION,
                            )
        except Exception:
            pass
        finally:
            if version_process:
                with suppress(Exception):
                    version_process.terminate()
                with suppress(Exception):
                    await version_process.wait()


class _RunLocalClaudeSDKClient(ClaudeSDKClient):
    """Construct the pinned SDK Query with a run-owned timeout instead of ambient configuration.
    Inherited error unwind, message handling, disconnect, and context-manager lifecycle remain intact.
    """

    def __init__(
        self,
        *,
        options: ClaudeAgentOptions,
        transport: _RunLocalSubprocessCLITransport,
        initialize_timeout_s: float,
    ) -> None:
        super().__init__(options=options, transport=transport)
        self._run_initialize_timeout_s = initialize_timeout_s

    async def _connect_inner(
        self,
        prompt: str | AsyncIterable[dict[str, Any]] | None,
        actual_prompt: AsyncIterable[dict[str, Any]],
    ) -> None:
        assert self._custom_transport is not None
        self._transport = self._custom_transport
        await self._transport.connect()

        agents_dict = (
            {
                name: {key: value for key, value in asdict(agent_definition).items() if value is not None}
                for name, agent_definition in self.options.agents.items()
            }
            if self.options.agents
            else None
        )
        exclude_dynamic_sections: bool | None = None
        system_prompt = self.options.system_prompt
        if isinstance(system_prompt, dict) and system_prompt.get("type") == "preset":
            candidate = system_prompt.get("exclude_dynamic_sections")
            if isinstance(candidate, bool):
                exclude_dynamic_sections = candidate

        self._query = Query(
            transport=self._transport,
            is_streaming_mode=True,
            can_use_tool=self.options.can_use_tool,
            hooks=(_hooks_to_internal_format(self.options.hooks) if self.options.hooks else None),
            sdk_mcp_servers={},
            initialize_timeout=self._run_initialize_timeout_s,
            agents=agents_dict,
            exclude_dynamic_sections=exclude_dynamic_sections,
            skills=self.options.skills,
            forward_subagent_text=self.options.forward_subagent_text,
        )
        await self._query.start()
        await self._query.initialize()
        if isinstance(prompt, str):
            message = {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": "default",
            }
            await self._transport.write(json.dumps(message) + "\n")
        elif prompt is not None and isinstance(prompt, AsyncIterable):
            self._query.spawn_task(self._query.stream_input(actual_prompt))


def _run_local_initialize_timeout_s(environment: dict[str, str]) -> float:
    """Parse the SDK initialization window from one run-owned environment."""
    raw = environment.get("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "60000")
    try:
        timeout_ms = int(raw)
    except ValueError:
        timeout_ms = 60000
    return max(timeout_ms / 1000.0, 60.0)
