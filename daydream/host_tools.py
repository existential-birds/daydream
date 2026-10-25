"""One exact-file capability and its invocation-local lifetime.

The fixed operation has fixed protocol bounds. Live authority, admissions and the
isolated reader belong to one session; this does not grant snapshot isolation.
The official MCP SDK is loaded only when a session is explicitly requested.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from daydream import clock

if TYPE_CHECKING:
    from daydream.host_tools_server import PiHostToolServer

_MAX_CALLS = 8
_MAX_FRAME_BYTES = 16384
_MAX_ARGUMENT_BYTES = 4096
_MAX_RESULT_BYTES = 12288
_MAX_ITEMS = 128
_MAX_HTTP_REQUESTS = 128
_CALLBACK_TIMEOUT_S = 3.0
_SETUP_TIMEOUT_S = 5.0
_CLEANUP_TIMEOUT_S = 2.0
NATIVE_READ_FILE = "mcp__daydream_host__read_file"
_READ_DESCRIPTION = "Read the exact UTF-8 contents of an explicitly permitted absolute file path."


def _read_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 2048}},
            "required": ["path"], "additionalProperties": False}


class HostToolRejected(Exception):
    """A bounded correctable native tool error, never a supervisor veto."""


class HostToolBinding:
    """Single-use exact-path authority, intersected with the original caller policy."""

    def __init__(self, permitted_paths: tuple[Path, ...], *, deadline: float) -> None:
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("host tools require a finite absolute monotonic deadline")
        paths = tuple(str(path.absolute()) for path in permitted_paths)
        if (not 1 <= len(paths) <= 128 or len(set(paths)) != len(paths)
                or any(path.startswith("//") or len(path.encode()) > 2048
                       or os.path.normpath(path) != path for path in paths)):
            raise ValueError("host tools require 1..128 distinct bounded normalized paths starting with exactly one /")
        self._paths = frozenset(paths)
        self.deadline = float(deadline)
        self.calls = self.rejections = 0
        self.authority_open = self.native_settled = self.cleanup_complete = False
        self._claimed = self._cleanup_failed = False
        self._call_limit = _MAX_CALLS
        self._admitted: set[str] = set()
        self._pending: dict[str, list[str]] = {}
        self._reading = False
        self._reader: asyncio.subprocess.Process | None = None
        self._late_cleanup: set[asyncio.Task[None]] = set()

    def restrict(self, *, deadline: float | None, tool_call_budget: int | None) -> None:
        """Intersect caller policy; progress cannot replenish the original allowance."""
        if deadline is not None:
            self.deadline = min(self.deadline, deadline)
        if tool_call_budget is not None:
            self._call_limit = min(self._call_limit, tool_call_budget)

    @staticmethod
    def validate_platform() -> None:
        if (os.name != "posix" or not all(hasattr(os, name) for name in (
                "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")) or os.open not in os.supports_dir_fd):
            raise NotImplementedError("Host reads require POSIX O_NOFOLLOW, O_DIRECTORY and dir_fd support")

    def _check_open(self) -> None:
        if not self.authority_open or clock.monotonic() >= self.deadline:
            self.close_authority()
            raise self._reject("Host-tool authority is closed or its absolute deadline expired")

    def _reject(self, message: str) -> HostToolRejected:
        self.rejections += 1
        return HostToolRejected(message)

    def admit(self, call_id: Any, name: Any, arguments: Any, *, malformed: bool = False) -> None:
        """Charge before native validation, including bounded malformed admissions."""
        self._check_open()
        try:
            valid_id = isinstance(call_id, str) and 0 < len(call_id.encode()) <= 512
        except UnicodeError:
            valid_id = False
        if not valid_id or not isinstance(name, str) or len(name) > 256:
            self.close_authority()
            raise self._reject("Invalid native tool identity")
        if call_id in self._admitted:
            raise self._reject("Native tool identity was already charged")
        if len(self._admitted) >= _MAX_CALLS:
            self.close_authority()
            raise self._reject("Host-tool call capacity exhausted")
        self.calls += 1
        self._admitted.add(call_id)
        if self.calls > self._call_limit:
            raise self._reject("Original cumulative tool allowance exhausted")
        if malformed:
            raise self._reject("Invalid bounded host frame")
        if name == NATIVE_READ_FILE:
            self._pending.setdefault(self._arguments_key(arguments), []).append(call_id)

    def _arguments_key(self, arguments: Any) -> str:
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                 separators=(",", ":"))
            byte_count = len(encoded.encode("utf-8", errors="strict"))
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise self._reject("Invalid host-tool arguments") from None
        if byte_count > _MAX_ARGUMENT_BYTES:
            raise self._reject("Host-tool argument byte limit exceeded")
        return encoded

    async def read_file(self, arguments: Any) -> str:
        self._check_open()
        key = self._arguments_key(arguments)
        pending = self._pending.get(key)
        if not pending:
            raise self._reject("Host callback has no native admission")
        pending.pop(0)
        if not pending:
            self._pending.pop(key)
        if (not isinstance(arguments, dict) or set(arguments) != {"path"}
                or not isinstance(arguments["path"], str) or not 1 <= len(arguments["path"]) <= 2048):
            raise self._reject("Expected exactly one bounded string path")
        if arguments["path"] not in self._paths:
            raise self._reject("Path is not permitted by this invocation")
        if self._reading:
            raise self._reject("Host read concurrency exhausted; no callbacks are queued")
        self._reading = True
        proc = self._reader
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        try:
            async with asyncio.timeout(min(_CALLBACK_TIMEOUT_S, self.deadline - clock.monotonic())):
                self._check_open()
                # This command is the read gate. An unreturned startup never gets it.
                proc.stdin.write(json.dumps(arguments["path"]).encode() + b"\n")
                await proc.stdin.drain()
                header = await proc.stdout.readline()
                if header == b"E\n":
                    raise self._reject("Permitted file unavailable, nonregular, symlinked, or exceeds the byte limit")
                size = int(header)
                if not 0 <= size <= _MAX_RESULT_BYTES:
                    raise ValueError("invalid reader frame")
                data = await proc.stdout.readexactly(size)
                self._check_open()
                try:
                    text = data.decode("utf-8", errors="strict")
                except UnicodeError:
                    raise self._reject("Permitted file is not UTF-8") from None
                if len(json.dumps(text, ensure_ascii=False).encode()) + 1024 > _MAX_RESULT_BYTES:
                    raise self._reject("Host-tool serialized result byte limit exceeded")
                return text
        except (TimeoutError, OSError, ValueError, asyncio.IncompleteReadError):
            self.close_authority()
            self._kill_reader()
            raise self._reject("Host read callback deadline expired or reader failed") from None
        except asyncio.CancelledError:
            self.close_authority()
            self._kill_reader()
            raise
        finally:
            self._reading = False

    @asynccontextmanager
    async def session(self) -> AsyncIterator[PiHostToolServer]:
        """Own the optional SDK endpoint and one inert-until-command isolated reader."""
        from daydream.backends.pi import PiError

        if self._claimed:
            raise ValueError("host-tool binding has already been used")
        self._claimed = True
        server: PiHostToolServer | None = None
        try:
            guidance = "Pi host tools require daydream[host-tools] with official mcp==1.30.0 and its bundled FastMCP"
            try:
                if version("mcp") != "1.30.0":
                    raise ImportError("unsupported SDK version")
                module = import_module("daydream.host_tools_server")
            except (ImportError, PackageNotFoundError):
                raise PiError(guidance + "; install a compatible host-tools extra", category="HOST_TOOLS") from None
            self.validate_platform()
            server = module.PiHostToolServer(self)
            async with asyncio.timeout(min(_SETUP_TIMEOUT_S, max(0, self.deadline - clock.monotonic()))):
                self._reader = await self.start(asyncio.create_subprocess_exec(
                    sys.executable, "-I", str(Path(__file__).resolve().with_name("host_tools_read.py")),
                    str(_MAX_RESULT_BYTES), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, start_new_session=True, limit=_MAX_RESULT_BYTES + 1024),
                    self._stop_reader)
                await server.start()
            self.authority_open = True
            self._check_open()
            yield server
        finally:
            self.close_authority()
            self._kill_reader()
            with anyio.CancelScope(shield=True):
                try:
                    try:
                        async with asyncio.timeout(_CLEANUP_TIMEOUT_S):
                            if self._reader is not None:
                                await self._reader.wait()
                            await asyncio.gather(*self._late_cleanup, return_exceptions=True)
                    finally:
                        # Reader EOF releases active callbacks; SDK shutdown must
                        # still run if reader reap or a late cleanup times out.
                        if server is not None:
                            await server.close()
                except TimeoutError:
                    self.mark_cleanup_failed()
            self.cleanup_complete = (not self._cleanup_failed and not self._reading
                                     and (self._reader is None or self._reader.returncode is not None)
                                     and not self._late_cleanup)

    async def start[T](self, operation: Coroutine[Any, Any, T],
                       cleanup: Callable[[T], Coroutine[Any, Any, None]]) -> T:
        """Keep ownership when native or reader startup returns after cancellation."""
        task = asyncio.create_task(operation)
        try:
            return await asyncio.shield(task)
        except BaseException:
            self.close_authority()
            joined = False
            try:
                with anyio.CancelScope(shield=True):
                    async with asyncio.timeout(_CLEANUP_TIMEOUT_S):
                        await cleanup(await asyncio.shield(task))
                        joined = True
            except Exception:
                pass  # Preserve the original startup failure.
            finally:
                if not joined and not task.cancelled() and (not task.done() or task.exception() is None):
                    self.mark_cleanup_failed()

                    def finish(completed: asyncio.Task[T]) -> None:
                        if not completed.cancelled() and completed.exception() is None:
                            self.track_cleanup(cleanup(completed.result()))

                    task.add_done_callback(finish)
            raise

    def _kill_reader(self) -> None:
        if self._reader is not None:
            if self._reader.stdin is not None:
                self._reader.stdin.close()
            if self._reader.returncode is None:
                try:
                    self._reader.kill()
                except ProcessLookupError:
                    pass

    async def _stop_reader(self, proc: asyncio.subprocess.Process) -> None:
        self._reader = proc
        self._kill_reader()
        await proc.wait()

    def track_cleanup(self, cleanup: Coroutine[Any, Any, Any]) -> None:
        """Own late isolated transport cleanup without writing frozen evidence."""
        task = asyncio.create_task(cleanup)
        self._late_cleanup.add(task)
        def finished(completed: asyncio.Task[None]) -> None:
            self._late_cleanup.discard(completed)
            if completed.cancelled() or completed.exception() is not None:
                self.mark_cleanup_failed()
        task.add_done_callback(finished)

    def mark_cleanup_failed(self) -> None:
        self._cleanup_failed = True
        self.cleanup_complete = False

    def close_authority(self) -> None:
        """Synchronous revocation precedes every shutdown await."""
        self.authority_open = False
        self._pending.clear()
