"""One explicitly granted host read, with invocation-local authority and bounds.

Definitions carry no live state. A binding is single-use and cannot be placed in
configuration: its paths, deadline, admissions and child processes are private to
one invocation. This seam does not grant snapshot isolation or review admission.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio

from daydream import clock


@dataclass(frozen=True)
class HostToolDefinition:
    name: str
    description: str
    schema_json: str

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name)
                or not 1 <= len(self.description.encode()) <= 1024 or len(self.schema_json.encode()) > 4096):
            raise ValueError("invalid bounded host-tool definition")
        if not isinstance(json.loads(self.schema_json), dict):
            raise ValueError("host-tool schema must be an object")

    def schema(self) -> dict[str, Any]:
        """Return a fresh schema; callers cannot mutate the shared definition."""
        value: dict[str, Any] = json.loads(self.schema_json)
        return value


READ_FILE = HostToolDefinition(
    "read_file", "Read the exact UTF-8 contents of an explicitly permitted absolute file path.",
    '{"type":"object","properties":{"path":{"type":"string","minLength":1,"maxLength":2048}},'
    '"required":["path"],"additionalProperties":false}',
)
NATIVE_READ_FILE = "mcp__daydream_host__read_file"


@dataclass(frozen=True)
class HostToolLimits:
    max_calls: int = 8
    max_frame_bytes: int = 16384
    max_argument_bytes: int = 4096
    max_result_bytes: int = 12288
    max_items: int = 128
    max_in_flight: int = 2
    max_http_requests: int = 128
    callback_timeout_s: float = 3.0
    setup_timeout_s: float = 5.0
    cleanup_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        for name, maximum in {
            "max_calls": 256, "max_frame_bytes": 65536, "max_argument_bytes": 8192,
            "max_result_bytes": 16384, "max_items": 1024, "max_in_flight": 8, "max_http_requests": 2048,
        }.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid host-tool limit: {name}")
        if self.max_frame_bytes < 256:
            raise ValueError("host-tool frames require at least 256 bytes for bounded protocol errors")
        if self.max_result_bytes < 256:
            raise ValueError("host-tool results require at least 256 bytes for bounded protocol errors")
        for name in ("callback_timeout_s", "setup_timeout_s", "cleanup_timeout_s"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 30:
                raise ValueError(f"invalid host-tool limit: {name}")


class HostToolRejected(Exception):
    """A bounded correctable native tool error, never a supervisor veto."""


class HostToolBinding:
    """An exact-path read grant and cumulative authority for a single invocation."""

    def __init__(self, permitted_paths: tuple[Path, ...], *, deadline: float,
                 limits: HostToolLimits = HostToolLimits()) -> None:
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("host tools require a finite absolute monotonic deadline")
        if not 1 <= len(permitted_paths) <= 128:
            raise ValueError("host tools require 1..128 exact permitted paths")
        paths = tuple(str(path.absolute()) for path in permitted_paths)
        if any(len(path.encode()) > 2048 or os.path.normpath(path) != path for path in paths):
            raise ValueError("host-tool paths must be bounded normalized absolute paths")
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate host-tool path")
        self._paths = frozenset(paths)
        self.deadline = float(deadline)
        self.limits = limits
        self.calls = 0
        self.rejections = 0
        self.authority_open = False
        self.native_settled = False
        self.cleanup_complete = False
        self._claimed = False
        self._call_limit = limits.max_calls
        self._admitted: dict[str, str | None] = {}
        self._pending: dict[str, list[str]] = {}
        self._callbacks: set[asyncio.Task[Any]] = set()
        self._children: set[asyncio.subprocess.Process] = set()
        self._spawns: set[asyncio.Task[asyncio.subprocess.Process]] = set()
        self._cleanup_failed = False
        self._late_cleanup: set[asyncio.Task[None]] = set()
        self._abandoned_spawns: set[asyncio.Task[asyncio.subprocess.Process]] = set()

    def restrict(self, *, deadline: float | None, tool_call_budget: int | None) -> None:
        """Intersect caller policy; neither retries nor progress can replenish it."""
        if deadline is not None:
            self.deadline = min(self.deadline, deadline)
        if tool_call_budget is not None:
            self._call_limit = min(self._call_limit, tool_call_budget)

    def claim(self) -> None:
        self.validate_platform()
        if self._claimed:
            raise ValueError("host-tool binding has already been used")
        self._claimed = True
        self.authority_open = True
        self._check_open()

    @staticmethod
    def validate_platform() -> None:
        """Reject missing filesystem guarantees before any model work."""
        if (os.name != "posix" or not all(hasattr(os, name) for name in (
                "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")) or os.open not in os.supports_dir_fd):
            raise NotImplementedError("Host reads require POSIX O_NOFOLLOW, O_DIRECTORY and dir_fd support")

    def _check_open(self) -> None:
        if not self.authority_open:
            raise self._reject("Host-tool authority is closed")
        if clock.monotonic() >= self.deadline:
            self.authority_open = False
            raise self._reject("Host-tool absolute deadline expired")

    def _reject(self, message: str) -> HostToolRejected:
        self.rejections += 1
        return HostToolRejected(message)

    def admit(self, call_id: Any, name: Any, arguments: Any) -> None:
        """Charge the native callback once, before the native tool can execute."""
        self._charge(call_id, name)
        if name == NATIVE_READ_FILE:
            key = self._arguments_key(arguments)
            self._admitted[call_id] = key
            self._pending.setdefault(key, []).append(call_id)

    def reject_admission(self, call_id: Any, name: Any) -> None:
        """Charge a bounded identity whose frame cannot grant a callback."""
        self._charge(call_id, name)
        raise self._reject("Invalid bounded host frame")

    def _charge(self, call_id: Any, name: Any) -> None:
        self._check_open()
        try:
            identity_bytes = len(call_id.encode()) if isinstance(call_id, str) else 0
        except UnicodeError:
            self.close_authority()
            raise self._reject("Invalid native tool identity") from None
        if (not isinstance(call_id, str) or not call_id or identity_bytes > 512
                or not isinstance(name, str) or len(name) > 256):
            self.close_authority()
            raise self._reject("Invalid native tool identity")
        if call_id in self._admitted:
            raise self._reject("Native tool identity was already charged")
        if len(self._admitted) >= self.limits.max_calls:
            self.close_authority()
            raise self._reject("Host-tool call capacity exhausted")
        self.calls += 1
        self._admitted[call_id] = None
        if self.calls > self._call_limit:
            raise self._reject("Original cumulative tool allowance exhausted")

    def _arguments_key(self, arguments: Any) -> str:
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                 separators=(",", ":"))
            byte_count = len(encoded.encode("utf-8", errors="strict"))
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise self._reject("Invalid host-tool arguments") from None
        if byte_count > self.limits.max_argument_bytes:
            raise self._reject("Host-tool argument byte limit exceeded")
        return encoded

    async def read_file(self, arguments: Any) -> str:
        """Authorize before spawning an isolated bounded regular-file reader."""
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("host callback requires an asyncio task")
        self._check_open()
        key = self._arguments_key(arguments)
        pending = self._pending.get(key)
        if not pending:
            raise self._reject("Host callback has no native admission")
        pending.pop(0)
        if not pending:
            self._pending.pop(key, None)
        if (not isinstance(arguments, dict) or set(arguments) != {"path"}
                or not isinstance(arguments["path"], str) or not 1 <= len(arguments["path"]) <= 2048):
            raise self._reject("Expected exactly one bounded string path")
        path = arguments["path"]
        if path not in self._paths:
            raise self._reject("Path is not permitted by this invocation")
        if len(self._callbacks) >= self.limits.max_in_flight:
            raise self._reject("Host read concurrency exhausted; no callbacks are queued")
        self._callbacks.add(task)
        proc: asyncio.subprocess.Process | None = None
        spawn: asyncio.Task[asyncio.subprocess.Process] | None = None
        try:
            self._check_open()
            timeout = min(self.limits.callback_timeout_s, self.deadline - clock.monotonic())
            async with asyncio.timeout(timeout):
                spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                    sys.executable, "-I", str(Path(__file__).resolve().with_name("host_tools_read.py")),
                    path, str(self.limits.max_result_bytes),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True, limit=self.limits.max_result_bytes + 1024,
                ))
                self._spawns.add(spawn)
                spawn.add_done_callback(self._track_spawn)
                proc = await asyncio.shield(spawn)
                self._children.add(proc)
                self._check_open()
                assert proc.stdin is not None
                proc.stdin.write(b"R")
                proc.stdin.close()
                data, _ = await proc.communicate()
                self._check_open()
                if proc.returncode != 0:
                    raise self._reject("Permitted file unavailable, nonregular, symlinked, or exceeds the byte limit")
                try:
                    text = data.decode("utf-8", errors="strict")
                except UnicodeError:
                    raise self._reject("Permitted file is not UTF-8") from None
                # Includes the native MCP envelope margin. Never truncate bytes.
                if len(json.dumps(text, ensure_ascii=False).encode()) + 1024 > self.limits.max_result_bytes:
                    raise self._reject("Host-tool serialized result byte limit exceeded")
                return text
        except TimeoutError:
            raise self._reject("Host read callback deadline expired") from None
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    try:
                        async with asyncio.timeout(self.limits.cleanup_timeout_s):
                            if spawn is not None:
                                proc = await asyncio.shield(spawn)
                                self._spawns.discard(spawn)
                                self._children.add(proc)
                            if proc is not None:
                                await self._reap(proc)
                    except TimeoutError:
                        self.mark_cleanup_failed()
                        if spawn is not None:
                            # An unreturned child has never received the read gate.
                            # Its eventual transport is killed/reaped by this owner;
                            # it cannot read or write invocation evidence meanwhile.
                            self._abandoned_spawns.add(spawn)
            finally:
                # Direct repeated Task.cancel() is independent of anyio's
                # shield. Bookkeeping must survive a second cancellation.
                self._callbacks.discard(task)

    def _track_spawn(self, task: asyncio.Task[asyncio.subprocess.Process]) -> None:
        self._spawns.discard(task)
        if task.cancelled() or task.exception() is not None:
            return
        proc = task.result()
        self._children.add(proc)
        if not self.authority_open or task in self._abandoned_spawns:
            self._kill(proc)
            cleanup = asyncio.create_task(self._reap(proc))
            self._late_cleanup.add(cleanup)
            cleanup.add_done_callback(self._finish_late_cleanup)

    def _finish_late_cleanup(self, task: asyncio.Task[None]) -> None:
        self._late_cleanup.discard(task)
        if task.cancelled() or task.exception() is not None:
            self.mark_cleanup_failed()

    def track_cleanup(self, cleanup: Coroutine[Any, Any, None]) -> None:
        """Own a late isolated transport cleanup without writing frozen evidence."""
        task = asyncio.create_task(cleanup)
        self._late_cleanup.add(task)
        task.add_done_callback(self._finish_late_cleanup)

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def mark_cleanup_failed(self) -> None:
        """A failed cleanup observation is sticky across idempotent closes."""
        self._cleanup_failed = True
        self.cleanup_complete = False

    async def _reap(self, proc: asyncio.subprocess.Process) -> None:
        self._kill(proc)
        await asyncio.wait_for(proc.wait(), timeout=self.limits.cleanup_timeout_s)
        self._children.discard(proc)

    def close_authority(self) -> None:
        """Synchronous revocation precedes every shutdown await."""
        self.authority_open = False
        self._pending.clear()

    async def close(self) -> None:
        self.close_authority()
        with anyio.CancelScope(shield=True):
            callbacks = [task for task in self._callbacks if task is not asyncio.current_task()]
            for task in callbacks:
                task.cancel()
            for proc in tuple(self._children):
                self._kill(proc)
            try:
                async with asyncio.timeout(self.limits.cleanup_timeout_s):
                    await asyncio.gather(*callbacks, return_exceptions=True)
                    await asyncio.gather(*(asyncio.shield(spawn) for spawn in tuple(self._spawns)),
                                         return_exceptions=True)
                    await asyncio.gather(*(self._reap(proc) for proc in tuple(self._children)))
                    await asyncio.gather(*self._late_cleanup, return_exceptions=True)
            except TimeoutError:
                self.mark_cleanup_failed()
        self.cleanup_complete = (not self._cleanup_failed and not self._callbacks and not self._children
                                 and not self._spawns and not self._late_cleanup)
