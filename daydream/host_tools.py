"""Invocation-local synchronous admission for native Pi review submissions.

Pi owns ordinary reads and tool execution. This binding owns one domain callback,
its original deadline and cumulative allowance, and the short-lived transport.
Acceptance runs on the event-loop thread without an await, so authority closure
and state installation have an unambiguous order.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio

from daydream import clock
from daydream.json_utils import validates_schema

if TYPE_CHECKING:
    from daydream.host_tools_server import PiHostToolServer

_MAX_FRAME_BYTES = 256 * 1024
_MAX_ARGUMENT_BYTES = 128 * 1024
_MAX_RESULT_BYTES = 128 * 1024
_MAX_ITEMS = 32768
_MAX_DEPTH = 64
_CALLBACK_TIMEOUT_S = 3.0
_SETUP_TIMEOUT_S = 5.0
_CLEANUP_TIMEOUT_S = 2.0
NATIVE_SUBMIT = "structured_output"


class HostToolRejected(Exception):
    """A bounded correctable native tool error that leaves admitted state intact."""


class HostToolBinding:
    """One review acceptance callback with attempt-local native correlation."""

    def __init__(self, *, deadline: float, accept: Callable[[Any], dict[str, Any]]) -> None:
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("host tools require a finite absolute monotonic deadline")
        if not callable(accept):
            raise TypeError("host tools require a synchronous acceptance callback")
        self.deadline = float(deadline)
        self.calls = self.rejections = 0
        self.authority_open = self.native_settled = self.cleanup_complete = False
        self._accept = accept
        self._call_limit: int | None = None
        self._schema: dict[str, Any] | None = None
        self._session_active = self._cleanup_failed = self._acceptance_started = False
        self._accepted_result: dict[str, Any] | None = None
        self._accepted_identity: tuple[str, str] | None = None
        self._admitted: set[str] = set()
        self._pending: dict[str, str] = {}
        self._late_cleanup: set[asyncio.Task[None]] = set()

    @property
    def accepted(self) -> bool:
        return self._accepted_result is not None

    @property
    def accepted_result(self) -> dict[str, Any] | None:
        return self._accepted_result

    @property
    def possibly_accepted(self) -> bool:
        """Unexpected callback failures cannot authorize replay of a mutation."""
        return self._acceptance_started

    @property
    def schema(self) -> dict[str, Any] | None:
        return self._schema

    def configure_schema(self, schema: dict[str, Any]) -> None:
        """Bind readiness and native argument validation to the model-facing schema."""
        if self._session_active or self.possibly_accepted:
            raise ValueError("cannot change an active host submission schema")
        self._schema = json.loads(json.dumps(schema, ensure_ascii=False, allow_nan=False))

    def restrict(self, *, deadline: float | None, tool_call_budget: int | None) -> None:
        """Intersect original cumulative policy; retries cannot replenish it."""
        if deadline is not None:
            self.deadline = min(self.deadline, deadline)
        if tool_call_budget is not None:
            self._call_limit = (tool_call_budget if self._call_limit is None
                                else min(self._call_limit, tool_call_budget))

    def _check_open(self) -> None:
        if not self.authority_open or clock.monotonic() >= self.deadline:
            self.close_authority()
            raise self._reject("Host-tool authority is closed or its absolute deadline expired")

    def ensure_open(self) -> None:
        """Recheck authority immediately before the domain installs prepared state."""
        self._check_open()

    def _reject(self, message: str) -> HostToolRejected:
        self.rejections += 1
        return HostToolRejected(message[:1024].encode("utf-8", errors="replace").decode())

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

    def admit(self, call_id: Any, name: Any, arguments: Any, *, terminal: bool = False,
              malformed: bool = False) -> None:
        """Charge real starts before native validation; only sole final calls can submit."""
        self._check_open()
        try:
            valid_id = isinstance(call_id, str) and 0 < len(call_id.encode()) <= 512
        except UnicodeError:
            valid_id = False
        if not valid_id or not isinstance(name, str) or not name or len(name) > 256:
            self.close_authority()
            raise self._reject("Invalid native tool identity")
        if call_id in self._admitted:
            raise self._reject("Native tool identity was already charged")
        self.calls += 1
        self._admitted.add(call_id)
        if self._call_limit is not None and self.calls > self._call_limit:
            raise self._reject("Original cumulative tool allowance exhausted")
        if malformed:
            raise self._reject("Invalid bounded host frame")
        if name == NATIVE_SUBMIT:
            if terminal is not True:
                raise self._reject("Submit structured_output alone in a completed native generation")
            self._pending[call_id] = self._arguments_key(arguments)

    def submit(self, call_id: Any, arguments: Any) -> dict[str, Any]:
        """Synchronously accept once, immediately after the final authority/deadline check."""
        key = self._arguments_key(arguments)
        if self._accepted_identity == (call_id, key):
            assert self._accepted_result is not None
            return self._accepted_result
        self._check_open()
        if self.possibly_accepted:
            raise self._reject("This invocation has already accepted its terminal submission")
        if not isinstance(call_id, str) or self._pending.pop(call_id, None) != key:
            raise self._reject("Host callback has no matching completed native terminal call")
        if not isinstance(arguments, dict) or (self._schema is not None
                                                and not validates_schema(arguments, self._schema)):
            raise self._reject("Invalid terminal submission schema")
        # All payload/identity checks precede this final synchronous boundary.
        # There is deliberately no await or detached thread between the check
        # and domain installation. The callback must preflight its result before
        # changing state; uncertain unexpected failures prohibit automatic replay.
        self._check_open()
        self._acceptance_started = True
        try:
            result = self._accept(arguments)
        except (HostToolRejected, ValueError) as exc:
            self._acceptance_started = False
            raise self._reject(str(exc)) from None
        except BaseException:
            self.close_authority()
            raise
        try:
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if not isinstance(result, dict) or len(encoded.encode("utf-8", errors="strict")) > _MAX_RESULT_BYTES:
                raise ValueError("invalid canonical result")
            admitted: dict[str, Any] = json.loads(encoded)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            self.close_authority()
            raise self._reject("Host acceptance returned an invalid canonical result") from None
        self._accepted_result = admitted
        self._accepted_identity = (call_id, key)
        self._pending.clear()
        return admitted

    @asynccontextmanager
    async def session(self) -> AsyncIterator[PiHostToolServer]:
        """Open one attempt's endpoint; unaccepted retries retain counters and deadline."""
        from daydream.host_tools_server import PiHostToolServer

        if self._session_active or self.possibly_accepted or self._cleanup_failed:
            raise ValueError("host-tool binding cannot start another native attempt")
        self._session_active = True
        self.cleanup_complete = self.native_settled = False
        self._admitted.clear()
        self._pending.clear()
        server = PiHostToolServer(self)
        try:
            async with asyncio.timeout(min(_SETUP_TIMEOUT_S, max(0, self.deadline - clock.monotonic()))):
                await server.start()
            self.authority_open = True
            self._check_open()
            yield server
        finally:
            self.close_authority()
            with anyio.CancelScope(shield=True):
                try:
                    async with asyncio.timeout(_CLEANUP_TIMEOUT_S):
                        await server.close()
                        await asyncio.gather(*self._late_cleanup, return_exceptions=True)
                except TimeoutError:
                    self.mark_cleanup_failed()
            self._session_active = False
            self.cleanup_complete = not self._cleanup_failed and not self._late_cleanup and server.closed

    async def start[T](self, operation: Coroutine[Any, Any, T],
                       cleanup: Callable[[T], Coroutine[Any, Any, None]]) -> T:
        """Keep process ownership when native startup returns after cancellation."""
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

    def track_cleanup(self, cleanup: Coroutine[Any, Any, Any]) -> None:
        """Own late transport cleanup without writing frozen evidence."""
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
