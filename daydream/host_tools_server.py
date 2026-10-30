"""Bounded authenticated loopback callbacks for one native Pi invocation.

The model sees only Pi's existing terminal tool. Its extension confirms the
active schema before provider work, charges actual native starts, then invokes
one synchronous host operation. This endpoint has no filesystem reader or MCP
server, and never schedules domain installation in a background thread.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import Any

from daydream import clock, host_tools as policy
from daydream.host_tools import HostToolBinding, HostToolFatal, HostToolRejected


def _bounded_items(value: Any, remaining: int, depth: int = 0) -> int:
    if depth > policy._MAX_DEPTH:
        raise ValueError(f"Host request nesting exceeds {policy._MAX_DEPTH} levels; flatten the submitted arguments")
    remaining -= 1
    if remaining < 0:
        raise ValueError(f"The entire host request exceeds {policy._MAX_ITEMS} items; submit fewer argument items")
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
    if isinstance(value, dict):
        for key in value:
            remaining = _bounded_items(key, remaining, depth + 1)
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    for child in children:
        remaining = _bounded_items(child, remaining, depth + 1)
    return remaining


class PiHostToolServer:
    """A closable callback endpoint owned by one execution, never a cached backend."""

    def __init__(self, binding: HostToolBinding) -> None:
        self.binding = binding
        self.ready = False
        self.closed = False
        self._closing = False
        self._secret = secrets.token_urlsafe(32)
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self.url = ""

    def packet(self) -> dict[str, Any]:
        """Private transport material, never persisted in request configuration."""
        return {
            "url": self.url, "authorization": "Bearer " + self._secret,
            "tool_name": policy.NATIVE_SUBMIT, "schema": self.binding.schema,
            "setup_timeout_ms": int(policy._SETUP_TIMEOUT_S * 1000),
            "request_timeout_ms": max(1, int(policy._CALLBACK_TIMEOUT_S * 1000)),
            "max_frame_bytes": policy._MAX_FRAME_BYTES,
            "max_argument_bytes": policy._MAX_ARGUMENT_BYTES,
            **({"allowed_tools": sorted(self.binding.allowed_tools)} if self.binding.allowed_tools is not None else {}),
        }

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._connected, "127.0.0.1", 0,
                                                 limit=policy._MAX_FRAME_BYTES + 1)
        socket = self._server.sockets[0]
        self.url = f"http://127.0.0.1:{socket.getsockname()[1]}"

    def _connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(self._tasks) >= 3 or self._closing:
            writer.close()
            return
        # These tasks own transport I/O only. The submit dispatch remains fully
        # synchronous on this loop, so cancellation cannot leave a late writer.
        task = asyncio.create_task(self._request(reader, writer))
        self._tasks.add(task)
        self._writers.add(writer)

        def finished(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            if not self._closing and not completed.cancelled() and completed.exception() is not None:
                error = HostToolFatal("Native host callback transport failed unexpectedly", code="callback_failed",
                                      preacceptance=not self.binding.possibly_accepted)
                error.__cause__ = completed.exception()
                self._fail(error)

        task.add_done_callback(finished)

    async def _reply(self, writer: asyncio.StreamWriter, status: int, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > policy._MAX_FRAME_BYTES:
            status = 413
            data = b'{"error":"Host response frame byte limit exceeded"}'
        reasons = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
                   405: "Method Not Allowed", 413: "Content Too Large", 431: "Request Header Fields Too Large"}
        writer.write((f"HTTP/1.1 {status} {reasons.get(status, 'Error')}\r\n"
                      "Content-Type: application/json\r\nConnection: close\r\n"
                      f"Content-Length: {len(data)}\r\n\r\n").encode() + data)
        await writer.drain()

    def _error(self, error: HostToolRejected | HostToolFatal, call_id: Any = None) -> dict[str, Any]:
        response: dict[str, Any] = {
            "outcome": error.outcome if isinstance(error, HostToolFatal) else "correctable",
            "code": error.code, "error": str(error),
        }
        if not isinstance(error, HostToolFatal) or error.preacceptance:
            response.update(allowed=False, accepted=False)
        if policy._valid_call_id(call_id):
            response["id"] = call_id
        return response

    def _fatal(self, message: str, *, code: policy.HostToolFailureCode) -> HostToolFatal:
        return self._fail(HostToolFatal(message, code=code, preacceptance=not self.binding.possibly_accepted))

    def _fail(self, error: HostToolFatal) -> HostToolFatal:
        # A retired endpoint cannot revoke a newer attempt on the same binding.
        if self._closing or not self.binding.authority_open:
            return self.binding.fatal_error or error
        return self.binding.fail(error)

    async def _request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        admitting = complete = False
        authenticated = False
        submitting = False
        submission_id: Any = None
        try:
            async with asyncio.timeout(policy._CALLBACK_TIMEOUT_S) as request_timeout:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except asyncio.LimitOverrunError:
                    await self._reply(writer, 431, {"error": "Host header byte limit exceeded"})
                    return
                if len(head) > policy._MAX_FRAME_BYTES:
                    await self._reply(writer, 431, {"error": "Host header byte limit exceeded"})
                    return
                lines = head.decode("ascii", errors="strict").split("\r\n")
                method, path, protocol = lines[0].split(" ")
                if protocol != "HTTP/1.1":
                    raise ValueError("unsupported protocol")
                headers: dict[str, str] = {}
                for line in lines[1:-2]:
                    key, value = line.split(":", 1)
                    key = key.lower()
                    if key in headers:
                        raise ValueError("duplicate header")
                    headers[key] = value.strip()
                if not secrets.compare_digest(headers.get("authorization", ""), "Bearer " + self._secret):
                    await self._reply(writer, 403, {"error": "Unauthorized transport"})
                    return
                authenticated = True
                admitting = path in {"/admit", "/submit"}
                submitting = path == "/submit"
                if "transfer-encoding" in headers:
                    raise ValueError("chunked requests are unsupported")
                length = headers.get("content-length", "0")
                if len(length) > 20 or not length.isdigit() or int(length) > policy._MAX_FRAME_BYTES:
                    await self._reply(writer, 413, self._error(self._fatal(
                        f"Host request exceeds the {policy._MAX_FRAME_BYTES} whole-frame byte limit",
                        code="invalid_frame")))
                    return
                body = await reader.readexactly(int(length))
                if self._closing or (not self.binding.authority_open and not self.binding.accepted
                                     and self.binding.fatal_error is None):
                    return
                value = json.loads(body) if body else {}
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                submission_id = value.get("id")
                if path in {"/admit", "/submit"}:
                    if path == "/admit":
                        self.binding.ensure_open()
                    if not self.ready:
                        raise self._fatal("Native tool readiness was not established before admission/submission",
                                          code="native_readiness")
                try:
                    _bounded_items(value, policy._MAX_ITEMS)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    if path == "/admit":
                        message = ("Host arguments contain invalid UTF-8 text; replace invalid characters"
                                   if isinstance(exc, UnicodeError) else str(exc))
                        self.binding.admit(value.get("id"), value.get("name"), None, malformed=message)
                    raise
                if method == "GET" and path == "/status":
                    self.binding.ensure_open()
                    # Node timers use signed 32-bit milliseconds. This is a
                    # projection of the original deadline, never a new allowance.
                    remaining_ms = min(2_147_483_647,
                                       max(0, int((self.binding.deadline - clock.monotonic()) * 1000)))
                    await self._reply(writer, 200, {"ready": self.ready, "remaining_ms": remaining_ms})
                    return
                if method != "POST":
                    await self._reply(writer, 405, self._error(self._fatal(
                        "Native host endpoint requires POST", code="invalid_frame"), submission_id))
                    return
                if path == "/ready":
                    self.binding.ensure_open()
                    if (set(value) != {"tool_name", "schema"} or self.binding.schema is None
                            or value["tool_name"] != policy.NATIVE_SUBMIT or value["schema"] != self.binding.schema):
                        raise self._fatal("Native terminal schema differs from the bound schema",
                                          code="native_readiness")
                    self.ready = True
                    await self._reply(writer, 200, {"ready": True})
                    return
                if path == "/admit":
                    oversized = (f"The entire structured_output argument object exceeds {policy._MAX_ARGUMENT_BYTES} "
                                 "UTF-8 bytes; reduce the submitted arguments before retrying"
                                 if value.get("arguments_oversized") is True else False)
                    self.binding.admit(value.get("id"), value.get("name"), value.get("arguments"),
                                       terminal=value.get("terminal") is True, malformed=oversized)
                    complete = True
                    await self._reply(writer, 200, {"id": submission_id, "allowed": True})
                    return
                if path == "/submit":
                    if not self.binding.accepted:
                        # Frame/control I/O stays short; synchronous source
                        # validation and admission use the assignment's deadline.
                        request_timeout.reschedule(asyncio.get_running_loop().time()
                            + max(0.0, self.binding.deadline - clock.monotonic()))
                    result = self.binding.submit(value.get("id"), value.get("arguments"))
                    complete = True
                    await self._reply(writer, 200, {"id": submission_id, "accepted": True, "result": result})
                    return
                await self._reply(writer, 404, self._error(self._fatal(
                    "Unknown native host endpoint", code="invalid_frame"), submission_id))
        except (HostToolRejected, HostToolFatal) as exc:
            complete = True
            if isinstance(exc, HostToolRejected) and submitting and self.binding.possibly_accepted:
                # A trusted callback failure after its mutation boundary cannot
                # be presented as a correctable semantic rejection.
                exc = self._fatal("Host acceptance outcome is unconfirmed", code="callback_failed")
            await self._reply(writer, 200, self._error(exc, submission_id))
        except (ValueError, UnicodeError, RecursionError, asyncio.IncompleteReadError) as exc:
            if authenticated:
                failure = HostToolFatal("Invalid bounded native host request cannot be correlated safely",
                                        code="invalid_frame", preacceptance=not self.binding.possibly_accepted)
                failure.__cause__ = exc
                error = self._fail(failure)
                await self._reply(writer, 400, self._error(error, submission_id))
            else:
                await self._reply(writer, 400, {"error": "Invalid bounded host frame"})
        except (TimeoutError, OSError):
            pass  # A timed-out/disconnected transport cannot establish success.
        except Exception as exc:
            failure = HostToolFatal("Native host callback failed unexpectedly", code="callback_failed",
                                    preacceptance=not self.binding.possibly_accepted)
            failure.__cause__ = exc
            error = self._fail(failure)
            # Unexpected callback failure may have crossed the acceptance boundary;
            # keep its native result generic and prohibit replay via the binding.
            try:
                await self._reply(writer, 400, self._error(error, submission_id))
            except OSError:
                pass
        finally:
            if admitting and not complete and not self._closing:
                self.binding.close_attempt()
            self._writers.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def close(self) -> None:
        if not self._closing:
            self.binding.close_attempt()
        self._closing = True
        if self._server is not None:
            self._server.close()
            # Python's server wait_closed also waits for active clients. End
            # their I/O before awaiting it, including connections whose stream
            # callback has not run yet; the closing gate prevents new handlers.
            self._server.close_clients()
        writers = tuple(self._writers)
        for writer in writers:
            writer.close()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.difference_update(tasks)
        if writers:
            await asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True)
        self._writers.clear()
        if self._server is not None:
            await self._server.wait_closed()
        self.closed = not self._tasks
