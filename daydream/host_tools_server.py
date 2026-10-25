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
from daydream.host_tools import HostToolBinding, HostToolRejected


def _bounded_items(value: Any, remaining: int, depth: int = 0) -> int:
    if depth > policy._MAX_DEPTH:
        raise ValueError("item depth exceeded")
    remaining -= 1
    if remaining < 0:
        raise ValueError("item limit exceeded")
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
            if not completed.cancelled() and completed.exception() is not None:
                self.binding.close_authority()

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

    async def _request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        admitting = complete = False
        submitting = False
        submission_id: Any = None
        try:
            async with asyncio.timeout(policy._CALLBACK_TIMEOUT_S):
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
                admitting = path in {"/admit", "/submit"}
                submitting = path == "/submit"
                if "transfer-encoding" in headers:
                    raise ValueError("chunked requests are unsupported")
                length = headers.get("content-length", "0")
                if len(length) > 20 or not length.isdigit() or int(length) > policy._MAX_FRAME_BYTES:
                    await self._reply(writer, 413, {"error": "Host frame byte limit exceeded"})
                    return
                body = await reader.readexactly(int(length))
                value = json.loads(body) if body else {}
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                submission_id = value.get("id")
                try:
                    _bounded_items(value, policy._MAX_ITEMS)
                except (ValueError, UnicodeError, RecursionError):
                    if path == "/admit":
                        self.binding.admit(value.get("id"), value.get("name"), None, malformed=True)
                    raise
                if method == "GET" and path == "/status":
                    await self._reply(writer, 200, {"ready": self.ready and self.binding.authority_open
                                                 and clock.monotonic() < self.binding.deadline})
                    return
                if method != "POST":
                    await self._reply(writer, 405, {"error": "Expected POST"})
                    return
                if path == "/ready":
                    self.binding.ensure_open()
                    if (set(value) != {"tool_name", "schema"} or self.binding.schema is None
                            or value["tool_name"] != policy.NATIVE_SUBMIT or value["schema"] != self.binding.schema):
                        raise ValueError("Native terminal schema differs from the bound schema")
                    self.ready = True
                    await self._reply(writer, 200, {"ready": True})
                    return
                if path == "/admit":
                    self.binding.admit(value.get("id"), value.get("name"), value.get("arguments"),
                                       terminal=value.get("terminal") is True)
                    complete = True
                    await self._reply(writer, 200, {"allowed": True})
                    return
                if path == "/submit":
                    if not self.ready:
                        raise HostToolRejected("Native terminal readiness was not established")
                    result = self.binding.submit(value.get("id"), value.get("arguments"))
                    complete = True
                    await self._reply(writer, 200, {"id": submission_id, "accepted": True, "result": result})
                    return
                await self._reply(writer, 404, {"error": "Unknown host endpoint"})
        except HostToolRejected as exc:
            complete = True
            if submitting and self.binding.possibly_accepted:
                # A trusted callback failure after its mutation boundary cannot
                # be presented as a correctable semantic rejection.
                await self._reply(writer, 400, {"id": submission_id, "error": "Host acceptance outcome is unconfirmed"})
                return
            response: dict[str, Any] = {"allowed": False, "accepted": False, "error": str(exc)}
            if submitting:
                response["id"] = submission_id
            await self._reply(writer, 200, response)
        except (ValueError, UnicodeError, RecursionError, asyncio.IncompleteReadError):
            await self._reply(writer, 400, {"error": "Invalid bounded host frame"})
        except (TimeoutError, OSError):
            pass  # A timed-out/disconnected transport cannot establish success.
        except Exception:
            self.binding.close_authority()
            # Unexpected callback failure may have crossed the acceptance boundary;
            # keep its native result generic and prohibit replay via the binding.
            try:
                await self._reply(writer, 400, {"error": "Host callback failed"})
            except OSError:
                pass
        finally:
            if admitting and not complete:
                self.binding.close_authority()
            self._writers.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def close(self) -> None:
        self._closing = True
        self.binding.close_authority()
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
