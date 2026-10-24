"""Invocation-local native Pi MCP transport using the official SDK's FastMCP.

The outer ASGI admission bounds bytes before SDK JSON parsing. Control requests
cannot establish readiness: only successful native initialize/tools-list traffic
at this private endpoint can. No callbacks are queued beyond the admitted HTTP
and read concurrency bounds.
"""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import anyio
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import InitializeResult, ListToolsResult, Tool, ToolAnnotations
from starlette.types import Message, Receive, Scope, Send

from daydream import clock
from daydream.host_tools import READ_FILE, HostToolBinding, HostToolRejected


class _ReadServer(FastMCP[None]):
    async def list_tools(self) -> list[Tool]:
        return [Tool(name=READ_FILE.name, description=READ_FILE.description, inputSchema=READ_FILE.schema(),
                     annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))]


class _LocalUvicorn(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        # The run host owns process signals; sibling endpoints must not change them.
        yield


def _bounded_items(value: Any, remaining: int, depth: int = 0) -> int:
    if depth > 16:
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
    """A closable server owned by one Pi execution, never a cached backend."""

    def __init__(self, binding: HostToolBinding) -> None:
        self.binding = binding
        self.initialized = False
        self.listed = False
        self._secret = secrets.token_urlsafe(32)
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._requests = 0
        self._active = 0
        self.url = ""
        self._closed = False
        mcp = _ReadServer("daydream-host", stateless_http=True, json_response=True)

        async def read_file(path: str) -> str:
            try:
                return await binding.read_file({"path": path})
            except HostToolRejected as exc:
                raise ToolError(str(exc)) from None

        mcp.add_tool(read_file, name=READ_FILE.name, description=READ_FILE.description, structured_output=False)
        self.app = mcp.streamable_http_app()

    def packet(self) -> dict[str, Any]:
        """Private transport material, never part of recorded request configuration."""
        return {
            "url": self.url, "authorization": "Bearer " + self._secret,
            "server_name": "daydream_host", "tool_name": "mcp__daydream_host__read_file",
            "schema": READ_FILE.schema(), "setup_timeout_ms": int(self.binding.limits.setup_timeout_s * 1000),
            "request_timeout_ms": max(1, int(self.binding.limits.callback_timeout_s * 1000)),
            "max_frame_bytes": self.binding.limits.max_frame_bytes,
        }

    async def start(self) -> None:
        self.binding.claim()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket = sock
        sock.bind(("127.0.0.1", 0))
        sock.listen(self.binding.limits.max_in_flight + 4)
        sock.setblocking(False)
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self._server = _LocalUvicorn(uvicorn.Config(
            self, log_level="critical", access_log=False, lifespan="on", ws="none",
            limit_concurrency=self.binding.limits.max_in_flight + 4,
            timeout_keep_alive=1, timeout_graceful_shutdown=math.ceil(self.binding.limits.cleanup_timeout_s),
            h11_max_incomplete_event_size=self.binding.limits.max_frame_bytes,
        ))
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
        async with asyncio.timeout(self.binding.limits.setup_timeout_s):
            while not self._server.started:
                if self._task.done():
                    await self._task
                    raise RuntimeError("Host MCP endpoint failed to start")
                await asyncio.sleep(0.01)

    async def _reply(self, send: Send, status: int, value: dict[str, Any], *, max_bytes: int | None = None) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        limit = min(self.binding.limits.max_frame_bytes, max_bytes) if max_bytes is not None else (
            self.binding.limits.max_frame_bytes)
        if len(data) > limit:
            # Reject the whole response, never shorten an authoritative identity.
            status = 413
            data = b'{"error":"Host response frame byte limit exceeded"}'
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
        await send({"type": "http.response.body", "body": data})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        authorized = headers.get(b"authorization", b"") == ("Bearer " + self._secret).encode()
        if sum(len(key) + len(value) for key, value in scope["headers"]) > self.binding.limits.max_frame_bytes:
            if authorized and scope["path"] == "/admit":
                self.binding.close_authority()
            await self._reply(send, 431, {"error": "Host header byte limit exceeded"})
            return
        if not authorized:
            await self._reply(send, 403, {"error": "Unauthorized transport"})
            return
        self._requests += 1
        if (self._requests > self.binding.limits.max_http_requests
                or self._active >= self.binding.limits.max_in_flight + 2):
            if scope["path"] == "/admit":
                self.binding.close_authority()
            await self._reply(send, 429, {"error": "Host transport capacity exhausted"})
            return
        self._active += 1
        try:
            async with asyncio.timeout(self.binding.limits.callback_timeout_s):
                if scope["method"] == "GET" and scope["path"] == "/status":
                    await self._reply(send, 200, {"ready": self.initialized and self.listed
                                                 and self.binding.authority_open
                                                 and clock.monotonic() < self.binding.deadline})
                    return
                body = bytearray()
                length = headers.get(b"content-length")
                if length is not None and (len(length) > 20 or not length.isdigit()
                                           or int(length) > self.binding.limits.max_frame_bytes):
                    if scope["path"] == "/admit":
                        self.binding.close_authority()
                    await self._reply(send, 413, {"error": "Host frame byte limit exceeded"})
                    return
                while True:
                    chunk = await receive()
                    if chunk["type"] == "http.disconnect":
                        if scope["path"] == "/admit":
                            self.binding.close_authority()
                        return
                    data = chunk.get("body", b"")
                    if len(body) + len(data) > self.binding.limits.max_frame_bytes:
                        if scope["path"] == "/admit":
                            self.binding.close_authority()
                        await self._reply(send, 413, {"error": "Host frame byte limit exceeded"})
                        return
                    body.extend(data)
                    if not chunk.get("more_body", False):
                        break
                try:
                    value = json.loads(body) if body else {}
                    if not isinstance(value, dict):
                        raise ValueError("expected object")
                except (ValueError, UnicodeError, RecursionError):
                    if scope["path"] == "/admit":
                        self.binding.close_authority()
                    await self._reply(send, 400, {"error": "Invalid bounded host frame"})
                    return
                if scope["path"] == "/admit" and scope["method"] == "POST":
                    response: dict[str, Any]
                    try:
                        try:
                            _bounded_items(value, self.binding.limits.max_items)
                        except (ValueError, UnicodeError, RecursionError):
                            self.binding.reject_admission(value.get("id"), value.get("name"))
                        self.binding.admit(value.get("id"), value.get("name"), value.get("arguments"))
                        response = {"allowed": True}
                    except HostToolRejected as exc:
                        response = {"allowed": False, "error": str(exc)}
                    await self._reply(send, 200, response)
                    return
                try:
                    _bounded_items(value, self.binding.limits.max_items)
                except (ValueError, UnicodeError, RecursionError):
                    await self._reply(send, 400, {"error": "Invalid bounded host frame"})
                    return
                if scope["path"] != "/mcp":
                    await self._reply(send, 404, {"error": "Unknown host endpoint"})
                    return
                request_id = value.get("id")
                if request_id is not None and (
                    (type(request_id) is not str and type(request_id) is not int)
                    or (type(request_id) is str and len(request_id.encode()) > 512)
                    or (type(request_id) is int and abs(request_id) > 2 ** 53 - 1)
                ):
                    await self._reply(send, 400, {"error": "Invalid bounded MCP request identity"})
                    return
                method = value.get("method")
                if method == "tools/call":
                    params = value.get("params")
                    arguments = params.get("arguments") if isinstance(params, dict) else None
                    # Validate the original JSON shape before FastMCP's coercion.
                    if (not isinstance(arguments, dict) or set(arguments) != {"path"}
                            or not isinstance(arguments["path"], str)
                            or not 1 <= len(arguments["path"]) <= 2048):
                        await self._reply(send, 200, {"jsonrpc": "2.0", "id": value.get("id"), "result": {
                            "isError": True, "content": [{"type": "text", "text": "Invalid host read schema"}]}},
                            max_bytes=self.binding.limits.max_result_bytes)
                        return
                replayed = False

                async def replay() -> Message:
                    nonlocal replayed
                    if replayed:
                        return await receive()
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}

                success = False
                response_messages: list[Any] = []
                response_bytes = 0
                response_limit = (min(self.binding.limits.max_frame_bytes, self.binding.limits.max_result_bytes)
                                  if method == "tools/call" else self.binding.limits.max_frame_bytes)

                async def observe(message: Any) -> None:
                    nonlocal success, response_bytes
                    if message["type"] == "http.response.start":
                        success = 200 <= message["status"] < 300
                    response_bytes += len(message.get("body", b""))
                    if response_bytes > response_limit or len(response_messages) >= 8:
                        raise ValueError("Host response frame limit exceeded")
                    response_messages.append(message)

                try:
                    await self.app(scope, replay, observe)
                except ValueError:
                    await self._reply(send, 500, {"error": "Host response frame limit exceeded"})
                    return
                for message in response_messages:
                    await send(message)
                if success and request_id is not None and method in {"initialize", "tools/list"}:
                    # HTTP 200 also carries JSON-RPC errors. Readiness belongs
                    # only to a successful, correlated SDK protocol result.
                    try:
                        result = json.loads(b"".join(message.get("body", b"") for message in response_messages))
                        _bounded_items(result, self.binding.limits.max_items)
                        if (not isinstance(result, dict) or result.get("jsonrpc") != "2.0"
                                or type(result.get("id")) is not type(request_id)
                                or result.get("id") != request_id or "error" in result):
                            return
                        if method == "initialize":
                            InitializeResult.model_validate(result.get("result"))
                            self.initialized = True
                        elif self.initialized:
                            tools = ListToolsResult.model_validate(result.get("result")).tools
                            self.listed = (len(tools) == 1 and tools[0].name == READ_FILE.name
                                           and tools[0].inputSchema == READ_FILE.schema())
                    except (ValueError, UnicodeError, RecursionError):
                        return
        except TimeoutError:
            if scope["path"] == "/admit":
                self.binding.close_authority()
            # Do not replace an already-started SDK response with another frame.
            return
        finally:
            self._active -= 1

    async def close(self) -> None:
        self.binding.close_authority()
        if self._closed:
            return
        self._closed = True
        with anyio.CancelScope(shield=True):
            try:
                await self.binding.close()
            finally:
                if self._server is not None:
                    self._server.should_exit = True
                if self._task is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(self._task), self.binding.limits.cleanup_timeout_s)
                    except TimeoutError:
                        self._task.cancel()
                        await asyncio.gather(self._task, return_exceptions=True)
                        self.binding.mark_cleanup_failed()
                if self._socket is not None:
                    self._socket.close()
