"""Invocation-local native Pi MCP transport using the official SDK's FastMCP.

The outer ASGI admission bounds bytes before SDK JSON parsing. Control requests
cannot establish readiness: only successful native initialize/tools-list traffic
at this private endpoint can. No callbacks are queued beyond the admitted HTTP
and read concurrency bounds.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, ToolAnnotations
from starlette.exceptions import HTTPException
from starlette.types import Message, Receive, Scope, Send

from daydream import clock, host_tools as policy
from daydream.host_tools import HostToolBinding, HostToolRejected
from daydream.json_utils import validates_schema


class _ReadServer(FastMCP[None]):
    async def list_tools(self) -> list[Tool]:
        # Stateful SDK dispatch admits this only on the initialized session.
        if self.get_context().session.client_params is None:
            raise ValueError("Native MCP session is not initialized")
        return [Tool(name="read_file", description=policy._READ_DESCRIPTION, inputSchema=policy._read_schema(),
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
        self.listed = False
        self._secret = secrets.token_urlsafe(32)
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._requests = 0
        self._active = 0
        self.url = ""
        mcp = _ReadServer("daydream-host", stateless_http=False, json_response=True)

        async def read_file(path: str) -> str:
            return await binding.read_file({"path": path})

        mcp.add_tool(read_file, name="read_file", description=policy._READ_DESCRIPTION, structured_output=False)
        self.app = mcp.streamable_http_app()

    def packet(self) -> dict[str, Any]:
        """Private transport material, never part of recorded request configuration."""
        return {
            "url": self.url, "authorization": "Bearer " + self._secret,
            "server_name": "daydream_host", "tool_name": policy.NATIVE_READ_FILE,
            "schema": policy._read_schema(), "setup_timeout_ms": int(policy._SETUP_TIMEOUT_S * 1000),
            "request_timeout_ms": max(1, int(policy._CALLBACK_TIMEOUT_S * 1000)),
            "max_frame_bytes": policy._MAX_FRAME_BYTES,
        }

    async def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket = sock
        sock.bind(("127.0.0.1", 0))
        sock.listen(5)
        sock.setblocking(False)
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self._server = _LocalUvicorn(uvicorn.Config(
            self, log_level="critical", access_log=False, lifespan="on", ws="none",
            limit_concurrency=5,
            timeout_keep_alive=1, timeout_graceful_shutdown=int(policy._CLEANUP_TIMEOUT_S) + 1,
            h11_max_incomplete_event_size=policy._MAX_FRAME_BYTES,
        ))
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
        async with asyncio.timeout(policy._SETUP_TIMEOUT_S):
            while not self._server.started:
                if self._task.done():
                    await self._task
                    raise RuntimeError("Host MCP endpoint failed to start")
                await asyncio.sleep(0.01)

    async def _reply(self, send: Send, status: int, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        limit = min(policy._MAX_FRAME_BYTES, policy._MAX_RESULT_BYTES)
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
        admitting = authorized and scope["path"] == "/admit"
        complete = active = False
        try:
            if sum(len(key) + len(value) for key, value in scope["headers"]) > policy._MAX_FRAME_BYTES:
                raise HTTPException(431, "Host header byte limit exceeded")
            if not authorized:
                raise HTTPException(403, "Unauthorized transport")
            self._requests += 1
            if self._requests > policy._MAX_HTTP_REQUESTS or self._active >= 3:
                raise HTTPException(429, "Host transport capacity exhausted")
            self._active += 1
            active = True
            async with asyncio.timeout(policy._CALLBACK_TIMEOUT_S):
                if scope["method"] == "GET" and scope["path"] == "/mcp":
                    raise HTTPException(405, "Standalone SSE unsupported")
                if scope["method"] == "GET" and scope["path"] == "/status":
                    await self._reply(send, 200, {"ready": self.listed and self.binding.authority_open
                                                 and clock.monotonic() < self.binding.deadline})
                    return
                body = bytearray()
                length = headers.get(b"content-length")
                if length is not None and (len(length) > 20 or not length.isdigit()
                                           or int(length) > policy._MAX_FRAME_BYTES):
                    raise HTTPException(413, "Host frame byte limit exceeded")
                while True:
                    chunk = await receive()
                    if chunk["type"] == "http.disconnect":
                        return
                    data = chunk.get("body", b"")
                    if len(body) + len(data) > policy._MAX_FRAME_BYTES:
                        raise HTTPException(413, "Host frame byte limit exceeded")
                    body.extend(data)
                    if not chunk.get("more_body", False):
                        break
                value = json.loads(body) if body else {}
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                if admitting and scope["method"] == "POST":
                    response: dict[str, Any]
                    try:
                        try:
                            _bounded_items(value, policy._MAX_ITEMS)
                        except (ValueError, UnicodeError, RecursionError):
                            self.binding.admit(value.get("id"), value.get("name"), None, malformed=True)
                        self.binding.admit(value.get("id"), value.get("name"), value.get("arguments"))
                        response = {"allowed": True}
                    except HostToolRejected as exc:
                        response = {"allowed": False, "error": str(exc)}
                    complete = True
                    await self._reply(send, 200, response)
                    return
                _bounded_items(value, policy._MAX_ITEMS)
                if scope["path"] != "/mcp":
                    raise HTTPException(404, "Unknown host endpoint")
                request_id = value.get("id")
                if request_id is not None and (
                    (type(request_id) is not str and type(request_id) is not int)
                    or (type(request_id) is str and len(request_id.encode()) > 512)
                    or (type(request_id) is int and abs(request_id) > 2 ** 53 - 1)
                ):
                    raise HTTPException(400, "Invalid bounded MCP request identity")
                method = value.get("method")
                if method == "tools/call":
                    params = value.get("params")
                    arguments = params.get("arguments") if isinstance(params, dict) else None
                    # Reject original JSON shape before the SDK can coerce it.
                    if not validates_schema(arguments, policy._read_schema()):
                        await self._reply(send, 200, {"jsonrpc": "2.0", "id": request_id, "result": {
                            "isError": True, "content": [{"type": "text", "text": "Invalid host read schema"}]}})
                        return
                replayed = False

                async def replay() -> Message:
                    nonlocal replayed
                    if replayed:
                        return await receive()
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}

                messages: list[Any] = []
                response_bytes = 0
                response_limit = policy._MAX_RESULT_BYTES if method == "tools/call" else policy._MAX_FRAME_BYTES

                async def observe(message: Any) -> None:
                    nonlocal response_bytes
                    response_bytes += len(message.get("body", b""))
                    if response_bytes > response_limit or len(messages) >= 8:
                        raise HTTPException(500, "Host response frame limit exceeded")
                    messages.append(message)

                await self.app(scope, replay, observe)
                if scope["method"] == "POST" and method == "tools/list" and request_id is not None:
                    response = json.loads(b"".join(message.get("body", b"") for message in messages))
                    if isinstance(response.get("result", {}).get("tools"), list):
                        self.listed = True
                for message in messages:
                    await send(message)
        except HTTPException as exc:
            await self._reply(send, exc.status_code, {"error": exc.detail})
        except (ValueError, UnicodeError, RecursionError):
            await self._reply(send, 400, {"error": "Invalid bounded host frame"})
        except TimeoutError:
            pass  # Do not replace a response whose transmission already started.
        finally:
            if admitting and not complete:
                self.binding.close_authority()
            if active:
                self._active -= 1

    async def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        try:
            if self._task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(self._task), policy._CLEANUP_TIMEOUT_S)
                except TimeoutError:
                    self._task.cancel()
                    await asyncio.gather(self._task, return_exceptions=True)
                    self.binding.mark_cleanup_failed()
        finally:
            if self._socket is not None:
                self._socket.close()
