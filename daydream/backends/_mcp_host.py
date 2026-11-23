"""Loopback MCP host-tool endpoint shared by the CLI adapters that register an MCP server.

Codex and Antigravity each serve one attempt's host tools from this authenticated
loopback server. It is not an adapter: ``uvicorn`` and ``mcp`` load only inside
``start()``, for an invocation that carries host tools.
"""

from __future__ import annotations

import asyncio
import secrets
import socket
from contextlib import suppress
from typing import Any

from daydream.host_tools import (
    NATIVE_SUBMIT,
    HostToolBinding,
    HostToolCallBudgetExceeded,
    HostToolEndpoint,
    HostToolFatal,
    HostToolRejected,
)

HOST_SERVER = "daydream"


def exempt_loopback_from_proxy(environment: dict[str, str]) -> None:
    """Keep the loopback callback and its secret away from any proxy, in place.

    Each name keeps the user's exemptions, from whichever name carries them: clients read either first.
    """
    exemptions = environment.get("NO_PROXY"), environment.get("no_proxy")
    for name, own, other in zip(("NO_PROXY", "no_proxy"), exemptions, reversed(exemptions)):
        environment[name] = ",".join(filter(None, (own or other, "127.0.0.1")))


class McpHostEndpoint(HostToolEndpoint):
    """One attempt's loopback MCP server for a binding, served on the host event loop.

    The native CLI (Codex or Antigravity) offers no host admission before its own tools
    run, so the MCP handler is both admission and service for a host call: it charges the
    start under a host-assigned identity, then serves it without an await. Rejections and
    host failures become tool errors the model reads. ``ready`` means this exact server
    answered ``tools/list``.
    """

    def __init__(self, binding: HostToolBinding, *, native_owner: str) -> None:
        super().__init__(binding)
        self.token = secrets.token_urlsafe(32)
        self.url = ""
        self._native_owner = native_owner
        self._calls = 0
        self._stop = asyncio.Event()
        self._serving: asyncio.Task[None] | None = None

    async def start(self) -> None:
        # Loaded only by an invocation that carries host tools.
        import uvicorn
        from mcp import types
        from mcp.server.lowlevel import Server
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        tools = [types.Tool(name=name, description=self.description(name), input_schema=schema)
                 for name, schema in self.schemas().items()]

        async def list_tools(
            _ctx: Any, _params: types.PaginatedRequestParams | None,
        ) -> types.ListToolsResult:
            self.ready = True
            return types.ListToolsResult(tools=tools)

        async def call_tool(
            _ctx: Any, params: types.CallToolRequestParams,
        ) -> types.CallToolResult:
            name = params.name
            failed, blocks = await self.admitted(name, lambda: self._call(name, params.arguments or {}))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text) for text in blocks], is_error=failed)

        server: Server[Any] = Server(HOST_SERVER, on_list_tools=list_tools, on_call_tool=call_tool)
        # Opt out of MCP's default global-provider OTel middleware; Daydream owns per-run tracing.
        server.middleware.clear()
        # Disable MCP's 30-minute idle timeout; the binding deadline and manager.run() own session lifetime.
        manager = StreamableHTTPSessionManager(app=server, json_response=True, session_idle_timeout=None)
        authorization = b"Bearer " + self.token.encode()

        async def app(scope: Any, receive: Any, send: Any) -> None:
            authorized = secrets.compare_digest(dict(scope["headers"]).get(b"authorization", b""), authorization)
            if authorized and scope["method"] == "POST":
                await manager.handle_request(scope, receive, send)
                return
            # Only this attempt's JSON requests are served; the server never opens a stream of its own.
            await send({"type": "http.response.start", "status": 405 if authorized else 403,
                        "headers": [(b"content-length", b"0")]})
            await send({"type": "http.response.body", "body": b""})

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{listener.getsockname()[1]}/mcp"
        # No logging configuration is installed: uvicorn registers its TRACE level name and logs
        # through the standard logging tree. startup()/shutdown() rather than serve(), which
        # replaces the process signal handlers.
        config = uvicorn.Config(app, lifespan="off", ws="none", log_config=None)
        config.load()
        http = uvicorn.Server(config)
        http.lifespan = config.lifespan_class(config)
        started: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def serve() -> None:
            # The session manager's task group lives in this task, never in a suspended event generator.
            async with manager.run():
                await http.startup(sockets=[listener])
                started.set_result(None)
                await self._stop.wait()
            # The manager has cancelled its handlers, one of which may await a slow check; nothing is left to drain.
            await http.shutdown(sockets=[listener])

        self._serving = asyncio.create_task(serve())
        await asyncio.wait({self._serving, started}, return_when=asyncio.FIRST_COMPLETED)
        if not started.done():
            await self._serving  # raises the startup failure

    def _call(self, name: str, arguments: dict[str, Any]) -> tuple[bool, list[str]]:
        """Admit and serve one host call synchronously, returning its error flag and text blocks."""
        binding = self.binding
        self._calls += 1
        call_id = f"{self._native_owner.lower()}-mcp-{self._calls}"
        try:
            if not self.current():
                raise HostToolRejected("This attempt's host-tool authority is closed; the call was not served",
                                       code="authority_closed")
            if binding.accepted:
                # Acceptance ended the assignment: observe overshoot without charging or failing it.
                binding.observe_start(call_id, name)
                raise HostToolRejected("structured_output was accepted and this assignment has ended; "
                                       "the call was not served", code="authority_closed")
            binding.admit(call_id, name, arguments, terminal=name == NATIVE_SUBMIT)
            if name != NATIVE_SUBMIT:
                return False, self.result_blocks(binding.call(call_id, name, arguments))
            result = binding.submit(call_id, arguments)
            # Acceptance closes authority at once: later starts become observed overshoot.
            binding.close_authority()
            return False, self.result_blocks(dict(result))
        except (HostToolRejected, HostToolFatal) as exc:
            return True, [str(exc)]

    def observe_native_start(self, item_id: str, name: str) -> None:
        """Count once a start the native CLI ran itself; it cannot be denied here before it runs.

        Acceptance closes host authority, not the CLI's own tools, so the ledger keeps
        charging them: one past the allowance fails the binding, accepted or not.
        """
        binding = self.binding
        if not binding.accepted:
            with suppress(HostToolFatal):  # retained; the event loop stops the invocation
                binding.observe_start(item_id, name)
        elif not binding.budget.charge():
            binding.fail(HostToolCallBudgetExceeded(
                f"Tool-call allocation exhausted by {self._native_owner}-owned tools after acceptance",
                preacceptance=False))

    async def close(self) -> None:
        await super().close()
        self._stop.set()
        if self._serving is not None:
            self.closed = False
            with suppress(Exception):  # a server that failed to start already failed start()
                await self._serving
                self.closed = True
