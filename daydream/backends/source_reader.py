"""Invocation-owned MCP transport for the shared frozen-source reader."""
from __future__ import annotations

import asyncio
import secrets
import socket
from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from types import TracebackType
from typing import Any
from uuid import uuid4

import anyio
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from daydream.review_source import SourceRecipe

SOURCE_TOOL_NAME = 'mcp__daydream_source__read_source'


def source_tool_output(content: Any) -> str:
    """Decode the owned tool's single text frame; malformed frames cannot ground source."""
    if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
            and content[0].get('type') == 'text' and isinstance(content[0].get('text'), str)):
        return str(content[0]['text'])
    return ''


class _SourceServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        # The runner owns process signals; an invocation must not replace them.
        yield


class SourceReader:
    """Serve only this invocation's frozen authority; join requests before teardown."""

    def __init__(self, recipe: SourceRecipe) -> None:
        self.recipe = recipe
        self.server_name = 'daydream_src_' + uuid4().hex
        self.token = secrets.token_urlsafe(32)
        self.url = ''
        self._stack = AsyncExitStack()
        self._server: _SourceServer | None = None
        self._task: asyncio.Task[None] | None = None
        self._validation: asyncio.Task[None] | None = None
        self._reads: set[asyncio.Task[str]] = set()
        server: Server[None] = Server('daydream_source')

        @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def list_tools() -> list[Tool]:
            return [Tool(name='read_source', description=(
                'Read one frozen catalog selector and before/after side. '
                'Tracked dependency paths also allow after reads.'),
                annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                            idempotentHint=True, openWorldHint=False), inputSchema={
                    'type': 'object', 'properties': {'target_id': {'type': 'string', 'minLength': 1},
                                                   'side': {'enum': ['before', 'after']}},
                    'required': ['target_id', 'side'], 'additionalProperties': False})]

        @server.call_tool()  # type: ignore[untyped-decorator]
        async def read_source(name: str, arguments: dict[str, Any]) -> CallToolResult:
            try:
                if name != 'read_source' or set(arguments) != {'target_id', 'side'}:
                    raise ValueError('Frozen source selector unavailable')
                read = asyncio.create_task(asyncio.to_thread(
                    recipe.read_source, arguments['target_id'], arguments['side']))
                self._reads.add(read)
                def finished(task: asyncio.Task[str]) -> None:
                    self._reads.discard(task)
                    if not task.cancelled():
                        task.exception()
                read.add_done_callback(finished)
                body = await asyncio.shield(read)
            except ValueError as exc:
                source_free = (set(arguments) == {'target_id', 'side'} and recipe.unavailable_lookup(
                    'read_source', arguments, recipe.repo, set()))
                return CallToolResult(content=[TextContent(type='text', text=str(exc))], isError=True,
                                      _meta={'source_free_disposition': 'zero_match'} if source_free else None)
            return CallToolResult(content=[TextContent(type='text', text=body)], isError=False)

        self._sessions = StreamableHTTPSessionManager(server, stateless=True, json_response=True)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = dict(scope.get('headers', []))
        authorization = headers.get(b'authorization', b'')
        if scope.get('path') != '/mcp':
            await Response(status_code=404)(scope, receive, send)
            return
        if not secrets.compare_digest(authorization, ('Bearer ' + self.token).encode()):
            await Response(status_code=401)(scope, receive, send)
            return
        await self._sessions.handle_request(scope, receive, send)

    async def __aenter__(self) -> SourceReader:
        try:
            self._validation = asyncio.create_task(asyncio.to_thread(self.recipe.revalidate))
            await asyncio.shield(self._validation)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._stack.callback(listener.close)
            listener.bind(('127.0.0.1', 0))
            self.url = f'http://127.0.0.1:{listener.getsockname()[1]}/mcp'
            config = uvicorn.Config(self, lifespan='off', access_log=False, log_level='error', ws='none')
            self._server = _SourceServer(config)
            async def serve() -> None:
                async with self._sessions.run():
                    assert self._server is not None
                    await self._server.serve(sockets=[listener])

            self._task = asyncio.create_task(serve())
            while not self._server.started:
                if self._task.done():
                    await self._task
                    raise RuntimeError('Frozen source reader failed to start')
                await asyncio.sleep(0)
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, traceback: TracebackType | None) -> None:
        if self._server is not None:
            self._server.should_exit = True
        # AnyIO wall scopes cancel repeatedly. Keep provider teardown, MCP
        # requests and their off-loop Git workers joined before archive cleanup.
        with anyio.CancelScope(shield=True):
            try:
                if self._task is not None:
                    try:
                        await asyncio.shield(self._task)
                    except asyncio.CancelledError:
                        await self._task
                        raise
            finally:
                try:
                    jobs = [*self._reads]
                    if self._validation is not None:
                        await asyncio.gather(self._validation, return_exceptions=True)
                    if jobs:
                        await asyncio.gather(*jobs, return_exceptions=True)
                finally:
                    await self._stack.aclose()
