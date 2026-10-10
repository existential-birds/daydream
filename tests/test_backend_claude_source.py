"""Claude's owned MCP source transport retains frozen authority and invocation lifetime."""
from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk.types import AssistantMessage, ResultMessage, ToolResultBlock, ToolUseBlock, UserMessage
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from daydream.backends import RequestEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.claude import ClaudeBackend
from daydream.git_ops.source import frozen_source
from daydream.review_source import SourceRecipe, SourceWindow
from tests.harness.git_helpers import git, seed_feature_branch


@pytest.fixture
def frozen_recipe(tmp_path: Path) -> SourceRecipe:
    repo = tmp_path / 'claude_source'
    seed_feature_branch(repo, base={'retired.py': 'def greeting():\n    return "世界"\n', 'keep.py': 'VALUE = 1\n'},
                        feature={'keep.py': 'VALUE = 2\n'})
    git(repo, 'rm', 'retired.py')
    git(repo, 'commit', '-m', 'Delete obsolete source')
    revision = git(repo, 'rev-parse', 'main')
    oid, raw = frozen_source(repo, revision, 'retired.py')
    return SourceRecipe((SourceWindow(
        target_ids=('retired.py',), file='retired.py', source_path='retired.py', side='before', revision=revision,
        content_sha256=hashlib.sha256(raw).hexdigest(), blob_oid=oid, start_line=1, end_line=2,
        start_byte=0, end_byte=len(raw), body=raw.decode(),
    ),), repo)


@pytest.mark.parametrize('finish', ['complete', 'close'])
async def test_claude_owned_source_reads_frozen_deleted_source_and_closes_after_provider(
    frozen_recipe: SourceRecipe, monkeypatch: pytest.MonkeyPatch, finish: str,
) -> None:
    owned_name = 'mcp__daydream_source__read_source'
    arguments = {'target_id': 'retired.py', 'side': 'before'}
    observed: dict[str, Any] = {}

    class SourceClient:
        def __init__(self, options: ClaudeAgentOptions) -> None:
            self.options = options

        async def __aenter__(self) -> SourceClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            # The invocation server must outlive SDK teardown.
            async with httpx.AsyncClient() as client:
                response = await client.post(observed['url'])
            assert response.status_code != 404
            observed['provider_closed'] = True

        async def interrupt(self) -> None:
            observed['interrupted'] = True

        async def query(self, prompt: str) -> None:
            servers = self.options.mcp_servers
            assert isinstance(servers, dict)
            source = servers['daydream_source']
            assert source['type'] == 'http'
            assert owned_name in self.options.allowed_tools
            observed['url'] = source['url']
            hooks: Any = self.options.hooks
            assert hooks is not None
            for tool, allowed in ((owned_name, True), ('mcp__foreign__read_source', False)):
                decisions = [await hook({'tool_name': tool, 'tool_input': arguments}, 'owned-1', {})
                             for matcher in hooks['PreToolUse'] for hook in matcher.hooks]
                assert any(decision.get('hookSpecificOutput', {}).get('permissionDecision') == 'deny'
                           for decision in decisions) is (not allowed)
            async with httpx.AsyncClient(headers=source['headers']) as client:
                async with streamable_http_client(source['url'], http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        assert [tool.name for tool in tools.tools] == ['read_source']
                        result = await session.call_tool('read_source', arguments)
            assert not result.isError
            observed['content'] = [block.model_dump(exclude_none=True) for block in result.content]

        async def receive_response(self) -> AsyncIterator[Any]:
            # Even if an SDK stream contains a foreign MCP result, its content
            # keeps the normal codec and never inherits the owned reader's name.
            yield AssistantMessage(content=[ToolUseBlock(
                id='foreign-1', name='mcp__foreign__read_source', input=arguments)], model='fixture')
            yield UserMessage(content=[ToolResultBlock(tool_use_id='foreign-1', content=observed['content'])])
            yield AssistantMessage(content=[ToolUseBlock(id='owned-1', name=owned_name, input=arguments)],
                                   model='fixture')
            yield UserMessage(content=[ToolResultBlock(tool_use_id='owned-1', content=observed['content'])])
            yield ResultMessage(subtype='success', duration_ms=1, duration_api_ms=1, is_error=False,
                                num_turns=1, session_id='source-session')

    monkeypatch.setattr('daydream.backends.claude.ClaudeSDKClient', SourceClient)
    backend = ClaudeBackend('fixture')
    stream = backend.execute(frozen_recipe.repo, 'Inspect deleted source.', read_only=True, source_recipe=frozen_recipe)
    events = []
    async for event in stream:
        events.append(event)
        if finish == 'close' and isinstance(event, ToolResultEvent) and event.id == 'owned-1':
            await stream.aclose()
            break
    request, = [event for event in events if isinstance(event, RequestEvent)]
    assert request.config is not None and request.config.source_tool_enabled is True
    start, = [event for event in events if isinstance(event, ToolStartEvent) and event.id == 'owned-1']
    result, = [event for event in events if isinstance(event, ToolResultEvent) and event.id == 'owned-1']
    foreign_start, = [event for event in events if isinstance(event, ToolStartEvent) and event.id == 'foreign-1']
    foreign_result, = [event for event in events if isinstance(event, ToolResultEvent) and event.id == 'foreign-1']
    assert foreign_start.name == 'mcp__foreign__read_source'
    assert json.loads(foreign_result.output) == observed['content']
    assert (start.id, start.name, start.input) == ('owned-1', 'read_source', arguments)
    window, = frozen_recipe.windows
    assert json.loads(result.output) == {'source': window.metadata(), 'body': 'def greeting():\n    return "世界"\n'}
    assert result.id == start.id and not result.is_error
    assert not (frozen_recipe.repo / 'retired.py').exists()
    assert observed['provider_closed']
    assert observed.get('interrupted', False) is (finish == 'close')
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError):
            await client.post(observed['url'])


@pytest.mark.parametrize('mode', ['no-recipe', 'finalization', 'audit'])
async def test_claude_inactive_source_tool_stays_blocked_and_keeps_normal_sdk_content(
    frozen_recipe: SourceRecipe, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    owned_name = 'mcp__daydream_source__read_source'
    content = [{'type': 'text', 'text': 'Unowned source-shaped output'}]

    class RestrictedClient:
        def __init__(self, options: ClaudeAgentOptions) -> None:
            self.options = options

        async def __aenter__(self) -> RestrictedClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def query(self, prompt: str) -> None:
            assert self.options.mcp_servers == {}
            assert owned_name not in self.options.allowed_tools
            hooks: Any = self.options.hooks
            assert hooks is not None
            decisions = [await hook({'tool_name': owned_name, 'tool_input': {}}, 'unowned-1', {})
                         for matcher in hooks['PreToolUse'] for hook in matcher.hooks]
            assert any(decision.get('hookSpecificOutput', {}).get('permissionDecision') == 'deny'
                       for decision in decisions)

        async def receive_response(self) -> AsyncIterator[Any]:
            yield AssistantMessage(content=[ToolUseBlock(id='unowned-1', name=owned_name, input={})], model='fixture')
            yield UserMessage(content=[ToolResultBlock(tool_use_id='unowned-1', content=content, is_error=True)])
            yield ResultMessage(subtype='success', duration_ms=1, duration_api_ms=1, is_error=False,
                                num_turns=1, session_id='source-session')

    monkeypatch.setattr('daydream.backends.claude.ClaudeSDKClient', RestrictedClient)
    backend = ClaudeBackend('fixture', audit_root=frozen_recipe.repo if mode == 'audit' else None)
    events = [event async for event in backend.execute(
        frozen_recipe.repo, 'Serialize only.' if mode == 'finalization' else 'Inspect the audit root.',
        read_only=True, finalization=mode == 'finalization',
        source_recipe=None if mode == 'no-recipe' else frozen_recipe,
    )]
    request, = [event for event in events if isinstance(event, RequestEvent)]
    assert request.config is not None and request.config.source_tool_enabled is False
    start, = [event for event in events if isinstance(event, ToolStartEvent)]
    result, = [event for event in events if isinstance(event, ToolResultEvent)]
    assert start.name == owned_name and result.id == start.id and result.is_error
    assert json.loads(result.output) == content
