"""Owned frozen source tools through Codex's real subprocess transport."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from daydream.backends import RequestEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.codex import CodexBackend
from daydream.git_ops.source import frozen_source
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.review_evidence import ReviewEvidence
from daydream.review_source import SourceRecipe, SourceWindow
from tests.harness.codex_replay import make_mock_process
from tests.harness.git_helpers import git, seed_feature_branch, tracked_source_state


@pytest.fixture
def frozen_recipe(tmp_path: Path) -> SourceRecipe:
    repo = tmp_path / 'source_repo'
    body = 'export const username = "世界";\n'
    seed_feature_branch(repo, base={'user.$name.tsx': body}, feature={'user.$name.tsx': body + '// changed\n'})
    revision = git(repo, 'rev-parse', 'HEAD')
    oid, raw = frozen_source(repo, revision, 'user.$name.tsx')
    return SourceRecipe((SourceWindow(
        ('user.$name.tsx',), 'user.$name.tsx', 'user.$name.tsx', 'after', revision,
        hashlib.sha256(raw).hexdigest(), oid, 1, 2, 0, len(raw), raw.decode(),
    ),), repo)


async def test_codex_owned_source_reader_preserves_frozen_bytes_and_disposable_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_recipe: SourceRecipe,
) -> None:
    recipe = frozen_recipe
    repo = recipe.repo
    original = tracked_source_state(repo)
    binary_dir = tmp_path / 'bin'
    binary_dir.mkdir()
    executable = binary_dir / 'codex'
    observed = tmp_path / 'observation.json'
    config_home = tmp_path / 'operator_codex_home'
    config_home.mkdir()
    config = config_home / 'config.toml'
    operator_config = ('[mcp_servers.daydream_source]\ncommand = "operator-source-driver"\n'
                       'disabled_tools = ["read_source"]\n')
    config.write_text(operator_config)
    monkeypatch.setenv('CODEX_HOME', str(config_home))
    executable.write_text(f'#!{sys.executable}\n' + '''import json, os, sys, tomllib, urllib.request
from pathlib import Path
args = sys.argv[1:]
prompt = sys.stdin.read()
settings = dict(arg.split('=', 1) for index, arg in enumerate(args) if index and args[index - 1] == '-c')
servers = tomllib.loads((Path(os.environ['CODEX_HOME']) / 'config.toml').read_text())['mcp_servers']
for key, value in settings.items():
    if key.startswith('mcp_servers.'):
        _, name, field = key.split('.')
        servers.setdefault(name, {})[field] = json.loads(value)
server_name, = [key[len('mcp_servers.'):-len('.url')] for key in settings
                if key.startswith('mcp_servers.') and key.endswith('.url')]
assert server_name != 'daydream_source', 'The owned reader collided with the configured operator MCP server'
assert 'command' not in servers[server_name], 'Inherited stdio configuration invalidates the HTTP source reader'
url = servers[server_name]['url']
token_name = servers[server_name]['bearer_token_env_var']
token = os.environ[token_name]
payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
           'params': {'name': 'read_source', 'arguments': {'target_id': 'user.$name.tsx', 'side': 'after'}}}
request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
    'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream'})
with urllib.request.urlopen(request, timeout=10) as response:
    result = json.load(response)['result']
source = json.loads(result['content'][0]['text'])
original = ''' + repr(str(repo)) + '''
Path(''' + repr(str(observed)) + ''').write_text(json.dumps({
    'cwd': args[args.index('--cd') + 1], 'body': source['body'],
    'source_path_exposed': original in prompt or any(original in arg for arg in args)
                           or any(original in value for value in os.environ.values()),
    'token_exposed': token in prompt or any(token in arg for arg in args),
    'server_name': server_name, 'operator_server': servers['daydream_source'],
}))
item = {'id': 'source-1', 'type': 'mcp_tool_call', 'server': server_name, 'tool': 'read_source',
        'arguments': payload['params']['arguments'], 'status': 'in_progress'}
print(json.dumps({'type': 'item.started', 'item': item}), flush=True)
print(json.dumps({'type': 'item.completed', 'item': {**item, 'status': 'completed', 'result': result}}), flush=True)
print(json.dumps({'type': 'turn.completed', 'usage': {}}), flush=True)
''')
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', f'{binary_dir}{os.pathsep}{os.environ["PATH"]}')
    backend = CodexBackend(model='fixture-model')
    events = [event async for event in backend.execute(
        repo, 'Inspect the supplied frozen source.', read_only=True, source_recipe=recipe,
    )]
    call, = [event for event in events if isinstance(event, ToolStartEvent)]
    result, = [event for event in events if isinstance(event, ToolResultEvent)]
    request_event, = [event for event in events if isinstance(event, RequestEvent)]
    assert call.name == 'read_source' and call.id == result.id == 'source-1'
    assert call.input == {'target_id': 'user.$name.tsx', 'side': 'after'}
    assert not result.is_error and result.status == 'completed'
    assert recipe.native_result(call.input, result.output) == recipe.windows[0]
    observation = json.loads(observed.read_text())
    assert observation['body'] == 'export const username = "世界";\n// changed\n'
    assert observation['source_path_exposed'] is observation['token_exposed'] is False
    assert observation['cwd'] != str(repo) and not Path(observation['cwd']).exists()
    assert observation['server_name'] != 'daydream_source'
    assert observation['operator_server'] == {'command': 'operator-source-driver', 'disabled_tools': ['read_source']}
    assert config.read_text() == operator_config
    assert request_event.config is not None and request_event.config.source_tool_enabled
    assert tracked_source_state(repo) == original
    assert backend._transports == []


@pytest.mark.parametrize('fault', ['failed-status', 'native-error', 'transport-error', 'multiple-blocks',
                                  'foreign-server', 'fixed-server', 'finalization'])
async def test_codex_source_failures_and_foreign_tools_cannot_ground_coverage(
    frozen_recipe: SourceRecipe, fault: str,
) -> None:
    recipe = frozen_recipe
    source = recipe.windows[0]
    packet = json.dumps({'source': source.metadata(), 'body': source.body})
    content = [{'type': 'text', 'text': packet}]
    if fault == 'multiple-blocks':
        content.append({'type': 'text', 'text': 'An unrelated warning.'})
    status = 'failed' if fault == 'failed-status' else 'completed'

    async def spawn_cli(*arguments: str, **_kwargs: Any) -> MagicMock:
        settings = dict(argument.split('=', 1) for index, argument in enumerate(arguments)
                        if index and arguments[index - 1] == '-c')
        names = [key[len('mcp_servers.'):-len('.url')] for key in settings
                 if key.startswith('mcp_servers.') and key.endswith('.url')]
        server_name = names[0] if names else 'daydream_source'
        item: dict[str, Any] = {
            'id': 'source-fault', 'type': 'mcp_tool_call',
            'server': 'foreign' if fault == 'foreign-server' else 'daydream_source'
            if fault == 'fixed-server' else server_name,
            'tool': 'read_source', 'arguments': {'target_id': source.file, 'side': 'after'}, 'status': 'in_progress',
        }
        completed = {**item, 'status': status, 'result': {'content': content, 'isError': fault == 'native-error'},
                     'error': {'message': 'Native MCP transport failure'} if fault == 'transport-error' else None}
        return make_mock_process([
            json.dumps({'type': 'item.started', 'item': item}),
            json.dumps({'type': 'item.completed', 'item': completed}),
            json.dumps({'type': 'turn.completed', 'usage': {}}),
        ])

    backend = CodexBackend(model='fixture-model')
    with patch('daydream.backends._transport.asyncio.create_subprocess_exec', side_effect=spawn_cli) as spawn:
        events = [event async for event in backend.execute(
            recipe.repo, 'Inspect frozen source.', source_recipe=recipe, finalization=fault == 'finalization',
        )]
    call, = [event for event in events if isinstance(event, ToolStartEvent)]
    result, = [event for event in events if isinstance(event, ToolResultEvent)]
    assert call.id == result.id == 'source-fault'
    assert result.is_error is (fault in {'failed-status', 'native-error', 'transport-error'})
    assert result.status == status
    if fault == 'foreign-server':
        assert call.name == 'mcp__foreign__read_source'
    if fault == 'fixed-server':
        assert call.name == 'mcp__daydream_source__read_source'
    if fault == 'multiple-blocks':
        assert result.output == ''
    if fault == 'finalization':
        assert 'DAYDREAM_SOURCE_TOKEN' not in spawn.call_args.kwargs['env']
        assert not any('mcp_servers.' in str(argument) for argument in spawn.call_args.args)
    evidence = ReviewEvidence(None)
    inputs = prepare_sanctioned_inputs(backend, recipe.repo, {}, read_only=True, source_recipe=recipe)
    evidence.configure_capture(recipe.repo, sanctioned_inputs=inputs)
    for event in events:
        evidence.observe(event)
    assert evidence.capture_failure([source.file])
    assert backend._transports == []


async def test_codex_closing_source_invocation_reaps_process_reader_and_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_recipe: SourceRecipe,
) -> None:
    binary_dir = tmp_path / 'bin'
    binary_dir.mkdir()
    observed = tmp_path / 'lifecycle.json'
    executable = binary_dir / 'codex'
    executable.write_text(f'#!{sys.executable}\n' + '''import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
sys.stdin.read()
settings = dict(arg.split('=', 1) for index, arg in enumerate(args) if index and args[index - 1] == '-c')
server_name, = [key[len('mcp_servers.'):-len('.url')] for key in settings
                if key.startswith('mcp_servers.') and key.endswith('.url')]
Path(''' + repr(str(observed)) + ''').write_text(json.dumps({
    'cwd': args[args.index('--cd') + 1], 'pid': os.getpid(),
    'url': json.loads(settings[f'mcp_servers.{server_name}.url']),
}))
print(json.dumps({'type': 'item.started', 'item': {'id': 'pending-source', 'type': 'mcp_tool_call',
    'server': server_name, 'tool': 'read_source', 'arguments': {
        'target_id': 'user.$name.tsx', 'side': 'after'}, 'status': 'in_progress'}}), flush=True)
time.sleep(60)
''')
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', f'{binary_dir}{os.pathsep}{os.environ["PATH"]}')
    backend = CodexBackend(model='fixture-model')
    stream = backend.execute(frozen_recipe.repo, 'Inspect source.', read_only=True, source_recipe=frozen_recipe)
    try:
        async for event in stream:
            if isinstance(event, ToolStartEvent):
                assert event.name == 'read_source'
                break
        observation = json.loads(observed.read_text())
        assert Path(observation['cwd']).exists()
        # A reachable reader requires its invocation bearer token.
        try:
            await asyncio.to_thread(urllib.request.urlopen, observation['url'], timeout=1)
        except urllib.error.HTTPError as exc:
            try:
                assert exc.code == 401
            finally:
                exc.close()
        else:
            pytest.fail('Source reader did not enforce bearer authentication')
    finally:
        await stream.aclose()
    assert backend._transports == []
    assert not Path(observation['cwd']).exists()
    with pytest.raises(ProcessLookupError):
        os.kill(observation['pid'], 0)
    with pytest.raises(urllib.error.URLError) as unavailable:
        urllib.request.urlopen(observation['url'], timeout=1)
    assert isinstance(unavailable.value.reason, ConnectionRefusedError)
