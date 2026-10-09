"""The installed native Pi CLI loads the shipped bounded frozen-source tool."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
from importlib.resources import as_file, files
from pathlib import Path

import pytest

from daydream.backends import PiRequestConfig, RequestEvent, ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from daydream.git_ops.source import frozen_source
from daydream.review_source import SourceRecipe, SourceWindow
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.protocol_cli import install_protocol_cli

_SOURCE_OUTPUT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'observed_body': {'type': 'string'}, 'source_error': {'type': 'boolean'},
                   'active_tools': {'type': 'array', 'items': {'type': 'string'}}},
    'required': ['observed_body', 'source_error', 'active_tools'],
}

@pytest.fixture
def isolated_pi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    native_pi = shutil.which('pi')
    if native_pi is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    config = tmp_path / 'isolated-pi-config'
    config.mkdir()
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(config))
    bin_dir = config / 'bin'
    bin_dir.mkdir()
    shim = bin_dir / 'pi'
    pids = config / 'pids'
    shim.write_text(f'#!{sys.executable}\nimport os, sys\n'
                   f'with open({str(pids)!r}, "a") as file: file.write(str(os.getpid()) + "\\n")\n'
                   f'os.execv({native_pi!r}, [{native_pi!r}, *sys.argv[1:]])\n')
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    for name in ('PI_API_KEY', 'NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    return config


@pytest.fixture
def native_source_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_pi: Path) -> None:
    native_pi = shutil.which('pi')
    assert native_pi is not None
    provider = Path(__file__).parent / 'fixtures/pi_native/source_provider.ts'
    bin_dir = tmp_path / 'provider-boundary'
    bin_dir.mkdir()
    shim = bin_dir / 'pi'
    shim.write_text(f'#!{sys.executable}\nimport os, sys\n'
                    f'os.execv({native_pi!r}, [{native_pi!r}, "--extension", {str(provider)!r}, *sys.argv[1:]])\n')
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PI_PROVIDER', 'source-fixture')


@pytest.mark.usefixtures('native_source_provider')
@pytest.mark.parametrize('selector', ['valid', 'unknown', 'arbitrary-path', 'wrong-side'])
async def test_installed_pi_reads_only_its_frozen_packet_and_preserves_native_call_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str,
) -> None:
    repo = tmp_path / 'native_deleted_source'
    body = 'def greeting():\n    return "世界"\n'
    seed_feature_branch(repo, base={'retired.py': body, 'keep.py': 'VALUE = 1\n'},
                        feature={'keep.py': 'VALUE = 2\n'})
    revision = git(repo, 'rev-parse', 'main')
    git(repo, 'rm', 'retired.py')
    git(repo, 'commit', '-m', 'delete greeting')
    oid, raw = frozen_source(repo, revision, 'retired.py')
    recipe = SourceRecipe((SourceWindow(
        target_ids=('retired.py',), file='retired.py', source_path='retired.py', side='before', revision=revision,
        content_sha256=hashlib.sha256(raw).hexdigest(), blob_oid=oid, start_line=1, end_line=2,
        start_byte=0, end_byte=len(raw), body=raw.decode('utf-8'),
    ),), repo)
    arguments = {'target_id': 'unknown.py' if selector == 'unknown' else 'retired.py',
                 'side': 'after' if selector == 'wrong-side' else 'before'}
    if selector == 'arbitrary-path':
        arguments['path'] = str(tmp_path / 'outside-packet.txt')
        (tmp_path / 'outside-packet.txt').write_text('OUTSIDE_PACKET_MUST_NOT_BE_READ\n')
    monkeypatch.setenv('DAYDREAM_TEST_SOURCE_SELECTOR', json.dumps(arguments))
    backend = PiBackend(model='source-model', reasoning_effort='high')
    events = [event async for event in backend.execute(
        repo, 'Read the supplied deleted source window.', output_schema=_SOURCE_OUTPUT_SCHEMA,
        read_only=True, persist_session=False, source_recipe=recipe,
    )]
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    final = next(event for event in reversed(events) if isinstance(event, ResultEvent)).structured_output
    assert len(starts) == len(results) == 2
    assert starts[1].name == 'structured_output' and not results[1].is_error
    assert starts[0].id == results[0].id == 'native-frozen-source-001'
    assert starts[0].name == 'read_source' and starts[0].input == arguments
    assert results[0].is_error is (selector != 'valid')
    assert not results[0].truncated and not results[0].cancelled
    assert final == {'observed_body': body if selector == 'valid' else '', 'source_error': selector != 'valid',
                     'active_tools': ['find', 'grep', 'ls', 'read', 'read_source', 'structured_output']}
    assert not (repo / 'retired.py').exists()
    if selector == 'valid':
        assert recipe.native_result(starts[0].input, results[0].output) == recipe.windows[0]
    with as_file(files('daydream.backends').joinpath('pi_read_source.ts')) as resource:
        assert resource.is_file()


@pytest.mark.parametrize('contract', ['matching', 'mismatched', 'plain'])
async def test_pi_request_has_one_matching_invocation_schema_and_keeps_custom_output_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: str,
) -> None:
    fixture = install_protocol_cli(tmp_path / 'external-provider', 'pi')
    monkeypatch.setenv('PATH', str(fixture.bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PI_PROVIDER', 'nous')
    monkeypatch.delenv('PI_API_KEY', raising=False)
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'status': {'type': 'string'}}, 'required': ['status']}
    other = {'type': 'object', 'additionalProperties': False,
             'properties': {'obsolete': {'type': 'string'}}, 'required': ['obsolete']}
    prompt = 'Review the assigned source.'
    if contract != 'plain':
        prompt += '\nHost review stage:\n' + json.dumps(
            {'response_contract': {'schema': schema if contract == 'matching' else other}})
    events = [event async for event in PiBackend(model='fixture-model').execute(
        tmp_path, prompt, output_schema=schema, read_only=True, require_complete_root=True,
    )]
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert isinstance(request.config, PiRequestConfig)
    assert request.output_schema == schema and request.config.schema_emulated is False
    assert request.prompt.count('"additionalProperties"') == (0 if contract == 'plain' else 1)
    assert ('"status"' in request.prompt) is (contract == 'matching')
    assert ('"obsolete"' in request.prompt) is (contract == 'mismatched')


@pytest.mark.usefixtures('native_source_provider')
async def test_installed_pi_bounded_read_footer_is_verified_against_actual_source_lines_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'native_bounded_read'
    body = 'VALUE = 1\ndef greeting():\n    return "world"\nEXPECTED = "world"\nassert greeting() == EXPECTED\n'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': body})
    revision = git(repo, 'rev-parse', 'HEAD')
    oid, raw = frozen_source(repo, revision, 'api.py')
    recipe = SourceRecipe((SourceWindow(
        target_ids=('api.py',), file='api.py', source_path='api.py', side='after', revision=revision,
        content_sha256=hashlib.sha256(raw).hexdigest(), blob_oid=oid, start_line=1, end_line=5,
        start_byte=0, end_byte=len(raw), body=body,
    ),), repo)
    monkeypatch.setenv('DAYDREAM_TEST_SOURCE_TOOL', 'read')
    monkeypatch.setenv('DAYDREAM_TEST_SOURCE_SELECTOR', json.dumps({'path': 'api.py', 'offset': 1, 'limit': 2}))
    events = [event async for event in PiBackend(model='source-model').execute(
        repo, 'Read the first two source lines.', output_schema=_SOURCE_OUTPUT_SCHEMA, read_only=True,
        persist_session=False, source_recipe=recipe,
    )]
    start = next(event for event in events if isinstance(event, ToolStartEvent))
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert start.id == result.id == 'native-frozen-source-001' and start.name == 'read'
    assert result.output == 'VALUE = 1\ndef greeting():\n\n[4 more lines in file. Use offset=3 to continue.]'
    assert result.is_error is False and result.truncated is False
    matched = recipe.match_read(start.input, result.output, repo)
    assert matched is not None, 'genuine bounded Pi read must yield independently verified source evidence'
    assert matched.body == 'VALUE = 1\ndef greeting():\n'
    assert (matched.start_line, matched.end_line, matched.start_byte, matched.end_byte) == (1, 2, 0, 26)


@pytest.mark.parametrize(('mode', 'defective'), [
    ('submitted', False), ('submitted', True), ('corrected', True), ('normalized', True),
    ('length', True), ('prose', True), ('mixed', True), ('mixed-read-first', True),
    ('multiple', True), ('later-failed', True),
    ('host-rejected', True), ('exhausted', True), ('boundary-corrected', True),
    ('over-limit-corrected', True), ('ordinary', True),
])
async def test_installed_pi_native_output_through_review_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_pi: Path, mode: str, defective: bool,
) -> None:
    """Only external HTTP responses are synthetic: Pi parses, validates, runs, and corrects tools."""
    import re
    import threading
    from collections.abc import AsyncIterator
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from typing import Any

    from daydream.backends import AgentEvent
    from daydream.phases.schemas import SUPERVISE_SCHEMA
    from tests.deep_orchestrator.test_review_completion import scopes
    from tests.deep_orchestrator.test_review_investigation import InvestigationRun, stage_ends
    from tests.deep_orchestrator.test_review_native_sources import NativeSourceBackend
    from tests.harness.stub_backend import review_stage_state, stage_result

    repo = tmp_path / 'native_review'
    api = 'def greeting():\n    return "world"\nEXPECTED = "world"\nassert greeting() == EXPECTED\n'
    seed_feature_branch(repo, base={'api.py': api},
                        feature={'api.py': api.replace('return "world"', 'return "universe"') if defective
                                 else api + '# Preserve the documented greeting contract.\n'})
    requests: list[dict[str, Any]] = []
    title = 'Greeting violates the retained world contract'
    ordinary_title = 'Native ordinary schema verdict applied'
    ordinary_events: list[AgentEvent] = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            prompt = next(message['content'] for message in request['messages'] if message['role'] == 'user')
            if isinstance(prompt, list):
                prompt = ''.join(part.get('text', '') for part in prompt)
            stage = review_stage_state(prompt)
            results = [message for message in request['messages'] if message['role'] == 'tool']
            calls: list[tuple[str, str, Any]] = []
            text = ''
            finish = 'tool_calls'
            if stage is None:
                assert mode == 'ordinary'
                tools = {tool['function']['name']: tool['function'] for tool in request['tools']}
                assert 'read_source' not in tools and tools['structured_output']['parameters'] == SUPERVISE_SCHEMA
                if not results:
                    path = re.search(r'listed in (\S+supervise-input\.json)', prompt)
                    assert path is not None
                    calls = [('ordinary-read', 'read', {'path': path[1]})]
                else:
                    items = json.loads(results[0]['content'])
                    verdicts = [{'id': item['id'], 'action': 'edit', 'reason': 'Native ordinary output applied',
                                 'severity': None, 'confidence': None, 'description': ordinary_title,
                                 'rationale': None, 'evidence': None} for item in items]
                    calls = [('ordinary-submit', 'structured_output', {'verdicts': verdicts})]
            elif not results:
                window = next(window for window in stage['source_access'] if window['side'] == 'after')
                calls = [('source', 'read_source', {'target_id': window['target_ids'][0], 'side': 'after'})]
            else:
                source = json.loads(results[0]['content'])['body']
                assert 'EXPECTED = "world"' in source and 'assert greeting() == EXPECTED' in source
                is_defect = 'return "universe"' in source
                assert is_defect is defective
                output = stage_result(stage)
                if is_defect:
                    finding = {'id': 1, 'file': 'api.py', 'line': 2, 'severity': 'medium', 'confidence': 'MEDIUM',
                               'description': title, 'rationale': 'The retained assertion requires world.',
                               'evidence': 'api.py:2 returns universe; api.py:3-4 requires world.'}
                    output['candidates'] = [{'candidate_id': '', 'file': 'api.py', 'line': 2,
                                             'trigger': 'Call greeting through the retained assertion',
                                             'consequence': 'AssertionError', 'grounds': finding['evidence'],
                                             'disposition': 'confirmed', 'finding': finding}]
                if len(results) == 1:
                    if mode in {'corrected', 'length', 'exhausted', 'boundary-corrected', 'over-limit-corrected'}:
                        calls = [('invalid', 'structured_output', {})]
                        if mode == 'length':
                            finish = 'length'
                    elif mode == 'prose':
                        text, finish = json.dumps(output), 'stop'
                    elif mode == 'multiple':
                        calls = [('first', 'structured_output', stage_result(stage)),
                                 ('last', 'structured_output', output)]
                    elif mode == 'later-failed':
                        calls = [('first', 'structured_output', output), ('invalid', 'structured_output', {})]
                    else:
                        if mode == 'normalized':
                            output['candidates'][0]['finding']['line'] = '2'
                        if mode == 'host-rejected':
                            output['candidates'][0]['grounds'] = ''
                        calls = [('final', 'structured_output', output)]
                        if mode in {'mixed', 'mixed-read-first'}:
                            calls.append(('mixed-read', 'read', {'path': 'api.py', 'offset': 1, 'limit': 4}))
                            if mode == 'mixed-read-first':
                                calls.reverse()
                elif mode in {'corrected', 'length', 'exhausted', 'boundary-corrected', 'over-limit-corrected'}:
                    # The next response is offered only after genuine Pi validation feedback.
                    assert ('output token limit' if mode == 'length' else 'Validation failed') in results[-1]['content']
                    rejected = (mode == 'exhausted' or mode == 'boundary-corrected' and len(results) < 47
                                or mode == 'over-limit-corrected' and len(results) < 48)
                    calls = [(f'correction-{len(results)}', 'structured_output', {} if rejected else output)]
                else:
                    # Mixed and error batches naturally continue; a later failure supplies no replacement.
                    text, finish = 'Review complete.', 'stop'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            delta: dict[str, Any] = {'role': 'assistant'}
            if calls:
                delta['tool_calls'] = [{'index': index, 'id': call_id, 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments)}}
                    for index, (call_id, name, arguments) in enumerate(calls)]
            else:
                delta['content'] = text
            for payload in ({'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                            {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                             'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}):
                self.wfile.write(('data: ' + json.dumps({'id': 'synthetic-native', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'native-test', **payload}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (isolated_pi / 'models.json').write_text(json.dumps({'providers': {'native-test': {
        'baseUrl': f'http://127.0.0.1:{server.server_port}/v1', 'api': 'openai-completions',
        'apiKey': 'synthetic-loopback-only', 'models': [{'id': 'native-test', 'reasoning': False,
            'input': ['text'], 'contextWindow': 32768, 'maxTokens': 8192}],
    }}}))
    monkeypatch.setenv('PI_PROVIDER', 'native-test')
    review = InvestigationRun(repo, tmp_path, monkeypatch)

    class NativeReview(NativeSourceBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            ordinary = mode == 'ordinary' and 'supervisor adjudication' in prompt.lower()
            if ordinary or stage is not None and stage['scope_id'] == 'python':
                if stage is not None:
                    self.stages.append(stage)
                if ordinary:
                    assert kwargs.get('source_recipe') is None
                self.calls.append({'prompt': prompt, **kwargs})
                async for event in PiBackend(model='native-test').execute(cwd, prompt, *args, **kwargs):
                    if ordinary:
                        ordinary_events.append(event)
                    yield event
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = NativeReview(repo)
    try:
        assert await review.run() == 0
        completed_requests = len(requests)
        for pid in (isolated_pi / 'pids').read_text().splitlines():
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid), 0)
        await asyncio.sleep(0.05)
        assert len(requests) == completed_requests, 'Native recovery must stop when cancellation and reaping finish'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    admitted = mode not in {'prose', 'host-rejected', 'exhausted', 'over-limit-corrected'}
    data = review.load()
    expected_title = ordinary_title if mode == 'ordinary' else title
    assert [finding['title'] for finding in data['findings']] == ([expected_title] if admitted and defective else [])
    assert scopes(data)['python']['status'] == ('complete' if admitted else 'incomplete')
    ends = stage_ends(review, 'python')
    assert len(ends) == 1, 'Pi owns native correction; the host must not start a schema-only replacement stage'
    metadata = ends[0]['metadata']
    assert metadata['admitted'] is admitted
    assert metadata['fresh_source_reads'] == (2 if mode in {'mixed', 'mixed-read-first'} else 1)
    expected_starts = (49 if mode in {'exhausted', 'over-limit-corrected'} else
                       48 if mode == 'boundary-corrected' else 1 if mode == 'prose' else 3 if mode in {
                           'corrected', 'length', 'mixed', 'mixed-read-first', 'multiple', 'later-failed'} else 2)
    assert metadata['observed_tool_starts'] == expected_starts
    assert metadata['remaining_tool_calls'] == max(0, metadata['hard_tool_call_allowance'] - expected_starts)
    if mode in {'exhausted', 'over-limit-corrected'}:
        assert metadata['hard_tool_call_allowance'] == 48 and metadata['submission_starts'] == 48
        assert scopes(data)['python']['reason_codes'] == ['host_tool_budget_exhaustion']
        assert len(requests) >= 49
    elif mode == 'boundary-corrected':
        assert metadata['hard_tool_call_allowance'] == 48 and metadata['submission_starts'] == 47
        assert metadata['failed_submissions'] == 46 and metadata['successful_submissions'] == 1
        assert len(requests) == 48
    elif mode == 'ordinary':
        assert len(requests) == 4
        assert [event.name for event in ordinary_events if isinstance(event, ToolStartEvent)] == [
            'read', 'structured_output']
        ordinary_request = next(event for event in ordinary_events if isinstance(event, RequestEvent))
        assert ordinary_request.output_schema == SUPERVISE_SCHEMA
        result = next(event for event in ordinary_events if isinstance(event, ResultEvent))
        assert result.structured_output_origin == 'native'
        assert isinstance(result.structured_output, dict)
        assert result.structured_output['verdicts'][0]['description'] == ordinary_title
    else:
        continued = mode in {'corrected', 'length', 'mixed', 'mixed-read-first', 'later-failed'}
        assert len(requests) == (3 if continued else 2)
    if mode == 'prose':
        assert scopes(data)['python']['reason_codes'] == ['missing_output']
    elif mode == 'host-rejected':
        assert scopes(data)['python']['reason_codes'] == ['evidence_incomplete']
    if admitted and defective:
        assert data['findings'][0]['line'] == 2
