"""The installed native Pi CLI loads the shipped bounded frozen-source tool."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
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
    'properties': {'observed_body': {'type': 'string'}, 'source_error': {'type': 'boolean'}},
    'required': ['observed_body', 'source_error'],
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
    shim.write_text(f'#!{sys.executable}\nimport os, sys, json\nfrom pathlib import Path\n'
                   'packet = os.environ.get("DAYDREAM_PI_SOURCE_PACKET")\n'
                   'fault = os.environ.get("DAYDREAM_TEST_PACKET_FAULT")\n'
                   'if packet and fault:\n'
                   '    path = Path(packet)\n'
                   '    if fault == "digest": path.write_bytes(path.read_bytes() + b" ")\n'
                   '    else:\n'
                   '        value = json.loads(path.read_text())\n'
                   '        value.pop("windows", None)\n'
                   '        if fault == "initialization": value["windows"] = [{}]\n'
                   '        path.write_text(json.dumps(value))\n'
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
@pytest.mark.parametrize('selector', ['valid', 'unknown', 'wrong-side'])
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
    assert final == {'observed_body': body if selector == 'valid' else '', 'source_error': selector != 'valid'}
    assert not (repo / 'retired.py').exists()
    if selector == 'valid':
        assert recipe.native_result(starts[0].input, results[0].output) == recipe.windows[0]


@pytest.mark.parametrize('contract', ['matching', 'mismatched', 'plain'])
async def test_pi_native_schema_transport_preserves_existing_prompt_contract(
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
    expected_prompt = prompt + '\n\nSubmit the final result using structured_output alone as your final action.'
    assert request.prompt == expected_prompt
    assert fixture.read_observations()[0]['prompt_sha256'] == hashlib.sha256(expected_prompt.encode()).hexdigest()


@pytest.mark.parametrize(('mode', 'defective'), [
    ('submitted', False), ('submitted', True),
    ('length', True), ('prose', True), ('reminder', True), ('invalid-prose', True),
    ('host-rejected', True), ('exhausted', True), ('boundary-corrected', True),
    ('over-limit-corrected', True), ('ordinary', True), ('ordinary-reminder', True), ('optional-unavailable', True),
    ('cancelled', True), ('deadline', True), ('checkout-unavailable', True),
    ('zero-selector', True), ('length-read', True), ('length-source', True),
    ('wrong-packet-digest', True), ('unloaded-source-tool', True), ('packet-initialization', True),
])
async def test_installed_pi_native_output_through_review_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_pi: Path, mode: str, defective: bool,
) -> None:
    """Daydream admits native submissions with verified source, bounded work, and process cleanup."""
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
    from tests.harness.fake_clock import FakeClock
    from tests.harness.stub_backend import review_stage_state, stage_result

    if mode in {'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'}:
        monkeypatch.setenv('DAYDREAM_TEST_PACKET_FAULT', {
            'wrong-packet-digest': 'digest', 'unloaded-source-tool': 'unloaded',
            'packet-initialization': 'initialization'}[mode])
    repo = tmp_path / 'native_review'
    api = 'def greeting():\n    return "world"\nEXPECTED = "world"\nassert greeting() == EXPECTED\n'
    seed_feature_branch(repo, base={'api.py': api},
                        feature={'api.py': api.replace('return "world"', 'return "universe"') if defective
                                 else api + '# Preserve the documented greeting contract.\n'})
    requests: list[dict[str, Any]] = []
    title = 'Greeting violates the retained world contract'
    ordinary_title = 'Native ordinary schema verdict applied'
    ordinary_events: list[AgentEvent] = []
    native_events: list[AgentEvent] = []
    prose_pending, release_prose = threading.Event(), threading.Event()
    clock = FakeClock().install(monkeypatch) if mode == 'deadline' else None

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
                assert mode in {'ordinary', 'ordinary-reminder'}
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
                    if mode == 'ordinary-reminder' and len(requests) == 4:
                        text, finish = json.dumps({'verdicts': verdicts}), 'stop'
                    else:
                        calls = [('ordinary-submit', 'structured_output', {'verdicts': verdicts})]
            elif not results:
                window = next(window for window in stage['source_access'] if window['side'] == 'after')
                calls = [('source', 'read_source', {'target_id': window['target_ids'][0], 'side': 'after'})]
                if mode in {'unloaded-source-tool', 'packet-initialization'}:
                    calls = [('source', 'read', {'path': 'api.py'})]
            else:
                source = (results[0]['content'] if mode in {'unloaded-source-tool', 'packet-initialization'} else
                          json.loads(results[0]['content'])['body'])
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
                    if mode in {'length', 'exhausted', 'boundary-corrected', 'over-limit-corrected',
                                'invalid-prose'}:
                        calls = [('invalid', 'structured_output', {})]
                        if mode == 'length':
                            finish = 'length'
                    elif mode in {'prose', 'cancelled', 'deadline'} or mode == 'reminder' and len(requests) == 2:
                        text, finish = json.dumps(output), 'stop'
                        if mode in {'cancelled', 'deadline'}:
                            prose_pending.set()
                            assert release_prose.wait(10), 'The runner must stop within its original allowance'
                    elif mode in {'optional-unavailable', 'checkout-unavailable', 'length-read',
                                  'zero-selector', 'length-source',
                                  'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'}:
                        source_lookup = mode in {'zero-selector', 'length-source',
                                                'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'}
                        calls = [('optional-read', 'read_source' if source_lookup else 'read',
                                  {'target_id': 'integration:structure', 'side': 'after'} if source_lookup else
                                  {'path': 'unrequested-missing.txt' if mode != 'optional-unavailable' else
                                   str(tmp_path / 'unrequested-missing.txt')})]
                        if mode in {'length-read', 'length-source'}:
                            finish = 'length'
                    else:
                        if mode == 'host-rejected':
                            output['candidates'][0]['grounds'] = ''
                        calls = [('final', 'structured_output', output)]
                elif mode in {'length', 'exhausted', 'boundary-corrected', 'over-limit-corrected'}:
                    # The next response is offered only after genuine Pi validation feedback.
                    assert ('output token limit' if mode == 'length' else 'Validation failed') in results[-1]['content']
                    rejected = (mode == 'exhausted' or mode == 'boundary-corrected' and len(results) < 47
                                or mode == 'over-limit-corrected' and len(results) < 48)
                    calls = [(f'correction-{len(results)}', 'structured_output', {} if rejected else output)]
                elif mode in {'optional-unavailable', 'checkout-unavailable', 'length-read',
                                  'zero-selector', 'length-source',
                                  'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'}:
                    assert len(results) == 2
                    assert ('output token limit' if mode.startswith('length-') else
                            'selector unavailable' if mode in {'zero-selector', 'wrong-packet-digest'} else
                            'not found' if mode in {'unloaded-source-tool', 'packet-initialization'} else 'ENOENT'
                            ) in results[-1]['content']
                    calls = [('after-optional', 'structured_output', output)]
                elif mode == 'invalid-prose':
                    assert len(results) == 2 and 'Validation failed' in results[-1]['content']
                    if len(requests) == 3:
                        text, finish = json.dumps(output), 'stop'
                    else:
                        calls = [('after-reminder', 'structured_output', output)]
                else:
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
            ordinary = mode in {'ordinary', 'ordinary-reminder'} and 'supervisor adjudication' in prompt.lower()
            if ordinary or stage is not None and stage['scope_id'] == 'python':
                if stage is not None:
                    self.stages.append(stage)
                if ordinary:
                    assert kwargs.get('source_recipe') is None
                self.calls.append({'prompt': prompt, **kwargs})
                async for event in PiBackend(model='native-test').execute(cwd, prompt, *args, **kwargs):
                    if ordinary:
                        ordinary_events.append(event)
                    else:
                        native_events.append(event)
                    yield event
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = NativeReview(repo)
    try:
        if mode in {'cancelled', 'deadline'}:
            task = asyncio.create_task(review.run())
            try:
                pending = await asyncio.to_thread(prose_pending.wait, 30)
                if task.done():
                    await task
                assert pending
                if mode == 'cancelled':
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    assert clock is not None
                    clock.advance(100_000)
                    release_prose.set()
                    assert await task == 0
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        else:
            assert await review.run() == 0
        completed_requests = len(requests)
        for pid in (isolated_pi / 'pids').read_text().splitlines():
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid), 0)
        await asyncio.sleep(0.05)
        assert len(requests) == completed_requests, 'Native recovery must stop when cancellation and reaping finish'
    finally:
        release_prose.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if mode == 'cancelled':
        assert not review.output.exists()
        interrupted = stage_ends(review, 'python')
        assert len(interrupted) == 1 and interrupted[0]['metadata']['admitted'] is False
        assert interrupted[0]['metadata']['observed_tool_starts'] == 1
        assert len(requests) == 2
        return
    admitted = mode not in {'prose', 'host-rejected', 'exhausted', 'over-limit-corrected', 'deadline',
                            'length-read', 'length-source',
                            'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'}
    data = review.load()
    expected_title = ordinary_title if mode in {'ordinary', 'ordinary-reminder'} else title
    assert [finding['title'] for finding in data['findings']] == ([expected_title] if admitted and defective else [])
    if mode == 'packet-initialization':
        assert scopes(data)['python']['status'] == 'failed'
        assert scopes(data)['python']['reason_codes'] == ['backend_failure']
        terminal, = stage_ends(review, 'python')
        assert terminal['metadata']['observed_tool_starts'] == 0
        assert terminal['metadata']['admitted'] is False
        assert not requests
        return
    assert scopes(data)['python']['status'] == ('complete' if admitted else 'incomplete')
    ends = stage_ends(review, 'python')
    assert len(ends) == 1, 'Pi owns native correction; the host must not start a schema-only replacement stage'
    metadata = ends[0]['metadata']
    assert metadata['admitted'] is admitted
    assert metadata['fresh_source_reads'] == 1
    expected_starts = (49 if mode in {'exhausted', 'over-limit-corrected'} else
                       48 if mode == 'boundary-corrected' else 1 if mode in {'prose', 'deadline'} else 3 if mode in {
                           'length', 'optional-unavailable', 'invalid-prose', 'checkout-unavailable', 'zero-selector',
                           'length-read', 'length-source',
                           'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'} else 2)
    assert metadata['observed_tool_starts'] == expected_starts
    assert metadata['remaining_tool_calls'] == max(0, metadata['hard_tool_call_allowance'] - expected_starts)
    native_request = next(event for event in native_events if isinstance(event, RequestEvent))
    assert isinstance(native_request.config, PiRequestConfig) and native_request.config.no_extensions is True
    if mode in {'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'}:
        failed = next(event for event in native_events if isinstance(event, ToolResultEvent)
                      and event.id == 'optional-read')
        assert failed.is_error and failed.source_free_disposition is None
        assert metadata['source_access_failures'] == 1
        assert metadata['nonblocking_unavailable_reads'] == 0
    if mode in {'length-read', 'length-source'}:
        incomplete = next(event for event in native_events if isinstance(event, ToolStartEvent)
                          and event.id == 'optional-read')
        assert getattr(incomplete, 'input_incomplete', False) is True
        assert metadata['blocking_opaque_receipts'] == 1
    if mode in {'optional-unavailable', 'checkout-unavailable', 'zero-selector'}:
        failed = next(event for event in native_events if isinstance(event, ToolResultEvent)
                      and event.id == 'optional-read')
        assert failed.is_error and not failed.cancelled and not failed.truncated
        assert metadata['source_access_failures'] == 0
        assert metadata['nonblocking_unavailable_reads'] == 1
        assert metadata['blocking_opaque_receipts'] == metadata['blocking_pending_receipts'] == 0
    if mode in {'exhausted', 'over-limit-corrected'}:
        assert metadata['hard_tool_call_allowance'] == 48 and metadata['submission_starts'] == 48
        assert scopes(data)['python']['reason_codes'] == ['host_tool_budget_exhaustion']
        assert len(requests) >= 49
    elif mode == 'boundary-corrected':
        assert metadata['hard_tool_call_allowance'] == 48 and metadata['submission_starts'] == 47
        assert metadata['failed_submissions'] == 46 and metadata['successful_submissions'] == 1
        assert len(requests) == 48
    elif mode in {'ordinary', 'ordinary-reminder'}:
        assert len(requests) == (5 if mode == 'ordinary-reminder' else 4)
        assert [event.name for event in ordinary_events if isinstance(event, ToolStartEvent)] == [
            'read', 'structured_output']
        ordinary_request = next(event for event in ordinary_events if isinstance(event, RequestEvent))
        assert ordinary_request.output_schema == SUPERVISE_SCHEMA
        result = next(event for event in ordinary_events if isinstance(event, ResultEvent))
        assert result.structured_output_origin == 'native'
        assert isinstance(result.structured_output, dict)
        assert result.structured_output['verdicts'][0]['description'] == ordinary_title
    elif mode == 'deadline':
        assert len(requests) in {2, 3}, 'A raced reminder must stop when the native process is reaped'
        assert scopes(data)['python']['reason_codes'] == ['host_wall_budget_exhaustion']
    elif mode == 'invalid-prose':
        assert len(requests) == 4
        assert metadata['failed_submissions'] == metadata['successful_submissions'] == 1
    else:
        continued = mode in {'length', 'reminder', 'prose',
                            'optional-unavailable', 'checkout-unavailable', 'zero-selector',
                            'length-read', 'length-source',
                            'wrong-packet-digest', 'unloaded-source-tool', 'packet-initialization'}
        assert len(requests) == (3 if continued else 2)
    reminders = [message for request in requests for message in request['messages']
                 if message['role'] == 'user' and 'Assistant prose is not a submission.' in str(message['content'])]
    if mode == 'deadline':
        assert len(reminders) <= 1
    else:
        assert len(reminders) == (1 if mode in {'reminder', 'prose', 'ordinary-reminder', 'invalid-prose'} else 0)
    if mode == 'prose':
        assert scopes(data)['python']['reason_codes'] == ['missing_output']
    elif mode == 'host-rejected':
        assert scopes(data)['python']['reason_codes'] == ['evidence_incomplete']
    if admitted and defective:
        assert data['findings'][0]['line'] == 2


@pytest.mark.parametrize('fault', ['none', 'metadata-only', 'mutated-catalog', 'missing-catalog', 'symlink-catalog'])
async def test_installed_pi_compaction_restores_exact_structure_access_and_settles_lookup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_pi: Path, fault: str,
) -> None:
    import threading
    from collections.abc import AsyncIterator
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from typing import Any

    from daydream.backends import AgentEvent
    from tests.deep_orchestrator.test_review_completion import scopes
    from tests.deep_orchestrator.test_review_investigation import InvestigationRun, stage_ends
    from tests.deep_orchestrator.test_review_native_sources import NativeSourceBackend
    from tests.harness.stub_backend import review_stage_state, stage_result

    repo = tmp_path / 'compacted_structure'
    before = {'api.py': 'VALUE = 0\n', 'retired.py': 'RETIRED = True\n',
              'legacy.py': '# Rename context\n' * 30 + 'VALUE = 0\n',
              'late.py': '# Stable context\n' * 9000 + 'VALUE = 0\n'}
    after = {**before, 'api.py': 'VALUE = 1\n',
             'late.py': before['late.py'] + '# New bounded context\n' * 2000}
    seed_feature_branch(repo, base=before, feature=after)
    git(repo, 'rm', 'retired.py')
    git(repo, 'mv', 'legacy.py', 'modern.py')
    (repo / 'modern.py').write_text(before['legacy.py'].replace('VALUE = 0', 'VALUE = 1'))
    git(repo, 'add', 'modern.py')
    git(repo, 'commit', '-m', 'retire and rename source')
    requests: list[dict[str, Any]] = []
    native_events: list[AgentEvent] = []
    saved_stage: dict[str, Any] = {}
    guide: dict[str, Any] = {}
    summary_calls = 0
    action_index = 0
    actions: list[tuple[str, str, dict[str, Any]]] = []
    catalog_paths: list[str] = []
    entries: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:
            nonlocal saved_stage, guide, summary_calls, action_index, actions
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            tools = {tool['function']['name'] for tool in request.get('tools', [])}
            calls: list[tuple[str, str, Any]] = []
            text, finish, tokens = '', 'tool_calls', 10
            if 'structured_output' not in tools:
                # The installed SDK asks the same loopback provider to summarize.
                summary_calls += 1
                text, finish = 'Earlier investigation is complete. Use persistent host guidance to continue.', 'stop'
            elif not saved_stage:
                user = next(message['content'] for message in request['messages'] if message['role'] == 'user')
                if isinstance(user, list):
                    user = ''.join(part.get('text', '') for part in user)
                saved_stage = review_stage_state(user) or {}
                assert saved_stage['scope_id'] == 'structure'
                window = next(window for window in saved_stage['source_access']
                              if window['file'] == 'api.py' and window['side'] == 'after')
                calls = [('initial-source', 'read_source', window['access']['arguments'])]
                if fault == 'metadata-only':
                    calls = [('initial-context', 'read', {'path': saved_stage['supporting_bundle']['path']})]
            elif summary_calls == 0:
                # A completed prose turn triggers threshold compaction, followed by
                # the owned missing-submission continuation under the same attempt.
                text, finish, tokens = 'Investigation complete.', 'stop', 30000
            else:
                system = next(message['content'] for message in request['messages'] if message['role'] == 'system')
                marker = 'Persistent exact access guide (supporting metadata): '
                guide = json.JSONDecoder().raw_decode(system.split(marker, 1)[1])[0] if marker in system else {}
                if not guide:
                    calls = [('unguided-submit', 'structured_output', stage_result(saved_stage))]
                else:
                    assert 'Read the diff first' not in system
                    assert len(json.dumps(guide, separators=(',', ':'), ensure_ascii=False).encode()) <= 8192
                    assert not any('Host review stage:' in str(message['content'])
                                   for message in request['messages'] if message['role'] == 'user')
                    if not catalog_paths:
                        root = guide['source_catalog']['path']
                        catalog_paths.append(root)
                        actions.append(('catalog-0', 'read', {'path': root}))
                    results = [message for message in request['messages'] if message['role'] == 'tool']
                    for result in results:
                        if not result.get('tool_call_id', '').startswith('catalog-'):
                            continue
                        catalog = json.loads(result['content'])
                        if 'windows' in catalog:
                            for entry in catalog['windows']:
                                if entry not in entries:
                                    entries.append(entry)
                        else:
                            for child in catalog['catalogs']:
                                if child['path'] not in catalog_paths:
                                    catalog_paths.append(child['path'])
                                    actions.append((f'catalog-{len(catalog_paths)-1}', 'read', {'path': child['path']}))
                    if action_index == len(actions) and not selected:
                        selected.extend([
                            next(entry for entry in entries if entry['file'] == 'retired.py'),
                            next(entry for entry in entries if entry['file'] == 'modern.py'
                                 and entry['side'] == 'before'),
                            next(entry for entry in entries if entry['file'] == 'modern.py'
                                 and entry['side'] == 'after'),
                            max((entry for entry in entries if entry['file'] == 'late.py'
                                 and entry['side'] == 'before'),
                                key=lambda entry: entry['start_byte']),
                        ])
                        assert selected[-1]['start_byte'] > 128000
                        assert selected[0]['side'] == 'before'
                        assert selected[1]['source_path'] == 'legacy.py' and selected[2]['source_path'] == 'modern.py'
                        assert 'integration:structure' not in {
                            target for entry in entries for target in entry['target_ids']}
                        if fault != 'metadata-only':
                            actions.extend((f'frozen-{i}', 'read_source', entry['access']['arguments'])
                                           for i, entry in enumerate(selected))
                        exact = next(context['path'] for context in guide['contexts'] if 'path' in context)
                        actions.extend([('exact-context', 'read', {'path': exact}),
                                        ('absent-context', 'read', {'path': '.daydream/exploration/absent.md'}),
                                        ('zero-selector', 'read_source',
                                         {'target_id': 'integration:structure', 'side': 'after'})])
                    if action_index < len(actions):
                        calls = [actions[action_index]]
                        action_index += 1
                    else:
                        root = Path(catalog_paths[-1])
                        if fault == 'mutated-catalog':
                            root.write_text('{}')
                        elif fault == 'missing-catalog':
                            root.unlink()
                        elif fault == 'symlink-catalog':
                            copy = tmp_path / 'catalog-copy.json'
                            copy.write_bytes(root.read_bytes())
                            root.unlink()
                            root.symlink_to(copy)
                        calls = [('submit', 'structured_output', stage_result(saved_stage))]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            delta: dict[str, Any] = {'role': 'assistant'}
            if calls:
                delta['tool_calls'] = [{'index': i, 'id': call_id, 'type': 'function',
                                       'function': {'name': name, 'arguments': json.dumps(arguments)}}
                                      for i, (call_id, name, arguments) in enumerate(calls)]
            else:
                delta['content'] = text
            for payload in ({'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                            {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                             'usage': {'prompt_tokens': tokens, 'completion_tokens': 5, 'total_tokens': tokens + 5}}):
                self.wfile.write(('data: ' + json.dumps({'id': 'synthetic-compaction',
                    'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'native-guide', **payload}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (isolated_pi / 'models.json').write_text(json.dumps({'providers': {'native-guide': {
        'baseUrl': f'http://127.0.0.1:{server.server_port}/v1', 'api': 'openai-completions',
        'apiKey': 'synthetic-loopback-only', 'models': [{'id': 'native-guide', 'reasoning': False,
            'input': ['text'], 'contextWindow': 32768, 'maxTokens': 8192}],
    }}}))
    (isolated_pi / 'settings.json').write_text(json.dumps({'compaction': {
        'enabled': True, 'reserveTokens': 8192, 'keepRecentTokens': 1}}))
    monkeypatch.setenv('PI_PROVIDER', 'native-guide')
    review = InvestigationRun(repo, tmp_path, monkeypatch)

    class NativeGuide(NativeSourceBackend):
        supports_review_instructions = True

        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is not None and stage['scope_id'] == 'structure':
                self.stages.append(stage)
                self.calls.append({'prompt': prompt, **kwargs})
                async for event in PiBackend(model='native-guide').execute(cwd, prompt, *args, **kwargs):
                    native_events.append(event)
                    yield event
                return
            kwargs.pop('review_instructions', None)
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = NativeGuide(repo)
    try:
        assert await review.run() == (1 if fault == 'symlink-catalog' else 0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert summary_calls == 1 and guide
    for pid in (isolated_pi / 'pids').read_text().splitlines():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid), 0)
    if fault == 'symlink-catalog':
        assert not review.output.exists(), 'Unsafe artifact publication must remain atomic'
        coverage_path = next(parent / 'review-coverage.json' for parent in Path(catalog_paths[0]).parents
                             if (parent / 'review-coverage.json').is_file())
        coverage = json.loads(coverage_path.read_text())
        structural = next(scope for scope in coverage['stack_outcomes'] if scope['scope_id'] == 'structure')
        assert structural['status'] == 'incomplete'
        assert structural['reason_codes'] == ['evidence_incomplete']
        return
    data = review.load()
    assert scopes(data)['structure']['status'] == ('complete' if fault == 'none' else 'incomplete')
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['alternatives']['status'] == ('complete' if fault == 'none' else 'failed')
    terminal, = stage_ends(review, 'structure')
    if fault in {'none', 'metadata-only'}:
        metadata = terminal['metadata']
        assert metadata['nonblocking_unavailable_reads'] == 2
        assert metadata['source_access_failures'] == 0
        assert metadata['fresh_source_reads'] == (5 if fault == 'none' else 0)
        assert metadata['admitted'] is (fault == 'none')
    failures = [event for event in native_events if isinstance(event, ToolResultEvent) and event.is_error]
    assert len(failures) == 2 and {event.id for event in failures} == {'absent-context', 'zero-selector'}
    assert next(event for event in failures if event.id == 'zero-selector').source_free_disposition == 'zero_match'
