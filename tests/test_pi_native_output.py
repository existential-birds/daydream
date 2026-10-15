"""Exercise the packaged native output extension with the installed Pi CLI."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import PiRequestConfig, RequestEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.otlp import _loopback_http_server, _QuietHTTPHandler

_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"verdict": {"type": "string"}, "findings": {"type": "array"}}, "required": ["verdict", "findings"]}


@pytest.mark.parametrize('native', [True, False], ids=['native', 'validation-optout'])
async def test_installed_pi_submits_native_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool,
) -> None:
    native_pi = shutil.which("pi")
    if native_pi is None or shutil.which("node") is None:
        pytest.skip("The installed official Pi CLI and Node are required for native tool loading.")
    config = tmp_path / "isolated-pi-config"
    config.mkdir()
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(config))
    for name in ("PI_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert not any('Live native Pi attempt budget:' in json.dumps(message)
                           for message in request['messages'])
            output = {"verdict": "complete", "findings": []}
            deltas: list[dict[str, Any]] = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": "native-output-001", "type": "function",
                    "function": {"name": "structured_output", "arguments": json.dumps(output)},
                }]}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            ]
            if not native:
                deltas[0]['choices'][0]['delta'] = {'role': 'assistant', 'content': json.dumps(output)}
                deltas[1]['choices'][0]['finish_reason'] = 'stop'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for payload in deltas:
                self.wfile.write(("data: " + json.dumps({
                    "id": "native-output", "object": "chat.completion.chunk", "created": 1,
                    "model": "output-model", **payload}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / "models.json").write_text(json.dumps({"providers": {"output-fixture": {
            "baseUrl": f"{base_url}/v1", "api": "openai-completions",
            "apiKey": "synthetic-loopback-only", "models": [{"id": "output-model", "reasoning": False,
                "input": ["text"], "contextWindow": 32768, "maxTokens": 8192}]}}}))
        monkeypatch.setenv("PI_PROVIDER", "output-fixture")
        events = [event async for event in PiBackend(model="output-model").execute(
            tmp_path, "Return the requested result.", output_schema=_SCHEMA, read_only=True, persist_session=False,
            validate_structured_output=native, tool_call_budget=None if native else 8)]

    request = next(event for event in events if isinstance(event, RequestEvent))
    assert isinstance(request.config, PiRequestConfig)
    assert request.config.no_extensions is native
    assert request.output_schema == _SCHEMA
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    assert [(event.id, event.name) for event in starts] == (
        [("native-output-001", "structured_output")] if native else [])
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == ({"verdict": "complete", "findings": []} if native else None)
    if not native:
        assert [event.text for event in events if isinstance(event, TextEvent)] == [
            '{"verdict": "complete", "findings": []}']


@pytest.mark.parametrize('allowance', [8, 7, 6, None],
                         ids=['ample', 'submission-only', 'exhausted', 'unbounded'])
async def test_native_pi_live_budget_counts_batch_failures_and_replaces_generation_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowance: int | None,
) -> None:
    if shutil.which('pi') is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    config = tmp_path / 'isolated-pi-config'
    config.mkdir()
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(config))
    for name in ('PI_API_KEY', 'NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    source = 'Captured source\n'
    (tmp_path / 'source.txt').write_text(source)
    prompt = 'Investigate the supplied source and submit the result.'
    requests: list[dict[str, Any]] = []
    output = {'verdict': 'complete', 'findings': []}
    batches: list[list[tuple[str, str, dict[str, Any]]]] = [
        [('read-a', 'read', {'path': 'source.txt'}), ('read-missing', 'read', {'path': 'missing.txt'}),
         ('read-b', 'read', {'path': 'source.txt'})],
        [('failed-submit', 'structured_output', {'verdict': 'incomplete', 'findings': False})],
        [('read-c', 'read', {'path': 'source.txt'}), ('read-d', 'read', {'path': 'source.txt'})],
        [('final-submit', 'structured_output', output)],
    ]

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            batch = batches[len(requests) - 1]
            deltas: list[dict[str, Any]] = [
                {'choices': [{'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [
                    {'index': index, 'id': call_id, 'type': 'function',
                     'function': {'name': name, 'arguments': json.dumps(arguments)}}
                    for index, (call_id, name, arguments) in enumerate(batch)]}, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
            ]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for delta in deltas:
                self.wfile.write(('data: ' + json.dumps({'id': 'pacing-output', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'output-model', **delta}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / 'models.json').write_text(json.dumps({'providers': {'output-fixture': {
            'baseUrl': f'{base_url}/v1', 'api': 'openai-completions', 'apiKey': 'synthetic-loopback-only',
            'models': [{'id': 'output-model', 'reasoning': False, 'input': ['text'],
                        'contextWindow': 32768, 'maxTokens': 8192}]}}}))
        monkeypatch.setenv('PI_PROVIDER', 'output-fixture')
        events = [event async for event in PiBackend(model='output-model').execute(
            tmp_path, prompt, output_schema=_SCHEMA,
            read_only=True, persist_session=False, tool_call_budget=allowance)]

    assert len(requests) == 4
    for index, (request, spent) in enumerate(zip(requests, [0, 3, 4, 6], strict=True)):
        assert any(prompt in json.dumps(message) for message in request['messages'] if message['role'] == 'user')
        history = {message['tool_call_id']: message['content'] for message in request['messages']
                   if message['role'] == 'tool'}
        previous = [call for batch in batches[:index] for call in batch]
        assert set(history) == {call_id for call_id, _, _ in previous}
        for call_id, name, arguments in previous:
            if name == 'read' and arguments['path'] == 'source.txt':
                assert history[call_id] == source
        notes = [message for message in request['messages']
                 if 'Live native Pi attempt budget:' in json.dumps(message)]
        assert len(notes) == (0 if allowance is None else 1)
        if notes:
            text = json.dumps(notes[0])
            assert allowance is not None
            remaining = max(0, allowance - spent)
            assert f'This attempt has {remaining} tool calls remaining' in text
            assert 'Each individual call counts, including failures and parallel batch members' in text
            assert 'Use this refreshed count without recounting history' in text
            assert 'observed' not in text and 'ledger' not in text and f'from {allowance}' not in text
            assert len(text.encode()) < 1024
            if remaining > 1:
                assert 'Keep one call for structured_output' in text
                assert 'Continue concrete unresolved checks for assigned work while tool/time capacity permits' in text
                assert 'submit when those checks settle' in text and 'Unused capacity is expected' in text
            elif remaining == 1:
                assert 'Only submission capacity remains' in text
                assert 'truthful incomplete decisions' in text
                assert 'Continue concrete unresolved' not in text
            else:
                assert 'Tool-call capacity is exhausted' in text and 'no call remains for structured_output' in text
                assert 'Keep one call' not in text and 'Submit now' not in text
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    assert [event.id for event in starts] == [call_id for batch in batches for call_id, _, _ in batch]
    failures = [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error]
    assert failures == ['read-missing', 'failed-submit']
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == output


async def test_native_pi_retry_refreshes_attempt_count_without_renewing_host_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real Pi restart resets local starts while earlier failed-attempt spend stays charged."""
    if shutil.which('pi') is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    config = tmp_path / 'isolated-pi-config'
    config.mkdir()
    (config / 'settings.json').write_text(json.dumps({'retry': {'enabled': False, 'maxRetries': 0}}))
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(config))
    for name in ('PI_API_KEY', 'NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / 'source.txt').write_text('Captured source\n')
    requests: list[dict[str, Any]] = []
    output = {'verdict': 'complete', 'findings': []}

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if len(requests) == 2:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': {'message': '503 Service Unavailable'}}).encode())
                return
            batch = ([('attempt-1-a', 'read', {'path': 'source.txt'}),
                      ('attempt-1-b', 'read', {'path': 'missing.txt'})] if len(requests) == 1 else
                     [('attempt-2-a', 'read', {'path': 'source.txt'})] if len(requests) == 3 else
                     [('attempt-2-submit', 'structured_output', output)])
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            deltas: list[dict[str, Any]] = [
                {'choices': [{'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [
                    {'index': index, 'id': call_id, 'type': 'function',
                     'function': {'name': name, 'arguments': json.dumps(arguments)}}
                    for index, (call_id, name, arguments) in enumerate(batch)]}, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
            ]
            for delta in deltas:
                self.wfile.write(('data: ' + json.dumps({'id': 'retry-output', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'output-model', **delta}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / 'models.json').write_text(json.dumps({'providers': {'output-fixture': {
            'baseUrl': f'{base_url}/v1', 'api': 'openai-completions', 'apiKey': 'synthetic-loopback-only',
            'models': [{'id': 'output-model', 'reasoning': False, 'input': ['text'],
                        'contextWindow': 32768, 'maxTokens': 8192}]}}}))
        monkeypatch.setenv('PI_PROVIDER', 'output-fixture')
        backend = PiBackend(model='output-model')
        backend.retry_attempts = 1
        backend.retry_base_delay_s = backend.retry_max_delay_s = 0
        budget = ReviewInvestigationBudget.from_limits(ReviewLimits(60, 3, 8))
        result, _, reason = await run_agent(
            backend, tmp_path, 'Investigate the supplied source and submit the result.',
            phase=DaydreamPhase.DEEP, output_schema=_SCHEMA, read_only=True, persist_session=False,
            investigation_budget=budget, tool_call_budget=8, progress_callback=lambda _: None)

    assert reason is None and result == output
    assert budget.observed_tool_starts == 4 and budget.remaining_tool_calls == 4
    assert len(requests) == 4
    for request, remaining in zip(requests, [8, 6, 6, 5], strict=True):
        notes = [message for message in request['messages'] if 'Live native Pi attempt budget:' in json.dumps(message)]
        assert len(notes) == 1
        text = json.dumps(notes[0])
        assert f'This attempt has {remaining} tool calls remaining' in text
        assert 'observed' not in text and 'ledger' not in text and len(text.encode()) < 1024
    assert 'attempt-1-a' not in json.dumps(requests[2]['messages'])
    assert 'attempt-2-a' in json.dumps(requests[3]['messages'])
