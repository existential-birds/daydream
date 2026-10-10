"""Exercise the packaged native output extension with the installed Pi CLI."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import PiRequestConfig, RequestEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
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
            assert not any('Live native Pi invocation budget:' in json.dumps(message)
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


@pytest.mark.parametrize('allowance', [8, None], ids=['bounded', 'unbounded'])
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
    (tmp_path / 'source.txt').write_text('Captured source\n')
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
            tmp_path, 'Investigate the supplied source and submit the result.', output_schema=_SCHEMA,
            read_only=True, persist_session=False, tool_call_budget=allowance)]

    assert len(requests) == 4
    for request, remaining in zip(requests, [8, 5, 4, 2], strict=True):
        notes = [message for message in request['messages']
                 if 'Live native Pi invocation budget:' in json.dumps(message)]
        assert len(notes) == (0 if allowance is None else 1)
        if notes:
            text = json.dumps(notes[0])
            assert f'{remaining} tool starts remain' in text
            assert 'structured_output costs 1' in text and 'Every parallel batch member counts' in text
            assert 'local native starts' in text and len(text.encode()) < 1024
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    assert [event.id for event in starts] == [call_id for batch in batches for call_id, _, _ in batch]
    failures = [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error]
    assert failures == ['read-missing', 'failed-submit']
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == output
