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


@pytest.mark.parametrize('allowance,submit', [(8, True), (7, True), (6, True), (2, True), (1, True), (None, True),
                                            (7, False), (6, False)],
                         ids=['ample', 'submission-only', 'batch-exhausted', 'batch-overdrawn',
                              'initial-only', 'unbounded', 'bounded-recovery', 'exhausted-no-recovery'])
async def test_native_pi_budget_warning_preserves_wire_roles_history_and_charged_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowance: int | None, submit: bool,
) -> None:
    """Actual wire requests catch custom notes becoming users or dropping genuine history.

    The immediate-submission journey misses threshold transitions, failed/invalid batch
    charges and missing-submission recovery at an unchanged remaining allowance.
    """
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
    assignment = {'objective': 'first_pass', 'target_ids': ['target-source'], 'candidate_ids': ['candidate-source']}
    policy = 'Stable original system policy: preserve this exact instruction.'
    instructions = policy + '\nDaydream final-submission assignment: ' + json.dumps(assignment)
    requests: list[dict[str, Any]] = []
    output = {'verdict': 'complete', 'findings': []}
    batches: list[list[tuple[str, str, dict[str, Any]]]] = [
        [('read-a', 'read', {'path': 'source.txt'}), ('read-missing', 'read', {'path': 'missing.txt'}),
         ('read-b', 'read', {'path': 'source.txt'})],
        [('failed-submit', 'structured_output', {'verdict': 'incomplete', 'findings': False})],
        [('read-c', 'read', {'path': 'source.txt'}), ('read-d', 'read', {'path': 'source.txt'})],
        *([[]] if allowance in (8, 7, None) else []),  # Repeat threshold through bounded recovery.
        [('final-submit', 'structured_output', output)] if submit else [],
    ]

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            index = len(requests) - 1
            batch = batches[index]
            assistant_delta: dict[str, Any] = {
                'role': 'assistant', 'content': f'Original assistant reasoning summary {index}.'}
            if batch:
                assistant_delta['tool_calls'] = [
                    {'index': index, 'id': call_id, 'type': 'function',
                     'function': {'name': name, 'arguments': json.dumps(arguments)}}
                    for index, (call_id, name, arguments) in enumerate(batch)]
            deltas: list[dict[str, Any]] = [
                {'choices': [{'index': 0, 'delta': assistant_delta, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls' if batch else 'stop'}],
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
            tmp_path, prompt, output_schema=_SCHEMA, review_instructions=instructions,
            read_only=True, persist_session=False, tool_call_budget=allowance)]

    assert len(requests) == len(batches)
    completions: dict[str, str] = {}
    for event in events:
        if isinstance(event, ToolResultEvent):
            # The host retains native details as JSON; OpenAI's tool content carries text only.
            completions[event.id] = (''.join(block['text'] for block in json.loads(event.output)['content'])
                                     if event.output.startswith('{"content":') else event.output)
    warning_requests: list[int] = []
    recovery = 'Submit the final result now by calling structured_output alone. Assistant prose is not a submission.'
    warning_instruction = ('Final-submission budget warning: Only submission capacity remains. '
                           'Submit now using structured_output alone; report unfinished assigned work '
                           'with truthful incomplete decisions. Assigned objective and identities: '
                           + json.dumps(assignment, separators=(',', ':')))
    original_system: dict[str, Any] | None = None
    for index, request in enumerate(requests):
        messages = request['messages']
        system = [message for message in messages if message['role'] in {'system', 'developer'}]
        assert system and messages[0] == system[0]
        assert policy in json.dumps(system)
        assert 'Attempt dispatch-time snapshot:' in json.dumps(system)
        preserved_system = dict(system[0])
        preserved_system['content'] = (
            system[0]['content'].replace(warning_instruction, '').replace(recovery, '').strip())
        if original_system is None:
            original_system = preserved_system
        assert preserved_system == original_system
        users = [message for message in messages if message['role'] == 'user']
        assert len(users) == 1 and prompt in json.dumps(users[0])
        assert 'Live native Pi attempt budget:' not in json.dumps(messages)
        assert 'daydream_native_tool_budget' not in json.dumps(messages)
        assert request['tools'] == requests[0]['tools']
        assert next(tool for tool in request['tools'] if tool['function']['name'] == 'structured_output')[
            'function']['parameters'] == _SCHEMA
        assistants = [message for message in messages if message['role'] == 'assistant']
        assert len(assistants) == index
        for earlier, assistant in enumerate(assistants):
            assert assistant['content'] == f'Original assistant reasoning summary {earlier}.'
            assert [(call['id'], call['function']['name'], json.loads(call['function']['arguments']))
                    for call in assistant.get('tool_calls', [])] == batches[earlier]
        history = {message['tool_call_id']: message['content'] for message in messages if message['role'] == 'tool'}
        previous = [call for batch in batches[:index] for call in batch]
        assert set(history) == {call_id for call_id, _, _ in previous}
        for call_id, name, arguments in previous:
            assert history[call_id] == completions[call_id]
            if name == 'read' and arguments['path'] == 'source.txt':
                assert history[call_id] == source
        warnings = [message for message in messages if 'Final-submission budget warning:' in json.dumps(message)]
        if warnings:
            warning_requests.append(index)
            assert len(warnings) == 1 and warnings[0]['role'] in {'system', 'developer'}
            warning = json.dumps(warnings[0])
            assert 'Only submission capacity remains' in warning and 'truthful incomplete decisions' in warning
            assignment_text = warnings[0]['content'].split('Assigned objective and identities: ', 1)[1]
            assert json.JSONDecoder().raw_decode(assignment_text)[0] == assignment
        recovery_messages = [message for message in messages if recovery in json.dumps(message)]
        assert len(recovery_messages) == (1 if index == 4 else 0)
        assert all(message['role'] in {'system', 'developer'} for message in recovery_messages)
    assert warning_requests == ([3] if allowance == 7 else [0] if allowance == 1 else [])
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    assert [event.id for event in starts] == [call_id for batch in batches for call_id, _, _ in batch]
    failures = [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error]
    assert failures == ['read-missing', 'failed-submit']
    assert completions['read-missing'] == f"ENOENT: no such file or directory, access '{tmp_path / 'missing.txt'}'"
    assert completions['failed-submit'].startswith('Validation failed for tool "structured_output":\n')
    assert completions['failed-submit'].endswith(
        'Received arguments:\n{\n  "verdict": "incomplete",\n  "findings": false\n}')
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == (output if submit else None)


async def test_native_pi_retry_refreshes_attempt_count_without_renewing_host_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart warning state resets without refunding successful or failed tool starts.

    The batch/history journey never restarts the CLI, so it cannot detect a warning flag
    leaking between attempts or a retry packet renewing the cumulative host allowance.
    """
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
        budget = ReviewInvestigationBudget.from_limits(ReviewLimits(60, 3, 3))
        result, _, reason = await run_agent(
            backend, tmp_path, 'Investigate the supplied source and submit the result.',
            phase=DaydreamPhase.DEEP, output_schema=_SCHEMA, read_only=True, persist_session=False,
            investigation_budget=budget, tool_call_budget=3, progress_callback=lambda _: None)

    assert reason is None and result == output
    assert budget.observed_tool_starts == 3 and budget.remaining_tool_calls == 0
    assert len(requests) == 3
    for index, request in enumerate(requests):
        notes = [message for message in request['messages']
                 if 'Final-submission budget warning:' in json.dumps(message)]
        assert len(notes) == (0 if index == 0 else 1)
        assert all(message['role'] in {'system', 'developer'} for message in notes)
        assert len([message for message in request['messages'] if message['role'] == 'user']) == 1
        assert 'Live native Pi attempt budget:' not in json.dumps(request['messages'])
    assert 'tool-call allocation is 3 calls' in json.dumps(requests[0]['messages'])
    assert 'tool-call allocation is 1 calls' in json.dumps(requests[2]['messages'])
    assert 'attempt-1-a' not in json.dumps(requests[2]['messages'])
