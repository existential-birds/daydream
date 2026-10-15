"""Exercise the packaged native output extension with the installed Pi CLI."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import PiRequestConfig, RequestEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.fake_clock import FakeClock
from tests.harness.otlp import _loopback_http_server, _QuietHTTPHandler

_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"verdict": {"type": "string"}, "findings": {"type": "array"}}, "required": ["verdict", "findings"]}


def _native_clocks(tmp: Path, patch: pytest.MonkeyPatch) -> Path:
    """Control independent child clocks at the real Node process boundary."""
    state = tmp / 'native-clock.json'
    state.write_text(json.dumps({'wall_ms': 2500, 'native_ms': 0}))
    preload = tmp / 'native-clock.cjs'
    preload.write_text('''
const fs = require('node:fs');
const wallNow = Date.now;
const nativeNow = process.hrtime.bigint;
const state = () => JSON.parse(fs.readFileSync(process.env.DAYDREAM_TEST_CLOCK, 'utf8'));
Date.now = () => wallNow() + state().wall_ms;
process.hrtime.bigint = () => nativeNow() + 900000000000000000n + BigInt(state().native_ms) * 1000000n;
''')
    patch.setenv('DAYDREAM_TEST_CLOCK', str(state))
    patch.setenv('NODE_OPTIONS', f'--require={preload}')
    return state


def _wall_estimate(text: str) -> tuple[int, int]:
    match = re.search(r'Wall estimate: (\d+)s elapsed since dispatch; (\d+)s remaining\.', text)
    assert match is not None, text
    return int(match[1]), int(match[2])


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
            validate_structured_output=native, tool_call_budget=None if native else 8, wall_budget_s=None)]

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


@pytest.mark.parametrize(('allowance', 'wall'), [
    (8, 'finite'), (None, 'finite'), (8, 'unbounded'), (7, 'unbounded'), (6, 'unbounded'),
    (None, 'unbounded'), (8, 'missing'), (8, 'malformed'), (8, 'backward-handoff'), (8, 'compacted'),
], ids=['finite', 'wall-only', 'ample', 'submission-only', 'exhausted', 'unbounded',
        'missing-wall', 'malformed-wall', 'backward-epoch-handoff', 'native-compaction'])
async def test_native_pi_live_budget_counts_batch_failures_and_replaces_generation_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowance: int | None, wall: str,
) -> None:
    if shutil.which('pi') is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    config = tmp_path / 'isolated-pi-config'
    config.mkdir()
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(config))
    for name in ('PI_API_KEY', 'NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    clocks = _native_clocks(tmp_path, monkeypatch)
    if wall == 'backward-handoff':
        clocks.write_text(json.dumps({'wall_ms': -60_000, 'native_ms': 0}))
    real_spawn = asyncio.create_subprocess_exec

    async def spawn(*args: Any, **kwargs: Any) -> Any:
        if args[0] == 'pi' and wall == 'compacted':
            compaction = tmp_path / 'compact.ts'
            # This public boundary appends a native CompactionEntry and refreshes
            # Pi's session projection; it does not merely filter request messages.
            compaction.write_text('''
export default function (pi) {
  let compacted = false;
  pi.on('turn_end', (event) => {
    if (compacted || event.message.stopReason !== 'stop') return;
    compacted = true;
    return {entries: [...event.entries, {type: 'compaction', firstKeptEntryId: null,
      summary: ''' + json.dumps('NATIVE_COMPACTION_MARKER. ' + prompt) + '''}], continue: true};
  });
}
''')
            args = (args[0], '--extension', str(compaction), *args[1:])
        # Malformed/missing handoffs must degrade at the packaged extension boundary.
        if args[0] == 'pi' and wall in {'missing', 'malformed'}:
            packet_path = Path(kwargs['env']['DAYDREAM_PI_OUTPUT_PACKET'])
            packet = json.loads(packet_path.read_text())
            if wall == 'missing':
                packet.pop('wall_time', None)
            else:
                packet['wall_time'] = {'remaining_s': '60', 'elapsed_s': -1, 'sent_unix_ms': None}
            packet_path.write_text(json.dumps(packet))
        return await real_spawn(*args, **kwargs)

    monkeypatch.setattr('daydream.backends._transport.asyncio.create_subprocess_exec', spawn)
    source = 'Captured source\n'
    (tmp_path / 'source.txt').write_text(source)
    prompt = 'Investigate the supplied source and submit the result.'
    requests: list[dict[str, Any]] = []
    output = {'verdict': 'complete', 'findings': []}
    batches: list[list[tuple[str, str, dict[str, Any]]]] = [
        [('read-a', 'read', {'path': 'source.txt'}), ('read-missing', 'read', {'path': 'missing.txt'}),
         ('read-b', 'read', {'path': 'source.txt'})],
        [],  # A prose-only turn resumes without another tool start.
        [('failed-submit', 'structured_output', {'verdict': 'incomplete', 'findings': False})],
        [('read-c', 'read', {'path': 'source.txt'}), ('read-d', 'read', {'path': 'source.txt'})],
        [('final-submit', 'structured_output', output)],
    ]

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            batch = batches[len(requests) - 1]
            # Native time advances, then regresses, then crosses the allowance.
            # Wall-clock jumps after the initial bridge must not renew the estimate.
            offset = [7000, 20_000, 16_000, 90_000, 90_000][len(requests) - 1]
            clocks.write_text(json.dumps({'wall_ms': -60_000 if len(requests) % 2 else 120_000,
                                         'native_ms': offset}))
            deltas: list[dict[str, Any]] = [
                {'choices': [{'index': 0, 'delta': {'role': 'assistant', 'tool_calls': [
                    {'index': index, 'id': call_id, 'type': 'function',
                     'function': {'name': name, 'arguments': json.dumps(arguments)}}
                    for index, (call_id, name, arguments) in enumerate(batch)]}, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
            ]
            if not batch:
                deltas[0]['choices'][0]['delta'] = {'role': 'assistant', 'content': 'The investigation continues.'}
                deltas[1]['choices'][0]['finish_reason'] = 'stop'
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
            read_only=True, persist_session=False, tool_call_budget=allowance,
            wall_budget_s=None if wall == 'unbounded' else 60)]

    assert len(requests) == 5
    estimates: list[tuple[int, int]] = []
    for index, (request, spent) in enumerate(zip(requests, [0, 3, 3, 4, 6], strict=True)):
        assert any(prompt in json.dumps(message) for message in request['messages'] if message['role'] == 'user')
        history = {message['tool_call_id']: message['content'] for message in request['messages']
                   if message['role'] == 'tool'}
        compacted = wall == 'compacted' and index >= 2
        previous = [call for batch in batches[2 if compacted else 0:index] for call in batch]
        assert set(history) == {call_id for call_id, _, _ in previous}
        if compacted:
            assert 'NATIVE_COMPACTION_MARKER' in json.dumps(request['messages'])
            assert 'read-a' not in json.dumps(request['messages'])
            schema = next(tool['function']['parameters'] for tool in request['tools']
                          if tool['function']['name'] == 'structured_output')
            assert schema == _SCHEMA
        for call_id, name, arguments in previous:
            if name == 'read' and arguments['path'] == 'source.txt':
                assert history[call_id] == source
        notes = [message for message in request['messages']
                 if 'Live native Pi attempt budget:' in json.dumps(message)]
        assert len(notes) == (0 if allowance is None and wall == 'unbounded' else 1)
        if notes:
            text = json.dumps(notes[0])
            assert len(text.encode()) < 1536
            assert 'observed' not in text and 'ledger' not in text
            if wall in {'finite', 'compacted'}:
                estimates.append(_wall_estimate(text))
                assert 'Epoch bridge then native monotonic' in text
                assert 'Host deadline/cancellation remain authoritative' in text
            else:
                assert 'Wall estimate:' not in text
                assert ('Wall allowance is unbounded' in text) is (wall == 'unbounded')
            if allowance is None:
                assert 'Tool-call allocation is uncapped' in text
                assert 'tool calls remaining' not in text
                remaining = None
            else:
                remaining = max(0, allowance - spent)
                assert f'This attempt has {remaining} tool calls remaining' in text
                assert 'Each individual call counts, including failures and parallel batch members' in text
                assert 'Use this refreshed count without recounting history' in text and f'from {allowance}' not in text
            if wall in {'finite', 'compacted'} and index == 4:
                assert 'Estimated wall allowance is exhausted; no extra time or submission capacity is granted' in text
                assert 'Unfinished assigned work remains incomplete' in text and 'Submit now' not in text
            elif remaining is None or remaining > 1:
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
    if estimates:
        assert 2 <= estimates[0][0] < 10 and 50 <= estimates[0][1] <= 57
        assert estimates[1][0] >= estimates[0][0] + 6
        assert estimates[1][1] <= estimates[0][1] - 6
        assert estimates[2][0] > estimates[1][0] and estimates[2][1] < estimates[1][1]
        assert [elapsed for elapsed, _ in estimates] == sorted(elapsed for elapsed, _ in estimates)
        remaining_values = [remaining for _, remaining in estimates]
        assert remaining_values == sorted(remaining_values, reverse=True)
        assert estimates[-1][1] == 0
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
    timer = FakeClock(monotonic_value=1000).install(monkeypatch)
    clocks = _native_clocks(tmp_path, monkeypatch)
    requests: list[dict[str, Any]] = []
    output = {'verdict': 'complete', 'findings': []}

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if len(requests) == 2:
                timer.advance(12)  # Earlier provider work remains charged across a real Pi restart.
                clocks.write_text(json.dumps({'wall_ms': 2500, 'native_ms': 0}))
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': {'message': '503 Service Unavailable'}}).encode())
                return
            clocks.write_text(json.dumps({'wall_ms': -60_000, 'native_ms': 7000}))
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
    estimates: list[tuple[int, int]] = []
    for request, remaining in zip(requests, [8, 6, 6, 5], strict=True):
        notes = [message for message in request['messages'] if 'Live native Pi attempt budget:' in json.dumps(message)]
        assert len(notes) == 1
        text = json.dumps(notes[0])
        assert f'This attempt has {remaining} tool calls remaining' in text
        assert 'observed' not in text and 'ledger' not in text and len(text.encode()) < 1536
        estimates.append(_wall_estimate(text))
    assert estimates[0][0] >= 2 and estimates[0][1] <= 57  # Child startup/bridge consumed capacity.
    assert estimates[1][1] <= estimates[0][1] - 6
    assert estimates[2][1] <= estimates[0][1] - 11  # Retry receives the original host deadline's remainder.
    assert estimates[3][1] <= estimates[2][1] - 6
    assert 'attempt-1-a' not in json.dumps(requests[2]['messages'])
    assert 'attempt-2-a' in json.dumps(requests[3]['messages'])
