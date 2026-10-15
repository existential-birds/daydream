"""Codex capacity recovery and batch isolation through the real runner and public transport."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent, create_backend
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes
from tests.deep_orchestrator.test_review_investigation import candidate, many_file_review, stage_ends
from tests.harness.codex_replay import make_mock_process
from tests.harness.fake_clock import FakeClock, patch_retry_sleep
from tests.harness.stub_backend import review_stage_state, stage_result


class CodexReviewProvider:
    """Answer stdin requests as public JSONL at the external process boundary."""

    def __init__(self, repo: Path, *, exhausted: bool, error: dict[str, Any] | None = None) -> None:
        self.fallback = EmptyReviewBackend(repo, forbid_merge=False, forbid_supervise=False)
        self.fallback.merge_echo_records = True
        self.exhausted = exhausted
        self.error = error
        self.batches: list[tuple[str, ...]] = []
        self.attempts: Counter[tuple[str, ...]] = Counter()
        self.models: list[str] = []
        self.stages: list[dict[str, Any]] = []

    async def response(self, process: MagicMock, argv: tuple[str, ...]) -> list[str]:
        prompt = process.stdin.write.call_args.args[0].decode()
        cwd = Path(argv[argv.index('--cd') + 1])
        stage = review_stage_state(prompt)
        output: dict[str, Any] | None = None
        text = ''
        if stage is not None:
            output = stage_result(stage)
            if stage['scope_id'] == 'python':
                self.stages.append(stage)
            if stage['scope_id'] == 'python' and stage['stage'] == 'first_pass':
                key = tuple(stage['assigned_target_ids'])
                if key not in self.batches:
                    self.batches.append(key)
                index = self.batches.index(key)
                self.attempts[key] += 1
                failed_index = 1 if self.exhausted else 0
                if index == failed_index and (self.exhausted or self.attempts[key] == 1):
                    return [json.dumps({'type': 'turn.failed', 'error': self.error or {
                        'message': 'Selected model is at capacity. Please try a different model.',
                        'codex_error_info': 'server_overloaded',
                    }})]
                if index in (0, 2):
                    path = stage['assigned_files'][0]
                    finding = dict(record(1), file=path, evidence=f'{path}:1',
                                   description='Early defect' if index == 0 else 'Later defect')
                    output['candidates'] = [candidate(file=path, line=1, disposition='confirmed',
                                                      grounds=f'{path}:1', finding=finding)]
        else:
            # Reuse existing provider fixtures for intent/synthesis while keeping
            # the production runner, agent, Codex parser and admission intact.
            async for event in self.fallback.execute(cwd, prompt):
                if isinstance(event, ResultEvent):
                    output = event.structured_output
                elif isinstance(event, TextEvent):
                    text += event.text
                else:
                    raise AssertionError(f'Unexpected fixture event: {type(event).__name__}')
        return [
            json.dumps({'type': 'item.completed', 'item': {
                'type': 'agent_message', 'id': 'reply', 'text': json.dumps(output) if output is not None else text,
            }}),
            json.dumps({'type': 'turn.completed', 'usage': {}}),
        ]

    async def spawn(self, *argv: str, **_kwargs: Any) -> MagicMock:
        self.models.append(argv[argv.index('--model') + 1])
        process = make_mock_process([])
        provider = self

        class Stdout:
            lines: Iterator[str] | None = None

            async def readline(self) -> bytes:
                if self.lines is None:
                    self.lines = iter(await provider.response(process, argv))
                return (line + '\n').encode() if (line := next(self.lines, None)) is not None else b''

        process.stdout = Stdout()
        return process


@pytest.mark.parametrize('exhausted', [False, True], ids=['recovered', 'middle-batch-exhausted'])
async def test_codex_overload_recovery_and_later_independent_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exhausted: bool,
) -> None:
    """Retry the native failure without losing findings or falsely completing failed targets."""
    seeded = many_file_review(tmp_path, monkeypatch, count=9, docs_count=0)
    review = ReviewRun(seeded.repo, tmp_path, monkeypatch)
    # ReviewRun supplies local GitHub metadata; backend construction remains real.
    monkeypatch.setattr(runner, 'create_backend', create_backend)
    review.config = replace(seeded.config, backend='codex', model='capacity-test-model')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '1')
    fake_clock = FakeClock().install(monkeypatch)
    delays = patch_retry_sleep(monkeypatch, fake_clock)
    monkeypatch.setattr('daydream.agent_retry._sample_retry_delay', lambda cap: cap)
    provider = CodexReviewProvider(review.repo, exhausted=exhausted)
    monkeypatch.setattr('daydream.backends._transport.asyncio.create_subprocess_exec', provider.spawn)

    assert await review.run() == 0
    data = review.load()
    assert len(provider.batches) >= 3
    failed_key = provider.batches[1 if exhausted else 0]
    assert provider.attempts[failed_key] == 2
    assert all(attempts == 1 for key, attempts in provider.attempts.items() if key != failed_key)
    assert delays == [1.0]
    assert set(provider.models) == {'capacity-test-model'}
    assert {finding['title'] for finding in data['findings']} == {'Early defect', 'Later defect'}
    scope = scopes(data)['python']
    assert scope['status'] == ('incomplete' if exhausted else 'complete')
    assert scope['reason_codes'] == (['backend_failure'] if exhausted else [])
    assert data['terminal_result']['analysis_state'] == ('incomplete' if exhausted else 'complete')
    terminal = stage_ends(review, 'python')
    assert sum(event['status'] == 'failed' for event in terminal) == int(exhausted)
    saved = json.loads((review.repo / '.daydream/deep/stack-python-records.json').read_text())
    assert bool(saved.get('incomplete')) is exhausted
    assert {finding['description'] for finding in saved['issues']} == {'Early defect', 'Later defect'}
    if exhausted:
        assert not set(failed_key).intersection(provider.stages[-1]['completed_target_ids'])
        coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
        assert 'CodexError: review invocation failed (backend_failure).' in coverage['diagnostics']['scopes']['python']


@pytest.mark.parametrize('process_exit', [False, True], ids=['turn-failed', 'process-exit'])
async def test_codex_permanent_public_failures_stop_later_independent_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_exit: bool,
) -> None:
    """Native permanent failures stop the scope while retaining an earlier admitted finding."""
    seeded = many_file_review(tmp_path, monkeypatch, count=13, docs_count=0)
    review = ReviewRun(seeded.repo, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'create_backend', create_backend)
    review.config = replace(seeded.config, backend='codex', model='capacity-test-model')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    class AuthenticationProvider(CodexReviewProvider):
        async def response(self, process: MagicMock, argv: tuple[str, ...]) -> list[str]:
            lines = await super().response(process, argv)
            if process_exit and json.loads(lines[0])['type'] == 'turn.failed':
                process.returncode = 1
                process.wait.return_value = 1
                return ['Error: authentication required. Run `codex login` to authenticate.']
            return lines

    provider = AuthenticationProvider(review.repo, exhausted=True, error={
        'message': 'authentication required. Run `codex login` to authenticate'})
    monkeypatch.setattr('daydream.backends._transport.asyncio.create_subprocess_exec', provider.spawn)

    assert await review.run() == 0
    data = review.load()
    assert len(provider.batches) == 2
    assert all(attempts == 1 for attempts in provider.attempts.values())
    assert [finding['title'] for finding in data['findings']] == ['Early defect']
    scope = scopes(data)['python']
    assert scope['status'] == 'incomplete' and scope['partial_evidence'] is True
    assert scope['reason_codes'] == ['authentication_failure']
    assert [event['metadata']['admitted'] for event in stage_ends(review, 'python')] == [True, False]
    coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    diagnostic = coverage['diagnostics']['scopes']['python']
    assert 'authentication_failure' in diagnostic and 'codex login' not in diagnostic
