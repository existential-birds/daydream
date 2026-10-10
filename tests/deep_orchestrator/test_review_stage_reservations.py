"""Host-enforced stage allocation through real review runs and native provider events."""
from __future__ import annotations

import re
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, PiRequestConfig, RequestEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import (
    StagedBackend,
    candidate,
    many_file_review,
    read_events,
    stage_ends,
)


class NativeStageBackend(StagedBackend, PiBackend):
    """Replace only Pi's external execution, retaining host transport planning."""

    supports_budget_preamble = False

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


@pytest.mark.parametrize('late_work', ['read-and-submit', 'inline', 'new-triage', 'overrun', 'retry',
                                     'parallel-open', 'parallel-open-only'])
async def test_prior_borrowing_preserves_later_native_stage_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_work: str,
) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    backend = NativeStageBackend(review.repo)
    review.backend = backend
    if late_work == 'inline':
        backend.sandbox = True
        backend.merge_echo_records = False
        backend.merge_items = [dict(record(), lens='per-stack', source_uids=['python:1'])]
    allowances: list[int] = []
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')

    class TransportFailure(RuntimeError):
        retryable = True

    @backend.script()
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        prompt = backend.calls[-1]['prompt']
        bound = re.search(r'and (\d+) (?:remaining cumulative tool starts|tool starts for this invocation)', prompt)
        assert bound is not None
        allowance = int(bound[1])
        allowances.append(allowance)
        yield RequestEvent(prompt=prompt, output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        first = not stage['progress']
        parallel = late_work.startswith('parallel-')
        reads = allowance - 2 if first else allowance + 1 if late_work == 'overrun' or parallel else (
            allowance - 1 if late_work == 'retry' else 1)
        if not first and late_work == 'inline' and stage['remaining_work']['stages'] > 1:
            reads = 0
        if parallel and not first:
            output['candidates'] = [candidate(disposition='confirmed', finding=dict(
                record(), description='Unsubmitted defect'))]
            for index in range(reads):
                yield ToolStartEvent(id=f'read-{index}', name='Read',
                                     input={'file_path': stage['assigned_files'][0]})
            for index in range(reads):
                yield ToolResultEvent(id=f'read-{index}', output='Read completed.', is_error=False)
        else:
            for index in range(reads):
                yield from read_events(review.repo, stage['assigned_files'][0], event_id=f'read-{index}')
        if not first and late_work == 'retry':
            raise TransportFailure('provider disconnected after reads')
        if first:
            output['candidates'] = ([] if late_work == 'parallel-open-only' else
                                    [candidate(disposition='confirmed', finding=record())])
            if parallel:
                output['candidates'].append(candidate())
            if late_work == 'new-triage':
                output['candidates'].extend(candidate() for _ in range(17))
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    incomplete = late_work not in {'read-and-submit', 'inline'}
    data = await review.finish('python', reason='host_tool_budget_exhaustion' if incomplete else None,
                              findings=() if late_work == 'parallel-open-only' else ('Grounded defect',))
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert allowances[0] < 48 and allowances[0] > 8
    assert metadata[0]['admitted'] is True
    assert metadata[0]['hard_tool_call_allowance'] == allowances[0]
    if late_work == 'new-triage':
        assert len(allowances) == 1
        assert metadata[-1]['attempt_tool_starts'] == 0
        assert metadata[-1]['admitted'] is False
    elif late_work == 'overrun' or late_work.startswith('parallel-'):
        assert len(allowances) == 2
        assert metadata[-1]['attempt_tool_starts'] == allowances[-1] + 1
        assert metadata[-1]['observed_tool_starts'] < 48
        assert metadata[-1]['admitted'] is False
        if late_work.startswith('parallel-'):
            assert allowances[-1] <= 3
            assert metadata[-1]['submission_starts'] == metadata[-1]['successful_submissions'] == 0
            expected_candidates = 1 if late_work == 'parallel-open-only' else 2
            assert metadata[0]['admitted_candidate_count'] == metadata[-1]['admitted_candidate_count'] == (
                expected_candidates)
            assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == [
                'first_pass', 'first_pass']
    elif late_work == 'retry':
        assert len(allowances) == 2
        assert metadata[-1]['attempt_tool_starts'] == allowances[-1] - 1
        assert metadata[-1]['admitted'] is False
    else:
        assert len(allowances) == 5
        assert all(item['admitted'] for item in metadata)
        assert metadata[-1]['successful_submissions'] == 1
        assert scopes(data)['python']['files'] == sorted(path.name for path in review.repo.glob('*.py'))
    if incomplete:
        assert scopes(data)['python']['partial_evidence'] is True
        assert data['terminal_result']['analysis_state'] == 'incomplete'


async def test_initial_native_structure_exhaustion_keeps_unsent_work_and_folded_alternatives_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    backend = NativeStageBackend(review.repo)
    review.backend = backend

    @backend.script('python')
    def python_response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        yield RequestEvent(prompt=backend.calls[-1]['prompt'], output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        if not stage['progress']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    @backend.script('structure')
    def structure_response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        assert stage['stage'] == 'integration' and stage['folded_alternatives'] is True
        yield RequestEvent(prompt=backend.calls[-1]['prompt'], output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        output['candidates'] = [candidate(disposition='confirmed', finding=dict(
            record(), description='Unsubmitted structure defect'))]
        for index in range(47):
            yield from read_events(review.repo, 'api.py', event_id=f'read-{index}')
        for index in range(2):
            yield ToolStartEvent(id=f'grep-{index}', name='grep', input={'pattern': 'hello', 'path': 'api.py'})
        for index in range(2):
            yield ToolResultEvent(id=f'grep-{index}', output='hello', is_error=False)
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    data = await review.finish('structure', reason='host_tool_budget_exhaustion', findings=('Grounded defect',))
    event, = stage_ends(review, 'structure')
    metadata = event['metadata']
    assert event['status'] == 'failed' and metadata['logical_stage'] == 'integration'
    assert metadata['observed_tool_starts'] == 49 and metadata['hard_tool_call_allowance'] == 48
    assert metadata['submission_starts'] == 0
    assert metadata['successful_submissions'] == 0
    assert metadata['admitted'] is False and metadata['admitted_candidate_count'] == 0
    assert scopes(data)['structure']['partial_evidence'] is False
    alternatives, = [phase for phase in data['terminal_result']['phase_outcomes'] if phase['phase'] == 'alternatives']
    assert alternatives['status'] == 'failed'
    assert alternatives['reason_codes'] == scopes(data)['structure']['reason_codes']
    assert alternatives['usable_evidence'] is False


async def test_completed_unresolved_triage_chunk_does_not_reserve_another_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    backend = NativeStageBackend(review.repo)
    review.backend = backend
    triage_chunks: list[list[str]] = []

    @backend.script()
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        prompt = backend.calls[-1]['prompt']
        yield RequestEvent(prompt=prompt, output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        if stage['stage'] == 'triage':
            triage_chunks.append(stage['assigned_candidate_ids'])
            output['candidates'] = stage['candidates']
            # Spend the first chunk's allowance; only the reserved final submission remains.
            reads = stage['hard_tool_call_allowance'] - 1 if len(triage_chunks) == 1 else 0
        else:
            reads = 1
            if not stage['progress']:
                output['candidates'] = [candidate(disposition='confirmed', finding=record())] + [
                    candidate(disposition='unresolved') for _ in range(9)]
        for index in range(reads):
            yield from read_events(review.repo, 'api.py', event_id=f'read-{index}')
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    await review.finish('python', reason='evidence_incomplete', findings=('Grounded defect',))
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert [len(chunk) for chunk in triage_chunks] == [8, 1]
    assert all(item['admitted'] for item in metadata)
    assert metadata[-1]['hard_tool_call_allowance'] == metadata[-1]['attempt_tool_starts'] == 1
    assert metadata[-1]['observed_tool_starts'] == 48
