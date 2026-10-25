"""Host-enforced stage allocation through real review runs and native provider events."""
from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncGenerator, Iterable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import review_result
from daydream.artifact_visibility import artifact_dir_for
from daydream.backends import AgentEvent, PiRequestConfig, RequestEvent, ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from daydream.host_tools import HostToolBinding, HostToolRejected
from daydream.review_investigation import HANDOFF_MAX_BYTES, ReviewInvestigation
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import (
    StagedBackend,
    candidate,
    many_file_review,
    read_events,
    stage_ends,
)
from tests.harness.review_result import saved_coverage
from tests.harness.trajectory import read_trajectory


class NativeStageBackend(StagedBackend, PiBackend):
    """Replace only Pi's external execution, retaining host transport planning."""

    supports_budget_preamble = False

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentEvent, None]:
        binding: HostToolBinding | None = kwargs.get('host_tools')
        if binding is None:
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
            return
        schema = args[0] if args else kwargs['output_schema']
        binding.configure_schema(schema)
        seen_request = False
        submissions: dict[str, Any] = {}
        admission_errors: dict[str, str] = {}
        async with binding.session():
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                if isinstance(event, RequestEvent):
                    seen_request = True
                if not seen_request:
                    yield RequestEvent(prompt=prompt, output_schema=schema,
                                       config=PiRequestConfig(schema_emulated=False, no_tools=False))
                    seen_request = True
                if isinstance(event, ToolStartEvent):
                    try:
                        binding.admit(event.id, event.name, event.input, terminal=event.name == 'structured_output')
                    except HostToolRejected as exc:
                        admission_errors[event.id] = str(exc)
                    if event.name == 'structured_output':
                        submissions[event.id] = deepcopy(event.input)
                elif isinstance(event, ToolResultEvent) and event.id in submissions and not event.is_error:
                    try:
                        if event.id in admission_errors:
                            raise HostToolRejected(admission_errors[event.id])
                        canonical = binding.submit(event.id, submissions[event.id])
                    except HostToolRejected as exc:
                        event = replace(event, is_error=True, output=str(exc))
                    else:
                        event = replace(event, output=str(canonical))
                elif isinstance(event, ResultEvent):
                    if not submissions:
                        # Default scripted provider follows the same terminal protocol.
                        submitted = event.structured_output
                        assert isinstance(submitted, dict)
                        binding.admit('default-submit', 'structured_output', submitted, terminal=True)
                        yield ToolStartEvent(id='default-submit', name='structured_output', input=submitted)
                        canonical = binding.submit('default-submit', submitted)
                        yield ToolResultEvent(id='default-submit', output=str(canonical), is_error=False)
                    event = replace(event, structured_output=binding.accepted_result)
                yield event
            binding.native_settled = True


@pytest.mark.parametrize('fault', ['foreign-source', 'invalid-judgment', 'tampered-original',
                                 'prior-admitted', 'handoff-bytes'])
async def test_native_correction_admits_only_the_valid_whole_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Native transport tests cannot prove rejected stage state and shared capacity stay untouched."""
    review = many_file_review(tmp_path, monkeypatch, count=9 if fault == 'prior-admitted' else 1, docs_count=0)
    backend = NativeStageBackend(review.repo)
    backend.fanout_concurrency = 1
    review.backend = backend
    checked = False
    investigations: dict[str, ReviewInvestigation] = {}
    run_investigation = ReviewInvestigation.run

    async def observe_run(investigation: ReviewInvestigation, *args: Any,
                          **kwargs: Any) -> tuple[dict[str, Any], str | None]:
        investigations[investigation.scope_id] = investigation
        return await run_investigation(investigation, *args, **kwargs)

    monkeypatch.setattr(ReviewInvestigation, 'run', observe_run)

    @backend.script()
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal checked
        if fault == 'prior-admitted' and not stage['progress']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            yield ToolStartEvent(id='submit', name='structured_output', input=output)
            yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)
            return
        if checked:
            yield ToolStartEvent(id='submit', name='structured_output', input=output)
            yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)
            return
        checked = True
        binding = backend.calls[-1]['host_tools']
        investigation = investigations['python']

        def retained_state() -> Any:
            return deepcopy((investigation.target_progress, investigation.handoff, investigation.admitted_stages,
                             investigation.coverage.to_dict(), investigation.coverage._target_decision_bytes))

        before = retained_state()
        # Count the corrected decisions in their actual published nesting. The
        # rejected plan has these same valid targets, so handoff rejection must
        # happen after a shared-capacity preview that fits exactly.
        expected = {**investigation.target_progress, **{
            target['target_id']: {**target, 'evidence_refs': deepcopy(stage['source_refs'][target['target_id']][:1])}
            for target in output['targets']}}
        envelope: dict[str, Any] = {'terminal_result': {'stack_outcomes': [{'scope_id': 'python'}]}}
        without_decisions = len(json.dumps(envelope, indent=2, ensure_ascii=False).encode())
        envelope['terminal_result']['stack_outcomes'][0]['target_decisions'] = [
            expected[key] for key in sorted(expected)]
        corrected_bytes = len(json.dumps(envelope, indent=2, ensure_ascii=False).encode()) - without_decisions
        allocated = investigation.coverage._target_decision_bytes
        retained_limit = review_result.RUN_TARGET_DECISIONS_MAX_BYTES
        tight_limit = sum(allocated.values()) - allocated['python'] + corrected_bytes
        monkeypatch.setattr(review_result, 'RUN_TARGET_DECISIONS_MAX_BYTES', tight_limit)
        bad = deepcopy(output)
        bad['notes'] = 'REJECTED SEMANTIC NOTE'
        bad['candidates'] = [candidate(disposition='confirmed', finding=dict(
            record(), description='Rejected defect'))]
        original_path: Path | None = None
        original_bytes: bytes | None = None
        original_times: tuple[int, int] | None = None
        if fault == 'foreign-source':
            bad['targets'][0]['evidence_refs'][0]['source_id'] = 'source:' + '0' * 64
        elif fault in {'invalid-judgment', 'prior-admitted'}:
            bad['candidates'][0]['grounds'] = ''
        elif fault == 'handoff-bytes':
            bad['notes'] += 'x' * HANDOFF_MAX_BYTES
        else:
            # Corrupt the canonical diff itself, leaving prepared copies intact.
            original_path = artifact_dir_for(review.repo, allow_standalone=True) / 'diff.patch'
            assert original_path.is_file()
            original_stat = original_path.stat()
            original_times = (original_stat.st_atime_ns, original_stat.st_mtime_ns)
            original_bytes = original_path.read_bytes()
            original_path.write_bytes(original_bytes + b'corrupt canonical input\n')
        try:
            yield ToolStartEvent(id='rejected-submit', name='structured_output', input=bad)
            yield ToolResultEvent(id='rejected-submit', output='Receipt.', is_error=False)
            assert binding.accepted is False and binding.possibly_accepted is False
            assert retained_state() == before, 'rejected plans must preserve all admitted state and shared capacity'
            if fault == 'handoff-bytes':
                assert investigation.assertion_failure_class == 'admission_failure'
            if original_path is not None:
                assert original_bytes is not None and original_times is not None
                original_path.write_bytes(original_bytes)
                os.utime(original_path, ns=original_times)
            if fault != 'prior-admitted':
                output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            yield ToolStartEvent(id='corrected-submit', name='structured_output', input=output)
            yield ToolResultEvent(id='corrected-submit', output='Submitted.', is_error=False)
            assert binding.accepted is True
            assert binding.accepted_result is not None
            assert binding.accepted_result['targets'][0]['evidence_refs'][0]['role'] == 'assigned'
            assert investigation.target_progress == expected
            assert sum(investigation.coverage._target_decision_bytes.values()) == tight_limit
            accepted = retained_state()
            # Callback replay returns the same canonical result without admitting again.
            assert binding.submit('corrected-submit', output) == binding.accepted_result
            assert retained_state() == accepted
        finally:
            monkeypatch.setattr(review_result, 'RUN_TARGET_DECISIONS_MAX_BYTES', retained_limit)
            if original_path is not None:
                assert original_bytes is not None and original_times is not None
                original_path.write_bytes(original_bytes)
                os.utime(original_path, ns=original_times)

    data = await review.finish('python', findings=('Grounded defect',))
    assert checked
    python_stages = [event['metadata'] for event in stage_ends(review, 'python')]
    assert all(item['admitted'] for item in python_stages)
    corrected = python_stages[1 if fault == 'prior-admitted' else 0]
    assert corrected['submission_starts'] == 2
    assert corrected['failed_submissions'] == 1
    assert corrected['successful_submissions'] == 1
    assert corrected['admitted_candidate_count'] == 1
    assert 'REJECTED SEMANTIC NOTE' not in str(data)
    assert all('REJECTED SEMANTIC NOTE' not in call['prompt'] for call in backend.calls)
    assert all('source_id' not in str(decision) for decision in scopes(data)['python']['target_decisions'])
    for path in (review.output, review.repo / '.review-output.md',
                 review.repo / '.daydream/deep/review-coverage.json'):
        assert 'REJECTED SEMANTIC NOTE' not in path.read_text()
    root = read_trajectory(review.tmp / 'trajectory.json')
    children = [read_trajectory(review.repo / '.daydream' / summary['sibling_trajectory_ref'])
                for summary in root['extra']['subtrajectories'] if summary.get('sibling_trajectory_ref')]
    rejected_steps = [step for child in children for step in child['steps']
                      if any(call['tool_call_id'] == 'rejected-submit' for call in step.get('tool_calls') or [])]
    assert len(rejected_steps) == 1
    assert 'REJECTED SEMANTIC NOTE' in json.dumps(rejected_steps[0])
    rejected_result, = [result for result in rejected_steps[0]['observation']['results']
                       if result['source_call_id'] == 'rejected-submit']
    assert rejected_result['extra']['is_error'] is True
    assert rejected_result['content'] and len(rejected_result['content'].encode()) <= 1024


@pytest.mark.parametrize('order', ['closure-first', 'acceptance-first'])
async def test_native_stage_authority_order_preserves_truthful_retained_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: str,
) -> None:
    """Binding lifecycle tests cannot prove accepted findings survive failed runner settlement."""
    review = many_file_review(tmp_path, monkeypatch, count=1, docs_count=0)
    backend = NativeStageBackend(review.repo)
    review.backend = backend
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '2')
    attempted = 0

    @backend.script()
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempted
        attempted += 1
        binding = backend.calls[-1]['host_tools']
        output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        if order == 'closure-first':
            binding.close_authority()
        yield ToolResultEvent(id='submit', output='Receipt.', is_error=False)
        if order == 'acceptance-first':
            assert binding.accepted
            binding.close_authority()
            class SettlementFailure(RuntimeError):
                retryable = True
            raise SettlementFailure('Native settlement failed after domain acceptance')

    reason = 'backend_failure' if order == 'acceptance-first' else 'evidence_incomplete'
    data = await review.finish('python', reason=reason,
        findings=('Grounded defect',) if order == 'acceptance-first' else (), statuses=('failed', 'incomplete'))
    assert attempted == 1
    scope = scopes(data)['python']
    assert scope['partial_evidence'] is (order == 'acceptance-first')
    assert bool(scope['target_decisions']) is (order == 'acceptance-first')
    assert data['terminal_result']['analysis_state'] in {'incomplete', 'failed'}


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
        bound = re.search(r'current invocation allocation: (\d+) tool calls', prompt)
        assert bound is not None
        allowance = int(bound[1])
        assert allowance == stage['hard_tool_call_allowance']
        assert 'Future-stage reserves are already excluded from this allocation; do not subtract them again.' in prompt
        assert prompt.count('current invocation allocation:') == 1
        assert 'Cumulative observed starts:' not in prompt
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
    assert alternatives['status'] == 'incomplete'
    assert alternatives['noop'] is False
    assert alternatives['reason_codes'] == scopes(data)['structure']['reason_codes']
    assert alternatives['usable_evidence'] is False
    persisted = saved_coverage(review.repo / '.daydream/deep')
    assert persisted.phases['alternatives'] == alternatives
    assert 'folded into Structure' in persisted.diagnostics['phases']['alternatives']
    assert persisted.diagnostics['scopes']['structure'] in persisted.diagnostics['phases']['alternatives']


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
            if len(triage_chunks) == 2:
                assert stage['hard_tool_call_allowance'] == 1
                assert 'current invocation allocation: 1 tool calls' in prompt
                assert 'When only submission capacity remains, submit truthful unfinished decisions.' in (
                    ' '.join(prompt.split()))
        else:
            reads = 1
            if not stage['progress']:
                output['candidates'] = [candidate(disposition='confirmed', finding=record())] + [
                    candidate(disposition='unresolved') for _ in range(9)]
        for index in range(reads):
            yield from read_events(review.repo, 'api.py', event_id=f'read-{index}')
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    data = await review.finish('python', reason='evidence_incomplete', findings=('Grounded defect',))
    assert data['terminal_result']['analysis_state'] == 'incomplete'
    assert scopes(data)['python']['partial_evidence'] is True
    assert scopes(data)['python']['reason_codes'] == ['evidence_incomplete']
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert [len(chunk) for chunk in triage_chunks] == [8, 1]
    assert all(item['admitted'] for item in metadata)
    assert metadata[-1]['hard_tool_call_allowance'] == metadata[-1]['attempt_tool_starts'] == 1
    assert metadata[-1]['observed_tool_starts'] == 48
