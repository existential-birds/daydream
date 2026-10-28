"""Host-enforced stage allocation through real review runs and native provider events."""
from __future__ import annotations

import json
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
from daydream.host_tools import HostToolBinding, HostToolCallBudgetExceeded, HostToolFatal, HostToolRejected
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
        admission_errors: dict[str, HostToolRejected | HostToolFatal] = {}
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
                    except (HostToolRejected, HostToolFatal) as exc:
                        admission_errors[event.id] = exc
                    if event.name == 'structured_output':
                        submissions[event.id] = deepcopy(event.input)
                elif isinstance(event, ToolResultEvent) and event.id in submissions and not event.is_error:
                    try:
                        if event.id in admission_errors:
                            raise admission_errors[event.id]
                        canonical = binding.submit(event.id, submissions[event.id])
                    except (HostToolRejected, HostToolFatal) as exc:
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
                        try:
                            canonical = binding.submit('default-submit', submitted)
                        except (HostToolRejected, HostToolFatal) as exc:
                            yield ToolResultEvent(id='default-submit', output=str(exc), is_error=True)
                        else:
                            yield ToolResultEvent(id='default-submit', output=str(canonical), is_error=False)
                        if binding.fatal_error is not None:
                            raise binding.fatal_error
                    event = replace(event, structured_output=binding.accepted_result)
                yield event
                if binding.fatal_error is not None:
                    if isinstance(event, ToolStartEvent):
                        yield ToolResultEvent(id=event.id, output=str(binding.fatal_error), is_error=True)
                    raise binding.fatal_error
            binding.native_settled = True


@pytest.mark.parametrize('fault', ['foreign-source', 'foreign-valid-source', 'invalid-judgment', 'not-reviewed-reason',
                                 'prior-admitted', 'handoff-bytes', 'foreign-contradiction',
                                 'duplicate-contradiction'])
async def test_native_correction_admits_only_the_valid_whole_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Transport feedback alone misses actionable domain corrections that preserve prior state and shared capacity."""
    from daydream.review_investigation import HANDOFF_MAX_BYTES, ReviewInvestigation

    triage = fault in {'foreign-contradiction', 'duplicate-contradiction'}
    prior_admitted = fault in {'prior-admitted', 'foreign-valid-source'}
    review = many_file_review(tmp_path, monkeypatch, count=9 if prior_admitted else 1, docs_count=0)
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
        if (prior_admitted and not stage['progress']) or (triage and stage['stage'] == 'first_pass'):
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            if triage:
                output['candidates'].append(candidate())
            yield ToolStartEvent(id='submit', name='structured_output', input=output)
            yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)
            return
        if checked:
            yield ToolStartEvent(id='submit', name='structured_output', input=output)
            yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)
            return
        checked = True
        if triage:
            output['candidates'] = [candidate(candidate_id=item['candidate_id'], disposition='rejected')
                                    for item in stage['candidates']]
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
        if not triage:
            bad['candidates'] = [candidate(disposition='confirmed', finding=dict(
                record(), description='Rejected defect'))]
        if fault == 'foreign-source':
            bad['targets'][0]['evidence_refs'][0]['source_id'] = 'source:' + '0' * 64
        elif fault == 'foreign-valid-source':
            # This handle belongs to the already admitted api.py assignment,
            # not this later target. A globally valid alias cannot widen scope.
            bad['targets'][0]['evidence_refs'][0]['source_id'] = 'source:1'
        elif fault in {'invalid-judgment', 'prior-admitted'}:
            bad['candidates'][0]['grounds'] = ''
        elif fault == 'not-reviewed-reason':
            bad['targets'][0].update(status='not_reviewed', reason='', evidence_refs=[])
        elif fault == 'handoff-bytes':
            bad['notes'] += 'x' * HANDOFF_MAX_BYTES
        elif triage:
            closed_id, = stage['closed_candidate_ids']
            bad['contradictions'] = ['foreign-candidate'] if fault == 'foreign-contradiction' else [closed_id] * 2
        try:
            yield ToolStartEvent(id='rejected-submit', name='structured_output', input=bad)
            yield ToolResultEvent(id='rejected-submit', output='Receipt.', is_error=False)
            assert binding.accepted is False and binding.possibly_accepted is False
            assert binding.fatal_error is None
            assert retained_state() == before, 'rejected plans must preserve all admitted state and shared capacity'
            if fault == 'handoff-bytes':
                assert investigation.assertion_failure_class == 'admission_failure'
            if not prior_admitted and not triage:
                output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            yield ToolStartEvent(id='corrected-submit', name='structured_output', input=output)
            yield ToolResultEvent(id='corrected-submit', output='Submitted.', is_error=False)
            assert binding.accepted is True
            assert binding.accepted_result is not None
            if triage:
                assert binding.accepted_result['targets'] == []
            else:
                assert binding.accepted_result['targets'][0]['evidence_refs'][0]['role'] == 'assigned'
            assert investigation.target_progress == expected
            assert sum(investigation.coverage._target_decision_bytes.values()) == tight_limit
            accepted = retained_state()
            # Callback replay returns the same canonical result without admitting again.
            assert binding.submit('corrected-submit', output) == binding.accepted_result
            assert retained_state() == accepted
        finally:
            monkeypatch.setattr(review_result, 'RUN_TARGET_DECISIONS_MAX_BYTES', retained_limit)

    data = await review.finish('python', findings=('Grounded defect',))
    assert checked
    python_stages = [event['metadata'] for event in stage_ends(review, 'python')]
    assert all(item['admitted'] for item in python_stages)
    corrected = python_stages[1 if prior_admitted or triage else 0]
    assert corrected['submission_starts'] == 2
    assert corrected['failed_submissions'] == 1
    assert corrected['successful_submissions'] == 1
    assert corrected['admitted_candidate_count'] == (2 if triage else 1)
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
    diagnostic = rejected_result['content']
    if fault in {'foreign-source', 'foreign-valid-source'}:
        assert 'targets[0]' in diagnostic and 'evidence_refs' in diagnostic and 'choose' in diagnostic
        if fault == 'foreign-valid-source':
            first = next(stage for stage in backend.stages if stage['scope_id'] == 'python')
            assert first['source_options']['api.py'][0]['source_id'] == 'source:1'
            assert 'source:1' in diagnostic and 'outside this target' in diagnostic
    elif fault in {'invalid-judgment', 'prior-admitted'}:
        assert 'candidates[0].grounds' in diagnostic and 'nonblank' in diagnostic and 'supply' in diagnostic
    elif fault == 'not-reviewed-reason':
        assert 'targets[0].reason' in diagnostic and 'not_reviewed' in diagnostic and 'supply' in diagnostic
    elif triage:
        assert 'contradictions' in diagnostic and 'unique previously closed_candidate_ids' in diagnostic
        assert 'remove duplicate or foreign IDs' in diagnostic
    else:
        assert str(HANDOFF_MAX_BYTES) in diagnostic and 'cumulative' in diagnostic and 'shorten' in diagnostic


@pytest.mark.parametrize('fault', ['corrupt-canonical', 'delete-canonical', 'target-capacity', 'handoff-capacity',
                                 'last-call-handoff-capacity',
                                 'handoff-byte-capacity', 'triage-byte-capacity', 'triage-byte-correction',
                                 'closed-contradiction'])
async def test_native_stage_capacity_and_input_failure_preserve_prior_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Native corrections cannot erase failed inputs, retained-capacity bounds or valid closed contradictions."""
    from daydream import review_investigation

    contradiction = fault == 'closed-contradiction'
    triage = fault.startswith('triage-') or contradiction
    correction = fault == 'triage-byte-correction'
    review = many_file_review(tmp_path, monkeypatch, count=1 if triage else 9, docs_count=0,
                             supporting_sources={'a': 'X\n'} if triage else None)
    backend = NativeStageBackend(review.repo)
    backend.fanout_concurrency = 1
    review.backend = backend
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '2')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    admitted: list[dict[str, Any]] = []
    bindings: list[HostToolBinding] = []
    unfinished: list[str] = []
    investigations: dict[str, review_investigation.ReviewInvestigation] = {}
    run_investigation = review_investigation.ReviewInvestigation.run

    async def observe_run(investigation: review_investigation.ReviewInvestigation, *args: Any,
                          **kwargs: Any) -> tuple[dict[str, Any], str | None]:
        investigations[investigation.scope_id] = investigation
        return await run_investigation(investigation, *args, **kwargs)

    monkeypatch.setattr(review_investigation.ReviewInvestigation, 'run', observe_run)

    @backend.script()
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        binding = backend.calls[-1]['host_tools']
        bindings.append(binding)
        if not admitted:
            admitted.extend({**target, 'evidence_refs': deepcopy(stage['source_refs'][target['target_id']][:1])}
                            for target in output['targets'])
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            if triage:
                output['candidates'].extend(candidate(file='a', line=1, trigger='x', consequence='x', grounds='x')
                                            for _ in range(8))
                if not contradiction:
                    output['notes'] = 'x' * (62_000 if correction else 62_800)
            if fault == 'handoff-byte-capacity':
                # One admitted large note leaves too little room for the next
                # stage's unavoidable host progress, even with no optional text.
                output['notes'] = 'x' * 63_500
            yield ToolStartEvent(id='accepted-submit', name='structured_output', input=output)
            yield ToolResultEvent(id='accepted-submit', output='Submitted.', is_error=False)
            return
        assert len(bindings) == 2, 'invalid frozen inputs must stop replacement and later-stage dispatch'
        unfinished.extend(stage['assigned_target_ids'])
        original = artifact_dir_for(review.repo, allow_standalone=True) / 'diff.patch'
        if fault == 'corrupt-canonical':
            original.write_bytes(original.read_bytes() + b'permanently changed input\n')
        elif fault == 'delete-canonical':
            original.unlink()
        elif fault == 'target-capacity':
            # Earlier publication remains valid, but no new required target ID
            # can fit the shared allocation, even without evidence or prose.
            allocated = investigations['python'].coverage._target_decision_bytes
            monkeypatch.setattr(review_result, 'RUN_TARGET_DECISIONS_MAX_BYTES', sum(allocated.values()))
        elif fault in {'handoff-capacity', 'last-call-handoff-capacity'}:
            monkeypatch.setattr(review_investigation, 'HANDOFF_MAX_ITEMS', 2)
            if fault == 'last-call-handoff-capacity':
                for index in range(stage['hard_tool_call_allowance'] - 1):
                    yield from read_events(review.repo, 'api.py', event_id=f'last-call-read-{index}')
        elif triage:
            assert stage['stage'] == 'triage'
            output.update(notes='', supporting_files=[], candidates=[candidate(
                candidate_id=item['candidate_id'], file='a', line=1, trigger='x', consequence='x', grounds='x',
                disposition='rejected' if contradiction else 'unresolved') for item in stage['candidates']])
            if contradiction:
                closed_id, = stage['closed_candidate_ids']
                output.update(notes='api.py:2 already returns a greeting accepted by callers.',
                              contradictions=[closed_id])
                yield ToolStartEvent(id='contradicted-submit', name='structured_output', input=output)
                yield ToolResultEvent(id='contradicted-submit', output='Submitted.', is_error=False)
                # A correctable rejection would let the provider erase its
                # counterexample and falsely declare this review complete.
                output['contradictions'] = []
                yield ToolStartEvent(id='cleared-submit', name='structured_output', input=output)
                yield ToolResultEvent(id='cleared-submit', output='Submitted.', is_error=False)
                return
            if correction:
                before = deepcopy(investigations['python'].handoff)
                bad = {**output, 'notes': 'OPTIONAL_TRIAGE_TEXT' * 300}
                yield ToolStartEvent(id='oversized-submit', name='structured_output', input=bad)
                yield ToolResultEvent(id='oversized-submit', output='Submitted.', is_error=False)
                assert not binding.accepted and not binding.possibly_accepted and binding.fatal_error is None
                assert investigations['python'].handoff == before
                yield ToolStartEvent(id='corrected-submit', name='structured_output', input=output)
                yield ToolResultEvent(id='corrected-submit', output='Submitted.', is_error=False)
                assert binding.accepted and binding.fatal_error is None
                return
        else:
            output.update(notes='', candidates=[], supporting_files=[])
        yield ToolStartEvent(id='fatal-submit', name='structured_output', input=output)
        yield ToolResultEvent(id='fatal-submit', output='Submitted.', is_error=False)
        pytest.fail('native continuation must stop after an unrecoverable host failure')

    assert await review.run() == (1 if fault == 'delete-canonical' else 0)
    data = review.load()
    scope = scopes(data)['python']
    assert scope['status'] == 'incomplete' and scope['partial_evidence'] is True
    assert scope['reason_codes'] == ['backend_failure' if fault.endswith('capacity') else 'evidence_incomplete']
    assert scope['target_decisions'] == admitted
    assert not set(unfinished).intersection(target['target_id'] for target in scope['target_decisions'])
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    assert len(bindings) == 2
    failed = bindings[-1]
    assert failed.authority_open is False
    if correction:
        assert failed.accepted and failed.fatal_error is None
    else:
        assert failed.accepted is False and failed.possibly_accepted is False
        fatal = failed.fatal_error
        assert fatal is not None and fatal.code == (
            'evidence_incomplete' if contradiction else
            'capacity_exhausted' if fault.endswith('capacity') else 'input_unavailable') and fatal.preacceptance is True
        if not fault.endswith('capacity') and not contradiction:
            assert type(fatal.__cause__).__name__ == 'SanctionedInputUnavailable'
            assert 'canonical-diff' in str(fatal)
        assert 'Changing the submission cannot resolve' in str(fatal)
        assert str(review.repo) not in str(fatal)
    if triage:
        retained = investigations['python'].handoff['candidates']
        assert retained[0]['disposition'] == 'confirmed' and len(retained) == 9
        assert all(item['disposition'] == ('unresolved' if correction else 'open') for item in retained[1:])
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert [item['admitted'] for item in metadata] == [True, correction]
    assert all(item['attempt'] == 1 for item in metadata)
    if fault == 'last-call-handoff-capacity':
        assert metadata[-1]['attempt_tool_starts'] == metadata[-1]['hard_tool_call_allowance']
        assert metadata[-1]['failure_class'] == 'backend_failure'
    if contradiction:
        assert metadata[-1]['stop_reason'] == 'evidence_incomplete'
        assert metadata[-1]['submission_starts'] == metadata[-1]['failed_submissions'] == 1
        assert metadata[-1]['observed_tool_starts'] == 2
        assert len(investigations['python'].handoff['notes']) == 1
        root = read_trajectory(review.tmp / 'trajectory.json')
        children = [read_trajectory(review.repo / '.daydream' / summary['sibling_trajectory_ref'])
                    for summary in root['extra']['subtrajectories'] if summary.get('sibling_trajectory_ref')]
        results = [result for child in children for step in child['steps']
                   for result in (step.get('observation') or {}).get('results', [])
                   if result['source_call_id'] == 'contradicted-submit']
        rejected, = results
        assert rejected['extra']['is_error'] is True
        assert 'source evidence contradicts a previously closed judgment' in rejected['content']
        assert 'Changing the submission cannot resolve' in rejected['content']
    persisted = saved_coverage(review.repo / '.daydream/deep').to_dict()['stack_outcomes']
    persisted_scope = next(item for item in persisted if item['scope_id'] == 'python')
    assert persisted_scope['target_decisions'] == admitted
    assert persisted_scope['status'] == scope['status'] and persisted_scope['reason_codes'] == scope['reason_codes']


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

    data = await review.finish('python', reason='backend_failure',
        findings=('Grounded defect',) if order == 'acceptance-first' else (), statuses=('failed', 'incomplete'))
    assert attempted == 1
    scope = scopes(data)['python']
    assert scope['partial_evidence'] is (order == 'acceptance-first')
    assert bool(scope['target_decisions']) is (order == 'acceptance-first')
    assert data['terminal_result']['analysis_state'] in {'incomplete', 'failed'}


@pytest.mark.parametrize('late_work', ['read-and-submit', 'inline', 'new-triage', 'overrun', 'retry',
                                     'rejected-last-call', 'unicode-last-call', 'parallel-open', 'parallel-open-only'])
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
            allowance - 1 if late_work in {'retry', 'rejected-last-call', 'unicode-last-call'} else 1)
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
        if not first and late_work == 'rejected-last-call':
            output['targets'][0]['evidence_refs'][0]['source_id'] = 'source:' + '0' * 64
        if not first and late_work == 'unicode-last-call':
            output['notes'] = '\ud800'
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
    elif late_work in {'rejected-last-call', 'unicode-last-call'}:
        assert len(allowances) == 2
        assert metadata[-1]['attempt_tool_starts'] == allowances[-1]
        assert metadata[-1]['remaining_tool_calls'] > 0
        assert metadata[-1]['admitted'] is False
        assert metadata[-1]['failure_class'] == 'quantitative_exhaustion'
        fatal = next(call['host_tools'].fatal_error for call in backend.calls
                     if call.get('host_tools') is not None and call['host_tools'].fatal_error is not None)
        assert isinstance(fatal, HostToolCallBudgetExceeded)
        if late_work == 'unicode-last-call':
            assert fatal.__cause__ is None
        coverage = saved_coverage(review.repo / '.daydream/deep')
        diagnostic = coverage.diagnostics['scopes']['python']
        detail = 'outside this target' if late_work == 'rejected-last-call' else 'valid UTF-8'
        assert 'no calls remain' in diagnostic and detail in diagnostic
        root = read_trajectory(review.tmp / 'trajectory.json')
        children = [read_trajectory(review.repo / '.daydream' / summary['sibling_trajectory_ref'])
                    for summary in root['extra']['subtrajectories'] if summary.get('sibling_trajectory_ref')]
        failures = [result for child in children for step in child['steps']
                    for result in (step.get('observation') or {}).get('results', [])
                    if result['source_call_id'] == 'submit' and result['extra']['is_error']]
        failure, = failures
        assert detail in failure['content'] and 'no calls remain' in failure['content']
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
