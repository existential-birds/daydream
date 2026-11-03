"""Host-enforced stage allocation through real review runs and native provider events."""
from __future__ import annotations

import json
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
from daydream.host_tools import HostToolBinding, HostToolFatal, HostToolRejected
from daydream.prompt_budget import SanctionedInputUnavailable
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import (
    StagedBackend,
    candidate,
    many_file_review,
    stage_ends,
)
from tests.harness.review_result import saved_coverage
from tests.harness.trajectory import read_trajectory


class NativeStageBackend(StagedBackend, PiBackend):
    """Replace only Pi's external execution, retaining host transport planning."""

    supports_budget_preamble = False

    def __init__(self, repo: Path) -> None:
        PiBackend.__init__(self, model='mock-model')
        StagedBackend.__init__(self, repo)

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


async def test_native_correction_admits_only_the_valid_whole_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport feedback alone misses actionable domain corrections that preserve prior state and shared capacity."""
    from daydream.review_investigation import ReviewInvestigation

    review = many_file_review(tmp_path, monkeypatch, count=1, docs_count=0)
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
        bad['candidates'] = [candidate(disposition='confirmed', finding=dict(record(), description='Rejected defect'))]
        bad['targets'][0]['evidence_refs'][0]['source_id'] = 'source:' + '0' * 64
        try:
            yield ToolStartEvent(id='rejected-submit', name='structured_output', input=bad)
            yield ToolResultEvent(id='rejected-submit', output='Receipt.', is_error=False)
            assert binding.accepted is False and binding.possibly_accepted is False
            assert binding.fatal_error is None
            assert retained_state() == before, 'rejected plans must preserve all admitted state and shared capacity'
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

    data = await review.finish('python', findings=('Grounded defect',))
    assert checked
    corrected = stage_ends(review, 'python')[0]['metadata']
    assert corrected['admitted']
    assert (corrected['submission_starts'], corrected['failed_submissions'], corrected['successful_submissions'],
            corrected['admitted_candidate_count']) == (2, 1, 1, 1)
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
    diagnostic = rejected_result['content']
    assert diagnostic and len(diagnostic.encode()) <= 1024
    assert 'targets[0]' in diagnostic and 'evidence_refs' in diagnostic and 'choose' in diagnostic


async def test_native_stage_capacity_and_input_failure_preserve_prior_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native correction cannot erase a failed frozen input or the prior admitted stage."""
    review = many_file_review(tmp_path, monkeypatch, count=9, docs_count=0)
    backend = NativeStageBackend(review.repo)
    backend.fanout_concurrency = 1
    review.backend = backend
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '2')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    admitted: list[dict[str, Any]] = []
    bindings: list[HostToolBinding] = []
    unfinished: list[str] = []

    @backend.script()
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        bindings.append(backend.calls[-1]['host_tools'])
        if not admitted:
            admitted.extend({**target, 'evidence_refs': deepcopy(stage['source_refs'][target['target_id']][:1])}
                            for target in output['targets'])
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            yield ToolStartEvent(id='accepted-submit', name='structured_output', input=output)
            yield ToolResultEvent(id='accepted-submit', output='Submitted.', is_error=False)
            return
        assert len(bindings) == 2, 'invalid frozen inputs must stop replacement and later-stage dispatch'
        unfinished.extend(stage['assigned_target_ids'])
        (artifact_dir_for(backend.calls[-1]['cwd'], allow_standalone=True) / 'diff.patch').unlink()
        output.update(notes='', candidates=[], supporting_files=[])
        yield ToolStartEvent(id='fatal-submit', name='structured_output', input=output)
        yield ToolResultEvent(id='fatal-submit', output='Submitted.', is_error=False)
        pytest.fail('native continuation must stop after an unrecoverable host failure')

    assert await review.run() == 1
    data = review.load()
    scope = scopes(data)['python']
    assert scope['status'] == 'incomplete' and scope['partial_evidence'] is True
    assert scope['reason_codes'] == ['evidence_incomplete']
    assert scope['target_decisions'] == admitted
    assert not set(unfinished).intersection(target['target_id'] for target in scope['target_decisions'])
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    failed = bindings[-1]
    assert failed.authority_open is False
    assert failed.accepted is False and failed.possibly_accepted is False
    fatal = failed.fatal_error
    assert fatal is not None and fatal.preacceptance is True and fatal.code == 'input_unavailable'
    assert isinstance(fatal.__cause__, SanctionedInputUnavailable) and 'canonical-diff' in str(fatal)
    assert 'Changing the submission cannot resolve' in str(fatal)
    assert str(review.repo) not in str(fatal)
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert [item['admitted'] for item in metadata] == [True, False]
    assert all(item['attempt'] == 1 for item in metadata)
    persisted = saved_coverage(review.repo / '.daydream/deep').to_dict()['stack_outcomes']
    persisted_scope = next(item for item in persisted if item['scope_id'] == 'python')
    assert persisted_scope['target_decisions'] == admitted
    assert persisted_scope['status'] == scope['status'] and persisted_scope['reason_codes'] == scope['reason_codes']
