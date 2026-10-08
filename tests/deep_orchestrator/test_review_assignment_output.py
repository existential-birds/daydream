"""Invocation schemas and terminal output faults at the public runner seam."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from daydream.backends import AgentEvent, ResultEvent, TextEvent
from tests.deep_orchestrator.test_review_capture_and_retry import read_source, source_review
from tests.deep_orchestrator.test_review_investigation import StagedBackend, candidate, stage_ends
from tests.harness.stub_backend import review_stage_state, stage_result


@pytest.mark.parametrize('extra_target', [False, True], ids=['compliant-triage', 'terminal-extra-target'])
async def test_discovery_candidates_reach_exact_empty_target_triage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_target: bool,
) -> None:
    review = source_review(tmp_path, monkeypatch, source_bytes=400, files=2)
    schema_results: list[bool] = []
    contracts: list[dict[str, Any]] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        for path in stage['assigned_files']:
            yield from read_source(review, path, f'{stage["stage"]}-{path}')
        body = (review.repo / 'api.py').read_text()
        assert "return 'universe'" in body
        schema = review.backend.calls[-1]['output_schema']
        contracts.append(stage.get('response_contract', {}))
        if stage['stage'] == 'first_pass':
            output['candidates'] = [candidate()]
            schema_results.append(Draft202012Validator(schema).is_valid(output))
            invalid = dict(output, candidates=[dict(output['candidates'][0], candidate_id='invented-native-id')])
            schema_results.append(Draft202012Validator(schema).is_valid(invalid))
        else:
            pending, = stage['candidates']
            assert stage['assigned_target_ids'] == [] and pending['candidate_id']
            output['candidates'] = [dict(pending, disposition='rejected',
                                        grounds='api.py:2 returns universe; no incompatible caller is present.')]
            if extra_target:
                output['targets'] = [{'target_id': 'api.py', 'status': 'reviewed', 'reason': ''}]
            schema_results.append(Draft202012Validator(schema).is_valid(output))

    review.backend.stage_response = response
    await review.finish('python', reason='malformed_output' if extra_target else None)
    assert schema_results == [True, False, not extra_target]
    assert len(contracts) == 2 and all(contract.get('schema') and contract.get('skeleton') for contract in contracts)
    assert contracts[-1]['skeleton']['targets'] == []
    phases = stage_ends(review, 'python')
    assert [event['metadata']['attempt'] for event in phases] == [1, 1]
    assert phases[-1]['metadata']['failure_class'] == ('assignment_mismatch' if extra_target else None)


@pytest.mark.parametrize('malformed', [False, True], ids=['compliant-json', 'terminal-extra-closing-brace'])
async def test_misplaced_closing_brace_is_terminal_syntax_failure_without_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: bool,
) -> None:
    review = source_review(tmp_path, monkeypatch, source_bytes=400, files=2)

    class TextBackend(StagedBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is None or stage['scope_id'] != 'python':
                async for event in super().execute(cwd, prompt, *args, **kwargs):
                    yield event
                return
            self.stages.append(stage)
            self.calls.append({'prompt': prompt, **kwargs})
            for path in stage['assigned_files']:
                for event in read_source(review, path, f'grounded-{path}'):
                    yield event
            output = stage_result(stage)
            text = (json.dumps({'targets': output['targets'], 'notes': 'Reviewed real source'})
                    + ', "candidates": [], "contradictions": []}' if malformed else json.dumps(output))
            yield TextEvent(text=text)
            yield ResultEvent(structured_output=None, continuation=None)

    review.backend = TextBackend(review.repo)
    await review.finish('python', reason='malformed_output' if malformed else None)
    phases = stage_ends(review, 'python')
    assert len(phases) == 1 and phases[0]['metadata']['attempt'] == 1
    assert phases[0]['metadata']['failure_class'] == ('syntax_failure' if malformed else None)
    assert phases[0]['metadata']['schema_rejection'] is None
