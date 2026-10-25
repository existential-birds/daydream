"""Public Pi failure evidence through real staged review and frozen record storage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from daydream.runner import run
from tests.deep_orchestrator.empty_synthesis_support import empty_review_config
from tests.deep_orchestrator.test_review_completion import record
from tests.harness.git_helpers import seed_feature_branch
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.review_result import merge_result
from tests.harness.stub_backend import review_stage_result, review_stage_state, silence
from tests.test_backend_pi import _capture_pi_subprocess


async def test_staged_pi_terminal_failure_survives_trajectory_and_frozen_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter's lost diagnostic must survive stage admission and the real run freeze."""
    repo = tmp_path / 'repo'
    before = {'api.py': "def hello():\n    return 'world'\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(7)})
    seed_feature_branch(repo, base=before, feature={name: body.replace('world', 'universe').replace('= 0', '= 1')
                                                 for name, body in before.items()})
    silence(monkeypatch)
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '0')
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(tmp_path / 'pi-agent'))
    terminal_message = 'Provider finish_reason: error'
    stages: list[dict[str, Any]] = []

    def response(argv: list[str]) -> MagicMock:
        prompt = Path(argv[-1][1:]).read_text()
        stage = review_stage_state(prompt)
        if stage is not None:
            stages.append(stage)
            if stage['scope_id'] == 'python' and stage['progress']:
                message = {'role': 'assistant', 'content': [], 'stopReason': 'error',
                           'rawStopReason': 'error', 'errorMessage': terminal_message,
                           'usage': {'input': 100, 'output': 10, 'cost': {'total': 0.01}}}
                return make_mock_process([json.dumps(row) for row in [
                    {'type': 'session', 'sessionId': 'failed-stage'}, {'type': 'agent_start'},
                    {'type': 'turn_start'}, {'type': 'message_start', 'message': {'role': 'assistant'}},
                    {'type': 'message_end', 'message': message}, {'type': 'turn_end', 'message': message},
                ]])
            payload = review_stage_result(prompt, [record()] if stage['scope_id'] == 'python' else [])
        elif '--extension' not in argv:
            return make_mock_process_from_fixture('simple_text.jsonl')
        elif 'cross-stack merge agent' in prompt.lower():
            payload = merge_result([dict(record(), lens='per-stack', source_uids=['python:1'])])
        elif 'supervisor adjudication' in prompt.lower():
            payload = {'verdicts': [{'id': 1, 'action': 'allow', 'reason': 'Confirmed by source',
                        'severity': None, 'confidence': None, 'description': None, 'rationale': None,
                        'evidence': None}]}
        else:
            payload = {'issues': []}
        assistant = {'role': 'assistant', 'content': [
            {'type': 'toolCall', 'id': 'submit', 'name': 'structured_output', 'arguments': payload}],
            'stopReason': 'toolUse'}
        tool = {'content': [{'type': 'text', 'text': 'Submitted.'}], 'details': payload}
        return make_mock_process([json.dumps(row) for row in [
            {'type': 'session', 'sessionId': 'healthy-stage'}, {'type': 'agent_start'}, {'type': 'turn_start'},
            {'type': 'message_end', 'message': assistant},
            {'type': 'tool_execution_start', 'toolCallId': 'submit', 'toolName': 'structured_output', 'args': payload},
            {'type': 'tool_execution_end', 'toolCallId': 'submit', 'toolName': 'structured_output',
             'isError': False, 'result': tool},
            {'type': 'message_end', 'message': {'role': 'toolResult', 'toolCallId': 'submit',
             'toolName': 'structured_output', 'isError': False, **tool}},
            {'type': 'turn_end', 'message': assistant}, {'type': 'agent_end'}, {'type': 'agent_settled'},
        ]])

    spawns: list[list[str]] = []
    _capture_pi_subprocess(monkeypatch, spawns, response)
    store = tmp_path / 'records'
    trajectory = tmp_path / 'trajectory.json'
    assert await run(empty_review_config(
        repo, trajectory, backend='pi', model='fixture-model', deep_shard_enabled=False,
        dataset_capture=True, dataset_store_path=store,
    )) == 0
    python_stages = [stage for stage in stages if stage['scope_id'] == 'python']
    assert len(python_stages) == 2 and python_stages[1]['progress']
    assert 'Grounded defect' in (repo / '.review-output.md').read_text()
    assert terminal_message not in (repo / '.review-output.md').read_text()
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    python_scope = next(scope for scope in coverage['stack_outcomes'] if scope['scope_id'] == 'python')
    assert python_scope['status'] == 'incomplete' and python_scope['reason_codes'] == ['backend_failure']
    assert all(scope['status'] == 'complete' for scope in coverage['stack_outcomes'] if scope['scope_id'] != 'python')
    root = json.loads(trajectory.read_text())
    refs = [ref for ref in root['extra']['subtrajectories'] if ref.get('descriptor') == 'deep-python']
    assert len(refs) == 1
    stage_ends = [event for event in refs[0]['phase_events']
                  if event['event'] == 'phase_end' and event['metadata'].get('review_scope_id') == 'python']
    assert [event['metadata']['admitted'] for event in stage_ends] == [True, False]
    child_path = repo / '.daydream' / refs[0]['sibling_trajectory_ref']
    child = json.loads(child_path.read_text())
    diagnostics = [(step, diagnostic) for step in child['steps']
                   for diagnostic in step.get('extra', {}).get('backend_diagnostics', [])]
    terminal_diagnostics = [(step, diagnostic) for step, diagnostic in diagnostics
                            if diagnostic['code'] == 'pi_terminal_error']
    assert len(terminal_diagnostics) == 1
    lifecycle_diagnostics = [diagnostic for _, diagnostic in diagnostics
                             if diagnostic['code'] == 'pi_host_tools_lifecycle']
    assert len(lifecycle_diagnostics) == 1
    lifecycle = lifecycle_diagnostics[0]['metadata']
    assert lifecycle['authority_closed'] and lifecycle['native_settled'] and lifecycle['endpoint_cleanup_complete']
    assert lifecycle['calls'] == 1 and lifecycle['rejections'] == 0
    step, diagnostic = terminal_diagnostics[0]
    assert diagnostic['message'] == terminal_message
    assert diagnostic['metadata']['category'] == 'UNKNOWN' and diagnostic['metadata']['retryable'] is False
    invocations = child['extra']['subtrajectories']
    assert len(invocations) == 2
    failed_lifecycle = invocations[1]['backend_lifecycle']
    assert failed_lifecycle['host_authority_closed'] and failed_lifecycle['host_cleanup_complete']
    assert failed_lifecycle['host_native_settled'] is False
    assert step['step_id'] in invocations[1]['step_ids']
    assert step['step_id'] not in invocations[0]['step_ids']
    failed_steps = [item for item in child['steps'] if item.get('extra', {}).get('error_subtype') == 'PiError']
    assert len(failed_steps) == 1 and failed_steps[0]['step_id'] in invocations[1]['step_ids']
    drafts = invocations[1]['generation_lifecycle']['drafts']
    assert diagnostic['metadata']['generation_id'] == drafts[0]['generation_id']
    assert invocations[1]['generation_lifecycle']['authoritative_total'] == {
        'input_tokens': 100, 'output_tokens': 10, 'cost_usd': 0.01,
    }
    frozen_paths = list((store / 'runs').glob('*.jsonl'))
    assert len(frozen_paths) == 1
    frozen = json.loads(frozen_paths[0].read_text())
    documents = frozen['trajectories']['value']['documents']
    frozen_child = next(document for document in documents if document['trajectory_id'] == child['trajectory_id'])
    assert frozen_child['steps'] == child['steps']
    assert terminal_message in json.dumps(frozen_child)
