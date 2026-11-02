"""Public Pi failure evidence through real staged review and frozen record storage."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import FrameType
from typing import Any
from unittest.mock import MagicMock

import pytest

from daydream.agent import run_agent
from daydream.backends.pi import PiBackend, PiError
from daydream.review_stage_inputs import StageInputFactory
from daydream.runner import run
from daydream.trajectory import DaydreamPhase
from tests.deep_orchestrator.empty_synthesis_support import empty_review_config
from tests.deep_orchestrator.test_review_completion import record
from tests.harness.git_helpers import seed_feature_branch
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.review_result import merge_result
from tests.harness.stub_backend import review_stage_result, review_stage_state, silence
from tests.harness.trajectory import make_recorder
from tests.test_backend_pi import _capture_pi_subprocess


@pytest.mark.parametrize("failure_kind", ["native-error", "process-exit"])
async def test_pi_failure_records_redacted_process_tail_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str,
) -> None:
    """A real child preserves late stderr and distinguishes host cleanup from exit.

    Mock process replays cannot prove late merged output is drained or that
    exit/cleanup observations describe the actual process on both paths.
    The graceful child ignores repeated SIGTERM during interpreter shutdown
    so cleanup still observes its promised zero exit.
    """
    binary = tmp_path / "bin"
    binary.mkdir()
    child = binary / "pi"
    child.write_text(f"#!{sys.executable}\n" + '''
import json, signal, sys, time
def stop(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('late cleanup diagnostic', file=sys.stderr, flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
for index in range(25):
    print(f'early diagnostic {index}', file=sys.stderr)
print('TOKEN=private-stderr-canary; ' + 'safe ' * 1000, file=sys.stderr, flush=True)
print('-----BEGIN PRIVATE KEY-----', file=sys.stderr)
for index in range(30):
    print(f'private-pem-body-canary-{index}', file=sys.stderr)
''' + ("print('-----END PRIVATE KEY-----', file=sys.stderr, flush=True)\n" if failure_kind == "native-error" else "")
        + ('''print(json.dumps({'type': 'turn_end', 'message': {
    'role': 'assistant', 'stopReason': 'error', 'errorMessage': 'terminated',
    'usage': {'input': 10, 'output': 2, 'cost': {'total': 0.01}}}}), flush=True)
time.sleep(300)
''' if failure_kind == "native-error" else "raise SystemExit(7)\n"))
    child.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    backend = PiBackend(model="fixture-model")
    backend.retry_attempts = 0
    recorder = make_recorder(tmp_path)
    async with recorder:
        with pytest.raises(PiError) as failure:
            await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.REVIEW,
                            progress_callback=lambda _: None)
    assert failure.value.category == ("STREAM_DROP" if failure_kind == "native-error" else "PROCESS_EXIT")
    document = json.loads(recorder.path.read_text())
    diagnostic = next(item for step in document["steps"]
                      for item in step.get("extra", {}).get("backend_diagnostics", [])
                      if item["code"] == "pi_process_failure")
    metadata = diagnostic["metadata"]
    assert metadata["process_running_at_cleanup"] is (failure_kind == "native-error")
    assert metadata["process_returncode"] == (0 if failure_kind == "native-error" else 7)
    assert len(metadata["diagnostic_tail"]) == 20 and metadata["diagnostic_tail_truncated"] is True
    assert all(len(line) <= 1024 for line in metadata["diagnostic_tail"])
    assert "private-pem-body-canary" not in json.dumps(document)
    assert "private-stderr-canary" not in json.dumps(document)
    assert any("[REDACTED_ENV_VAR]" in line for line in metadata["diagnostic_tail"])
    assert any("[REDACTED_PEM_KEY]" in line for line in metadata["diagnostic_tail"])
    if failure_kind == "native-error":
        assert metadata["diagnostic_tail"][-1] == "late cleanup diagnostic"
        assert document["final_metrics"]["total_cost_usd"] == 0.01
    lifecycle = document["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert lifecycle["process_returncode"] == metadata["process_returncode"]
    assert lifecycle["teardown_started_s"] is not None and lifecycle["teardown_completed_s"] is not None
    assert lifecycle["stdout_eof_s"] is not None
    if failure_kind == "native-error":
        assert lifecycle["stdout_eof_s"] >= lifecycle["teardown_started_s"]


@pytest.mark.parametrize('failure_kind', ['provider', 'host-input'])
async def test_staged_pi_terminal_failure_survives_trajectory_and_frozen_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str,
) -> None:
    """Native and host diagnostics survive stage admission and the real run freeze.

    Direct native tests do not persist fatal host causes into frozen JSONL; the
    provider case cannot expose lost canonical-input evidence or continuation
    after a confirmed preacceptance host failure.
    """
    repo = tmp_path / 'repo'
    before = {'api.py': "def hello():\n    return 'world'\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(7)})
    seed_feature_branch(repo, base=before, feature={name: body.replace('world', 'universe').replace('= 0', '= 1')
                                                 for name, body in before.items()})
    silence(monkeypatch)
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '0')
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(tmp_path / 'pi-agent'))
    # Sibling stages read the same diff.patch, so concurrency could spend the fault outside python.
    monkeypatch.setenv('DAYDREAM_PI_FANOUT_CONCURRENCY', '1')
    terminal_message = 'Provider finish_reason: error'
    stages: list[dict[str, Any]] = []
    fail_next_canonical_open = False
    canonical_faults = 0
    real_open = os.open
    def python_stage_caller() -> bool:
        # Stacks share the canonical diff and revalidate concurrently; fault only the python stage's read.
        frame: FrameType | None = sys._getframe(2)
        while frame is not None:
            owner = frame.f_locals.get('self')
            if isinstance(owner, StageInputFactory):
                return owner.stack.stack_name == 'python'
            frame = frame.f_back
        return False
    def open_source(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal fail_next_canonical_open, canonical_faults
        if (fail_next_canonical_open and isinstance(path, (str, Path)) and Path(path).name == 'diff.patch'
                and python_stage_caller()):
            fail_next_canonical_open = False
            canonical_faults += 1
            raise OSError('Frozen canonical bytes unavailable at admission')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(os, 'open', open_source)

    def response(argv: list[str]) -> MagicMock:
        nonlocal fail_next_canonical_open
        prompt = Path(argv[-1][1:]).read_text()
        stage = review_stage_state(prompt)
        if stage is not None:
            stages.append(stage)
            if stage['scope_id'] == 'python' and stage['progress'] and failure_kind == 'provider':
                message = {'role': 'assistant', 'content': [], 'stopReason': 'error',
                           'rawStopReason': 'error', 'errorMessage': terminal_message,
                           'usage': {'input': 100, 'output': 10, 'cost': {'total': 0.01}}}
                return make_mock_process([json.dumps(row) for row in [
                    {'type': 'session', 'sessionId': 'failed-stage'}, {'type': 'agent_start'},
                    {'type': 'turn_start'},
                    {'type': 'daydream_provider_response', 'status': 200,
                     'headers': {'x-request-id': 'frozen-provider-request'}},
                    {'type': 'message_start', 'message': {'role': 'assistant'}},
                    {'type': 'message_end', 'message': message}, {'type': 'turn_end', 'message': message},
                ]])
            payload = review_stage_result(prompt, [record()] if stage['scope_id'] == 'python' else [])
            if stage['scope_id'] == 'python' and stage['progress'] and failure_kind == 'host-input':
                fail_next_canonical_open = True
        elif 'Submit the final result using structured_output alone as your final action.' not in prompt:
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
            'stopReason': 'toolUse', 'usage': {'input': 100, 'output': 10, 'cost': {'total': 0.01}}}
        tool = {'content': [{'type': 'text', 'text': 'Submitted.'}], 'details': payload}
        return make_mock_process([json.dumps(row) for row in [
            {'type': 'session', 'sessionId': 'healthy-stage'}, {'type': 'agent_start'}, {'type': 'turn_start'},
            {'type': 'message_start', 'message': {'role': 'assistant'}},
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
    assert python_scope['status'] == 'incomplete'
    assert python_scope['reason_codes'] == (
        ['backend_failure'] if failure_kind == 'provider' else ['evidence_incomplete'])
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
                            if diagnostic['code'] == ('pi_terminal_error' if failure_kind == 'provider'
                                                       else 'pi_host_tool_fatal')]
    assert len(terminal_diagnostics) == 1
    lifecycle_diagnostics = [diagnostic for _, diagnostic in diagnostics
                             if diagnostic['code'] == 'pi_host_tools_lifecycle']
    assert len(lifecycle_diagnostics) == (1 if failure_kind == 'provider' else 2)
    lifecycle = lifecycle_diagnostics[-1]['metadata']
    assert lifecycle['authority_closed'] and lifecycle['native_settled'] and lifecycle['endpoint_cleanup_complete']
    assert lifecycle['calls'] == 1 and lifecycle['rejections'] == 0
    step, diagnostic = terminal_diagnostics[0]
    if failure_kind == 'provider':
        assert diagnostic['message'] == terminal_message
        assert diagnostic['metadata']['category'] == 'UNKNOWN' and diagnostic['metadata']['retryable'] is False
        provider_response = next(item for _, item in diagnostics if item['code'] == 'pi_provider_response')
        assert provider_response['metadata'] == {'status': 200, 'headers': {'x-request-id': 'frozen-provider-request'},
            'generation_id': diagnostic['metadata']['generation_id'],
            'observed_at_unix_ns': provider_response['metadata']['observed_at_unix_ns']}
    else:
        assert canonical_faults == 1 and 'canonical-diff' in diagnostic['message']
        assert diagnostic['metadata']['code'] == 'input_unavailable' and diagnostic['metadata']['retryable'] is False
        assert diagnostic['metadata']['cause_type'] == 'SanctionedInputUnavailable'
        assert 'canonical-diff' in diagnostic['metadata']['cause_message']
    invocations = child['extra']['subtrajectories']
    assert len(invocations) == 2
    failed_lifecycle = invocations[1]['backend_lifecycle']
    assert failed_lifecycle['host_authority_closed'] and failed_lifecycle['host_cleanup_complete']
    assert failed_lifecycle['host_native_settled'] is (failure_kind == 'host-input')
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
    assert (terminal_message if failure_kind == 'provider' else 'canonical-diff') in json.dumps(frozen_child)
