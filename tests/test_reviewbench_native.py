from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest

from daydream import git_ops, runner
from daydream.backends import BackendExecutionInput, RetryPolicy, create_backend
from daydream.backends.pi import native_pi_settings
from daydream.benchmark.reviewbench.native import (
    convert_findings,
    freeze_reviewer,
    parse_native_config,
)
from daydream.config_file import DaydreamFileConfig
from daydream.review_result import FinalizedReview
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, tracked_source_state, write_and_stage
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.reviewbench import finding, frozen_case, public_input
from tests.harness.stub_backend import review_stage_state, stage_result


def test_export_refuses_native_controls_without_an_admitted_runtime_capability(tmp_path: Path) -> None:
    for arguments in (['--test-command', 'true'], ['--trace-to', 'otlp']):
        config = parse_native_config([str(tmp_path), '--backend', 'codex', *arguments])
        with pytest.raises(ValueError, match='unsupported benchmark native policy'):
            freeze_reviewer(config)


def test_head_conversion_retains_logical_records_and_exposes_invalid_locations(tmp_path: Path) -> None:
    repo, initial = frozen_case(tmp_path)
    # A later frozen PR includes a same-number replacement, unchanged context,
    # a pure-deleted file, and new-only lines beyond the old hunk's bounds.
    write_and_stage(repo, "deleted.py", "removed\n")
    base = commit(repo, "second base")
    write_and_stage(repo, "api.py", "def resolve(value):\n    return value[-1]\n# new\n# tail\n")
    git(repo, "rm", "deleted.py")
    head = commit(repo, "second candidate")
    case = replace(initial, base=base, head=head)
    items = [finding(id=1, line=1), finding(id=2, line=2), finding(id=3, line=3),
        finding(id=4, file="deleted.py", line=1), finding(id=5, line=0), finding(id=6, line=999),
        finding(id=7, line=3, location_cited_line=2), finding(id=8, file="../escape", line=1),
        finding(id=9, line=4, description="", rationale="", evidence="")]
    converted, omitted = convert_findings(items, repository=repo, case=case, producer="fixture",
        comparison=git_ops.diff_paths(repo, base, head, ["."]))
    assert [(row["start_line"], row["end_line"]) for row in converted] == [(1, 1), (2, 2), (3, 3)]
    assert all(row["producer"] == "fixture" and row["file"] == "api.py" for row in converted)
    assert {row["id"]: row["reason"] for row in omitted} == {
        4: "location_absent_from_head_regular_file",
        5: "missing_repository_relative_positive_location", 6: "location_absent_from_head_regular_file",
        7: "native_location_was_relocated_or_distrusted", 8: "missing_repository_relative_positive_location",
        9: "missing_finding_message"}


async def test_frozen_pi_defaults_refuse_case_thinking_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daydream.backends.pi import PiBackend, PiError

    repo, _ = frozen_case(tmp_path)
    write_and_stage(repo, '.pi/settings.json', '{"defaultThinkingLevel":"high"}')
    commit(repo, 'case-local thinking default')
    operator = tmp_path / 'operator'
    (operator / '.pi').mkdir(parents=True)
    (operator / '.pi/settings.json').write_text(
        '{"defaultProvider":"nous","defaultModel":"deepseek/deepseek-v4.1-flash"}')
    monkeypatch.delenv('PI_THINKING', raising=False)
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(tmp_path / 'pi-agent'))

    execution = BackendExecutionInput.from_environment(os.environ, backend='pi')
    execution = replace(execution, pi_settings=native_pi_settings(operator, agent_dir=execution.pi_agent_dir))
    backend = PiBackend(cwd=repo, execution_input=execution)
    with pytest.raises(PiError, match='Case Pi thinking conflicts with unavailable operator effort'):
        async for _ in backend.execute(repo, 'Review this change'):
            pytest.fail('case-local effort reached the Pi transport')


@pytest.mark.parametrize('phase_model,idle_override', [(None, '0')])
async def test_native_pi_retains_nous_provider_model_and_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
    phase_model: str | None, idle_override: str | None,
) -> None:
    from daydream.backends.pi import PiBackend
    from daydream.benchmark.reviewbench.harbor_entrypoint import main
    from daydream.benchmark.reviewbench.harbor_profile import canonical_digest

    if idle_override is None:
        monkeypatch.delenv('DAYDREAM_STREAM_IDLE_TIMEOUT_S', raising=False)
    else:
        monkeypatch.setenv('DAYDREAM_STREAM_IDLE_TIMEOUT_S', idle_override)
    retry_controls = {
        'DAYDREAM_PI_RETRY_ATTEMPTS': '0', 'DAYDREAM_PI_RETRY_BASE_DELAY_S': '0.25',
        'DAYDREAM_PI_RETRY_MAX_DELAY_S': '4', 'DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S': '0',
    }
    for key, value in retry_controls.items():
        if idle_override is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    expected_retry = RetryPolicy(20, 10, 120, None) if idle_override is None else RetryPolicy(0, 0.25, 4, 0)
    environment = public_input(tmp_path, additional_sources=3 if phase_model else 0)
    repo = Path(environment['DAYDREAM_RB_REPOSITORY'])
    agent_dir = tmp_path / 'operator-pi'
    agent_dir.mkdir()
    model = 'deepseek/deepseek-v4.1-flash'
    expected_models = {model, *([phase_model] if phase_model else [])}
    (agent_dir / 'models.json').write_text(json.dumps({'providers': {'nous': {
        'baseUrl': 'https://inference-api.nousresearch.com/v1', 'api': 'openai-completions',
        'models': [{'id': name, 'name': name, 'reasoning': True, 'contextWindow': 100000, 'maxTokens': 8192}
                   for name in sorted(expected_models)]}}}))
    auth_body = '{"nous":{"type":"api_key","key":"offline-file-auth-canary"}}'
    (agent_dir / 'auth.json').write_text(auth_body)
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(agent_dir))
    monkeypatch.setenv('PI_PROVIDER', 'nous')
    monkeypatch.setenv('NOUS_API_KEY', 'offline-native-auth-canary')
    monkeypatch.delenv('PI_THINKING', raising=False)
    captured: list[list[str]] = []
    expected_usage: list[tuple[int, int, int, float]] = []
    real_exec: Any = asyncio.create_subprocess_exec

    async def native_process(*args: object, **kwargs: Any) -> MagicMock:
        if not args or args[0] != 'pi':
            return cast(MagicMock, await real_exec(*args, **kwargs))
        argv = [str(arg) for arg in args]
        if argv[1:] == ["--version"]:
            return make_mock_process(["1.4.0"])
        captured.append(argv)
        assert Path(kwargs['env']['DAYDREAM_RB_PI_AUTH_PATH']).read_text() == auth_body
        delivered = json.loads((Path(kwargs['env']['PI_CODING_AGENT_DIR']) / 'models.json').read_text())
        assert {item['id'] for item in delivered['providers']['nous']['models']} == expected_models
        assert argv[argv.index('--provider') + 1] == 'nous'
        assert argv[argv.index('--thinking') + 1] == 'low'
        assert 'NOUS_API_KEY' not in kwargs['env']
        stage = review_stage_state(Path(argv[-1][1:]).read_text())
        output_packet = kwargs['env'].get('DAYDREAM_PI_OUTPUT_PACKET')
        output_schema = json.loads(Path(output_packet).read_text()).get('output_schema') if output_packet else None
        if stage is None and output_schema is None:
            expected_usage.append((110, 50, 10, 0.0003))
            return make_mock_process_from_fixture('simple_text.jsonl')
        expected_usage.append((13, 3, 2, 0.125))
        payload = stage_result(stage) if stage is not None else {'issues': []}
        call = {'type': 'toolCall', 'id': 'submit', 'name': 'structured_output', 'arguments': payload}
        assistant = {'role': 'assistant', 'content': [call], 'stopReason': 'toolUse',
            'usage': {'input': 11, 'output': 3, 'cacheRead': 2, 'cost': {'total': 0.125}}}
        admitted = payload
        ready = []
        if packet_path := kwargs['env'].get('DAYDREAM_PI_HOST_PACKET'):
            packet = json.loads(Path(packet_path).read_text())
            async with httpx.AsyncClient(trust_env=False, headers={'Authorization': packet['authorization']}) as client:
                response = await client.post(packet['url'] + '/ready',
                    json={'tool_name': packet['tool_name'], 'schema': packet['schema']})
                assert response.json() == {'ready': True}
                response = await client.post(packet['url'] + '/admit', json={
                    'id': 'submit', 'name': packet['tool_name'], 'arguments': payload, 'terminal': True})
                assert response.json()['allowed'] is True
                response = await client.post(packet['url'] + '/submit', json={'id': 'submit', 'arguments': payload})
                assert response.json()['accepted'] is True
                admitted = response.json()['result']
            ready = [{'type': 'daydream_host_tools_ready'}]
        tool = {'content': [{'type': 'text', 'text': 'Submitted.'}], 'details': admitted}
        return make_mock_process([json.dumps(row) for row in [
            {'type': 'session', 'sessionId': 'offline-fixture'}, *ready,
            {'type': 'agent_start'}, {'type': 'turn_start'},
            {'type': 'message_end', 'message': assistant},
            {'type': 'tool_execution_start', 'toolCallId': 'submit', 'toolName': 'structured_output', 'args': payload},
            {'type': 'tool_execution_end', 'toolCallId': 'submit', 'toolName': 'structured_output',
             'isError': False, 'result': tool},
            {'type': 'message_end', 'message': {'role': 'toolResult', 'toolCallId': 'submit',
             'toolName': 'structured_output', 'isError': False, **tool}},
            {'type': 'turn_end', 'message': assistant}, {'type': 'agent_end', 'messages': []},
            {'type': 'agent_settled'}]])

    monkeypatch.setattr('daydream.backends._transport.asyncio.create_subprocess_exec', native_process)
    config = empty_review_config(repo, tmp_path / 'trajectory', backend='pi',
        file_config=DaydreamFileConfig(model=model, reasoning_effort='low',
            phases={'intent': {'model': phase_model}} if phase_model else {}))
    profile_path = Path(environment['DAYDREAM_RB_INPUTS']) / 'daydream-profile.json'
    profile = json.loads(profile_path.read_text())
    profile['reviewer'] = freeze_reviewer(config)
    profile['profile_sha256'] = canonical_digest({key: value for key, value in profile.items()
                                                  if key != 'profile_sha256'})
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setenv('DAYDREAM_STREAM_IDLE_TIMEOUT_S', '6000')
    for key in retry_controls:
        monkeypatch.setenv(key, '1')
    real_create = create_backend

    def create(name: str, **kwargs: Any) -> PiBackend:
        backend = real_create(name, **kwargs)
        assert isinstance(backend, PiBackend)
        assert backend.retry_policy == expected_retry
        return backend

    monkeypatch.setattr(runner, 'create_backend', create)
    environment['DAYDREAM_RB_REVIEWER_HOME'] = str(agent_dir)
    assert await main(environment) == 0
    result = json.loads((Path(environment['DAYDREAM_RB_EVIDENCE']) / 'execution.json').read_text())
    assert result['status'] == 'completed_empty', result['diagnostics']
    assert (result['reviewer_backend'], result['reviewer_provider'], result['reviewer_authentication']) == (
        'pi', 'nous', 'native-pi')
    assert {argv[argv.index('--model') + 1] for argv in captured} == expected_models
    assert result['candidate']['usage'] == {'prompt_tokens': sum(row[0] for row in expected_usage),
        'completion_tokens': sum(row[1] for row in expected_usage),
        'cached_tokens': sum(row[2] for row in expected_usage),
        'cost_usd': pytest.approx(sum(row[3] for row in expected_usage)),
        'total_tokens': sum(row[0] + row[1] for row in expected_usage)}
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    root = json.loads((evidence / 'trajectory.json').read_bytes())
    documents = [root, *[json.loads((evidence / row['sibling_trajectory_ref']).read_bytes())
                        for row in root['extra']['subtrajectories'] if 'sibling_trajectory_ref' in row]]
    requests = [request for document in documents for invocation in document['extra']['subtrajectories']
                for request in invocation.get('requests', [])]
    lifecycles = [invocation['backend_lifecycle'] for document in documents
                  for invocation in document['extra']['subtrajectories'] if invocation.get('backend_lifecycle')]
    assert lifecycles
    assert {(row['tool_idle_timeout_s'], row['response_idle_timeout_s']) for row in lifecycles} == {
        (2700, 600) if idle_override is None else (0, 0)}
    assert {row['idle_detection_enabled'] for row in lifecycles} == {idle_override is None}
    assert {row['model'] for row in requests} == expected_models
    assert {row['provider'] for row in requests} == {'nous'}
    assert 'offline-file-auth-canary' not in json.dumps(result)
    assert 'offline-native-auth-canary' not in json.dumps(result)
    assert os.environ['PI_CODING_AGENT_DIR'] == str(agent_dir)
    assert (agent_dir / 'auth.json').read_text() == auth_body


async def test_ignored_patch_comparison_never_becomes_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """Ignoring the frozen patch changes the comparison; the run fails instead of publishing empty success."""
    from daydream.benchmark.reviewbench.harbor_entrypoint import main
    from daydream.benchmark.reviewbench.harbor_profile import canonical_digest

    environment = public_input(tmp_path)
    repo = Path(environment['DAYDREAM_RB_REPOSITORY'])
    original = tracked_source_state(repo)
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **kwargs: EmptyReviewBackend(kwargs.get("cwd", repo)))
    config = empty_review_config(repo, tmp_path / "trajectory", backend="codex", ignore_paths=["api.py"])
    profile_path = Path(environment['DAYDREAM_RB_INPUTS']) / 'daydream-profile.json'
    profile = json.loads(profile_path.read_text())
    profile['reviewer'] = freeze_reviewer(config)
    profile['profile_sha256'] = canonical_digest({key: value for key, value in profile.items()
                                                  if key != 'profile_sha256'})
    profile_path.write_text(json.dumps(profile))
    assert await main(environment) == 1
    result = SimpleNamespace(**json.loads((Path(environment['DAYDREAM_RB_EVIDENCE']) / 'execution.json').read_text()))
    assert result.status == "failed", result.diagnostics
    assert result.candidate is None and result.terminal_result is not None
    assert any("comparison differs" in diagnostic for diagnostic in result.diagnostics)
    assert tracked_source_state(repo) == original


@pytest.mark.parametrize('consumer_fails', [True])
async def test_result_observer_preserves_terminal_report_and_invocation_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, consumer_fails: bool,
) -> None:
    repo, case = frozen_case(tmp_path)
    received: list[str] = []
    backend = EmptyReviewBackend(repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={"structure": [finding()]})
    backend.merge_echo_records = True
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)

    def observe(result: FinalizedReview) -> None:
        received.append(result.terminal_result["run_id"])
        result.terminal_result["run_id"] = "observer-mutated-run"
        result.terminal_result["analysis_state"] = "failed"
        result.items[0]['description'] = 'observer-mutated-finding'
        result.usage['prompt_tokens'] = -1
        if consumer_fails:
            raise RuntimeError('consumer failed')

    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi", base=case.base,
                                 model="native-original-model", pr_number=None)
    if consumer_fails:
        with pytest.raises(RuntimeError, match='consumer failed'):
            await runner.run(config, review_result_handler=observe)
    else:
        assert await runner.run(config, review_result_handler=observe) == 0
    assert len(received) == 1
    coverage = json.loads((repo / ".daydream/deep/review-coverage.json").read_text())
    assert coverage["run_id"] == received[0]
    assert all(scope["status"] == "complete" for scope in coverage["stack_outcomes"])
    assert json.loads((tmp_path / 'trajectory').read_bytes())['final_metrics'].get('total_prompt_tokens') != -1
    if not consumer_fails:
        report = (repo / ".review-output.md").read_text()
        assert "observer-mutated" not in report and received[0] in report
        assert "<code>complete</code>" in report
    assert await runner.run(config) == 0
    assert len(received) == 1, "a later ordinary invocation must not inherit the observer"
