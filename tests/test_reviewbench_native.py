from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent, BackendExecutionInput, CostEvent, RetryPolicy, create_backend
from daydream.backends.pi import native_pi_settings
from daydream.benchmark.reviewbench.contract import Case
from daydream.benchmark.reviewbench.native import (
    convert_findings,
    freeze_reviewer,
    parse_native_config,
    reviewer_config,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.artifacts import diff_key
from daydream.review_result import FinalizedReview
from daydream.run_config import _resolved_model
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.dataset_hub import FakeDatasetHub
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, tracked_source_state, write_and_stage
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.reviewbench import finding, frozen_case, public_input
from tests.harness.stub_backend import review_stage_state, stage_result


@pytest.mark.parametrize('outcome,status', [
    ('empty', 'completed_empty'), ('finding', 'complete'), ('partial', 'incomplete'),
    ('incomplete', 'incomplete'),
    ('failed', 'failed'), ('relocated', 'conversion_limited'), ('pipeline_failure', 'incomplete'),
    ('divergent', 'completed_empty'), ('extension', 'completed_empty'),
])
async def test_admitted_review_preserves_terminal_authority_and_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, outcome: str, status: str,
    dataset_store: Path, dataset_hub: FakeDatasetHub, request: pytest.FixtureRequest,
) -> None:
    from daydream.backends.codex import CodexBackend
    from daydream.benchmark.reviewbench.harbor_entrypoint import main

    controls = {
        'DAYDREAM_PI_RETRY_ATTEMPTS': '0', 'DAYDREAM_PI_RETRY_BASE_DELAY_S': '0.25',
        'DAYDREAM_PI_RETRY_MAX_DELAY_S': '4', 'DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S': '0',
        'DAYDREAM_STREAM_IDLE_TIMEOUT_S': '17', 'DAYDREAM_CODEX_TERMINAL_DRAIN_TIMEOUT_S': '19',
        'DAYDREAM_CODEX_PROCESS_REAP_TIMEOUT_S': '3', 'DAYDREAM_CODEX_CLEANUP_TIMEOUT_S': '7',
    }
    for key, value in controls.items():
        monkeypatch.setenv(key, value)
    environment = public_input(tmp_path, divergent_base=outcome == 'divergent')
    for key in controls:
        monkeypatch.setenv(key, '1')
    if outcome == 'extension':
        request.getfixturevalue('ext_dir').write_module(
            'def register(registry):\n'
            '    original = registry.prompt("per-stack")\n'
            '    registry.override_prompt("per-stack", lambda **kw: original(**kw) + "\\nNATIVE_OVERRIDE")\n')
    repo, evidence = Path(environment['DAYDREAM_RB_REPOSITORY']), Path(environment['DAYDREAM_RB_EVIDENCE'])
    case = Case.from_dict(json.loads((Path(environment['DAYDREAM_RB_INPUTS']) / 'pr.json').read_text()))
    original = tracked_source_state(repo)
    protected = tmp_path / 'operator-archive'
    protected.mkdir()
    (protected / 'keep').write_text('operator history')
    monkeypatch.setenv('DAYDREAM_ARCHIVE_DIR', str(protected))
    environment['DAYDREAM_ARCHIVE_DIR'] = str(protected)
    active = peak = 0
    archives: set[Path] = set()
    prompts: list[str] = []
    class Backend(EmptyReviewBackend):
        async def execute(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                prompts.append(args[1])
                await asyncio.sleep(0.01)
                if outcome not in {'empty', 'divergent'}:
                    yield CostEvent(cost_usd=0.0, input_tokens=None, output_tokens=0, cached_tokens=None)
                async for event in super().execute(*args, **kwargs):
                    yield event
            finally:
                active -= 1

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert name == 'codex' and kwargs['execution_input'].fanout_concurrency == 1
        capability = kwargs['execution_input']
        assert capability.retry_policy == RetryPolicy(0, 0.25, 4, 0)
        assert capability.stream_idle_timeout_s == capability.pi_response_idle_timeout_s == 17
        assert (capability.codex_terminal_drain_timeout_s, capability.codex_process_reap_timeout_s,
                capability.codex_cleanup_timeout_s) == (19, 3, 7)
        child = capability.child_environment()
        assert not {'OPENAI_API_KEY', 'GH_TOKEN', 'OPENAI_BASE_URL', 'DAYDREAM_ARCHIVE_DIR'} & child.keys()
        assert not controls.keys() & child.keys()
        home = Path(child['CODEX_HOME'])
        archives.add(Path(os.environ['DAYDREAM_ARCHIVE_DIR']))
        auth = json.loads((home / 'auth.json').read_text())
        assert home != Path(environment['DAYDREAM_RB_CODEX_HOME']) and auth['auth_mode'] == 'chatgpt'
        backend = Backend(repo, forbid_merge=False, forbid_supervise=False,
            review_by_stack={'structure': [finding(line=500 if outcome == 'relocated' else 2)]}
                if outcome in {'finding', 'partial', 'incomplete', 'relocated', 'pipeline_failure'} else {},
            fail_stack='python' if outcome == 'partial' else None,
            incomplete_stack='python' if outcome == 'incomplete' else None,
            responder=lambda prompt: [RuntimeError('offline failure')] if (
                outcome == 'failed' and ('you are reviewing the ' in prompt.lower()
                    or 'you are the structural reviewer' in prompt.lower())
                or outcome == 'pipeline_failure' and 'supervisor adjudication' in prompt.lower()) else None)
        native_backend = CodexBackend(model=kwargs['model'], reasoning_effort=kwargs['reasoning_effort'],
            execution_input=capability)
        assert native_backend.retry_policy == RetryPolicy(0, 0.25, 4, 0)
        backend.fanout_concurrency = native_backend.fanout_concurrency
        backend.merge_echo_records = True
        return backend

    monkeypatch.setattr(runner, 'create_backend', create)
    code = await main(environment)
    result = json.loads((evidence / 'execution.json').read_text())
    successful = status in {'complete', 'completed_empty'}
    assert result['status'] == status and code == (0 if successful else 1), result['diagnostics']
    assert result['analyzed_comparison'] == {'head_sha': case.head,
        'merge_base_sha': git_ops.merge_base(repo, case.base, case.head),
        'diff_key': diff_key(git_ops.diff_paths(repo, case.base, case.head, ['.'], merge_base_diff=True))}
    assert result['terminal_result']['analyzed_revision']['head_sha'] == case.head
    assert result['task_id'] == 7 and result['evaluation_id'] == 99 and active == 0 and peak == 1
    assert (result['reviewer_backend'], result['reviewer_provider'], result['reviewer_authentication']) == (
        'codex', 'openai', 'chatgpt-oauth')
    assert result['usage'] == ({} if outcome in {'empty', 'divergent'} else {'completion_tokens': 0, 'cost_usd': 0.0})
    output = Path(environment['DAYDREAM_RB_OUTPUT'])
    assert output.exists() == successful
    if output.exists():
        candidate = json.loads(output.read_text())
        assert candidate['pr'] == asdict(case) and bool(candidate['findings']) == (outcome == 'finding')
        assert candidate.get('usage', {}) == result['usage']
        if not result['usage']:
            assert 'usage' not in candidate
        if source := os.environ.get('DAYDREAM_REVIEWBENCH_SOURCE'):
            checked = subprocess.run(['npm', 'run', 'harbor:validate-findings', '--', '--candidate', str(output),
                '--pr', str(Path(environment['DAYDREAM_RB_INPUTS']) / 'pr.json')], cwd=source,
                capture_output=True, text=True, timeout=60)
            assert checked.returncode == 0, checked.stdout + checked.stderr
            assert 'No judging or inference performed' in checked.stdout
    if outcome in {'finding', 'partial', 'incomplete', 'relocated', 'pipeline_failure'}:
        assert result['retained_findings'][0]['description'] == finding()['description']
    if outcome == 'relocated':
        assert result['conversion_omissions'][0]['reason'] == 'native_location_was_relocated_or_distrusted'
    assert any('Frozen supplied author explanation' in prompt for prompt in prompts)
    if outcome == 'extension':
        assert any('NATIVE_OVERRIDE' in prompt for prompt in prompts)
    assert archives and all(path != protected and not path.exists() for path in archives)
    assert os.environ['DAYDREAM_ARCHIVE_DIR'] == str(protected)
    assert (protected / 'keep').read_text() == 'operator history'
    assert list(protected.iterdir()) == [protected / 'keep']
    files = [path for path in evidence.rglob('*') if path.is_file()]
    assert all(path.name != 'auth.json' for path in files)
    retained = b''.join(path.read_bytes() for path in files)
    assert not any(value in retained for value in (b'private-refresh-canary', b'forbidden-api-canary'))
    assert tracked_source_state(repo) == original
    assert not dataset_store.exists() and dataset_hub.commits == []
    assert not fake_gh.calls('POST')


def test_frozen_native_profile_survives_removed_operator_files(tmp_path: Path) -> None:
    """Export owns serialization; ordinary profile resolution owns precedence and path admission."""
    operator = tmp_path / 'operator'
    operator.mkdir()
    profile = operator / 'policy.toml'
    profile.write_text('schema_version=1\nname="frozen-operator-policy"\n')
    (operator / '.daydream.toml').write_text('backend="codex"\n[phases.per_stack_review]\nmodel="phase-model"\n')
    frozen = freeze_reviewer(parse_native_config([str(operator), '--backend', 'codex',
        '--model', 'explicit-model', '--review-profile', str(profile), '--diagram', 'off']))
    assert frozen['file_config']['phases']['per_stack_review']['model'] == 'phase-model'
    profile.unlink()
    config = reviewer_config(json.loads(json.dumps(frozen)), tmp_path / 'reviewed-source')
    assert _resolved_model(config, 'per_stack_review') == 'explicit-model'
    assert config.review_profile is not None
    assert config.review_profile.source_kind == 'explicit'
    assert config.review_profile.profile.name == 'frozen-operator-policy'


@pytest.mark.parametrize('arguments', [
    ['--copy', 'extra.py'], ['--test-command', 'true'], ['--approve-on-clean'],
    ['--file-scope-issues'], ['--trace-to', 'otlp'],
])
def test_export_refuses_native_controls_without_an_admitted_runtime_capability(
    tmp_path: Path, arguments: list[str],
) -> None:
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


@pytest.mark.parametrize('phase_model,idle_override', [(None, None), ('native-intent-model', None), (None, '0')])
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
        if stage is None and '--extension' not in argv:
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


@pytest.mark.parametrize("fault", ["missing_terminal", "malformed_projection", "ignored_patch", "moved_head"])
async def test_native_artifact_failure_or_changed_comparison_never_becomes_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    from daydream.benchmark.reviewbench.harbor_entrypoint import main
    from daydream.benchmark.reviewbench.harbor_profile import canonical_digest

    environment = public_input(tmp_path)
    repo = Path(environment['DAYDREAM_RB_REPOSITORY'])
    case = Case.from_dict(json.loads((Path(environment['DAYDREAM_RB_INPUTS']) / 'pr.json').read_text()))
    original = tracked_source_state(repo)
    injected: list[str] = []
    real_read = Path.read_text
    real_replace = os.replace

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        contents = real_read(path, *args, **kwargs)
        if fault == "malformed_projection" and path.name == "merged-items.json":
            injected.append(path.name)
            return '{"items": "not canonical findings"}'
        return contents

    def replace_file(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        # atomic_write_json stages complete coverage bytes before installing them
        # into place; failure there occurs before the real terminal handoff.
        if fault == "missing_terminal" and Path(destination).name == "review-coverage.json":
            injected.append(Path(destination).name)
            raise OSError("synthetic terminal coverage installation failure")
        real_replace(source, destination, *args, **kwargs)

    def create(*_args: Any, **kwargs: Any) -> EmptyReviewBackend:
        checkout = kwargs.get("cwd", repo)
        if fault == "moved_head":
            git_ops.checkout_detach(checkout, case.base)
            injected.append("head")
        return EmptyReviewBackend(checkout)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "replace", replace_file)
    monkeypatch.setattr(runner, "create_backend", create)
    config = empty_review_config(repo, tmp_path / "trajectory", backend="codex",
        ignore_paths=["api.py"] if fault == "ignored_patch" else [])
    profile_path = Path(environment['DAYDREAM_RB_INPUTS']) / 'daydream-profile.json'
    profile = json.loads(profile_path.read_text())
    profile['reviewer'] = freeze_reviewer(config)
    profile['profile_sha256'] = canonical_digest({key: value for key, value in profile.items()
                                                  if key != 'profile_sha256'})
    profile_path.write_text(json.dumps(profile))
    assert await main(environment) == 1
    result = SimpleNamespace(**json.loads((Path(environment['DAYDREAM_RB_EVIDENCE']) / 'execution.json').read_text()))
    assert result.status == "failed", result.diagnostics
    assert result.candidate is None and result.diagnostics
    if fault != "moved_head":
        assert tracked_source_state(repo) == original
    if fault == "missing_terminal":
        assert injected and result.terminal_result is None
        assert "native reviewer did not produce a finalized terminal result" in result.diagnostics
    elif fault == "malformed_projection":
        assert injected and result.terminal_result is not None
        assert not result.terminal_result["projection_valid"]
    elif fault == "ignored_patch":
        assert result.terminal_result is not None, result.diagnostics
        assert any("comparison differs" in diagnostic for diagnostic in result.diagnostics)
    else:
        assert injected
        assert any("changed the frozen checkout revision" in diagnostic for diagnostic in result.diagnostics)


@pytest.mark.parametrize('consumer_fails', [False, True])
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
