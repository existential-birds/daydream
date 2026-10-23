"""Native ReviewBench contracts using real Git and the ordinary runner."""
from __future__ import annotations

import asyncio
import errno
import json
import os
import subprocess
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent, BackendExecutionInput, CostEvent, EffectiveRequestConfig, RequestEvent
from daydream.backends.pi import native_pi_settings
from daydream.benchmark.reviewbench.contract import REVIEWBENCH_REVISION, Case
from daydream.benchmark.reviewbench.native import (
    CaseExecution,
    PublicHarborTask,
    convert_findings,
    freeze_reviewer,
    parse_native_config,
    review_case,
    reviewer_config,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.artifacts import diff_key
from daydream.github_app import GitHubExecutionInput
from daydream.review_result import FinalizedReview, deliver_finalized_review
from daydream.run_config import RunConfig, _resolved_backend_name
from daydream.run_context import InteractionPolicy, RunContext, bind_run_context
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.dataset_hub import FakeDatasetHub
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git, tracked_source_state, write_and_stage
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.review_result import terminal_result
from tests.harness.reviewbench import finding, frozen_case, public_input
from tests.harness.stub_backend import review_stage_state, stage_result


def _execution_input(config: RunConfig) -> runner.RunnerExecutionInput:
    backend = _resolved_backend_name(config, 'per_stack_review')
    environment = dict(os.environ)
    if environment.get('PI_CODING_AGENT_DIR'):
        environment['PI_CODING_AGENT_DIR'] = str(Path(environment['PI_CODING_AGENT_DIR']).resolve())
    native = BackendExecutionInput.from_environment(environment, backend=backend)
    if backend == 'pi':
        native = replace(native, pi_settings=native_pi_settings(Path(config.target or Path.cwd()),
                                                               agent_dir=native.pi_agent_dir))
    return runner.RunnerExecutionInput(native, GitHubExecutionInput(), backends={backend: native})


async def _review(repo: Path, case: Case, config: RunConfig, *,
                  execution_input: runner.RunnerExecutionInput | None = None) -> CaseExecution:
    task = PublicHarborTask(case, REVIEWBENCH_REVISION,
        git_ops.diff_paths(repo, git_ops.merge_base(repo, case.base, case.head) or case.base, case.head, ['.']),
        7, 99, 'a' * 64)
    return await review_case(task, repo, config, execution_input=execution_input or _execution_input(config),
        evidence_directory=repo.parent / 'native-evidence')


@pytest.mark.parametrize('outcome,status', [
    ('empty', 'completed_empty'), ('finding', 'complete'), ('partial', 'incomplete'),
    ('incomplete', 'incomplete'),
    ('failed', 'failed'), ('relocated', 'conversion_limited'), ('pipeline_failure', 'incomplete'),
    ('divergent', 'completed_empty'),
])
async def test_admitted_review_preserves_terminal_authority_and_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, outcome: str, status: str,
    dataset_store: Path, dataset_hub: FakeDatasetHub,
) -> None:
    """One real admitted runner journey owns terminal, public-output and C5 policy."""
    from daydream.artifacts import external
    from daydream.backends import TextEvent, ToolResultEvent, ToolStartEvent
    from daydream.backends.codex import CodexBackend
    from daydream.benchmark.reviewbench.harbor_entrypoint import main
    from daydream.trajectory import run_directory, run_document_path

    environment = public_input(tmp_path, divergent_base=outcome == 'divergent')
    repo, evidence = Path(environment['DAYDREAM_RB_REPOSITORY']), Path(environment['DAYDREAM_RB_EVIDENCE'])
    case = Case.from_dict(json.loads((Path(environment['DAYDREAM_RB_INPUTS']) / 'pr.json').read_text()))
    original = tracked_source_state(repo)
    protected = tmp_path / 'operator-archive'
    protected.mkdir()
    (protected / 'keep').write_text('operator history')
    monkeypatch.setenv('DAYDREAM_ARCHIVE_DIR', str(protected))
    environment['DAYDREAM_ARCHIVE_DIR'] = str(protected)
    active = peak = 0
    homes: set[Path] = set()
    archives: set[Path] = set()
    checkouts: set[Path] = set()
    prompts: list[str] = []
    exchange = external._AtomicNameExchange.call

    def unavailable_exchange(binding: Any, parent_fd: int, staged: str, target: str) -> Any:
        if evidence.exists() and os.fstat(parent_fd).st_ino == evidence.stat().st_ino:
            return external._NameExchangeResult(-1, errno.ENOTSUP)
        return exchange(binding, parent_fd, staged, target)

    monkeypatch.setattr(external._AtomicNameExchange, 'call', unavailable_exchange)

    @dataclass(frozen=True)
    class PrivateConfig(EffectiveRequestConfig):
        arbitrary: str = 'unstructured-private-canary'

    class Backend(EmptyReviewBackend):
        async def execute(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                checkout = args[0]
                checkouts.add(checkout)
                prompts.append(args[1])
                assert checkout != repo and not git_ops.remote_urls(checkout)
                assert not git_ops.list_local_branches(checkout) and git_ops.commit_exists(checkout, case.base)
                await asyncio.sleep(0.01)
                if outcome not in {'empty', 'divergent'}:
                    yield RequestEvent(prompt='private request', config=PrivateConfig())
                    yield CostEvent(cost_usd=0.0, input_tokens=None, output_tokens=0, cached_tokens=None)
                yield TextEvent('retained public activity')
                yield ToolStartEvent(id='read', name='Read', input={'file_path': 'api.py'})
                yield ToolResultEvent(id='read', output='retained public tool result', is_error=False)
                async for event in super().execute(*args, **kwargs):
                    yield event
            finally:
                active -= 1

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert name == 'codex' and kwargs['execution_input'].fanout_concurrency == 1
        child = kwargs['execution_input'].child_environment()
        assert not {'OPENAI_API_KEY', 'GH_TOKEN', 'OPENAI_BASE_URL', 'DAYDREAM_ARCHIVE_DIR'} & child.keys()
        home = Path(child['CODEX_HOME'])
        homes.add(home)
        archives.add(Path(os.environ['DAYDREAM_ARCHIVE_DIR']))
        auth = json.loads((home / 'auth.json').read_text())
        assert home != Path(environment['DAYDREAM_RB_CODEX_HOME']) and auth['auth_mode'] == 'chatgpt'
        auth['tokens']['refresh_token'] = 'private-rotated-canary'
        (home / 'auth.json').write_text(json.dumps(auth))
        backend = Backend(repo, forbid_merge=False, forbid_supervise=False,
            review_by_stack={'structure': [finding(line=500 if outcome == 'relocated' else 2)]}
                if outcome in {'finding', 'partial', 'incomplete', 'relocated', 'pipeline_failure'} else {},
            fail_stack='python' if outcome == 'partial' else None,
            incomplete_stack='python' if outcome == 'incomplete' else None,
            responder=lambda prompt: [RuntimeError('offline failure')] if (
                outcome == 'failed' and ('you are reviewing the ' in prompt.lower()
                    or 'you are the structural reviewer' in prompt.lower())
                or outcome == 'pipeline_failure' and 'supervisor adjudication' in prompt.lower()) else None)
        backend.fanout_concurrency = CodexBackend(model=kwargs['model'], reasoning_effort=kwargs['reasoning_effort'],
            execution_input=kwargs['execution_input']).fanout_concurrency
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
    assert result['configuration']['phases']['per_stack_review']['model'] == 'gpt-6.1-sol'
    assert result['usage'] == ({} if outcome in {'empty', 'divergent'} else {'completion_tokens': 0, 'cost_usd': 0.0})
    if outcome not in {'empty', 'divergent'}:
        assert result['configuration']['observed_requests']
        assert all(row['controls'] == {} for row in result['configuration']['observed_requests'])
    assert 'unstructured-private-canary' not in json.dumps(result)
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
    root = json.loads((evidence / 'trajectory.json').read_text())
    bundle = run_directory(evidence, root['session_id'])
    assert (bundle / 'manifest.json').is_file() and (bundle / 'diff.patch').is_file()
    assert run_document_path(bundle).read_bytes() == (
        evidence.parent / 'trajectory.json').read_bytes() == (evidence / 'trajectory.json').read_bytes()
    siblings = [fork for fork in root['extra']['subtrajectories'] if 'sibling_trajectory_ref' in fork]
    assert {fork['descriptor'] for fork in siblings} == {'deep-python', 'deep-structure'}
    for fork in siblings:
        content = (evidence / fork['sibling_trajectory_ref']).read_text()
        child = json.loads(content)
        assert child['session_id'] == root['session_id'] and child['trajectory_id'] == fork['trajectory_id']
        assert 'retained public activity' in content and 'retained public tool result' in content
    assert any('Frozen supplied author explanation' in prompt for prompt in prompts)
    assert checkouts and all(not checkout.exists() for checkout in checkouts)
    assert homes and all(not home.exists() for home in homes)
    assert archives and all(path != protected and not path.exists() for path in archives)
    assert os.environ['DAYDREAM_ARCHIVE_DIR'] == str(protected)
    assert (protected / 'keep').read_text() == 'operator history'
    assert list(protected.iterdir()) == [protected / 'keep']
    assert json.loads((Path(environment['DAYDREAM_RB_CODEX_HOME']) / 'auth.json').read_text())[
        'tokens']['refresh_token'] == 'private-rotated-canary'
    files = [path for path in evidence.rglob('*') if path.is_file()]
    assert all(path.name != 'auth.json' for path in files)
    retained = b''.join(path.read_bytes() for path in files)
    assert not any(value in retained for value in (b'private-refresh-canary', b'private-rotated-canary',
                                                   b'forbidden-api-canary', b'unstructured-private-canary'))
    assert tracked_source_state(repo) == original and not (repo / '.daydream').exists()
    assert not dataset_store.exists() and dataset_hub.commits == []
    assert not fake_gh.calls('POST')


@pytest.mark.parametrize('profile_source', ['explicit', 'env', 'repo', 'default', 'escape', 'symlink'])
async def test_frozen_native_profile_preserves_operator_precedence_and_confines_repository_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, profile_source: str,
) -> None:
    """Frozen CLI/env/repository precedence survives removed operator files; profile paths stay confined."""
    from daydream.review_profile import ProfileError

    repo, case = frozen_case(tmp_path)
    root = tmp_path / 'operator'
    root.mkdir()
    monkeypatch.delenv('DAYDREAM_REVIEW_PROFILE', raising=False)
    for path, name in ((root / 'repo.toml', 'repository-policy'),
                       (tmp_path / 'env.toml', 'environment-policy'),
                       (tmp_path / 'explicit.toml', 'explicit-policy')):
        path.write_text(f'schema_version=1\nname="{name}"\n')
    selected = '../env.toml' if profile_source == 'escape' else 'repo.toml'
    if profile_source == 'symlink':
        (root / 'linked.toml').symlink_to(tmp_path / 'env.toml')
        selected = 'linked.toml'
    (root / '.daydream.toml').write_text('backend="codex"\nsupervisor="llm"\n'
        + ('' if profile_source == 'default' else f'review_profile="{selected}"\n')
        + '[phases.per_stack_review]\nbackend="codex"\nmodel="phase-model"\n')
    if profile_source in {'explicit', 'env'}:
        monkeypatch.setenv('DAYDREAM_REVIEW_PROFILE', str(tmp_path / 'env.toml'))
    arguments = [str(root), '--backend', 'codex', '--model', 'explicit-model', '--diagram', 'off']
    if profile_source == 'explicit':
        arguments += ['--review-profile', str(tmp_path / 'explicit.toml')]
    if profile_source in {'escape', 'symlink'}:
        with pytest.raises(ProfileError, match='escapes the repository root'):
            parse_native_config(arguments)
        return
    frozen = freeze_reviewer(parse_native_config(arguments))
    assert frozen['file_config']['phases']['per_stack_review']['model'] == 'phase-model'
    for path in (root / 'repo.toml', tmp_path / 'env.toml', tmp_path / 'explicit.toml'):
        path.unlink()
    config = reviewer_config(json.loads(json.dumps(frozen)), repo)
    monkeypatch.setattr(runner, 'create_backend', lambda *_args, **_kwargs: EmptyReviewBackend(repo))
    result = await _review(repo, case, config)
    assert result.status == 'completed_empty', result.diagnostics
    assert result.configuration['phases']['per_stack_review']['model'] == 'explicit-model'
    profile = result.configuration['profile']
    assert profile['source_kind'] == profile_source
    if profile_source != 'default':
        assert profile['name'] == {'explicit': 'explicit-policy', 'env': 'environment-policy',
                                   'repo': 'repository-policy'}[profile_source]
    assert not fake_gh.calls('POST')


@pytest.mark.parametrize('arguments', [
    ['--copy', 'extra.py'], ['--test-command', 'true'], ['--approve-on-clean'],
    ['--file-scope-issues'], ['--trace-to', 'otlp'],
])
def test_export_refuses_native_controls_without_an_admitted_runtime_capability(
    tmp_path: Path, arguments: list[str],
) -> None:
    """Reject copying, tests, publication and trace controls whose capabilities cannot be delivered."""
    config = parse_native_config([str(tmp_path), '--backend', 'codex', *arguments])
    with pytest.raises(ValueError, match='unsupported benchmark native policy'):
        freeze_reviewer(config)


def test_head_conversion_retains_logical_records_and_exposes_invalid_locations(tmp_path: Path) -> None:
    """Convert proven HEAD locations and retain explicit omissions for unsafe hints."""
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


async def test_native_review_preserves_ordinary_extension_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, ext_dir: ExtDir,
) -> None:
    """Benchmark reviews retain ordinary extension loading, prompts and custom phase models.

    The ordinary extension integration tests do not enter review_case's isolated
    checkout or deliver its frozen per-backend execution capabilities.
    """
    repo, case = frozen_case(tmp_path)
    ext_dir.write_module(
        'from daydream.agent import run_agent\n'
        'from daydream.extensions import FlowStep\n'
        'from daydream.trajectory import DaydreamPhase\n'
        'async def probe(ctx):\n'
        '    await run_agent(ctx.backend_for("extension-review"), ctx.work.repo, "NATIVE_EXTENSION", '
        'phase=DaydreamPhase.REVIEW, run_context=ctx.run_context)\n'
        'def register(registry):\n'
        '    intent = registry.prompt("intent")\n'
        '    registry.override_prompt("intent", lambda *args, **kwargs: '
        'intent(*args, **kwargs) + "\\nNATIVE_PROMPT_OVERRIDE")\n'
        '    registry.register_phase(FlowStep(name="native-probe", run=probe))\n'
        '    registry.insert_before("deep", anchor="intent", step="native-probe")\n')
    config = empty_review_config(repo, tmp_path / 'trajectory', file_config=DaydreamFileConfig(
        backend='pi', supervisor='llm', phases={'extension-review': {'backend': 'claude', 'model': 'claude-sonnet-5'}}))
    frozen = _execution_input(config)
    frozen = replace(frozen, backends={**frozen.backends,
        'claude': BackendExecutionInput.from_environment(os.environ, backend='claude')})
    created: list[tuple[str, EmptyReviewBackend]] = []

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        backend = EmptyReviewBackend(kwargs.get('cwd', repo), forbid_merge=False, forbid_supervise=False)
        backend.model = kwargs['model']
        created.append((name, backend))
        return backend

    monkeypatch.setattr(runner, 'create_backend', create)
    result = await _review(repo, case, config, execution_input=frozen)
    assert result.status == 'completed_empty', result.diagnostics
    assert result.candidate is not None
    assert any(name == 'claude' and any(call['prompt'] == 'NATIVE_EXTENSION' for call in backend.calls)
               for name, backend in created)
    assert any('NATIVE_PROMPT_OVERRIDE' in call['prompt'] for _, backend in created for call in backend.calls)
    assert result.configuration['phases']['extension-review']['model'] == 'claude-sonnet-5'
    assert not fake_gh.calls('POST')


async def test_frozen_pi_defaults_refuse_case_thinking_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable frozen operator effort cannot inherit higher case-local Pi settings."""
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

    config = empty_review_config(operator, tmp_path / 'trajectory', backend='pi')
    execution = _execution_input(config)
    backend = PiBackend(cwd=repo, execution_input=execution.backends['pi'])
    with pytest.raises(PiError, match='Case Pi thinking conflicts with unavailable operator effort'):
        async for _ in backend.execute(repo, 'Review this change'):
            pytest.fail('case-local effort reached the Pi transport')


@pytest.mark.parametrize(("selection", "provider"), [
    ("explicit", "nous"), ("project", "nous"), ("project_phase", "nous"),
    ("project_cli", "openrouter"), ("project_cli_missing", "nous"), ("project_env", "nous"), ("agent_relative", "nous"),
    ("harbor", "nous"), ("harbor_phase", "nous"),
])
async def test_native_pi_retains_nous_provider_model_and_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, selection: str, provider: str,
) -> None:
    """Freeze operator configuration and cwd-relative credentials across isolated runs."""
    is_public = selection in {'harbor', 'harbor_phase'}
    public_environment = None
    if is_public:
        # Ordinary Pi uses deterministic author context for <=3 source files.
        # Four real changed sources require the intent phase to execute, proving
        # its distinct native model reaches the subprocess and observed evidence.
        public_environment = public_input(tmp_path, additional_sources=3 if selection == 'harbor_phase' else 0)
        repo = Path(public_environment['DAYDREAM_RB_REPOSITORY'])
        case = Case.from_dict(json.loads((Path(public_environment['DAYDREAM_RB_INPUTS']) / 'pr.json').read_text()))
    else:
        repo, case = frozen_case(tmp_path)
    if selection in {"explicit", "project_env"} or is_public:
        monkeypatch.setenv("PI_PROVIDER", "nous")
    else:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_API_KEY", raising=False)
    monkeypatch.delenv("PI_THINKING", raising=False)
    monkeypatch.setenv("NOUS_API_KEY", "offline-native-auth-canary")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-other-auth-canary")
    agent_dir = tmp_path / 'operator-pi'
    if selection == 'agent_relative':
        operator_cwd = tmp_path / 'operator-cwd'
        operator_cwd.mkdir()
        monkeypatch.chdir(operator_cwd)
        agent_dir = operator_cwd / '.operator-pi'
        monkeypatch.delenv('NOUS_API_KEY')
        monkeypatch.delenv('OPENROUTER_API_KEY')
    agent_dir.mkdir()
    if is_public:
        (agent_dir / 'models.json').write_text(json.dumps({'providers': {'nous': {
            'baseUrl': 'https://inference-api.nousresearch.com/v1', 'api': 'openai-completions',
            'models': [{'id': model, 'name': model, 'reasoning': True, 'contextWindow': 100000, 'maxTokens': 8192}
                       for model in ('deepseek/deepseek-v4.1-flash', 'native-intent-model')]}}}))
    auth_body = '{"nous":{"type":"api_key","key":"offline-file-auth-canary"}}'
    (agent_dir / 'auth.json').write_text(auth_body)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", '.operator-pi' if selection == 'agent_relative' else str(agent_dir))
    real_exec: Any = asyncio.create_subprocess_exec
    captured: list[tuple[list[str], dict[str, Any]]] = []
    auth_bodies: list[str | None] = []
    expected_usage: list[tuple[int, int, int, float]] = []
    operator = tmp_path / "operator"
    operator.mkdir()
    overridden = selection in {"project_cli", "project_phase", "project_cli_missing"}
    if selection != "explicit" and not is_public:
        settings_dir = agent_dir if selection == 'agent_relative' else operator / '.pi'
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings = {"defaultModel": "operator-default-model" if overridden else "deepseek/deepseek-v4.1-flash",
                    "defaultThinkingLevel": "low", "apiKey": "must-not-fingerprint-this-secret"}
        if selection != "project_cli_missing":
            settings["defaultProvider"] = "openrouter" if selection == "project_env" else provider
        (settings_dir / 'settings.json').write_text(json.dumps(settings))
        write_and_stage(repo, ".pi/settings.json", '{"defaultProvider":"other","defaultModel":"wrong-model"}')
        case = replace(case, head=commit(repo, "case native settings cannot replace operator settings"))
        git_ops.update_refs(repo, {'refs/reviewbench/head': case.head})

    async def native_process(*args: object, **kwargs: Any) -> MagicMock:
        if not args or args[0] != "pi":
            return cast(MagicMock, await real_exec(*args, **kwargs))
        argv = [str(arg) for arg in args]
        captured.append((argv, kwargs))
        auth_path = (Path(kwargs['env']['DAYDREAM_RB_PI_AUTH_PATH']) if is_public else
                     Path(kwargs['cwd']) / kwargs['env']['PI_CODING_AGENT_DIR'] / 'auth.json')
        auth_bodies.append(auth_path.read_text() if auth_path.is_file() else None)
        if is_public:
            models = json.loads((Path(kwargs['env']['PI_CODING_AGENT_DIR']) / 'models.json').read_text())
            delivered_models = {item['id'] for item in models['providers']['nous']['models']}
            assert delivered_models == ({'deepseek/deepseek-v4.1-flash', 'native-intent-model'}
                                        if selection == 'harbor_phase' else {'deepseek/deepseek-v4.1-flash'})
            assert argv[argv.index('--model') + 1] in delivered_models

        prompt = Path(argv[-1][1:]).read_text()
        stage = review_stage_state(prompt)
        if stage is None and "--extension" not in argv:
            expected_usage.append((110, 50, 10, 0.0003))
            return make_mock_process_from_fixture("simple_text.jsonl")
        expected_usage.append((13, 3, 2, 0.125))
        payload = stage_result(stage) if stage is not None else {"issues": []}
        call = {"type": "toolCall", "id": "submit", "name": "structured_output", "arguments": payload}
        assistant = {"role": "assistant", "content": [call], "stopReason": "toolUse",
            "usage": {"input": 11, "output": 3, "cacheRead": 2, "cost": {"total": 0.125}}}
        tool = {"content": [{"type": "text", "text": "Submitted."}], "details": payload}
        return make_mock_process([json.dumps(row) for row in [
            {"type": "session", "sessionId": "offline-fixture"}, {"type": "agent_start"}, {"type": "turn_start"},
            {"type": "message_end", "message": assistant},
            {"type": "tool_execution_start", "toolCallId": "submit", "toolName": "structured_output", "args": payload},
            {"type": "tool_execution_end", "toolCallId": "submit", "toolName": "structured_output",
             "isError": False, "result": tool},
            {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "submit",
             "toolName": "structured_output", "isError": False, **tool}},
            {"type": "turn_end", "message": assistant}, {"type": "agent_end", "messages": []},
            {"type": "agent_settled"}]])

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", native_process)
    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi",
        file_config=DaydreamFileConfig(model="deepseek/deepseek-v4.1-flash", reasoning_effort="low",
            phases={"intent": {"model": "native-intent-model"}} if selection == "harbor_phase" else {})
        if selection == "explicit" or is_public else DaydreamFileConfig())
    if selection != "explicit" and not is_public:
        config.target = str(operator)
    if selection in {"project_cli", "project_cli_missing"}:
        config.model = "deepseek/deepseek-v4.1-flash"
    elif selection == "project_phase":
        config.file_config = DaydreamFileConfig(phases={"per_stack_review": {"model": "deepseek/deepseek-v4.1-flash"}})
    if is_public:
        from daydream.benchmark.reviewbench.harbor_entrypoint import main
        from daydream.benchmark.reviewbench.harbor_profile import canonical_digest

        assert public_environment is not None
        profile_path = Path(public_environment['DAYDREAM_RB_INPUTS']) / 'daydream-profile.json'
        profile = json.loads(profile_path.read_text())
        profile['reviewer'] = freeze_reviewer(config)
        assert profile['reviewer']['phase_models']['intent'] == (
            'native-intent-model' if selection == 'harbor_phase' else 'deepseek/deepseek-v4.1-flash')
        profile['profile_sha256'] = canonical_digest({key: value for key, value in profile.items()
                                                      if key != 'profile_sha256'})
        profile_path.write_text(json.dumps(profile))
        public_environment['DAYDREAM_RB_REVIEWER_HOME'] = str(agent_dir)
        assert await main(public_environment) == 0
        evidence = Path(public_environment['DAYDREAM_RB_EVIDENCE']) / 'execution.json'
        result: Any = SimpleNamespace(**json.loads(evidence.read_text()))
        assert result.reviewer_backend == 'pi' and result.reviewer_provider == 'nous'
        assert result.reviewer_authentication == 'native-pi'
    else:
        result = await _review(repo, case, config)
    assert result.status == "completed_empty", result.diagnostics
    assert captured
    assert auth_bodies == [auth_body] * len(captured)
    expected_models = {"deepseek/deepseek-v4.1-flash"}
    if selection == "harbor_phase":
        expected_models.add("native-intent-model")
    for argv, kwargs in captured:
        assert argv[argv.index("--provider") + 1] == provider
        efforts = {"low"} if selection == "explicit" or is_public else {"low", "medium"}
        assert argv[argv.index("--thinking") + 1] in efforts
        if selection != 'agent_relative' and not is_public:
            assert kwargs["env"]["NOUS_API_KEY"] == "offline-native-auth-canary"
            if provider == "openrouter":
                assert kwargs["env"]["OPENROUTER_API_KEY"] == "offline-other-auth-canary"
    assert {argv[argv.index("--model") + 1] for argv, _ in captured} == expected_models
    assert result.candidate is not None
    assert result.candidate['usage'] == {"prompt_tokens": sum(row[0] for row in expected_usage),
        "completion_tokens": sum(row[1] for row in expected_usage),
        "cached_tokens": sum(row[2] for row in expected_usage),
        "cost_usd": pytest.approx(sum(row[3] for row in expected_usage)),
        "total_tokens": sum(row[0] + row[1] for row in expected_usage)}
    assert {row['provider'] for row in result.configuration['observed_requests']} == {provider}
    assert {row['model'] for row in result.configuration['observed_requests']} == expected_models
    assert {'wonder', 'arbiter', 'parse'} <= result.configuration['phases'].keys()
    if selection == "explicit" or is_public:
        assert config.file_config is not None and config.file_config.model == "deepseek/deepseek-v4.1-flash"
        assert os.environ["PI_PROVIDER"] == "nous"
    else:
        assert 'must-not-fingerprint-this-secret' not in json.dumps(asdict(result), default=str)
        if selection == "project_env":
            assert os.environ["PI_PROVIDER"] == "nous"
        else:
            assert 'PI_PROVIDER' not in os.environ
    retained = json.dumps(vars(result) if is_public else asdict(result), default=str)
    assert 'offline-native-auth-canary' not in retained
    assert 'offline-other-auth-canary' not in retained
    assert 'offline-file-auth-canary' not in retained
    assert os.environ['PI_CODING_AGENT_DIR'] == ('.operator-pi' if selection == 'agent_relative' else str(agent_dir))


@pytest.mark.parametrize("fault", ["missing_terminal", "malformed_projection", "ignored_patch", "moved_head"])
async def test_native_artifact_failure_or_changed_comparison_never_becomes_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    """Artifact faults and changed comparisons cannot produce scorable empty success."""
    repo, case = frozen_case(tmp_path)
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
        checkout = kwargs["cwd"]
        if fault == "moved_head":
            git_ops.checkout_detach(checkout, case.base)
            injected.append("head")
        return EmptyReviewBackend(checkout)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "replace", replace_file)
    monkeypatch.setattr(runner, "create_backend", create)
    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi",
        ignore_paths=["api.py"] if fault == "ignored_patch" else [])
    result = await _review(repo, case, config)
    assert result.status == "failed", result.diagnostics
    assert result.candidate is None and result.diagnostics
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


async def test_result_observer_preserves_terminal_report_and_invocation_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """Observer mutations leave native reporting and later invocation scope intact."""
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

    config = empty_review_config(repo, tmp_path / "trajectory", backend="pi", base=case.base,
                                 model="native-original-model", pr_number=None)
    assert await runner.run(config, review_result_handler=observe) == 0
    assert len(received) == 1
    coverage = json.loads((repo / ".daydream/deep/review-coverage.json").read_text())
    report = (repo / ".review-output.md").read_text()
    assert coverage["run_id"] == received[0]
    assert all(scope["status"] == "complete" for scope in coverage["stack_outcomes"])
    assert "observer-mutated" not in report and received[0] in report
    assert "<code>complete</code>" in report
    assert await runner.run(config) == 0
    assert len(received) == 1, "a later ordinary invocation must not inherit the observer"


async def test_result_delivery_follows_nested_and_concurrent_run_contexts() -> None:
    """Nested ordinary runs mask observers; sibling tasks keep their own deliveries."""
    received: dict[str, list[str]] = {name: [] for name in ("first", "second")}

    def fail(_result: FinalizedReview) -> None:
        raise RuntimeError("consumer failed")

    async def deliver(name: str) -> None:
        items = [finding(source_uids=['native-source'])]
        configuration: dict[str, Any] = {'owner': name, 'phases': {'review': {'model': 'native-model'}}}

        def observe(result: FinalizedReview) -> None:
            received[name].append(result.configuration['owner'])
            result.items[0]['source_uids'].append('observer-source')
            result.configuration['phases']['review']['model'] = 'observer-model'

        with bind_run_context(RunContext(InteractionPolicy(), review_result_handler=observe)):
            with bind_run_context(RunContext(InteractionPolicy())):
                deliver_finalized_review(terminal_result(), [], {"owner": "nested"})
            await asyncio.sleep(0)
            deliver_finalized_review(terminal_result(), items, configuration)
            with pytest.raises(RuntimeError, match="consumer failed"):
                with bind_run_context(RunContext(InteractionPolicy(), review_result_handler=fail)):
                    deliver_finalized_review(terminal_result(), [], {})
            deliver_finalized_review(terminal_result(), items, configuration)
        assert items[0]['source_uids'] == ['native-source']
        assert configuration['phases']['review']['model'] == 'native-model'

    async with asyncio.TaskGroup() as tasks:
        for name in received:
            tasks.create_task(deliver(name))
    deliver_finalized_review(terminal_result(), [], {"owner": "outside"})
    assert received == {name: [name, name] for name in received}
