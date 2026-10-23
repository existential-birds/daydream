"""Harbor launch, exact selection and wheel artifact contracts without inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_build, harbor_cli, harbor_profile
from daydream.benchmark.reviewbench.contract import PUBLIC_PROFILE_SCHEMA, Case, load_cases
from daydream.benchmark.reviewbench.harbor_profile import canonical_digest
from daydream.benchmark.reviewbench.native import freeze_reviewer
from daydream.run_config import RunConfig
from tests.harness.reviewbench import native_auth, supplied_source as supplied_source
from tests.test_reviewbench_seccomp import moby_policy as moby_policy


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> argparse.Namespace:
    from harbor.environments.docker.docker import DockerEnvironment

    launches: list[dict[str, Any]] = []
    format_checks: list[Path] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        assert command[1:3] in (['-m', 'reviewbench_harbor.check-format'], ['-m', 'reviewbench_harbor.check-job'])
        if command[2].endswith('check-format'):
            format_checks.append(Path(command[3]))
        return SimpleNamespace(returncode=0)

    def call(command: list[str], *, env: dict[str, str]) -> int:
        args = command[1:]
        (Path(args[args.index('-o') + 1]) / args[args.index('--job-name') + 1]).mkdir(parents=True)
        launches.append({'args': args,
            'metric_config': json.loads(Path(args[args.index('-c') + 1]).read_text()),
            'environment': {key: value for key, value in env.items() if key.startswith('RB_HARBOR_')
                or key in {'DOCKER_DEFAULT_PLATFORM', 'OPENAI_API_KEY', 'OPENAI_BASE_URL',
                           'CODEX_BEARER_TOKEN', 'OPENAI_WORKLOAD_IDENTITY_TOKEN'}}})
        return int(env.get('DAYDREAM_TEST_HARBOR_EXIT', '0'))

    monkeypatch.setattr(harbor_cli, 'subprocess', SimpleNamespace(run=run, call=call))
    monkeypatch.setattr(DockerEnvironment, '_egress_control_kernel_support', staticmethod(lambda: True))
    distribution, version = importlib.metadata.distribution, importlib.metadata.version
    harbor_distribution = SimpleNamespace(
        version='0.24.0',
        read_text=lambda _file: json.dumps({'vcs_info': {'commit_id': harbor_build.HARBOR_REVISION}}))
    monkeypatch.setattr(importlib.metadata, 'distribution',
                        lambda name: harbor_distribution if name == 'harbor' else distribution(name))
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda name: '0.1.0rc10' if name == 'reviewbench-harbor' else version(name))
    backend = getattr(request, 'param', 'codex')
    dataset, source = tmp_path / 'dataset', tmp_path / 'subscription'
    dataset.mkdir()
    (dataset / 'metric.py').write_text('EXPECTED_TASKS = {}\n')
    if backend == 'pi':
        source.mkdir()
        (source / 'auth.json').write_text('{"nous":{"type":"api_key","key":"synthetic-nous-native"}}')
    else:
        native_auth(source)
    judge = tmp_path / 'judge-auth'
    native_auth(judge, 'judge')
    cases = [Case('https://github.com/example/repository', number, 'b' * 40, f'{number:040x}')
             for number in range(1, 220)]
    policy = {'schema': PUBLIC_PROFILE_SCHEMA, 'source_prs': 219, 'final_comparison_rounds': 3,
              'candidate': {'wheel_name': 'daydream-candidate.whl', 'wheel_sha256': 'a' * 64},
              'reviewer': {'backend': backend, 'provider': 'nous' if backend == 'pi' else None},
              'judge': {'provider': 'openai-codex', 'model': 'gpt-6.1-sol', 'authentication': 'subscription'},
              'cases': {case.key: asdict(case) for case in cases},
              'test_tasks': [case.key for case in cases[::8]][:25]}
    policy_sha = canonical_digest(policy)
    (dataset / 'daydream-profile.json').write_text(json.dumps({**policy, 'policy_sha256': policy_sha}))
    for number, case in enumerate(cases, 1):
        directory = dataset / case.key
        (directory / 'tests').mkdir(parents=True)
        (directory / 'environment').mkdir()
        (directory / 'task.toml').touch()
        task = {'pr': asdict(case), 'task_id': number, 'evaluation_id': number + 219}
        (directory / 'tests/task.json').write_text(json.dumps(task))
        (directory / 'environment/daydream-profile.json').write_text(json.dumps({
            **task, 'policy_sha256': policy_sha,
        }))
    parser = argparse.ArgumentParser()
    harbor_cli.add_parsers(parser.add_subparsers(dest='reviewbench_command', required=True))
    args = parser.parse_args([
        'harbor-run', str(dataset), '--reviewer-pi-home' if backend == 'pi' else '--codex-home', str(source),
        '--pi-home', str(judge),
        '--jobs-dir', str(tmp_path / 'jobs'), '--job-name', 'selected-case', '--task', cases[0].key,
    ])
    args.launch_records, args.format_checks = launches, format_checks
    return args


@pytest.mark.parametrize(('launch', 'rounds', 'exit_code'), [('codex', 1, 0), ('codex', 3, 23), ('pi', 1, 0)],
                         indirect=['launch'])
def test_launch_delivers_selection_rounds_and_subscription_environment(
    launch: argparse.Namespace, monkeypatch: pytest.MonkeyPatch,
    rounds: int, exit_code: int,
) -> None:
    """Deliver exact native launch settings, scrub billing routes and propagate Harbor failure."""
    args = launch
    args.rounds = rounds
    native_variable = 'RB_HARBOR_CODEX_AUTH_DIR' if args.codex_home else 'RB_HARBOR_REVIEWER_PI_AUTH_DIR'
    native_home = args.codex_home or args.reviewer_pi_home
    unselected = 'RB_HARBOR_REVIEWER_PI_AUTH_DIR' if args.codex_home else 'RB_HARBOR_CODEX_AUTH_DIR'
    monkeypatch.setenv(unselected, '/unselected/private/credentials')
    monkeypatch.setenv('DAYDREAM_TEST_HARBOR_EXIT', str(exit_code))
    monkeypatch.setenv('DOCKER_DEFAULT_PLATFORM', 'linux/amd64')
    monkeypatch.setenv('RB_HARBOR_SECCOMP_PROFILE', 'unconfined')
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CODEX_BEARER_TOKEN', 'OPENAI_WORKLOAD_IDENTITY_TOKEN'):
        monkeypatch.setenv(key, 'synthetic-forbidden-route')
    assert harbor_cli.handle(args) == exit_code
    actual = args.launch_records[0]
    local_config = Path(actual['args'][actual['args'].index('-c') + 1])
    assert local_config.parent == args.jobs_dir.resolve() and local_config.is_file()
    assert actual == {
        'args': ['run', '--yes', '-c', str(local_config), '-p', str(args.directory.resolve()),
                 '-a', 'daydream.benchmark.reviewbench.harbor_agent:DaydreamPublicReviewAgent',
                 '-n', '1', '-k', str(rounds), '-r', '0', '-o', str(args.jobs_dir.resolve()),
                 '--job-name', args.job_name, '--plugin', 'reviewbench_harbor.reviewbench_plugin:ReviewBenchPlugin',
                 '-i', args.task[0]],
        'metric_config': {'metrics': [{'type': 'uv-script',
                                      'kwargs': {'script_path': str(args.directory.resolve() / 'metric.py')}}]},
        'environment': {native_variable: str(native_home.resolve()),
                        'RB_HARBOR_PI_AUTH_DIR': str(args.pi_home.resolve()),
                        'RB_HARBOR_PI_AUTH_PATH': '/run/reviewbench-pi/auth.json',
                        'RB_HARBOR_SECCOMP_PROFILE': str(args.directory.resolve() / 'reviewer-seccomp.json')},
    }
    assert args.format_checks == [args.directory.resolve()]


@pytest.mark.parametrize('selection', ['test', 'full'])
def test_launch_passes_exact_supplied_membership_to_harbor(
    launch: argparse.Namespace, selection: str,
) -> None:
    """Select exact supplied task names and independent rounds, never globs or count limits."""
    args = launch
    args.task, args.set, args.rounds = None, selection, 3
    profile = json.loads((args.directory / 'daydream-profile.json').read_text())
    assert harbor_cli.handle(args) == 0
    actual = args.launch_records[0]
    selected = [actual['args'][index + 1] for index, token in enumerate(actual['args']) if token == '-i']
    assert selected == sorted(profile['test_tasks'] if selection == 'test' else profile['cases'])
    assert len(selected) == (25 if selection == 'test' else 219)
    assert actual['args'][actual['args'].index('-k') + 1] == '3'
    assert '-l' not in actual['args']


@pytest.mark.parametrize('fault', [
    'foreign', 'glob', 'duplicate', 'missing-task', 'base', 'head', 'evaluation',
    'schema-missing', 'schema-1', 'schema-2', 'schema-3',
    'runtime-released', 'runtime-commit', 'runtime-reviewbench',
    'auth-dataset', 'auth-jobs', 'kernel', 'same-store', 'nested-store', 'copied-session', 'personal-home', 'judge-api',
])
def test_launch_rejects_unbound_inputs_before_process_or_credentials(
    launch: argparse.Namespace, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Every admission boundary rejects before format checking, jobs, or credential delivery."""
    from harbor.environments.docker.docker import DockerEnvironment

    args = launch
    path = args.directory / 'daydream-profile.json'
    profile = json.loads(path.read_text())
    expected = 'exact|identity|membership'
    if fault in {'foreign', 'glob', 'duplicate'}:
        args.task = ['foreign'] if fault == 'foreign' else ['*'] if fault == 'glob' else args.task * 2
    elif fault in {'missing-task', 'base', 'head', 'evaluation'}:
        directory = args.directory / sorted(profile['cases'])[-1]
        if fault == 'missing-task':
            (directory / 'task.toml').unlink()
        else:
            task_path = directory / 'tests/task.json'
            task = json.loads(task_path.read_text())
            if fault == 'evaluation':
                task['evaluation_id'] += 1
            else:
                task['pr'][fault] = 'c' * 40
            task_path.write_text(json.dumps(task))
    elif fault.startswith('schema-'):
        profile.pop('policy_sha256')
        profile['schema'] = None if fault == 'schema-missing' else int(fault.rsplit('-', 1)[1])
        path.write_text(json.dumps({**profile, 'policy_sha256': canonical_digest(profile)}))
        expected = 'Incompatible public profile schema'
    elif fault.startswith('runtime-'):
        expected = 'pinned Harbor|reviewbench-harbor'
        if fault == 'runtime-reviewbench':
            monkeypatch.setattr(importlib.metadata, 'version', lambda _name: '0.1.0')
        else:
            direct = {} if fault == 'runtime-released' else {'vcs_info': {'commit_id': '0' * 40}}
            monkeypatch.setattr(importlib.metadata, 'distribution', lambda _name: SimpleNamespace(
                version='0.24.0', read_text=lambda _file: json.dumps(direct)))
    elif fault.startswith('auth-'):
        args.pi_home = (args.directory if fault == 'auth-dataset' else args.jobs_dir) / 'credentials'
        expected = 'outside dataset and job evidence'
    elif fault == 'kernel':
        monkeypatch.setattr(DockerEnvironment, '_egress_control_kernel_support', staticmethod(lambda: False))
        expected = 'exact-host isolation'
    else:
        expected = 'separate|copied OAuth|mount only|dedicated Pi'
        if fault == 'same-store':
            args.pi_home = args.codex_home
        elif fault == 'nested-store':
            args.pi_home = args.codex_home / 'judge'
        elif fault == 'personal-home':
            (args.codex_home / 'config.toml').write_text('model = "unrelated"')
        else:
            auth = json.loads((args.pi_home / 'auth.json').read_text())
            if fault == 'copied-session':
                auth['openai-codex']['refresh'] = 'private-refresh-canary'
            else:
                auth['openai-codex'] = {'type': 'api_key', 'key': 'synthetic-forbidden-key'}
            (args.pi_home / 'auth.json').write_text(json.dumps(auth))
    with pytest.raises(ValueError, match=expected):
        harbor_cli.handle(args)
    assert not args.jobs_dir.exists() and not args.format_checks and not args.launch_records


@pytest.mark.parametrize('selection', ['test', 'full'])
def test_local_dataset_launch_builds_actual_harbor_reviewbench_plan_before_models(
    launch: argparse.Namespace, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, supplied_source: Path, moby_policy: bytes, selection: str,
) -> None:
    """Resolve the actual pinned CLI, Job, plugin, task identities and metric binding."""
    selected = os.environ.get('DAYDREAM_REVIEWBENCH_RUNTIME')
    if not selected:
        pytest.skip('set DAYDREAM_REVIEWBENCH_RUNTIME to the pinned public runtime Python')
    runtime = Path(selected).absolute()
    assert runtime.is_file()
    args = launch
    args.directory = tmp_path / 'actual-full-export'
    args.task, args.set, args.rounds = None, selection, 3
    monkeypatch.setattr(harbor_profile, 'reviewer_build', lambda *args, **kwargs: ('', {}, {}))
    harbor_profile.export_harbor_profile(supplied_source, args.directory, wheel=tmp_path / 'unused.whl',
        reviewer=freeze_reviewer(RunConfig(backend='codex', model='gpt-6.1-sol', reasoning_effort='high')),
        base_image=harbor_build.NODE_IMAGE)
    monkeypatch.setattr(harbor_cli, 'sys', SimpleNamespace(executable=str(runtime)))
    script = tmp_path / 'resolve-plan.py'
    script.write_text('''import asyncio, hashlib, json, sys
from pathlib import Path
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.cli.job_plugins import attach_job_plugins, PluginConfig
from daydream.benchmark.reviewbench.harbor_cli import verify_runtime

async def main():
    verify_runtime()
    config = JobConfig.model_validate_json(Path(sys.argv[1]).read_text())
    job = await Job.create(config)
    await attach_job_plugins(job, [
        PluginConfig(import_path="reviewbench_harbor.reviewbench_plugin:ReviewBenchPlugin"),
    ])
    plan = json.loads((job.job_dir / "reviewbench-plan.json").read_text())
    expected = json.loads(Path(sys.argv[3]).read_text())
    assert plan["source_prs"] == 219 and plan["expected_tasks"] == expected["tasks"]
    assert plan["n_attempts"] == 3 and len(plan["trials"]) == len(expected["tasks"]) * 3
    assert {(row["task_id"], row["evaluation_id"], row["round"]) for row in plan["trials"].values()} == {
        (int(task), evaluation, number) for task, evaluation in expected["tasks"].items() for number in (1, 2, 3)
    }
    assert set(config.datasets[0].task_names) == set(expected["cases"])
    metric_digest = hashlib.sha256(Path(config.metrics[0].kwargs["script_path"]).read_bytes()).hexdigest()
    assert plan["source_metric_sha256"] == [metric_digest]
    assert (job.job_dir / "reviewbench-metric.py").is_file()
    assert job.config.n_concurrent_trials == 1
    Path(sys.argv[2]).write_text(json.dumps({"plan": plan, "metric": config.metrics[0].model_dump(mode="json")}))

asyncio.run(main())
''')
    result_path = tmp_path / 'resolved-plan.json'
    merged_config = tmp_path / 'resolved-config.json'
    repository = Path(__file__).resolve().parents[1]
    cases = load_cases(supplied_source, selection)
    expected: dict[str, Any] = {'cases': {case.key: asdict(case) for case in cases}, 'tasks': {}}
    for case in cases:
        task = json.loads((args.directory / case.key / 'tests/task.json').read_text())
        assert Case.from_dict(task['pr']) == case
        expected['tasks'][str(task['task_id'])] = task['evaluation_id']
    expected_path = tmp_path / 'expected-selection.json'
    expected_path.write_text(json.dumps(expected))

    def resolve_only(command: list[str], *, env: dict[str, str]) -> int:
        # Resolve configuration without running trials, containers or models.
        environment = {**env, 'PYTHONPATH': str(repository)}
        printed = subprocess.run([*command, '--print-config'], env=environment, check=True,
                                 text=True, capture_output=True, timeout=60)
        merged_config.write_text(printed.stdout)
        subprocess.run([str(runtime), str(script), str(merged_config), str(result_path), str(expected_path)],
                       env=environment, check=True, text=True, capture_output=True, timeout=120)
        return 0

    monkeypatch.setattr(harbor_cli, 'subprocess', subprocess)
    monkeypatch.setattr(subprocess, 'call', resolve_only)
    # A plan without retained results cannot establish completed work.
    assert harbor_cli.handle(args) == 1
    actual = json.loads(result_path.read_text())
    assert actual['plan']['source_prs'] == 219
    assert actual['plan']['expected_tasks'] == expected['tasks']
    assert len(actual['plan']['trials']) == (75 if selection == 'test' else 657)
    assert actual['metric'] == {'type': 'uv-script', 'kwargs': {'script_path': str(args.directory / 'metric.py')}}
    assert len(list(args.jobs_dir.glob('daydream-reviewbench-launch-*.json'))) == 1


@pytest.mark.parametrize('checker_exit', [0, 29])
def test_launch_propagates_completed_job_checker_without_reusing_prior_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checker_exit: int,
) -> None:
    """Deliver exact checker arguments and reserve a fresh report without replacing prior bytes."""
    job = tmp_path / 'retained-job'
    job.mkdir()
    prior = job / 'daydream-reviewbench-check-prior.json'
    prior.write_text('preserve prior report bytes')
    requests: list[list[str]] = []

    def checker(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        requests.append(command)
        return SimpleNamespace(returncode=checker_exit)

    monkeypatch.setattr(harbor_cli, 'subprocess', SimpleNamespace(run=checker))
    assert harbor_cli._check_completed_job(job) == checker_exit
    assert prior.read_text() == 'preserve prior report bytes'
    command, = requests
    report = Path(command[-1])
    assert command == [sys.executable, '-m', 'reviewbench_harbor.check-job', str(job),
                       str(job / 'reviewbench-metric.py'), '--output', str(report)]
    assert report.parent == job and report.name.startswith('daydream-reviewbench-check-') and report != prior
    assert report.is_file()


def test_reviewer_build_captures_fork_wheel_without_host_extension_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture a different candidate artifact and execute its packaged extension/resources."""
    repository = Path(__file__).resolve().parents[1]
    fork = tmp_path / 'fork'
    fork.mkdir()
    shutil.copytree(repository / 'daydream', fork / 'daydream', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(repository / 'pyproject.toml', fork / 'pyproject.toml')
    extension = fork / 'daydream_ext'
    extension.mkdir()
    (extension / '__init__.py').write_text(
        'from pathlib import Path\nDAYDREAM_EXT_API = 8\n'
        'def prompt(**kwargs): return Path(__file__).with_name("policy.txt").read_text()\n'
        'def register(registry): registry.override_prompt("reviewbench_artifact_probe", prompt)\n',
    )
    (extension / 'policy.txt').write_text('candidate policy')
    subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(tmp_path / 'wheels')],
                   cwd=fork, check=True, capture_output=True)
    wheel = next((tmp_path / 'wheels').glob('*.whl'))
    admitted = wheel.read_bytes()
    # The launching process may select unrelated extension code. Export captures
    # the artifact without importing that code or copying its private state.
    outside = tmp_path / 'host-extension'
    outside.mkdir()
    sentinel = tmp_path / 'host-extension-executed'
    (outside / '__init__.py').write_text(f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n')
    monkeypatch.setenv('DAYDREAM_EXT_DIR', str(outside))
    _, assets, identity = harbor_build.reviewer_build(wheel)
    assert assets[wheel.name] == admitted
    assert identity['wheel_sha256'] == hashlib.sha256(admitted).hexdigest()
    assert not sentinel.exists()
    installed = tmp_path / 'installed'
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    environment = {**os.environ, 'PYTHONPATH': str(installed)}
    environment.pop('DAYDREAM_EXT_DIR', None)
    script = (
        'from daydream.extensions.loader import build_registry; '
        'print(build_registry().prompt("reviewbench_artifact_probe")())'
    )
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=environment,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == 'candidate policy'
