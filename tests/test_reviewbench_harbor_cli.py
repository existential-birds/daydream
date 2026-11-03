from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_build, harbor_cli
from daydream.benchmark.reviewbench.contract import PUBLIC_PROFILE_SCHEMA, Case
from daydream.benchmark.reviewbench.harbor_profile import canonical_digest
from tests.harness.reviewbench import native_auth
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


@pytest.mark.parametrize('launch', ['pi'], indirect=True)
def test_launch_binds_runtime_selection_rounds_membership_and_subscription_environment(
    launch: argparse.Namespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.environments.docker.docker import DockerEnvironment

    args = launch
    profile_path = args.directory / 'daydream-profile.json'
    profile = json.loads(profile_path.read_text())
    last_task = args.directory / sorted(profile['cases'])[-1] / 'tests/task.json'
    # Unbound runtime, membership, identity, schema, credential or kernel inputs are
    # refused before any job evidence, format check, launch or credential use.
    with monkeypatch.context() as unpinned:
        unpinned.setattr(importlib.metadata, 'version', lambda _name: '0.1.0')
        with pytest.raises(ValueError, match='pinned Harbor|reviewbench-harbor'):
            harbor_cli.handle(args)
    with monkeypatch.context() as released:
        released.setattr(importlib.metadata, 'distribution', lambda _name: SimpleNamespace(
            version='0.24.0', read_text=lambda _file: '{}'))
        with pytest.raises(ValueError, match='pinned Harbor'):
            harbor_cli.handle(args)
    with monkeypatch.context() as kernel:
        kernel.setattr(DockerEnvironment, '_egress_control_kernel_support', staticmethod(lambda: False))
        with pytest.raises(ValueError, match='exact-host isolation'):
            harbor_cli.handle(args)
    selected_task, judge_home = args.task, args.pi_home
    args.task = ['foreign']
    with pytest.raises(ValueError, match='exact'):
        harbor_cli.handle(args)
    args.task, args.pi_home = selected_task, args.directory / 'credentials'
    with pytest.raises(ValueError, match='outside dataset and job evidence'):
        harbor_cli.handle(args)
    args.pi_home = judge_home
    for path, damaged, expected in (
        (profile_path, json.dumps({**profile, 'schema': None, 'policy_sha256': canonical_digest(
            {**{key: value for key, value in profile.items() if key != 'policy_sha256'}, 'schema': None})}),
         'Incompatible public profile schema'),
        (last_task, json.dumps({**json.loads(last_task.read_text()), 'evaluation_id': 0}), 'frozen full identity'),
    ):
        original = path.read_text()
        path.write_text(damaged)
        with pytest.raises(ValueError, match=expected):
            harbor_cli.handle(args)
        path.write_text(original)
    extra = args.directory / 'unexported-task'
    extra.mkdir()
    (extra / 'task.toml').touch()
    with pytest.raises(ValueError, match='membership'):
        harbor_cli.handle(args)
    (extra / 'task.toml').unlink()
    extra.rmdir()
    assert not args.jobs_dir.exists() and not args.format_checks and not args.launch_records

    args.rounds = 1
    native_variable = 'RB_HARBOR_CODEX_AUTH_DIR' if args.codex_home else 'RB_HARBOR_REVIEWER_PI_AUTH_DIR'
    native_home = args.codex_home or args.reviewer_pi_home
    unselected = 'RB_HARBOR_REVIEWER_PI_AUTH_DIR' if args.codex_home else 'RB_HARBOR_CODEX_AUTH_DIR'
    monkeypatch.setenv(unselected, '/unselected/private/credentials')
    monkeypatch.setenv('DOCKER_DEFAULT_PLATFORM', 'linux/amd64')
    monkeypatch.setenv('RB_HARBOR_SECCOMP_PROFILE', 'unconfined')
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CODEX_BEARER_TOKEN', 'OPENAI_WORKLOAD_IDENTITY_TOKEN'):
        monkeypatch.setenv(key, 'synthetic-forbidden-route')
    assert harbor_cli.handle(args) == 0
    actual = args.launch_records[0]
    local_config = Path(actual['args'][actual['args'].index('-c') + 1])
    assert local_config.parent == args.jobs_dir.resolve() and local_config.is_file()
    assert actual == {
        'args': ['run', '--yes', '-c', str(local_config), '-p', str(args.directory.resolve()),
                 '-a', 'daydream.benchmark.reviewbench.harbor_agent:DaydreamPublicReviewAgent',
                 '-n', '1', '-k', '1', '-r', '0', '-o', str(args.jobs_dir.resolve()),
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

    # The full set passes the exact supplied membership, not a glob or the test split.
    args.task, args.set, args.rounds, args.job_name = None, 'full', 3, 'full-set'
    profile = json.loads((args.directory / 'daydream-profile.json').read_text())
    assert harbor_cli.handle(args) == 0
    actual = args.launch_records[-1]
    selected = [actual['args'][index + 1] for index, token in enumerate(actual['args']) if token == '-i']
    assert selected == sorted(profile['cases']) and len(selected) == 219
    assert actual['args'][actual['args'].index('-k') + 1] == '3'
    assert '-l' not in actual['args']

    # The supplied test split selects exactly its 25 tasks; a failed Harbor run is propagated.
    args.set, args.job_name = 'test', 'test-set'
    monkeypatch.setenv('DAYDREAM_TEST_HARBOR_EXIT', '29')
    assert harbor_cli.handle(args) == 29
    actual = args.launch_records[-1]
    assert [actual['args'][index + 1] for index, token in enumerate(actual['args']) if token == '-i'] == sorted(
        profile['test_tasks'])


@pytest.mark.parametrize('checker_exit', [29])
def test_launch_propagates_completed_job_checker_without_reusing_prior_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checker_exit: int,
) -> None:
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
    """Export freezes the candidate wheel's exact bytes without importing host extension code."""
    wheel = tmp_path / 'wheels/daydream-0.0.0-py3-none-any.whl'
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('daydream/benchmark/reviewbench/harbor_entrypoint.py', '')
        archive.writestr('daydream_ext/__init__.py', 'DAYDREAM_EXT_API = 8\n')
        archive.writestr('daydream-0.0.0.dist-info/METADATA', 'Metadata-Version: 2.1\nName: daydream\nVersion: 0.0.0\n')
    admitted = wheel.read_bytes()
    outside = tmp_path / 'host-extension'
    outside.mkdir()
    sentinel = tmp_path / 'host-extension-executed'
    (outside / '__init__.py').write_text(f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n')
    monkeypatch.setenv('DAYDREAM_EXT_DIR', str(outside))
    _, assets, identity = harbor_build.reviewer_build(wheel)
    assert assets[wheel.name] == admitted
    assert identity['wheel_sha256'] == hashlib.sha256(admitted).hexdigest()
    assert not sentinel.exists()
