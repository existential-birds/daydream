"""Harbor launch, exact selection and wheel artifact contracts without inference."""
from __future__ import annotations

import argparse
import base64
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
from tests.harness.reviewbench import supplied_source as supplied_source
from tests.test_reviewbench_seccomp import moby_policy as moby_policy


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    from harbor.environments.docker.docker import DockerEnvironment

    binary = tmp_path / 'runtime' / 'bin'
    binary.mkdir(parents=True)
    interpreter = sys.executable
    format_checker = binary / 'python'
    format_checker.write_text(
        f'#!{interpreter}\n'
        'import json, os, pathlib, runpy, sys\n'
        'if sys.argv[1:3] == ["-m", "reviewbench_harbor.check-job"]:\n'
        ' (pathlib.Path(sys.argv[3]) / "checker-invoked").touch()\n'
        ' if checker := os.environ.get("DAYDREAM_TEST_REVIEWBENCH_CHECKER"):\n'
        '  sys.argv = [checker, *sys.argv[3:]]\n'
        '  runpy.run_path(checker, run_name="__main__")\n'
        ' else:\n'
        '  pass  # Argument/environment tests make no scoring assertion.\n'
        ' raise SystemExit(0)\n'
        'assert sys.argv[1:3] == ["-m", "reviewbench_harbor.check-format"]\n'
        'assert (pathlib.Path(sys.argv[3]) / "daydream-profile.json").is_file()\n'
        '(pathlib.Path(sys.argv[3]) / "format-checked").touch()\n'
    )
    format_checker.chmod(0o755)
    harbor = binary / 'harbor'
    harbor.write_text(
        f'#!{interpreter}\n'
        'import json, os, pathlib, shutil, sys\n'
        'args = sys.argv[1:]\n'
        'job = pathlib.Path(args[args.index("-o") + 1]) / args[args.index("--job-name") + 1]\n'
        'job.mkdir(parents=True)\n'
        'if results := os.environ.get("DAYDREAM_TEST_HARBOR_RESULTS"):\n'
        ' shutil.copytree(results, job, dirs_exist_ok=True)\n'
        'record = {"args": args, "metric_config": json.loads('
        'pathlib.Path(args[args.index("-c") + 1]).read_text()),\n'
        ' "environment": {key: os.environ[key] for key in ("RB_HARBOR_CODEX_AUTH_DIR",\n'
        ' "RB_HARBOR_REVIEWER_PI_AUTH_DIR",\n'
        ' "RB_HARBOR_PI_AUTH_DIR",\n'
        ' "RB_HARBOR_PI_AUTH_PATH", "RB_HARBOR_SECCOMP_PROFILE", "DOCKER_DEFAULT_PLATFORM", "OPENAI_API_KEY",\n'
        ' "OPENAI_BASE_URL", "CODEX_BEARER_TOKEN", "OPENAI_WORKLOAD_IDENTITY_TOKEN") if key in os.environ}}\n'
        '(job / "launch.json").write_text(json.dumps(record))\n'
        'raise SystemExit(int(os.environ.get("DAYDREAM_TEST_HARBOR_EXIT", "0")))\n'
    )
    harbor.chmod(0o755)
    monkeypatch.setattr(harbor_cli, 'sys', SimpleNamespace(executable=str(format_checker)))
    monkeypatch.setattr(DockerEnvironment, '_egress_control_kernel_support', staticmethod(lambda: True))
    distribution, version = importlib.metadata.distribution, importlib.metadata.version
    harbor_distribution = SimpleNamespace(
        version='0.24.0',
        read_text=lambda _file: json.dumps({'vcs_info': {'commit_id': harbor_build.HARBOR_REVISION}}))
    monkeypatch.setattr(importlib.metadata, 'distribution',
                        lambda name: harbor_distribution if name == 'harbor' else distribution(name))
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda name: '0.1.0rc10' if name == 'reviewbench-harbor' else version(name))
    dataset, source = tmp_path / 'dataset', tmp_path / 'subscription'
    dataset.mkdir()
    (dataset / 'metric.py').write_text('EXPECTED_TASKS = {}\n')
    source.mkdir()
    claims = base64.urlsafe_b64encode(b'{"exp":1}').decode().rstrip('=')
    (source / 'auth.json').write_text(json.dumps({
        'auth_mode': 'chatgpt', 'OPENAI_API_KEY': None, 'tokens': {
            'access_token': f'test.{claims}.test', 'refresh_token': 'synthetic-original-refresh',
            'account_id': 'test-account', 'id_token': 'synthetic-id',
        },
    }))
    judge = tmp_path / 'judge-auth'
    judge.mkdir()
    (judge / 'auth.json').write_text(json.dumps({'openai-codex': {
        'type': 'oauth', 'access': 'synthetic-pi-access', 'refresh': 'synthetic-pi-refresh',
        'expires': 1, 'accountId': 'test-account',
    }}))
    cases = [Case('https://github.com/example/repository', number, 'b' * 40, f'{number:040x}')
             for number in range(1, 220)]
    policy = {'schema': PUBLIC_PROFILE_SCHEMA, 'source_prs': 219, 'final_comparison_rounds': 3,
              'candidate': {'wheel_name': 'daydream-candidate.whl', 'wheel_sha256': 'a' * 64},
              'reviewer': {'backend': 'codex', 'provider': None},
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
        'harbor-run', str(dataset), '--codex-home', str(source), '--pi-home', str(judge),
        '--jobs-dir', str(tmp_path / 'jobs'), '--job-name', 'selected-case', '--task', cases[0].key,
    ])
    return args


@pytest.mark.parametrize(('rounds', 'exit_code'), [(1, 0), (3, 23)])
def test_launch_delivers_selection_rounds_and_subscription_environment(
    launch: argparse.Namespace, monkeypatch: pytest.MonkeyPatch,
    rounds: int, exit_code: int,
) -> None:
    """Deliver exact native launch settings, scrub billing routes and propagate Harbor failure."""
    args = launch
    args.rounds = rounds
    monkeypatch.setenv('DAYDREAM_TEST_HARBOR_EXIT', str(exit_code))
    monkeypatch.setenv('DOCKER_DEFAULT_PLATFORM', 'linux/amd64')
    monkeypatch.setenv('RB_HARBOR_SECCOMP_PROFILE', 'unconfined')
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CODEX_BEARER_TOKEN', 'OPENAI_WORKLOAD_IDENTITY_TOKEN'):
        monkeypatch.setenv(key, 'synthetic-forbidden-route')
    assert harbor_cli.handle(args) == exit_code
    actual = json.loads((args.jobs_dir / args.job_name / 'launch.json').read_text())
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
        'environment': {'RB_HARBOR_CODEX_AUTH_DIR': str(args.codex_home.resolve()),
                        'RB_HARBOR_PI_AUTH_DIR': str(args.pi_home.resolve()),
                        'RB_HARBOR_PI_AUTH_PATH': '/run/reviewbench-pi/auth.json',
                        'RB_HARBOR_SECCOMP_PROFILE': str(args.directory.resolve() / 'reviewer-seccomp.json')},
    }
    assert (args.directory / 'format-checked').is_file()


@pytest.mark.parametrize('backend', ['pi', 'codex'])
def test_selected_reviewer_delivers_native_auth_separately_from_common_judge(
    launch: argparse.Namespace, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    """Select only the native reviewer capability while preserving the independent common judge."""
    args = launch
    path = args.directory / 'daydream-profile.json'
    profile = json.loads(path.read_text())
    judge = dict(profile['judge'])
    if backend == 'pi':
        home = args.codex_home.parent / 'native-pi'
        home.mkdir()
        (home / 'auth.json').write_text(json.dumps({'nous': {'type': 'api_key', 'key': 'synthetic-nous-native'}}))
        args.codex_home, args.reviewer_pi_home = None, home
        profile.pop('policy_sha256')
        profile['reviewer'] = {'backend': 'pi', 'provider': 'nous', 'model': 'deepseek/deepseek-v4.1-flash'}
        profile['policy_sha256'] = canonical_digest(profile)
        path.write_text(json.dumps(profile))
        for name in profile['cases']:
            delivered_path = args.directory / name / 'environment/daydream-profile.json'
            delivered = json.loads(delivered_path.read_text())
            delivered['policy_sha256'] = profile['policy_sha256']
            delivered_path.write_text(json.dumps(delivered))
    monkeypatch.setenv('RB_HARBOR_REVIEWER_PI_AUTH_DIR' if backend == 'codex' else 'RB_HARBOR_CODEX_AUTH_DIR',
                       '/unrelated/private/credentials')
    assert harbor_cli.handle(args) == 0
    actual = json.loads((args.jobs_dir / args.job_name / 'launch.json').read_text())
    environment = actual['environment']
    selected = 'RB_HARBOR_CODEX_AUTH_DIR' if backend == 'codex' else 'RB_HARBOR_REVIEWER_PI_AUTH_DIR'
    unselected = 'RB_HARBOR_REVIEWER_PI_AUTH_DIR' if backend == 'codex' else 'RB_HARBOR_CODEX_AUTH_DIR'
    assert environment[selected] == str((args.codex_home if backend == 'codex' else args.reviewer_pi_home).resolve())
    assert unselected not in environment
    assert environment['RB_HARBOR_PI_AUTH_DIR'] == str(args.pi_home.resolve())
    assert environment['RB_HARBOR_PI_AUTH_PATH'] == '/run/reviewbench-pi/auth.json'
    assert profile['judge'] == judge == {
        'provider': 'openai-codex', 'model': 'gpt-6.1-sol', 'authentication': 'subscription',
    }
    assert 'synthetic-nous-native' not in json.dumps(actual) + path.read_text()


@pytest.mark.parametrize('selection', ['test', 'full'])
def test_launch_passes_exact_supplied_membership_to_harbor(
    launch: argparse.Namespace, selection: str,
) -> None:
    """Select exact supplied task names and independent rounds, never globs or count limits."""
    args = launch
    args.task, args.set, args.rounds = None, selection, 3
    profile = json.loads((args.directory / 'daydream-profile.json').read_text())
    assert harbor_cli.handle(args) == 0
    actual = json.loads((args.jobs_dir / args.job_name / 'launch.json').read_text())
    selected = [actual['args'][index + 1] for index, token in enumerate(actual['args']) if token == '-i']
    assert selected == sorted(profile['test_tasks'] if selection == 'test' else profile['cases'])
    assert len(selected) == (25 if selection == 'test' else 219)
    assert actual['args'][actual['args'].index('-k') + 1] == '3'
    assert '-l' not in actual['args']


@pytest.mark.parametrize('fault', ['foreign', 'glob', 'duplicate', 'missing-task', 'base', 'head', 'evaluation'])
def test_launch_rejects_changed_exact_selection_before_models(
    launch: argparse.Namespace, fault: str,
) -> None:
    """Reject changed identities even in unselected tasks before launching models."""
    args = launch
    profile = json.loads((args.directory / 'daydream-profile.json').read_text())
    if fault in {'foreign', 'glob', 'duplicate'}:
        args.task = ['foreign'] if fault == 'foreign' else (['*'] if fault == 'glob' else args.task * 2)
    else:
        directory = args.directory / sorted(profile['cases'])[-1]
        if fault == 'missing-task':
            (directory / 'task.toml').unlink()
        else:
            path = directory / 'tests/task.json'
            task = json.loads(path.read_text())
            if fault == 'evaluation':
                task['evaluation_id'] += 1
            else:
                task['pr'][fault] = 'c' * 40
            path.write_text(json.dumps(task))
    with pytest.raises(ValueError, match='exact|identity|membership'):
        harbor_cli.handle(args)
    assert not args.jobs_dir.exists() and not (args.directory / 'format-checked').exists()


@pytest.mark.parametrize('schema', [None, 1, 2, 3])
def test_launch_rejects_incompatible_profile_before_process_delivery(
    launch: argparse.Namespace, schema: int | None,
) -> None:
    """A valid digest cannot admit an incompatible profile schema."""
    args = launch
    path = args.directory / 'daydream-profile.json'
    profile = json.loads(path.read_text())
    profile.pop('policy_sha256')
    profile['schema'] = schema
    path.write_text(json.dumps({**profile, 'policy_sha256': canonical_digest(profile)}))
    with pytest.raises(ValueError, match='Incompatible public profile schema; create a fresh Harbor export'):
        harbor_cli.handle(args)
    assert not (args.directory / 'format-checked').exists()
    assert not args.jobs_dir.exists()


@pytest.mark.parametrize('wrong', ['released', 'different-commit', 'different-reviewbench'])
def test_unpinned_runtime_refuses_before_process_or_credential_delivery(
    launch: argparse.Namespace, monkeypatch: pytest.MonkeyPatch, wrong: str,
) -> None:
    """Reject same-version runtimes missing the pinned Harbor or ReviewBench contract."""
    args = launch
    if wrong == 'different-reviewbench':
        monkeypatch.setattr(importlib.metadata, 'version', lambda _name: '0.1.0')
    else:
        direct = {} if wrong == 'released' else {'vcs_info': {'commit_id': '0' * 40}}
        monkeypatch.setattr(importlib.metadata, 'distribution', lambda _name: SimpleNamespace(
            version='0.24.0', read_text=lambda _file: json.dumps(direct)))
    with pytest.raises(ValueError, match='pinned Harbor|reviewbench-harbor'):
        harbor_cli.handle(args)
    assert not (args.directory / 'format-checked').exists()
    assert not args.jobs_dir.exists()


@pytest.mark.parametrize('boundary', ['dataset', 'jobs'])
def test_auth_mount_cannot_live_in_retained_dataset_or_job_evidence(
    launch: argparse.Namespace, boundary: str,
) -> None:
    """Private credential stores must stay outside recursive evidence roots."""
    args = launch
    args.pi_home = (args.directory if boundary == 'dataset' else args.jobs_dir) / 'credentials'
    with pytest.raises(ValueError, match='outside dataset and job evidence'):
        harbor_cli.handle(args)
    assert not (args.directory / 'format-checked').exists()


def test_kernel_isolation_failure_stops_before_auth_delivery(
    launch: argparse.Namespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing exact-host kernel enforcement must prevent credential delivery."""
    from harbor.environments.docker.docker import DockerEnvironment

    args = launch
    monkeypatch.setattr(DockerEnvironment, '_egress_control_kernel_support', staticmethod(lambda: False))
    with pytest.raises(ValueError, match='exact-host isolation'):
        harbor_cli.handle(args)
    assert not (args.directory / 'format-checked').exists()


@pytest.mark.parametrize('selection', ['test', 'full'])
def test_local_dataset_launch_builds_actual_harbor_reviewbench_plan_before_models(
    launch: argparse.Namespace, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, supplied_source: Path, moby_policy: bytes, selection: str,
) -> None:
    """Resolve actual pinned CLI, Job, plugin and metrics; reject the empty-metric control."""
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
from reviewbench_harbor.reviewbench_plugin import ReviewBenchPlugin, RoundMetric
from daydream.benchmark.reviewbench.harbor_cli import verify_runtime

async def main():
    verify_runtime()
    config = JobConfig.model_validate_json(Path(sys.argv[1]).read_text())
    control = config.model_copy(deep=True)
    control.job_name = "no-metric-control"
    control.metrics = []
    control_job = await Job.create(control)
    try:
        await ReviewBenchPlugin().on_job_start(control_job)
    except ValueError as error:
        assert "frozen ReviewBench uv-script metric" in str(error)
    else:
        raise AssertionError("unbound local metric unexpectedly produced a ReviewBench plan")
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
    for trial in job._trial_configs:
        downloaded = job._task_download_results[trial.task.get_task_id()]
        task = json.loads((downloaded.path / "tests/task.json").read_text())
        actual_case = {key: task["pr"][key] for key in ("repo", "pr_number", "base", "head")}
        assert actual_case == expected["cases"][downloaded.path.name]
    metric_digest = hashlib.sha256(Path(config.metrics[0].kwargs["script_path"]).read_bytes()).hexdigest()
    assert plan["source_metric_sha256"] == [metric_digest]
    assert all(isinstance(metric, RoundMetric) for group in job._metrics.values() for metric in group)
    metric = next(metric for group in job._metrics.values() for metric in group)
    rewards = [{
        "rb_schema": 1, "rb_task": row["task_id"], "rb_evaluation": row["evaluation_id"],
        "rb_round": row["round"], "rb_failed": 0,
        "overall.candidate_count": 0, "overall.golden_tp_count": 1, "overall.matched_count": 0,
        "overall.matched_tp_count": 0, "overall.unmatched_count": 0, "overall.unmatched_tp_count": 0,
        "overall.unmatched_fp_count": 0, "overall.golden_covered_count": 0,
    } for row in plan["trials"].values()]
    computed = metric.compute(rewards)
    assert computed["complete"] == 1 and computed["expected_trials"] == len(rewards)
    assert computed["source_prs"] == 219 and computed["completed_rounds"] == 3
    assert computed["summary.micro.overall.grounded_precision.defined_rounds"] == 0
    assert "summary.micro.overall.grounded_precision.mean" not in computed
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

    monkeypatch.setattr(subprocess, 'call', resolve_only)
    # A plan without retained results cannot establish completed work.
    assert harbor_cli.handle(args) == 1
    actual = json.loads(result_path.read_text())
    assert actual['plan']['source_prs'] == 219
    assert actual['plan']['expected_tasks'] == expected['tasks']
    assert len(actual['plan']['trials']) == (75 if selection == 'test' else 657)
    assert actual['metric'] == {'type': 'uv-script', 'kwargs': {'script_path': str(args.directory / 'metric.py')}}
    assert len(list(args.jobs_dir.glob('daydream-reviewbench-launch-*.json'))) == 1


@pytest.mark.parametrize('outcome', [
    'complete', 'reviewer-error', 'missing-score', 'missing-trial', 'wrong-round', 'duplicate', 'null',
    'transport-error',
])
def test_successful_harbor_process_requires_actual_complete_retained_job(
    launch: argparse.Namespace, supplied_source: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    """Use the supplied checker and metric to reject incomplete retained jobs despite Harbor success."""
    args = launch
    retained = tmp_path / 'harbor-results'
    retained.mkdir()
    metric = (supplied_source / 'scripts/harbor/metric.py').read_text().replace(
        'EXPECTED_TASKS = {}', 'EXPECTED_TASKS = {"1": 2}',
    )
    (retained / 'reviewbench-metric.py').write_text(metric)
    metric_sha = hashlib.sha256(metric.encode()).hexdigest()
    job: dict[str, Any] = {
        'id': 'fixture-job', 'finished_at': '2026-10-21T00:00:00Z', 'n_total_trials': 3,
        'stats': {'n_errored_trials': 0, 'n_running_trials': 0, 'n_pending_trials': 0, 'n_cancelled_trials': 0},
    }
    plan = {
        'schema_version': 1, 'job_id': job['id'], 'metric_sha256': metric_sha,
        'source_metric_sha256': [metric_sha], 'expected_tasks': {'1': 2}, 'n_attempts': 3, 'source_prs': 219,
        'trials': {f'trial-{number}': {'task_id': 1, 'evaluation_id': 2, 'round': number}
                   for number in (1, 2, 3)},
    }
    reward = {
        'rb_schema': 1, 'rb_task': 1, 'rb_evaluation': 2, 'rb_round': 1, 'rb_failed': 0,
        'overall.candidate_count': 1, 'overall.golden_tp_count': 1, 'overall.matched_count': 1,
        'overall.matched_tp_count': 1, 'overall.unmatched_count': 0, 'overall.unmatched_tp_count': 0,
        'overall.unmatched_fp_count': 0, 'overall.golden_covered_count': 1,
    }
    trial: dict[str, Any] = {
        'trial_name': 'trial-1', 'finished_at': job['finished_at'], 'exception_info': None,
        'verifier_environment_mode': 'separate', 'step_results': [], 'verifier_result': {'rewards': reward},
    }
    if outcome == 'reviewer-error':
        job['stats']['n_errored_trials'] = 1
        trial['exception_info'] = {'exception_type': 'RuntimeError', 'exception_message': 'native reviewer failed'}
    elif outcome == 'missing-score':
        trial['verifier_result'] = None
    elif outcome == 'wrong-round':
        reward['rb_round'] = 4
    elif outcome == 'null':
        for key in reward:
            if key.startswith('overall.') and key != 'overall.golden_tp_count':
                reward[key] = 0
    elif outcome == 'transport-error':
        monkeypatch.setenv('DAYDREAM_TEST_HARBOR_EXIT', '23')
    for relative, value in [('result.json', job), ('reviewbench-plan.json', plan)]:
        (retained / relative).write_text(json.dumps(value))
    for number in (1, 2, 3):
        if outcome == 'missing-trial' and number == 3:
            continue
        current = json.loads(json.dumps(trial))
        current['trial_name'] = f'trial-{1 if outcome == "duplicate" else number}'
        if outcome != 'wrong-round' and current['verifier_result']:
            current['verifier_result']['rewards']['rb_round'] = number
        directory = retained / f'trial-{number}'
        directory.mkdir()
        (directory / 'result.json').write_text(json.dumps(current))
    prior_name = 'daydream-reviewbench-check-prior.json'
    (retained / prior_name).write_text('preserve prior report bytes')
    monkeypatch.setenv('DAYDREAM_TEST_HARBOR_RESULTS', str(retained))
    monkeypatch.setenv('DAYDREAM_TEST_REVIEWBENCH_CHECKER', str(supplied_source / 'scripts/harbor/check-job.py'))
    result = harbor_cli.handle(args)
    job_directory = args.jobs_dir / args.job_name
    assert (job_directory / prior_name).read_text() == 'preserve prior report bytes'
    assert (job_directory / 'result.json').read_bytes() == (retained / 'result.json').read_bytes()
    reports = [path for path in job_directory.glob('daydream-reviewbench-check-*.json') if path.name != prior_name]
    if outcome in {'complete', 'null'}:
        assert result == 0 and len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report['complete'] == 1 and report['scored_trials'] == report['expected_trials'] == 3
        assert report['source_prs'] == 219 and report['expected_prs'] == 1
        assert report['n_attempts'] == 3 and report['failed_trials'] == 0
        assert report['completed_rounds'] == 3
        assert report['summary.micro.overall.grounded_precision.defined_rounds'] == (0 if outcome == 'null' else 3)
        if outcome == 'null':
            assert 'summary.micro.overall.grounded_precision.mean' not in report
        else:
            assert report['summary.micro.overall.grounded_precision.mean'] == 1
    else:
        assert result == (23 if outcome == 'transport-error' else 1)
        assert reports == []
    assert (job_directory / 'checker-invoked').exists() is (outcome != 'transport-error')


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


@pytest.mark.parametrize('fault', ['same-store', 'nested-store', 'copied-session', 'personal-home', 'judge-api'])
def test_launch_refuses_shared_sessions_or_broad_credential_mounts(
    launch: argparse.Namespace, fault: str,
) -> None:
    """Reject overlapping stores, copied OAuth sessions, personal state and judge API fallback."""
    args = launch
    if fault == 'same-store':
        args.pi_home = args.codex_home
    elif fault == 'nested-store':
        args.pi_home = args.codex_home / 'judge'
    elif fault == 'copied-session':
        path = args.pi_home / 'auth.json'
        auth = json.loads(path.read_text())
        auth['openai-codex']['refresh'] = 'synthetic-original-refresh'
        path.write_text(json.dumps(auth))
    elif fault == 'personal-home':
        (args.codex_home / 'config.toml').write_text('model = "unrelated"')
    else:
        (args.pi_home / 'auth.json').write_text(json.dumps({'openai-codex': {
            'type': 'api_key', 'key': 'synthetic-forbidden-key',
        }}))
    with pytest.raises(ValueError, match='separate|copied OAuth|mount only|dedicated Pi'):
        harbor_cli.handle(args)
    assert not (args.directory / 'format-checked').exists()
    assert not args.jobs_dir.exists()
