"""Public Harbor launch/package contracts at external-process boundaries.

Existing export tests own the 219 supplied tasks and format digests. These tests
protect the later operator launch, which can otherwise select the wrong runtime,
package, rounds or authentication route despite a valid export. Harmless process
replays replace Harbor and its format checker; no container or inference runs.
"""
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_build, harbor_cli
from daydream.benchmark.reviewbench.harbor_profile import canonical_digest
from tests.test_reviewbench_evaluator import supplied_source as supplied_source
from tests.test_reviewbench_sandbox_probe import moby_policy as moby_policy


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, dict[str, Any]]:
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
        '  pathlib.Path(sys.argv[-1]).write_text(json.dumps({"complete": 1}))\n'
        ' raise SystemExit(0)\n'
        'assert sys.argv[1:3] == ["-m", "reviewbench_harbor.check-format"]\n'
        'assert (pathlib.Path(sys.argv[3]) / "daydream-profile.json").is_file()\n'
        '(pathlib.Path(sys.argv[3]) / "format-checked").touch()\n'
    )
    format_checker.chmod(0o755)
    harbor = binary / 'harbor'
    harbor.write_text(
        f'#!{interpreter}\n'
        'import argparse, json, os, pathlib, shutil\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("command", choices=["run"])\n'
        'for flag, name in [("-c", "config"), ("-p", "dataset"), ("-a", "agent"), ("-n", "concurrency"),\n'
        ' ("-k", "rounds"), ("-r", "retries"), ("-o", "jobs"),\n'
        ' ("--job-name", "job"), ("-i", "task"), ("-l", "limit")]:\n'
        ' parser.add_argument(flag, dest=name)\n'
        'parser.add_argument("--yes", action="store_true")\n'
        'parser.add_argument("--plugin", dest="plugins", action="append")\n'
        'args = parser.parse_args()\n'
        'auth_dir = pathlib.Path(os.environ["RB_HARBOR_AUTH_DIR"])\n'
        'auth = json.loads((auth_dir / "auth.json").read_text())\n'
        'assert auth["auth_mode"] == "chatgpt" and not auth["OPENAI_API_KEY"]\n'
        'auth["tokens"]["refresh_token"] = "synthetic-process-rotation"\n'
        '(auth_dir / "auth.json").write_text(json.dumps(auth))\n'
        'job = pathlib.Path(args.jobs) / args.job\n'
        'job.mkdir(parents=True)\n'
        'if results := os.environ.get("DAYDREAM_TEST_HARBOR_RESULTS"):\n'
        ' shutil.copytree(results, job, dirs_exist_ok=True)\n'
        'record = vars(args)\n'
        'record["metric_config"] = json.loads(pathlib.Path(args.config).read_text())\n'
        'record.update(round_membership=list(range(1, int(args.rounds) + 1)),\n'
        ' auth_home=os.environ["RB_HARBOR_SUBSCRIPTION_HOME"],\n'
        ' seccomp_profile=os.environ["RB_HARBOR_SECCOMP_PROFILE"],\n'
        ' platform=os.environ.get("DOCKER_DEFAULT_PLATFORM"),\n'
        ' forbidden_env=sorted(k for k in os.environ if k in ["OPENAI_API_KEY",\n'
        ' "OPENAI_BASE_URL", "CODEX_BEARER_TOKEN", "OPENAI_WORKLOAD_IDENTITY_TOKEN"]))\n'
        '(job / "launch.json").write_text(json.dumps(record))\n'
        'raise SystemExit(int(os.environ.get("DAYDREAM_TEST_HARBOR_EXIT", "0")))\n'
    )
    harbor.chmod(0o755)
    monkeypatch.setattr(harbor_cli, 'sys', SimpleNamespace(executable=str(format_checker)))
    monkeypatch.setattr(DockerEnvironment, '_egress_control_kernel_support', staticmethod(lambda: True))
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda _name: SimpleNamespace(
        version='0.24.0',
        read_text=lambda _file: json.dumps({'vcs_info': {'commit_id': harbor_build.HARBOR_REVISION}})))
    monkeypatch.setattr(importlib.metadata, 'version', lambda _name: '0.1.0rc10')
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
    candidate = {'artifact_sha256': 'a' * 64, 'version': '0.30.0'}
    policy = {'source_prs': 219, 'final_comparison_rounds': 3, 'candidate': candidate}
    (dataset / 'daydream-profile.json').write_text(json.dumps({**policy, 'policy_sha256': canonical_digest(policy)}))
    parser = argparse.ArgumentParser()
    harbor_cli.add_parsers(parser.add_subparsers(dest='reviewbench_command', required=True))
    args = parser.parse_args([
        'harbor-run', str(dataset), '--codex-home', str(source), '--auth-runtime', str(tmp_path / 'private-auth'),
        '--jobs-dir', str(tmp_path / 'jobs'), '--job-name', 'selected-case', '--task', 'owner_repo*',
        '--n-tasks', '1', '--concurrency', '2',
    ])
    return args, candidate


@pytest.mark.parametrize(('rounds', 'exit_code'), [(1, 0), (3, 23)])
def test_launch_delivers_selection_independent_rounds_and_returns_refresh(
    launch: tuple[argparse.Namespace, dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
    rounds: int, exit_code: int,
) -> None:
    """A valid export alone misses launch flag drift, inherited billing and failed-job rotation loss."""
    args, candidate = launch
    args.rounds = rounds
    monkeypatch.setenv('DAYDREAM_TEST_HARBOR_EXIT', str(exit_code))
    monkeypatch.setenv('DOCKER_DEFAULT_PLATFORM', 'linux/amd64')
    monkeypatch.setenv('RB_HARBOR_SECCOMP_PROFILE', 'unconfined')
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CODEX_BEARER_TOKEN', 'OPENAI_WORKLOAD_IDENTITY_TOKEN'):
        monkeypatch.setenv(key, 'synthetic-forbidden-route')
    assert harbor_cli.handle(args, candidate) == exit_code
    actual = json.loads((args.jobs_dir / args.job_name / 'launch.json').read_text())
    local_config = Path(actual.pop('config'))
    assert local_config.parent == args.jobs_dir.resolve() and local_config.is_file()
    assert actual == {
        'command': 'run', 'dataset': str(args.directory.resolve()),
        'metric_config': {'metrics': [{'type': 'uv-script',
                                      'kwargs': {'script_path': str(args.directory.resolve() / 'metric.py')}}]},
        'agent': 'daydream.benchmark.reviewbench.harbor_agent:DaydreamPublicReviewAgent',
        'concurrency': '2', 'rounds': str(rounds), 'retries': '0', 'jobs': str(args.jobs_dir.resolve()),
        'job': args.job_name, 'yes': True, 'plugins': [
            'reviewbench_harbor.reviewbench_plugin:ReviewBenchPlugin',
            'daydream.benchmark.reviewbench.harbor_subscription_plugin:SubscriptionAdmissionPlugin'],
        'task': 'owner_repo*', 'limit': '1', 'round_membership': list(range(1, rounds + 1)),
        'auth_home': '/run/reviewbench-codex', 'platform': None, 'forbidden_env': [],
        'seccomp_profile': str(args.directory.resolve() / 'reviewer-seccomp.json'),
    }
    assert (args.directory / 'format-checked').is_file()
    assert not (args.auth_runtime / 'auth.json').exists()
    source = json.loads((args.codex_home / 'auth.json').read_text())
    assert source['tokens']['refresh_token'] == 'synthetic-process-rotation'
    assert 'synthetic-process-rotation' not in (args.jobs_dir / args.job_name / 'launch.json').read_text()


@pytest.mark.parametrize('wrong', ['released', 'different-commit', 'different-reviewbench'])
def test_unpinned_runtime_refuses_before_process_or_credential_delivery(
    launch: tuple[argparse.Namespace, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, wrong: str,
) -> None:
    """Released Harbor reports the same version while lacking the required metric contract."""
    args, candidate = launch
    if wrong == 'different-reviewbench':
        monkeypatch.setattr(importlib.metadata, 'version', lambda _name: '0.1.0')
    else:
        direct = {} if wrong == 'released' else {'vcs_info': {'commit_id': '0' * 40}}
        monkeypatch.setattr(importlib.metadata, 'distribution', lambda _name: SimpleNamespace(
            version='0.24.0', read_text=lambda _file: json.dumps(direct)))
    with pytest.raises(ValueError, match='pinned Harbor|reviewbench-harbor'):
        harbor_cli.handle(args, candidate)
    assert not (args.directory / 'format-checked').exists()
    assert not args.auth_runtime.exists()
    assert not args.jobs_dir.exists()


@pytest.mark.parametrize('boundary', ['dataset', 'jobs'])
def test_auth_mount_cannot_live_in_retained_dataset_or_job_evidence(
    launch: tuple[argparse.Namespace, dict[str, Any]], boundary: str,
) -> None:
    """A valid OAuth capability must never be published under recursive evidence roots."""
    args, candidate = launch
    args.auth_runtime = (args.directory if boundary == 'dataset' else args.jobs_dir) / 'credentials'
    with pytest.raises(ValueError, match='outside dataset and job evidence'):
        harbor_cli.handle(args, candidate)
    assert not args.auth_runtime.exists()
    assert not (args.directory / 'format-checked').exists()


@pytest.mark.parametrize('changed', ['artifact_sha256', 'version'])
def test_launch_refuses_changed_executing_candidate_before_models(
    launch: tuple[argparse.Namespace, dict[str, Any]], changed: str,
) -> None:
    """A host adapter from another package must not run under the frozen wheel's identity."""
    args, candidate = launch
    with pytest.raises(ValueError, match='Executing candidate differs'):
        harbor_cli.handle(args, {**candidate, changed: 'different'})
    assert not args.auth_runtime.exists()
    assert not (args.directory / 'format-checked').exists()


def test_kernel_isolation_failure_stops_before_auth_delivery(
    launch: tuple[argparse.Namespace, dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact-host enforcement cannot silently degrade into a public-network model run."""
    from harbor.environments.docker.docker import DockerEnvironment

    args, candidate = launch
    monkeypatch.setattr(DockerEnvironment, '_egress_control_kernel_support', staticmethod(lambda: False))
    with pytest.raises(ValueError, match='exact-host isolation'):
        harbor_cli.handle(args, candidate)
    assert not args.auth_runtime.exists()
    assert not (args.directory / 'format-checked').exists()


@pytest.mark.parametrize('damage', ['missing-module', 'changed-module', 'injected-extension'])
def test_reviewer_wheel_must_equal_executing_package(tmp_path: Path, damage: str) -> None:
    """Checking only wheel-present bytes admits truncated wheels under a full-package identity.

    Existing installed-wheel coverage proves a good wheel executes; this negative
    control protects the export admission boundary against absent or changed code.
    """
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(tmp_path / 'built')],
                   cwd=repository, check=True, capture_output=True)
    original = next((tmp_path / 'built').glob('*.whl'))
    changed = tmp_path / original.name
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(changed, 'w') as destination:
        for name in source.namelist():
            data = source.read(name)
            if name == 'daydream/__init__.py':
                if damage == 'missing-module':
                    continue
                if damage == 'changed-module':
                    data += b'\n# Different candidate package bytes\n'
            destination.writestr(name, data)
        if damage == 'injected-extension':
            destination.writestr('daydream_ext/__init__.py', 'DAYDREAM_EXT_API = 8\ndef register(registry): pass\n')
    with pytest.raises(ValueError, match='package membership|same candidate package'):
        harbor_build.reviewer_build(changed)


def test_local_dataset_launch_builds_actual_harbor_reviewbench_plan_before_models(
    launch: tuple[argparse.Namespace, dict[str, Any]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, supplied_source: Path, moby_policy: bytes,
) -> None:
    """A local -p export needs explicit metrics before the upstream plugin can plan.

    The argv replay cannot detect Harbor's different metric resolution for local
    and Hub datasets. Resolve the actual CLI configuration, Job and both plugins
    in the pinned optional runtime without running any trial or container. The
    identical empty-metrics control reproduces the observed pre-inference failure.
    """
    from daydream.benchmark.reviewbench.harbor_profile import export_harbor_profile

    selected = os.environ.get('DAYDREAM_REVIEWBENCH_RUNTIME')
    if not selected:
        pytest.skip('set DAYDREAM_REVIEWBENCH_RUNTIME to the pinned public runtime Python')
    runtime = Path(selected).absolute()
    assert runtime.is_file()
    args, candidate = launch
    args.directory = tmp_path / 'actual-full-export'
    args.task = None
    export_harbor_profile(supplied_source, args.directory, candidate=candidate, base_image=harbor_build.NODE_IMAGE)
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
        PluginConfig(import_path="daydream.benchmark.reviewbench.harbor_subscription_plugin:SubscriptionAdmissionPlugin"),
    ])
    plan = json.loads((job.job_dir / "reviewbench-plan.json").read_text())
    assert plan["source_prs"] == 219 and len(plan["expected_tasks"]) == 1
    assert plan["n_attempts"] == 1 and len(plan["trials"]) == 1
    metric_digest = hashlib.sha256(Path(config.metrics[0].kwargs["script_path"]).read_bytes()).hexdigest()
    assert plan["source_metric_sha256"] == [metric_digest]
    assert all(isinstance(metric, RoundMetric) for group in job._metrics.values() for metric in group)
    assert (job.job_dir / "reviewbench-metric.py").is_file()
    assert json.loads((job.job_dir / "daydream-subscription-admission.json").read_text())["effective_concurrency"] == 1
    Path(sys.argv[2]).write_text(json.dumps({"plan": plan, "metric": config.metrics[0].model_dump(mode="json")}))

asyncio.run(main())
''')
    result_path = tmp_path / 'resolved-plan.json'
    merged_config = tmp_path / 'resolved-config.json'
    repository = Path(__file__).resolve().parents[1]

    def resolve_only(command: list[str], *, env: dict[str, str]) -> int:
        # --print-config returns before execution. No Job.run, Trial.create,
        # environment build, reviewer or verifier is invoked by this proof.
        environment = {**env, 'PYTHONPATH': str(repository)}
        printed = subprocess.run([*command, '--print-config'], env=environment, check=True,
                                 text=True, capture_output=True, timeout=60)
        merged_config.write_text(printed.stdout)
        subprocess.run([str(runtime), str(script), str(merged_config), str(result_path)], env=environment,
                       check=True, text=True, capture_output=True, timeout=120)
        return 0

    monkeypatch.setattr(subprocess, 'call', resolve_only)
    # This proof resolves a plan without running trials; a successful Harbor
    # process with no retained result must now fail closed at the CLI boundary.
    assert harbor_cli.handle(args, candidate) == 1
    actual = json.loads(result_path.read_text())
    assert actual['plan']['source_prs'] == 219
    assert len(actual['plan']['expected_tasks']) == 1
    assert actual['metric'] == {'type': 'uv-script', 'kwargs': {'script_path': str(args.directory / 'metric.py')}}
    assert len(list(args.jobs_dir.glob('daydream-reviewbench-launch-*.json'))) == 1


@pytest.mark.parametrize('outcome', ['complete', 'reviewer-error', 'missing-score', 'wrong-round', 'transport-error'])
def test_successful_harbor_process_requires_actual_complete_retained_job(
    launch: tuple[argparse.Namespace, dict[str, Any]], supplied_source: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    """A real Harbor exit zero can hide reviewer exceptions and absent rewards.

    Retain the existing process-boundary launch, use the actual supplied checker
    and metric against real job/plan/trial files, and vary only the failed evidence.
    Previous argv replay coverage modeled process success as benchmark success.
    """
    args, candidate = launch
    retained = tmp_path / 'harbor-results'
    (retained / 'trial').mkdir(parents=True)
    metric = (supplied_source / 'scripts/harbor/metric.py').read_text().replace(
        'EXPECTED_TASKS = {}', 'EXPECTED_TASKS = {"1": 2}',
    )
    (retained / 'reviewbench-metric.py').write_text(metric)
    metric_sha = hashlib.sha256(metric.encode()).hexdigest()
    job: dict[str, Any] = {
        'id': 'fixture-job', 'finished_at': '2026-10-21T00:00:00Z', 'n_total_trials': 1,
        'stats': {'n_errored_trials': 0, 'n_running_trials': 0, 'n_pending_trials': 0, 'n_cancelled_trials': 0},
    }
    plan = {
        'schema_version': 1, 'job_id': job['id'], 'metric_sha256': metric_sha,
        'source_metric_sha256': [metric_sha], 'expected_tasks': {'1': 2}, 'n_attempts': 1, 'source_prs': 219,
        'trials': {'trial': {'task_id': 1, 'evaluation_id': 2, 'round': 1}},
    }
    reward = {
        'rb_schema': 1, 'rb_task': 1, 'rb_evaluation': 2, 'rb_round': 1, 'rb_failed': 0,
        'overall.candidate_count': 1, 'overall.golden_tp_count': 1, 'overall.matched_count': 1,
        'overall.matched_tp_count': 1, 'overall.unmatched_count': 0, 'overall.unmatched_tp_count': 0,
        'overall.unmatched_fp_count': 0, 'overall.golden_covered_count': 1,
    }
    trial: dict[str, Any] = {
        'trial_name': 'trial', 'finished_at': job['finished_at'], 'exception_info': None,
        'verifier_environment_mode': 'separate', 'step_results': [], 'verifier_result': {'rewards': reward},
    }
    if outcome == 'reviewer-error':
        job['stats']['n_errored_trials'] = 1
        trial['exception_info'] = {'exception_type': 'RuntimeError', 'exception_message': 'native reviewer failed'}
    elif outcome == 'missing-score':
        trial['verifier_result'] = None
    elif outcome == 'wrong-round':
        reward['rb_round'] = 3
    elif outcome == 'transport-error':
        monkeypatch.setenv('DAYDREAM_TEST_HARBOR_EXIT', '23')
    for relative, value in [('result.json', job), ('reviewbench-plan.json', plan), ('trial/result.json', trial)]:
        (retained / relative).write_text(json.dumps(value))
    prior_name = 'daydream-reviewbench-check-prior.json'
    (retained / prior_name).write_text('preserve prior report bytes')
    monkeypatch.setenv('DAYDREAM_TEST_HARBOR_RESULTS', str(retained))
    monkeypatch.setenv('DAYDREAM_TEST_REVIEWBENCH_CHECKER', str(supplied_source / 'scripts/harbor/check-job.py'))
    result = harbor_cli.handle(args, candidate)
    job_directory = args.jobs_dir / args.job_name
    assert (job_directory / prior_name).read_text() == 'preserve prior report bytes'
    assert (job_directory / 'result.json').read_bytes() == (retained / 'result.json').read_bytes()
    reports = [path for path in job_directory.glob('daydream-reviewbench-check-*.json') if path.name != prior_name]
    if outcome == 'complete':
        assert result == 0 and len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report['complete'] == 1 and report['scored_trials'] == report['expected_trials'] == 1
        assert report['source_prs'] == 219 and report['expected_prs'] == 1
        assert report['n_attempts'] == 1 and report['failed_trials'] == 0
    else:
        assert result == (23 if outcome == 'transport-error' else 1)
        assert reports == []
    assert (job_directory / 'checker-invoked').exists() is (outcome != 'transport-error')
    # Authentication refresh still reconciles before result validation, including
    # when the retained job subsequently fails its strict completion checks.
    source_auth = json.loads((args.codex_home / 'auth.json').read_text())
    assert source_auth['tokens']['refresh_token'] == 'synthetic-process-rotation'
    assert not (args.auth_runtime / 'auth.json').exists()


@pytest.mark.parametrize('selection', ['environment', 'importable-default'])
def test_controlled_export_refuses_unowned_extensions_before_source_or_wheel_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], selection: str,
) -> None:
    """External extension inputs must not silently disappear from the image's reviewer.

    The existing package guard tested core-only wheel damage. Enter the real CLI
    with a valid external extension selection and prove admission stops before
    exporting tasks, reading unrelated inputs, or executing extension code.
    """
    from daydream.benchmark.cli import _handle_benchmark_command

    outside = tmp_path / 'outside' / 'daydream_ext'
    outside.mkdir(parents=True)
    sentinel = tmp_path / 'extension-executed'
    (outside / '__init__.py').write_text(
        f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n'
        'DAYDREAM_EXT_API = 8\ndef register(registry): pass\n',
    )
    (outside / 'auth.json').write_text('{"access_token":"unrelated-private-fixture"}')
    monkeypatch.delitem(sys.modules, 'daydream_ext', raising=False)
    if selection == 'environment':
        monkeypatch.setenv('DAYDREAM_EXT_DIR', str(outside))
    else:
        monkeypatch.delenv('DAYDREAM_EXT_DIR', raising=False)
        monkeypatch.syspath_prepend(str(outside.parent))
    output = tmp_path / 'export'
    assert _handle_benchmark_command([
        'reviewbench', 'harbor-export', str(output), '--source', str(tmp_path / 'unused-source'),
        '--wheel', str(tmp_path / 'unused-wheel.whl'),
    ]) == 1
    error = capsys.readouterr().err
    assert 'package the extension in the candidate wheel and unset DAYDREAM_EXT_DIR' in error
    assert not output.exists() and not sentinel.exists()
    assert 'unrelated-private-fixture' not in error


def test_packaged_extension_code_and_resources_bind_candidate_and_reviewer_wheel(tmp_path: Path) -> None:
    """A real fork package must retain native extension behavior with complete identity.

    Fresh isolated interpreters load the actual ordinary registry and candidate
    CLI from the fork. Resource and code changes alter genuine prompt behavior;
    both must change identity and invalidate the older reviewer wheel.
    """
    repository = Path(__file__).resolve().parents[1]
    fork = tmp_path / 'fork'
    fork.mkdir()
    shutil.copytree(repository / 'daydream', fork / 'daydream', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(repository / 'pyproject.toml', fork / 'pyproject.toml')
    extension = fork / 'daydream_ext'
    extension.mkdir()
    initial = (
        'from pathlib import Path\nDAYDREAM_EXT_API = 8\n'
        'def prompt(**kwargs): return Path(__file__).with_name("policy.txt").read_text()\n'
        'def register(registry): registry.override_prompt("reviewbench_identity_probe", prompt)\n'
    )
    (extension / '__init__.py').write_text(initial)
    (extension / 'policy.txt').write_text('original policy')
    subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(tmp_path / 'wheels')],
                   cwd=fork, check=True, capture_output=True)
    wheel = next((tmp_path / 'wheels').glob('*.whl'))
    environment = {**os.environ, 'PYTHONPATH': str(fork)}
    environment.pop('DAYDREAM_EXT_DIR', None)
    script = '''import json,sys
from pathlib import Path
from daydream.benchmark.reviewbench.cli import candidate_identity
from daydream.benchmark.reviewbench.harbor_build import reviewer_build
from daydream.extensions.loader import build_registry
error=None
try:
    reviewer_build(Path(sys.argv[1]))
except ValueError as failure:
    error=str(failure)
print(json.dumps({"candidate":candidate_identity(), "wheel_error":error,
                  "prompt":build_registry().prompt("reviewbench_identity_probe")()}))
'''

    def inspect(wheel_path: Path) -> dict[str, Any]:
        result = subprocess.run([sys.executable, '-c', script, str(wheel_path)], cwd=tmp_path,
                                env=environment, check=True, capture_output=True, text=True)
        value: dict[str, Any] = json.loads(result.stdout)
        return value

    original = inspect(wheel)
    assert original['wheel_error'] is None and original['prompt'] == 'original policy'
    assert original['candidate']['default_extension']['source_kind'] == 'packaged'
    assert original['candidate']['default_extension']['files'] == 2
    missing = tmp_path / wheel.name
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(missing, 'w') as destination:
        for name in source.namelist():
            if not name.startswith('daydream_ext/'):
                destination.writestr(name, source.read(name))
    assert 'package membership' in inspect(missing)['wheel_error']
    previous = original
    for path, contents, expected in (
        (extension / 'policy.txt', 'changed resource', 'changed resource'),
        (extension / '__init__.py', initial.replace('return Path(', 'return "changed code: " + Path('),
         'changed code: changed resource'),
    ):
        path.write_text(contents)
        changed = inspect(wheel)
        assert changed['prompt'] == expected
        assert changed['candidate']['artifact_sha256'] != previous['candidate']['artifact_sha256']
        assert changed['candidate']['default_extension']['sha256'] != (
            previous['candidate']['default_extension']['sha256'])
        assert 'same candidate package' in changed['wheel_error']
        previous = changed
