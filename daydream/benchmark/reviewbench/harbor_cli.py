"""The single supplied ReviewBench workflow: native export and Harbor launch."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from daydream.benchmark.reviewbench.candidate import candidate_identity
from daydream.benchmark.reviewbench.contract import Case
from daydream.benchmark.reviewbench.harbor_build import (
    HARBOR_REVISION,
    NODE_IMAGE,
    REVIEWBENCH_PLATFORMS,
    reviewer_build,
)


def add_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    export = commands.add_parser(
        'harbor-export', help='export all 219 tasks with a selected native reviewer; no inference',
    )
    export.add_argument('directory', type=Path)
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--wheel', type=Path, required=True)
    export.epilog = 'Append -- followed by ordinary native review flags and configuration root.'
    export.add_argument('--base-image', default=NODE_IMAGE)
    export.add_argument('--platform', choices=REVIEWBENCH_PLATFORMS, default='linux/amd64',
                        help='frozen task architecture')
    export.add_argument('--rounds', type=int, choices=(1, 3), default=3)
    run = commands.add_parser('harbor-run', help='run the exported public profile; launches models')
    run.add_argument('directory', type=Path)
    run.add_argument('--codex-home', type=Path,
                     help='persistent auth-only directory with a dedicated Codex subscription login')
    run.add_argument('--reviewer-pi-home', type=Path,
                     help='dedicated native Pi reviewer credential directory, separate from the judge')
    run.add_argument('--pi-home', type=Path, required=True,
                     help='persistent auth-only directory with an independently signed-in Pi judge login')
    run.add_argument('--jobs-dir', type=Path, required=True)
    run.add_argument('--job-name', required=True)
    selection = run.add_mutually_exclusive_group()
    selection.add_argument('--set', choices=('test', 'full'),
                           help='exact supplied test25 or full219 membership; default full')
    selection.add_argument('--task', action='append',
                           help='exact task name for an explicitly identified subset; repeatable')
    run.add_argument('--rounds', type=int, choices=(1, 3), default=1)


def verify_runtime() -> None:
    """Released Harbor has the same version but lacks the public metric contract."""
    try:
        distribution = importlib.metadata.distribution('harbor')
        direct = json.loads(distribution.read_text('direct_url.json') or '{}')
        if distribution.version != '0.24.0' or direct.get('vcs_info', {}).get('commit_id') != HARBOR_REVISION:
            raise ValueError('Public ReviewBench requires the shipped pinned Harbor runtime, not released 0.24.0')
        if importlib.metadata.version('reviewbench-harbor') != '0.1.0rc10':
            raise ValueError('Public ReviewBench requires reviewbench-harbor==0.1.0rc10')
    except importlib.metadata.PackageNotFoundError:
        raise ValueError('Install the shipped harbor-runtime.lock in an isolated environment') from None


def _local_metric_config(dataset: Path, jobs_dir: Path) -> Path:
    """Retain an explicit local-dataset metric binding without changing its bytes.

    Harbor discovers package-Hub metrics automatically, but local ``-p`` datasets
    require a JobConfig metric. Unique private-created filenames cannot reuse or
    overwrite another launch's configuration; Harbor also retains this binding
    in its resolved job config.
    """
    metric = dataset.resolve() / 'metric.py'
    if not metric.is_file():
        raise ValueError('Public local dataset lacks its frozen metric.py')
    jobs_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='daydream-reviewbench-launch-', suffix='.json', dir=jobs_dir.resolve())
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'metrics': [{'type': 'uv-script', 'kwargs': {'script_path': str(metric)}}]}, stream, indent=2)
        stream.write('\n')
    return Path(name)


def _check_completed_job(job_directory: Path) -> int:
    """Harbor process success alone does not establish scored trial success.

    Use the supplied strict checker and this job's retained metric/round plan.
    Every invocation owns a fresh report name so an earlier verification result
    remains intact when a later check fails or another command checks the job.
    """
    if not job_directory.is_dir():
        print('ReviewBench: Harbor returned without a retained job directory', file=sys.stderr)
        return 1
    descriptor, report_name = tempfile.mkstemp(
        prefix='daydream-reviewbench-check-', suffix='.json', dir=job_directory,
    )
    os.close(descriptor)
    result = subprocess.run([
        sys.executable, '-m', 'reviewbench_harbor.check-job', str(job_directory),
        str(job_directory / 'reviewbench-metric.py'), '--output', report_name,
    ], check=False)
    if result.returncode == 0:
        print(f'ReviewBench verified report: {report_name}')
    return result.returncode


def _selected_tasks(dataset: Path, profile: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Admit full frozen case identities before Harbor resolves the chosen trials.

    Harbor and the supplied plugin own task resolution and round/evaluation
    membership. This boundary checks that the exact names sent to Harbor still
    refer to the full supplied PR identities frozen by the exporter.
    """
    raw_cases = profile.get('cases')
    test_tasks = profile.get('test_tasks')
    if not isinstance(raw_cases, dict) or len(raw_cases) != 219:
        raise ValueError('Full supplied case identity catalog differs; create a fresh Harbor export')
    cases = {key: Case.from_dict(value) for key, value in raw_cases.items()}
    if any(key != case.key for key, case in cases.items()):
        raise ValueError('Frozen task names differ from their supplied full case identities')
    if (not isinstance(test_tasks, list) or len(test_tasks) != 25
            or len(set(test_tasks)) != 25 or any(key not in cases for key in test_tasks)):
        raise ValueError('Exact supplied test25 membership differs; create a fresh Harbor export')
    actual_names = {path.name for path in dataset.iterdir() if path.is_dir() and (path / 'task.toml').is_file()}
    if actual_names != set(cases):
        raise ValueError('Delivered task membership differs from the frozen full219 catalog')
    for key, case in cases.items():
        task = json.loads((dataset / key / 'tests/task.json').read_text())
        delivered = json.loads((dataset / key / 'environment/daydream-profile.json').read_text())
        if (Case.from_dict(task['pr']) != case or Case.from_dict(delivered['pr']) != case
                or delivered.get('policy_sha256') != profile['policy_sha256']
                or any(type(task.get(field)) is not int or delivered.get(field) != task[field]
                       for field in ('task_id', 'evaluation_id'))):
            raise ValueError(f'Delivered task differs from its frozen full identity: {key}')
    requested = args.task
    if requested:
        if len(set(requested)) != len(requested) or any(key not in cases for key in requested):
            raise ValueError('--task requires distinct exact exported task names; '
                             'globs and foreign names are unsupported')
        selected, scope = sorted(requested), 'explicit-subset'
    elif args.set == 'test':
        selected, scope = sorted(test_tasks), 'supplied-test25'
    else:
        selected, scope = sorted(cases), 'supplied-full219'
    print(f'ReviewBench selection: {scope}; {len(selected)} PRs x {args.rounds} independent rounds')
    return selected


def _current_candidate(candidate: dict[str, Any]) -> None:
    if candidate_identity() != candidate:
        raise ValueError('Executing candidate changed during Harbor launch; retain evidence and use a fresh export')


def handle(args: argparse.Namespace, candidate: dict[str, Any]) -> int:
    if args.reviewbench_command == 'harbor-export':
        from daydream.benchmark.reviewbench.harbor_profile import export_harbor_profile
        from daydream.benchmark.reviewbench.native import freeze_reviewer, parse_native_config

        reviewer = freeze_reviewer(parse_native_config(getattr(args, 'native_args', [])))
        suffix, assets, identity = reviewer_build(args.wheel.resolve(), platform=args.platform, reviewer=reviewer)
        result = export_harbor_profile(
            args.source, args.directory, candidate={**candidate, **identity}, base_image=args.base_image,
            reviewer=reviewer, reviewer_dockerfile_suffix=suffix, reviewer_assets=assets, rounds=args.rounds,
            platform=args.platform,
        )
        print(json.dumps(result, indent=2))
        return 0
    verify_runtime()
    from harbor.environments.docker.docker import DockerEnvironment

    from daydream.benchmark.reviewbench.harbor_auth import subscription_environment, validate_credential_stores
    from daydream.benchmark.reviewbench.harbor_profile import canonical_digest

    profile = json.loads((args.directory / 'daydream-profile.json').read_text())
    if profile.get('schema') != 3:
        raise ValueError('Incompatible public profile schema; create a fresh Harbor export')
    policy_sha = profile['policy_sha256']
    if canonical_digest({key: value for key, value in profile.items() if key != 'policy_sha256'}) != policy_sha:
        raise ValueError('Full public profile identity differs')
    if profile.get('source_prs') != 219:
        raise ValueError('Full public profile identity differs')
    frozen_candidate = profile.get('candidate', {})
    if any(not candidate.get(key) or frozen_candidate.get(key) != candidate[key]
           for key in ('artifact_sha256', 'version')):
        raise ValueError('Executing candidate differs from the frozen public profile')
    if args.rounds > profile['final_comparison_rounds']:
        raise ValueError('Requested rounds exceed the frozen profile')
    backend = profile['reviewer']['backend']
    if backend not in {'codex', 'pi'}:
        raise ValueError('Selected reviewer has no supported native Harbor capability')
    reviewer_path = args.codex_home if backend == 'codex' else args.reviewer_pi_home
    unused_path = args.reviewer_pi_home if backend == 'codex' else args.codex_home
    if reviewer_path is None or unused_path is not None:
        raise ValueError('Supply only the selected reviewer credential directory: --codex-home or --reviewer-pi-home')
    reviewer_home, judge_home = reviewer_path.resolve(), args.pi_home.resolve()
    for home in (reviewer_home, judge_home):
        if any(home.is_relative_to(root.resolve()) or root.resolve().is_relative_to(home)
               for root in (args.directory, args.jobs_dir)):
            raise ValueError('Private credential stores must be outside dataset and job evidence')
    selected = _selected_tasks(args.directory.resolve(), profile, args)
    validate_credential_stores(reviewer_home, judge_home, backend=backend, provider=profile['reviewer'].get('provider'))
    if not DockerEnvironment._egress_control_kernel_support():
        raise ValueError('Docker kernel lacks Harbor exact-host isolation support')
    # Schema and package validation precede credential delivery and model work.
    subprocess.run([sys.executable, '-m', 'reviewbench_harbor.check-format', str(args.directory.resolve())], check=True)
    local_config = _local_metric_config(args.directory, args.jobs_dir)
    command = [str(Path(sys.executable).with_name('harbor')), 'run', '--yes', '-c', str(local_config),
               '-p', str(args.directory.resolve()),
               '-a', 'daydream.benchmark.reviewbench.harbor_agent:DaydreamPublicReviewAgent',
               '-n', '1', '-k', str(args.rounds), '-r', '0',
               '-o', str(args.jobs_dir.resolve()), '--job-name', args.job_name,
               '--plugin', 'reviewbench_harbor.reviewbench_plugin:ReviewBenchPlugin']
    for task in selected:
        command.extend(['-i', task])
    environment = subscription_environment(os.environ)
    # Harbor's sidecar retains host-native architecture; task platform is frozen.
    environment.pop('DOCKER_DEFAULT_PLATFORM', None)
    for variable in ('RB_HARBOR_CODEX_AUTH_DIR', 'RB_HARBOR_REVIEWER_PI_AUTH_DIR'):
        environment.pop(variable, None)
    auth_variable = 'RB_HARBOR_CODEX_AUTH_DIR' if backend == 'codex' else 'RB_HARBOR_REVIEWER_PI_AUTH_DIR'
    environment[auth_variable] = str(reviewer_home)
    environment['RB_HARBOR_PI_AUTH_DIR'] = str(judge_home)
    environment['RB_HARBOR_PI_AUTH_PATH'] = '/run/reviewbench-pi/auth.json'
    environment['RB_HARBOR_SECCOMP_PROFILE'] = str(args.directory.resolve() / 'reviewer-seccomp.json')
    _current_candidate(candidate)
    returncode = subprocess.call(command, env=environment)
    if returncode:
        return returncode
    checked = _check_completed_job(args.jobs_dir.resolve() / args.job_name)
    _current_candidate(candidate)
    return checked
