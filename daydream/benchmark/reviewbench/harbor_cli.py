"""Explicit public Harbor export and subscription-backed local execution."""

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

from daydream.benchmark.reviewbench.harbor_build import HARBOR_REVISION, NODE_IMAGE, reviewer_build


def add_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    export = commands.add_parser('harbor-export', help='export all 219 custom Codex tasks; no inference')
    export.add_argument('directory', type=Path)
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--wheel', type=Path, required=True)
    export.add_argument('--reviewer-model', default='gpt-6.1-sol')
    export.add_argument('--reviewer-effort', choices=('low', 'medium', 'high', 'xhigh'), default='high')
    export.add_argument('--base-image', default=NODE_IMAGE)
    export.add_argument('--rounds', type=int, choices=(1, 3), default=3)
    run = commands.add_parser('harbor-run', help='run the exported public profile; launches models')
    run.add_argument('directory', type=Path)
    run.add_argument('--codex-home', type=Path, required=True, help='existing subscription login; exclusively borrowed')
    run.add_argument('--auth-runtime', type=Path, required=True,
                     help='new private credential directory outside results')
    run.add_argument('--jobs-dir', type=Path, required=True)
    run.add_argument('--job-name', required=True)
    run.add_argument('--task', help='supplied task name/glob, applied to the full export')
    run.add_argument('--n-tasks', type=int)
    run.add_argument('--concurrency', type=int, default=1)
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


def handle(args: argparse.Namespace, candidate: dict[str, Any]) -> int:
    if args.reviewbench_command == 'harbor-export':
        from daydream.benchmark.reviewbench.harbor_profile import export_harbor_profile

        suffix, assets, identity = reviewer_build(args.wheel.resolve())
        result = export_harbor_profile(
            args.source, args.directory, candidate={**candidate, **identity}, base_image=args.base_image,
            reviewer_model=args.reviewer_model, reviewer_effort=args.reviewer_effort,
            reviewer_dockerfile_suffix=suffix, reviewer_assets=assets, rounds=args.rounds,
        )
        print(json.dumps(result, indent=2))
        return 0
    verify_runtime()
    from harbor.environments.docker.docker import DockerEnvironment

    from daydream.benchmark.reviewbench.harbor_auth import subscription_environment, subscription_session
    from daydream.benchmark.reviewbench.harbor_profile import canonical_digest

    auth_directory = args.auth_runtime.resolve()
    if any(auth_directory.is_relative_to(root.resolve()) for root in (args.directory, args.jobs_dir)):
        raise ValueError('Private auth runtime must be outside dataset and job evidence')
    profile = json.loads((args.directory / 'daydream-profile.json').read_text())
    policy_sha = profile.pop('policy_sha256')
    if canonical_digest(profile) != policy_sha or profile.get('source_prs') != 219:
        raise ValueError('Full public profile identity differs')
    frozen_candidate = profile.get('candidate', {})
    if any(not candidate.get(key) or frozen_candidate.get(key) != candidate[key]
           for key in ('artifact_sha256', 'version')):
        raise ValueError('Executing candidate differs from the frozen public profile')
    if args.rounds > profile['final_comparison_rounds']:
        raise ValueError('Requested rounds exceed the frozen profile')
    if args.concurrency < 1 or (args.n_tasks is not None and args.n_tasks < 1):
        raise ValueError('Concurrency and task count must be positive')
    if not DockerEnvironment._egress_control_kernel_support():
        raise ValueError('Docker kernel lacks Harbor exact-host isolation support')
    # Schema and package validation precede credential delivery and model work.
    subprocess.run([sys.executable, '-m', 'reviewbench_harbor.check-format', str(args.directory.resolve())], check=True)
    local_config = _local_metric_config(args.directory, args.jobs_dir)
    command = [str(Path(sys.executable).with_name('harbor')), 'run', '--yes', '-c', str(local_config),
               '-p', str(args.directory.resolve()),
               '-a', 'daydream.benchmark.reviewbench.harbor_agent:DaydreamPublicReviewAgent',
               '-n', str(args.concurrency), '-k', str(args.rounds), '-r', '0',
               '-o', str(args.jobs_dir.resolve()), '--job-name', args.job_name,
               '--plugin', 'reviewbench_harbor.reviewbench_plugin:ReviewBenchPlugin',
               '--plugin', 'daydream.benchmark.reviewbench.harbor_subscription_plugin:SubscriptionAdmissionPlugin']
    if args.task:
        command.extend(['-i', args.task])
    if args.n_tasks is not None:
        command.extend(['-l', str(args.n_tasks)])
    with subscription_session(args.codex_home.resolve(), args.auth_runtime.resolve()) as home:
        environment = subscription_environment(os.environ)
        # Task-authored main platform is frozen; Harbor's network sidecar must
        # retain the host-native architecture rather than inherit an override.
        environment.pop('DOCKER_DEFAULT_PLATFORM', None)
        environment.update(RB_HARBOR_AUTH_DIR=str(home), RB_HARBOR_SUBSCRIPTION_HOME='/run/reviewbench-codex')
        returncode = subprocess.call(command, env=environment)
    if returncode:
        return returncode
    return _check_completed_job(args.jobs_dir.resolve() / args.job_name)
