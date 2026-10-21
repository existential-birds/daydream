"""Public Harbor launch/package contracts at external-process boundaries.

Existing export tests own the 219 supplied tasks and format digests. These tests
protect the later operator launch, which can otherwise select the wrong runtime,
package, rounds or authentication route despite a valid export. Harmless process
replays replace Harbor and its format checker; no container or inference runs.
"""
from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_build, harbor_cli
from daydream.benchmark.reviewbench.harbor_profile import canonical_digest


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[argparse.Namespace, dict[str, Any]]:
    from harbor.environments.docker.docker import DockerEnvironment

    binary = tmp_path / 'runtime' / 'bin'
    binary.mkdir(parents=True)
    interpreter = sys.executable
    format_checker = binary / 'python'
    format_checker.write_text(
        f'#!{interpreter}\n'
        'import pathlib, sys\n'
        'assert sys.argv[1:3] == ["-m", "reviewbench_harbor.check-format"]\n'
        'assert (pathlib.Path(sys.argv[3]) / "daydream-profile.json").is_file()\n'
        '(pathlib.Path(sys.argv[3]) / "format-checked").touch()\n'
    )
    format_checker.chmod(0o755)
    harbor = binary / 'harbor'
    harbor.write_text(
        f'#!{interpreter}\n'
        'import argparse, json, os, pathlib\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("command", choices=["run"])\n'
        'for flag, name in [("-p", "dataset"), ("-a", "agent"), ("-n", "concurrency"),\n'
        ' ("-k", "rounds"), ("-r", "retries"), ("-o", "jobs"),\n'
        ' ("--job-name", "job"), ("--plugin", "plugin"), ("-i", "task"), ("-l", "limit")]:\n'
        ' parser.add_argument(flag, dest=name)\n'
        'args = parser.parse_args()\n'
        'auth_dir = pathlib.Path(os.environ["RB_HARBOR_AUTH_DIR"])\n'
        'auth = json.loads((auth_dir / "auth.json").read_text())\n'
        'assert auth["auth_mode"] == "chatgpt" and not auth["OPENAI_API_KEY"]\n'
        'auth["tokens"]["refresh_token"] = "synthetic-process-rotation"\n'
        '(auth_dir / "auth.json").write_text(json.dumps(auth))\n'
        'job = pathlib.Path(args.jobs) / args.job\n'
        'job.mkdir(parents=True)\n'
        'record = vars(args)\n'
        'record.update(round_membership=list(range(1, int(args.rounds) + 1)),\n'
        ' auth_home=os.environ["RB_HARBOR_SUBSCRIPTION_HOME"],\n'
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
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CODEX_BEARER_TOKEN', 'OPENAI_WORKLOAD_IDENTITY_TOKEN'):
        monkeypatch.setenv(key, 'synthetic-forbidden-route')
    assert harbor_cli.handle(args, candidate) == exit_code
    actual = json.loads((args.jobs_dir / args.job_name / 'launch.json').read_text())
    assert actual == {
        'command': 'run', 'dataset': str(args.directory.resolve()),
        'agent': 'daydream.benchmark.reviewbench.harbor_agent:DaydreamPublicReviewAgent',
        'concurrency': '2', 'rounds': str(rounds), 'retries': '0', 'jobs': str(args.jobs_dir.resolve()),
        'job': args.job_name, 'plugin': 'reviewbench_harbor.reviewbench_plugin:ReviewBenchPlugin',
        'task': 'owner_repo*', 'limit': '1', 'round_membership': list(range(1, rounds + 1)),
        'auth_home': '/run/reviewbench-codex', 'platform': None, 'forbidden_env': [],
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


@pytest.mark.parametrize('damage', ['missing-module', 'changed-module'])
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
                data += b'\n# Different candidate package bytes\n'
            destination.writestr(name, data)
    with pytest.raises(ValueError, match='package membership|same candidate package'):
        harbor_build.reviewer_build(changed)
