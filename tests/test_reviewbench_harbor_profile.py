"""Public custom export contracts against the pinned supplied exporter, without inference."""

from __future__ import annotations

import hashlib
import json
import runpy
import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream import credential_patterns, json_utils, redaction
from daydream.benchmark.reviewbench import harbor_profile
from daydream.benchmark.reviewbench.contract import verify_source
from daydream.benchmark.reviewbench.harbor_build import NODE_IMAGE
from daydream.benchmark.reviewbench.harbor_profile import export_harbor_profile
from tests.test_reviewbench_evaluator import supplied_source as supplied_source


@pytest.fixture(params=['linux/amd64', 'linux/arm64'])
def public_profile(supplied_source: Path, tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Use real supplied inputs/export process, not a fake task generator."""
    output = tmp_path / 'public'
    export_harbor_profile(
        supplied_source, output,
        candidate={'revision': 'test-candidate', 'wheel_sha256': 'a' * 64, 'platform': request.param},
        base_image=NODE_IMAGE, reviewer_assets={'runtime.lock': b'offline test build input\n'},
        reviewer_dockerfile_suffix='\nCOPY runtime.lock /opt/runtime.lock\n',
        **({'platform': request.param} if request.param != 'linux/amd64' else {}),
    )
    return output


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_custom_profile_passes_supplied_harbor_packager(
    supplied_source: Path, public_profile: Path,
) -> None:
    """A judge-bootstrap change must update task, evaluation, metric and dataset versions.

    Existing standalone tests never package Harbor tasks; stale upstream digests
    would make the delivered custom export unusable before any model call.
    """
    subprocess.run(['python', str(supplied_source / 'scripts/harbor/check-format.py'), str(public_profile)], check=True)
    dataset = tomllib.loads((public_profile / 'dataset.toml').read_text())
    manifest = json.loads((supplied_source / 'corpus/manifest.json').read_text())
    assert len(dataset['tasks']) == len(manifest) == 219
    policy = json.loads((public_profile / 'daydream-profile.json').read_text())
    assert policy['source_prs'] == 219
    assert policy['final_comparison_rounds'] == 3
    assert policy['judge']['provider'] == 'openai-codex'
    assert policy['judge']['max_output_tokens_enforced'] is False
    assert policy['reviewer']['fanout_concurrency'] == 1
    assert policy['runtime']['platform'] == policy['candidate']['platform']
    assert policy['runtime']['reviewer_seccomp_sha256'] == _hash(public_profile / 'reviewer-seccomp.json')
    assert policy['runtime']['reviewer_seccomp_provenance']['policy_sha256'] == (
        policy['runtime']['reviewer_seccomp_sha256'])
    assert policy['subscription_admission']['effective_concurrency'] == 1
    assert policy['subscription_admission']['scope'] == 'whole-trial-before-create-and-phase-timers'
    expected = runpy.run_path(str(public_profile / 'metric.py'))['EXPECTED_TASKS']
    assert len(expected) == 219
    for reference in dataset['tasks']:
        directory = public_profile / reference['name'].split('/', 1)[1]
        task = json.loads((directory / 'tests/task.json').read_text())
        assert expected[str(task['task_id'])] == task['evaluation_id']
        exported = json.loads((directory / 'environment/daydream-profile.json').read_text())
        digest = exported.pop('profile_sha256')
        encoded = json.dumps(exported, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest
        assert exported['evaluation_id'] == task['evaluation_id']
        assert exported['policy_sha256'] == policy['policy_sha256']
        assert exported['runtime']['platform'] == policy['runtime']['platform']
        assert _hash(directory / 'environment/reviewer-seccomp.json') == policy['runtime']['reviewer_seccomp_sha256']
        assert not (directory / 'environment/golden').exists()
        task_config = tomllib.loads((directory / 'task.toml').read_text())
        assert task_config['verifier']['environment_mode'] == 'separate'
        assert task_config['agent']['allowed_hosts'] == ['chatgpt.com', 'auth.openai.com']
        assert task_config['verifier']['allowed_hosts'] == ['chatgpt.com', 'auth.openai.com']
    for reference in dataset['files']:
        assert reference['digest'] == f'sha256:{_hash(public_profile / reference["path"])}'


def test_custom_bootstrap_keeps_supplied_judge_semantics_and_loads_current_model(
    supplied_source: Path, public_profile: Path,
) -> None:
    """A custom provider without usable Pi metadata can silently skip actual judging.

    Prove model resolution using the pinned real SDK, and protect upstream source
    parity independently of the evaluator's standalone fixtures.
    """
    dataset = tomllib.loads((public_profile / 'dataset.toml').read_text())
    task_dir = public_profile / dataset['tasks'][0]['name'].split('/', 1)[1]
    judge = task_dir / 'tests/judge'
    atomic_asset = task_dir / 'tests/daydream_json_utils.py'
    assert atomic_asset.read_bytes() == Path(json_utils.__file__).read_bytes()
    policy = json.loads((public_profile / 'daydream-profile.json').read_text())
    assert policy['judge']['atomic_writer_sha256'] == _hash(atomic_asset)
    redaction_asset = task_dir / 'tests/daydream/redaction.py'
    credential_patterns_asset = task_dir / 'tests/daydream/credential_patterns.py'
    assert redaction_asset.read_bytes() == Path(redaction.__file__).read_bytes()
    assert credential_patterns_asset.read_bytes() == Path(credential_patterns.__file__).read_bytes()
    assert (task_dir / 'tests/daydream/__init__.py').read_bytes() == b''
    # Auth-wrapper tests own session publication. The shipped task must actually
    # opt into that contract, otherwise successful verifier cleanup loses its
    # classifier conversations despite a working retention helper.
    bootstrap = (task_dir / 'tests/test.sh').read_text().splitlines()[-1]
    arguments = shlex.split(bootstrap)
    source = arguments[arguments.index('--classifier-sessions') + 1]
    destination = arguments[arguments.index('--judge-evidence') + 1]
    assert source == '/tests/judge/classifier-output/sessions'
    assert destination == '/logs/verifier/judge-sessions'
    assert policy['judge']['evidence'] == {
        'classifier_sessions_source': source, 'classifier_sessions_destination': destination,
        'format': 'sanitized-top-level-pi-jsonl', 'publication': 'exclusive-index-last',
        'redaction_sha256': _hash(redaction_asset), 'credential_patterns_sha256': _hash(credential_patterns_asset),
        'matcher_sessions': 'unavailable_upstream_in_memory', 'matcher_usage': 'not_retained_by_eval',
        'monetary_usage_source': 'api-equivalent-estimate',
    }
    for relative in ('scripts/eval/matcher.ts', 'scripts/eval/eval.ts', 'scripts/eval/scorer.ts',
                     'scripts/classifier/classify.ts', 'scripts/classifier/prompts.ts', 'scripts/harbor/verify.ts',
                     'package.json', 'package-lock.json'):
        assert (judge / relative).read_bytes() == (supplied_source / relative).read_bytes()
    result = subprocess.run([
        'node', '--input-type=module', '-e',
        "import {AuthStorage,ModelRegistry} from '@earendil-works/pi-coding-agent';"
        "const registry=ModelRegistry.create(AuthStorage.inMemory(),process.argv[1]);"
        "if(registry.getError()) throw new Error(registry.getError());"
        "console.log(JSON.stringify(registry.find('openai-codex','gpt-6.1-sol')));",
        str(judge / '.daydream-codex/models.json'),
    ], cwd=supplied_source, check=True, capture_output=True, text=True)
    model = json.loads(result.stdout)
    assert model['id'] == 'gpt-6.1-sol'
    assert model['api'] == 'openai-codex-responses'
    assert model['baseUrl'] == 'https://chatgpt.com/backend-api'
    assert model['reasoning'] is True
    dockerfile = (task_dir / 'environment/Dockerfile').read_text()
    assert dockerfile.index('COPY runtime.lock') < dockerfile.index('RUN git init -q .')
    # Both roles share only auth. Only the reviewer receives the narrowly
    # extended syscall profile needed to bootstrap Codex's native sandbox.
    reviewer = yaml.safe_load((task_dir / 'environment/docker-compose.yaml').read_text())['services']['main']
    verifier = yaml.safe_load((task_dir / 'tests/docker-compose.yaml').read_text())['services']['main']
    assert set(reviewer) == {'platform', 'security_opt', 'volumes'}
    assert set(verifier) == {'platform', 'volumes'}
    assert reviewer['platform'] == verifier['platform'] == policy['runtime']['platform']
    assert reviewer['volumes'] == verifier['volumes'] == [{
        'type': 'bind', 'source': '${RB_HARBOR_AUTH_DIR:?Set a dedicated subscription runtime directory}',
        'target': '/run/reviewbench-codex',
    }]
    assert reviewer['security_opt'] == [
        'seccomp=${RB_HARBOR_SECCOMP_PROFILE:?Set the frozen reviewer seccomp profile}',
    ]


@pytest.mark.parametrize('option', [
    {'reviewer_model': 'stale-model'}, {'reviewer_effort': 'unsupported'}, {'rounds': 0},
    {'base_image': 'node:latest'}, {'reviewer_timeout': 0}, {'verifier_timeout': 0},
    {'candidate': {'refresh_token': 'must-never-be-retained'}},
    {'reviewer_assets': {'../escaped': b'bytes'}},
    {'platform': 'linux/ppc64le'}, {'platform': 'linux/arm64', 'candidate': {'platform': 'linux/amd64'}},
])
def test_invalid_public_profile_preserves_source_and_publishes_nothing(
    supplied_source: Path, tmp_path: Path, option: dict[str, Any],
) -> None:
    """Invalid transport/billing policy and escaping assets must fail before publication."""
    output = tmp_path / 'refused'
    options = {'candidate': {'revision': 'offline'}, 'base_image': NODE_IMAGE, **option}
    with pytest.raises(ValueError):
        export_harbor_profile(supplied_source, output, **options)
    assert not output.exists()
    assert not (tmp_path / 'escaped').exists()
    assert not list(tmp_path.glob('.reviewbench-export-*'))


def test_competing_export_destination_preserves_the_winning_profile(
    supplied_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination created during export must not receive a nested candidate.

    Successful format tests miss the race between admission and publication: a
    move into an existing directory returns the winner's path with our policy.
    """
    output = tmp_path / 'competing-profile'
    winning_files = {
        'daydream-profile.json': b'{"winner": "other-export"}\n',
        'retained.txt': b'preserve competing export bytes\n',
    }
    verifications = 0

    def competing_destination(source: Path) -> dict[str, Any]:
        nonlocal verifications
        identity = verify_source(source)
        verifications += 1
        if verifications == 2:
            # The real export is complete and source verification still passes.
            # Publish competing bytes at the last boundary before installation.
            output.mkdir()
            for name, content in winning_files.items():
                (output / name).write_bytes(content)
        return identity

    monkeypatch.setattr(harbor_profile, 'verify_source', competing_destination)
    with pytest.raises(FileExistsError):
        export_harbor_profile(supplied_source, output, candidate={'revision': 'losing-export'}, base_image=NODE_IMAGE)
    assert verifications == 2
    assert {path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob('*') if path.is_file()} == winning_files
    assert not (output / 'dataset').exists()
    assert not list(tmp_path.glob('.reviewbench-export-*'))
