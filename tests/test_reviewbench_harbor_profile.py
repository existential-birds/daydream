"""Public custom export contracts against the pinned supplied exporter, without inference."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from daydream import json_utils
from daydream.benchmark.reviewbench.harbor_build import NODE_IMAGE
from daydream.benchmark.reviewbench.harbor_profile import export_harbor_profile
from tests.test_reviewbench_evaluator import supplied_source as supplied_source


@pytest.fixture
def public_profile(supplied_source: Path, tmp_path: Path) -> Path:
    """Use real supplied inputs/export process, not a fake task generator."""
    output = tmp_path / 'public'
    export_harbor_profile(
        supplied_source, output, candidate={'revision': 'test-candidate', 'wheel_sha256': 'a' * 64},
        base_image=NODE_IMAGE, reviewer_assets={'runtime.lock': b'offline test build input\n'},
        reviewer_dockerfile_suffix='\nCOPY runtime.lock /opt/runtime.lock\n',
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
    # Both roles share only the auth directory; each still owns its isolated
    # filesystem and exact-host network as validated above.
    assert (task_dir / 'environment/docker-compose.yaml').read_bytes() == (
        task_dir / 'tests/docker-compose.yaml').read_bytes()


@pytest.mark.parametrize('option', [
    {'reviewer_model': 'stale-model'}, {'reviewer_effort': 'unsupported'}, {'rounds': 0},
    {'base_image': 'node:latest'}, {'reviewer_timeout': 0}, {'verifier_timeout': 0},
    {'candidate': {'refresh_token': 'must-never-be-retained'}},
    {'reviewer_assets': {'../escaped': b'bytes'}},
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
