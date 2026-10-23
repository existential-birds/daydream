"""Public custom export contracts against the pinned supplied exporter, without inference."""

from __future__ import annotations

import hashlib
import json
import runpy
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream.benchmark.reviewbench import harbor_profile
from daydream.benchmark.reviewbench.contract import PUBLIC_PROFILE_SCHEMA, verify_source
from daydream.benchmark.reviewbench.harbor_build import NODE_IMAGE
from daydream.benchmark.reviewbench.harbor_profile import export_harbor_profile
from daydream.benchmark.reviewbench.native import freeze_reviewer
from daydream.run_config import RunConfig
from tests.harness.reviewbench import supplied_source as supplied_source
from tests.test_reviewbench_seccomp import moby_policy as moby_policy

pytestmark = pytest.mark.usefixtures('moby_policy')


@pytest.fixture(params=[('linux/amd64', 'codex'), ('linux/arm64', 'codex'),
                        ('linux/amd64', 'pi'), ('linux/arm64', 'pi')])
def public_profile(supplied_source: Path, tmp_path: Path, request: pytest.FixtureRequest,
                   monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the actual pinned full219 exporter for each supported reviewer and architecture."""
    output = tmp_path / 'public'
    platform, backend = request.param
    model = 'gpt-6-sol' if backend == 'codex' else 'deepseek/deepseek-v4.1-flash'
    model_ids = [model] if backend == 'codex' else [model, 'deepseek/deepseek-v4.1-flash-1017']
    agent = tmp_path / 'operator-pi'
    agent.mkdir()
    (agent / 'models.json').write_text(json.dumps({'providers': {'nous': {
        'baseUrl': 'https://inference-api.nousresearch.com/v1', 'apiKey': 'synthetic-native-secret',
        'models': [{'id': name, 'name': name, 'api': 'openai-completions', 'reasoning': True,
                    'input': ['text'], 'contextWindow': 1000000, 'maxTokens': 8192,
                    'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}} for name in model_ids],
    }}}))
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(agent))
    monkeypatch.setenv('PI_PROVIDER', 'nous')
    reviewer = freeze_reviewer(RunConfig(target=str(tmp_path), backend=backend, model=model,
                                        parse_model=model_ids[-1],
                                        reasoning_effort='high'))
    monkeypatch.setattr(harbor_profile, 'reviewer_build', lambda _wheel, *, platform, backend: (
        '\nCOPY runtime.lock /opt/runtime.lock\n', {'runtime.lock': b'offline test build input\n'},
        {'version': '0.30.0', 'wheel_name': 'daydream-0.30.0-py3-none-any.whl',
         'wheel_sha256': 'a' * 64, 'platform': platform}))
    export_harbor_profile(supplied_source, output, wheel=tmp_path / 'unused.whl', reviewer=reviewer,
                          base_image=NODE_IMAGE, platform=platform)
    return output


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_custom_profile_passes_supplied_harbor_packager(
    supplied_source: Path, public_profile: Path,
) -> None:
    """Validate real task, evaluation, metric and dataset identities with the supplied packager."""
    subprocess.run(['python', str(supplied_source / 'scripts/harbor/check-format.py'), str(public_profile)], check=True)
    dataset = tomllib.loads((public_profile / 'dataset.toml').read_text())
    manifest = json.loads((supplied_source / 'corpus/manifest.json').read_text())
    assert len(dataset['tasks']) == len(manifest) == 219
    policy = json.loads((public_profile / 'daydream-profile.json').read_text())
    assert policy['schema'] == PUBLIC_PROFILE_SCHEMA
    assert policy['final_comparison_rounds'] == 3
    assert policy['judge']['provider'] == 'openai-codex'
    assert policy['judge']['model'] == 'gpt-6.1-sol'
    assert policy['reviewer']['model'] != policy['judge']['model']
    if policy['reviewer']['backend'] == 'pi':
        assert {model['id'] for model in policy['reviewer']['pi_models']['providers']['nous']['models']} == {
            'deepseek/deepseek-v4.1-flash', 'deepseek/deepseek-v4.1-flash-1017'}
    assert 'synthetic-native-secret' not in (public_profile / 'daydream-profile.json').read_text()
    assert len(policy['test_tasks']) == 25
    assert policy['runtime']['reviewer_seccomp_sha256'] == _hash(public_profile / 'reviewer-seccomp.json')
    assert policy['runtime']['reviewer_seccomp_provenance']['policy_sha256'] == (
        policy['runtime']['reviewer_seccomp_sha256'])
    expected = runpy.run_path(str(public_profile / 'metric.py'))['EXPECTED_TASKS']
    assert len(expected) == 219
    for reference in dataset['tasks']:
        directory = public_profile / reference['name'].split('/', 1)[1]
        task = json.loads((directory / 'tests/task.json').read_text())
        assert expected[str(task['task_id'])] == task['evaluation_id']
        exported = json.loads((directory / 'environment/daydream-profile.json').read_text())
        assert exported['schema'] == PUBLIC_PROFILE_SCHEMA
        assert exported['candidate'] == policy['candidate']
        digest = exported.pop('profile_sha256')
        encoded = json.dumps(exported, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest
        assert exported['evaluation_id'] == task['evaluation_id']
        assert exported['policy_sha256'] == policy['policy_sha256']
        assert _hash(directory / 'environment/reviewer-seccomp.json') == policy['runtime']['reviewer_seccomp_sha256']
        assert not (directory / 'environment/golden').exists()
        task_config = tomllib.loads((directory / 'task.toml').read_text())
        assert task_config['verifier']['environment_mode'] == 'separate'
        assert task_config['agent']['allowed_hosts'] == policy['reviewer']['allowed_hosts']
        assert task_config['verifier']['allowed_hosts'] == ['chatgpt.com', 'auth.openai.com']
    for reference in dataset['files']:
        assert reference['digest'] == f'sha256:{_hash(public_profile / reference["path"])}'


def test_custom_bootstrap_keeps_supplied_judge_semantics_and_loads_current_model(
    supplied_source: Path, public_profile: Path,
) -> None:
    """Resolve the actual judge SDK model and preserve supplied sources and isolated bootstrap wiring."""
    dataset = tomllib.loads((public_profile / 'dataset.toml').read_text())
    task_dir = public_profile / dataset['tasks'][0]['name'].split('/', 1)[1]
    judge = task_dir / 'tests/judge'
    # The verifier wrapper is self-contained and imports no host/site packages.
    subprocess.run([sys.executable, '-E', '-S', str(task_dir / 'tests/daydream-auth.py'), '--help'],
                   cwd=public_profile, check=True, capture_output=True)
    policy = json.loads((public_profile / 'daydream-profile.json').read_text())
    arguments = shlex.split((task_dir / 'tests/test.sh').read_text().splitlines()[-1])
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
    # Only the reviewer receives the extended sandbox bootstrap policy.
    reviewer = yaml.safe_load((task_dir / 'environment/docker-compose.yaml').read_text())['services']['main']
    verifier = yaml.safe_load((task_dir / 'tests/docker-compose.yaml').read_text())['services']['main']
    assert set(reviewer) == {'platform', 'security_opt', 'volumes'}
    assert set(verifier) == {'platform', 'volumes'}
    assert reviewer['platform'] == verifier['platform'] == policy['runtime']['platform']
    reviewer_role = 'REVIEWER_PI' if policy['reviewer']['backend'] == 'pi' else 'CODEX'
    reviewer_target = '/run/reviewbench-reviewer' if reviewer_role == 'REVIEWER_PI' else '/run/reviewbench-codex'
    for role, config, target in [(reviewer_role, reviewer, reviewer_target),
                                 ('PI', verifier, '/run/reviewbench-pi')]:
        assert config['volumes'] == [{
            'type': 'bind', 'source': '${RB_HARBOR_' + role + '_AUTH_DIR:?Set a dedicated '
                                    + role + ' credential directory}', 'target': target,
        }]
    assert arguments[arguments.index('--canonical-home') + 1] == '/run/reviewbench-pi'
    assert arguments[arguments.index('--pi-bootstrap') + 1] == '/tests/judge/daydream-auth.mjs'
    binding = (judge / 'daydream-auth.mjs').read_bytes()
    assert binding == Path(harbor_profile.__file__).with_name('judge_auth.mjs').read_bytes()
    assert reviewer['security_opt'] == [
        'seccomp=${RB_HARBOR_SECCOMP_PROFILE:?Set the frozen reviewer seccomp profile}',
    ]


@pytest.mark.parametrize(('option', 'error'), [
    ({'reviewer': {'backend': 'unsupported'}}, 'admitted frozen native'),
    ({'rounds': 0}, 'rounds must be positive'),
    ({'base_image': 'node:latest'}, 'pinned by sha256'),
    ({'platform': 'linux/ppc64le'}, 'supported Linux platform'),
])
def test_invalid_public_profile_preserves_source_and_publishes_nothing(
    tmp_path: Path, option: dict[str, Any], error: str,
) -> None:
    """Invalid reviewer, rounds, image or architecture must publish nothing."""
    output = tmp_path / 'refused'
    options = {'wheel': tmp_path / 'unused.whl', 'base_image': NODE_IMAGE,
               'reviewer': freeze_reviewer(RunConfig(backend='codex', model='gpt-6.1-sol')), **option}
    with pytest.raises(ValueError, match=error):
        export_harbor_profile(tmp_path / 'unread-source', output, **options)
    assert not output.exists()
    assert not list(tmp_path.glob('.reviewbench-export-*'))


def test_competing_export_destination_preserves_the_winning_profile(
    supplied_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A competing publication keeps its exact bytes and cannot receive a nested losing export."""
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
            output.mkdir()
            for name, content in winning_files.items():
                (output / name).write_bytes(content)
        return identity

    monkeypatch.setattr(harbor_profile, 'verify_source', competing_destination)
    monkeypatch.setattr(harbor_profile, 'reviewer_build', lambda *args, **kwargs: ('', {}, {}))
    with pytest.raises(FileExistsError):
        export_harbor_profile(supplied_source, output, wheel=tmp_path / 'unused.whl', base_image=NODE_IMAGE,
            reviewer=freeze_reviewer(RunConfig(backend='codex', model='gpt-6.1-sol')))
    assert verifications == 2
    assert {path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob('*') if path.is_file()} == winning_files
    assert not (output / 'dataset').exists()
    assert not list(tmp_path.glob('.reviewbench-export-*'))


@pytest.mark.parametrize('change', ['missing-model', 'http', 'model-host', 'headers'])
def test_native_pi_export_rejects_unavailable_or_unisolated_operator_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    """Admit native model availability and exact HTTPS routing without mutating operator files."""
    provider: dict[str, Any] = {'baseUrl': 'https://inference-api.nousresearch.com/v1',
        'apiKey': 'synthetic-native-secret', 'models': [{'id': 'deepseek/deepseek-v4.1-flash',
                                                       'api': 'openai-completions'}]}
    if change == 'missing-model':
        provider['models'] = []
    elif change == 'http':
        provider['baseUrl'] = 'http://inference-api.nousresearch.com/v1'
    elif change == 'model-host':
        provider['models'][0]['baseUrl'] = 'https://secondary.invalid/v1'
    else:
        provider['headers'] = {'Authorization': 'synthetic-header-secret'}
    operator = tmp_path / 'operator'
    operator.mkdir()
    path = operator / 'models.json'
    original = json.dumps({'providers': {'nous': provider}})
    path.write_text(original)
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(operator))
    monkeypatch.setenv('PI_PROVIDER', 'nous')
    with pytest.raises(ValueError) as caught:
        freeze_reviewer(RunConfig(target=str(tmp_path), backend='pi', model='deepseek/deepseek-v4.1-flash'))
    assert 'synthetic-' not in str(caught.value)
    assert path.read_text() == original
