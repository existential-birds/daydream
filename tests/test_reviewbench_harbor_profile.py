"""Public custom export contracts against the pinned supplied exporter, without inference."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
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
        {'platform': platform}))
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
    assert policy['schema'] == 3
    assert policy['source_prs'] == 219
    assert policy['final_comparison_rounds'] == 3
    assert policy['judge']['provider'] == 'openai-codex'
    assert policy['judge']['model'] == 'gpt-6.1-sol'
    assert policy['reviewer']['model'] != policy['judge']['model']
    if policy['reviewer']['backend'] == 'pi':
        assert {model['id'] for model in policy['reviewer']['pi_models']['providers']['nous']['models']} == {
            'deepseek/deepseek-v4.1-flash', 'deepseek/deepseek-v4.1-flash-1017'}
    assert 'synthetic-native-secret' not in (public_profile / 'daydream-profile.json').read_text()
    assert len(policy['cases']) == 219
    assert len(policy['test_tasks']) == 25
    assert policy['judge']['max_output_tokens_enforced'] is False
    assert policy['reviewer']['fanout_concurrency'] == 1
    assert policy['runtime']['platform'] == policy['candidate']['platform']
    assert policy['runtime']['reviewer_seccomp_sha256'] == _hash(public_profile / 'reviewer-seccomp.json')
    assert policy['runtime']['reviewer_seccomp_provenance']['policy_sha256'] == (
        policy['runtime']['reviewer_seccomp_sha256'])
    assert policy['trial_concurrency'] == 1
    expected = runpy.run_path(str(public_profile / 'metric.py'))['EXPECTED_TASKS']
    assert len(expected) == 219
    for reference in dataset['tasks']:
        directory = public_profile / reference['name'].split('/', 1)[1]
        task = json.loads((directory / 'tests/task.json').read_text())
        assert expected[str(task['task_id'])] == task['evaluation_id']
        exported = json.loads((directory / 'environment/daydream-profile.json').read_text())
        assert exported['schema'] == 3
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
    atomic_asset = task_dir / 'tests/daydream/json_utils.py'
    assert atomic_asset.read_bytes() == Path(json_utils.__file__).read_bytes()
    # Shipped verifier imports must work without host packages or PYTHONPATH.
    subprocess.run([sys.executable, '-E', '-S', str(task_dir / 'tests/daydream-auth.py'), '--help'],
                   cwd=public_profile, check=True, capture_output=True)
    policy = json.loads((public_profile / 'daydream-profile.json').read_text())
    redaction_asset = task_dir / 'tests/daydream/redaction.py'
    credential_patterns_asset = task_dir / 'tests/daydream/credential_patterns.py'
    assert redaction_asset.read_bytes() == Path(redaction.__file__).read_bytes()
    assert credential_patterns_asset.read_bytes() == Path(credential_patterns.__file__).read_bytes()
    assert (task_dir / 'tests/daydream/__init__.py').read_bytes() == b''
    # The shipped verifier must enable classifier retention.
    bootstrap = (task_dir / 'tests/test.sh').read_text().splitlines()[-1]
    arguments = shlex.split(bootstrap)
    source = arguments[arguments.index('--classifier-sessions') + 1]
    destination = arguments[arguments.index('--judge-evidence') + 1]
    assert source == '/tests/judge/classifier-output/sessions'
    assert destination == '/logs/verifier/judge-sessions'
    assert policy['judge']['evidence'] == {
        'classifier_sessions_source': source, 'classifier_sessions_destination': destination,
        'format': 'sanitized-top-level-pi-jsonl', 'publication': 'exclusive-index-last',
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
    supplied_source: Path, tmp_path: Path, option: dict[str, Any], error: str,
) -> None:
    """Invalid reviewer, rounds, image or architecture must publish nothing."""
    output = tmp_path / 'refused'
    options = {'wheel': tmp_path / 'unused.whl', 'base_image': NODE_IMAGE,
               'reviewer': freeze_reviewer(RunConfig(backend='codex', model='gpt-6.1-sol')), **option}
    with pytest.raises(ValueError, match=error):
        export_harbor_profile(supplied_source, output, **options)
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


@pytest.mark.parametrize('mutation', ['before-admission', 'during-bootstrap', 'before-publication'])
def test_export_refuses_changed_candidate_before_publication(
    supplied_source: Path, tmp_path: Path, mutation: str,
) -> None:
    """Real package mutations across build, export and publication cannot retain a stale identity."""
    copied = tmp_path / 'candidate'
    copied.mkdir()
    package = Path(harbor_profile.__file__).resolve().parents[2]
    shutil.copytree(package, copied / 'daydream', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(package.parent / 'pyproject.toml', copied / 'pyproject.toml')
    subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(copied / 'wheels')],
                   cwd=copied, check=True, capture_output=True)
    script = '''import json,sys
from pathlib import Path
from daydream.benchmark.reviewbench import harbor_profile
from daydream.benchmark.reviewbench.harbor_build import NODE_IMAGE
from daydream.benchmark.reviewbench.native import freeze_reviewer
from daydream.run_config import RunConfig
root,source,mutation=Path.cwd(),Path(sys.argv[1]),sys.argv[2]
reviewer=freeze_reviewer(RunConfig(target=str(root),backend='codex',model='gpt-6.1-sol'))
helper=root/'daydream/benchmark/reviewbench/harbor_auth.py'
marker=b'\\n# competing candidate helper bytes\\n'
verify=harbor_profile.verify_source
build=harbor_profile.reviewer_build
checks=0
def captured_build(*args,**kwargs):
 _,_,identity=build(*args,**kwargs)
 if mutation=='before-admission':helper.write_bytes(helper.read_bytes()+marker)
 return '',{},identity
harbor_profile.reviewer_build=captured_build
def mutate(path):
 global checks
 identity=verify(path);checks+=1
 if (checks==1 and mutation=='during-bootstrap') or (checks==2 and mutation=='before-publication'):
  helper.write_bytes(helper.read_bytes()+marker)
 return identity
harbor_profile.verify_source=mutate
# This HTTP-only dependency does not own candidate or supplied-source identity.
harbor_profile.seccomp_profile=lambda:(b'{}\\n',b'{"policy_sha256":"synthetic-control"}\\n')
output=root/'exported'
error=None
try:
 harbor_profile.export_harbor_profile(source,output,wheel=next((root/'wheels').glob('*.whl')),
  base_image=NODE_IMAGE,reviewer=reviewer)
except ValueError as exc:error=str(exc)
print(json.dumps({'error':error,'published':output.exists(),'source_checks':checks,
 'competing_bytes_preserved':helper.read_bytes().endswith(marker),
 'staging_preserved':bool(list(root.glob('.reviewbench-export-*')))}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(supplied_source), mutation], cwd=copied,
        env={**os.environ, 'PYTHONPATH': str(copied)}, check=True, capture_output=True, text=True)
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence['error'] == ('Executing candidate differs from the frozen export identity'
                                 if mutation == 'before-admission' else 'Executing candidate changed during export')
    assert evidence['source_checks'] == (0 if mutation == 'before-admission' else 2)
    assert evidence['competing_bytes_preserved'] is True
    assert evidence['published'] is False
    assert evidence['staging_preserved'] is False


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
