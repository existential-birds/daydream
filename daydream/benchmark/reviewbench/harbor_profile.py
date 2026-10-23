"""Content-addressed public ReviewBench tasks with a subscription-only Codex judge.

The supplied exporter and judging sources are executed unchanged. This module owns
only the explicitly distinct local evaluation profile and its runtime bootstrap.
It never reads credentials and never publishes a dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from daydream.benchmark.reviewbench.contract import (
    PUBLIC_PROFILE_SCHEMA,
    REVIEWBENCH_REVISION,
    load_cases,
    verify_source,
)
from daydream.benchmark.reviewbench.harbor_build import reviewer_build
from daydream.benchmark.reviewbench.harbor_seccomp import seccomp_profile

CODEX_MODEL = 'gpt-6.1-sol'
JUDGE_HOME = '/run/reviewbench-pi'
PROFILE_PATH = '/work/pr/daydream-profile.json'


def _compose(platform: str, *, reviewer: bool, backend: str = 'codex') -> str:
    """Keep the judge on Docker's default syscall policy and share only auth."""
    security = ('    security_opt:\n'
                '      - "seccomp=${RB_HARBOR_SECCOMP_PROFILE:?Set the frozen reviewer seccomp profile}"\n'
                if reviewer else '')
    role = ("REVIEWER_PI" if backend == 'pi' else "CODEX") if reviewer else "PI"
    target = (("/run/reviewbench-reviewer" if backend == 'pi' else "/run/reviewbench-codex")
              if reviewer else JUDGE_HOME)
    return f'''services:
  main:
    platform: {platform}
{security}    volumes:
      - type: bind
        source: ${{RB_HARBOR_{role}_AUTH_DIR:?Set a dedicated {role} credential directory}}
        target: {target}
'''


def canonical_digest(value: Any) -> str:
    """Digest JSON policy and identities, never credential contents."""
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def package_digest(directory: Path) -> str:
    """Match Harbor's sorted POSIX path/content digest protocol."""
    entries = []
    for path in sorted(directory.rglob('*'), key=lambda item: item.relative_to(directory).as_posix()):
        if path.is_symlink():
            raise ValueError('Public task assets must not contain symlinks')
        if path.is_file():
            entries.append(f'{path.relative_to(directory).as_posix()}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n')
    return hashlib.sha256(''.join(entries).encode()).hexdigest()


def codex_models() -> dict[str, Any]:
    """Pi model metadata; costs are API-equivalent and maxTokens is unenforced."""
    return {'providers': {'openai-codex': {
        'api': 'openai-codex-responses',
        'baseUrl': 'https://chatgpt.com/backend-api',
        'models': [{
            'id': CODEX_MODEL, 'name': CODEX_MODEL, 'reasoning': True, 'input': ['text', 'image'],
            'contextWindow': 1050000, 'maxTokens': 8192,
            'cost': {'input': 2, 'output': 10, 'cacheRead': 0.1, 'cacheWrite': 2.5},
        }],
    }}}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def selected_reviewer_policy(reviewer: Mapping[str, Any], *,
                             environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Bind the selected native endpoint/model, without reading authentication.

    Nous is an operator-defined native Pi provider. Copy only its selected public
    model definition; credentials remain in Pi's native auth store. Other native
    providers need an explicit isolation contract before this public lane admits
    them. Ordinary Daydream backend support is unaffected.
    """
    policy = dict(reviewer)
    if policy.get('backend') == 'codex':
        policy['allowed_hosts'] = ['chatgpt.com', 'auth.openai.com']
        return policy
    if policy.get('backend') != 'pi' or policy.get('provider') != 'nous':
        raise ValueError('Public ReviewBench supports native Codex or the configured Pi nous provider')
    env = os.environ if environment is None else environment
    agent_dir = Path(env.get('PI_CODING_AGENT_DIR') or Path(env.get('HOME', str(Path.home()))) / '.pi/agent').resolve()
    try:
        providers = json.loads((agent_dir / 'models.json').read_text())['providers']
        provider = providers['nous']
        required = {policy['model'], *policy.get('phase_models', {}).values()}
        catalog = {item['id']: item for item in provider['models']}
        models = [catalog[name] for name in sorted(required)]
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        raise ValueError('Selected native Pi nous model definition is unavailable') from None
    if set(provider) - {'apiKey', 'baseUrl', 'models', 'api'}:
        raise ValueError('Selected Pi provider has unsupported routing or authentication extensions')
    allowed_model = {'id', 'name', 'api', 'reasoning', 'input', 'cost', 'contextWindow', 'maxTokens',
                     'compat', 'thinkingLevelMap'}
    if any(set(model) - allowed_model for model in models):
        raise ValueError('Selected Pi model has unsupported routing or authentication extensions')
    address = urlsplit(provider.get('baseUrl', ''))
    if (address.scheme != 'https' or not address.hostname or address.username or address.password
            or address.query or address.fragment or address.port not in (None, 443)):
        raise ValueError('Selected native Pi provider requires an exact HTTPS endpoint')
    public_provider = {'baseUrl': provider['baseUrl'], 'models': models}
    if 'api' in provider:
        public_provider['api'] = provider['api']
    policy['pi_models'] = {'providers': {'nous': public_provider}}
    policy['allowed_hosts'] = [address.hostname]
    return policy


def export_harbor_profile(
    source: Path,
    output: Path,
    *,
    wheel: Path,
    reviewer: Mapping[str, Any],
    base_image: str,
    rounds: int = 3,
    platform: str = 'linux/amd64',
) -> dict[str, Any]:
    """Freeze the supplied wheel and all 219 supplied tasks into a new local profile."""
    source = source.resolve()
    output = output.absolute()
    if output.exists():
        raise ValueError('Export destination already exists; preserve existing evidence')
    reviewer = dict(reviewer)
    if (reviewer.get('backend') not in {'codex', 'pi'} or reviewer.get('fanout_concurrency') != 1
            or not reviewer.get('allowed_hosts')):
        raise ValueError('Reviewer must be an admitted frozen native configuration')
    if type(rounds) is not int or rounds < 1:
        raise ValueError('Independent comparison rounds must be positive')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[a-f0-9]{64}', base_image):
        raise ValueError('Runtime images must be pinned by sha256 digest')
    reviewer_dockerfile_suffix, reviewer_assets, candidate = reviewer_build(
        wheel.resolve(), platform=platform, backend=reviewer['backend'],
    )
    source_identity = verify_source(source)
    cases = load_cases(source, 'full')
    models = codex_models()
    from daydream import credential_patterns, json_utils, redaction

    seccomp_asset, seccomp_provenance_asset = seccomp_profile()
    seccomp_provenance = json.loads(seccomp_provenance_asset)
    seccomp_sha = hashlib.sha256(seccomp_asset).hexdigest()
    bootstrap = (
        '#!/bin/bash\nset -euo pipefail\ncd /tests/judge\n'
        'exec python3 /tests/daydream-auth.py judge '
        f'--canonical-home {JUDGE_HOME} '
        '--classifier-sessions /tests/judge/classifier-output/sessions '
        '--judge-evidence /logs/verifier/judge-sessions '
        '--models /tests/judge/.daydream-codex/models.json '
        '--pi-bootstrap /tests/judge/daydream-auth.mjs -- '
        'node --import tsx scripts/harbor/verify.ts /tests/task.json\n'
    )
    task_assets = {
        **{f'environment/{name}': content for name, content in reviewer_assets.items()},
        'environment/reviewer-seccomp.json': seccomp_asset,
        'environment/reviewer-seccomp-provenance.json': seccomp_provenance_asset,
        'environment/docker-compose.yaml': _compose(platform, reviewer=True, backend=reviewer['backend']).encode(),
        'tests/docker-compose.yaml': _compose(platform, reviewer=False).encode(),
        'tests/daydream-auth.py': Path(__file__).with_name('harbor_auth.py').read_bytes(),
        'tests/daydream/__init__.py': b'',
        **{f'tests/daydream/{Path(filename).name}': Path(filename).read_bytes()
           for filename in (json_utils.__file__, redaction.__file__, credential_patterns.__file__)},
        'tests/judge/.daydream-codex/models.json': (json.dumps(models, sort_keys=True, indent=2) + '\n').encode(),
        'tests/judge/daydream-auth.mjs': Path(__file__).with_name('judge_auth.mjs').read_bytes(),
        'tests/test.sh': bootstrap.encode(),
    }
    policy: dict[str, Any] = {
        'schema': PUBLIC_PROFILE_SCHEMA, 'name': 'daydream-reviewbench-native',
        'source': source_identity, 'source_prs': len(cases),
        'cases': {case.key: asdict(case) for case in cases},
        'test_tasks': [case.key for case in load_cases(source, 'test')],
        'candidate': dict(candidate), 'final_comparison_rounds': rounds,
        'trial_concurrency': 1,
        'reviewer': reviewer,
        'judge': {
            'provider': 'openai-codex', 'model': CODEX_MODEL, 'auth': 'subscription-oauth',
            'matcher_effort': 'low', 'classifier_effort': 'medium',
            'call_timeout_ms': 600000, 'classifier_extra_per_finding_ms': 120000,
            'verifier_timeout_sec': 7200, 'models': models,
            'max_output_tokens_enforced': False, 'cost_source': 'api-equivalent-estimate',
            'refresh_policy': 'independent-native-stores-pi-canonical-locks',
            'evidence': {
                'classifier_sessions_source': '/tests/judge/classifier-output/sessions',
                'classifier_sessions_destination': '/logs/verifier/judge-sessions',
                'format': 'sanitized-top-level-pi-jsonl', 'publication': 'exclusive-index-last',
                'matcher_sessions': 'unavailable_upstream_in_memory', 'matcher_usage': 'not_retained_by_eval',
                'monetary_usage_source': 'api-equivalent-estimate',
            },
        },
        'runtime': {'platform': platform,
                    'reviewer_seccomp_sha256': seccomp_sha,
                    'reviewer_seccomp_provenance_sha256': hashlib.sha256(seccomp_provenance_asset).hexdigest(),
                    'reviewer_seccomp_provenance': seccomp_provenance,
                    'verifier_seccomp': 'docker-default',
                    'base_image': base_image},
    }
    policy_sha = canonical_digest(policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reviewbench-export-', dir=output.parent) as temporary:
        exported = Path(temporary) / 'dataset'
        subprocess.run([
            'node', '--import', 'tsx', str(source / 'scripts/harbor/export.ts'),
            '--output', str(exported), '--organization', 'daydream-local', '--split', 'full',
            '--judge-provider', 'openai-codex', '--judge-model', CODEX_MODEL,
            '--base-image', base_image,
            *[argument for host in reviewer['allowed_hosts'] for argument in ('--agent-host', host)],
            '--judge-host', 'chatgpt.com', '--judge-host', 'auth.openai.com',
            '--judge-env', 'DAYDREAM_RB_PI_AUTH_PATH=RB_HARBOR_PI_AUTH_PATH',
        ], cwd=source, check=True)
        expected: dict[str, int] = {}
        references: list[str] = []
        for case in sorted(cases, key=lambda value: value.key):
            task_dir = exported / case.key
            task_path = task_dir / 'tests/task.json'
            task = json.loads(task_path.read_text())
            # Bind changed runtime policy to the evaluation ID while retaining the
            # upstream task ID and supplied source/golden identity.
            task['evaluation_id'] = int(canonical_digest({'upstream_evaluation_id': task['evaluation_id'],
                                                         'policy_sha256': policy_sha})[:13], 16)
            expected[str(task['task_id'])] = task['evaluation_id']
            _write_json(task_path, task)
            profile = {
                'schema': PUBLIC_PROFILE_SCHEMA, 'upstream_revision': REVIEWBENCH_REVISION, 'pr': task['pr'],
                'task_id': task['task_id'], 'evaluation_id': task['evaluation_id'],
                'reviewer': policy['reviewer'], 'candidate': dict(candidate), 'policy_sha256': policy_sha,
            }
            profile['profile_sha256'] = canonical_digest(profile)
            _write_json(task_dir / 'environment/daydream-profile.json', profile)
            docker_path = task_dir / 'environment/Dockerfile'
            docker = docker_path.read_text()
            docker = docker.replace('RUN git init -q .', reviewer_dockerfile_suffix + '\nRUN git init -q .', 1)
            docker += '\nCOPY daydream-profile.json /work/pr/daydream-profile.json\n'
            docker_path.write_text(docker, encoding='utf-8')
            for relative, content in task_assets.items():
                destination = task_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            toml_path = task_dir / 'task.toml'
            toml = toml_path.read_text()
            toml = toml.replace('[metadata]\n', f'[metadata]\ndaydream_policy_sha256 = "{policy_sha}"\n')
            toml_path.write_text(toml, encoding='utf-8')
            references.append(f'[[tasks]]\nname = {json.dumps("daydream-local/" + case.key)}\n'
                              f'digest = "sha256:{package_digest(task_dir)}"\n')
        metric = (source / 'scripts/harbor/metric.py').read_text().replace(
            'EXPECTED_TASKS = {}', 'EXPECTED_TASKS = ' + json.dumps(expected, sort_keys=True),
        )
        (exported / 'metric.py').write_text(metric, encoding='utf-8')
        metric_sha = hashlib.sha256(metric.encode()).hexdigest()
        _write_json(exported / 'daydream-profile.json', {**policy, 'policy_sha256': policy_sha})
        (exported / 'reviewer-seccomp.json').write_bytes(seccomp_asset)
        (exported / 'reviewer-seccomp-provenance.json').write_bytes(seccomp_provenance_asset)
        (exported / 'dataset.toml').write_text(
            'schema_version = "1.0"\n[dataset]\n'
            f'name = {json.dumps("daydream-local/reviewbench-native")}\n'
            'description = "Full public ReviewBench: 219 frozen PRs; custom subscription Codex judge."\n'
            'keywords = ["code-review", "reviewbench", "daydream"]\n\n'
            + '\n'.join(references) + ''.join(
                f'\n[[files]]\npath = {json.dumps(name)}\ndigest = "sha256:'
                f'{hashlib.sha256((exported / name).read_bytes()).hexdigest()}"\n'
                for name in ('metric.py', 'daydream-profile.json', 'reviewer-seccomp.json',
                             'reviewer-seccomp-provenance.json')
            ), encoding='utf-8',
        )
        # Recheck the supplied checkout after generation so concurrent source
        # changes cannot silently claim the originally admitted revision.
        if verify_source(source) != source_identity:
            raise ValueError('Supplied source changed during export')
        shutil.copytree(exported, output)
    return {'tasks': len(cases), 'policy_sha256': policy_sha, 'metric_sha256': metric_sha,
            'output': str(output), 'profile': policy}
