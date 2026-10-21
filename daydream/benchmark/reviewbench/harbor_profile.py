"""Content-addressed public ReviewBench tasks with a subscription-only Codex judge.

The supplied exporter and judging sources are executed unchanged. This module owns
only the explicitly distinct local evaluation profile and its runtime bootstrap.
It never reads credentials and never publishes a dataset.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from daydream.benchmark.reviewbench.contract import (
    REVIEWBENCH_REVISION,
    load_cases,
    verify_source,
)

CODEX_MODEL = 'gpt-6.1-sol'
SUBSCRIPTION_HOME = '/run/reviewbench-codex'
PROFILE_PATH = '/work/pr/daydream-profile.json'
COMPOSE = '''services:
  main:
    platform: linux/amd64
    volumes:
      - type: bind
        source: ${RB_HARBOR_AUTH_DIR:?Set a dedicated subscription runtime directory}
        target: /run/reviewbench-codex
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


def codex_models(model: str = CODEX_MODEL) -> dict[str, Any]:
    """Explicit Pi metadata; maxTokens is metadata, not a transport-enforced cap.

    Prices are API-equivalent estimates only. Subscription cash charges are not
    available from the native transport and must not be inferred from these rates.
    """
    if model != CODEX_MODEL:
        raise ValueError(f'This frozen subscription profile supports only {CODEX_MODEL}')
    return {'providers': {'openai-codex': {
        'api': 'openai-codex-responses',
        'baseUrl': 'https://chatgpt.com/backend-api',
        'models': [{
            'id': model, 'name': model, 'reasoning': True, 'input': ['text', 'image'],
            'contextWindow': 1050000, 'maxTokens': 8192,
            'cost': {'input': 2, 'output': 10, 'cacheRead': 0.1, 'cacheWrite': 2.5},
        }],
    }}}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def _image(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[a-f0-9]{64}', value):
        raise ValueError('Runtime images must be pinned by sha256 digest')
    return value


def export_harbor_profile(
    source: Path,
    output: Path,
    *,
    candidate: Mapping[str, Any],
    base_image: str,
    reviewer_image: str | None = None,
    reviewer_model: str = CODEX_MODEL,
    reviewer_effort: str = 'high',
    reviewer_timeout: int = 900,
    verifier_timeout: int = 7200,
    reviewer_dockerfile_suffix: str = '',
    reviewer_assets: Mapping[str, bytes] | None = None,
    organization: str = 'daydream-local',
    rounds: int = 3,
) -> dict[str, Any]:
    """Export all 219 supplied tasks without inference into a NEW local profile.

    ``candidate`` must contain only nonsecret immutable package/runtime identity.
    Additional reviewer build assets belong beneath its environment directory.
    Selection/concurrency/independent rounds are Harbor run controls applied AFTER
    this full export; no tiny export can be mislabeled as the full benchmark.
    """
    source = source.resolve()
    output = output.absolute()
    if output.exists():
        raise ValueError('Export destination already exists; preserve existing evidence')
    source_identity = verify_source(source)
    cases = load_cases(source, 'full')
    models = codex_models(reviewer_model)
    if type(rounds) is not int or rounds < 1:
        raise ValueError('Independent comparison rounds must be positive')
    _image(base_image)
    if reviewer_image is not None:
        _image(reviewer_image)
    if reviewer_effort not in {'low', 'medium', 'high', 'xhigh'}:
        raise ValueError('Unsupported native Codex reasoning effort')
    if (type(reviewer_timeout) is not int or reviewer_timeout < 1
            or type(verifier_timeout) is not int or verifier_timeout < 1):
        raise ValueError('Reviewer and verifier wall budgets must be positive integers')
    # No arbitrary kwargs/environment escape hatch: profile data is retained in
    # images and jobs. Explicitly refuse common credential-bearing identities.
    serialized_candidate = json.dumps(dict(candidate), sort_keys=True)
    if re.search(r'(?i)(api.?key|access.?token|refresh.?token|id.?token|password|secret)', serialized_candidate):
        raise ValueError('Candidate identity must contain no credentials or secret fields')
    from daydream import json_utils
    from daydream.benchmark.reviewbench.harbor_subscription_plugin import ADMISSION_POLICY

    auth_asset = Path(__file__).with_name('harbor_auth.py').read_bytes()
    atomic_asset = Path(json_utils.__file__).read_bytes()
    policy = {
        'schema': 1, 'name': 'daydream-reviewbench-codex-subscription-v1',
        'source': source_identity, 'source_prs': len(cases),
        'candidate': dict(candidate), 'final_comparison_rounds': rounds,
        'subscription_admission': {**ADMISSION_POLICY, 'plugin_sha256': hashlib.sha256(
            Path(__file__).with_name('harbor_subscription_plugin.py').read_bytes()).hexdigest()},
        'profile_builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reviewer': {'model': reviewer_model, 'reasoning_effort': reviewer_effort, 'timeout_sec': reviewer_timeout},
        'judge': {
            'provider': 'openai-codex', 'model': reviewer_model, 'auth': 'subscription-oauth',
            'matcher_effort': 'low', 'classifier_effort': 'medium',
            'call_timeout_ms': 600000, 'classifier_extra_per_finding_ms': 120000,
            'verifier_timeout_sec': verifier_timeout, 'models': models,
            'max_output_tokens_enforced': False, 'cost_source': 'api-equivalent-estimate',
            'auth_bootstrap_sha256': hashlib.sha256(auth_asset).hexdigest(),
            'atomic_writer_sha256': hashlib.sha256(atomic_asset).hexdigest(),
            'refresh_policy': 'canonical-exclusive-full-invocation-lease',
        },
        'runtime': {'platform': 'linux/amd64', 'auth_compose_sha256': hashlib.sha256(COMPOSE.encode()).hexdigest(),
                    'verifier_image': base_image, 'reviewer_image': reviewer_image or base_image,
                    'reviewer_build_suffix_sha256': hashlib.sha256(reviewer_dockerfile_suffix.encode()).hexdigest(),
                    'reviewer_assets': {key: hashlib.sha256(value).hexdigest()
                                        for key, value in sorted((reviewer_assets or {}).items())}},
    }
    policy_sha = canonical_digest(policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reviewbench-export-', dir=output.parent) as temporary:
        exported = Path(temporary) / 'dataset'
        subprocess.run([
            'node', '--import', 'tsx', str(source / 'scripts/harbor/export.ts'),
            '--output', str(exported), '--organization', organization, '--split', 'full',
            '--judge-provider', 'openai-codex', '--judge-model', reviewer_model,
            '--base-image', base_image, '--agent-host', 'chatgpt.com', '--agent-host', 'auth.openai.com',
            '--judge-host', 'chatgpt.com', '--judge-host', 'auth.openai.com',
            '--judge-env', 'DAYDREAM_SUBSCRIPTION_HOME=RB_HARBOR_SUBSCRIPTION_HOME',
            '--verifier-timeout', str(verifier_timeout),
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
                'schema': 1, 'upstream_revision': REVIEWBENCH_REVISION, 'pr': task['pr'],
                'task_id': task['task_id'], 'evaluation_id': task['evaluation_id'],
                'reviewer': policy['reviewer'], 'candidate': dict(candidate), 'policy_sha256': policy_sha,
            }
            profile['profile_sha256'] = canonical_digest(profile)
            _write_json(task_dir / 'environment/daydream-profile.json', profile)
            docker_path = task_dir / 'environment/Dockerfile'
            docker = docker_path.read_text()
            if reviewer_image:
                docker = docker.replace(f'FROM {base_image}\n', f'FROM {reviewer_image}\n', 1)
            docker = docker.replace('RUN git init -q .', reviewer_dockerfile_suffix + '\nRUN git init -q .', 1)
            docker += '\nCOPY daydream-profile.json /work/pr/daydream-profile.json\n'
            docker_path.write_text(docker, encoding='utf-8')
            for relative, content in (reviewer_assets or {}).items():
                relative_path = Path(relative)
                if relative_path.is_absolute() or '..' in relative_path.parts:
                    raise ValueError('Reviewer build assets must remain within the environment directory')
                destination = task_dir / 'environment' / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            (task_dir / 'tests/daydream-auth.py').write_bytes(auth_asset)
            (task_dir / 'tests/daydream_json_utils.py').write_bytes(atomic_asset)
            for role_directory in ('environment', 'tests'):
                (task_dir / role_directory / 'docker-compose.yaml').write_text(COMPOSE, encoding='utf-8')
            _write_json(task_dir / 'tests/judge/.daydream-codex/models.json', models)
            bootstrap = (
                '#!/bin/bash\nset -euo pipefail\ncd /tests/judge\n'
                'exec python3 /tests/daydream-auth.py judge '
                f'--canonical-home {SUBSCRIPTION_HOME} '
                '--models /tests/judge/.daydream-codex/models.json -- '
                'node --import tsx scripts/harbor/verify.ts /tests/task.json\n'
            )
            (task_dir / 'tests/test.sh').write_text(bootstrap, encoding='utf-8')
            toml_path = task_dir / 'task.toml'
            toml = toml_path.read_text().replace('[agent]\ntimeout_sec = 900\n',
                                                  f'[agent]\ntimeout_sec = {reviewer_timeout}\n')
            toml = toml.replace('[metadata]\n', f'[metadata]\ndaydream_policy_sha256 = "{policy_sha}"\n')
            toml_path.write_text(toml, encoding='utf-8')
            references.append(f'[[tasks]]\nname = {json.dumps(organization + "/" + case.key)}\n'
                              f'digest = "sha256:{package_digest(task_dir)}"\n')
        metric = (source / 'scripts/harbor/metric.py').read_text().replace(
            'EXPECTED_TASKS = {}', 'EXPECTED_TASKS = ' + json.dumps(expected, sort_keys=True),
        )
        (exported / 'metric.py').write_text(metric, encoding='utf-8')
        metric_sha = hashlib.sha256(metric.encode()).hexdigest()
        _write_json(exported / 'daydream-profile.json', {**policy, 'policy_sha256': policy_sha})
        profile_sha = hashlib.sha256((exported / 'daydream-profile.json').read_bytes()).hexdigest()
        (exported / 'dataset.toml').write_text(
            'schema_version = "1.0"\n[dataset]\n'
            f'name = {json.dumps(organization + "/reviewbench-codex-subscription")}\n'
            'description = "Full public ReviewBench: 219 frozen PRs; custom subscription Codex judge."\n'
            'keywords = ["code-review", "reviewbench", "daydream"]\n\n'
            + '\n'.join(references) + '\n[[files]]\npath = "metric.py"\n'
            f'digest = "sha256:{metric_sha}"\n'
            '\n[[files]]\npath = "daydream-profile.json"\n'
            f'digest = "sha256:{profile_sha}"\n', encoding='utf-8',
        )
        # Recheck the supplied checkout after generation so concurrent source
        # changes cannot silently claim the originally admitted revision.
        if verify_source(source) != source_identity:
            raise ValueError('Supplied source changed during export')
        shutil.move(str(exported), output)
    return {'tasks': len(cases), 'policy_sha256': policy_sha, 'metric_sha256': metric_sha,
            'output': str(output), 'profile': policy}
