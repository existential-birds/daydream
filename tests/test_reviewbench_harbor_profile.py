from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.reviewbench.harbor_build import NODE_IMAGE
from daydream.benchmark.reviewbench.harbor_profile import export_harbor_profile
from daydream.benchmark.reviewbench.native import freeze_reviewer
from daydream.run_config import RunConfig
from tests.test_reviewbench_seccomp import moby_policy as moby_policy

pytestmark = pytest.mark.usefixtures('moby_policy')


def test_invalid_public_profile_preserves_source_and_publishes_nothing(tmp_path: Path) -> None:
    output = tmp_path / 'refused'
    reviewer = freeze_reviewer(RunConfig(backend='codex', model='gpt-6.1-sol'))
    invalid: list[tuple[dict[str, Any], str]] = [
        ({'reviewer': {'backend': 'unsupported'}}, 'admitted frozen native'),
        ({'rounds': 0}, 'rounds must be positive'),
        ({'base_image': 'node:latest'}, 'pinned by sha256'),
        ({'platform': 'linux/ppc64le'}, 'supported Linux platform'),
    ]
    for option, error in invalid:
        options = {'wheel': tmp_path / 'unused.whl', 'base_image': NODE_IMAGE, 'reviewer': reviewer, **option}
        with pytest.raises(ValueError, match=error):
            export_harbor_profile(tmp_path / 'unread-source', output, **options)
        assert not output.exists()
        assert not list(tmp_path.glob('.reviewbench-export-*'))


def test_native_pi_export_rejects_unavailable_or_unisolated_operator_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PI_PROVIDER', 'nous')
    for change in ('missing-model', 'http', 'model-host', 'headers'):
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
        operator = tmp_path / change
        operator.mkdir()
        path = operator / 'models.json'
        original = json.dumps({'providers': {'nous': provider}})
        path.write_text(original)
        monkeypatch.setenv('PI_CODING_AGENT_DIR', str(operator))
        with pytest.raises(ValueError) as caught:
            freeze_reviewer(RunConfig(target=str(tmp_path), backend='pi', model='deepseek/deepseek-v4.1-flash'))
        assert 'synthetic-' not in str(caught.value), change
        assert path.read_text() == original
