"""Ordinary file inputs keep useful before-side context for file-only backends."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.deep_orchestrator.test_review_completion import record
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, StagedBackend, candidate
from tests.harness.git_helpers import git, seed_feature_branch, tracked_source_state


class PiBackend(StagedBackend):
    """File-only transport name activates the runner's ordinary before-side input path."""


@pytest.mark.parametrize('renamed', [False, True], ids=['deleted', 'renamed'])
@pytest.mark.parametrize('broken', [False, True], ids=['compatible', 'consumer-defect'])
async def test_before_source_is_furnished_as_a_private_ordinary_file_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, renamed: bool, broken: bool,
) -> None:
    repo = tmp_path / 'before_source'
    old_path = 'legacy.py' if renamed else 'retired.py'
    old = (''.join(f'# Stable API contract context {index:02}\n' for index in range(30)) if renamed else
           '# Public greeting used by retained consumers.\n') + 'def hello():\n    return "world"\n'
    base_consumer = 'EXPECTED_GREETING = "world"\n' if renamed else (
        'from retired import hello\nprint(hello())\n' if broken else 'print("world")\n')
    consumer = 'EXPECTED_GREETING = "universe"\n' if renamed and broken else base_consumer
    after = {'consumer.py': consumer + '# Existing public contract retained.\n'}
    if renamed:
        after['modern.py'] = old
    seed_feature_branch(repo, base={old_path: old, 'consumer.py': base_consumer}, feature=after)
    git(repo, 'rm', old_path)
    git(repo, 'commit', '-m', 'remove old greeting path')
    before = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = PiBackend(repo)
    title = ('Renamed greeting no longer satisfies the consumer contract' if renamed else
             'Retained consumer imports a deleted greeting module')

    @review.backend.script('python')
    def respond(stage: dict[str, Any], output: dict[str, Any]) -> None:
        contexts = json.loads(stage['access_guide'])['contexts']
        context, = [item for item in contexts if item['label'].startswith('before-context-')]
        path = Path(context['path'])
        captured = path.read_text()
        assert context['label'] in stage['context_inputs']
        assert f'# original_path: {old_path}' in captured and '# side: before' in captured
        assert captured.endswith(old) and not path.is_relative_to(repo)
        if broken:
            finding = dict(record(1), file='consumer.py', description=title,
                           rationale='The retained consumer disagrees with the removed source contract.',
                           evidence=('consumer.py:1 expects universe; legacy.py:32 returned world.' if renamed else
                                     'consumer.py:1 imports retired.hello; the supplied before-context defined it.'))
            output['candidates'] = [dict(candidate(disposition='confirmed', finding=finding),
                                         file='consumer.py', line=1,
                                         trigger='Run the retained consumer against the greeting contract',
                                         consequence=('The greeting contract disagrees' if renamed else
                                                      'ModuleNotFoundError'),
                                         grounds=finding['evidence'])]

    await review.finish('python', findings=(title,) if broken else ())
    assert not (repo / old_path).exists()
    assert (repo / 'modern.py').is_file() is renamed
    assert tracked_source_state(repo) == before
