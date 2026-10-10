"""Ordinary file inputs keep useful before-side context for file-only backends."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.deep_orchestrator.test_review_completion import scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, StagedBackend
from tests.harness.git_helpers import seed_feature_branch, tracked_source_state


class PiBackend(StagedBackend):
    """File-only transport name activates the runner's ordinary before-side input path."""


@pytest.mark.parametrize('broken', [False, True], ids=['retired-clean', 'live-caller-defect'])
async def test_deleted_source_is_furnished_as_an_ordinary_file_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bool,
) -> None:
    repo = tmp_path / 'deleted_contract'
    old = '# Public greeting used by retained consumers.\ndef hello():\n    return "world"\n'
    consumer = 'from retired import hello\nprint(hello())\n' if broken else 'print("world")\n'
    seed_feature_branch(repo, base={'retired.py': old, 'consumer.py': consumer},
                        feature={'consumer.py': consumer + '# audited consumer\n'})
    from tests.harness.git_helpers import git
    git(repo, 'rm', 'retired.py')
    git(repo, 'commit', '-m', 'remove greeting module')
    before = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = PiBackend(repo)

    def respond(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        contexts = json.loads(stage['access_guide'])['contexts']
        old_contexts = [item for item in contexts if item['label'].startswith('before-context-')]
        assert len(old_contexts) == 1, (type(review.backend).__name__, stage['context_statuses'], stage['access_guide'])
        context = Path(old_contexts[0]['path'])
        captured = context.read_text()
        assert old_contexts[0]['label'] in stage['context_inputs']
        assert '# original_path: retired.py' in captured and '# side: before' in captured, repr(captured)
        assert captured.endswith(old), repr(captured)
        assert not context.is_relative_to(repo)
        if broken:
            finding = {'id': 1, 'file': 'consumer.py', 'line': 1, 'severity': 'medium',
                       'description': 'Retained consumer imports a deleted greeting module',
                       'confidence': 'MEDIUM',
                       'rationale': 'The import cannot resolve after retired.py was deleted.',
                       'evidence': 'consumer.py:1 imports retired.hello; the supplied before-context defined it.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'consumer.py', 'line': 1,
                                     'trigger': 'Run the retained consumer', 'consequence': 'ModuleNotFoundError',
                                     'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = respond
    data = await review.finish('python', findings=('Retained consumer imports a deleted greeting module',)
                               if broken else ())
    assert scopes(data)['python']['status'] == 'complete'
    assert [finding['title'] for finding in data['findings']] == (
        ['Retained consumer imports a deleted greeting module'] if broken else [])
    assert not (repo / 'retired.py').exists()
    assert tracked_source_state(repo) == before


@pytest.mark.parametrize('broken', [False, True], ids=['compatible-rename', 'renamed-source-defect'])
async def test_renamed_source_keeps_its_before_context_as_an_ordinary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bool,
) -> None:
    repo = tmp_path / 'renamed_source'
    prefix = ''.join(f'# Stable API contract context {index:02}\n' for index in range(30))
    old = prefix + 'def hello():\n    return "world"\n'
    new = old
    base_consumer = 'EXPECTED_GREETING = "world"\n'
    changed_consumer = 'EXPECTED_GREETING = "universe"\n' if broken else base_consumer
    seed_feature_branch(repo, base={'legacy.py': old, 'consumer.py': base_consumer},
                        feature={'modern.py': new, 'consumer.py': changed_consumer +
                                 '# Existing public contract retained.\n'})
    from tests.harness.git_helpers import git
    git(repo, 'rm', 'legacy.py')
    git(repo, 'commit', '-m', 'remove old path after relocating module')
    before = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = PiBackend(repo)

    def respond(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        contexts = json.loads(stage['access_guide'])['contexts']
        old_contexts = [item for item in contexts if item['label'].startswith('before-context-')]
        assert len(old_contexts) == 1, (type(review.backend).__name__, stage['context_statuses'], stage['access_guide'])
        context = Path(old_contexts[0]['path'])
        captured = context.read_text()
        assert old_contexts[0]['label'] in stage['context_inputs']
        assert '# original_path: legacy.py' in captured and '# side: before' in captured
        assert captured.endswith(old)
        if broken:
            finding = {'id': 1, 'file': 'consumer.py', 'line': 1, 'severity': 'medium',
                       'description': 'Renamed greeting no longer satisfies the consumer contract',
                       'confidence': 'MEDIUM', 'rationale': 'The consumer expects a different greeting.',
                       'evidence': 'consumer.py:1 expects universe; legacy.py:32 returned world.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'consumer.py', 'line': 1,
                                     'trigger': 'Compare the consumer contract with the relocated hello',
                                     'consequence': 'The greeting contract disagrees',
                                     'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = respond
    data = await review.finish('python', findings=('Renamed greeting no longer satisfies the consumer contract',)
                               if broken else ())
    assert scopes(data)['python']['status'] == 'complete'
    assert [finding['title'] for finding in data['findings']] == (
        ['Renamed greeting no longer satisfies the consumer contract'] if broken else [])
    assert not (repo / 'legacy.py').exists() and (repo / 'modern.py').is_file()
    assert tracked_source_state(repo) == before
