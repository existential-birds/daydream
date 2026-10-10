"""The installed Pi source tool delegates tracked dependencies to the host reader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from daydream.review_source import SourceRecipe
from tests.harness.git_helpers import git, seed_feature_branch
from tests.test_pi_native_source import isolated_pi as isolated_pi, native_source_provider as native_source_provider


@pytest.mark.usefixtures('native_source_provider')
async def test_installed_pi_reads_frozen_tracked_dependency_outside_assigned_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'pi_dependency'
    body = 'VALUE = "世界"\n'
    seed_feature_branch(repo, base={'dependency.py': body, 'api.py': 'VALUE = 1\n'},
                        feature={'api.py': 'VALUE = 2\n'})
    revision = git(repo, 'rev-parse', 'HEAD')
    recipe = SourceRecipe((), repo, head_revision=revision, repository_files=('api.py', 'dependency.py'),
                          repository_inventory_revision=revision)
    # A mutable checkout body is neither the selection authority nor the returned source.
    (repo / 'dependency.py').write_text('VALUE = "unbound checkout"\n')
    arguments = {'target_id': 'dependency.py', 'side': 'after'}
    monkeypatch.setenv('DAYDREAM_TEST_SOURCE_SELECTOR', json.dumps(arguments))
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'observed_body': {'type': 'string'}, 'source_error': {'type': 'boolean'}},
              'required': ['observed_body', 'source_error']}
    events = [event async for event in PiBackend(model='source-model').execute(
        repo, 'Read the tracked frozen dependency.', output_schema=schema,
        read_only=True, persist_session=False, source_recipe=recipe,
    )]
    start = next(event for event in events if isinstance(event, ToolStartEvent))
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    final = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert start.name == 'read_source' and start.input == arguments
    assert result.id == start.id and not result.is_error
    assert final.structured_output == {'observed_body': body, 'source_error': False}
    bound = recipe.native_result(arguments, result.output)
    assert bound is not None and bound.file == 'dependency.py' and bound.body == body
    assert bound.revision == revision
