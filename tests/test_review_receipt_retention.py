"""Supplement the runner's capture-loss proof with the retained receipt bound."""
from __future__ import annotations

from pathlib import Path

import pytest

from daydream.backends import ToolResultEvent, ToolStartEvent
from daydream.review_evidence import ReviewEvidence


@pytest.mark.parametrize('field', ['call-id', 'tool-name', 'native-status'])
def test_oversized_native_metadata_cannot_remain_in_retained_receipts(tmp_path: Path, field: str) -> None:
    source = 'VALUE = 1\n'
    (tmp_path / 'api.py').write_text(source)
    evidence = ReviewEvidence(None)
    evidence.configure_capture(tmp_path)
    too_large = 'x' * 2_049
    call_id = too_large if field == 'call-id' else 'read-source'
    name = too_large if field == 'tool-name' else 'Read'
    evidence.observe(ToolStartEvent(id=call_id, name=name, input={'file_path': 'api.py'}))
    evidence.observe(ToolResultEvent(id=call_id, output=source, is_error=False,
                                     status=too_large if field == 'native-status' else None))
    assert evidence.retention_overflow_results == 1
    assert evidence.capture_failure(['api.py']) is True
    assert all(len(value.encode()) <= 2_048 for receipt in evidence.receipts
               for value in (receipt.call.id, receipt.call.name, receipt.result.id, receipt.result.status or ''))


@pytest.mark.parametrize(('operand', 'inventory', 'expected'), [
    ('missing.json', 'complete', True), ('missing.json', 'default', False),
    ('missing.json', 'wrong-revision', False), ('api.py', 'complete', False),
    ('API.py', 'complete', False), ('root-case', 'complete', False), ('double-root', 'complete', False),
    ('api.py', 'empty-head', True),
])
def test_frozen_absence_needs_complete_head_authority_and_rejects_path_aliases(
    tmp_path: Path, operand: str, inventory: str, expected: bool,
) -> None:
    from daydream.git_ops.queries import ls_tree_files
    from daydream.review_source import SourceRecipe
    from tests.harness.git_helpers import git, seed_feature_branch

    repo = tmp_path / 'frozen_checkout'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': 'VALUE = 1\n'})
    if inventory == 'empty-head':
        git(repo, 'rm', 'api.py')
        git(repo, 'commit', '-m', 'empty the frozen tree')
    else:
        (repo / 'api.py').unlink()
    head = git(repo, 'rev-parse', 'HEAD')
    binding = (None if inventory == 'default' else git(repo, 'rev-parse', 'main')
               if inventory == 'wrong-revision' else head)
    files = tuple(ls_tree_files(repo, head, strict=True))
    recipe = SourceRecipe((), repo, head_revision=head, repository_files=files,
                          repository_inventory_revision=binding)
    if operand == 'root-case':
        operand = str(repo.parent / repo.name.upper() / 'api.py')
    elif operand == 'double-root':
        operand = '/' + str(repo / 'api.py')
    assert recipe.unavailable_lookup('read', {'path': operand}, repo, set()) is expected
