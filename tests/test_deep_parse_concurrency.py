"""Automatically planned reviewers retain stable records and merge provenance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes


async def test_automatic_shard_records_keep_stable_scope_ids_and_merge_order(
    shard_many_python_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = ReviewRun(shard_many_python_target, tmp_path, monkeypatch)
    review.backend.forbid_merge = review.backend.forbid_supervise = False
    review.backend.merge_echo_records = True
    review.backend.review_by_stack = {
        f'python#{index}': [dict(record(), file=f'mod{index}.py', evidence=f'mod{index}.py:2',
                                description=f'Defect in module {index}')]
        for index in range(3)}
    deep = review.repo / '.daydream/deep'
    observed: list[dict[str, Any]] = []
    for _ in range(2):
        assert await review.run(deep_shard_max_files=1, deep_shard_max_bytes=10**9) == 0
        data = review.load()
        result = data['terminal_result']
        assert list(scopes(data)) == ['generic', 'python#0', 'python#1', 'python#2', 'structure']
        assert all(scope['status'] == 'complete' for scope in scopes(data).values())
        assert [finding['title'] for finding in data['findings']] == [
            'Defect in module 0', 'Defect in module 1', 'Defect in module 2']
        records = {}
        for index in range(3):
            scope_id = f'python#{index}'
            saved = json.loads((deep / f'stack-{scope_id}-records.json').read_text())
            assert saved['scope_id'] == scope_id
            assert saved['analyzed_revision'] == result['analyzed_revision']
            assert [item['uid'] for item in saved['issues']] == [f'{scope_id}:1']
            records[scope_id] = saved['issues']
        merged = json.loads((deep / 'merged-items.json').read_text())
        assert [item['source_uids'] for item in merged['items']] == [
            ['python#0:1'], ['python#1:1'], ['python#2:1']]
        observed.append({'revision': result['analyzed_revision'], 'scopes': result['stack_outcomes'],
                         'records': records, 'source_uids': [item['source_uids'] for item in merged['items']]})
    assert observed[0] == observed[1]
