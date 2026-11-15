"""Report context collection: author context and bounded verification evidence."""

import json
from pathlib import Path
from typing import Any

from tests.harness.review_result import review_coverage


def test_report_collection_admits_only_valid_author_context(tmp_path: Path) -> None:
    """Retained optional prose is shape-checked before it can enter rendering or a canonical handoff."""
    from daydream.deep.report_context import collect_report_context
    from daydream.review_reporting import render_review_context

    coverage = review_coverage()
    author: Any = {'source': 'pull_request_body', 'head_sha': coverage.revision.head_sha,
                   'body': 'Retained author assertion', 'truncated': False, 'invented': True}
    for author_kind, raw in (('extra-field', json.dumps(author)), ('file-size', ' ' * 65536 + 'null')):
        root = tmp_path / author_kind
        root.mkdir()
        author_path = root / 'review-author-context.json'
        author_path.write_text(raw)
        collected = collect_report_context(root, coverage.to_dict())
        assert collected['author_reported'] is None
        assert 'Author-reported check context unavailable.' in render_review_context(collected)
        assert [scope['scope_id'] for scope in collected['scopes']] == ['python', 'structure']
        assert all(scope['review_context'] is None for scope in collected['scopes'])
        assert author_path.read_text() == raw
        assert collected['verification'] == {'status': 'unproduced', 'value': None}


def test_report_collection_bounds_existing_artifacts_and_prioritizes_test_identity(
    tmp_path: Path,
) -> None:
    """Bound aggregate handoff size while explicitly accounting for omitted existing evidence."""
    from daydream.deep.report_context import collect_report_context
    from daydream.findings import MAX_ARTIFACT_BYTES, write_findings_artifact
    from daydream.review_reporting import render_review_context
    from tests.harness.review_result import findings_artifact

    coverage = review_coverage(scope_ids=[f'python-{index}' for index in range(32)])
    names = ('test-verdict.json', 'remote-ci-verdict.json', 'fix-outcomes.json', 'fix-failures.json',
             'fix-quality-gate.json', 'recommended-capture.json')
    for name in names:
        payload = {'existing_evidence': name, 'padding': 'x' * 60_000}
        (tmp_path / name).write_text(json.dumps(payload, separators=(',', ':')))
    captured = collect_report_context(tmp_path, coverage.to_dict())
    assert (len(captured['scopes']), captured['scopes_omitted']) == (32, 0)
    assert len(json.dumps(captured).encode()) < 400_000
    verification = captured['verification']
    assert verification['status'] == 'failed' and verification['omitted_artifacts']
    assert set(verification['omitted_artifacts']) | set(verification['value']) >= set(names)
    assert {'test-verdict.json', 'remote-ci-verdict.json'} <= verification['value'].keys()
    output = tmp_path / 'findings.json'
    write_findings_artifact(output, findings_artifact(
        review_context=captured, terminal_result=coverage.finalize('completed')))
    assert output.stat().st_size < MAX_ARTIFACT_BYTES
    report = render_review_context(captured)
    assert 'omitted due to the report evidence budget' in report
