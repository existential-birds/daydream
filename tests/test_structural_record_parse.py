"""Structural findings remain partitioned through review and resume."""

import json
from pathlib import Path
from typing import Any

from tests.harness.review_result import records_artifact, review_coverage


def test_report_collection_keeps_each_scope_bound_and_optional_faults_isolated(tmp_path: Path) -> None:
    """One read acquires independent sidecars without discarding or mutating canonical siblings."""
    from daydream.deep.report_context import collect_report_context
    from daydream.phases.review import valid_record_artifact

    kinds = [
        "missing",
        "current",
        "unattributed",
        "counts",
        "notes",
        "secret",
        "origin",
        "scope",
        "revision",
        "run",
        "nested-sidecar",
        "arabic-ordinal", "zero-padded-ordinal",
    ]
    coverage = review_coverage(scope_ids=["python-" + kind for kind in kinds])
    canonical, expected, raw_sidecars = {}, {}, {}
    for kind in kinds:
        scope = "python-" + kind
        ordinal = {"arabic-ordinal": "١", "zero-padded-ordinal": "01"}.get(kind, "1")
        artifact = records_artifact(coverage, scope, [{
            "id": 1, "uid": f"{scope}:{ordinal}", "file": "api.py", "line": 1,
            "description": "Boundary mismatch", "severity": "high", "confidence": "HIGH",
            "rationale": "Shared contract", "evidence": "api.py:1",
        }] if ordinal != "1" else [])
        records_path = tmp_path / f"stack-{scope}-records.json"
        canonical[records_path] = json.dumps(artifact)
        records_path.write_text(canonical[records_path])
        context: dict[str, Any] = {
            "admitted_stages": 1,
            "target_decisions": {"required": 1, "reviewed": 1, "not_reviewed": 0},
            "notes": [
                {
                    "text": "Source inspection found compatible callers.",
                    "files": ["api.py"],
                    "supporting_files": [],
                    "origin": {
                        "scope_id": scope,
                        "stage": "first_pass",
                        "sequence": 1,
                        "judgment_origin": "reviewer",
                        "analyzed_revision": coverage.revision.to_dict(),
                    },
                }
            ],
            "notes_omitted": 0,
        }
        sidecar = {key: value for key, value in artifact.items() if key != "issues"}
        sidecar["review_context"] = context
        if kind == "unattributed":
            context["notes"][0]["origin"] = None
        elif kind == "counts":
            context["target_decisions"]["not_reviewed"] = 1
        elif kind == "notes":
            context["notes"] *= 9
        elif kind == "secret":
            context["notes"][0]["text"] = "ghp_1234567890"
        elif kind == "origin":
            context["notes"][0]["origin"] = {"scope_id": scope}
        elif kind == "scope":
            sidecar["scope_id"] = "foreign"
        elif kind == "revision":
            sidecar["analyzed_revision"] = {**coverage.revision.to_dict(), "head_sha": "foreign"}
        elif kind == "run":
            sidecar["originating_run_id"] = "foreign"
        expected[scope] = context if kind in {"current", "unattributed"} else None
        if kind != "missing":
            if kind == "nested-sidecar":
                context["notes"] = "nested_placeholder"
                raw = json.dumps(sidecar).replace('"nested_placeholder"', "[" * 30000 + "0" + "]" * 30000)
                assert len(raw.encode()) < 65536
            else:
                raw = json.dumps(sidecar)
            path = tmp_path / f"stack-{scope}-report-context.json"
            raw_sidecars[path] = raw
            path.write_text(raw)
        if ordinal == "1":
            assert valid_record_artifact(artifact, scope_id=scope, analyzed_revision=coverage.revision.to_dict())
        assert not valid_record_artifact(
            {**artifact, "review_context": context}, scope_id=scope, analyzed_revision=coverage.revision.to_dict()
        )
    (tmp_path / "test-verdict.json").write_text("[" * 2000 + "]" * 2000)

    collected = collect_report_context(tmp_path, coverage.to_dict())

    assert collected["verification"] == {"status": "failed", "value": None}
    assert {entry["scope_id"]: entry["review_context"] for entry in collected["scopes"]} == expected
    rejected = {"python-arabic-ordinal", "python-zero-padded-ordinal"}
    assert {entry["scope_id"]: (entry["originating_run_id"], entry["analyzed_revision"])
            for entry in collected["scopes"]} == {
        scope: (None, None) if scope in rejected else (coverage.run_id, coverage.revision.to_dict())
        for scope in expected
    }
    assert all(path.read_text() == raw for path, raw in (canonical | raw_sidecars).items())


def test_report_collection_admits_only_valid_author_context(tmp_path: Path) -> None:
    """Retained optional prose is shape-checked before it can enter rendering or a canonical handoff."""
    from daydream.deep.report_context import collect_report_context
    from daydream.review_reporting import render_review_context

    coverage = review_coverage()
    record = records_artifact(coverage, 'python')
    author: Any = {'source': 'pull_request_body', 'head_sha': coverage.revision.head_sha,
                   'body': 'Retained author assertion', 'truncated': False, 'invented': True}
    for author_kind, raw in (('extra-field', json.dumps(author)), ('file-size', ' ' * 65536 + 'null')):
        root = tmp_path / author_kind
        root.mkdir()
        records_path = root / 'stack-python-records.json'
        records_path.write_text(json.dumps(record))
        author_path = root / 'review-author-context.json'
        author_path.write_text(raw)
        collected = collect_report_context(root, coverage.to_dict())
        assert collected['author_reported'] is None
        assert 'Author-reported check context unavailable.' in render_review_context(collected)
        assert collected['scopes'][0]['originating_run_id'] == coverage.run_id
        assert collected['scopes'][0]['analyzed_revision'] == coverage.revision.to_dict()
        assert json.loads(records_path.read_text()) == record and author_path.read_text() == raw
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
    for scope in coverage.planned_scopes:
        artifact = records_artifact(coverage, scope.scope_id)
        (tmp_path / f'stack-{scope.scope_id}-records.json').write_text(json.dumps(artifact))
        (tmp_path / f'stack-{scope.scope_id}-report-context.json').write_text(json.dumps({
            **{key: value for key, value in artifact.items() if key != 'issues'}, 'review_context': {
                'admitted_stages': 8, 'target_decisions': {'required': 8, 'reviewed': 8, 'not_reviewed': 0},
                'notes': [{'text': 'x' * 512, 'files': [], 'supporting_files': [], 'origin': None}] * 8,
                'notes_omitted': 0}}))
    names = ('test-verdict.json', 'remote-ci-verdict.json', 'fix-outcomes.json', 'fix-failures.json',
             'fix-quality-gate.json', 'recommended-capture.json')
    for name in names:
        payload = {'existing_evidence': name, 'padding': 'x' * 60_000}
        (tmp_path / name).write_text(json.dumps(payload, separators=(',', ':')))
    captured = collect_report_context(tmp_path, coverage.to_dict())
    assert 0 < len(captured['scopes']) < 32
    assert captured['scopes_omitted'] == 32 - len(captured['scopes'])
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
