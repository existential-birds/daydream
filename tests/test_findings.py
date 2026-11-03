"""Strict current findings envelopes, atomic writes, and production Phase A exports."""
import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.findings import (
    MAX_ARTIFACT_BYTES,
    FindingsValidationError,
    build_findings_artifact,
    load_findings_artifact,
    write_findings_artifact,
)
from daydream.pr_review import PRInfo
from tests.harness.execution import test_execution_identity as execution_identity
from tests.harness.review_result import findings_artifact, review_coverage, terminal_result
from tests.test_review_result import attribution_bytes, large_attributed_terminal


def _load(path: Path, *, head_sha: str = "h" * 40, **options: Any) -> Any:
    return load_findings_artifact(path, expected_repo="o/r", expected_pr_number=7,
                                 expected_head_sha=head_sha, **options)


@pytest.fixture
def valid_artifact() -> dict[str, Any]:
    return findings_artifact([{
        "fingerprint": "f" * 64, "path": "a.py", "line": 12, "placement": "inline", "title": "T", "body": "B",
        "severity": "high", "confidence": "HIGH", "is_cross_stack": False,
    }], run_info=None)


def _diagram_payload() -> dict[str, Any]:
    return {"eligibility": {"sequence": {"eligible": True, "rule": "forced", "reason": "r"}},
            "results": {"sequence": {"status": "rendered", "reason": None, "omit_reasons": [],
                                     "spec_final": {"participants": [], "messages": [], "blocks": []}},
                        "flowchart": None}}


def immutable_findings_artifact(
    findings: list[dict[str, Any]] | None = None, *, state: str = 'complete',
) -> dict[str, Any]:
    """A trusted synthetic input keeps original PR commits outside local attribution."""
    immutable: dict[str, Any] = {
        'schema_version': 1,
        'provenance': {'repository': 'o/r', 'pr_number': 7, 'head_sha': 'a' * 40,
                       'requested_base_sha': 'b' * 40, 'comparison_base_sha': 'c' * 40,
                       'title': 'Historical PR title', 'body': 'Historical PR body', 'context': ''},
        'materialization': {'kind': 'synthetic', 'head_sha': '1' * 40, 'requested_base_sha': '2' * 40,
                            'comparison_base_sha': '2' * 40, 'head_tree_sha': '3' * 40,
                            'comparison_tree_sha': '4' * 40,
                            'review_diff_key': '6' * 64, 'bundle_sha256': '7' * 64},
    }
    terminal = terminal_result(state, head_sha=immutable['materialization']['head_sha'])
    terminal['analyzed_revision'].update(merge_base_sha='2' * 40, diff_key='6' * 64)
    terminal['stack_outcomes'][0]['target_decisions'] = [{
        'target_id': 'part:000001', 'status': 'reviewed', 'reason': '',
        'evidence_refs': [{'artifact_sha256': '8' * 64, 'part_sha256': None,
                          'revision': '1' * 40, 'path': 'a.py', 'side': 'new',
                          'source_range': {'start': 1, 'end': 1}, 'fragment': None,
                          'basis': 'source_inspection', 'role': 'assigned'}],
    }]
    return findings_artifact(findings or [], schema_version=3, head_sha='a' * 40,
                             terminal_result=terminal, immutable_input=immutable)


def test_immutable_handoff_rejects_mixed_identities_before_replacing_prior_output(tmp_path: Path) -> None:
    """The ordinary same-head guard cannot bind synthetic source or the execution diff."""
    outer_head = immutable_findings_artifact()
    outer_head['head_sha'] = 'd' * 40
    comparison = immutable_findings_artifact()
    comparison['terminal_result']['analyzed_revision']['merge_base_sha'] = 'd' * 40
    path = tmp_path / 'prior.json'
    path.write_bytes(b'prior completed output')
    for artifact in (outer_head, comparison):
        with pytest.raises(FindingsValidationError):
            write_findings_artifact(path, artifact)
        assert path.read_bytes() == b'prior completed output'


def test_build_write_load_preserves_the_diagram_contract(tmp_path: Path, git_repo: Path) -> None:
    head = git_ops.head_sha(git_repo)
    pr = PRInfo(7, head, head, "main", "feature", "o", "r", "u")
    payload = _diagram_payload()
    artifact = build_findings_artifact(git_repo, pr, [], run_info=None, kind="diagram", diagrams=payload)
    assert (artifact["repo"], artifact["pr_number"], artifact["head_sha"]) == ("o/r", 7, head)
    path = tmp_path / "findings.json"
    write_findings_artifact(path, artifact)
    loaded = _load(path, head_sha=head)
    assert json.loads(path.read_text())["schema_version"] == loaded.schema_version == 2
    assert loaded.kind == "diagram" and loaded.diagrams == payload
    assert loaded.findings == [] and not loaded.analysis_complete


def test_review_defaults_and_known_run_freshness(tmp_path: Path) -> None:
    path = tmp_path / "reused.json"
    artifact = findings_artifact(terminal_result=large_attributed_terminal())
    write_findings_artifact(path, artifact)
    loaded = _load(path, expected_run_id="run-1")
    assert loaded.kind == "review" and loaded.diagrams is None and loaded.analysis_complete
    assert MAX_ARTIFACT_BYTES < path.stat().st_size < 9 * MAX_ARTIFACT_BYTES
    assert loaded.terminal_result == artifact['terminal_result']
    with pytest.raises(FindingsValidationError, match="run_id"):
        _load(path, expected_run_id="current-run")
    with pytest.raises(FindingsValidationError):
        _load(tmp_path / "fresh-current-run.json")
    damaged = copy.deepcopy(artifact)
    damaged['terminal_result']['stack_outcomes'][-1]['target_decisions'][-1]['evidence_refs'][0][
        'revision'] = 'x' * 40
    path.write_text(json.dumps(damaged))
    with pytest.raises(FindingsValidationError, match='revision differs'):
        _load(path)


def test_truncated_nested_or_unencodable_input_is_never_clean(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    for raw in ('{"schema_version": 2', '[' * 2000 + '0' + ']' * 2000):
        path.write_text(raw)
        with pytest.raises(FindingsValidationError):
            _load(path)
    path.write_text(json.dumps(findings_artifact(run_info='\ud800')))
    with pytest.raises(FindingsValidationError, match='encoding failed'):
        _load(path)


def test_oversized_input_is_never_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('[' + ' ' * (9 * MAX_ARTIFACT_BYTES))
    with monkeypatch.context() as unparsed:
        def forbidden_parse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('Oversized raw input reached JSON parsing')
        unparsed.setattr(json, 'loads', forbidden_parse)
        with pytest.raises(FindingsValidationError, match="size"):
            _load(path)
    path.write_text('{}')
    real_stat = Path.stat
    real_open = Path.open
    observations = []
    read_sizes: list[int] = []

    def grow_after_stat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        metadata = real_stat(self, *args, **kwargs)
        if self == path:
            observations.append(metadata.st_size)
            self.write_bytes(b'x' * (9 * MAX_ARTIFACT_BYTES + 1))
        return metadata

    def observed_open(self: Path, mode: str = 'r', *args: Any, **kwargs: Any) -> Any:
        stream: Any = real_open(self, mode, *args, **kwargs)
        if self == path and mode == 'rb':
            real_read = stream.read

            def observed_read(size: int = -1) -> bytes:
                read_sizes.append(size)
                captured: bytes = real_read(size)
                return captured

            stream.read = observed_read
        return stream

    with monkeypatch.context() as growing:
        growing.setattr(Path, 'stat', grow_after_stat)
        growing.setattr(Path, 'open', observed_open)
        with pytest.raises(FindingsValidationError, match='size'):
            _load(path)
    assert observations == [2], 'The actual file must grow after a successful small-file stat'
    assert read_sizes == [9 * MAX_ARTIFACT_BYTES + 1], 'Bound the actual read before decode or parsing'
    path.write_bytes(b'\xff')
    with pytest.raises(FindingsValidationError, match='read failed'):
        _load(path)


def test_semantic_invariants_rejected_before_write(tmp_path: Path, valid_artifact: dict[str, Any]) -> None:
    """A PR envelope cannot accept local source identity just because the underlying HEAD matches."""
    mutations: list[Any] = [
        lambda a: a.pop("terminal_result"),
        lambda a: (a['terminal_result'].update(schema_version=2),
                   a['terminal_result']['analyzed_revision'].update(source_id='d' * 64)),
        lambda a: a["terminal_result"]["stack_outcomes"].clear(),
    ]
    path = tmp_path / "invalid.json"
    for mutate in mutations:
        artifact = findings_artifact()
        mutate(artifact)
        with pytest.raises(FindingsValidationError):
            write_findings_artifact(path, artifact)
        assert not path.exists()
    # An untrustworthy projection cannot contain findings.
    c = review_coverage(scope_ids=("python",))
    c.record_scope("python", "complete")
    c.record_phase("merge", "failed", reasons=["malformed_artifact"])
    valid_artifact["terminal_result"] = c.finalize("failed", projection_valid=False)
    with pytest.raises(FindingsValidationError, match="projection"):
        write_findings_artifact(path, valid_artifact)
    assert not path.exists()


def test_failed_write_preserves_prior_bytes_and_leaves_no_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "atomic.json"
    path.write_bytes(b"prior current result")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("cannot serialize")

    artifact = findings_artifact()
    with monkeypatch.context() as serializer:
        serializer.setattr(json, "dumps", fail)
        with pytest.raises(FindingsValidationError):
            write_findings_artifact(path, artifact)
    assert path.read_bytes() == b"prior current result"
    path.unlink()
    oversized = findings_artifact()
    oversized["run_info"] = "é" * (MAX_ARTIFACT_BYTES // 2)
    assert len(oversized["run_info"]) < MAX_ARTIFACT_BYTES
    with pytest.raises(FindingsValidationError, match='size'):
        write_findings_artifact(path, oversized)
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def _review_context(artifact: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        'scopes': [{'scope_id': 'python', 'originating_run_id': 'run-1',
            'analyzed_revision': artifact['terminal_result']['analyzed_revision'], 'review_context': {
                'admitted_stages': 8, 'target_decisions': {'required': 8, 'reviewed': 8, 'not_reviewed': 0},
                'notes': [{'text': 'Existing reviewer rationale', 'origin': None,
                           'files': ['a' * 300] * 32, 'supporting_files': ['b' * 300] * 32} for _ in range(8)],
                'notes_omitted': 0}}], 'scopes_omitted': 0,
        'verification': {'status': 'available', 'value': {'test-verdict.json': verdict}},
        'author_reported': None,
    }


def test_optional_context_fits_remaining_budget_without_changing_findings(
    tmp_path: Path, valid_artifact: dict[str, Any], caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional prose cannot block a valid canonical handoff or weaken its existing byte cap."""
    identity = execution_identity(head_sha=valid_artifact['head_sha'])
    verdict = {'attempts': [{'kind': 'host', 'command': list(identity.argv), 'passed': True,
                             'identity': identity.payload()}]}
    # Trim: dropping scope prose alone fits, and verification survives.
    small = copy.deepcopy(valid_artifact)
    trimmed = _review_context(small, verdict) | {'scopes': [], 'scopes_omitted': 1}
    fitted_size = len((json.dumps({**small, 'review_context': trimmed}, indent=2, ensure_ascii=False)
                       + '\n').encode('utf-8'))
    small['review_context'] = _review_context(small, verdict)
    with monkeypatch.context() as capped:
        capped.setattr('daydream.findings.MAX_ARTIFACT_BYTES', fitted_size)
        write_findings_artifact(tmp_path / 'trimmed.json', small)
    context = json.loads((tmp_path / 'trimmed.json').read_text())['review_context']
    assert context['scopes'] == [] and context['scopes_omitted'] == 1
    assert context['verification']['value']['test-verdict.json'] == verdict
    # Omit: core exactly fills the cap, so all optional context is dropped.
    artifact = valid_artifact
    artifact['terminal_result'] = large_attributed_terminal()
    artifact['run_info'] = ''
    size = (len((json.dumps(artifact, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
            - attribution_bytes(artifact))
    artifact['run_info'] = 'x' * (MAX_ARTIFACT_BYTES - size)
    core = copy.deepcopy(artifact)
    artifact['review_context'] = _review_context(artifact, verdict)
    original_context = copy.deepcopy(artifact['review_context'])
    path = tmp_path / 'fitted.json'
    write_findings_artifact(path, core)
    write_findings_artifact(path, artifact)
    assert MAX_ARTIFACT_BYTES < path.stat().st_size <= 9 * MAX_ARTIFACT_BYTES
    loaded = json.loads(path.read_text())
    assert {key: loaded[key] for key in core} == core
    assert _load(path).analysis_complete
    assert artifact['review_context'] == original_context
    assert 'Optional review context omitted' in caplog.text
    assert 'review_context' not in loaded
    assert path.stat().st_size - attribution_bytes(loaded) == MAX_ARTIFACT_BYTES
