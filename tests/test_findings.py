"""Strict current findings envelopes, atomic writes, and production Phase A exports."""
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.backends import TextEvent
from daydream.backends.claude import ClaudeAgentError
from daydream.findings import (
    MAX_ARTIFACT_BYTES,
    FindingsValidationError,
    build_findings_artifact,
    load_findings_artifact,
    write_findings_artifact,
)
from daydream.pr_review import ParsedIssue, PRInfo
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.harness.backend import ScriptedBackend
from tests.harness.execution import test_execution_identity as execution_identity
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit, git
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


def test_immutable_export_places_against_local_source_without_relabeling_it(
    git_repo: Path, tmp_path: Path, fake_gh: FakeGh,
) -> None:
    """Schema-2 round trips cannot detect placement against unavailable original commits."""
    (git_repo / 'a.py').write_text('value = 1\n')
    git(git_repo, 'add', 'a.py')
    base = commit(git_repo, 'base source')
    (git_repo / 'a.py').write_text('value = 2\n')
    git(git_repo, 'add', 'a.py')
    head = commit(git_repo, 'changed source')
    diff = git_ops.diff_paths(git_repo, base, head, ['.'])
    immutable = immutable_findings_artifact()['immutable_input']
    local = immutable['materialization']
    local.update(head_sha=head, requested_base_sha=base, comparison_base_sha=base,
                 head_tree_sha=git(git_repo, 'rev-parse', f'{head}^{{tree}}'),
                 comparison_tree_sha=git(git_repo, 'rev-parse', f'{base}^{{tree}}'),
                 review_diff_key=hashlib.sha256(diff.encode()).hexdigest())
    terminal = terminal_result(head_sha=head)
    terminal['analyzed_revision'].update(merge_base_sha=base, diff_key=local['review_diff_key'])
    pr = PRInfo(7, 'a' * 40, 'c' * 40, 'main', 'feature', 'o', 'r', '')
    issue = ParsedIssue(path='a.py', line=1, title='Changed `value`', body='Missing guard',
                        severity='high', confidence='HIGH', fingerprint='f' * 64, is_cross_stack=False)
    artifact = build_findings_artifact(git_repo, pr, [issue], run_info=None, snapshot_diff=diff,
                                       terminal_result=terminal, immutable_input=immutable)
    path = tmp_path / 'immutable-findings.json'
    write_findings_artifact(path, artifact)
    loaded = _load(path, head_sha='a' * 40, expected_base_sha='b' * 40, expected_comparison_sha='c' * 40)
    assert loaded.schema_version == 3 and loaded.head_sha == 'a' * 40
    assert [(f.path, f.line, f.placement) for f in loaded.findings] == [('a.py', 1, 'inline')]
    assert loaded.terminal_result == terminal and loaded.immutable_input == immutable
    assert loaded.terminal_result['analyzed_revision']['head_sha'] == head != loaded.head_sha
    assert fake_gh.process_calls() == []


@pytest.mark.parametrize('field', ['outer-head', 'local-head', 'comparison', 'diff-key', 'source-ref', 'version'])
def test_immutable_handoff_rejects_mixed_identities_before_replacing_prior_output(
    tmp_path: Path, field: str,
) -> None:
    """The ordinary same-head guard cannot bind synthetic source or the execution diff."""
    artifact = immutable_findings_artifact()
    if field == 'outer-head':
        artifact['head_sha'] = 'd' * 40
    elif field == 'version':
        artifact['immutable_input']['schema_version'] = 99
    elif field == 'source-ref':
        artifact['terminal_result']['stack_outcomes'][0]['target_decisions'][0]['evidence_refs'][0][
            'revision'] = artifact['head_sha']
    else:
        revision = artifact['terminal_result']['analyzed_revision']
        key = {'local-head': 'head_sha', 'comparison': 'merge_base_sha', 'diff-key': 'diff_key'}[field]
        revision[key] = 'd' * (64 if field == 'diff-key' else 40)
    path = tmp_path / 'prior.json'
    path.write_bytes(b'prior completed output')
    with pytest.raises(FindingsValidationError):
        write_findings_artifact(path, artifact)
    assert path.read_bytes() == b'prior completed output'


@pytest.mark.parametrize("kind", ["review", "diagram"])
def test_build_write_load_preserves_target_finding_and_diagram_contract(
    tmp_path: Path, git_repo: Path, kind: str,
) -> None:
    head = git_ops.head_sha(git_repo)
    pr = PRInfo(7, head, head, "main", "feature", "o", "r", "u")
    issues = [ParsedIssue(path="a.py", line=None, title="T", body="B", severity="high",
                          confidence="HIGH", fingerprint="f" * 64, is_cross_stack=True)] if kind == "review" else []
    payload = _diagram_payload() if kind == "diagram" else None
    options: dict[str, Any] = {}
    if kind == "review":
        options = {"terminal_result": terminal_result(head_sha=head),
                   "snapshot_diff": git_ops.diff_paths(git_repo, head, head, ["."])}
    artifact = build_findings_artifact(git_repo, pr, issues, run_info=None,
                                       kind=kind, diagrams=payload, **options)
    assert (artifact["repo"], artifact["pr_number"], artifact["head_sha"]) == ("o/r", 7, head)
    path = tmp_path / "findings.json"
    write_findings_artifact(path, artifact)
    loaded = _load(path, head_sha=head)
    assert json.loads(path.read_text())["schema_version"] == loaded.schema_version == 2
    assert loaded.kind == kind and loaded.diagrams == payload
    if kind == "review":
        finding = artifact["findings"][0]
        assert (finding["fingerprint"], finding["placement"], finding["line"]) == ("f" * 64, "body", None)
        assert loaded.analysis_complete and loaded.terminal_result["analysis_state"] == "complete"
    else:
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


@pytest.mark.parametrize("raw", [None, '{"schema_version": 2', '[' * 2000 + '0' + ']' * 2000,
                                 pytest.param('[' + ' ' * (9 * MAX_ARTIFACT_BYTES), id="oversized-input")])
def test_missing_truncated_nested_and_oversized_input_is_never_clean(
    tmp_path: Path, raw: str | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "invalid.json"
    if raw is not None:
        path.write_text(raw)
    oversized = raw is not None and len(raw) > 9 * MAX_ARTIFACT_BYTES
    if oversized:
        def forbidden_parse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('Oversized raw input reached JSON parsing')
        monkeypatch.setattr(json, 'loads', forbidden_parse)
    with pytest.raises(FindingsValidationError, match="size" if oversized else None):
        _load(path)
    if oversized:
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


@pytest.mark.parametrize("mutate", [
    lambda a: a.pop("terminal_result"), lambda a: a.update(schema_version=99), lambda a: a.update(kind="diagram"),
    lambda a: a["terminal_result"]["analyzed_revision"].update(head_sha="x" * 40),
    lambda a: (a['terminal_result'].update(schema_version=2),
               a['terminal_result']['analyzed_revision'].update(source_id='d' * 64)),
    lambda a: a["terminal_result"]["stack_outcomes"].clear(),
    lambda a: a["terminal_result"].update(reason_codes=["invented"]),
])
def test_semantic_invariants_rejected_before_write(tmp_path: Path, mutate: Any) -> None:
    """A PR envelope cannot accept local source identity just because the underlying HEAD matches."""
    artifact = findings_artifact()
    mutate(artifact)
    path = tmp_path / "invalid.json"
    with pytest.raises(FindingsValidationError):
        write_findings_artifact(path, artifact)
    assert not path.exists()


@pytest.mark.parametrize("fault,prior", [("replace", False), ("replace", True), ("object", True),
    ("serializer", True), ("utf8", False), ("diagram-size", False)])
def test_failed_write_preserves_prior_bytes_and_leaves_no_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, prior: bool,
) -> None:
    path = tmp_path / "atomic.json"
    if prior:
        path.write_bytes(b"prior current result")
    artifact = findings_artifact(terminal_result=large_attributed_terminal()) if fault == 'replace' else (
        findings_artifact())
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("rename unavailable") if fault == "replace" else ValueError("cannot serialize")
    if fault in {"replace", "serializer"}:
        monkeypatch.setattr(os if fault == "replace" else json, "replace" if fault == "replace" else "dumps", fail)
    elif fault == "object":
        artifact["run_info"] = object()
    elif fault == "utf8":
        artifact["run_info"] = "é" * (MAX_ARTIFACT_BYTES // 2)
        assert len(artifact["run_info"]) < MAX_ARTIFACT_BYTES
    else:
        payload = _diagram_payload()
        payload["results"]["sequence"]["spec_final"]["messages"] = [{"label": "x" * 512} for _ in range(4000)]
        artifact = findings_artifact(kind="diagram", diagrams=payload)
    expected = 'rename unavailable' if fault == 'replace' else (
        'size' if fault in {'utf8', 'diagram-size'} else None)
    with pytest.raises(FindingsValidationError, match=expected):
        write_findings_artifact(path, artifact)
    assert path.read_bytes() == b"prior current result" if prior else not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_untrustworthy_projection_cannot_contain_findings(tmp_path: Path, valid_artifact: dict[str, Any]) -> None:
    c = review_coverage(scope_ids=("python",))
    c.record_scope("python", "complete")
    c.record_phase("merge", "failed", reasons=["malformed_artifact"])
    valid_artifact["terminal_result"] = c.finalize("failed", projection_valid=False)
    with pytest.raises(FindingsValidationError, match="projection"):
        write_findings_artifact(tmp_path / "bad-projection.json", valid_artifact)


@pytest.mark.parametrize('capacity', ['trim', 'omit', 'core-oversized'])
def test_optional_context_fits_remaining_budget_without_changing_findings(
    tmp_path: Path, valid_artifact: dict[str, Any], caplog: pytest.LogCaptureFixture, capacity: str,
) -> None:
    """Optional prose cannot block a valid canonical handoff or weaken its existing byte cap."""
    artifact = valid_artifact
    artifact['terminal_result'] = large_attributed_terminal()
    artifact['findings'][0]['body'] = 'Canonical finding. ' * 47000 if capacity == 'trim' else 'B'
    if capacity != 'trim':
        artifact['run_info'] = ''
        size = (len((json.dumps(artifact, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
                - attribution_bytes(artifact))
        artifact['run_info'] = 'x' * (MAX_ARTIFACT_BYTES - size + (capacity == 'core-oversized'))
    core = copy.deepcopy(artifact)
    identity = execution_identity(head_sha=artifact['head_sha'])
    verdict = {'attempts': [{'kind': 'host', 'command': list(identity.argv), 'passed': True,
                             'identity': identity.payload()}]}
    artifact['review_context'] = {
        'scopes': [{'scope_id': 'python', 'originating_run_id': 'run-1',
            'analyzed_revision': artifact['terminal_result']['analyzed_revision'], 'review_context': {
                'admitted_stages': 8, 'target_decisions': {'required': 8, 'reviewed': 8, 'not_reviewed': 0},
                'notes': [{'text': 'Existing reviewer rationale', 'origin': None,
                           'files': ['a' * 300] * 32, 'supporting_files': ['b' * 300] * 32} for _ in range(8)],
                'notes_omitted': 0}}], 'scopes_omitted': 0,
        'verification': {'status': 'available', 'value': {'test-verdict.json': verdict}},
        'author_reported': None,
    }
    original_context = copy.deepcopy(artifact['review_context'])
    path = tmp_path / 'fitted.json'
    if capacity == 'core-oversized':
        path.write_bytes(b'prior canonical bytes')
        with pytest.raises(FindingsValidationError, match='size'):
            write_findings_artifact(path, artifact)
        assert path.read_bytes() == b'prior canonical bytes'
        assert not list(tmp_path.glob('*.tmp'))
        path.write_text(json.dumps(core))
        with pytest.raises(FindingsValidationError, match='size'):
            _load(path)
        return
    write_findings_artifact(path, core)
    write_findings_artifact(path, artifact)
    assert MAX_ARTIFACT_BYTES < path.stat().st_size <= 9 * MAX_ARTIFACT_BYTES
    loaded = json.loads(path.read_text())
    assert path.stat().st_size - attribution_bytes(loaded) <= MAX_ARTIFACT_BYTES
    assert {key: loaded[key] for key in core} == core
    assert _load(path).analysis_complete
    assert artifact['review_context'] == original_context
    assert 'Optional review context omitted' in caplog.text
    if capacity == 'trim':
        context = loaded['review_context']
        assert context['scopes'] == [] and context['scopes_omitted'] == 1
        assert context['verification']['value']['test-verdict.json'] == verdict
    else:
        assert 'review_context' not in loaded
        assert path.stat().st_size - attribution_bytes(loaded) == MAX_ARTIFACT_BYTES


@pytest.mark.parametrize('failure', ['parser-recursion', 'escaped-surrogate'])
def test_parser_resource_failure_is_controlled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    path = tmp_path / "parse-resource.json"
    path.write_text('{}' if failure == 'parser-recursion' else json.dumps(findings_artifact(run_info='\ud800')))
    if failure == 'parser-recursion':
        def fail_parse(*args: Any, **kwargs: Any) -> Any:
            raise RecursionError("nested input")
        monkeypatch.setattr(json, "loads", fail_parse)
    with pytest.raises(FindingsValidationError,
                       match='parse failed' if failure == 'parser-recursion' else 'encoding failed'):
        _load(path)


@pytest.mark.parametrize("errored", [False, True])
async def test_real_review_export_preserves_findings_or_backend_failure(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, errored: bool,
) -> None:
    """Only provider and GitHub boundaries are doubled; Git/placement/writes run unchanged."""
    record = {"id": 1, "description": "Greeting changed without regression coverage", "severity": "medium",
              "confidence": "HIGH", "file": "main.py", "line": 1, "rationale": "", "evidence": "main.py:1"}
    records = [record]
    if not errored:
        (feature_branch_repo / "db.py").write_text("def query():\n    return 1\n")
        git(feature_branch_repo, "add", "db.py")
        commit(feature_branch_repo, "add second reviewed source")
        record = {**record, "description": "Missing check on `token_env`", "rationale": "NPE"}
        records = [record, {**record, "id": 2, "line": 2},
                   {**record, "id": 3, "severity": "low", "confidence": "MEDIUM"},
                   {**record, "id": 4, "file": "db.py", "evidence": "db.py:1"},
                   {**record, "id": 5, "rationale": "second rationale"},
                   {**record, "id": 6, "description": "`alpha` and `bravo`"},
                   {**record, "id": 7, "description": "`bravo` and `alpha`"}]
    backend: Any = ScriptedBackend(events=[TextEvent(text="Invalid API key · Fix external API key"),
        ClaudeAgentError("Claude agent run failed: Invalid API key · Fix external API key")],
        model=None) if errored else (
        EmptyReviewBackend(feature_branch_repo, forbid_merge=False, forbid_supervise=False,
                           review_by_stack={"python": records}))
    if not errored:
        backend.merge_echo_records, backend.merge_items = True, None
    head = git_ops.head_sha(feature_branch_repo)
    pr = PRInfo(7, head, git(feature_branch_repo, "rev-parse", "main"),
                "main", "feature", "o", "r", "https://example.invalid/pr/7")
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr("daydream.github_app.resolve_user_identity", lambda *_args, **_kwargs: "tester")
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", lambda *_args, **_kwargs: pr)
    output = tmp_path / "findings.json"
    config = RunConfig(target=str(feature_branch_repo), output_mode="review", pr_number=7,
                       findings_out=str(output), non_interactive=True)
    if errored:
        with pytest.raises(ClaudeAgentError, match="Invalid API key"):
            await run(config)
    else:
        assert await run(config) == 0
    loaded = load_findings_artifact(output, expected_repo="o/r", expected_pr_number=7, expected_head_sha=head)
    assert loaded.pr_number == 7 and loaded.head_sha == git_ops.head_sha(feature_branch_repo)
    if errored:
        assert loaded.terminal_result is not None
        assert loaded.terminal_result["analysis_state"] == loaded.terminal_result["pipeline_state"] == "failed"
        assert not loaded.analysis_complete
    else:
        assert loaded.findings and all(re.fullmatch(r"[0-9a-f]{64}", f.fingerprint) for f in loaded.findings)
        by_record = {(f.path, f.line, f.title, f.severity, f.confidence, f.body.rsplit("\n\n", 1)[-1]): f.fingerprint
                     for f in loaded.findings}
        assert len(by_record) == len(loaded.findings) == 7
        base = by_record[("main.py", 1, "Missing check on `token_env`", "medium", "HIGH", "NPE")]
        assert base == "d7aeacf21bddb6765801dde1c9b488cbf12a7bfabd9ee818c09861e0198484c7"
        assert by_record[("main.py", 2, "Missing check on `token_env`", "medium", "HIGH", "NPE")] == base
        assert by_record[("main.py", 1, "Missing check on `token_env`", "low", "MEDIUM", "NPE")] == base
        other_file = by_record[("db.py", 1, "Missing check on `token_env`", "medium", "HIGH", "NPE")]
        other_rationale = by_record[
            ("main.py", 1, "Missing check on `token_env`", "medium", "HIGH", "second rationale")]
        alpha_first = by_record[("main.py", 1, "`alpha` and `bravo`", "medium", "HIGH", "NPE")]
        bravo_first = by_record[("main.py", 1, "`bravo` and `alpha`", "medium", "HIGH", "NPE")]
        assert other_file != base and other_rationale != base
        assert alpha_first != bravo_first
        assert len({base, other_file, other_rationale, alpha_first, bravo_first}) == 5
