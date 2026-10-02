"""Tests for the findings artifact (build/write/load) in `daydream/findings.py`."""
import json
import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

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


def test_build_artifact_declares_target_envelope(tmp_path: Path) -> None:
    """build_findings_artifact stamps the PR identity envelope and passes each
    issue's fingerprint through. A cross-stack issue routes to body placement
    without touching the diff, so no git/collaborator mocking is needed; inline
    snapping is exercised by the real-path test below.
    """
    pr = PRInfo(number=7, head_sha="h" * 40, base_sha="b" * 40, base_ref="main", head_ref="feature",
                owner="o", repo="r", url="u")
    issues = [ParsedIssue(path="a.py", line=None, title="T", body="B", severity="high",
                          confidence="HIGH", fingerprint="f" * 64, is_cross_stack=True)]
    artifact = build_findings_artifact(tmp_path, pr, issues, run_info=None)
    assert (artifact["repo"], artifact["pr_number"], artifact["head_sha"]) == ("o/r", 7, "h" * 40)
    f = artifact["findings"][0]
    assert (f["fingerprint"], f["placement"], f["line"]) == ("f" * 64, "body", None)

def test_write_artifact_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "findings.json"
    write_findings_artifact(path, {"schema_version": 1, "repo": "o/r",
                                   "pr_number": 7, "head_sha": "h" * 40, "findings": []})
    assert json.loads(path.read_text())["schema_version"] == 1


# --- Load + validation (confused-deputy gate) ---------------------------------


@pytest.fixture
def valid_artifact() -> dict[str, Any]:
    """Artifact dict with one inline finding."""
    return {"schema_version": 1, "repo": "o/r", "pr_number": 7, "head_sha": "h" * 40,
        "run_info": None,
        "findings": [{
                "fingerprint": "f" * 64, "path": "a.py", "line": 12, "placement": "inline", "title": "T", "body": "B",
                "severity": "high", "confidence": "HIGH", "is_cross_stack": False,
            }
        ],
    }

@pytest.mark.parametrize("mutate, match", [
    (lambda a: a.pop("head_sha"), "schema"), (lambda a: a.update(head_sha="e" * 40), "does not match"),
    (lambda a: a.update(pr_number=8), "does not match"), (lambda a: a.update(unexpected=1), "schema"),
    (lambda a: a["findings"][0].update(fingerprint="nope"), "schema"),
])
def test_load_rejects_invalid_artifacts(tmp_path: Path, valid_artifact: dict[str, Any], mutate: Any, match: Any,
) -> None:
    mutate(valid_artifact)
    p = tmp_path / "f.json"
    p.write_text(json.dumps(valid_artifact))
    with pytest.raises(FindingsValidationError, match=match):
        load_findings_artifact(p, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40)

def test_load_rejects_oversized_artifact(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text("[" + " " * MAX_ARTIFACT_BYTES)
    with pytest.raises(FindingsValidationError, match="size"):
        load_findings_artifact(p, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40)


# --- Real-path Phase A emission (--findings-out via runner.run) --------------


@contextmanager
def _review_run_env(feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, out: Any, backend: Any, pr: Any,
) -> Iterator[Any]:
    """Shared `--review --findings-out` real-path setup: env, config, patch stack.

    Clears the GitHub App env, builds the review-mode RunConfig, and patches the
    backend seam plus the GitHub lookups. Each test supplies only its backend,
    PRInfo, and assertions.
    """
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    config = RunConfig(target=str(feature_branch_repo), output_mode="review",
                       pr_number=7, findings_out=str(out), non_interactive=True)
    with patch("daydream.runner.create_backend", return_value=backend), \
         patch("daydream.github_app.resolve_user_identity", return_value="tester"), \
         patch("daydream.pr_review.find_pr_by_number", return_value=pr):
        yield config

async def test_review_mode_writes_findings_artifact(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exercise the event-bound findings handoff through runner.run and real Git.

    Only backend and GitHub lookups are doubled; classification, fingerprinting,
    and artifact writing run unchanged.
    """
    out = tmp_path / "findings.json"
    issue = {"id": 1, "title": "Greeting changed without tests",
        "description": "`hello` now returns a different greeting with no test coverage",
        "recommendation": "Add a regression test for the new greeting", "severity": "medium", "confidence": "HIGH",
        "files": ["main.py"], "file": "main.py", "line": 1, "rationale": "", "evidence": "main.py:1",
    }
    record = {key: issue[key] for key in (
        "id", "description", "file", "line", "severity", "confidence", "rationale", "evidence")}
    backend = EmptyReviewBackend(feature_branch_repo, forbid_merge=False, forbid_supervise=False,
                                 review_by_stack={"python": [record]})
    backend.merge_echo_records = True
    backend.merge_items = None
    head = git_ops.head_sha(feature_branch_repo)
    base = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", "rev-parse", "main"],  # noqa: S607 - git is a trusted command
        cwd=feature_branch_repo, capture_output=True, text=True, check=True,
    ).stdout.strip()
    pr = PRInfo(number=7, head_sha=head, base_sha=base, base_ref="main", head_ref="feature",
                owner="o", repo="r", url="https://example.invalid/pr/7")

    with _review_run_env(feature_branch_repo, monkeypatch, out, backend, pr) as config:
        assert await run(config) == 0

    data = json.loads(out.read_text())
    assert data["pr_number"] == 7
    assert data["head_sha"] == git_ops.head_sha(feature_branch_repo)
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["fingerprint"]) for f in data["findings"])
    assert data["findings"], "scripted issue must survive to the artifact"

async def test_review_mode_errored_agent_exports_failed_terminal_artifact(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An errored backend must abort rather than publish an empty, apparently clean artifact.

    Drive runner.run with real Git; only the Claude error boundary and GitHub lookups
    are doubled. This protects privileged posting from accepting failed analysis.
    """
    out = tmp_path / "findings.json"

    backend = ScriptedBackend(events=[TextEvent(text="Invalid API key · Fix external API key"),
            ClaudeAgentError("Claude agent run failed: Invalid API key · Fix external API key"),
        ], model=None,
    )

    head = git_ops.head_sha(feature_branch_repo)
    pr = PRInfo(number=7, head_sha=head, base_sha=head, base_ref="main", head_ref="feature",
                owner="o", repo="r", url="https://example.invalid/pr/7")

    with _review_run_env(feature_branch_repo, monkeypatch, out, backend, pr) as config:
        with pytest.raises(ClaudeAgentError, match="Invalid API key"):
            await run(config)

    loaded = load_findings_artifact(out, expected_repo="o/r", expected_pr_number=7, expected_head_sha=head)
    assert loaded.terminal_result is not None
    assert loaded.terminal_result["analysis_state"] == "failed"
    assert loaded.terminal_result["pipeline_state"] == "failed"
    assert not loaded.analysis_complete


# --- Grounded diagrams (issue #1113) ----------------------------------------


def _diagram_payload() -> dict[str, Any]:
    """A minimal but shape-real diagram payload."""
    return {"eligibility": {"sequence": {"eligible": True, "rule": "forced", "reason": "r"}},
        "results": {"sequence": {
                "status": "rendered", "reason": None, "spec_final": {"participants": [], "messages": [], "blocks": []},
                "omit_reasons": [],
            }, "flowchart": None,
        },
    }

def test_diagram_artifact_round_trips_kind_and_payload(tmp_path: Path) -> None:
    pr = PRInfo(number=7, head_sha="h" * 40, base_sha="b" * 40, base_ref="main", head_ref="feature",
                owner="o", repo="r", url="u")
    payload = _diagram_payload()
    artifact = build_findings_artifact(tmp_path, pr, [], run_info=None, kind="diagram", diagrams=payload)
    assert artifact["kind"] == "diagram"
    assert artifact["diagrams"] == payload

    path = tmp_path / "diagram-findings.json"
    write_findings_artifact(path, artifact)
    loaded = load_findings_artifact(path, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40)
    assert loaded.kind == "diagram"
    assert loaded.diagrams == payload
    assert loaded.findings == []

def test_review_artifact_defaults_kind_and_diagrams(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
                "schema_version": 1, "repo": "o/r", "pr_number": 7, "head_sha": "h" * 40,
                "findings": [],
            }
        ), encoding="utf-8",
    )
    loaded = load_findings_artifact(path, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40)
    assert loaded.kind == "review"
    assert loaded.diagrams is None

def test_unknown_artifact_kind_is_rejected_by_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad-kind.json"
    path.write_text(json.dumps({
                "schema_version": 1, "repo": "o/r", "pr_number": 7, "head_sha": "h" * 40,
                "kind": "sabotage", "findings": [],
            }
        ), encoding="utf-8",
    )
    with pytest.raises(FindingsValidationError, match="schema validation"):
        load_findings_artifact(path, expected_repo="o/r", expected_pr_number=7, expected_head_sha="h" * 40)

def test_write_rejects_an_oversized_artifact(tmp_path: Path) -> None:
    """The size cap fails in the job that produced the artifact, not one job later."""
    pr = PRInfo(number=7, head_sha="h" * 40, base_sha="b" * 40, base_ref="main", head_ref="feature",
                owner="o", repo="r", url="u")
    payload = _diagram_payload()
    payload["results"]["sequence"]["spec_final"]["messages"] = [{"label": "x" * 512} for _ in range(4000)]
    artifact = build_findings_artifact(tmp_path, pr, [], run_info=None, kind="diagram", diagrams=payload)
    path = tmp_path / "huge.json"
    with pytest.raises(FindingsValidationError, match="size check failed"):
        write_findings_artifact(path, artifact)
    assert not path.exists(), "an over-cap artifact must not be left on disk"


def _terminal_result(state: str = 'complete') -> dict[str, Any]:
    from daydream.review_result import AnalyzedRevision, PlannedScope, ReviewCoverage
    coverage = ReviewCoverage('run-1', AnalyzedRevision('h' * 40, 'b' * 40, 'diff-1'),
                              [PlannedScope('python', 'python', ('a.py',))], ['merge'])
    coverage.record_scope('python', state, reasons=[] if state == 'complete' else ['backend_failure'])
    coverage.record_phase('merge', 'complete', noop=True)
    return coverage.finalize('completed')


def _v2_artifact() -> dict[str, Any]:
    return {'schema_version': 2, 'repo': 'o/r', 'pr_number': 7, 'head_sha': 'h' * 40,
            'findings': [], 'terminal_result': _terminal_result()}


def test_v2_terminal_contract_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / 'v2.json'
    write_findings_artifact(path, _v2_artifact())
    loaded = load_findings_artifact(path, expected_repo='o/r', expected_pr_number=7, expected_head_sha='h' * 40)
    assert loaded.terminal_result is not None
    assert loaded.terminal_result['analysis_state'] == 'complete'
    assert loaded.analysis_complete


@pytest.mark.parametrize('mutate', [
    lambda a: a.pop('terminal_result'),
    lambda a: a.update(schema_version=99),
    lambda a: a.update(kind='diagram'),
    lambda a: a['terminal_result']['analyzed_revision'].update(head_sha='x' * 40),
    lambda a: a['terminal_result']['stack_outcomes'].clear(),
    lambda a: a['terminal_result'].update(reason_codes=['invented']),
])
def test_v2_semantic_invariants_rejected_before_write(tmp_path: Path, mutate: Any) -> None:
    artifact = _v2_artifact()
    mutate(artifact)
    with pytest.raises(FindingsValidationError):
        write_findings_artifact(tmp_path / 'invalid.json', artifact)
    assert not (tmp_path / 'invalid.json').exists()


@pytest.mark.parametrize('prior', [None, b'prior complete result'])
def test_atomic_replacement_failure_preserves_prior_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           prior: bytes | None) -> None:
    path = tmp_path / 'atomic.json'
    if prior is not None:
        path.write_bytes(prior)
    def fail_replace(*args: Any) -> None:
        raise OSError('rename unavailable')
    monkeypatch.setattr(os, 'replace', fail_replace)
    with pytest.raises(FindingsValidationError, match='write'):
        write_findings_artifact(path, _v2_artifact())
    assert path.read_bytes() == prior if prior is not None else not path.exists()
    assert not list(tmp_path.glob('*.tmp'))


def test_serialization_failure_is_controlled_and_preserves_prior(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    path.write_bytes(b'prior')
    artifact = _v2_artifact()
    artifact['run_info'] = object()
    with pytest.raises(FindingsValidationError):
        write_findings_artifact(path, artifact)
    assert path.read_bytes() == b'prior'


@pytest.mark.parametrize('raw', [None, '{"schema_version": 2'])
def test_absent_or_truncated_result_is_never_clean(tmp_path: Path, raw: str | None) -> None:
    path = tmp_path / 'missing.json'
    if raw is not None:
        path.write_text(raw)
    with pytest.raises(FindingsValidationError):
        load_findings_artifact(path, expected_repo='o/r', expected_pr_number=7, expected_head_sha='h' * 40)


def test_reused_path_requires_separately_known_run_id(tmp_path: Path) -> None:
    path = tmp_path / 'reused.json'
    write_findings_artifact(path, _v2_artifact())
    with pytest.raises(FindingsValidationError, match='run_id'):
        load_findings_artifact(path, expected_repo='o/r', expected_pr_number=7,
                               expected_head_sha='h' * 40, expected_run_id='current-run')
    loaded = load_findings_artifact(path, expected_repo='o/r', expected_pr_number=7,
                                    expected_head_sha='h' * 40, expected_run_id='run-1')
    assert loaded.analysis_complete
    with pytest.raises(FindingsValidationError):
        load_findings_artifact(tmp_path / 'fresh-current-run.json', expected_repo='o/r',
                               expected_pr_number=7, expected_head_sha='h' * 40)


def test_utf8_byte_cap_uses_exact_serialized_bytes(tmp_path: Path) -> None:
    artifact = _v2_artifact()
    artifact['run_info'] = 'é' * (MAX_ARTIFACT_BYTES // 2)
    assert len(artifact['run_info']) < MAX_ARTIFACT_BYTES
    with pytest.raises(FindingsValidationError, match='size'):
        write_findings_artifact(tmp_path / 'utf8.json', artifact)
    assert not (tmp_path / 'utf8.json').exists()


def test_serializer_exception_leaves_current_artifact_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'serialized.json'
    path.write_bytes(b'old current result')
    def fail_serialization(*args: Any, **kwargs: Any) -> str:
        raise ValueError('cannot serialize')
    monkeypatch.setattr(json, 'dumps', fail_serialization)
    with pytest.raises(FindingsValidationError, match='write failed'):
        write_findings_artifact(path, _v2_artifact())
    assert path.read_bytes() == b'old current result'


def test_excessively_nested_json_is_a_controlled_artifact_error(tmp_path: Path) -> None:
    path = tmp_path / 'nested.json'
    path.write_text('[' * 2000 + '0' + ']' * 2000)
    with pytest.raises(FindingsValidationError):
        load_findings_artifact(path, expected_repo='o/r', expected_pr_number=7, expected_head_sha='h' * 40)


def test_untrustworthy_projection_cannot_contain_findings(tmp_path: Path, valid_artifact: dict[str, Any]) -> None:
    artifact = _v2_artifact()
    artifact['findings'] = valid_artifact['findings']
    from daydream.review_result import AnalyzedRevision, PlannedScope, ReviewCoverage
    c = ReviewCoverage('bad-projection', AnalyzedRevision('h' * 40, 'b' * 40, 'diff-1'),
                       [PlannedScope('python', 'python')], ['merge'])
    c.record_scope('python', 'complete')
    c.record_phase('merge', 'failed', reasons=['malformed_artifact'])
    artifact['terminal_result'] = c.finalize('failed', projection_valid=False)
    with pytest.raises(FindingsValidationError, match='projection'):
        write_findings_artifact(tmp_path / 'bad-projection.json', artifact)


def test_parser_resource_failure_is_a_controlled_artifact_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'parse-resource.json'
    path.write_text('{}')
    def fail_parse(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError('nested input')
    monkeypatch.setattr(json, 'loads', fail_parse)
    with pytest.raises(FindingsValidationError, match='parse failed'):
        load_findings_artifact(path, expected_repo='o/r', expected_pr_number=7, expected_head_sha='h' * 40)
