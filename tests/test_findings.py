"""Strict current findings envelopes, atomic writes, and production Phase A exports."""
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
from tests.harness.review_result import findings_artifact, review_coverage, terminal_result


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


@pytest.mark.parametrize("mutate,match", [
    (lambda a: a.pop("head_sha"), "schema"),
    (lambda a: (a.update(head_sha="e" * 40),
                a["terminal_result"]["analyzed_revision"].update(head_sha="e" * 40)), "does not match"),
    (lambda a: a.update(pr_number=8), "does not match"), (lambda a: a.update(unexpected=1), "schema"),
    (lambda a: a["findings"][0].update(fingerprint="nope"), "schema"),
    (lambda a: a.pop("kind"), "schema"), (lambda a: a.update(kind="sabotage"), "schema"),
    (lambda a: a.update(schema_version=1), "version|schema"),
])
def test_load_rejects_invalid_artifacts(
    tmp_path: Path, valid_artifact: dict[str, Any], mutate: Any, match: str,
) -> None:
    mutate(valid_artifact)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(valid_artifact))
    with pytest.raises(FindingsValidationError, match=match):
        _load(path)


def test_review_defaults_and_known_run_freshness(tmp_path: Path) -> None:
    path = tmp_path / "reused.json"
    artifact = findings_artifact()
    write_findings_artifact(path, artifact)
    loaded = _load(path, expected_run_id="run-1")
    assert loaded.kind == "review" and loaded.diagrams is None and loaded.analysis_complete
    with pytest.raises(FindingsValidationError, match="run_id"):
        _load(path, expected_run_id="current-run")
    with pytest.raises(FindingsValidationError):
        _load(tmp_path / "fresh-current-run.json")


@pytest.mark.parametrize("raw", [None, '{"schema_version": 2', '[' * 2000 + '0' + ']' * 2000,
                                 '[' + ' ' * MAX_ARTIFACT_BYTES])
def test_missing_truncated_nested_and_oversized_input_is_never_clean(tmp_path: Path, raw: str | None) -> None:
    path = tmp_path / "invalid.json"
    if raw is not None:
        path.write_text(raw)
    with pytest.raises(FindingsValidationError, match="size" if raw and len(raw) > MAX_ARTIFACT_BYTES else None):
        _load(path)


@pytest.mark.parametrize("mutate", [
    lambda a: a.pop("terminal_result"), lambda a: a.update(schema_version=99), lambda a: a.update(kind="diagram"),
    lambda a: a["terminal_result"]["analyzed_revision"].update(head_sha="x" * 40),
    lambda a: a["terminal_result"]["stack_outcomes"].clear(),
    lambda a: a["terminal_result"].update(reason_codes=["invented"]),
])
def test_semantic_invariants_rejected_before_write(tmp_path: Path, mutate: Any) -> None:
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
    artifact = findings_artifact()
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
    with pytest.raises(FindingsValidationError, match="size" if fault in {"utf8", "diagram-size"} else None):
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


def test_parser_resource_failure_is_controlled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "parse-resource.json"
    path.write_text("{}")
    def fail_parse(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError("nested input")
    monkeypatch.setattr(json, "loads", fail_parse)
    with pytest.raises(FindingsValidationError, match="parse failed"):
        _load(path)


@pytest.mark.parametrize("errored", [False, True])
async def test_real_review_export_preserves_findings_or_backend_failure(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, errored: bool,
) -> None:
    """Only provider and GitHub boundaries are doubled; Git/placement/writes run unchanged."""
    record = {"id": 1, "description": "Greeting changed without regression coverage", "severity": "medium",
              "confidence": "HIGH", "file": "main.py", "line": 1, "rationale": "", "evidence": "main.py:1"}
    backend: Any = ScriptedBackend(events=[TextEvent(text="Invalid API key · Fix external API key"),
        ClaudeAgentError("Claude agent run failed: Invalid API key · Fix external API key")],
        model=None) if errored else (
        EmptyReviewBackend(feature_branch_repo, forbid_merge=False, forbid_supervise=False,
                           review_by_stack={"python": [record]}))
    if not errored:
        backend.merge_echo_records, backend.merge_items = True, None
    head = git_ops.head_sha(feature_branch_repo)
    pr = PRInfo(7, head, head, "main", "feature", "o", "r", "https://example.invalid/pr/7")
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
