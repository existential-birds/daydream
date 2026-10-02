"""Strict findings handoff from unprivileged analysis to privileged posting.

Artifacts carry raw finding fields and placement, never rendered comments. The
poster validates size, schema and event identity before rendering without PR
checkout access. Dependencies flow from findings to pr_review.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import jsonschema

from daydream import git_ops, pr_review
from daydream.json_utils import atomic_write_bytes
from daydream.output_schema import strict_object
from daydream.pr_review import ParsedIssue, PRInfo
from daydream.review_result import TERMINAL_RESULT_SCHEMA, validate_terminal_result

FINDINGS_SCHEMA_VERSION = 2

MAX_ARTIFACT_BYTES = 1_048_576


def _enforce_max_artifact_bytes(size: int) -> None:
    if size > MAX_ARTIFACT_BYTES:
        raise FindingsValidationError(
            f"artifact size check failed: {size} bytes exceeds the {MAX_ARTIFACT_BYTES}-byte cap"
        )

_FINDING_PROPERTIES = {
    'fingerprint': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
    **{name: {'type': 'string'} for name in ('path', 'title', 'body')},
    'line': {'type': ['integer', 'null']}, 'placement': {'enum': ['inline', 'file', 'body']},
    **{name: {'type': ['string', 'null']} for name in ('severity', 'confidence', 'severity_before_demotion')},
    **{name: {'type': 'boolean'} for name in ('is_cross_stack', 'location_distrust', 'severity_off_vocabulary')},
}
FINDINGS_SCHEMA: dict[str, Any] = strict_object({
    'schema_version': {'const': FINDINGS_SCHEMA_VERSION}, 'repo': {'type': 'string'},
    'pr_number': {'type': 'integer'}, 'head_sha': {'type': 'string'}, 'kind': {'enum': ['review', 'diagram']},
    'run_info': {'type': ['string', 'null']},
    'review_warnings': {'type': 'array', 'items': {'type': 'string'}},
    'diagrams': {'type': ['object', 'null']},
    'findings': {'type': 'array', 'items': strict_object(_FINDING_PROPERTIES)},
    'terminal_result': TERMINAL_RESULT_SCHEMA,
})
FINDINGS_SCHEMA['required'].remove('terminal_result')
FINDINGS_SCHEMA['allOf'] = [{
    'if': {'properties': {'kind': {'const': 'review'}}},
    'then': {'required': ['terminal_result']},
    'else': {'not': {'required': ['terminal_result']}, 'properties': {'findings': {'maxItems': 0}}},
}]


def _validate_artifact(data: Any) -> None:
    """Validate the current kind-specific envelope and its coverage semantics."""
    version = data.get("schema_version") if isinstance(data, dict) else None
    if type(version) is not int or version != FINDINGS_SCHEMA_VERSION:
        raise FindingsValidationError(f"artifact failed schema validation: unsupported schema version {version!r}")
    try:
        jsonschema.validate(data, FINDINGS_SCHEMA)
        if data["kind"] == "review":
            validate_terminal_result(data["terminal_result"], expected_head_sha=data["head_sha"])
            if not data["terminal_result"]["projection_valid"] and data["findings"]:
                raise ValueError("untrustworthy projection must not contain findings")
    except (jsonschema.ValidationError, ValueError) as exc:
        message = exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
        raise FindingsValidationError(f"artifact failed schema validation: {message}") from exc


class FindingsValidationError(Exception):
    """Artifact read/write size, parse, schema or event-binding validation failed."""


@dataclass
class ArtifactFinding:
    """Validated finding with placement, identity and approval-gate provenance.

    Location and severity provenance preserve the current approval gate inputs.
    severity_before_demotion preserves whether a demoted finding was blocking.
    Non-inline placements have no line; body remains raw until posting.
    """

    fingerprint: str
    path: str
    line: int | None
    placement: str
    title: str
    body: str
    severity: str | None
    confidence: str | None
    is_cross_stack: bool
    location_distrust: bool = False
    severity_before_demotion: str | None = None
    severity_off_vocabulary: bool = False


@dataclass
class FindingsArtifact:
    """Event-bound findings and optional diagrams for the privileged poster.

    review_warnings block approval/stale resolution but permit surviving findings.
    Diagram payloads omit stored Mermaid;
    the poster validates and renders their final specs.
    """

    repo: str
    pr_number: int
    head_sha: str
    run_info: str | None
    findings: list[ArtifactFinding]
    kind: str = "review"
    diagrams: dict[str, Any] | None = None
    review_warnings: tuple[str, ...] = ()
    schema_version: int = FINDINGS_SCHEMA_VERSION
    terminal_result: dict[str, Any] | None = None

    @property
    def analysis_complete(self) -> bool:
        """Only explicit validated code review evidence establishes completeness."""
        return self.terminal_result is not None and self.terminal_result["analysis_state"] == "complete"

    @property
    def coverage_notice(self) -> tuple[str, ...]:
        """Render typed incomplete/failed coverage for review notices."""
        if self.terminal_result is None or self.analysis_complete:
            return ()
        state = self.terminal_result["analysis_state"]
        reasons = ", ".join(self.terminal_result["reason_codes"])
        return (f"Review analysis is {state}; required coverage was not completed ({reasons}).",)


def _finding_dict(issue: ParsedIssue, *, placement: str, line: int | None) -> dict[str, Any]:
    """Map one classified issue onto an artifact finding entry."""
    finding = {field.name: getattr(issue, field.name) for field in fields(ArtifactFinding)
               if field.name not in {'placement', 'line'}}
    return {**finding, 'placement': placement, 'line': line}


def build_findings_artifact(
    target_dir: Path,
    pr: PRInfo,
    issues: list[ParsedIssue],
    *,
    run_info: str | None,
    review_warnings: tuple[str, ...] = (),
    kind: str = "review",
    diagrams: dict[str, Any] | None = None,
    renderers: pr_review.ReviewRenderers | None = None,
    terminal_result: dict[str, Any] | None = None,
    snapshot_diff: str | None = None,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> dict[str, Any]:
    """Classify against the PR diff before producing the handoff artifact.

    Inline findings retain snapped lines; file/body findings have line=None.
    Diagram-only callers provide empty issues and specs without stored Mermaid.
    Classification runs where PR Git objects are available, before privileged posting.
    """
    if (kind == 'review') != (terminal_result is not None):
        raise FindingsValidationError('review requires terminal_result; diagram forbids code coverage')
    if kind == 'review' and snapshot_diff is None:
        raise FindingsValidationError('review requires the captured snapshot diff')
    placement_options = {"snapshot_diff": snapshot_diff} if snapshot_diff is not None else {}
    classified = pr_review.classify(target_dir, pr, issues, auth=auth, renderers=renderers, **placement_options)
    findings = [
        _finding_dict(issue, placement="inline", line=entry.line)
        for entry, issue in zip(classified.inline, classified.inline_issues, strict=True)
    ]
    findings.extend(_finding_dict(issue, placement="file", line=None) for issue in classified.file_level)
    findings.extend(_finding_dict(issue, placement="body", line=None) for issue in classified.body_only)
    return {
        "schema_version": FINDINGS_SCHEMA_VERSION,
        "repo": f"{pr.owner}/{pr.repo}",
        "pr_number": pr.number,
        "head_sha": pr.head_sha,
        "run_info": run_info,
        "kind": kind,
        "diagrams": diagrams,
        "findings": findings,
        "review_warnings": list(review_warnings),
        **({"terminal_result": copy.deepcopy(terminal_result)} if terminal_result is not None else {}),
    }


def write_findings_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write UTF-8 JSON with parent creation, enforcing the cap on exact output bytes."""
    try:
        _validate_artifact(artifact)
        payload = (json.dumps(artifact, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        _enforce_max_artifact_bytes(len(payload))
        atomic_write_bytes(path, payload)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise FindingsValidationError(f"artifact write failed: {path}: {exc}") from exc


def load_findings_artifact(
    path: Path,
    *,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_run_id: str | None = None,
) -> FindingsArtifact:
    """Validate untrusted output against event facts before privileged posting.

    Check size before reading, then JSON, schema and repo/PR/head identity, in that
    order. Any failure raises FindingsValidationError naming the check.
    """
    try:
        size = path.stat().st_size
        _enforce_max_artifact_bytes(size)
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FindingsValidationError(f"artifact read failed: {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise FindingsValidationError(f"artifact JSON parse failed: {exc}") from exc

    _validate_artifact(data)

    for field_name, expected in (
        ("repo", expected_repo),
        ("pr_number", expected_pr_number),
        ("head_sha", expected_head_sha),
    ):
        declared = data[field_name]
        if declared != expected:
            raise FindingsValidationError(
                f"artifact {field_name} {declared!r} does not match event-derived {field_name} {expected!r}"
            )

    if expected_run_id is not None:
        declared = data.get("terminal_result", {}).get("run_id")
        if declared != expected_run_id:
            raise FindingsValidationError(
                f"artifact run_id {declared!r} does not match expected run_id {expected_run_id!r}"
            )

    return FindingsArtifact(**{**data, 'findings': [ArtifactFinding(**f) for f in data['findings']],
                               'review_warnings': tuple(data['review_warnings'])})
