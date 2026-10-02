"""Strict findings handoff from unprivileged analysis to privileged posting.

Artifacts carry raw finding fields and placement, never rendered comments. The
poster validates size, schema and event identity before rendering without PR
checkout access. Dependencies flow from findings to pr_review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from daydream import git_ops, pr_review
from daydream.pr_review import ParsedIssue, PRInfo

FINDINGS_SCHEMA_VERSION = 1

MAX_ARTIFACT_BYTES = 1_048_576


def _enforce_max_artifact_bytes(size: int) -> None:
    if size > MAX_ARTIFACT_BYTES:
        raise FindingsValidationError(
            f"artifact size check failed: {size} bytes exceeds the {MAX_ARTIFACT_BYTES}-byte cap"
        )

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "repo", "pr_number", "head_sha", "findings"],
    "properties": {
        "schema_version": {"const": FINDINGS_SCHEMA_VERSION},
        "repo": {"type": "string"},
        "pr_number": {"type": "integer"},
        "head_sha": {"type": "string"},
        "run_info": {"type": ["string", "null"]},
        "review_warnings": {"type": "array", "items": {"type": "string"}},
        # Optional for older artifacts; absent kind means review.
        "kind": {"enum": ["review", "diagram"]},
        # Envelope permits schema evolution; the poster strictly validates each
        # model-authored spec_final before rendering.
        "diagrams": {"type": ["object", "null"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "fingerprint",
                    "path",
                    "line",
                    "placement",
                    "title",
                    "body",
                    "severity",
                    "confidence",
                    "is_cross_stack",
                ],
                "properties": {
                    "fingerprint": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "path": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                    "placement": {"enum": ["inline", "file", "body"]},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "severity": {"type": ["string", "null"]},
                    "confidence": {"type": ["string", "null"]},
                    "is_cross_stack": {"type": "boolean"},
                    # Optional legacy defaults preserve approval-gate provenance.
                    "location_distrust": {"type": "boolean"},
                    "severity_off_vocabulary": {"type": "boolean"},
                    "severity_before_demotion": {"type": ["string", "null"]},
                },
            },
        },
    },
}


class FindingsValidationError(Exception):
    """Artifact read/write size, parse, schema or event-binding validation failed."""


@dataclass
class ArtifactFinding:
    """Validated finding with placement, identity and approval-gate provenance.

    location_distrust and severity_off_vocabulary default false for older artifacts.
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
    Older artifacts default to kind=review. Diagram payloads omit stored Mermaid;
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


def _finding_dict(issue: ParsedIssue, *, placement: str, line: int | None) -> dict[str, Any]:
    """Map one classified issue onto an artifact finding entry."""
    return {
        "fingerprint": issue.fingerprint,
        "path": issue.path,
        "line": line,
        "placement": placement,
        "title": issue.title,
        "body": issue.body,
        "severity": issue.severity,
        "confidence": issue.confidence,
        "is_cross_stack": issue.is_cross_stack,
        "location_distrust": issue.location_distrust,
        "severity_before_demotion": issue.severity_before_demotion,
        "severity_off_vocabulary": issue.severity_off_vocabulary,
    }


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
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> dict[str, Any]:
    """Classify against the PR diff before producing the handoff artifact.

    Inline findings retain snapped lines; file/body findings have line=None.
    Diagram-only callers provide empty issues and specs without stored Mermaid.
    Classification runs where PR Git objects are available, before privileged posting.
    """
    classified = pr_review.classify(target_dir, pr, issues, auth=auth, renderers=renderers)
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
        **({"review_warnings": list(review_warnings)} if review_warnings else {}),
    }


def write_findings_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write UTF-8 JSON with parent creation, enforcing the cap on exact output bytes."""
    text = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    size = len(text.encode("utf-8"))
    _enforce_max_artifact_bytes(size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_findings_artifact(
    path: Path,
    *,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_sha: str,
) -> FindingsArtifact:
    """Validate untrusted output against event facts before privileged posting.

    Check size before reading, then JSON, schema and repo/PR/head identity, in that
    order. Any failure raises FindingsValidationError naming the check.
    """
    try:
        size = path.stat().st_size
        _enforce_max_artifact_bytes(size)
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FindingsValidationError(f"artifact read failed: {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FindingsValidationError(f"artifact JSON parse failed: {exc}") from exc

    try:
        jsonschema.validate(data, FINDINGS_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise FindingsValidationError(f"artifact failed schema validation: {exc.message}") from exc

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

    return FindingsArtifact(
        repo=data["repo"],
        pr_number=data["pr_number"],
        head_sha=data["head_sha"],
        run_info=data.get("run_info"),
        findings=[ArtifactFinding(**f) for f in data["findings"]],
        kind=data.get("kind") or "review",
        diagrams=data.get("diagrams"),
        review_warnings=tuple(data.get("review_warnings", [])),
    )
