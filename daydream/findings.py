"""Findings artifact for the Phase A → Phase B review handoff.

Phase A (unprivileged analyze job, has the PR checkout) classifies review
issues into inline / file-level / body-only placement and serializes it as a
strict-schema JSON artifact. Phase B (privileged poster, never touches PR
code) consumes the artifact and renders/posts from artifact data only.

The artifact carries raw issue fields, never rendered comment bodies —
rendering stays in the poster (`daydream/pr_review.py`).

Imports are strictly one-way: ``findings`` → ``pr_review``, never the
reverse — no cycle.

Exports:
    FINDINGS_SCHEMA_VERSION: Current artifact schema version (1).
    FINDINGS_SCHEMA: Strict JSON Schema for the artifact
        (``additionalProperties: False`` at every level).
    MAX_ARTIFACT_BYTES: Size cap enforced before the artifact is read.
    FindingsValidationError: Raised when an artifact fails any load check.
    ArtifactFinding: Typed view of one finding entry.
    FindingsArtifact: Typed view of a validated artifact.
    build_findings_artifact: Classify issues and build the artifact dict.
    write_findings_artifact: Write the artifact as pretty-printed JSON.
    load_findings_artifact: Load + validate an artifact against event facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from daydream import pr_review
from daydream.pr_review import ParsedIssue, PRInfo

FINDINGS_SCHEMA_VERSION = 1

MAX_ARTIFACT_BYTES = 1_048_576

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
        "provenance": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "target_kind",
                "candidate_sha",
                "candidate_tree_digest",
                "base_sha",
                "full_diff_digest",
            ],
            "properties": {
                "target_kind": {"enum": ["pr_head", "merge_group"]},
                "candidate_sha": {"type": "string", "pattern": "^[0-9a-fA-F]{40}$"},
                "candidate_tree_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{40}$"},
                "base_sha": {"type": "string", "pattern": "^[0-9a-fA-F]{40}$"},
                "full_diff_digest": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
            },
        },
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
                },
            },
        },
    },
}


class FindingsValidationError(Exception):
    """An artifact failed a load-time check (size, parse, schema, or event match)."""


@dataclass
class ArtifactFinding:
    """One validated finding entry from the artifact.

    Attributes:
        fingerprint: 64-hex cross-run dedup identity.
        path: Repo-relative file path the finding targets.
        line: Snapped inline line, or None for file-level / body-only findings.
        placement: "inline", "file" (file-level comment), or "body".
        title: Finding title.
        body: Finding body (raw, unrendered).
        severity: Severity label, or None.
        confidence: Confidence label, or None.
        is_cross_stack: Whether the finding came from the cross-stack merge.
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


@dataclass
class FindingsArtifact:
    """A validated findings artifact, typed so downstream code never touches raw dicts.

    Attributes:
        repo: Declared "owner/repo" slug.
        pr_number: Declared target PR number.
        head_sha: Declared PR head SHA the findings were computed against.
        run_info: Phase A's rendered run-info markdown, or None.
        provenance: Optional exact-review-target provenance block
            (``REVIEW_TARGET_V1`` fields: ``target_kind``, ``candidate_sha``,
            ``candidate_tree_digest``, ``base_sha``, ``full_diff_digest``), or
            None. Additive: pre-provenance artifacts load with None.
        findings: Validated finding entries.
    """

    repo: str
    pr_number: int
    head_sha: str
    run_info: str | None
    findings: list[ArtifactFinding]
    provenance: dict[str, Any] | None = None


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
    }


def build_findings_artifact(
    target_dir: Path,
    pr: PRInfo,
    issues: list[ParsedIssue],
    *,
    run_info: str | None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify issues against the PR diff and build the findings artifact.

    Runs the existing :func:`daydream.pr_review.classify` placement logic
    (anchor line resolution + hunk snapping) in the job that has the PR
    checkout, so the privileged poster never needs PR git objects.

    Args:
        issues: Parsed issues, fingerprinted for cross-run dedup.
        provenance: Optional exact-review-target provenance block
            (``REVIEW_TARGET_V1`` fields), recorded verbatim so Phase B can
            verify it reviewed the exact same target. Additive: when omitted,
            the artifact is byte-identical to a pre-provenance build.

    Returns:
        The artifact dict, matching ``FINDINGS_SCHEMA``: inline findings
        carry ``placement="inline"`` with the snapped line; findings with no
        line home but whose file is in the PR diff carry ``placement="file"``;
        the remainder carry ``placement="body"``. Both non-inline placements
        have ``line=None``.
    """
    classified = pr_review.classify(target_dir, pr, issues)
    findings = [
        _finding_dict(issue, placement="inline", line=entry["line"])
        for entry, issue in zip(classified.inline, classified.inline_issues, strict=True)
    ]
    findings.extend(_finding_dict(issue, placement="file", line=None) for issue in classified.file_level)
    findings.extend(_finding_dict(issue, placement="body", line=None) for issue in classified.body_only)
    artifact: dict[str, Any] = {
        "schema_version": FINDINGS_SCHEMA_VERSION,
        "repo": f"{pr.owner}/{pr.repo}",
        "pr_number": pr.number,
        "head_sha": pr.head_sha,
        "run_info": run_info,
        "findings": findings,
    }
    if provenance is not None:
        artifact["provenance"] = provenance
    return artifact


def write_findings_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write the artifact as pretty-printed UTF-8 JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_findings_artifact(
    path: Path,
    *,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_provenance: dict[str, Any] | None = None,
) -> FindingsArtifact:
    """Load an artifact and validate it against event-derived facts.

    This is the confused-deputy gate for the privileged poster: the artifact
    is untrusted Phase A output, so every check runs before its content is
    acted on. Checks run in order: file size (stat, before reading), JSON
    parse, strict schema validation, then equality of the declared
    ``repo``/``pr_number``/``head_sha`` against the expected (event-derived)
    values. Artifact content is never executed or interpolated.

    Args:
        expected_provenance: When provided, the artifact MUST declare a
            ``provenance`` block whose exact-SHA/digest fields
            (``candidate_sha``, ``candidate_tree_digest``, ``base_sha``,
            ``full_diff_digest``) match the event-derived target exactly. A
            missing block or any mismatch is a hard validation failure.

    Raises:
        FindingsValidationError: On any failed check, naming the check.
    """
    try:
        size = path.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            raise FindingsValidationError(
                f"artifact size check failed: {size} bytes exceeds the {MAX_ARTIFACT_BYTES}-byte cap"
            )
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

    provenance = data.get("provenance")
    if expected_provenance is not None:
        if provenance is None:
            raise FindingsValidationError(
                "artifact provenance is missing but the event-derived target requires it"
            )
        for field_name in ("target_kind", "candidate_sha", "candidate_tree_digest", "base_sha", "full_diff_digest"):
            if provenance.get(field_name) != expected_provenance.get(field_name):
                raise FindingsValidationError(
                    f"artifact provenance {field_name} {provenance.get(field_name)!r} does not match "
                    f"event-derived {field_name} {expected_provenance.get(field_name)!r}"
                )

    return FindingsArtifact(
        repo=data["repo"],
        pr_number=data["pr_number"],
        head_sha=data["head_sha"],
        run_info=data.get("run_info"),
        findings=[ArtifactFinding(**f) for f in data["findings"]],
        provenance=provenance,
    )
