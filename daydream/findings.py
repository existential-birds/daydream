"""Findings artifact for the Phase A → Phase B review handoff.

Phase A (unprivileged analyze job, has the PR checkout) classifies review
issues into inline vs body-only placement and serializes the result as a
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
    build_findings_artifact: Classify issues and build the artifact dict.
    write_findings_artifact: Write the artifact as pretty-printed JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream import pr_review
from daydream.pr_review import ParsedIssue, PRInfo

FINDINGS_SCHEMA_VERSION = 1

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
                    "placement": {"enum": ["inline", "body"]},
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
) -> dict[str, Any]:
    """Classify issues against the PR diff and build the findings artifact.

    Runs the existing :func:`daydream.pr_review.classify` placement logic
    (anchor line resolution + hunk snapping) in the job that has the PR
    checkout, so the privileged poster never needs PR git objects.

    Args:
        target_dir: Repo root containing the PR checkout.
        pr: The target PR (declares repo / pr_number / head_sha identity).
        issues: Parsed issues, fingerprinted for cross-run dedup.
        run_info: Phase A's rendered run-info markdown, or None.

    Returns:
        The artifact dict, matching ``FINDINGS_SCHEMA``: inline findings
        carry ``placement="inline"`` with the snapped line; body-only
        findings carry ``placement="body"`` and ``line=None``.
    """
    classified = pr_review.classify(target_dir, pr, issues)
    findings = [
        _finding_dict(issue, placement="inline", line=entry["line"])
        for entry, issue in zip(classified.inline, classified.inline_issues, strict=True)
    ]
    findings.extend(_finding_dict(issue, placement="body", line=None) for issue in classified.body_only)
    return {
        "schema_version": FINDINGS_SCHEMA_VERSION,
        "repo": f"{pr.owner}/{pr.repo}",
        "pr_number": pr.number,
        "head_sha": pr.head_sha,
        "run_info": run_info,
        "findings": findings,
    }


def write_findings_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write the artifact as pretty-printed UTF-8 JSON, creating parent dirs.

    Args:
        path: Destination file path.
        artifact: Artifact dict from :func:`build_findings_artifact`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
