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
        # Optional (issue #1113). Absent means "review" -- the only kind that
        # existed before grounded diagrams. Both keys are OPTIONAL in
        # ``required`` but MANDATORY here: ``additionalProperties: false``
        # above would otherwise reject every artifact that carries them.
        "kind": {"enum": ["review", "diagram"]},
        # Deliberately permissive. The payload is ``diagram.json`` minus the
        # rendered mermaid, whose shape is owned by four other modules'
        # ``to_dict`` methods; declaring it here with
        # ``additionalProperties: false`` would break on any field they add.
        # The real check on its model-derived content is the per-kind
        # ``spec_final`` validation in ``pr_review.validate_diagram_payload``,
        # which runs before the privileged poster renders anything.
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
                    # Optional (issue #972 R2): findings written by a Phase A
                    # that ran location validation carry the demotion mark so
                    # the poster's approval gate stays demotion-aware. Absent
                    # on older artifacts — treated as False on load.
                    "location_distrust": {"type": "boolean"},
                    # Optional (issue #972): present-but-off-canonical severity
                    # strings fold into ``None`` at the boundary but must still
                    # block the poster's approval gate; the raw signal rides
                    # through here. Absent on older artifacts -> False on load.
                    "severity_off_vocabulary": {"type": "boolean"},
                    # Optional: the original severity before a location-
                    # validation demotion, so the poster's approval gate only
                    # re-blocks a demoted finding when that original severity
                    # was itself blocking. Absent on older artifacts -> None on
                    # load.
                    "severity_before_demotion": {"type": ["string", "null"]},
                },
            },
        },
    },
}


class FindingsValidationError(Exception):
    """An artifact failed a validation check.

    Raised on load (size, parse, schema, or event match) and on write (size --
    issue #1113), so the size cap fails in the job that produced the artifact
    rather than one job later in the privileged poster.
    """


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
        location_distrust: True when location validation demoted the finding
            (citation beyond tolerance); absent on older artifacts -> False.
        severity_before_demotion: Original severity before a location-
            validation demotion, if any; the poster's approval gate checks it
            so only a demotion from a blocking severity stays blocking. Absent
            on older artifacts -> None.
        severity_off_vocabulary: True when the finding carried a present
            severity string outside the canonical vocabulary (e.g.
            "critical") that was folded into ``None`` at the boundary. The
            poster's approval gate must still block on it, so the raw signal
            rides through the artifact; absent on older artifacts -> False.
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
    """A validated findings artifact, typed so downstream code never touches raw dicts.

    Attributes:
        repo: Declared "owner/repo" slug.
        pr_number: Declared target PR number.
        head_sha: Declared PR head SHA the findings were computed against.
        run_info: Phase A's rendered run-info markdown, or None.
        findings: Validated finding entries.
        kind: ``"review"`` (findings to post as a PR review) or ``"diagram"``
            (issue #1113: a grounded-diagram payload to post as a standalone
            issue comment). Defaults to ``"review"`` so a pre-#1113 artifact
            loads unchanged.
        diagrams: The ``diagram.json`` payload without the rendered mermaid,
            or None. Only meaningful when ``kind == "diagram"``.
    """

    repo: str
    pr_number: int
    head_sha: str
    run_info: str | None
    findings: list[ArtifactFinding]
    kind: str = "review"
    diagrams: dict[str, Any] | None = None


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
    kind: str = "review",
    diagrams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify issues against the PR diff and build the findings artifact.

    Runs the existing :func:`daydream.pr_review.classify` placement logic
    (anchor line resolution + hunk snapping) in the job that has the PR
    checkout, so the privileged poster never needs PR git objects.

    Args:
        issues: Parsed issues, fingerprinted for cross-run dedup.
        run_info: Phase A's rendered run-info markdown, or None.
        kind: Artifact kind (issue #1113). ``"diagram"`` artifacts carry an
            empty ``findings`` list and a ``diagrams`` payload instead.
        diagrams: The ``diagram.json`` payload with every ``mermaid`` string
            removed (the poster re-renders from the specs), or None.

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
    return {
        "schema_version": FINDINGS_SCHEMA_VERSION,
        "repo": f"{pr.owner}/{pr.repo}",
        "pr_number": pr.number,
        "head_sha": pr.head_sha,
        "run_info": run_info,
        "kind": kind,
        "diagrams": diagrams,
        "findings": findings,
    }


def write_findings_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write the artifact as pretty-printed UTF-8 JSON, creating parent dirs.

    The :data:`MAX_ARTIFACT_BYTES` cap is enforced here as well as on load
    (issue #1113): without a write-side check an oversized payload -- a large
    grounded-diagram payload is the realistic case -- would succeed in Phase A
    and fail one job later inside the privileged poster, where the failure is
    far harder to attribute. Measured on the exact bytes about to be written.

    Raises:
        FindingsValidationError: When the rendered artifact exceeds the cap.
    """
    text = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    size = len(text.encode("utf-8"))
    if size > MAX_ARTIFACT_BYTES:
        raise FindingsValidationError(
            f"artifact size check failed: {size} bytes exceeds the {MAX_ARTIFACT_BYTES}-byte cap"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_findings_artifact(
    path: Path,
    *,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_sha: str,
) -> FindingsArtifact:
    """Load an artifact and validate it against event-derived facts.

    This is the confused-deputy gate for the privileged poster: the artifact
    is untrusted Phase A output, so every check runs before its content is
    acted on. Checks run in order: file size (stat, before reading), JSON
    parse, strict schema validation, then equality of the declared
    ``repo``/``pr_number``/``head_sha`` against the expected (event-derived)
    values. Artifact content is never executed or interpolated.

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

    return FindingsArtifact(
        repo=data["repo"],
        pr_number=data["pr_number"],
        head_sha=data["head_sha"],
        run_info=data.get("run_info"),
        findings=[ArtifactFinding(**f) for f in data["findings"]],
        kind=data.get("kind") or "review",
        diagrams=data.get("diagrams"),
    )
