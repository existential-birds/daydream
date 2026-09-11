"""Validated archive rows and immutable acquired harvest evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from daydream.training._immutable_json import freeze_json
from daydream.training.reward import ScoringInputs
from daydream.training.rubric import Rubric

BaseShaStatus = Literal["available", "unavailable", "failed"]


@dataclass(frozen=True)
class HarvestRow:
    """The consumed archive columns, validated before acquisition side effects."""

    session_id: str
    archive_path: Path
    source_path: Path | None = None
    remote_url: str | None = None
    repo_slug: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    head_sha: str | None = None
    base_sha: str | None = None
    pr_repo: str | None = None
    pr_number: int | None = None
    grounding_rate: float | None = None
    changed_files: tuple[str, ...] = ()
    findings_fingerprints: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_files", tuple(self.changed_files))
        if self.findings_fingerprints is not None:
            object.__setattr__(self, "findings_fingerprints", tuple(self.findings_fingerprints))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, row_number: int) -> HarvestRow:
        """Validate identity/path columns without probing files or GitHub.

        Optional historical columns may be absent. Malformed changed-files JSON
        retains its established empty-list fallback; invalid identity/path types
        do not silently select an ambient directory or external repository.
        """
        session = raw.get("session_id") if isinstance(raw, Mapping) else None
        label = repr(session) if isinstance(session, str) else "<unknown>"

        def invalid(field: str, reason: str) -> ValueError:
            return ValueError(f"harvest row {row_number} session {label}: {field} {reason}")

        if not isinstance(raw, Mapping):
            raise invalid("row", "must be a mapping")
        if (
            not isinstance(session, str) or not session.strip()
            or session in (".", "..") or any(char in session for char in "/\\\0")
        ):
            raise invalid("session_id", "must be a nonempty safe path segment")

        def text(field: str) -> str | None:
            value = raw.get(field)
            if value is None:
                return None
            if not isinstance(value, str):
                raise invalid(field, "must be a string or null")
            if "\0" in value:
                raise invalid(field, "must not contain NUL")
            return value

        def path(field: str, *, required: bool = False) -> Path | None:
            value = text(field)
            if value is None or value == "":
                if required:
                    raise invalid(field, "must be a nonempty absolute path")
                return None
            result = Path(value)
            if required and not result.is_absolute():
                raise invalid(field, "must be an absolute path")
            return result

        def slug(field: str) -> str | None:
            value = text(field)
            if value is None or value == "":
                return value
            parts = value.split("/")
            if len(parts) != 2 or any(part in ("", ".", "..") for part in parts) or "\\" in value:
                raise invalid(field, "must contain safe owner/repository components")
            return value

        archive_path = path("archive_path", required=True)
        assert archive_path is not None
        number = raw.get("pr_number")
        if number is not None and (type(number) is not int or number <= 0):
            raise invalid("pr_number", "must be a positive integer or null")
        grounding = raw.get("grounding_rate")
        if grounding is not None and type(grounding) not in (int, float):
            raise invalid("grounding_rate", "must be a number or null")
        changed = raw.get("changed_files")
        if changed is None:
            changed = []
        elif not isinstance(changed, list):
            try:
                changed = json.loads(changed)
            except (TypeError, json.JSONDecodeError):
                changed = []
        if not isinstance(changed, list):
            changed = []
        if any(not isinstance(item, str) for item in changed):
            raise invalid("changed_files", "must contain strings")
        fingerprints = raw.get("findings_fingerprints")
        return cls(
            session_id=session, archive_path=archive_path,
            source_path=path("source_path"), remote_url=text("remote_url"),
            repo_slug=slug("repo_slug"), branch=text("branch"),
            base_branch=text("base_branch"), head_sha=text("head_sha"),
            base_sha=text("base_sha"), pr_repo=slug("pr_repo"),
            pr_number=number, grounding_rate=grounding,
            changed_files=tuple(changed),
            findings_fingerprints=(
                tuple(str(item) for item in fingerprints) if isinstance(fingerprints, list) else None
            ),
        )

    @property
    def is_pr(self) -> bool:
        """Preserve the existing PR discriminator, including incomplete links."""
        return bool(self.pr_repo) and self.pr_number is not None

    def as_signal_row(self) -> dict[str, Any]:
        """Project a fresh legacy row for the unchanged external signal helpers."""
        row: dict[str, Any] = {
            "session_id": self.session_id, "archive_path": str(self.archive_path),
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "remote_url": self.remote_url, "repo_slug": self.repo_slug,
            "branch": self.branch, "base_branch": self.base_branch,
            "head_sha": self.head_sha, "base_sha": self.base_sha,
            "pr_repo": self.pr_repo, "pr_number": self.pr_number,
            "grounding_rate": self.grounding_rate, "changed_files": list(self.changed_files),
        }
        if self.findings_fingerprints is not None:
            row["findings_fingerprints"] = list(self.findings_fingerprints)
        return row


@dataclass(frozen=True)
class HarvestEvidence:
    """Completed acquisition, with owned immutable nested signal collections."""

    scoring_inputs: ScoringInputs
    rubric: Rubric
    reviewer_logins: tuple[str, ...] = ()
    pooled_prior: float | None = None
    prior_n: int = 0
    repo_resolution: Path | None = None
    base_sha_status: BaseShaStatus = "unavailable"
    valid_at_override: str | None = None

    def __post_init__(self) -> None:
        verdicts = self.scoring_inputs.verifier_verdicts
        object.__setattr__(self, "scoring_inputs", replace(
            self.scoring_inputs,
            verifier_verdicts=None if verdicts is None else tuple(freeze_json(item) for item in verdicts),
        ))
        resolutions = self.rubric.per_finding_resolutions
        object.__setattr__(self, "rubric", replace(
            self.rubric,
            fix_applied=replace(self.rubric.fix_applied, window_commits=tuple(self.rubric.fix_applied.window_commits)),
            per_finding_resolutions=(
                None if resolutions is None else tuple(
                    replace(resolution, evidence=tuple(freeze_json(entry) for entry in resolution.evidence))
                    for resolution in resolutions
                )
            ),
        ))
        object.__setattr__(self, "reviewer_logins", tuple(self.reviewer_logins))
