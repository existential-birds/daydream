"""Validated record inputs and immutable acquired harvest evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from daydream.training._immutable_json import freeze_json
from daydream.training.reward import ScoringInputs
from daydream.training.rubric import Rubric


@dataclass(frozen=True)
class HarvestRow:
    """Captured run fields, validated before acquisition side effects."""

    session_id: str
    recommended_patch: str = ""
    source_path: Path | None = None
    remote_url: str | None = None
    repo_slug: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    head_sha: str | None = None
    base_sha: str | None = None
    pr_repo: str | None = None
    pr_number: int | None = None
    changed_files: tuple[str, ...] = ()
    findings_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_files", tuple(self.changed_files))
        object.__setattr__(self, "findings_fingerprints", tuple(self.findings_fingerprints))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, row_number: int) -> HarvestRow:
        """Validate current record shapes before probing files or GitHub."""
        session = raw.get("session_id") if isinstance(raw, Mapping) else None
        label = repr(session) if isinstance(session, str) else "<unknown>"

        def invalid(field: str, reason: str) -> ValueError:
            return ValueError(f"harvest row {row_number} session {label}: {field} {reason}")

        if not isinstance(raw, Mapping):
            raise invalid("row", "must be a mapping")
        if not isinstance(session, str) or not session.strip():
            raise invalid("session_id", "must be a nonempty string")

        def text(field: str) -> str | None:
            value = raw.get(field)
            if value is None:
                return None
            if not isinstance(value, str):
                raise invalid(field, "must be a string or null")
            if "\0" in value:
                raise invalid(field, "must not contain NUL")
            return value

        def path(field: str) -> Path | None:
            value = text(field)
            return Path(value) if value else None

        def slug(field: str) -> str | None:
            value = text(field)
            if value is None or value == "":
                return value
            parts = value.split("/")
            if len(parts) != 2 or any(part in ("", ".", "..") for part in parts) or "\\" in value:
                raise invalid(field, "must contain safe owner/repository components")
            return value

        number = raw.get("pr_number")
        if number is not None and (type(number) is not int or number <= 0):
            raise invalid("pr_number", "must be a positive integer or null")

        def strings(field: str) -> tuple[str, ...]:
            value = raw.get(field, [])
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise invalid(field, "must be a list of strings")
            return tuple(value)

        return cls(
            session_id=session,
            recommended_patch=text("recommended_patch") or "",
            source_path=path("source_path"),
            remote_url=text("remote_url"),
            repo_slug=slug("repo_slug"),
            branch=text("branch"),
            base_branch=text("base_branch"),
            head_sha=text("head_sha"),
            base_sha=text("base_sha"),
            pr_repo=slug("pr_repo"),
            pr_number=number,
            changed_files=strings("changed_files"),
            findings_fingerprints=strings("findings_fingerprints"),
        )

    @property
    def is_pr(self) -> bool:
        """Preserve the existing PR discriminator, including incomplete links."""
        return bool(self.pr_repo) and self.pr_number is not None

    def as_signal_row(self) -> dict[str, Any]:
        """Project captured fields for the pure signal helpers."""
        return {
            "recommended_patch": self.recommended_patch,
            "repo_slug": self.repo_slug,
            "branch": self.branch,
            "base_branch": self.base_branch,
            "head_sha": self.head_sha,
            "pr_repo": self.pr_repo,
            "pr_number": self.pr_number,
        }


@dataclass(frozen=True)
class HarvestEvidence:
    """Completed acquisition, with owned immutable nested signal collections."""

    scoring_inputs: ScoringInputs
    rubric: Rubric
    reviewer_logins: tuple[str, ...] = ()
    pooled_prior: float | None = None
    prior_n: int = 0

    def __post_init__(self) -> None:
        verdicts = self.scoring_inputs.verifier_verdicts
        object.__setattr__(
            self,
            "scoring_inputs",
            replace(
                self.scoring_inputs,
                verifier_verdicts=None if verdicts is None else tuple(freeze_json(item) for item in verdicts),
            ),
        )
        resolutions = self.rubric.per_finding_resolutions
        object.__setattr__(
            self,
            "rubric",
            replace(
                self.rubric,
                fix_applied=replace(
                    self.rubric.fix_applied, window_commits=tuple(self.rubric.fix_applied.window_commits)
                ),
                per_finding_resolutions=(
                    None
                    if resolutions is None
                    else tuple(
                        replace(
                            resolution,
                            evidence=tuple(freeze_json(entry) for entry in resolution.evidence),
                            reply_captures=tuple(freeze_json(capture) for capture in resolution.reply_captures),
                        )
                        for resolution in resolutions
                    )
                ),
            ),
        )
        object.__setattr__(self, "reviewer_logins", tuple(self.reviewer_logins))
