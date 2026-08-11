"""Idempotent GitHub issue publication for locally written Improve plans.

The local plan remains the plan writer's validated output. Publication copies
that Markdown verbatim into an issue body; it never creates branches, commits,
or pushes. A stable marker stored in GitHub makes reconciliation independent of
local ``daydream_plans`` state, which is important for fresh CI checkouts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from daydream import git_ops

PublicationDisposition = Literal["created", "existing", "reconciled"]
_PACKAGE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_MEMBER_ALIAS_MARKER_PREFIX = "<!-- daydream-improve-member: alias="
_MEMBER_FINGERPRINT_MARKER_PREFIX = "<!-- daydream-improve-member: fingerprint="


class ImprovePublishError(RuntimeError):
    """Raised when an Improve plan cannot be published without duplication risk."""


@dataclass(frozen=True)
class PublishResult:
    """Outcome of publishing one locally written plan."""

    package_id: str
    plan_path: Path
    disposition: PublicationDisposition
    issue_url: str


def package_marker(package_id: str) -> str:
    """Return the stable, injection-safe issue marker for a work package."""
    if _PACKAGE_ID_RE.fullmatch(package_id) is None:
        raise ValueError(
            "package_id must contain only letters, numbers, '.', '_', ':', or '-'"
        )
    return f"<!-- daydream-improve: package={package_id} -->"


def member_marker(member_alias: str) -> str:
    """Return an injection-safe marker for one stable package member alias."""
    if _PACKAGE_ID_RE.fullmatch(member_alias) is None:
        raise ValueError(
            "member_alias must contain only letters, numbers, '.', '_', ':', or '-'"
        )
    return f"{_MEMBER_ALIAS_MARKER_PREFIX}{member_alias} -->"


def member_fingerprint_marker(fingerprint: str) -> str:
    """Return an injection-safe marker for one audit-time member identity."""
    if _PACKAGE_ID_RE.fullmatch(fingerprint) is None:
        raise ValueError(
            "member fingerprint must contain only letters, numbers, '.', '_', ':', or '-'"
        )
    return f"{_MEMBER_FINGERPRINT_MARKER_PREFIX}{fingerprint} -->"


def _member_markers(member_aliases: Sequence[str]) -> tuple[str, ...]:
    return tuple(member_marker(alias) for alias in dict.fromkeys(member_aliases))


def issue_body(
    package_id: str,
    plan_markdown: str,
    *,
    member_aliases: Sequence[str] = (),
    member_fingerprints: Sequence[str] = (),
) -> str:
    """Prefix a complete plan with its durable GitHub reconciliation marker."""
    if not plan_markdown.strip():
        raise ValueError("plan Markdown must not be empty")
    markers = (
        package_marker(package_id),
        *_member_markers(member_aliases),
        *tuple(
            member_fingerprint_marker(fingerprint)
            for fingerprint in dict.fromkeys(member_fingerprints)
        ),
    )
    return f"{'\n'.join(markers)}\n\n{plan_markdown}"


def _repo_slug(repo: Path, explicit: str | None) -> str:
    if explicit is not None:
        parsed = git_ops.split_owner_repo(explicit)
        if parsed is None or "/" in parsed[1]:
            raise ImprovePublishError(f"Invalid GitHub repository slug {explicit!r}; expected owner/repo")
        return explicit
    try:
        inferred = git_ops.gh_repo_view(repo)
    except git_ops.GitError as exc:
        raise ImprovePublishError("Cannot infer the GitHub repository from this checkout") from exc
    if inferred is None:
        raise ImprovePublishError("Cannot infer the GitHub repository from this checkout")
    return f"{inferred[0]}/{inferred[1]}"


class IssuePublisher:
    """Publish plans to one repository with strict cross-run reconciliation.

    Construct with :meth:`connect` once per Improve run. The initial issue
    lookup doubles as a fail-closed preflight and is cached across all plans.
    """

    def __init__(
        self,
        repo: Path,
        repo_slug: str,
        issues: list[dict[str, object]],
    ) -> None:
        self._repo = repo
        self.repo_slug = repo_slug
        self._issues = issues

    @classmethod
    def connect(
        cls,
        repo: Path,
        *,
        repo_slug: str | None = None,
    ) -> IssuePublisher:
        """Resolve the repository and load open and closed issues strictly."""
        resolved = _repo_slug(repo, repo_slug)
        try:
            issues = git_ops.gh_issue_list_strict(
                repo,
                state="all",
                repo_slug=resolved,
            )
        except git_ops.GitError as exc:
            raise ImprovePublishError(
                "Cannot safely reconcile existing Improve issues; no issues were created"
            ) from exc
        return cls(repo, resolved, list(issues))

    def _matches(self, marker: str) -> list[dict[str, object]]:
        return [issue for issue in self._issues if marker in str(issue.get("body") or "")]

    @staticmethod
    def _issue_key(issue: dict[str, object]) -> str:
        return str(issue.get("url") or issue.get("number") or id(issue))

    def _existing(
        self,
        package: str,
        required_members: tuple[str, ...],
        related_members: tuple[str, ...],
    ) -> dict[str, object] | None:
        """Resolve one issue only when it covers the complete current package.

        Member aliases let a fresh checkout recognize a package whose wording
        or grouping changed. Any partial overlap fails closed: reusing that
        issue would lose new work, while creating another would give the user
        overlapping issues.
        """
        package_matches = self._matches(package)
        related: list[dict[str, object]] = []
        full_member_matches: list[dict[str, object]] = []
        if related_members:
            for issue in self._issues:
                body = str(issue.get("body") or "")
                if any(marker in body for marker in related_members):
                    related.append(issue)
                if required_members and all(
                    marker in body for marker in required_members
                ):
                    full_member_matches.append(issue)

        candidates = {
            self._issue_key(issue): issue
            for issue in [*package_matches, *full_member_matches]
        }
        related_keys = {self._issue_key(issue) for issue in related}
        if len(candidates) > 1 or (
            candidates and related_keys - set(candidates)
        ):
            ambiguous = {
                self._issue_key(issue): issue
                for issue in [*candidates.values(), *related]
            }
            urls = ", ".join(
                str(issue.get("url") or "unknown")
                for issue in ambiguous.values()
            )
            raise ImprovePublishError(
                "Multiple GitHub issues overlap the same Improve work package: "
                f"{urls}"
            )

        if package_matches:
            existing = package_matches[0]
            body = str(existing.get("body") or "")
            has_member_metadata = (
                _MEMBER_ALIAS_MARKER_PREFIX in body
                or _MEMBER_FINGERPRINT_MARKER_PREFIX in body
            )
            if required_members and has_member_metadata and not all(
                marker in body for marker in required_members
            ):
                raise ImprovePublishError(
                    "An existing GitHub issue only partially covers this Improve "
                    "work package; refusing to publish stale or overlapping plan text"
                )
            return existing
        if full_member_matches:
            return full_member_matches[0]
        if related:
            raise ImprovePublishError(
                "An existing GitHub issue only partially covers this Improve work "
                "package; refusing to create an overlapping issue"
            )
        return None

    def _refresh_after_failed_create(self, create_error: git_ops.GitError) -> None:
        try:
            self._issues = git_ops.gh_issue_list_strict(
                self._repo,
                state="all",
                repo_slug=self.repo_slug,
            )
        except git_ops.GitError as lookup_error:
            raise ImprovePublishError(
                "GitHub issue creation failed and its outcome could not be reconciled; "
                f"creation error: {create_error}; lookup error: {lookup_error}"
            ) from create_error

    def publish(
        self,
        *,
        package_id: str,
        title: str,
        plan_path: Path,
        member_aliases: Sequence[str] = (),
        member_fingerprints: Sequence[str] = (),
    ) -> PublishResult:
        """Copy one validated local plan into an idempotent GitHub issue.

        A failed create may have reached GitHub even when the client saw an
        error. The issue set is therefore refreshed before the error is exposed
        to a caller that might retry.
        """
        try:
            plan_markdown = plan_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ImprovePublishError(f"Cannot read Improve plan {plan_path}: {exc}") from exc
        try:
            body = issue_body(
                package_id,
                plan_markdown,
                member_aliases=member_aliases,
                member_fingerprints=member_fingerprints,
            )
        except ValueError as exc:
            raise ImprovePublishError(str(exc)) from exc
        marker = package_marker(package_id)
        alias_markers = _member_markers(member_aliases)
        fingerprint_markers = tuple(
            member_fingerprint_marker(fingerprint)
            for fingerprint in dict.fromkeys(member_fingerprints)
        )
        # A repeated stable alias is deliberately ambiguous: two independent
        # findings share the same semantic anchor. In that case only the raw
        # member fingerprints can prove that every member is covered.
        aliases_collide = len(set(member_aliases)) < len(member_aliases)
        required_markers = (
            fingerprint_markers if aliases_collide else alias_markers
        ) or fingerprint_markers
        related_markers = tuple(dict.fromkeys((*alias_markers, *fingerprint_markers)))
        existing = self._existing(marker, required_markers, related_markers)
        if existing is not None:
            return PublishResult(
                package_id=package_id,
                plan_path=plan_path,
                disposition="existing",
                issue_url=str(existing["url"]),
            )

        try:
            url = git_ops.gh_issue_create(
                self._repo,
                title=title,
                body=body,
                repo_slug=self.repo_slug,
            )
        except git_ops.GitError as create_error:
            self._refresh_after_failed_create(create_error)
            existing = self._existing(marker, required_markers, related_markers)
            if existing is None:
                raise ImprovePublishError(
                    f"GitHub issue creation failed and no matching issue appeared during reconciliation: {create_error}"
                ) from create_error
            return PublishResult(
                package_id=package_id,
                plan_path=plan_path,
                disposition="reconciled",
                issue_url=str(existing["url"]),
            )

        self._issues.append(
            {
                "number": None,
                "title": title,
                "body": body,
                "url": url,
                "state": "open",
            }
        )
        return PublishResult(
            package_id=package_id,
            plan_path=plan_path,
            disposition="created",
            issue_url=url,
        )
