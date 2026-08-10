"""Public publisher port for the review service (Plan 008 Step 5).

A publisher converts a fail-closed policy decision into an external review
artifact (e.g. a GitHub Check). The port is deliberately narrow and trust-
neutral: it accepts a decision, an immutable external id to bind, a conclusion,
and a bounded summary, and returns a receipt. The trusted adapter that actually
holds external write authority is separate (``daydream.github_app`` for GitHub
Checks).

The port is fail-closed: ``publish`` may raise ``PublishError`` and a caller
must treat any exception as not-published. A retry is an explicit second call
that re-raises — it never silently flips a failure to success.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from daydream.service.models import ReviewTarget

Conclusion = Literal["success", "failure", "neutral", "cancelled", "action_required"]


class PublishError(Exception):
    """Raised when a publisher could not durably record a decision.

    Fail-closed: the caller must treat this as not-published and must not
    fabricate success.
    """


class PublishRequest:
    """A request to durably publish one policy decision.

    Attributes:
        external_id: The immutable job/candidate id the external artifact binds
            to (used as the Check ``external_id`` by GitHub).
        conclusion: The terminal conclusion to publish.
        summary: A bounded, non-secret summary (never source, prompts, or
            secrets).
        repo: ``owner/repo`` slug the artifact belongs to.
        target_sha: The exact candidate SHA the artifact is bound to.
        check_name: Exact Check identity.
        target: The full ReviewTarget (kind + PR/merge-group identity) the
            publisher revalidates live identity against before publishing
            success.
    """

    __slots__ = (
        "external_id",
        "conclusion",
        "summary",
        "repo",
        "target_sha",
        "check_name",
        "target",
    )

    def __init__(
        self,
        *,
        external_id: str,
        conclusion: Conclusion,
        summary: str,
        repo: str = "",
        target_sha: str = "",
        check_name: str = "",
        target: "ReviewTarget | None" = None,
    ) -> None:
        if not external_id:
            raise ValueError("external_id must be non-empty (binds the artifact immutably)")
        self.external_id = external_id
        self.conclusion = conclusion
        self.summary = summary
        self.repo = repo
        self.target_sha = target_sha
        self.check_name = check_name
        self.target = target


class PublishReceipt:
    """Proof a decision was durably published.

    Attributes:
        external_id: The immutable id the artifact was bound to.
        check_run_id: External identifier of the created artifact (e.g. GitHub
            check run id), when the adapter knows one.
    """

    __slots__ = ("external_id", "check_run_id")

    def __init__(self, *, external_id: str, check_run_id: int | None = None) -> None:
        self.external_id = external_id
        self.check_run_id = check_run_id


@runtime_checkable
class Publisher(Protocol):
    """Port: durably publish a policy decision.

    Implementations hold the external write authority (e.g. Checks-write) and
    MUST revalidate live identity before publishing success; they never accept
    worker-asserted credentials. ``publish`` raises :class:`PublishError` on any
    failure.
    """

    def publish(self, req: PublishRequest) -> PublishReceipt:
        """Durably publish *req*; raises :class:`PublishError` on failure."""
        ...
