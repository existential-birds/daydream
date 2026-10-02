"""Fetch complete CI snapshots through the bounded GitHub transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from daydream.git_ops import (
    INHERIT_GITHUB_AUTH,
    GitError,
    GitHubAuth,
    GitHubPageLimits,
    GitHubRequestBudget,
    gh_active_branch_rules,
    gh_api_bounded_pages,
    gh_classic_required_checks,
    gh_pr_ci_snapshot,
)
from daydream.remote_ci.evidence import (
    CIObservation,
    RemoteCIIdentityMismatch,
    RemoteCILimits,
    RemoteCISnapshot,
    RemoteCITarget,
)
from daydream.remote_ci.parsing import (
    parse_active_workflows,
    parse_observations,
    parse_pr_ci_binding,
    parse_required_policy,
)


class RemoteCIFetcher(Protocol):
    """One complete remote-CI poll under a shared absolute request budget."""

    async def fetch(
        self, target: RemoteCITarget, *, budget: GitHubRequestBudget
    ) -> RemoteCISnapshot: ...


@dataclass(frozen=True)
class GitHubRemoteCIFetcher:
    """Fetch normalized CI evidence through the sole async GitHub boundary."""

    limits: RemoteCILimits = field(default_factory=RemoteCILimits)
    auth: GitHubAuth = field(
        default=INHERIT_GITHUB_AUTH,
        repr=False,
        compare=False,
    )

    async def fetch(
        self, target: RemoteCITarget, *, budget: GitHubRequestBudget
    ) -> RemoteCISnapshot:
        owner, name = target.base_repository.split("/", 1)
        pages = GitHubPageLimits(
            per_page=self.limits.per_page, max_pages=self.limits.max_pages
        )
        try:
            binding = parse_pr_ci_binding(
                await gh_pr_ci_snapshot(
                    target.target_dir,
                    owner,
                    name,
                    target.pr_number,
                    budget=budget,
                    auth=self.auth,
                ),
                target,
            )
            active_rules = await gh_active_branch_rules(
                target.target_dir,
                owner,
                name,
                target.base_ref,
                limits=pages,
                budget=budget,
                auth=self.auth,
            )
            classic = await gh_classic_required_checks(
                target.target_dir,
                owner,
                name,
                target.base_ref,
                budget=budget,
                auth=self.auth,
            )
            policy = parse_required_policy(active_rules, classic)
            workflows = parse_active_workflows(
                await gh_api_bounded_pages(
                    target.target_dir,
                    f"repos/{owner}/{name}/actions/workflows",
                    envelope='workflows',
                    limits=pages,
                    budget=budget,
                    auth=self.auth,
                )
            )
            head = await self._observations(
                target, owner, name, target.pushed_sha, pages=pages, budget=budget
            )
            merge: tuple[CIObservation, ...] = ()
            if binding.merge_sha is not None and binding.merge_sha != target.pushed_sha:
                merge = await self._observations(
                    target,
                    owner,
                    name,
                    binding.merge_sha,
                    pages=pages,
                    budget=budget,
                )
        except RemoteCIIdentityMismatch:
            raise
        except ValueError as exc:
            raise GitError("GitHub remote CI response failed strict validation") from exc
        return RemoteCISnapshot(
            target=target,
            binding=binding,
            policy=policy,
            active_workflows=workflows,
            head_observations=head,
            merge_observations=merge,
        )

    async def _observations(
        self,
        target: RemoteCITarget,
        owner: str,
        name: str,
        sha: str,
        *,
        pages: GitHubPageLimits,
        budget: GitHubRequestBudget,
    ) -> tuple[CIObservation, ...]:
        checks = await gh_api_bounded_pages(
            target.target_dir,
            f"repos/{owner}/{name}/commits/{sha}/check-runs?filter=latest",
            envelope='check_runs',
            limits=pages,
            budget=budget,
            auth=self.auth,
        )
        statuses = await gh_api_bounded_pages(
            target.target_dir,
            f"repos/{owner}/{name}/commits/{sha}/statuses",
            envelope=None,
            limits=pages,
            budget=budget,
            auth=self.auth,
        )
        return parse_observations(
            checks, statuses, expected_sha=sha, limit=self.limits.diagnostic_chars
        )
