"""Account for selected packages and publish their validated local plans as issues."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import trajectory
from daydream.agent import console
from daydream.config_file import DaydreamFileConfig
from daydream.improve import artifacts
from daydream.improve.publish import ImprovePublishError, IssuePublisher
from daydream.ui import print_error, print_success, print_warning

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext




def _automatic_issue_publishing(ctx: FlowContext) -> bool:
    """Return whether this Improve run is configured to publish plans."""
    file_config = ctx.config.file_config or DaydreamFileConfig()
    return bool(getattr(file_config, "improve_github_publish_issues", False))


def _local_plan_path(repo: Path, entry: dict[str, Any]) -> Path | None:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else repo / "daydream_plans" / path


def _publication_package_id(
    finding: dict[str, Any],
    entry: dict[str, Any] | None = None,
) -> str:
    """Return the plan's stored package identity before the current finding's."""
    return str(
        (entry or {}).get("package_fingerprint")
        or finding.get("package_fingerprint")
        or finding.get("fingerprint")
        or ""
    )


def _publication_members(
    entry: dict[str, Any],
    finding: dict[str, Any],
    field: str,
) -> tuple[str, ...]:
    """Read stored plan membership first, with current-finding compatibility."""
    raw = entry[field] if field in entry else finding.get(field)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(value for value in raw if isinstance(value, str) and value)


def _publication_identity(
    entry: dict[str, Any],
    finding: dict[str, Any],
    plan_path: Path | None,
) -> dict[str, Any]:
    """Extract the per-package identity shared by every publication record."""
    return {
        "package_id": _publication_package_id(finding, entry),
        "title": str(entry.get("title") or finding.get("title") or "Improve repository"),
        "plan_path": plan_path.name if plan_path is not None else None,
        "member_fingerprints": list(_publication_members(entry, finding, "member_fingerprints")),
        "member_aliases": list(_publication_members(entry, finding, "member_aliases")),
    }


async def _step_publish_issues(ctx: FlowContext) -> None:
    """Copy each validated local plan into one idempotent GitHub issue."""
    enabled = _automatic_issue_publishing(ctx)
    publication: dict[str, Any] = {
        "enabled": enabled,
        "status": "running" if enabled else "disabled",
        "repository": ctx.config.pr_repo,
        "published": [],
        "failed": [],
    }
    ctx.data["issue_publication"] = publication
    if not enabled:
        _save_publication(ctx, publication, final=True)
        return

    candidates: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    represented: set[str] = set()
    plan_write = ctx.data["plan_write"]
    for disposition in ("written", "skipped", "failed"):
        for entry in plan_write.get(disposition, []):
            if not isinstance(entry, dict):
                continue
            nested_finding = entry.get("finding")
            finding = nested_finding if isinstance(nested_finding, dict) else entry
            package_id = _publication_package_id(finding, entry)
            if package_id:
                represented.add(package_id)
            current_package_id = _publication_package_id(finding)
            if current_package_id:
                represented.add(current_package_id)
            plan_path = _local_plan_path(ctx.work.repo, entry)
            identity = _publication_identity(entry, finding, plan_path)
            if disposition == "failed":
                identity["plan_path"] = None
                publication["failed"].append(
                    {
                        **identity,
                        "stage": "plan-write",
                        "error": "Plan writing did not produce a validated local plan.",
                    }
                )
            elif plan_path is None:
                publication["failed"].append(
                    {
                        **identity,
                        "stage": "plan-reconciliation",
                        "error": "The selected package was skipped without an unambiguous validated local plan.",
                    }
                )
            elif not plan_path.is_file():
                publication["failed"].append(
                    {
                        **identity,
                        "stage": "local-plan",
                        "error": "The validated local plan file is unavailable.",
                    }
                )
            else:
                candidates.append((entry, finding, plan_path))

    for finding in ctx.data.get("selected_findings", []):
        if not isinstance(finding, dict):
            continue
        package_id = _publication_package_id(finding)
        if package_id in represented:
            continue
        publication["failed"].append(
            {
                "package_id": package_id,
                "title": str(finding.get("title") or "Improve repository"),
                "plan_path": None,
                "stage": "plan-accounting",
                "error": ("Plan writing produced no outcome for the selected package."),
            }
        )

    artifacts.write_artifact(
        ctx.data["improve_dir"] / artifacts.PUBLISHED_ISSUES_FILENAME,
        publication,
        phase=trajectory.DaydreamPhase.PLAN_WRITE,
    )
    if not candidates:
        _save_publication(ctx, publication, final=True)
        return

    try:
        publisher = IssuePublisher.connect(
            ctx.work.repo,
            repo_slug=ctx.config.pr_repo,
            auth=ctx.github_execution.auth,
        )
    except ImprovePublishError as exc:
        safe_error = trajectory.redact_text(str(exc))
        for entry, finding, plan_path in candidates:
            publication["failed"].append(
                {
                    **_publication_identity(entry, finding, plan_path),
                    "stage": "preflight",
                    "error": safe_error,
                }
            )
        _save_publication(ctx, publication, final=True)
        print_error(console, "Improve issue publishing failed", safe_error)
        return

    publication["repository"] = publisher.repo_slug
    for entry, finding, plan_path in sorted(
        candidates,
        key=lambda item: int(item[0].get("number") or 0),
    ):
        identity = _publication_identity(entry, finding, plan_path)
        try:
            result = publisher.publish(
                package_id=identity["package_id"],
                title=identity["title"],
                plan_path=plan_path,
                member_fingerprints=identity["member_fingerprints"],
                member_aliases=identity["member_aliases"],
            )
        except (ImprovePublishError, ValueError) as exc:
            publication["failed"].append(
                {**identity, "stage": "issue-create", "error": trajectory.redact_text(str(exc))}
            )
            print_warning(
                console,
                f"Issue publishing failed for {identity['title']}: {trajectory.redact_text(str(exc))}",
            )
        else:
            publication["published"].append(
                {
                    **identity,
                    "disposition": result.disposition,
                    "issue_url": result.issue_url,
                }
            )
            print_success(
                console,
                f"Issue {result.disposition} for {identity['title']}: {result.issue_url}",
            )
        _save_publication(ctx, publication)

    _save_publication(ctx, publication, final=True)


def _save_publication(ctx: FlowContext, publication: dict[str, Any], *, final: bool = False) -> None:
    """Persist each publication outcome and apply partial-failure exits at completion."""
    failed = bool(publication.get("failed"))
    published = bool(publication.get("published"))
    publication["status"] = (
        "disabled" if not publication.get("enabled")
        else "partial" if failed and published
        else "failed" if failed
        else "complete"
    )
    if final and failed:
        ctx.data["plan_exit_code"] = max(ctx.data["plan_exit_code"], 2 if published else 1)
    artifacts.write_artifact(
        ctx.data["improve_dir"] / artifacts.PUBLISHED_ISSUES_FILENAME,
        publication,
        phase=trajectory.DaydreamPhase.PLAN_WRITE,
    )
