"""Resolve the exact independent snapshot bound to an Improve flow."""

from pathlib import Path

from daydream.flows.engine import FlowContext


def _audit_repo(ctx: FlowContext) -> Path:
    """Return the independent snapshot used as cwd for every advisory model turn.

    The captured audit capability owns the canonical model working directory.
    Raise RuntimeError for an absent or invalid workspace.
    """
    audit = ctx.audit_workspace
    if audit is None or not isinstance(audit.repo, Path):
        raise RuntimeError("improve flow has no bound audit workspace")
    try:
        boundary_repo = audit.repo.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("improve flow has an invalid audit workspace") from exc
    return boundary_repo
