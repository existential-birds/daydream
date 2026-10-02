"""Resolve the exact independent snapshot bound to an Improve flow."""

from pathlib import Path

from daydream.flows.engine import FlowContext


def _audit_repo(ctx: FlowContext) -> Path:
    """Return the independent snapshot used as cwd for every advisory model turn.

    ``ctx.data["audit_repo"]`` must resolve to the exact canonical root carried by
    ``ctx.audit_workspace``. Raise RuntimeError for an absent, invalid, or mismatched binding.
    """
    audit = ctx.audit_workspace
    raw_repo = ctx.data.get("audit_repo")
    if audit is None or not isinstance(raw_repo, Path):
        raise RuntimeError("improve flow has no bound audit workspace")
    try:
        data_repo = raw_repo.resolve(strict=True)
        boundary_repo = audit.repo.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("improve flow has an invalid audit workspace") from exc
    if data_repo != boundary_repo:
        raise RuntimeError("improve flow audit workspace identity mismatch")
    return boundary_repo
