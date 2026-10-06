"""Pure comparison of successful host-test evidence with a gate's typed target.

Check absence, incomplete execution, absent components, then all identity fields
in declared order. A content-only tree-key match is insufficient: HEAD and branch
must also match unless the caller verified the post-commit retained tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from daydream.test_execution import TestExecutionIdentity

#: Bump whenever the decision/audit shape changes so a stale record can never
#: be read as the current contract (mirrors ``REUSE_KEY_FORMAT``).
EVIDENCE_REUSE_FORMAT: int = 1

ReuseResult = Literal[
    "reused",
    "no-evidence",
    "incomplete-evidence",
    "identity-mismatch",
    "absent-component",
]

#: The fixed component comparison order. ``tree_key`` is the one component
#: whose two identity fields (input/output) are both compared to the target's
#: single retained tree key.
_COMPONENT_ORDER: tuple[str, ...] = (
    "session_id",
    "argv",
    "cwd_relative",
    "runner",
    "interpreter",
    "config_digest",
    "tree_key",
    "head_sha",
    "branch",
)

@dataclass(frozen=True)
class ReuseTarget:
    """Resolved target; post_commit_verified attests the committed retained path/state set.

    Only that strict post-commit proof permits ignoring changed HEAD and branch.
    """

    session_id: str
    tree_key: str
    argv: tuple[str, ...]
    cwd_relative: str
    runner: str | None
    interpreter: str | None
    config_digest: str | None
    absent_components: tuple[str, ...]
    head_sha: str
    branch: str
    post_commit_verified: bool


@dataclass(frozen=True)
class ReuseDecision:
    """Reuse outcome with every mismatched or absent component named for audit."""

    reused: bool
    result: ReuseResult
    mismatched_components: tuple[str, ...] = ()


def _absent_names(
    identity: TestExecutionIdentity, target: ReuseTarget
) -> tuple[str, ...]:
    """Every named absent input across both sides, sorted and deduplicated."""
    return tuple(sorted(set(identity.absent_components) | set(target.absent_components)))


def _component_matches(
    name: str, identity: TestExecutionIdentity, target: ReuseTarget
) -> bool:
    """Compare one named component; ``tree_key`` binds both identity tree keys."""
    if name == "tree_key":
        return (
            identity.input_tree_key == target.tree_key
            and identity.output_tree_key == target.tree_key
        )
    return bool(getattr(identity, name) == getattr(target, name))


def _component_mismatches(
    identity: TestExecutionIdentity, target: ReuseTarget
) -> tuple[str, ...]:
    """Return the mismatching component names in the fixed compare order."""
    exempt = {"head_sha", "branch"} if target.post_commit_verified else set()
    return tuple(
        name
        for name in _COMPONENT_ORDER
        if name not in exempt and not _component_matches(name, identity, target)
    )


def decide_reuse(
    identity: TestExecutionIdentity | None, target: ReuseTarget
) -> ReuseDecision:
    """Decide whether ``identity`` authorizes reuse at ``target`` (pure).

    No I/O, no model/network call; every input is an already-resolved value.
    """
    if identity is None:
        return ReuseDecision(reused=False, result="no-evidence")
    if not identity.reusable:
        return ReuseDecision(reused=False, result="incomplete-evidence")
    absent = _absent_names(identity, target)
    if absent:
        return ReuseDecision(
            reused=False, result="absent-component", mismatched_components=absent
        )
    mismatches = _component_mismatches(identity, target)
    if mismatches:
        return ReuseDecision(
            reused=False, result="identity-mismatch", mismatched_components=mismatches
        )
    return ReuseDecision(reused=True, result="reused")


def audit_payload(
    decision: ReuseDecision,
    identity: TestExecutionIdentity | None,
    target: ReuseTarget,
) -> dict[str, Any]:
    """Record tree keys, HEADs, branch, and mismatch names only.

    Never include commands, config digests, secrets, or environment values.
    before_head_sha belongs to the evidence; after_head_sha belongs to the target.
    """
    return {
        "format_version": EVIDENCE_REUSE_FORMAT,
        "reused": decision.reused,
        "result": decision.result,
        "mismatched_components": list(decision.mismatched_components),
        "input_tree_key": None if identity is None else identity.input_tree_key,
        "output_tree_key": None if identity is None else identity.output_tree_key,
        "target_tree_key": target.tree_key,
        "before_head_sha": None if identity is None else identity.head_sha,
        "after_head_sha": target.head_sha,
        "branch": target.branch,
        "post_commit_verified": target.post_commit_verified,
        "post_commit_verification_used": (
            decision.reused and target.post_commit_verified
        ),
    }
