"""The pure evidence-reuse decision predicate for the fix/commit gates.

A successful host test run may stand in for a fresh validation at a later
decision gate only when the *whole* typed identity of that run still matches
the gate's target. This module owns that comparison and nothing else: it takes
already-resolved values, performs no I/O, and makes no model or network call,
so the same inputs always yield the same :class:`ReuseDecision`.

Check order (each earlier clause wins):

1. no evidence at all -> ``no-evidence`` (a fresh validation is required);
2. red or incomplete evidence (``identity.reusable`` false: an agent-reported
   verdict, a timeout, a truncated run) -> ``incomplete-evidence``;
3. any named absent component (Pattern C: an unreadable config input) ->
   ``absent-component``, naming the absent input(s);
4. component-wise comparison in the fixed order ``session_id``, ``argv``,
   ``cwd_relative``, ``runner``, ``interpreter``, ``config_digest``, tree keys,
   ``head_sha``, ``branch`` -> ``identity-mismatch``, naming every mismatch.

The tree key is content-only, so a commit can move ``HEAD`` without moving it.
A bare tree-key match therefore never authorizes reuse: ``head_sha``/``branch``
must also match, unless the caller has *verified* the post-commit tree with
``ReuseTarget.post_commit_verified``.
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
    """The already-resolved target a reuse decision is compared against.

    ``tree_key`` is the retained tree's content key (supplied by the caller,
    never recomputed here). ``post_commit_verified`` is set only after the
    strict post-commit verification proved the created commit carries exactly
    the retained path/state set that was tested; when set, the ``head_sha`` and
    ``branch`` components are not compared.
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
    """The pure outcome of one reuse comparison.

    ``mismatched_components`` names every component that failed (or the absent
    inputs when ``result == "absent-component"``), so the caller can explain a
    miss without re-deriving it.
    """

    reused: bool
    result: ReuseResult
    mismatched_components: tuple[str, ...] = ()


def _absent_names(
    identity: TestExecutionIdentity, target: ReuseTarget
) -> tuple[str, ...]:
    """Every named absent input across both sides, sorted and deduplicated."""
    return tuple(sorted(set(identity.absent_components) | set(target.absent_components)))


def _component_mismatches(
    identity: TestExecutionIdentity, target: ReuseTarget
) -> tuple[str, ...]:
    """Return the mismatching component names in the fixed compare order."""
    comparisons: dict[str, bool] = {
        "session_id": identity.session_id == target.session_id,
        "argv": identity.argv == target.argv,
        "cwd_relative": identity.cwd_relative == target.cwd_relative,
        "runner": identity.runner == target.runner,
        "interpreter": identity.interpreter == target.interpreter,
        "config_digest": identity.config_digest == target.config_digest,
        "tree_key": (
            identity.input_tree_key == target.tree_key
            and identity.output_tree_key == target.tree_key
        ),
        "head_sha": identity.head_sha == target.head_sha,
        "branch": identity.branch == target.branch,
    }
    exempt = {"head_sha", "branch"} if target.post_commit_verified else set()
    return tuple(
        name
        for name in _COMPONENT_ORDER
        if name not in exempt and not comparisons[name]
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
    """Build the strict-JSON audit record for one reuse decision.

    Only identity *facts* are recorded — tree keys, HEAD shas, branch, and the
    mismatched component names — never commands, config digests, secrets, or
    environment values. ``before_head_sha`` is the evidence's HEAD and
    ``after_head_sha`` the target's.
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
