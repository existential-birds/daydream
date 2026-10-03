"""Accepted fix-cycle baselines, retained-tree capture, and mutation confinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from daydream import git_ops
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.scope_issues import (
    ScopeEnforcementResult,
    enforce_authorized_fix_footprint,
)
from daydream.deep.settings import _resolve_opt_in
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.flows.engine import FlowContext
from daydream.generated_files import is_generated_file, related_manifest_paths
from daydream.git_ops import GitPathState, IndexSnapshot, WorktreeRollbackSnapshot
from daydream.json_utils import atomic_write_json
from daydream.phases import TestAttemptEvidence
from daydream.quote_scrub import scrub_smart_quotes_changed_files
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.run_config import RunConfig
    from daydream.test_execution import TestRecipe


@dataclass(frozen=True)
class EvidenceKey:
    """Tree and policy identity required by verifier evidence."""

    tree_key: str
    policy_revision: int


@dataclass(frozen=True)
class RetainedTreeSnapshot:
    """Authorized retained patch plus full observed-tree identity."""

    paths: frozenset[str]
    states: tuple[GitPathState, ...]
    tree_key: str
    verifier_patch: str
    recommended_patch: bytes


@dataclass(frozen=True)
class RepairCandidate:
    """One captured retained target and the verification that admitted it."""

    snapshot: RetainedTreeSnapshot
    key: EvidenceKey
    outcomes: dict[str, dict[str, Any]]
    test: TestAttemptEvidence | None = None


@dataclass
class FixCycleState:
    """One accepted fix gate's stable policy, baseline, and evidence."""

    work: WorkContext
    config: RunConfig
    recipe: TestRecipe | None
    session_id: str
    stable_ref: str
    stable_head: str
    initial_index: IndexSnapshot
    preexisting_untracked: dict[str, GitPathState]
    preexisting_gitlinks: tuple[GitPathState, ...]
    footprint: AuthorizedFixFootprint
    round_snapshot: RetainedTreeSnapshot | None = None
    candidate: RepairCandidate | None = None
    test_attempts: list[TestAttemptEvidence] = field(default_factory=list)
    test_retries: int = 0
    test_ignored: bool = False
    last_fix_target_by_uid: dict[str, str] = field(default_factory=dict)

    def capture_key(self) -> str:
        """Observe this accepted session's run-relative tree identity."""
        return git_ops.tree_key(
            git_ops.snapshot_worktree_delta(
                self.work.repo,
                self.stable_ref,
                preexisting_untracked=self.preexisting_untracked,
                preexisting_gitlinks=self.preexisting_gitlinks,
            )
        )

    @classmethod
    def require(cls, ctx: FlowContext) -> FixCycleState:
        """Consume only the capability published by an accepted fix gate."""
        state = ctx.data.get("fix_cycle_state")
        if not isinstance(state, cls):
            raise RuntimeError("fix cycle was not initialized at the accepted gate")
        return state


def capture_retained_tree(state: FixCycleState) -> RetainedTreeSnapshot:
    """Capture the authorized HEAD delta while keying the run-relative tree.

    ``stable_ref`` includes pre-gate tracked edits so it remains the authority
    for mutation attribution, rollback, and test identity.  Commit selection is
    deliberately relative to the original HEAD: authorized reviewed edits that
    predate the gate are part of the result even when no fixer touches them.
    Pre-existing untracked owner files remain protected even when a finding
    names them; authorization cannot silently enroll that private draft in a
    commit. New related files created after the gate are still retained.
    """
    changed = set(git_ops.changed_paths_z(state.work.repo, state.stable_head))
    paths = frozenset(
        (changed & set(state.footprint.run_allowed_paths)) - set(state.preexisting_untracked)
    )
    states = git_ops.snapshot_worktree_paths(state.work.repo, paths)
    tree_key = state.capture_key()
    recommended = git_ops.build_recommended_patch_strict(
        state.work.repo, state.stable_head, paths
    )
    return RetainedTreeSnapshot(
        paths=paths,
        states=states,
        tree_key=tree_key,
        verifier_patch=recommended.decode("utf-8", errors="replace"),
        recommended_patch=recommended,
    )


def _evidence_payload(key: EvidenceKey) -> dict[str, Any]:
    return {"tree_key": key.tree_key, "policy_revision": key.policy_revision}


def _write_footprint_audit(ctx: FlowContext, state: FixCycleState, key: EvidenceKey) -> None:
    deep_data = ctx.deep_data()
    atomic_write_json(
        DeepArtifact.FIX_FOOTPRINT.at(deep_data["dd"]),
        state.footprint.audit_payload(state.session_id, evidence_key=_evidence_payload(key)),
    )


def _persist_stabilization_failure(ctx: FlowContext, state: FixCycleState, reason: str) -> None:
    deep_data = ctx.deep_data()
    atomic_write_json(
        DeepArtifact.STABILIZATION_FAILED.at(deep_data["dd"]),
        {"session_id": state.session_id, "reason": reason},
    )


def _round_rollback_snapshot(state: FixCycleState, work: WorkContext) -> WorktreeRollbackSnapshot:
    """Capture one exact rollback point for all authorized group paths."""
    return WorktreeRollbackSnapshot(
        ref=state.stable_ref,
        index=git_ops.snapshot_index(work.repo),
        path_states=git_ops.snapshot_worktree_paths(
            work.repo, state.footprint.run_allowed_paths
        ),
        untracked=git_ops.snapshot_untracked_paths(
            work.repo, include_runtime_artifacts=False
        ),
    )


def _enforce_footprint(
    ctx: FlowContext,
    state: FixCycleState,
    *,
    phase: str,
    round_number: int | None,
) -> ScopeEnforcementResult:
    """Restore unauthorized state and return the enforced footprint result."""
    return enforce_authorized_fix_footprint(
        ctx.work,
        state.stable_ref,
        state.footprint,
        preexisting_untracked=state.preexisting_untracked,
        preexisting_gitlinks=state.preexisting_gitlinks,
        phase=phase,
        round_number=round_number,
        file_scope_issues=_resolve_opt_in(ctx.config, "scope_issue_filing"),
        auth=ctx.github_execution.auth,
    )


def _strict_scope_and_scrub(
    ctx: FlowContext,
    state: FixCycleState,
    *,
    phase: str,
    round_number: int | None,
) -> bool:
    """Apply the run-wide guard and quote scrub, returning whether bytes changed."""
    deep_data = ctx.deep_data()

    before = state.capture_key()
    generated_restores: list[str] = []
    for path in git_ops.changed_paths_z(ctx.work.repo, state.stable_ref):
        if path.startswith(".daydream/") or path == REVIEW_OUTPUT_FILE:
            continue
        try:
            baseline = git_ops.show(ctx.work.repo, state.stable_ref, path)
        except git_ops.GitError:
            absolute = ctx.work.repo / path
            try:
                current = absolute.read_bytes()
            except OSError:
                current = None
            if current is not None and is_generated_file(path, current):
                state.footprint.authorize_new_generated(
                    ctx.work.repo,
                    path,
                    phase=phase,
                    round_number=round_number,
                    reason="new generated output approved by generated-file policy",
                )
            continue
        if is_generated_file(path, baseline):
            generated_restores.append(path)
            generated_restores.extend(related_manifest_paths(path))
    if generated_restores:
        restore_paths = sorted(set(generated_restores))
        git_ops.restore_worktree_paths_from_ref(
            ctx.work.repo, state.stable_ref, restore_paths
        )
        for path in restore_paths:
            state.footprint.record_git_event(
                action="restore",
                path=path,
                origin="guard",
                phase=phase,
                round_number=round_number,
                reason="restored an edit to existing generated output or its manifest",
            )
        atomic_write_json(
            DeepArtifact.GENERATED_FILE_VIOLATIONS.at(deep_data["dd"]),
            {
                "session_id": state.session_id,
                "violations": restore_paths,
                "phase": phase,
                "round_number": round_number,
            },
            sort_keys=True,
        )
    enforced = _enforce_footprint(ctx, state, phase=phase, round_number=round_number)
    scrub_smart_quotes_changed_files(
        ctx.work.repo,
        sorted(enforced.retained_paths),
        pre_fix_ref=state.stable_ref,
    )
    after = state.capture_key()
    return bool(generated_restores) or enforced.mutated or before != after


def _enforce_terminal_confinement(
    ctx: FlowContext,
    state: FixCycleState,
    *,
    phase: str,
    round_number: int | None,
) -> str | None:
    """Restore all out-of-run/protected state and durably audit a failed exit."""
    try:
        _enforce_footprint(ctx, state, phase=phase, round_number=round_number)
        key = EvidenceKey(
            state.capture_key(),
            state.footprint.policy_revision,
        )
        _write_footprint_audit(ctx, state, key)
    except Exception as exc:
        return str(exc)
    return None
