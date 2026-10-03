"""Publish for review and fix phases."""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import daydream
from daydream import agent, git_ops, ui
from daydream.archive.git_safe import normalize_remote_url
from daydream.deep.artifacts import (
    DeepArtifact,
    deep_dir,
)
from daydream.deep.evidence_reuse import (
    EVIDENCE_REUSE_FORMAT,
    ReuseDecision,
    ReuseTarget,
    audit_payload,
    decide_reuse,
)
from daydream.git_ops import GitError
from daydream.json_utils import atomic_write_json, read_json_object
from daydream.phases.test_evidence import (
    TestAttemptEvidence,
    _canonical_test_cmd,
    _recipe_command,
    _run_host_test_command,
    reuse_target,
)
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    TestExecutionIdentity,
    TestRecipe,
)
from daydream.trajectory import (
    DaydreamPhase,
    host_phase_scope,
)
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.deep.fix_state import FixCycleState


def require_empty_staged_index(work: WorkContext) -> git_ops.IndexSnapshot:
    """Capture the pre-run index and reject any staged entry before mutation."""
    snapshot = git_ops.snapshot_index(work.repo)
    if snapshot.paths:
        raise GitError(
            "Cannot start the fix cycle with staged changes. Unstage or commit "
            "them first; Daydream did not modify the worktree."
        )
    return snapshot


def _stage_retained_once(
    work: WorkContext,
    *,
    retained_paths: frozenset[str],
    retained_states: tuple[git_ops.GitPathState, ...],
    initial_index: git_ops.IndexSnapshot,
) -> tuple[git_ops.GitPathState, ...]:
    """Validate the untouched index, then stage and validate retained paths once."""
    if initial_index.paths:
        raise GitError("Fix-cycle preflight index was not empty")
    current_index = git_ops.snapshot_index(work.repo)
    if current_index != initial_index:
        raise GitError("Index changed during the fix cycle; refusing to stage")
    expected_states = tuple(sorted(retained_states, key=lambda state: os.fsencode(state.path)))
    if frozenset(state.path for state in expected_states) != retained_paths:
        raise GitError("Retained path set does not match retained tree states")
    current_states = git_ops.snapshot_worktree_paths(work.repo, retained_paths)
    if current_states != expected_states:
        raise GitError("Worktree changed after retained-tree verification")
    if not retained_paths:
        return ()

    git_ops.stage_paths(work.repo, [Path(path) for path in sorted(retained_paths)])
    staged_index = git_ops.snapshot_index(work.repo)
    if frozenset(staged_index.paths) != retained_paths:
        raise GitError("Staged index path set does not match the retained path set")
    staged_states = git_ops.snapshot_index_paths(work.repo, retained_paths)
    # Git regular-file modes retain only the owner's executable bit. Keep
    # exact filesystem modes in retained/test/post-hook evidence and project
    # only this index comparison; staging does not chmod the worktree.
    expected_index_states = tuple(
        replace(state, mode=0o100755 if state.mode & 0o100 else 0o100644)
        if state.state == "regular" and state.mode is not None else state
        for state in expected_states
    )
    if staged_states != expected_index_states:
        raise GitError("Staged index content does not match the retained tree")
    return staged_states


def _verify_strict_commit_and_worktree(
    work: WorkContext,
    *,
    sha_before: str,
    retained_paths: frozenset[str],
    staged_states: tuple[git_ops.GitPathState, ...],
    precommit_paths: frozenset[str],
    precommit_states: tuple[git_ops.GitPathState, ...],
) -> None:
    """Fail closed when commit hooks alter the commit, index, or worktree."""
    committed_paths = frozenset(
        git_ops.diff_name_only_strict(work.repo, sha_before, "HEAD"),
    )
    if committed_paths != retained_paths:
        raise GitError("Committed path set does not match the validated staged index")
    if git_ops.snapshot_commit_paths(work.repo, "HEAD", retained_paths) != staged_states:
        raise GitError("Committed content does not match the validated staged index")

    post_index = git_ops.snapshot_index(work.repo)
    if post_index.paths:
        raise GitError("Commit hooks left staged index changes")
    post_changed = frozenset(
        git_ops.changed_paths_z(work.repo, "HEAD", include_runtime_artifacts=False)
    )
    universe = precommit_paths | post_changed
    expected_by_path = {state.path: state for state in precommit_states}
    expected = tuple(
        expected_by_path.get(
            path,
            git_ops.GitPathState(path=path, state="missing", mode=None, digest=None),
        )
        for path in sorted(universe, key=os.fsencode)
    )
    actual = git_ops.snapshot_worktree_paths(work.repo, universe)
    if actual != expected:
        raise GitError("Commit or validation hooks changed the worktree")


def _persist_reuse_audit(
    work: WorkContext, gate: str, record: dict[str, Any]
) -> None:
    """Write one gate's reuse record, merging it into any prior gates.

    Fail-soft: an unwritable artifact directory warns and never fails a commit
    or push. The merge preserves an earlier gate's decision (Pattern B).
    """
    try:
        path = DeepArtifact.EVIDENCE_REUSE.at(deep_dir(work.repo, allow_standalone=True))
        existing = read_json_object(path)
        gates = existing.get("gates")
        merged: dict[str, Any] = dict(gates) if isinstance(gates, dict) else {}
        merged[gate] = record
        atomic_write_json(
            path,
            {"format_version": EVIDENCE_REUSE_FORMAT, "gates": merged},
            indent=2,
            sort_keys=True,
            trailing_newline=True,
        )
    except Exception as exc:  # artifact write is never load-bearing
        ui.print_warning(agent.console, f"Evidence-reuse audit could not be written: {exc}")


def _report_reuse_decision(
    work: WorkContext,
    gate: str,
    decision: ReuseDecision,
    identity: TestExecutionIdentity | None,
    target: ReuseTarget,
) -> None:
    """Display reused/revalidated status with its deciding component and persist the audit."""
    if decision.reused:
        ui.print_success(
            agent.console,
            f"Evidence reuse ({gate}): reused matching evidence — "
            "no redundant suite run",
        )
    else:
        named = ", ".join(decision.mismatched_components) or decision.result
        ui.print_info(
            agent.console,
            f"Evidence reuse ({gate}): ran real validation "
            f"({decision.result}: {named})",
        )
    record = audit_payload(decision, identity, target)
    record["gate"] = gate
    _persist_reuse_audit(work, gate, record)


def _pre_push_reuse_decision(
    work: WorkContext,
    recipe: TestRecipe | None,
    evidence: TestAttemptEvidence | None,
    retained_tree_key: str | None,
) -> ReuseDecision | None:
    """Consider reuse only after strict post-commit retained-tree verification.

    That proof permits ignoring moved HEAD/branch. Missing recipe or usable evidence
    returns None and keeps the proactive test run.
    """
    if recipe is None or evidence is None:
        return None
    identity = evidence.identity
    if identity is None or retained_tree_key is None:
        return None
    target = reuse_target(
        work,
        recipe,
        session_id=identity.session_id,
        retained_tree_key=retained_tree_key,
        post_commit_verified=True,
    )
    if target is None:
        return None
    decision = decide_reuse(identity, target)
    _report_reuse_decision(work, "pre-push", decision, identity, target)
    return decision


async def _validate_declined_fixes(
    work: WorkContext,
    config: Any,
    *,
    recipe: TestRecipe | None = None,
    evidence: TestAttemptEvidence | None = None,
    retained_tree_key: str | None = None,
) -> None:
    """Validate fixes left uncommitted, using matching green evidence or a real host test.

    Compare revision identity because no commit moved it. Recipes supply command/cwd
    without re-resolution; otherwise resolve the canonical command. Red validation
    raises; no configured command leaves the decline without a fabricated verdict.
    """
    cmd = _canonical_test_cmd(config) if recipe is None else _recipe_command(recipe)
    if evidence is not None and evidence.identity is not None and retained_tree_key is not None:
        target = reuse_target(
            work,
            recipe,
            session_id=evidence.identity.session_id,
            retained_tree_key=retained_tree_key,
        )
        decision = None if target is None else decide_reuse(evidence.identity, target)
        if target is not None and decision is not None:
            _report_reuse_decision(
                work, "declined-commit", decision, evidence.identity, target
            )
        if decision is not None and decision.reused:
            return
    if cmd is None:
        return
    result = await _run_host_test_command(cmd, work, config, recipe=recipe)
    if result.passed:
        ui.print_success(
            agent.console,
            "Commit declined; post-fix validation passed (changes left uncommitted)",
        )
        return
    ui.print_error(
        agent.console,
        "Commit declined but post-fix validation failed",
        "The applied fixes did not pass the configured test command; the run "
        "cannot be reported as successful.",
    )
    raise RuntimeError(
        "Post-fix validation failed after the commit/push gate was declined: "
        "the configured test command exited non-zero."
    )


@dataclass(frozen=True)
class PushReceipt:
    """Exact identity of one ordinary push attempt."""

    remote: str
    branch: str
    sha: str
    pushed_repository: str | None


class PushAttemptError(GitError):
    """A push or its exact remote-ref verification failed."""

    def __init__(self, message: str, *, receipt: PushReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


@bind_resolved_run_context
async def phase_commit_push(
    session: "FixCycleState",
    *,
    items: list[dict[str, Any]] | None = None,
    run_context: RunContext | None = None,
) -> PushReceipt | None:
    """Gate and publish the retained tree through normal hooks and exact-state checks.

    The accepted session owns its verified candidate and initial empty index.
    Optional findings shape the deterministic commit message. Current target and
    matching test evidence are checked before any offer or publication is consumed.

    Declining leaves fixes uncommitted but still requires successful validation;
    validation errors propagate to the orchestrator's failed commit step. A push
    succeeds only when its captured branch, HEAD, and remote remain unchanged and
    the remote reports that exact commit. PushAttemptError retains the attempted
    receipt for the caller's audit when pushing or verification fails.
    """
    run_context = resolve_run_context(run_context)
    agent.console.print()
    ui.print_info(agent.console, "Committing and pushing changes...")
    candidate = session.candidate
    if candidate is None:
        raise ValueError("repair session has no verified retained tree")
    snapshot = candidate.snapshot
    evidence = candidate.test
    retained_tree_key = snapshot.tree_key
    if (
        candidate.key.tree_key != retained_tree_key
        or session.capture_key() != retained_tree_key
        or candidate.key.policy_revision != session.footprint.policy_revision
    ):
        raise ValueError("repair target changed after retained-tree verification")
    if evidence is not None and (
        evidence.session_id != session.session_id
        or evidence.input_tree_key != retained_tree_key
        or evidence.output_tree_key != retained_tree_key
        or (evidence.identity is not None and evidence.identity.output_tree_key != retained_tree_key)
    ):
        raise ValueError("test evidence does not match the verified retained tree")

    # Resolve both interaction axes here for every commit path: --yes commits,
    # unattended defaults decline, otherwise prompt with a decline default.
    decision = run_context.confirm(
        safe_default=False,
        question="Commit and push changes? [y/N]",
        default="n",
        console=agent.console,
    )
    if not decision:
        ui.print_dim(agent.console, "Skipping commit and push")
        # Declining commit still runs host tests; failures stop the run.
        await _validate_declined_fixes(
            session.work,
            session.config,
            recipe=session.recipe,
            evidence=evidence,
            retained_tree_key=retained_tree_key,
        )
        return None

    sha_before = git_ops.head_sha(session.work.repo)
    # Runtime output may change during host phases. Tracked and retained paths
    # remain protected even when their names belong to the runtime namespace.
    precommit_paths = snapshot.paths | frozenset(
        git_ops.changed_paths_z(session.work.repo, sha_before, include_runtime_artifacts=False)
    )
    precommit_states = git_ops.snapshot_worktree_paths(session.work.repo, precommit_paths)
    staged_states = _stage_retained_once(
        session.work,
        retained_paths=snapshot.paths,
        retained_states=snapshot.states,
        initial_index=session.initial_index,
    )
    if not staged_states:
        ui.print_info(agent.console, "Nothing to commit — no daydream changes")
        return None

    def _verify_strict(checked: str) -> None:
        try:
            _verify_strict_commit_and_worktree(
                session.work,
                sha_before=sha_before,
                retained_paths=snapshot.paths,
                staged_states=staged_states,
                precommit_paths=precommit_paths,
                precommit_states=precommit_states,
            )
        except GitError as exc:
            local_sha = git_ops.head_sha(session.work.repo)
            raise GitError(
                f"Local commit {local_sha} was created, but {checked} validation "
                f"failed; push blocked: {exc}"
            ) from exc

    message = build_commit_message(
        items=items or [], run_id=session.work.run_id, version=daydream.__version__,
    )
    # Issue #726 task 12: the commit is its own trajectory phase, so the
    # manifest can time it and tell it apart from test/hook/push phases.
    async with host_phase_scope(DaydreamPhase.COMMIT):
        git_ops.commit_staged(session.work.repo, message)
    _verify_strict("post-commit")

    # Reuse may skip Daydream's proactive test run after strict verification.
    # The repository's actual pre-push hook always runs during push_branch.
    if git_ops.has_executable_pre_push_hook(session.work.repo):
        cmd = _canonical_test_cmd(session.config) if session.recipe is None else _recipe_command(session.recipe)
        if cmd is None:
            ui.print_warning(
                agent.console,
                "Pre-push hook detected but no canonical test command is "
                "configured; the push-time suite run is skipped (set "
                "--test-command or the `test_command` config key).",
            )
        else:
            reuse = _pre_push_reuse_decision(
                session.work, session.recipe, evidence, retained_tree_key
            )
            if reuse is None or not reuse.reused:
                # Record the hook's own phase in addition to test-execution events.
                async with host_phase_scope(DaydreamPhase.HOOK_RUN):
                    result = await _run_host_test_command(cmd, session.work, session.config, recipe=session.recipe)
                if not result.passed:
                    raise RuntimeError(
                        "Pre-push validation failed: the configured test command "
                        "exited non-zero, so the commit was not pushed."
                    )

            _verify_strict("post-hook")

    # The push + remote verification is its own trajectory phase
    # (issue #726 task 12).
    async with host_phase_scope(DaydreamPhase.PUSH):
        remote = "origin"
        branch = git_ops.current_branch(session.work.repo)
        if branch is None:
            raise GitError(
                f"Cannot push: {session.work.repo} is in a detached-HEAD state "
                "with no current branch"
            )
        sha = git_ops.head_sha(session.work.repo)
        raw_remote = git_ops.remote_url(session.work.repo, remote)
        pushed_repository = None
        if raw_remote is not None:
            normalized_repository = normalize_remote_url(raw_remote)[0]
            if normalized_repository is not None:
                pushed_repository = normalized_repository.lower()
        attempted = PushReceipt(
            remote=remote,
            branch=branch,
            sha=sha,
            pushed_repository=pushed_repository,
        )
        try:
            git_ops.push_branch(session.work.repo, branch, remote=remote)
            if (
                git_ops.current_branch(session.work.repo) != branch
                or git_ops.head_sha(session.work.repo) != sha
                or git_ops.remote_url(session.work.repo, remote) != raw_remote
            ):
                raise GitError(
                    "Push verification failed: local branch, HEAD, or configured "
                    "remote URL changed during the push"
                )
            # Success requires the remote to actually hold the exact SHA
            # captured before push.
            if not git_ops.remote_contains_commit(
                session.work.repo, branch, sha, remote=remote
            ):
                raise GitError(
                    f"Push verification failed: remote {remote!r} does not report "
                    f"refs/heads/{branch} at {sha} after the push"
                )
        except GitError as exc:
            raise PushAttemptError(str(exc), receipt=attempted) from exc

    ui.print_success(agent.console, "Changes pushed; verifying remote CI...")
    return attempted


def build_commit_message(
    *,
    items: list[dict[str, Any]],
    run_id: str,
    version: str,
) -> str:
    """Build a deterministic conventional subject, finding list, and run/version trailers.

    No I/O; the dominant type and concise summary form a subject under 72 characters.
    """
    summary = "apply automated review fixes"
    if items:
        first = str(items[0].get("description") or "").strip()
        if first:
            summary = first[0].lower() + first[1:]

    subject = f"fix: {summary}"
    if len(subject) >= 72:
        subject = subject[:71].rstrip()

    lines = [subject, ""]
    for item in sorted(items, key=lambda i: (str(i.get("file", "")), str(i.get("description", "")))):
        file = str(item.get("file", "")).strip()
        desc = str(item.get("description", "")).strip()
        if file and desc:
            lines.append(f"- {file}: {desc}")
        elif desc:
            lines.append(f"- {desc}")
        elif file:
            lines.append(f"- {file}")
    lines.append("")
    lines.append(f"Daydream-Run: {run_id}")
    lines.append(f"Daydream-Version: {version}")
    return "\n".join(lines)
