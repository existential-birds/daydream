"""Workspace, extension registry, and flow dispatch for one run.

``run`` binds the per-run registry and opens the workspace. Explicit ``review``,
``shallow``, and ``deep`` aliases, plus review/comment/diagram/loop output modes,
use ``deep.orchestrator.run_deep``. ``improve`` opens a read-only audit workspace;
other registered flow names use the custom-flow preamble.
"""

import os
import sys
import uuid
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from rich.markup import escape as escape_markup

from daydream import git_ops, github_app, pr_review
from daydream.agent import console
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    OutputLabel,
    PrivateRootLocations,
    PrivateWorkspaceOwner,
    RoutedDestination,
    artifact_dir_for,
    open_artifact_session,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.backends import (
    AUDIT_ROOT_ISOLATION,
    AuditIsolationError,
    Backend,
    BackendExecutionInput,
    create_backend,
)
from daydream.config import (
    DEEP_PHASE_DEFAULT_EFFORT,
    EFFORT_TIERS,
    REVIEW_OUTPUT_FILE,
)
from daydream.extensions import (
    ExtensionError,
    build_registry,
    get_registry,
    set_registry,
)
from daydream.findings import (
    FindingsValidationError,
    build_findings_artifact,
    write_findings_artifact,
)
from daydream.flows import FlowContext, run_flow
from daydream.flows.engine import BackendCache, BackendFactory
from daydream.git_ops import GitError
from daydream.hunk_index import write_hunk_index
from daydream.observability.config import (
    ObservabilityError,
    resolve_observability_config,
)
from daydream.observability.runtime import trace_run
from daydream.phases.inputs import (
    _detect_default_branch,
    _git_branch,
    _git_log,
)
from daydream.run_artifacts import (
    _finalize_run_artifacts,
    _open_recorder,
    _resolve_review_profile,
    _RunArtifacts,
    _RunWriteCapture,
)
from daydream.run_config import (
    DEEP_FLOW_ALIASES,
    RunConfig,
    _file_config_or_empty,
    _resolved_backend_name,
    _resolved_model,
    _resolved_reasoning_effort,
)
from daydream.run_context import InteractionPolicy, RunContext, bind_run_context
from daydream.trajectory import (
    DaydreamRunFlow,
)
from daydream.ui import (
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
    print_phase_hero,
    print_success,
)
from daydream.workspace import (
    AuditWorkspace,
    WorkContext,
    open_audit_workspace,
    open_workspace,
)

if TYPE_CHECKING:
    from daydream.pr_review import ParsedIssue

# Output mode: ``loop`` runs review→fix→test; ``comment`` posts inline PR
# comments and exits; ``review`` writes a report and exits; ``diagram``
# (issue #1113) runs the diagram-only flow and posts a standalone comment.


@dataclass(frozen=True)
class RunnerExecutionInput:
    """Runtime-only backend and GitHub capabilities for an embedded run.

    Kept outside RunConfig and all persisted run metadata. Ordinary callers
    omit this input and retain native environment inheritance.
    """

    backend: BackendExecutionInput = field(repr=False, compare=False)
    github: github_app.GitHubExecutionInput = field(repr=False, compare=False)


def _resolve_backend(
    config: RunConfig,
    phase: str,
    cache: BackendCache | None = None,
    *,
    cwd: Path | None = None,
    audit_workspace: AuditWorkspace | None = None,
    execution_input: BackendExecutionInput | None = None,
    effort_override: str | None = None,
) -> Backend:
    """Resolve and optionally cache a phase backend.

    ``effort_override`` is a per-group arbiter override, gated by the same backend
    support as latency profiles. The resolved effort participates in the cache key,
    so a tuned arbiter cannot reuse the default-effort instance.
    """
    backend_name = _resolved_backend_name(config, phase)
    resolved_model = _resolved_model(config, phase)
    resolved_effort = (
        effort_override
        if effort_override is not None and backend_name in DEEP_PHASE_DEFAULT_EFFORT
        else _resolved_reasoning_effort(config, phase)
    )
    audit_root = (
        audit_workspace.repo.resolve(strict=True)
        if audit_workspace is not None
        else None
    )
    audit_outward_symlinks = (
        audit_workspace.outward_symlinks
        if audit_workspace is not None
        else frozenset()
    )

    def _make() -> Backend:
        # ``cwd`` stays pi-only: it exists solely to resolve Pi's configured
        # default model, and widening it churns every patched create_backend.
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "reasoning_effort": resolved_effort,
            "audit_root": audit_root,
            "audit_outward_symlinks": audit_outward_symlinks,
        }
        if backend_name == "pi":
            kwargs["cwd"] = cwd
        if execution_input is not None:
            kwargs["execution_input"] = execution_input
        return create_backend(backend_name, **kwargs)

    if cache is None:
        return _make()
    cache_key = (backend_name, resolved_model, resolved_effort, audit_root)
    if cache_key not in cache:
        cache[cache_key] = _make()
    return cache[cache_key]


_IMPROVE_MODEL_PHASES: tuple[str, ...] = ("recon", "audit", "vet", "plan_write")


def _preflight_improve_backends(ctx: FlowContext) -> None:
    """Validate every Improve backend inside its bound audit workspace before model calls.

    Reject unsupported direct contexts without an audit workspace.
    """
    audit = ctx.audit_workspace
    if audit is None:
        raise AuditIsolationError("claude", "wrong_root", phase="recon")
    expected_root = audit.repo.resolve(strict=True)
    for phase in _IMPROVE_MODEL_PHASES:
        backend_name = _resolved_backend_name(ctx.config, phase)
        try:
            backend = ctx.backend_for(phase)
        except AuditIsolationError as exc:
            raise AuditIsolationError(
                exc.backend_name,
                exc.reason,
                phase=phase,
            ) from exc
        marker = object()
        capability = getattr(backend, "audit_root_isolation", marker)
        if capability is marker:
            raise AuditIsolationError(
                backend_name,
                "missing_capability",
                phase=phase,
            )
        if capability != AUDIT_ROOT_ISOLATION:
            raise AuditIsolationError(
                backend_name,
                "wrong_capability",
                phase=phase,
            )
        bound_root = getattr(backend, "audit_root", None)
        try:
            resolved_root = (
                bound_root.resolve(strict=True)
                if isinstance(bound_root, Path)
                else None
            )
        except (OSError, RuntimeError, ValueError):
            resolved_root = None
        if resolved_root != expected_root:
            raise AuditIsolationError(
                backend_name,
                "wrong_root",
                phase=phase,
            )


def _truthy(value: str | None) -> bool:
    """Treat None, empty, 0, and false (case-insensitive) as false; other strings as true."""
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false")


def _stdin_isatty() -> bool:
    """Return whether stdin is a TTY; detached or closed stdin is noninteractive."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _resolve_interactive(config: "RunConfig") -> bool:
    """Prompt only with TTY stdin, no truthy CI, and no explicit --non-interactive flag."""
    if config.non_interactive:
        return False
    return _stdin_isatty() and not _truthy(os.environ.get("CI"))


def _compute_diff_ref(cwd: Path) -> str:
    """Use base_branch...HEAD when detected, otherwise HEAD for exploration diffs."""
    base_branch = _detect_default_branch(cwd)
    if base_branch:
        return f"{base_branch}...HEAD"
    return "HEAD"


def _run_posts_to_github(config: RunConfig) -> bool:
    """Return whether this run posts to GitHub."""
    if config.flow_name is not None:
        if config.flow_name == "improve":
            return _file_config_or_empty(config).improve_github_publish_issues
        return config.flow_name == "deep"

    if config.output_mode in ("comment", "diagram"):
        return True

    return config.output_mode == "loop" and not config.shallow


# Public entry points


async def run(
    config: RunConfig | None = None, *, private_roots: PrivateRootLocations | None = None,
    execution: RunnerExecutionInput | None = None,
) -> int:
    """Open a workspace and execute the selected daydream flow."""
    if config is None:
        config = RunConfig()
    if config.dataset_capture_disabled:
        config = replace(config, dataset_capture=False)
    else:
        from daydream.hub import resolve_hub_repo

        hub_repo = resolve_hub_repo(config)
        if hub_repo:
            config = replace(config, dataset_capture=True, trajectory_hub_repo=hub_repo)

    backend_factory: BackendFactory | None = None
    if execution is not None:
        backend_execution = execution.backend

        def create_for_context(
            flow_config: RunConfig, phase: str, cache: BackendCache,
            cwd: Path, audit: AuditWorkspace | None,
        ) -> Backend:
            return _resolve_backend(
                flow_config, phase, cache, cwd=cwd, audit_workspace=audit,
                execution_input=backend_execution,
            )

        backend_factory = create_for_context

    # Codex backends need shell output visible (the commands ARE the signal), so
    # disable quiet when any phase resolves to codex. Done before backend construction.
    quiet = config.quiet
    if quiet:
        codex_in_use = any(
            _resolved_backend_name(config, phase) == "codex"
            for phase in ("review", "fix", "test")
        )
        if codex_in_use:
            quiet = False
    run_context = RunContext(InteractionPolicy(
        assume=config.assume,
        interactive=_resolve_interactive(config),
        quiet=quiet,
        log_mode=config.log_mode,
    ))
    with bind_run_context(run_context):
        return await _run_with_context(
            config, private_roots=private_roots, run_context=run_context,
            backend_factory=backend_factory,
            backend_execution=None if execution is None else execution.backend,
            github_execution=None if execution is None else execution.github,
        )


async def _run_with_context(
    config: RunConfig, *, private_roots: PrivateRootLocations | None,
    run_context: RunContext,
    backend_factory: BackendFactory | None = None,
    backend_execution: BackendExecutionInput | None = None,
    github_execution: github_app.GitHubExecutionInput | None = None,
) -> int:
    """Execute with the runner's immutable policy bound before any output."""
    print_phase_hero(console, "DAYDREAM", phase_subtitle("DAYDREAM"))


    # Build the per-run registry (builtins + optional daydream_ext) and set it
    # on the ContextVar so every downstream phase resolves through it.
    try:
        registry = build_registry()
        file_config = _file_config_or_empty(config)
        if file_config.tool_supervisor == "rules":
            from daydream.supervision import RuleBasedToolSupervisor

            try:
                registry.register_tool_supervisor(
                    RuleBasedToolSupervisor(
                        deny_globs=file_config.supervisor_deny_globs,
                        bash_deny=file_config.tool_bash_deny,
                    )
                )
            except ExtensionError as exc:
                raise ExtensionError(
                    "tool supervisor conflict: config-enabled built-in "
                    "RuleBasedToolSupervisor cannot coexist with an "
                    f"extension-registered tool supervisor ({exc})"
                ) from exc
        set_registry(registry)
    except ExtensionError as exc:
        print_error(console, "Extension Error", str(exc))
        return 1

    # Resolve target dir outside the workspace context so path-validation errors
    # short-circuit before any git work.
    if config.target is not None:
        target_dir = Path(config.target).resolve()
    else:
        target_input = run_context.choice(
            "Enter target directory", default=".", safe_default=".",
        )
        target_dir = Path(target_input).resolve()

    if not target_dir.is_dir():
        print_error(console, "Invalid Path", f"'{target_dir}' is not a valid directory")
        return 1

    try:
        observability = config.observability if config.observability is not None else resolve_observability_config()
        async with trace_run(
            observability, registry, flow=config.flow_name or ("shallow" if config.shallow else "deep"),
        ) as observed:
            observed.attrs({"daydream.output_mode": config.output_mode})
            try:
                locations = private_root_locations() if private_roots is None else private_roots
                private_owner = resolve_private_workspace_owner(target_dir, locations=locations)
            except ArtifactVisibilityError as exc:
                print_error(console, "Artifact Storage", str(exc))
                observed.finish(1)
                return 1

            # Resolve the active GitHub identity only after source ownership
            # succeeds. Ordinary posting flows may acquire an installation
            # token; embedded callers retain their supplied auth capability.
            try:
                if github_execution is None:
                    session = github_app.resolve_run_identity(
                        target_dir, config.pr_repo, is_posting=_run_posts_to_github(config),
                    )
                else:
                    session = github_app.ResolvedGitHubSession(
                        identity=github_app.GitHubIdentity(github_app.resolve_user_identity(
                            target_dir, auth=github_execution.auth,
                        )),
                        execution=github_execution,
                    )
            except github_app.GitHubAppError as exc:
                print_error(console, "GitHub App", str(exc))
                observed.finish(1)
                return 1

            config.identity = session.identity.login
            run_context.github_identity = session.identity

            # Report-only flows skip the test phase and its .env copy.
            skip_tests = config.output_mode != "loop" or config.flow_name in ("review", "improve")
            result = await _run_workspace(
                config, target_dir, skip_tests=skip_tests, private_owner=private_owner,
                run_context=run_context, github_execution=session.execution,
                backend_factory=backend_factory,
                backend_execution=backend_execution,
            )
            observed.finish(result)
            return result
    except ObservabilityError as exc:
        print_error(console, "Observability Error", str(exc))
        return 1


def _absolute_output_path(path: str | Path | None) -> Path | None:
    """Anchor a CLI-supplied relative output path on the invocation directory."""
    if path is None:
        return None
    requested = Path(path)
    return requested if requested.is_absolute() else Path(os.path.abspath(requested))


async def _run_workspace(
    config: RunConfig, target_dir: Path, *, skip_tests: bool,
    private_owner: PrivateWorkspaceOwner, run_context: RunContext,
    github_execution: github_app.GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
    backend_execution: BackendExecutionInput | None = None,
) -> int:
    """Keep workspace errors inside the run span so returned failures are recorded."""
    # ``open_workspace`` runs ``assert_is_worktree`` and surfaces
    # ``NotAWorktreeError`` (a ``GitError``) caught below — a loud error instead of
    # a confusing "no diff found". ``WrongBranchError`` is raised in ``_dispatch``.
    try:
        async with open_workspace(
            source=target_dir,
            branch=config.branch,
            base=config.base,
            force_ephemeral=config.force_worktree,
            extra_copy=config.extra_copy,
            skip_tests=skip_tests,
            allow_unborn=config.flow_name == "improve",
            private_owner=private_owner, auth=github_execution.auth,
        ) as work:
            session_id = str(uuid.uuid4())
            async with open_artifact_session(work, session_id=session_id, owner=private_owner) as artifacts:
                def _route(path: str | Path | None, label: OutputLabel) -> RoutedDestination | None:
                    absolute = _absolute_output_path(path)
                    return None if absolute is None else artifacts.register_destination(absolute, label=label)

                _route(work.source / ".daydream", OutputLabel.PUBLIC_DAYDREAM)
                _route(work.source / REVIEW_OUTPUT_FILE, OutputLabel.PUBLIC_REVIEW_OUTPUT)
                trajectory = artifacts.register_trajectory_output(_absolute_output_path(config.trajectory_path))
                findings = _route(config.findings_out, OutputLabel.FINDINGS_OUTPUT)
                dump = _route(config.dump_artifacts, OutputLabel.DUMP_DIRECTORY)
                capture = _RunWriteCapture(session_id=session_id)
                run_artifacts = _RunArtifacts(
                    artifacts, private_owner, trajectory, capture, dump,
                    execution_input=backend_execution,
                )
                dispatch_config = (
                    config if findings is None else replace(config, findings_out=str(findings.write_path))
                )
                primary: BaseException | None = None
                result = 1
                try:
                    result = await _dispatch(
                        work, dispatch_config, run_artifacts, run_context=run_context,
                        github_execution=github_execution, backend_factory=backend_factory,
                    )
                except BaseException as exc:
                    primary = exc

                selected = (
                    capture.final if capture.validation_error is None and capture.final is not None
                    else capture.partial
                )
                finalization_error: Exception | None = capture.validation_error
                if selected is not None or config.dataset_capture:
                    successful = primary is None and result == 0 and finalization_error is None
                    try:
                        finalize = partial(
                            _finalize_run_artifacts, run_artifacts, selected=selected,
                            config=dispatch_config, work=work, successful=successful,
                            interrupted=primary is not None and not isinstance(primary, Exception),
                        )
                        with anyio.CancelScope(shield=True):
                            await anyio.to_thread.run_sync(finalize)
                    except BaseException as exc:
                        if primary is None:
                            raise
                        primary.add_note(
                            f"artifact finalization retained a secondary base failure ({type(exc).__name__})"
                        )

                if primary is not None:
                    if isinstance(primary, SystemExit) and primary.code == 2 and capture.validation_error is not None:
                        print_error(console, "Artifact Finalization", str(capture.validation_error))
                        return 1
                    if finalization_error is not None:
                        primary.add_note(
                            f"artifact finalization retained a closed failure ({type(finalization_error).__name__})"
                        )
                    raise primary
                if finalization_error is not None:
                    print_error(console, "Artifact Finalization", str(finalization_error))
                    return 1
                return result
    except git_ops.WrongBranchError:
        # Propagate to ``cli.main`` for the actionable error panel.
        raise
    except git_ops.GitError as exc:
        print_error(console, "Workspace Error", str(exc))
        return 1
    except ArtifactVisibilityError as exc:
        print_error(console, "Artifact Storage", str(exc))
        return 1
    except ExtensionError as exc:
        # ``run_flow``'s pre-flight resolve pass raises ``UnresolvedExtensionError``
        # naming flow + step before any step executes; the flow helpers let it
        # propagate here so every broken-extension abort renders the same panel.
        print_error(console, "Extension Error", str(exc))
        return 1


# Dispatch


def _require_reviewable_branch(work: WorkContext, config: RunConfig) -> None:
    """Raise WrongBranchError when a base-branch loop would review itself.

    cli.main renders the same actionable error for default, deep, and shallow flows.
    """
    if (
        config.branch is None
        and not config.force_worktree
        and work.head_branch is not None
        and work.head_branch == work.base_branch
    ):
        raise git_ops.WrongBranchError(
            f"cwd is on the base branch {work.base_branch!r} -- "
            "there's nothing to review against itself.\n"
            "Either:\n"
            f"  - check out a feature branch in this worktree and re-run, or\n"
            f"  - run with --branch <feature-branch> to review the server's version, or\n"
            f"  - run with --worktree to force ephemeral isolation."
        )


# Built-in deep-flow mode aliases (review / shallow / deep all route to the
# single deep flow in different modes, #330). Kept as a module constant so
# downstream consumers (e.g. archive manifest tier gate) share the same list.


async def _dispatch_selected_flow(
    work: WorkContext, config: RunConfig, run_artifacts: _RunArtifacts, *,
    run_context: RunContext,
    github_execution: github_app.GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
) -> int:
    """Resolve built-in aliases before registered flows.

    Review/shallow/deep aliases share the deep flow. Unknown names raise
    ``UnresolvedExtensionError`` for ``run`` to render as an Extension Error.
    """
    name = config.flow_name
    assert name is not None

    # Built-in mode aliases: resolve before the registry lookup so the deep
    # routing wins over the "not registered" error.
    if name in DEEP_FLOW_ALIASES:
        if name in ("shallow", "deep"):
            _require_reviewable_branch(work, config)
        return await _run_loop_deep(
            work, config, run_artifacts, run_context=run_context, github_execution=github_execution,
            backend_factory=backend_factory,
        )
    if name == "improve":
        return await _run_improve(
            work, config, run_artifacts, run_context=run_context, github_execution=github_execution,
            backend_factory=backend_factory,
        )

    # Resolve-check first; unknown names raise UnresolvedExtensionError, caught
    # by run()'s Extension Error panel (exit 1). Do not swallow it here.
    get_registry().flow(name)
    return await _run_custom_flow(
        work, config, run_artifacts, run_context=run_context, github_execution=github_execution,
        backend_factory=backend_factory,
    )


def _verify_approved_head(work: WorkContext, config: RunConfig) -> int:
    """Reject a checkout that has drifted from a maintainer-approved PR head."""
    approved = config.approved_head_sha
    if approved is None or work.head_sha == approved:
        return 0
    print_error(
        console,
        "Head Mismatch",
        f"Approved PR head is {approved}, but the checked-out head is {work.head_sha}. Re-approve the current head.",
    )
    return 1


async def _dispatch(
    work: WorkContext, config: RunConfig, run_artifacts: _RunArtifacts, *,
    run_context: RunContext,
    github_execution: github_app.GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
) -> int:
    """Verify the approved head and dispatch the selected flow."""
    if work.is_unborn:
        if config.flow_name != "improve" or config.approved_head_sha is not None:
            raise GitError("unborn checkout cannot satisfy a commit-anchored review")
        return await _run_improve(
            work, config, run_artifacts, run_context=run_context, github_execution=github_execution,
            backend_factory=backend_factory,
        )
    head_status = _verify_approved_head(work, config)
    if head_status != 0:
        return head_status

    if config.flow_name is not None:
        return await _dispatch_selected_flow(
            work, config, run_artifacts, run_context=run_context, github_execution=github_execution,
            backend_factory=backend_factory,
        )

    if config.output_mode not in ("comment", "review", "diagram"):
        _require_reviewable_branch(work, config)
    return await _run_loop_deep(
        work, config, run_artifacts, run_context=run_context, github_execution=github_execution,
        backend_factory=backend_factory,
    )


def _emit_diagram_findings(
    target_dir: Path, config: RunConfig, payload: dict[str, Any], *,
    run_info: str,
    renderers: "pr_review.ReviewRenderers",
    captured_pr: "pr_review.PRInfo",
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> int:
    """Write a diagram artifact with no findings or mermaid render.

    Phase B re-renders mermaid from the grounded specs in ``payload``.
    """
    return _write_findings_for_parsed(
        target_dir, config, [], kind="diagram", diagrams=payload, auth=auth,
        run_info=run_info, renderers=renderers, captured_pr=captured_pr,
    )


def _emit_findings_from_items(
    target_dir: Path,
    config: RunConfig,
    items: list[dict[str, Any]],
    *,
    diagrams: dict[str, Any] | None = None,
    review_warnings: tuple[str, ...] = (),
    run_info: str,
    renderers: "pr_review.ReviewRenderers",
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
    captured_pr: "pr_review.PRInfo",
    terminal_result: dict[str, Any],
    snapshot_diff: str,
) -> int:
    """Write canonical review items; grounded diagrams ride along for Phase B rendering."""
    parsed = pr_review.parsed_issues_from_items(items)
    return _write_findings_for_parsed(
        target_dir, config, parsed, diagrams=diagrams, auth=auth,
        run_info=run_info, renderers=renderers, review_warnings=review_warnings,
        captured_pr=captured_pr, terminal_result=terminal_result, snapshot_diff=snapshot_diff,
    )


def _write_findings_for_parsed(
    target_dir: Path,
    config: RunConfig,
    parsed: list["ParsedIssue"],
    *,
    kind: str = "review",
    diagrams: dict[str, Any] | None = None,
    review_warnings: tuple[str, ...] = (),
    run_info: str,
    renderers: "pr_review.ReviewRenderers",
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
    captured_pr: "pr_review.PRInfo",
    terminal_result: dict[str, Any] | None = None,
    snapshot_diff: str | None = None,
) -> int:
    """Write the artifact for the PR identity the caller captured at run start.

    The writer never resolves a target itself, so an artifact always declares
    the commit its analysis read. Empty findings still produce an artifact so
    Phase B can clear stale comments.
    """
    assert config.findings_out is not None  # caller gates on findings_out

    artifact = build_findings_artifact(
        target_dir,
        captured_pr,
        parsed,
        run_info=run_info,
        review_warnings=review_warnings,
        renderers=renderers,
        kind=kind,
        diagrams=diagrams,
        auth=auth,
        terminal_result=terminal_result,
        snapshot_diff=snapshot_diff,
    )
    out_path = Path(config.findings_out)
    try:
        write_findings_artifact(out_path, artifact)
    except FindingsValidationError as exc:
        print_error(console, "Findings Artifact", str(exc))
        return 1
    print_success(console, "Findings artifact prepared.")
    return 0


def _gather_diff_seed(work: WorkContext, config: RunConfig) -> tuple[str | None, str, str]:
    """Gather the (diff, log, branch) git seed for a flow preamble.

    The diff is None when the base branch cannot be resolved.
    """
    try:
        diff: str | None = git_ops.diff(work.repo, work.base_branch, exclude=config.ignore_paths)
    except GitError:
        diff = None
    log = _git_log(work.repo)
    branch = work.head_branch or _git_branch(work.repo)
    return diff, log, branch


# Helper: generic custom flow (--flow <name>)


async def _run_improve(
    work: WorkContext, config: RunConfig, run_artifacts: _RunArtifacts, *,
    run_context: RunContext,
    github_execution: github_app.GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
) -> int:
    """Preamble for the registered repository-wide improve flow."""
    from daydream.improve.artifacts import improve_dir

    target_dir = work.repo
    if work.is_unborn and config.improve_focus == "branch":
        print_error(
            console,
            "Unborn Improve Unsupported",
            "--focus branch requires a commit anchor; create the initial commit "
            "or run improve without branch focus.",
        )
        return 1
    if work.is_unborn and config.improve_plan_description is not None:
        print_error(
            console,
            "Unborn Improve Unsupported",
            "improve plan requires a planned-at commit; create the initial commit "
            "before requesting a plan.",
        )
        return 1

    directory = improve_dir(target_dir, session=run_artifacts.session, allow_standalone=False)
    tier = EFFORT_TIERS[config.improve_effort]

    async with _open_recorder(
        config=config,
        target_dir=target_dir,
        work=work,
        flow_kind=DaydreamRunFlow.IMPROVE,
        run_artifacts=run_artifacts,
        allow_standalone=False,
    ):
        _resolve_review_profile(config)
        # The standalone snapshot gives improve independent Git storage. The
        # root-bound backend capability below is the separate filesystem-tool
        # boundary; neither mechanism is described as an OS sandbox.
        branch_base_ref = (
            work.base_branch if config.improve_focus == "branch" else None
        )
        expected_head_sha = work.head_sha if branch_base_ref is not None else None
        async with open_audit_workspace(
            work.repo,
            run_id=work.run_id,
            branch_base_ref=branch_base_ref,
            expected_head_sha=expected_head_sha,
        ) as audit:
            ctx = FlowContext(
                config=config,
                work=work,
                registry=get_registry(),
                review_profile=config.review_profile,
                audit_workspace=audit,
                private_workspace_owner=run_artifacts.owner,
                artifacts=run_artifacts.session,
                run_context=run_context, github_execution=github_execution,
                _backend_factory=backend_factory,
                allow_standalone_artifacts=False,
            )
            ctx.data["audit_repo"] = audit.repo
            ctx.data["improve_dir"] = directory
            ctx.data["effort_tier"] = tier
            ctx.data["improve_publish_issues"] = (
                _file_config_or_empty(config).improve_github_publish_issues
            )
            ctx.data["github_repo"] = config.pr_repo

            try:
                _preflight_improve_backends(ctx)
            except AuditIsolationError as exc:
                phase = exc.phase or "unknown"
                print_error(
                    console,
                    "Improve Audit Isolation",
                    f"Phase {phase!r} selected backend {exc.backend_name!r}, "
                    f"which cannot provide strict audit isolation ({exc.reason}). "
                    "Use backend 'claude' for every improve model phase.",
                )
                return 1

            console.print()
            print_info(console, f"Target directory: {target_dir}")
            print_info(console, f"Effort: {config.improve_effort}")
            print_info(console, f"Focus: {config.improve_focus or 'all'}")
            print_info(
                console,
                "GitHub issue publishing: "
                f"{'enabled' if ctx.data['improve_publish_issues'] else 'disabled'}",
            )
            print_info(console, f"Model: {ctx.backend_for('recon').model}")
            print_info(
                console,
                f"GitHub identity: {escape_markup(config.identity)}",
            )
            console.print()

            return await run_flow(ctx.registry, "improve", ctx)


async def _run_custom_flow(
    work: WorkContext, config: RunConfig, run_artifacts: _RunArtifacts, *,
    run_context: RunContext,
    github_execution: github_app.GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
) -> int:
    """Seed a registered custom flow with diff/log/branch and a shared recorder.

    An unavailable diff becomes an empty seed: custom flows may not require one.
    """
    flow_name = config.flow_name
    assert flow_name is not None
    target_dir = work.repo

    diff, log, branch = _gather_diff_seed(work, config)

    if not diff:
        print_dim(console, "No diff found — custom flow will run without a diff seed.")
        diff = ""

    daydream_dir = artifact_dir_for(target_dir, session=run_artifacts.session, allow_standalone=False)
    daydream_dir.mkdir(exist_ok=True)
    diff_path = daydream_dir / "diff.patch"
    diff_path.write_text(diff)
    # Persist the hunk index alongside diff.patch so custom-flow steps can
    # source changed-line ranges from the run-time authority.
    write_hunk_index(daydream_dir, diff)

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=DaydreamRunFlow.CUSTOM,
        run_artifacts=run_artifacts,
        allow_standalone=False,
    ):
        _resolve_review_profile(config)
        ctx = FlowContext(
            config=config,
            work=work,
            registry=get_registry(),
            review_profile=config.review_profile,
            private_workspace_owner=run_artifacts.owner,
            artifacts=run_artifacts.session,
            run_context=run_context, github_execution=github_execution,
            _backend_factory=backend_factory,
            allow_standalone_artifacts=False,
        )
        ctx.data["post_to_pr"] = False  # custom flows do not post to PR by default
        ctx.data["diff"] = diff
        ctx.data["log"] = log
        ctx.data["branch"] = branch
        ctx.data["daydream_dir"] = daydream_dir
        ctx.data["diff_path"] = diff_path

        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Flow: {flow_name}")
        print_info(console, f"Branch: {branch}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        console.print()

        return await run_flow(ctx.registry, flow_name, ctx)


# Helper: deep (single-flow dispatch)


async def _run_loop_deep(
    work: WorkContext, config: RunConfig, run_artifacts: _RunArtifacts, *,
    run_context: RunContext,
    github_execution: github_app.GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
) -> int:
    """Delegate to the deep-mode orchestrator (the only PR-process flow, #330)."""
    from daydream.deep.orchestrator import run_deep

    _resolve_review_profile(config)
    return await run_deep(
        config, work, run_artifacts=run_artifacts, run_context=run_context, github_execution=github_execution,
        backend_factory=backend_factory,
        allow_standalone=False,
    )
