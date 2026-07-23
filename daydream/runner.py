"""Main orchestration logic for the review and fix loop.

The runner is unified around a single :func:`run` entry point. ``run`` opens
the workspace via :func:`daydream.workspace.open_workspace` and then dispatches
to a private helper based on ``config.bot`` / ``config.flow_name`` /
``config.output_mode`` / ``config.shallow``::

    bot set (feedback mode)  -> _run_pr_feedback (registered "pr-feedback" flow)
    flow_name set (--flow)   -> _dispatch_selected_flow:
        "review"             -> _run_review     (registered "review" flow, report only, no posting)
        "shallow"            -> _run_loop_shallow (registered "shallow" flow)
        "deep"               -> _run_loop_deep  (deep multi-stack pipeline)
        "improve"            -> _run_improve    (repo-wide read-only advisor)
        other registered     -> _run_custom_flow (fork-registered custom flow)
    output_mode == "comment" -> _run_comment    (registered "review" flow, posts inline PR comments)
    output_mode == "review"  -> _run_review     (registered "review" flow, report only, no posting)
    output_mode == "loop":
        config.shallow       -> _run_loop_shallow (registered "shallow" flow, single-stack review-fix-test)
        else                 -> _run_loop_deep    (deep multi-stack pipeline, default)

``run_feedback`` is the entry point used by the ``daydream feedback <pr#>``
subcommand and is a thin wrapper that sets ``pr_number`` and re-enters
:func:`run`.

``run`` builds the per-run extension registry (builtins + optional
``daydream_ext``) and sets it on the registry ContextVar before dispatch;
migrated flows (pr-feedback, review/comment, shallow) keep their preamble in
the flow helper and run their phase sequence through
:func:`daydream.flows.run_flow` against the registered flow definition.
"""

import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rich.markup import escape as escape_markup

from daydream import git_ops, github_app
from daydream.agent import (
    console,
    get_assume,
    get_non_interactive,
    resolve_or_prompt,
    set_assume,
    set_log_mode,
    set_non_interactive,
    set_quiet_mode,
)
from daydream.backends import Backend, create_backend
from daydream.config import (
    EFFORT_TIERS,
    PHASE_DEFAULT_EFFORT,
    PHASE_DEFAULT_MODELS,
    REVIEW_SKILLS,
    ReviewSkillChoice,
)
from daydream.config_file import DaydreamFileConfig
from daydream.exploration import ExplorationContext
from daydream.extensions import ExtensionError, build_registry, get_registry, set_registry
from daydream.flows import FlowContext, run_flow
from daydream.git_ops import GitError
from daydream.phases import (
    _detect_default_branch,
    _git_branch,
    _git_log,
    check_review_file_exists,
)
from daydream.trajectory import (
    DaydreamRunFlow,
    TrajectoryRecorder,
    default_trajectory_path,
)
from daydream.ui import (
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
    print_menu,
    print_phase_hero,
    print_skipped_phases,
    print_success,
    print_warning,
    prompt_user,
)
from daydream.workspace import WorkContext, open_workspace

if TYPE_CHECKING:
    from daydream.pr_review import ParsedIssue

# Output mode: ``loop`` runs review→fix→test; ``comment`` posts inline PR
# comments and exits; ``review`` writes a report and exits.
OutputMode = Literal["loop", "comment", "review"]


# Shallow reviews emit ``confidence``; map it onto the canonical ``severity``
# axis so the shallow fix loop orders findings like the deep pipeline.
_CONFIDENCE_TO_SEVERITY = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}


def to_canonical_shallow(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag shallow parse items with the canonical ``lens``/``severity`` axes.

    Sets ``lens="per-stack"`` on every item and derives ``severity`` from the
    item's ``confidence`` (HIGH→high, MEDIUM→medium, LOW→low), defaulting to
    ``"medium"`` when ``confidence`` is absent. Items are mutated in place and
    returned for convenience.
    """
    for item in items:
        item["lens"] = "per-stack"
        confidence = item.get("confidence") or ""
        item["severity"] = _CONFIDENCE_TO_SEVERITY.get(confidence, "medium")
    return items


@dataclass
class RunConfig:
    """Configuration for a daydream run.

    Attributes:
        target: Target directory path for the review. If None, prompts user.
        skill: Review skill to use ("python", "react", "elixir", "go", "rust",
            or "ios"). If None and shallow, prompts user.
        cleanup: Remove review output file after completion. If None, prompts user.
        quiet: Suppress verbose output from the agent.
        start_at: Phase to start at ("review", "parse", "fix", "test", "ttt",
            "per-stack", or "merge").
        pr_number: GitHub PR number for PR feedback mode. If None, normal mode.
        bot: Bot username whose comments to fetch (e.g. "coderabbitai[bot]").
        backend: Default backend to use ("claude" or "codex"). Default is None;
            ``_resolve_backend`` falls back through the config file to ``"claude"``.
        model: Global default model applied across phases when no explicit
            per-phase model is set. Resolved by ``_resolved_model`` below the
            per-phase field but above the config-file (phase then global) and
            table sources. Default None.
        reasoning_effort: Global default reasoning-effort override (e.g. "low",
            "medium", "high"), resolved by ``_resolved_reasoning_effort``
            (CLI > config-file phase > config-file global). Only the Codex
            backend applies it (forwarded as ``-c model_reasoning_effort=...``);
            ignored for claude/pi. Default None.
        file_config: File-sourced configuration (``[tool.daydream]`` /
            ``.daydream.toml``) feeding ``_resolved_model`` / ``_resolve_backend``
            as a low-precedence source. None is treated as an empty config.
        review_backend: Override backend for the review phase. If None, uses backend.
        fix_backend: Override backend for the fix phase. If None, uses backend.
        test_backend: Override backend for the test phase. If None, uses backend.
        review_model: Model override for the review phase. When None, the
            resolver falls back to ``PHASE_DEFAULT_MODELS[backend_name]["review"]``
            and then to the backend's own default.
        parse_model: Model override for the parse phase. When None, resolves via
            ``PHASE_DEFAULT_MODELS[backend_name]["parse"]`` then backend default.
        fix_model: Model override for the fix phase. When None, resolves via
            ``PHASE_DEFAULT_MODELS[backend_name]["fix"]`` then backend default.
        test_model: Model override for the test phase. When None, resolves via
            ``PHASE_DEFAULT_MODELS[backend_name]["test"]`` then backend default.
        loop: Enable continuous review/fix/test iterations. Default is False.
        max_iterations: Maximum number of loop iterations before exiting. Default is 5.
        exploration_model: Model override for exploration subagents. When set, a separate
            backend is created for the exploration phase using this model. Defaults to
            :data:`config.DEFAULT_EXPLORATION_MODEL`.
        ignore_paths: Paths to exclude from diffs (passed to `git :(exclude)` pathspecs
            and surfaced in review prompts). Default is an empty list.
        trajectory_path: Path to write the ATIF v1.7 trajectory JSON. Default-resolved
            by run flows to ``<target>/.daydream/runs/<session_id>/trajectory.json``
            when None.
        pr_repo: GitHub repository in ``owner/repo`` format. Auto-detected from ``gh``
            in deep (default) mode. Stored in trajectory metadata for eval linkage.
        archive: Archive run artifacts to centralized store. Default True.
        run_eval: Run deterministic evaluation on archived artifacts. Default True
            (``analyze_session`` is file-based and cheap); ``--no-eval`` opts out.
        branch: Specific branch to review. If None, uses cwd's HEAD.
        base: Base ref to compare against. If None, auto-resolves.
        output_mode: ``"loop"`` (review→fix→test, default), ``"comment"``
            (review + post inline PR comments), or ``"review"`` (review report only).
        findings_out: Path to write the Phase A findings artifact
            (``--findings-out``; review mode only). Default None.
        dump_artifacts: Directory to copy the full assembled run bundle into
            (trajectory, review-output, deep artifacts, diffs, findings, manifest,
            evaluation) so CI can upload it. Opt-in via ``--dump-artifacts`` because
            the logs may contain sensitive data. Default None.
        force_worktree: Force ephemeral worktree even when ``branch`` is None.
        shallow: Single-stack review (skip multi-stack auto-detection).
        extra_copy: Extra paths to copy into ephemeral worktrees.
        non_interactive: Run without prompting; take each prompt's safe default
            without reading stdin.
        assume: Forced yes/no answer for interactive gates — ``"yes"`` (``--yes``),
            ``"no"``, or ``None``. Orthogonal to ``non_interactive``: it supplies a
            pre-decided answer rather than controlling stdin access.
        shallow_fanout_threshold: Max changed-file count that triggers the
            tiny-diff short-circuit in deep mode (issue #172). ``None`` falls
            through to ``file_config.shallow_fanout_threshold`` then
            ``DEFAULT_SHALLOW_FANOUT_THRESHOLD`` (precedence CLI > file > default,
            mirroring ``_resolve_backend``). ``0`` disables the short-circuit.
        precision_mode: Opt-in precision suppression (issue #232). When True, the
            deep pipeline runs a skeptical LLM second opinion over borderline
            (LOW-confidence / low-severity uncontested) findings after the arbiter
            and drops any it cannot confirm (fail-closed). ``False`` falls through
            to ``file_config.precision_mode`` then the built-in default ``False``
            (precedence CLI > file > default, mirroring
            ``shallow_fanout_threshold``; resolved by ``_precision_mode``), so the
            suppression pass never runs and arbiter output is byte-identical.
        flow_name: Name of a registered flow to dispatch (``--flow``); built-in
            names route to their dedicated helper, other registered names to the
            generic custom-flow runner.
        improve_effort: Improve audit *breadth* tier (quick, standard, or deep),
            resolved to an ``EFFORT_TIERS`` entry that selects categories, audit
            fanout concurrency, confidence filtering, and finding caps. It does
            not select the model or reasoning effort — those are per-phase
            (``PHASE_DEFAULT_MODELS`` / ``PHASE_DEFAULT_EFFORT``).
        improve_focus: Optional improve focus mode.
        improve_scope: Optional service name/root/glob to audit.
        improve_plan_description: One-line request for ``daydream improve plan``;
            switches the flow to single-request investigation mode.

    """

    target: str | None = None
    skill: str | None = None  # "python", "react", "elixir", "go", "rust", "ios"
    cleanup: bool | None = None
    quiet: bool = True
    start_at: str = "review"
    pr_number: int | None = None
    bot: str | None = None
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    file_config: DaydreamFileConfig | None = None
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    review_model: str | None = None
    parse_model: str | None = None
    fix_model: str | None = None
    test_model: str | None = None
    loop: bool = False
    max_iterations: int = 5
    exploration_context: ExplorationContext | None = None
    exploration_depth: int = 1
    exploration_model: str | None = None
    ignore_paths: list[str] = field(default_factory=list)
    trajectory_path: Path | None = None
    pr_repo: str | None = None
    archive: bool = True
    run_eval: bool = True

    branch: str | None = None
    base: str | None = None
    output_mode: OutputMode = "loop"
    findings_out: str | None = None
    dump_artifacts: str | None = None
    force_worktree: bool = False
    shallow: bool = False
    extra_copy: list[Path] = field(default_factory=list)
    non_interactive: bool = False
    assume: str | None = None  # forced gate answer: "yes" (--yes), "no", or None
    log_mode: bool = False  # bypass Rich UI and emit plain text to stdout
    identity: str = "unknown"  # resolved GitHub identity; set once by run()
    # Issue #172: tiny-diff short-circuit gate (max changed files). CLI-tier
    # override; falls through to file-config scalar then the orchestrator
    # default (DEFAULT_SHALLOW_FANOUT_THRESHOLD). ``0`` disables the gate.
    shallow_fanout_threshold: int | None = None
    # Issue #232: opt-in precision mode. When True, the deep pipeline runs a
    # skeptical suppression pass over borderline (LOW-confidence / low-severity
    # uncontested) findings after the arbiter, dropping any the suppression agent
    # cannot confirm (fail-closed). Default False => byte-identical behavior; the
    # suppression predicate is never called and arbiter output is unchanged.
    precision_mode: bool = False
    flow_name: str | None = None
    improve_effort: str = "standard"
    improve_focus: str | None = None
    improve_scope: str | None = None
    improve_plan_description: str | None = None


def _print_missing_skill_error(skill_name: str) -> None:
    """Print error message for missing skill with installation instructions."""
    print_error(console, "Missing Skill", f"Skill '{skill_name}' is not available")

    if skill_name.startswith("beagle"):
        print_info(console, "The Beagle plugin is required but not installed or enabled.")
        console.print()
        print_dim(console, "To install Beagle:")
        print_dim(console, "  1. Open Claude Code in your terminal")
        print_dim(console, "  2. Run: /install-plugin beagle@existential-birds")
        print_dim(console, "  3. Restart Claude Code")
        console.print()
        print_dim(console, "Or enable it manually in ~/.claude/settings.json:")
        print_dim(console, '  "enabledPlugins": {')
        print_dim(console, '    "beagle@existential-birds": true')
        print_dim(console, "  }")
    else:
        print_info(console, f"The plugin providing '{skill_name}' is not installed.")
        print_dim(console, "Check your ~/.claude/settings.json for enabled plugins.")

    console.print()


def _make_archive_callback(
    config: RunConfig, target_dir: Path, work: WorkContext | None = None,
) -> Callable[[TrajectoryRecorder, str], None] | None:
    """Build the on_write archive callback, or None if archiving is disabled.

    ``--dump-artifacts`` reuses the same bundle assembly, so the callback also
    fires (to build and copy out the bundle) when a dump target is set even if
    the centralized archive is disabled.
    """
    if not config.archive and not config.dump_artifacts:
        return None

    def _cb(recorder: TrajectoryRecorder, status: str) -> None:
        from daydream.archive import archive_run

        archive_run(
            recorder=recorder,
            target_dir=target_dir,
            config=config,
            status=status,
            run_eval=config.run_eval,
            work=work,
        )

    return _cb


def _open_recorder(
    *,
    config: RunConfig,
    target_dir: Path,
    work: WorkContext | None,
    flow_kind: DaydreamRunFlow,
) -> TrajectoryRecorder:
    """Construct the run's ``TrajectoryRecorder`` with archival + dump wired in.

    The single construction site for every flow's recorder. Centralizing it here
    guarantees that centralized archival AND ``--dump-artifacts`` apply to every
    flow — the four built-ins today and any future custom/extension flow tomorrow.
    New flows MUST open their recorder through this factory rather than
    constructing ``TrajectoryRecorder`` directly, so the dump/archive callback can
    never be silently dropped. Session id and trajectory path are resolved here
    identically for all flows.
    """
    session_id = str(uuid.uuid4())
    trajectory_path = config.trajectory_path or default_trajectory_path(target_dir, session_id)
    return TrajectoryRecorder(
        path=trajectory_path,
        run_flow=flow_kind,
        target_dir=target_dir,
        agent_model_name="",
        session_id=session_id,
        explicit_path=config.trajectory_path is not None,
        pr_number=config.pr_number,
        pr_repo=config.pr_repo,
        on_write=_make_archive_callback(config, target_dir, work),
    )


def _file_config_or_empty(config: RunConfig) -> DaydreamFileConfig:
    """Return ``config.file_config``, or an empty config when it is None.

    A single accessor so resolution call sites never branch on ``None`` —
    an absent file config behaves identically to one with no keys set.
    """
    return config.file_config if config.file_config is not None else DaydreamFileConfig()


def _resolved_backend_name(config: RunConfig, phase: str) -> str:
    """Resolve the backend kind for ``phase`` across all precedence tiers.

    Order (highest first): explicit per-phase ``{phase}_backend``, global
    ``config.backend`` (``--backend``), file-config phase override, file-config
    global, then the terminal ``"claude"`` fallback.
    """
    file_config = _file_config_or_empty(config)
    return (
        getattr(config, f"{phase}_backend", None)
        or config.backend
        or file_config.phase_backend(phase)
        or file_config.backend
        or "claude"
    )


def _resolved_model(config: RunConfig, phase: str) -> str | None:
    """Resolve the model for ``phase`` across all precedence tiers.

    Order (highest first): explicit per-phase ``{phase}_model``, global
    ``config.model`` (``--model``), file-config phase override, file-config
    global, then ``PHASE_DEFAULT_MODELS[backend][phase]``. Returns ``None``
    only when no source supplies a model (the backend then applies its own
    default).

    The per-backend table lookup keys off the backend kind resolved by
    :func:`_resolved_backend_name`, so a config-selected backend still gets its
    own phase tier defaults.
    """
    file_config = _file_config_or_empty(config)
    backend_name = _resolved_backend_name(config, phase)
    return (
        getattr(config, f"{phase}_model", None)
        or config.model
        or file_config.phase_model(phase)
        or file_config.model
        or PHASE_DEFAULT_MODELS.get(backend_name, {}).get(phase)
    )


def _resolved_reasoning_effort(config: RunConfig, phase: str) -> str | None:
    """Resolve the reasoning effort for ``phase`` across all precedence tiers.

    Order (highest first): global ``config.reasoning_effort``
    (``--reasoning-effort``), file-config phase override, file-config global,
    then ``PHASE_DEFAULT_EFFORT[backend][phase]``. There is no per-phase
    RunConfig field. ``None`` means no source supplied one and the backend
    applies its own ambient default (e.g. Codex reads
    ``model_reasoning_effort`` from ``~/.codex/config.toml`` when daydream
    passes nothing).

    Like :func:`_resolved_model`, the default-table lookup keys off the backend
    kind resolved by :func:`_resolved_backend_name`. All three backends consume
    the resolved value through their own native knob.
    """
    file_config = _file_config_or_empty(config)
    backend_name = _resolved_backend_name(config, phase)
    return (
        config.reasoning_effort
        or file_config.phase_reasoning_effort(phase)
        or file_config.reasoning_effort
        or PHASE_DEFAULT_EFFORT.get(backend_name, {}).get(phase)
    )


def _resolve_backend(
    config: RunConfig,
    phase: str,
    cache: dict[tuple[str, str | None, str | None], Backend] | None = None,
    *,
    cwd: Path | None = None,
) -> Backend:
    """Get or create the backend for a given phase, respecting all precedence tiers.

    The backend kind, model, and reasoning effort are each resolved through the
    source-tiered precedence ``CLI > config-file > default``:

    - Backend kind via :func:`_resolved_backend_name`
      (per-phase flag → ``config.backend`` → file-config phase →
      file-config global → ``"claude"``).
    - Model via :func:`_resolved_model`
      (per-phase field → ``config.model`` → file-config phase →
      file-config global → ``PHASE_DEFAULT_MODELS`` →
      ``None``, where ``None`` falls through to the backend's own default).
    - Reasoning effort via :func:`_resolved_reasoning_effort`
      (``config.reasoning_effort`` → file-config phase → file-config global →
      ``PHASE_DEFAULT_EFFORT`` → ``None``, where ``None`` falls through to the
      backend's own default). All three backends apply the resolved value
      through their native knob.

    Args:
        config: Run configuration with backend/model/reasoning-effort and
            file-config sources.
        phase: Phase name (e.g. ``"review"``, ``"parse"``, ``"fix"``, ``"test"``,
            ``"intent"``, ``"wonder"``, ``"merge"``,
            ``"exploration"``, ``"pr_feedback"``).
        cache: Optional dict to cache backends by
            ``(backend_name, model, reasoning_effort)``. When provided,
            backends are reused only when the backend kind, resolved model,
            and resolved reasoning effort all match — so the same backend kind
            with two different models or effort levels yields two distinct
            instances.
        cwd: Target workspace used for backend-specific configuration.
    """
    backend_name = _resolved_backend_name(config, phase)
    resolved_model = _resolved_model(config, phase)
    resolved_effort = _resolved_reasoning_effort(config, phase)

    def _make() -> Backend:
        # ``cwd`` stays pi-only: it exists solely to resolve Pi's configured
        # default model, and widening it churns every patched create_backend.
        if backend_name == "pi":
            return create_backend(
                backend_name, model=resolved_model, cwd=cwd, reasoning_effort=resolved_effort
            )
        return create_backend(
            backend_name, model=resolved_model, reasoning_effort=resolved_effort
        )

    if cache is None:
        return _make()
    cache_key = (backend_name, resolved_model, resolved_effort)
    if cache_key not in cache:
        cache[cache_key] = _make()
    return cache[cache_key]


def _truthy(value: str | None) -> bool:
    """Interpret an environment-variable string as a boolean.

    Returns:
        False for None and for ``""``/``"0"``/``"false"`` (case-insensitive);
        True for any other non-empty value.
    """
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false")


def _stdin_isatty() -> bool:
    """Report whether stdin is an interactive TTY.

    Returns:
        True if stdin is attached to a terminal. A detached or closed stdin
        (raising ``AttributeError``/``ValueError``) is treated as not a TTY.
    """
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _resolve_interactive(config: "RunConfig") -> bool:
    """Resolve whether this run may prompt the user, from three sources.

    Precedence: an explicit ``--non-interactive`` flag forces False; otherwise
    the run is interactive only when stdin is a TTY and ``CI`` is not truthy.

    Returns:
        True if prompts may read stdin; False for unattended/harness runs.
    """
    if config.non_interactive:
        return False
    return _stdin_isatty() and not _truthy(os.environ.get("CI"))


def _compute_diff_ref(cwd: Path) -> str:
    """Compute the diff ref to hand to exploration specialists.

    Returns ``"{base_branch}...HEAD"`` when a default branch is detected, else
    falls back to ``"HEAD"`` so specialists can still run ``git diff HEAD -- <file>``.
    """
    base_branch = _detect_default_branch(cwd)
    if base_branch:
        return f"{base_branch}...HEAD"
    return "HEAD"


def _get_head_sha(cwd: Path) -> str | None:
    """Get the current HEAD commit SHA.

    Returns:
        The full SHA string, or None if the command fails.
    """
    try:
        return git_ops.head_sha(cwd)
    except GitError:
        return None


# Public entry points


async def run(config: RunConfig | None = None) -> int:
    """Execute a daydream run end-to-end.

    Opens the workspace via :func:`open_workspace` and dispatches to the
    appropriate flow helper based on ``config.bot`` / ``config.output_mode``
    / ``config.shallow``. Centralising workspace lifecycle means every flow gets
    a real :class:`WorkContext` (in-place or ephemeral) with consistent
    base/branch resolution.

    Args:
        config: Optional configuration. Defaults to a fresh :class:`RunConfig`
            (interactive prompts for target dir, skill, cleanup).
    """
    if config is None:
        config = RunConfig()

    print_phase_hero(console, "DAYDREAM", phase_subtitle("DAYDREAM"))

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
    set_quiet_mode(quiet)
    # Interactivity (--non-interactive flag, else non-TTY stdin, else CI) and the
    # orthogonal ``assume`` axis (--yes) both feed ``resolve_gate`` at each gate.
    set_non_interactive(not _resolve_interactive(config))
    set_assume(config.assume)
    set_log_mode(config.log_mode)

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
        target_input = prompt_user(console, "Enter target directory", ".")
        target_dir = Path(target_input).resolve()

    if not target_dir.is_dir():
        print_error(console, "Invalid Path", f"'{target_dir}' is not a valid directory")
        return 1

    # Resolve the active GitHub identity once onto config.identity. Under App
    # credentials this also mints + injects the installation token into every ``gh``
    # subprocess; every hard-abort case surfaces as GitHubAppError.
    # ``--flow review`` is equivalent to ``--review``: treat it as posting so the
    # GitHub App token is minted and injected the same way.
    _flow_is_review = config.flow_name == "review"
    is_posting = config.bot is not None or config.output_mode in ("comment", "review") or _flow_is_review
    try:
        identity = github_app.resolve_run_identity(target_dir, config.pr_repo, is_posting=is_posting)
    except github_app.GitHubAppError as exc:
        print_error(console, "GitHub App", str(exc))
        return 1
    config.identity = identity

    # ``--comment``/``--review`` skip the test phase, hence the .env copy too.
    # ``--flow review`` matches this behaviour for parity.
    skip_tests = (
        config.output_mode != "loop"
        or _flow_is_review
        or config.flow_name == "improve"
    )

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
        ) as work:
            return await _dispatch(work, config)
    except git_ops.WrongBranchError:
        # Propagate to ``cli.main`` for the actionable error panel.
        raise
    except git_ops.GitError as exc:
        print_error(console, "Workspace Error", str(exc))
        return 1
    except ExtensionError as exc:
        # ``run_flow``'s pre-flight resolve pass raises ``UnresolvedExtensionError``
        # naming flow + step before any step executes; the flow helpers let it
        # propagate here so every broken-extension abort renders the same panel.
        print_error(console, "Extension Error", str(exc))
        return 1


async def run_feedback(config: RunConfig, pr: int) -> int:
    """Entry point for the ``daydream feedback <pr#>`` subcommand.

    Sets ``config.pr_number`` and re-enters :func:`run` so the dispatch
    routes to :func:`_run_pr_feedback`. Kept as a thin wrapper so cli.py
    has a single named entry point per invocation shape.
    """
    config.pr_number = pr
    return await run(config)


# Dispatch


def _require_reviewable_branch(work: WorkContext, config: RunConfig) -> None:
    """Raise WrongBranchError when a loop run has nothing to review against.

    A worktree on the base branch with no --branch/--worktree would review
    itself. Raised for cli.main() to render the actionable panel. Extracted
    from _dispatch so both the default loop path and --flow deep/shallow reuse
    the identical guard.
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


async def _dispatch_selected_flow(work: WorkContext, config: RunConfig) -> int:
    """Route an explicit ``--flow <name>`` selection.

    Validates registration first (unknown names raise
    ``UnresolvedExtensionError``, which propagates to :func:`run`'s Extension
    Error panel). Built-in names route to their existing dedicated helper so
    behavior matches the default / ``--shallow`` / ``--review`` paths;
    ``pr-feedback`` is not selectable this way (it needs a PR number + bot);
    any other registered name runs the generic :func:`_run_custom_flow`.
    """
    name = config.flow_name
    assert name is not None

    # Resolve-check first; unknown names raise UnresolvedExtensionError, caught
    # by run()'s Extension Error panel (exit 1). Do not swallow it here.
    get_registry().flow(name)

    if name == "pr-feedback":
        print_error(
            console,
            "Flow not selectable",
            "The pr-feedback flow needs a PR number and bot identity; "
            "use: daydream feedback <pr#> --bot <name>.",
        )
        return 1
    if name == "review":
        return await _run_review(work, config)
    if name == "shallow":
        _require_reviewable_branch(work, config)
        return await _run_loop_shallow(work, config)
    if name == "deep":
        _require_reviewable_branch(work, config)
        return await _run_loop_deep(work, config)
    if name == "improve":
        return await _run_improve(work, config)

    return await _run_custom_flow(work, config)


async def _dispatch(work: WorkContext, config: RunConfig) -> int:
    """Pick the flow helper for the resolved workspace + config.

    Order matters: ``bot`` signals PR feedback mode (set only by the
    ``daydream feedback <pr#>`` subcommand). Then an explicit ``flow_name``
    (``--flow``) routes via :func:`_dispatch_selected_flow`. Then output_mode
    picks comment vs review vs loop. Inside loop, ``config.shallow`` selects
    the single-stack pipeline; otherwise the deep multi-stack pipeline runs
    (default).

    Note: ``config.pr_number`` can be auto-detected from the current branch
    for metadata (trajectory/archive) without implying feedback mode.

    Args:
        config: Run configuration (``config.identity`` carries the resolved
            GitHub identity set by :func:`run`).
    """
    if config.bot is not None:
        return await _run_pr_feedback(work, config)

    if config.flow_name is not None:
        return await _dispatch_selected_flow(work, config)

    if config.output_mode == "comment":
        return await _run_comment(work, config)

    if config.output_mode == "review":
        return await _run_review(work, config)

    # output_mode == "loop".
    _require_reviewable_branch(work, config)

    if config.shallow:
        return await _run_loop_shallow(work, config)
    # Default: deep multi-stack pipeline (--shallow opts into single-stack).
    return await _run_loop_deep(work, config)


# Helper: PR feedback flow


async def _run_pr_feedback(work: WorkContext, config: RunConfig) -> int:
    """PR-feedback preamble; the phase sequence runs through the flow registry.

    Validates args, opens the trajectory recorder, prints the info block,
    then delegates to the registered ``pr-feedback`` flow (fetch -> parse ->
    fix -> commit/push -> respond; steps in ``daydream/flows/pr_feedback.py``).
    """
    if config.pr_number is None or config.bot is None:
        print_error(
            console,
            "Invalid PR config",
            "PR number and --bot are required (use: daydream feedback <pr#> --bot <name>).",
        )
        return 1

    pr_number = config.pr_number
    bot = config.bot
    target_dir = work.repo

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=DaydreamRunFlow.PR,
    ):
        ctx = FlowContext(config=config, work=work, registry=get_registry())
        ctx.data["pr_number"] = pr_number
        ctx.data["bot"] = bot

        console.print()
        print_info(console, f"PR feedback mode: PR #{pr_number}")
        print_info(console, f"Bot: {bot}")
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Model: {ctx.backend_for('review').model}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        console.print()

        return await run_flow(ctx.registry, "pr-feedback", ctx)


# Helper: comment mode (--comment)


async def _run_comment(work: WorkContext, config: RunConfig) -> int:
    """Review + post inline PR comments + exit.

    Pre-flight: when a branch is explicitly requested but no open PR exists
    for it, refuse early with an actionable error rather than running a
    review that has nowhere to land.
    """
    if config.branch is not None:
        try:
            prs = git_ops.gh_pr_list_for_branch(work.source, config.branch)
        except GitError as exc:
            print_error(console, "GitHub Error", str(exc))
            return 1
        if not prs:
            print_error(
                console,
                "No Open PR",
                f"no open PR for branch {config.branch} — push first or use --review",
            )
            return 1

    return await _run_review_or_comment(work, config, post_to_pr=True)


# Helper: review mode (--review)


async def _run_review(work: WorkContext, config: RunConfig) -> int:
    """Review + write a report and exit. No PR posting, no fix, no test."""
    return await _run_review_or_comment(work, config, post_to_pr=False)


def _emit_findings_artifact(
    target_dir: Path, config: RunConfig, issues: list[dict[str, Any]],
) -> int:
    """Write the Phase A findings artifact declared by ``--findings-out``.

    Resolves the target PR — via :func:`daydream.pr_review.find_pr_by_number`
    when ``config.pr_number`` is pinned, else :func:`daydream.pr_review.find_open_pr`
    — then classifies the alt-review issues and writes the strict-schema
    artifact. The artifact must declare its target, so an unresolvable PR (or
    a ``GitError`` from the lookup) is an actionable error, never a silently
    absent artifact. An empty issue list still writes an (empty) artifact so
    Phase B can resolve all stale comments.

    Args:
        target_dir: Repo root containing the PR checkout.
        config: Run configuration; ``config.findings_out`` must be set.
        issues: Raw issue dicts from ``phase_alternative_review`` (may be empty).

    Returns:
        ``0`` on success, ``1`` when no PR is resolvable.
    """
    from daydream import pr_review

    parsed = pr_review.alt_issues_to_parsed(issues)
    return _write_findings_for_parsed(target_dir, config, parsed)


def _emit_findings_from_items(
    target_dir: Path, config: RunConfig, items: list[dict[str, Any]],
) -> int:
    """Write the Phase A findings artifact from canonical merged items.

    Sibling of :func:`_emit_findings_artifact` for the deep-review path:
    converts canonical merged items (``file``/``line`` already resolved) via
    :func:`daydream.pr_review.parsed_issues_from_items` and routes them through
    the same shared PR-resolution + build + write path.

    Args:
        target_dir: Repo root containing the PR checkout.
        config: Run configuration; ``config.findings_out`` must be set.
        items: Canonical merged finding dicts (may be empty).

    Returns:
        ``0`` on success, ``1`` when no PR is resolvable.
    """
    from daydream import pr_review

    parsed = pr_review.parsed_issues_from_items(items)
    return _write_findings_for_parsed(target_dir, config, parsed)


def _write_findings_for_parsed(
    target_dir: Path, config: RunConfig, parsed: list["ParsedIssue"],
) -> int:
    """Resolve the target PR and write the strict-schema findings artifact.

    Shared body for :func:`_emit_findings_artifact` and
    :func:`_emit_findings_from_items`. Resolves the target PR — via
    :func:`daydream.pr_review.find_pr_by_number` when ``config.pr_number`` is
    pinned, else :func:`daydream.pr_review.find_open_pr` — then writes the
    artifact. The artifact must declare its target, so an unresolvable PR (or a
    ``GitError`` from the lookup) is an actionable error, never a silently
    absent artifact. An empty ``parsed`` list still writes an (empty) artifact
    so Phase B can resolve all stale comments.

    Returns:
        ``0`` on success, ``1`` when no PR is resolvable.
    """
    from daydream import pr_review
    from daydream.findings import build_findings_artifact, write_findings_artifact

    assert config.findings_out is not None  # caller gates on findings_out
    try:
        if config.pr_number is not None:
            pr = pr_review.find_pr_by_number(target_dir, config.pr_number)
        else:
            pr = pr_review.find_open_pr(target_dir)
    except GitError as exc:
        print_error(console, "Findings Artifact", f"cannot resolve target PR: {exc}")
        return 1
    if pr is None:
        print_error(
            console,
            "Findings Artifact",
            "no PR resolvable for --findings-out — the artifact must declare its "
            "target (pass --pr-number or open a PR for this branch)",
        )
        return 1

    artifact = build_findings_artifact(
        target_dir, pr, parsed, run_info=pr_review._render_review_info_block(),
    )
    out_path = Path(config.findings_out)
    write_findings_artifact(out_path, artifact)
    print_success(console, f"Findings artifact written to {out_path}")
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


async def _run_review_or_comment(
    work: WorkContext, config: RunConfig, *, post_to_pr: bool,
) -> int:
    """Shared preamble for ``--comment`` and ``--review``.

    Gathers git context, writes the diff file, opens the trajectory recorder,
    prints the info block, then delegates to the registered ``review`` flow
    (exploration -> intent -> alternatives -> emit-findings -> no-issues-exit
    -> post-comments; steps in ``daydream/flows/review.py``).
    ``ctx.data["post_to_pr"]`` carries the mode: only ``--comment`` posts the
    alternative-review issues to the PR via
    :func:`daydream.pr_review.post_review_to_pr_from_alt_issues`, and only
    ``--review`` honours ``config.findings_out`` (Phase A artifact emission
    via :func:`_emit_findings_artifact`).
    """
    target_dir = work.repo

    # Gather git context using the resolved base branch from work (no
    # double-detection — base resolution is locked at workspace open time).
    diff, log, branch = _gather_diff_seed(work, config)

    if diff is None:
        print_error(console, "Git Error", "Unable to determine base branch for diff")
        return 1
    if not diff.strip():
        print_warning(console, "No diff found — nothing to review")
        return 0

    # Write diff to file to avoid exceeding prompt size limits
    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir(exist_ok=True)
    diff_path = daydream_dir / "diff.patch"
    diff_path.write_text(diff)

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=DaydreamRunFlow.TTT,
    ):
        ctx = FlowContext(config=config, work=work, registry=get_registry())
        ctx.data["post_to_pr"] = post_to_pr
        ctx.data["diff"] = diff
        ctx.data["log"] = log
        ctx.data["branch"] = branch
        ctx.data["daydream_dir"] = daydream_dir
        ctx.data["diff_path"] = diff_path

        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Branch: {branch}")
        print_info(console, f"Model: {ctx.backend_for('review').model}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        console.print()

        return await run_flow(ctx.registry, "review", ctx)


# Helper: generic custom flow (--flow <name>)


async def _run_improve(work: WorkContext, config: RunConfig) -> int:
    """Preamble for the registered repository-wide improve flow."""
    from daydream.improve.artifacts import improve_dir

    target_dir = work.repo
    directory = improve_dir(target_dir)
    tier = EFFORT_TIERS[config.improve_effort]

    async with _open_recorder(
        config=config,
        target_dir=target_dir,
        work=work,
        flow_kind=DaydreamRunFlow.IMPROVE,
    ):
        ctx = FlowContext(config=config, work=work, registry=get_registry())
        ctx.data["improve_dir"] = directory
        ctx.data["effort_tier"] = tier

        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Effort: {config.improve_effort}")
        print_info(console, f"Focus: {config.improve_focus or 'all'}")
        print_info(console, f"Model: {ctx.backend_for('recon').model}")
        print_info(
            console,
            f"GitHub identity: {escape_markup(config.identity)}",
        )
        console.print()

        return await run_flow(ctx.registry, "improve", ctx)


async def _run_custom_flow(work: WorkContext, config: RunConfig) -> int:
    """Generic preamble for a fork-registered flow selected via ``--flow``.

    Mirrors :func:`_run_review_or_comment`'s diff seed so custom flows composed
    of built-in review steps work, but an empty/unavailable diff is a dim note
    rather than an early return (a custom flow may not need a diff). Opens the
    recorder through the shared factory so ``--dump-artifacts``/archival apply.
    """
    flow_name = config.flow_name
    assert flow_name is not None
    target_dir = work.repo

    diff, log, branch = _gather_diff_seed(work, config)

    if not diff:
        print_dim(console, "No diff found — custom flow will run without a diff seed.")
        diff = ""

    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir(exist_ok=True)
    diff_path = daydream_dir / "diff.patch"
    diff_path.write_text(diff)

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=DaydreamRunFlow.CUSTOM,
    ):
        ctx = FlowContext(config=config, work=work, registry=get_registry())
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


# Helper: shallow loop (single-stack review-fix-test)


async def _run_loop_shallow(work: WorkContext, config: RunConfig) -> int:
    """Single-stack review → fix → test preamble.

    Keeps the preamble (skill resolution incl. the ``phase:review`` slot,
    review-file check, cleanup gate, trajectory recorder + info block,
    pre-fix snapshot/HEAD capture) and delegates the phase sequence to the
    registered ``shallow`` flow (exploration -> loop-preflight ->
    [review -> parse -> fix -> test -> commit-iteration]* -> loop-exhausted
    -> summary -> commit-gate; steps in ``daydream/flows/shallow.py``).
    The ``--loop`` iteration is the flow's ``iterate`` loop group, run once
    in single-pass mode.
    """
    target_dir = work.repo

    # Resolve skill only when the review phase will run.
    skill: str | None = None
    if config.start_at == "review":
        if config.skill is not None:
            if (resolved := get_registry().skill_if_registered(f"stack:{config.skill}")) is not None:
                skill = resolved
            elif config.skill in get_registry().skill_slots().values():
                skill = config.skill
            else:
                print_error(console, "Invalid Skill", f"'{config.skill}' is not a valid skill")
                return 1
        elif (slot_skill := get_registry().skill_if_registered("phase:review")) is not None:
            # Precedence: --skill (CLI) > phase:review slot (extension) > prompt/error.
            skill = slot_skill
        else:
            if get_non_interactive():
                print_error(
                    console,
                    "Missing --skill",
                    "Non-interactive mode requires --skill (e.g. --skill python). "
                    "Valid values: python, react, elixir, go, rust, ios.",
                )
                return 1
            console.print()
            print_menu(console, "Select review skill", [
                ("1", "Python/FastAPI backend (review-python)"),
                ("2", "React/TypeScript (review-frontend)"),
                ("3", "Elixir/Phoenix (review-elixir)"),
                ("4", "Go backend (review-go)"),
                ("5", "Rust (review-rust)"),
                ("6", "iOS/SwiftUI (review-ios)"),
            ])

            skill_choice = prompt_user(console, "Choice", "1")

            try:
                skill_enum = ReviewSkillChoice(skill_choice)
            except ValueError:
                print_error(console, "Invalid Choice", f"'{skill_choice}' is not a valid option")
                return 1

            skill = REVIEW_SKILLS[skill_enum]

    # Early validation: check review file exists when starting at parse or fix.
    if config.start_at in ("parse", "fix"):
        try:
            check_review_file_exists(target_dir)
        except FileNotFoundError as e:
            print_error(console, "Missing Review File", str(e))
            return 1

    # Explicit --cleanup/--no-cleanup wins; otherwise gate it (unattended keeps
    # the review output, safe_default=False).
    if config.cleanup is not None:
        cleanup_enabled = config.cleanup
    else:
        cleanup_enabled = resolve_or_prompt(
            assume=get_assume(),
            interactive=not get_non_interactive(),
            safe_default=False,
            question="Cleanup review output after completion? [y/N]",
            default="n",
        )

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=DaydreamRunFlow.NORMAL,
    ):
        ctx = FlowContext(config=config, work=work, registry=get_registry())
        ctx.data["skill"] = skill
        ctx.data["cleanup_enabled"] = cleanup_enabled
        ctx.data["feedback_items"] = []
        ctx.data["fixes_applied"] = 0
        ctx.data["test_retries"] = 0
        ctx.data["tests_passed"] = True
        ctx.data["iteration"] = 0
        ctx.data["diff_base"] = None
        ctx.data["exploration_dir"] = None
        ctx.data["loop_broke"] = False

        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Model: {ctx.backend_for('review').model}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        if skill:
            print_info(console, f"Review skill: {skill}")
        if config.start_at != "review":
            print_skipped_phases(console, config.start_at)
        console.print()

        # Recommended-change patch base: snapshot the tracked tree + HEAD BEFORE
        # any fix runs. stash_create returns None on a clean tree (the common
        # pre-fix case), so the pre-fix HEAD SHA is the fallback base. HEAD is
        # captured now because the commit gate advances it past the fixes.
        # Captured once before the flow so the final file is the cumulative fix
        # against the pre-first-fix base (recommended.patch is overwritten each iteration).
        try:
            ctx.data["pre_fix_snapshot"] = git_ops.stash_create(target_dir)
        except GitError:
            ctx.data["pre_fix_snapshot"] = None
        try:
            ctx.data["pre_fix_head"] = git_ops.head_sha(target_dir)
        except GitError:
            ctx.data["pre_fix_head"] = None

        return await run_flow(ctx.registry, "shallow", ctx)


# Helper: deep loop (multi-stack pipeline)


async def _run_loop_deep(work: WorkContext, config: RunConfig) -> int:
    """Delegate to the deep-mode orchestrator."""
    from daydream.deep.orchestrator import run_deep

    return await run_deep(config, work)
