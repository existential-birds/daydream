"""Main orchestration logic for the review and fix loop.

The runner is unified around a single :func:`run` entry point. ``run`` opens
the workspace via :func:`daydream.workspace.open_workspace` and then dispatches
to the single deep flow, which now carries every PR-process mode (#330)::

    bot set (feedback mode)   -> deep feedback mode
    flow_name set (--flow):
        "review" / "shallow"  -> deep review / shallow mode
        "deep"                -> deep (default)
        "improve"             -> _run_improve    (repo-wide read-only advisor)
        other registered      -> _run_custom_flow (fork-registered custom flow)
    output_mode == "comment"  -> deep comment mode (posts inline, no fix cycle)
    output_mode == "review"   -> deep review mode (report only, no fix cycle)
    output_mode == "loop":
        config.shallow        -> deep shallow mode (single-stack deep)
        else                  -> deep (default)

``run_feedback`` is the entry point used by the ``daydream feedback <pr#>``
subcommand and is a thin wrapper that sets ``pr_number`` and re-enters
:func:`run`.

``run`` builds the per-run extension registry (builtins + optional
``daydream_ext``) and sets it on the registry ContextVar before dispatch;
the ``deep`` flow's preambles stay in :func:`daydream.deep.orchestrator.run_deep`
and run the phase sequence through :func:`daydream.flows.run_flow` against the
registered flow definition.
"""

import os
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rich.markup import escape as escape_markup

from daydream import git_ops, github_app
from daydream.agent import (
    console,
    set_assume,
    set_log_mode,
    set_non_interactive,
    set_quiet_mode,
)
from daydream.backends import Backend, create_backend
from daydream.config import EFFORT_TIERS, PHASE_DEFAULT_EFFORT, PHASE_DEFAULT_MODELS
from daydream.config_file import DaydreamFileConfig
from daydream.exploration import ExplorationContext
from daydream.extensions import ExtensionError, build_registry, get_registry, set_registry
from daydream.flows import FlowContext, run_flow
from daydream.git_ops import GitError
from daydream.phases import (
    _detect_default_branch,
    _git_branch,
    _git_log,
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
    print_phase_hero,
    print_success,
    prompt_user,
)
from daydream.workspace import WorkContext, open_workspace

if TYPE_CHECKING:
    from daydream.pr_review import ParsedIssue
    from daydream.service.artifact import WorkerArtifactV1
    from daydream.service.models import ReviewJobV1

# Output mode: ``loop`` runs review→fix→test; ``comment`` posts inline PR
# comments and exits; ``review`` writes a report and exits.
OutputMode = Literal["loop", "comment", "review"]


@dataclass
class RunConfig:
    """Configuration for a daydream run.

    Attributes:
        target: Target directory path for the review. If None, prompts user.
        skill: Review skill to use ("python", "react", "elixir", "go", "rust",
            or "ios"). If None and shallow, prompts user.
        cleanup: Remove review output file after completion. If None, prompts user.
        quiet: Suppress verbose output from the agent.
        start_at: Phase to start at ("review", "fix", "ttt", "per-stack", or
            "merge"). parse/test are legacy shallow-loop stages with no mapping
            in the unified pipeline and are rejected at the CLI.
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
            (CLI > config-file phase > config-file global). Every backend
            applies it through its native knob (Codex as ``-c
            model_reasoning_effort=...``, Claude as ``--effort``, Pi as
            ``--thinking``). Default None.
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
        trajectory_hub_repo: HuggingFace dataset repo id (``owner/repo``) that each
            run's archive bundle is uploaded to, keyed by session id. Opt-in via
            ``--trajectory-hub-repo`` / ``DAYDREAM_TRAJECTORY_HUB_REPO`` /
            ``trajectory_hub_repo`` file config; requires ``HF_TOKEN``. Default None
            (feature off).
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
        approve_on_clean: Opt-in approval of clean deep reviews (issue #343). When
            True, a deep review with zero high/medium findings posts
            ``event: "APPROVE"`` (with a prepended approval line) instead of the
            default ``event: "COMMENT"``, satisfying a repo's
            ``required_approving_review_count`` without a human. ``False`` falls
            through to ``file_config.approve_on_clean`` then the built-in default
            ``False`` (precedence CLI > file > default, mirroring
            ``precision_mode``; resolved by ``_approve_on_clean``), so the posted
            event stays COMMENT unless a repo explicitly opts in.
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
        skill_availability: Stack keys with an installed Beagle review skill,
            resolved once by :func:`run` from ``get_installed_skills()``. ``None``
            means unresolved or registry-unreadable (→ optimistic routing in
            ``detect_stacks``). Set explicitly to inject availability and bypass
            the probe (tests).
        uncovered_sweep: Issue #309. Toggle the uncovered-diff-file sweep (the
            second-pass reviewer over diff files no per-stack reviewer read).
            ``None`` falls through to ``file_config.uncovered_sweep`` then the
            built-in default ``True`` (precedence CLI > file > default,
            mirroring ``shallow_fanout_threshold``; resolved by
            ``_uncovered_sweep_enabled``).
        uncovered_sweep_max_files: Issue #309. Cap on how many uncovered files
            are swept in one run; files beyond the cap are recorded in
            ``coverage-stats.json`` as ``sweep_skipped_capacity`` rather than
            silently dropped. ``None`` falls through to file config then
            ``DEFAULT_UNCOVERED_SWEEP_MAX_FILES`` (10).
        uncovered_sweep_min_hunk_lines: Issue #309. A file counts as sweepable
            only when its hunks contain at least this many added/removed lines.
            ``None`` falls through to file config then
            ``DEFAULT_UNCOVERED_SWEEP_MIN_HUNK_LINES`` (5).

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
    trajectory_hub_repo: str | None = None
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
    # Issue #343: opt-in approval of clean deep reviews. When True, a deep review
    # with zero high/medium findings posts event: "APPROVE" (prepended approval
    # line) instead of event: "COMMENT", satisfying a repo's
    # required_approving_review_count without a human. Default False =>
    # byte-identical behavior; the posted event stays COMMENT unless a repo
    # explicitly opts in.
    approve_on_clean: bool = False
    flow_name: str | None = None
    improve_effort: str = "standard"
    improve_focus: str | None = None
    improve_scope: str | None = None
    improve_plan_description: str | None = None
    skill_availability: frozenset[str] | None = None
    # Issue #309: uncovered-diff-file sweep (second-pass reviewer). CLI-tier
    # overrides; ``None`` falls through to the file-config scalar then the
    # orchestrator default (True / DEFAULT_UNCOVERED_SWEEP_MAX_FILES /
    # DEFAULT_UNCOVERED_SWEEP_MIN_HUNK_LINES).
    uncovered_sweep: bool | None = None
    uncovered_sweep_max_files: int | None = None
    uncovered_sweep_min_hunk_lines: int | None = None


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
            upload=status != "partial",
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


def _run_posts_to_github(config: RunConfig) -> bool:
    """Return whether the deep single flow (as selected by ``config``) can write to GitHub.

    This mirrors the mode dispatch's write-capable paths: feedback mode may
    reply to PR comments; an explicit ``--flow deep`` and the default
    non-shallow loop both execute deep's ``post-review`` step; and ``--comment``
    posts inline comments. ``--review``, shallow mode, and generic custom flows
    are report-only from the runner's perspective, so they retain the ambient
    ``gh`` identity. A custom flow that gains a GitHub write must explicitly
    add its dispatch contract here before it can use App credentials.
    """
    if config.bot is not None:
        return True

    if config.flow_name is not None:
        return config.flow_name == "deep"

    if config.output_mode == "comment":
        return True

    return config.output_mode == "loop" and not config.shallow


# Public entry points


async def run(config: RunConfig | None = None) -> int:
    """Execute a daydream run end-to-end.

    Opens the workspace via :func:`open_workspace` and dispatches to the single
    deep flow based on ``config.bot`` / ``config.output_mode`` / ``config.shallow``
    (feedback / review / comment / shallow modes, #330). Centralising workspace
    lifecycle means every flow gets a real :class:`WorkContext` (in-place or
    ephemeral) with consistent base/branch resolution.

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

    # Resolve installed-skill availability once, here at the composition root, so
    # the orchestrators consume it as data instead of each probing the filesystem.
    # None (unreadable registry) flows straight through to detect_stacks' optimistic
    # default. An explicitly injected value is kept (the probe is skipped).
    if config.skill_availability is None:
        from daydream.deep.orchestrator import get_installed_skills

        installed = get_installed_skills()
        config.skill_availability = frozenset(installed) if installed is not None else None

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
    # subprocess when the selected flow posts; every hard-abort case surfaces as
    # GitHubAppError. Read-only flows deliberately preserve the ambient identity.
    _flow_is_review = config.flow_name == "review"
    is_posting = _run_posts_to_github(config)
    try:
        identity = github_app.resolve_run_identity(target_dir, config.pr_repo, is_posting=is_posting)
    except github_app.GitHubAppError as exc:
        print_error(console, "GitHub App", str(exc))
        return 1
    config.identity = identity

    # ``--comment``/``--review`` (and ``--flow review``) stop after post-review,
    # so they skip the test phase, hence the .env copy too.
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

    ``review`` / ``shallow`` / ``deep`` route to the single deep flow (as
    review / shallow / default mode, #330) — these are not registered flow
    names, so they are resolved before the registry lookup. ``improve`` runs
    its own flow. Any other registered name runs the generic
    :func:`_run_custom_flow`; unknown names raise
    ``UnresolvedExtensionError``, which propagates to :func:`run`'s Extension
    Error panel.
    """
    name = config.flow_name
    assert name is not None

    # Built-in mode aliases: resolve before the registry lookup so the deep
    # routing wins over the "not registered" error.
    if name in ("review", "shallow", "deep"):
        if name in ("shallow", "deep"):
            _require_reviewable_branch(work, config)
        return await _run_loop_deep(work, config)
    if name == "improve":
        return await _run_improve(work, config)

    # Resolve-check first; unknown names raise UnresolvedExtensionError, caught
    # by run()'s Extension Error panel (exit 1). Do not swallow it here.
    get_registry().flow(name)
    return await _run_custom_flow(work, config)


async def _dispatch(work: WorkContext, config: RunConfig) -> int:
    """Route the resolved workspace + config to the single deep flow.

    Every PR-process mode routes to :func:`_run_loop_deep` (which delegates to
    :func:`daydream.deep.orchestrator.run_deep`): feedback mode (``bot`` set by
    the ``daydream feedback <pr#>`` subcommand) runs the feedback prefix,
    ``--review`` / ``--comment`` run the review spine and stop after
    post-review, ``--shallow`` forces single-stack mode, and the default loop
    mode is unchanged. An explicit ``flow_name`` (``--flow``) routes via
    :func:`_dispatch_selected_flow`.

    Note: ``config.pr_number`` can be auto-detected from the current branch
    for metadata (trajectory/archive) without implying feedback mode.

    Args:
        config: Run configuration (``config.identity`` carries the resolved
            GitHub identity set by :func:`run`).
    """
    if config.bot is not None:
        return await _run_loop_deep(work, config)

    if config.flow_name is not None:
        return await _dispatch_selected_flow(work, config)

    if config.output_mode in ("comment", "review"):
        return await _run_loop_deep(work, config)

    # output_mode == "loop" (default deep) and --shallow both fix against a
    # base branch, so both must refuse to review the base branch against
    # itself (the guard was shared by loop + shallow pre-collapse, #330).
    _require_reviewable_branch(work, config)
    return await _run_loop_deep(work, config)


def _emit_findings_from_items(
    target_dir: Path, config: RunConfig, items: list[dict[str, Any]],
) -> int:
    """Write the Phase A findings artifact from canonical merged items.

    Converts canonical merged items (``file``/``line`` already resolved) via
    :func:`daydream.pr_review.parsed_issues_from_items` and routes them through
    the shared PR-resolution + build + write path (:func:`_write_findings_for_parsed`).

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

    Resolves the target PR — via
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


# Helper: deep (single-flow dispatch)


async def _run_loop_deep(work: WorkContext, config: RunConfig) -> int:
    """Delegate to the deep-mode orchestrator (the only PR-process flow, #330)."""
    from daydream.deep.orchestrator import run_deep

    return await run_deep(config, work)


async def run_service(
    config: RunConfig, job: "ReviewJobV1", *, lens_inventory: Sequence[str] | None = None
) -> int:
    """Service-mode dispatch hook: run a read-only service review for *job*.

    The entrypoint for ``DAYDREAM_SERVICE_V1`` executor ports' ``start``:
    opens the workspace, detaches the checkout at the job's exact candidate
    SHA (the worker re-verifies every component), resolves the backend through
    the full precedence chain, calls the fail-closed worker, and returns the
    terminal exit code for ``job`` (see
    :func:`daydream.service.worker.terminal_exit_code`).

    The job itself is constructed by the controller leaf from a forge event
    (``REVIEW_TARGET_V1``); this hook only consumes a validated
    :class:`~daydream.service.models.ReviewJobV1`.

    Args:
        config: The resolved run configuration.
        job: The immutable job to run.
        lens_inventory: Every lens the executing environment can dispatch
            (the capability-constrained inventory). A required lens absent
            here fails as ``lens_unavailable`` before the backend ever runs.
            Defaults to ``job.required_lenses`` when the caller has no
            constrained inventory to declare.

    Returns:
        ``0`` for ``clean``/``findings``, ``1`` for ``infra_error`` (or any
        workspace/extension abort), ``2`` for ``cancelled``.
    """
    from daydream.service.worker import run_service_review, terminal_exit_code

    print_phase_hero(console, "DAYDREAM SERVICE", phase_subtitle("SERVICE REVIEW"))
    set_quiet_mode(config.quiet)
    set_non_interactive(not _resolve_interactive(config))
    set_assume(config.assume)
    set_log_mode(config.log_mode)

    try:
        registry = build_registry()
        set_registry(registry)
    except ExtensionError as exc:
        print_error(console, "Extension Error", str(exc))
        return 1

    if config.target is not None:
        target_dir = Path(config.target).resolve()
    else:
        target_dir = Path(".").resolve()
    if not target_dir.is_dir():
        print_error(console, "Invalid Path", f"'{target_dir}' is not a valid directory")
        return 1

    try:
        git_ops.assert_is_worktree(target_dir)
    except git_ops.GitError as exc:
        print_error(console, "Workspace Error", str(exc))
        return 1

    # Normalize the checkout to a detached HEAD at the exact candidate SHA; the
    # worker re-verifies the SHA, tree, diff digest, and pristine-ness.
    try:
        if git_ops.current_branch(target_dir) is not None or git_ops.head_sha(
            target_dir
        ) != job.target.candidate_sha:
            git_ops.checkout_detach(target_dir, job.target.candidate_sha)
    except git_ops.GitError as exc:
        print_error(
            console,
            "Workspace Error",
            f"cannot detach checkout at {job.target.candidate_sha}: {exc}",
        )
        return 1

    backend = _resolve_backend(config, "review", cwd=target_dir)
    artifact = await run_service_review(
        target_dir,
        job,
        backend,
        lens_inventory=job.required_lenses if lens_inventory is None else lens_inventory,
    )
    _print_service_outcome(artifact)
    return terminal_exit_code(artifact)


def _print_service_outcome(artifact: "WorkerArtifactV1") -> None:
    """Render the worker artifact terminal for the service-mode hook."""
    if artifact.terminal in ("clean", "findings"):
        print_success(
            console,
            f"Service review {artifact.terminal}: {len(artifact.completed_lenses)} lens(es) "
            f"completed, {len(artifact.findings)} finding(s).",
        )
    elif artifact.terminal == "cancelled":
        print_error(console, "Service Review", "cancelled")
    else:
        print_error(
            console,
            "Service Review",
            f"infra_error: {artifact.process_outcome} "
            f"(missing lenses: {', '.join(artifact.missing_lenses) or 'none'})",
        )
