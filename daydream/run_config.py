"""Run policy and model/backend precedence, independent of flow execution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from daydream.config import DEEP_PHASE_DEFAULT_EFFORT, PHASE_DEFAULT_EFFORT, PHASE_DEFAULT_MODELS
from daydream.config_file import DaydreamFileConfig
from daydream.deep.latency import LatencyRoute, ProfileResolution, resolve_latency_profile
from daydream.exploration import ExplorationContext
from daydream.observability.config import ObservabilityConfig
from daydream.review_profile import ResolvedProfile

OutputMode = Literal["loop", "comment", "review", "diagram"]


@dataclass
class RunConfig:
    """Configuration for a run. Phase overrides take precedence over file defaults; see README for CLI fields."""

    target: str | None = None
    observability: ObservabilityConfig | None = None
    stack: str | None = None  # "python", "react", "elixir", "go", "rust", "ios"
    cleanup: bool | None = None
    quiet: bool = True
    start_at: str = "review"
    pr_number: int | None = None
    approved_head_sha: str | None = None
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    # CLI > file > balanced; unknown names resolve upward to forensic and are recorded.
    latency_profile: str | None = None
    # Resolved once at the composition root (the deep preamble) and consumed by
    # the effort resolver; ``None`` until then.
    latency_route: LatencyRoute | None = None
    file_config: DaydreamFileConfig | None = None
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    review_model: str | None = None
    parse_model: str | None = None
    fix_model: str | None = None
    test_model: str | None = None
    exploration_context: ExplorationContext | None = None
    exploration_model: str | None = None
    ignore_paths: list[str] = field(default_factory=list)
    trajectory_path: Path | None = None
    pr_repo: str | None = None
    archive: bool = True
    run_eval: bool = True
    dataset_capture: bool = False
    dataset_store_path: Path | None = None

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
    log_mode: bool = False  # --verbose: redacted plain-text diagnostics bypass Rich UI
    identity: str = "unknown"  # resolved GitHub identity; set once by run()
    # CLI > file > DEFAULT_SHALLOW_FANOUT_THRESHOLD; zero disables the tiny-diff gate.
    shallow_fanout_threshold: int | None = None
    # Suppress borderline arbiter findings unless the skeptical agent confirms them.
    precision_mode: bool = False
    # Opt in to posting APPROVE for deep reviews with zero high/medium findings.
    approve_on_clean: bool = False
    # Opt in at CLI/file tier to filing pre-fix findings and post-fix reverted edits.
    scope_issue_filing: bool = False
    flow_name: str | None = None
    improve_effort: str = "standard"
    improve_focus: str | None = None
    improve_scope: str | None = None
    improve_plan_description: str | None = None
    improve_prune_name: str | None = None
    # CLI > file > defaults for dependency-aware per-language sharding; disabled by default.
    deep_shard_enabled: bool | None = None
    deep_shard_max_files: int | None = None
    deep_shard_max_bytes: int | None = None
    deep_shard_fanout_cap: int | None = None
    deep_shard_frontier_max: int | None = None
    # CLI > file > defaults; explicit False disables reuse without changing retention bounds.
    review_cache_enabled: bool | None = None
    review_cache_max_entries: int | None = None
    review_cache_max_bytes: int | None = None
    review_cache_max_age_days: int | None = None
    # File-config-only controls: risk categories widen selection and validate before verification.
    verify_all: bool | None = None
    verify_extra_risk_categories: list[str] | None = None
    # Resolve the CLI/env path once into a validated profile with source and digest.
    review_profile_path: str | Path | None = None
    review_profile: ResolvedProfile | None = None
    # None preserves file precedence, including a repository's explicit diagram mode=off.
    diagram: str | None = None
    test_command: str | None = None
    # Explicit suite declarations for the single test command; empty falls back to file policy.
    # resolve_test_recipe never invents a suite or creates a second runner.
    test_required_suites: list[str] = field(default_factory=list)


def _file_config_or_empty(config: RunConfig) -> DaydreamFileConfig:
    """Treat absent file_config as an empty policy for every resolver."""
    return config.file_config if config.file_config is not None else DaydreamFileConfig()


def _configured_phase_value(config: RunConfig, phase: str, setting: str) -> str | None:
    """Resolve CLI global > file phase > file global; empty strings are unset."""
    file_config = _file_config_or_empty(config)
    return (
        getattr(config, setting)
        or file_config.phases.get(phase, {}).get(setting)
        or getattr(file_config, setting)
    )


def _resolved_backend_name(config: RunConfig, phase: str) -> str:
    """Explicit per-phase backend > configured tiers > Claude fallback."""
    return getattr(config, f"{phase}_backend", None) or _configured_phase_value(config, phase, "backend") or "claude"


def _default_backend_name(config: RunConfig) -> str:
    """Resolve the general backend: CLI global, file global, then claude.

    Never inspect phase overrides: archive identity and the default-backend display
    must remain distinct from review-specific provenance.
    """
    file_config = _file_config_or_empty(config)
    return config.backend or file_config.backend or "claude"


def _resolved_review_backend_name(config: RunConfig) -> str | None:
    """Resolve an explicit review backend for archival, preserving None when unset.

    Consult CLI review_backend then the file review phase directly. The general
    resolver would let a CLI global mask this override and blur archive provenance.
    """
    file_config = _file_config_or_empty(config)
    return config.review_backend or file_config.phase_backend("review")


def _resolved_model(config: RunConfig, phase: str) -> str | None:
    """Explicit per-phase model > configured tiers > the resolved backend's table."""
    return (
        getattr(config, f"{phase}_model", None)
        or _configured_phase_value(config, phase, "model")
        or PHASE_DEFAULT_MODELS.get(_resolved_backend_name(config, phase), {}).get(phase)
    )


def _resolved_latency_profile(config: RunConfig) -> ProfileResolution:
    """Resolve CLI then file latency policy, promoting unknown names to forensic.

    Return the source (cli/file/default) and unknown spelling without raising.
    """
    if config.latency_profile is not None:
        return resolve_latency_profile(config.latency_profile, source="cli")
    file_config = _file_config_or_empty(config)
    if file_config.latency_profile is not None:
        return resolve_latency_profile(file_config.latency_profile, source="file")
    return resolve_latency_profile(None, source="default")


def _explicit_reasoning_effort_pin(config: RunConfig, phase: str) -> str | None:
    """Only explicit tiers; exclude latency routing and built-in defaults."""
    return _configured_phase_value(config, phase, "reasoning_effort")


def _profile_phase_effort(config: RunConfig, backend_name: str, phase: str) -> str | None:
    """Resolve wonder/arbiter route effort below explicit pins and above table defaults.

    Apply only to backends in the deep effort table; wonder=skip falls through.
    """
    if phase not in ("wonder", "arbiter") or backend_name not in DEEP_PHASE_DEFAULT_EFFORT:
        return None
    route = config.latency_route
    if route is None:
        return None
    if phase == "wonder":
        return None if route.wonder == "skip" else route.wonder
    return route.arbiter_effort


def _resolved_reasoning_effort(config: RunConfig, phase: str) -> str | None:
    """Resolve effort: CLI global > file phase > file global > route > backend default.

    Latency routes apply to wonder/arbiter; there is no per-phase RunConfig field.
    The default table uses the phase's resolved backend. None leaves the backend's
    ambient default intact, such as Codex's model_reasoning_effort config.
    """
    backend_name = _resolved_backend_name(config, phase)
    return (
        _explicit_reasoning_effort_pin(config, phase)
        or _profile_phase_effort(config, backend_name, phase)
        or PHASE_DEFAULT_EFFORT.get(backend_name, {}).get(phase)
    )


DEEP_FLOW_ALIASES = ("review", "shallow", "deep")
