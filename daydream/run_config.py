"""Run policy and model/backend precedence, independent of flow execution."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from daydream.config import DEFAULT_RETAINED_RUNS, PHASE_DEFAULT_EFFORT, PHASE_DEFAULT_MODELS
from daydream.config_file import DaydreamFileConfig
from daydream.exploration import ExplorationContext
from daydream.observability.config import ObservabilityConfig
from daydream.review_profile import ResolvedProfile

OutputMode = Literal["loop", "comment", "review", "diagram"]


@dataclass(frozen=True)
class FrozenAuthorDescription:
    """Supplied author context bound to a frozen source revision by an embedder."""

    head_sha: str
    description: str

    def for_head(self, head_sha: str | None) -> str:
        """Return the description, refusing an analyzed revision other than the one it was frozen for."""
        if self.head_sha != head_sha:
            raise ValueError("frozen author description does not match analyzed HEAD")
        return self.description


@dataclass
class RunConfig:
    """Configuration for a run. Phase overrides take precedence over file defaults; see README for CLI fields."""

    target: str | None = None
    observability: ObservabilityConfig | None = None
    cleanup: bool | None = None
    quiet: bool = True
    pr_number: int | None = None
    approved_head_sha: str | None = None
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    file_config: DaydreamFileConfig | None = None
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    review_model: str | None = None
    fix_model: str | None = None
    test_model: str | None = None
    exploration_context: ExplorationContext | None = None
    frozen_author_description: FrozenAuthorDescription | None = None
    exploration_model: str | None = None
    ignore_paths: list[str] = field(default_factory=list)
    trajectory_path: Path | None = None
    pr_repo: str | None = None
    # Internal CLI repository/PR discovery request, resolved after session authentication.
    discover_github_provenance: bool = False
    archive: bool = True
    run_eval: bool = True
    dataset_capture: bool = False
    dataset_capture_disabled: bool = False
    dataset_store_path: Path | None = None

    branch: str | None = None
    base: str | None = None
    output_mode: OutputMode = "review"
    findings_out: str | None = None
    dump_artifacts: str | None = None
    trajectory_hub_repo: str | None = None
    force_worktree: bool = False
    extra_copy: list[Path] = field(default_factory=list)
    non_interactive: bool = False
    assume: str | None = None  # forced gate answer: "yes" (--yes), "no", or None
    log_mode: bool = False  # --verbose: redacted plain-text diagnostics bypass Rich UI
    identity: str = "unknown"  # resolved GitHub identity; set once by run()
    # Opt in to posting APPROVE for a complete review with no findings.
    approve_on_clean: bool = False
    # Opt in at CLI/file tier to filing pre-fix findings and post-fix reverted edits.
    scope_issue_filing: bool = False
    flow_name: str | None = None
    improve_effort: str = "standard"
    improve_focus: str | None = None
    improve_scope: str | None = None
    improve_plan_description: str | None = None
    improve_prune_name: str | None = None
    # Resolve the CLI/env path once into a validated profile with source and digest.
    review_profile_path: str | Path | None = None
    review_profile: ResolvedProfile | None = None
    # None preserves file precedence, including a repository's explicit diagram mode=off.
    diagram: str | None = None
    test_command: str | None = None


def _file_config_or_empty(config: RunConfig) -> DaydreamFileConfig:
    """Treat absent file_config as an empty policy for every resolver."""
    return config.file_config if config.file_config is not None else DaydreamFileConfig()


def retained_runs(config: RunConfig) -> int:
    """Finished runs one workspace keeps in private runtime storage: file config, then the default."""
    return _file_config_or_empty(config).retained_runs or DEFAULT_RETAINED_RUNS


def _configured_phase_value(config: RunConfig, phase: str, setting: str) -> str | None:
    """Resolve CLI global > file phase > file global; empty strings are unset."""
    file_config = _file_config_or_empty(config)
    return (
        getattr(config, setting)
        or file_config.phases.get(phase, {}).get(setting)
        or getattr(file_config, setting)
    )


def resolved_backend_name(config: RunConfig, phase: str) -> str:
    """Explicit per-phase backend > configured tiers > Claude fallback."""
    return getattr(config, f"{phase}_backend", None) or _configured_phase_value(config, phase, "backend") or "claude"


def _default_backend_name(config: RunConfig) -> str:
    """Resolve the general backend: CLI global, file global, then claude.

    Never inspect phase overrides: archive identity and the default-backend display
    must remain distinct from review-specific provenance.
    """
    file_config = _file_config_or_empty(config)
    return config.backend or file_config.backend or "claude"


def resolved_model(config: RunConfig, phase: str) -> str | None:
    """Explicit per-phase model > configured tiers > the resolved backend's table."""
    return (
        getattr(config, f"{phase}_model", None)
        or _configured_phase_value(config, phase, "model")
        or PHASE_DEFAULT_MODELS.get(resolved_backend_name(config, phase), {}).get(phase)
    )


def resolved_reasoning_effort(config: RunConfig, phase: str) -> str | None:
    """Resolve effort: CLI global > file phase > file global > the resolved backend's table.

    None leaves the backend's ambient default intact, such as Codex's
    model_reasoning_effort config.
    """
    return (
        _configured_phase_value(config, phase, "reasoning_effort")
        or PHASE_DEFAULT_EFFORT.get(resolved_backend_name(config, phase), {}).get(phase)
    )


def resolved_phase_settings(config: RunConfig, phases: Iterable[str]) -> dict[str, dict[str, str | None]]:
    """Return each phase's config-resolved ``backend``, ``model`` and ``reasoning_effort``.

    These are the values Daydream hands the backend, not the model a provider served;
    ``None`` means nothing resolved and the adapter's native default applies.
    """
    return {phase: {"backend": resolved_backend_name(config, phase), "model": resolved_model(config, phase),
                    "reasoning_effort": resolved_reasoning_effort(config, phase)} for phase in phases}


DEEP_FLOW_ALIASES = ("deep",)


def runs_review(config: RunConfig) -> bool:
    """Whether this request runs the reviewer: every built-in flow mode except diagram-only."""
    return config.flow_name in (None, *DEEP_FLOW_ALIASES) and config.output_mode != "diagram"
