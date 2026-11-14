"""The exploration pre-scan step of the diagram-only flow."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from daydream.agent import console
from daydream.artifact_visibility import artifact_dir_for
from daydream.deep.diff import _read_full_diff
from daydream.deep.state import DeepState
from daydream.flows.engine import FlowContext
from daydream.trajectory import DaydreamPhase, phase_scope
from daydream.ui import (
    phase_subtitle,
    print_dim,
    print_phase_hero,
    print_warning,
    render_exploration_summary,
)

logger = logging.getLogger(__name__)

try:
    from daydream.exploration import ExplorationContext, safe_explore
    from daydream.exploration_runner import (
        count_changed_files as count_changed_files,
        pre_scan,
        select_tier as select_tier,
    )

    EXPLORATION_AVAILABLE = True
except ImportError:  # pragma: no cover -- optional exploration dependency
    EXPLORATION_AVAILABLE = False


async def _step_exploration(ctx: FlowContext) -> None:
    """Exploration pre-scan (D-43), reused on an exact key match."""
    deep_state = DeepState(ctx.data)
    from daydream.exploration import cache_key_path, exploration_cache_key, read_cache_key
    from daydream.runner import _compute_diff_ref

    config = ctx.config
    target_dir = ctx.work.repo
    daydream_dir = artifact_dir_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )
    # Explore and key the full persisted diff so omitted inline blocks remain covered.
    # Read failure warns and falls back to bounded text because exploration is optional.
    diff = deep_state.diff
    try:
        diff = _read_full_diff(ctx)
    except OSError as exc:
        print_warning(
            console,
            f"Could not read the full diff for the exploration pre-scan "
            f"({exc}); falling back to the bounded in-memory diff",
        )
    tier = deep_state.tier
    exploration_path = daydream_dir / "exploration"

    reuse_enabled = ctx.work.captured_source is None

    exploration_dir: Path | None = None
    if not EXPLORATION_AVAILABLE:
        print_warning(
            console,
            "Exploration infrastructure not installed; running deep pipeline "
            "without pre-scan grounding",
        )
    elif config.exploration_context is None:
        cache_key = exploration_cache_key(
            ctx.work.head_sha or "", diff, tier,
            strategies={name: ctx.strategy(name) for name in (
                "exploration.pattern_scan", "exploration.dependency_trace", "exploration.test_mapping",
            )},
        )
        if (
            reuse_enabled
            and exploration_path.is_dir()
            and read_cache_key(exploration_path) == cache_key
        ):
            # Early return BEFORE the pre_scan/write_to_dir block below: routing
            # a hit through it with an empty in-memory context would overwrite
            # the cached files with "No data collected" stubs.
            print_dim(console, f"Reusing exploration pre-scan from {exploration_path}")
            deep_state.exploration_dir = exploration_path
            return

        # Miss: drop any stale directory so a partial previous result cannot be
        # read as this run's grounding.
        if exploration_path.is_dir():
            shutil.rmtree(exploration_path, ignore_errors=True)

        if tier == "skip":
            print_dim(console, "Skipping exploration -- trivial diff")
            config.exploration_context = ExplorationContext()
        else:
            print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
            explore_backend = ctx.backend_for("exploration")
            async with phase_scope(DaydreamPhase.EXPLORATION):
                config.exploration_context = await safe_explore(
                    pre_scan,
                    explore_backend,
                    target_dir,
                    diff,
                    diff_ref=_compute_diff_ref(target_dir),
                    strategies={
                        "exploration.pattern_scan": ctx.strategy("exploration.pattern_scan"),
                        "exploration.dependency_trace": ctx.strategy("exploration.dependency_trace"),
                        "exploration.test_mapping": ctx.strategy("exploration.test_mapping"),
                        "exploration.repository_survey": ctx.strategy("exploration.repository_survey"),
                    },
                    run_context=ctx.run_context,
                )
            # Osprey resolves an omitted model from its own config and reports
            # the authoritative value in session_start, during safe_explore.
            # Log after that boundary so the backend name is never presented as
            # the model id.
            print_dim(console, f"Exploration model: {explore_backend.model}")
            console.print(render_exploration_summary(config.exploration_context))
        if config.exploration_context is not None:
            exploration_dir = config.exploration_context.write_to_dir(exploration_path)
            if reuse_enabled and config.exploration_context.completed:
                cache_key_path(exploration_path).write_text(cache_key, encoding="utf-8")
            deep_state.exploration_dir = exploration_dir
            return
    if EXPLORATION_AVAILABLE and config.exploration_context is not None:
        exploration_dir = config.exploration_context.write_to_dir(exploration_path)
    deep_state.exploration_dir = exploration_dir
