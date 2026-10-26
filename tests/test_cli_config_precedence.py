"""Resolve phase settings: CLI phase, CLI global, file phase, file global, defaults.

DAYDREAM_MODEL and DAYDREAM_BACKEND are not precedence tiers.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.backends import Backend
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from daydream.commands.improve import _parse_improve_args
from daydream.commands.review import _parse_args
from daydream.config_file import DaydreamFileConfig, load_file_config
from daydream.deep.adjudication_steps import _run_arbiter
from daydream.deep.latency import (
    PROFILE_ROUTES,
    ArbiterPlan,
    DiffSignals,
    PlannedGroup,
    diff_signals,
    route_for,
    summarize_risk,
)
from daydream.deep.state import DeepState
from daydream.extensions.registry import Registry
from daydream.flows.engine import FlowContext
from daydream.run_config import (
    RunConfig,
    _resolved_backend_name,
    _resolved_latency_profile,
    _resolved_model,
    _resolved_reasoning_effort,
    _resolved_review_backend_name,
)
from daydream.runner import _resolve_backend
from daydream.test_execution import MissingTestCommandError, canonical_test_command
from daydream.workspace import WorkContext
from tests.harness.review_result import review_coverage


def _routine_signals() -> DiffSignals:
    """A small diff with no escalation surface (A2: size never escalates)."""
    return diff_signals(diff="", changed_files=3, stack_count=1)



def test_model_precedence_cli_over_file_over_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_MODEL", "env-model")
    fc = DaydreamFileConfig(model="file-model", backend=None, phases={"fix": {"model": "file-fix"}})
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_model(cfg, "fix") == "file-fix"        # file phase override, nothing higher
    cfg.model = "cli-global"
    assert _resolved_model(cfg, "fix") == "cli-global"      # global --model beats file phase override
    cfg.fix_model = "cli-fix"
    assert _resolved_model(cfg, "fix") == "cli-fix"         # explicit per-phase beats global --model
    cfg.model = None
    cfg.fix_model = None
    assert _resolved_model(cfg, "review") == "file-model"   # no phase override -> file global
    cfg2 = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=DaydreamFileConfig())
    assert _resolved_model(cfg2, "parse") == "claude-haiku-4-5"   # falls through to table default


def test_backend_precedence_mirrors_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_BACKEND", "env-backend")
    fc = DaydreamFileConfig(model=None, backend="file-global", phases={"fix": {"backend": "file-fix"}})
    cfg = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=fc)
    assert _resolved_backend_name(cfg, "fix") == "file-fix"        # file phase override, nothing higher
    cfg.backend = "cli-global"
    assert _resolved_backend_name(cfg, "fix") == "cli-global"      # global --backend beats file phase override
    cfg.fix_backend = "cli-fix"
    assert _resolved_backend_name(cfg, "fix") == "cli-fix"         # explicit per-phase beats global --backend
    cfg.backend = None
    cfg.fix_backend = None
    assert _resolved_backend_name(cfg, "review") == "file-global"  # no phase override -> file global
    cfg2 = RunConfig(target=str(tmp_path), backend=None, model=None, file_config=DaydreamFileConfig())
    assert _resolved_backend_name(cfg2, "parse") == "claude"       # terminal fallback

def _codex_backend(
    cfg: RunConfig, phase: str, cache: dict[tuple[str, str | None, str | None, Path | None], Backend] | None = None,
) -> CodexBackend:
    """Resolve ``phase`` and narrow to the concrete Codex backend under test."""
    backend = _resolve_backend(cfg, phase, cache)
    assert isinstance(backend, CodexBackend), f"{phase} should resolve to a CodexBackend, got {type(backend)}"
    return backend

def test_phase_default_effort_table_is_the_lowest_precedence_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("daydream.run_config.PHASE_DEFAULT_EFFORT", {"codex": {"parse": "low", "fix": "high"}})
    bare = RunConfig(target=str(tmp_path), backend="codex", model="fixture-model", file_config=DaydreamFileConfig())
    assert _codex_backend(bare, "parse").reasoning_effort == "low"
    assert _codex_backend(bare, "fix").reasoning_effort == "high"
    assert _codex_backend(bare, "review").reasoning_effort is None
    assert _codex_backend(bare, "not_a_phase").reasoning_effort is None

    fc = DaydreamFileConfig(model=None, backend=None, phases={"parse": {"reasoning_effort": "xhigh"}})
    cfg = RunConfig(target=str(tmp_path), backend="codex", model="fixture-model", file_config=fc)
    assert _codex_backend(cfg, "parse").reasoning_effort == "xhigh"
    assert _codex_backend(cfg, "fix").reasoning_effort == "high"  # still the controlled table default

    cfg.file_config = DaydreamFileConfig(reasoning_effort="medium", phases=fc.phases)
    assert _codex_backend(cfg, "parse").reasoning_effort == "xhigh"  # phase beats file global
    assert _codex_backend(cfg, "fix").reasoning_effort == "medium"  # file global beats table

    cfg.reasoning_effort = "low"
    assert _codex_backend(cfg, "parse").reasoning_effort == "low"
    assert _codex_backend(cfg, "fix").reasoning_effort == "low"


@pytest.mark.parametrize("phase", ["wonder", "arbiter"])
def test_pi_review_effort_default_does_not_enable_latency_or_group_overrides(tmp_path: Path, phase: str) -> None:
    file_config = DaydreamFileConfig()
    cfg = RunConfig(target=str(tmp_path), backend="pi", file_config=file_config)
    cfg.latency_route = route_for("balanced", summarize_risk(_routine_signals()))
    backend = _resolve_backend(cfg, phase, effort_override="xhigh")
    assert isinstance(backend, PiBackend)
    assert backend.reasoning_effort is None
    file_config.phases[phase] = {"reasoning_effort": "low"}
    pinned = _resolve_backend(cfg, phase, effort_override="xhigh")
    assert isinstance(pinned, PiBackend)
    assert pinned.reasoning_effort == "low"


def test_backend_cache_splits_on_table_default_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.run_config.PHASE_DEFAULT_EFFORT", {"codex": {"review": "low"}})
    cfg = RunConfig(target=str(tmp_path), backend="codex", model="fixture-model", file_config=DaydreamFileConfig())
    cache: dict[tuple[str, str | None, str | None, Path | None], Backend] = {}
    review = _codex_backend(cfg, "review", cache)
    cfg.reasoning_effort = "high"
    pinned_review = _codex_backend(cfg, "review", cache)
    assert review.model == pinned_review.model == "fixture-model"
    assert review is not pinned_review
    assert (review.reasoning_effort, pinned_review.reasoning_effort) == ("low", "high")
    assert _resolve_backend(cfg, "review", cache) is pinned_review
    cfg.reasoning_effort = None
    assert _resolve_backend(cfg, "review", cache) is review



def test_review_backend_override_is_none_when_unset(tmp_path: Path) -> None:
    empty = DaydreamFileConfig()
    bare = RunConfig(target=str(tmp_path), backend="claude", review_backend=None, file_config=empty)
    assert _resolved_review_backend_name(bare) is None
    cfg = RunConfig(target=str(tmp_path), backend="claude", review_backend="codex", file_config=empty)
    assert _resolved_review_backend_name(cfg) == "codex"
    fc = DaydreamFileConfig(phases={"review": {"backend": "file-codex"}})
    cfg2 = RunConfig(target=str(tmp_path), backend="claude", review_backend=None, file_config=fc)
    assert _resolved_review_backend_name(cfg2) == "file-codex"
    fc_global = DaydreamFileConfig(backend="codex")
    cfg3 = RunConfig(target=str(tmp_path), backend=None, review_backend=None, file_config=fc_global)
    assert _resolved_review_backend_name(cfg3) is None

def test_trajectory_hub_repo_flag_reaches_runconfig(tmp_path: Path) -> None:
    target = str(tmp_path)

    deep = _parse_args([target, "--trajectory-hub-repo", "acme/dd-trajectories"])
    assert deep.trajectory_hub_repo == "acme/dd-trajectories"

    improve = _parse_improve_args(["improve", target, "--trajectory-hub-repo", "acme/dd-trajectories"])
    assert improve.trajectory_hub_repo == "acme/dd-trajectories"

def test_test_command_precedence_cli_over_file_config(tmp_path: Path) -> None:
    target = str(tmp_path)

    (tmp_path / "pyproject.toml").write_text('[tool.daydream]\ntest_command = "make test"\n')
    (tmp_path / ".daydream.toml").write_text('test_command = "pytest -q"\n')
    fc = load_file_config(tmp_path)
    assert fc.test_command == "pytest -q"

    cfg = _parse_args([target, "--test-command", "uv run pytest"])
    assert cfg.test_command == "uv run pytest"
    assert canonical_test_command(fc, cfg) == ["uv", "run", "pytest"]

    cfg_noflag = _parse_args([target])
    assert canonical_test_command(fc, cfg_noflag) == ["pytest", "-q"]

    empty_fc = load_file_config(tmp_path / "does-not-exist")
    with pytest.raises(MissingTestCommandError) as e:
        canonical_test_command(empty_fc, SimpleNamespace(test_command=None))
    msg = str(e.value)
    assert "test_command" in msg
    assert "tool.daydream" in msg
    assert "--test-command" in msg

def test_latency_profile_precedence_cli_over_file_over_default(tmp_path: Path) -> None:
    fc = DaydreamFileConfig(latency_profile="forensic")
    cfg = RunConfig(target=str(tmp_path), file_config=fc)
    assert _resolved_latency_profile(cfg).profile == "forensic"

    cfg.latency_profile = "fast"
    assert _resolved_latency_profile(cfg).profile == "fast"

    cfg.latency_profile = "turbo"
    resolved = _resolved_latency_profile(cfg)
    assert resolved.profile == "forensic" and resolved.fail_safe is True

    bare = RunConfig(target=str(tmp_path), file_config=DaydreamFileConfig())
    assert _resolved_latency_profile(bare).profile == "balanced"


def test_profile_route_sets_wonder_and_arbiter_effort_on_codex(tmp_path: Path) -> None:
    cfg = RunConfig(target=str(tmp_path), backend="codex", file_config=DaydreamFileConfig())
    assert _resolved_reasoning_effort(cfg, "wonder") == "medium"  # table baseline
    cfg.latency_route = route_for("balanced", summarize_risk(_routine_signals()))
    assert _resolved_reasoning_effort(cfg, "wonder") == "medium"
    assert _resolved_reasoning_effort(cfg, "arbiter") == "high"

    cfg.reasoning_effort = "low"  # explicit pin still wins
    assert _resolved_reasoning_effort(cfg, "wonder") == "low"




def _arbiter_flow_context(tmp_path: Path, backend: str) -> FlowContext:
    """The smallest FlowContext whose effort seam is production's (no test factory)."""
    work = WorkContext(
        repo=tmp_path, source=tmp_path, base_branch="main", base_sha="0" * 40, head_branch="main", head_sha="0" * 40,
        is_ephemeral=False, run_id="session-test",
    )
    return FlowContext(
        config=RunConfig(target=str(tmp_path), backend=backend, model=None, file_config=DaydreamFileConfig()),
        work=work, registry=Registry(),
    )

def test_arbiter_effort_override_is_codex_only(tmp_path: Path) -> None:
    codex = _arbiter_flow_context(tmp_path, "codex")
    assert getattr(codex.backend_for_effort("arbiter", "xhigh"), "reasoning_effort") == "xhigh"
    assert getattr(codex.backend_for_effort("arbiter", "medium"), "reasoning_effort") == "medium"

    claude = _arbiter_flow_context(tmp_path, "claude")
    assert getattr(claude.backend_for_effort("arbiter", "xhigh"), "reasoning_effort") is None
    assert getattr(claude.backend_for_effort("arbiter", "medium"), "reasoning_effort") is None

async def test_unsharded_arbiter_call_keeps_todays_xhigh_whatever_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise actual unsharded dispatch with a default, explicit and non-Codex effort."""
    observed: list[Any] = []

    async def adjudicate(backend: Any, *_args: Any, **_kwargs: Any) -> tuple[dict[int, dict[str, Any]], None]:
        observed.append(getattr(backend, "reasoning_effort"))
        return {}, None

    monkeypatch.setattr("daydream.deep.adjudication_steps.phase_arbiter_review", adjudicate)
    (tmp_path / "intent").write_text("intent")
    plan = ArbiterPlan(False, (PlannedGroup("arbiter-group-0", (), "xhigh", "test"),), "test")
    for backend, pin in (("codex", None), ("codex", "low"), ("claude", None)):
        ctx = _arbiter_flow_context(tmp_path, backend)
        ctx.config.latency_route = PROFILE_ROUTES["balanced"]
        ctx.config.reasoning_effort = pin
        ctx.data.update(dd=tmp_path, diff_path=tmp_path / "diff", intent_path=tmp_path / "intent",
                        alts_path=tmp_path / "alternatives", exploration_dir=None,
                        review_coverage=review_coverage())
        await _run_arbiter(ctx, DeepState(ctx.data), plan, [], [], effort_pin=pin,
                           targets_by_group={"arbiter-group-0": []})
    assert observed == ["xhigh", "low", None]
