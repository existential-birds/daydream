"""Resolve phase settings: CLI phase, CLI global, file phase, file global, defaults.

DAYDREAM_MODEL and DAYDREAM_BACKEND are not precedence tiers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.config_file import DaydreamFileConfig
from daydream.deep.adjudication_steps import _run_arbiter
from daydream.deep.latency import (
    PROFILE_ROUTES,
    ArbiterPlan,
    PlannedGroup,
)
from daydream.deep.state import DeepState
from daydream.extensions.registry import Registry
from daydream.flows.engine import FlowContext
from daydream.run_config import (
    RunConfig,
    _resolved_latency_profile,
)
from daydream.workspace import WorkContext
from tests.harness.review_result import review_coverage


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
