"""Default design review shares the structural pass without losing coverage."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.artifacts import per_stack_failures_path
from daydream.deep.detection import StackAssignment
from daydream.deep.prompts import build_structural_prompt
from daydream.deep.review_steps import _step_wonder_and_per_stack
from daydream.extensions import Registry
from daydream.flows.engine import FlowContext
from daydream.review_budget import review_warnings
from daydream.review_profile import ResolvedProfile, build_default_profile


@pytest.mark.parametrize("start_at", ["review", "per-stack"])
@pytest.mark.parametrize("custom_structure", [False, True])
async def test_default_alternatives_share_structural_review_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
    start_at: str, custom_structure: bool,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work, start_at=start_at,
                          custom_structure=custom_structure)
    if start_at == "per-stack":
        ctx.data["alts_path"].write_text('[{"title": "retained prior finding"}]')
    await _step_wonder_and_per_stack(ctx)
    assert not calls["alternatives"]
    assert len(calls["reviews"]) == 1
    strategy = calls["reviews"][0]["strategies"]["discovery.structural"]
    assert "confirmed intent" in strategy and "canonical implementation" in strategy
    if custom_structure:
        assert "CUSTOM STRUCTURAL POLICY" in strategy
    if start_at == "review":
        assert ctx.data["alts_path"].read_text() == "[]"
        assert not calls["reviews"][0]["include_alternatives"]
    else:
        assert "retained prior finding" in ctx.data["alts_path"].read_text()
        assert calls["reviews"][0]["include_alternatives"]


@pytest.mark.parametrize("custom_alternatives,structural", [(True, True), (False, False)])
async def test_custom_alternatives_and_structural_disabled_keep_independent_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
    custom_alternatives: bool, structural: bool,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work,
                          custom_alternatives=custom_alternatives, structural=structural)
    await _step_wonder_and_per_stack(ctx)
    assert len(calls["alternatives"]) == 1
    assert len(calls["reviews"]) == 1
    assert ctx.data["alts_path"].read_text() == "[]"


async def test_folded_structural_budget_failure_remains_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work,
                          failures={"structure": "budget exhausted: wall_budget_exceeded"})
    await _step_wonder_and_per_stack(ctx)
    assert not calls["alternatives"]
    assert per_stack_failures_path(ctx.data["dd"]).exists()
    assert review_warnings(ctx.data["dd"]) == ("structure: budget exhausted: wall_budget_exceeded",)


def _context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
    *, start_at: str = "review", custom_structure: bool = False,
    custom_alternatives: bool = False, structural: bool = True,
    failures: dict[str, str] | None = None,
) -> tuple[FlowContext, dict[str, list[Any]]]:
    profile = build_default_profile()
    strategies = dict(profile.strategies)
    if custom_alternatives:
        strategies["alternatives"] = replace(strategies["alternatives"], content="CUSTOM DESIGN REVIEW")
    if custom_structure:
        strategies["discovery.structural"] = replace(
            strategies["discovery.structural"], content="CUSTOM STRUCTURAL POLICY",
        )
    resolved = ResolvedProfile(profile=replace(profile, strategies=strategies), source_kind="test")
    dd = tmp_path / "deep"
    dd.mkdir()
    diff_path = tmp_path / "diff.patch"
    diff_path.write_text("diff --git a/app.py b/app.py\n+pass\n")
    stacks = [StackAssignment("python", ["app.py"])]
    if structural:
        stacks.append(StackAssignment("structure", ["app.py"]))
    registry = Registry()
    registry.override_prompt("structural", build_structural_prompt)
    ctx = FlowContext(
        config=make_config(tmp_path, start_at=start_at), work=make_work(tmp_path), registry=registry,
        review_profile=resolved,
        data={"dd": dd, "stacks": stacks, "tier": "single", "single_stack_mode": False,
              "intent_summary": "Preserve behavior", "intent_path": dd / "intent.md",
              "alts_path": dd / "alternatives.json", "diff_path": diff_path,
              "diff": diff_path.read_text(), "exploration_dir": None, "failed_stacks": {}},
    )
    calls: dict[str, list[Any]] = {"alternatives": [], "reviews": []}

    async def alternative(*args: Any, **kwargs: Any) -> list[Any]:
        calls["alternatives"].append(kwargs)
        return []

    async def reviews(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, str]]:
        calls["reviews"].append(kwargs)
        return {}, failures or {}

    monkeypatch.setattr(ctx, "backend_for", lambda _: None)
    monkeypatch.setattr("daydream.deep.review_steps.phase_alternative_review", alternative)
    monkeypatch.setattr("daydream.deep.review_steps.phase_per_stack_reviews", reviews)
    return ctx, calls


@pytest.mark.parametrize("start_at", ["review", "per-stack"])
async def test_custom_structural_builder_preserves_independent_alternatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
    start_at: str,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work, start_at=start_at)
    ctx.registry.override_prompt("structural", lambda **_: "CUSTOM STRUCTURAL BUILDER")
    original_strategy = ctx.strategy("discovery.structural")
    findings = [{"title": "retained design finding"}]

    async def alternatives(*args: Any, **kwargs: Any) -> list[Any]:
        calls["alternatives"].append(kwargs)
        return findings

    monkeypatch.setattr("daydream.deep.review_steps.phase_alternative_review", alternatives)
    messages: list[str] = []
    monkeypatch.setattr("daydream.deep.review_steps.print_dim", lambda _, message: messages.append(message))
    if start_at == "per-stack":
        ctx.data["alts_path"].write_text(json.dumps(findings))
    await _step_wonder_and_per_stack(ctx)
    assert len(calls["alternatives"]) == (1 if start_at == "review" else 0)
    assert json.loads(ctx.data["alts_path"].read_text()) == findings
    assert calls["reviews"][0]["registry"] is ctx.registry
    assert calls["reviews"][0]["strategies"]["discovery.structural"] == original_strategy
    assert not any("included in the structural review" in message for message in messages)
    if start_at == "per-stack":
        assert calls["reviews"][0]["include_alternatives"]
        assert calls["reviews"][0]["alternatives_path"] == ctx.data["alts_path"]
