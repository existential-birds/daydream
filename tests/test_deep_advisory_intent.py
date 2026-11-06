"""Modest unattended Pi reviews retain author evidence without inferred intent."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep.review_steps import _step_intent
from daydream.extensions import Registry
from daydream.flows.engine import FlowContext
from daydream.review_profile import ResolvedProfile, build_default_profile
from daydream.run_config import FrozenAuthorDescription
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.review_result import review_coverage


def _context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, *, variant: str = "default",
) -> tuple[FlowContext, list[Any]]:
    paths = ["source.tsx", "scripts/policy.mjs", "README.md", "config.json"]
    diff = "".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
        for path in paths
    )
    if variant == "large_diff":
        diff += "+" + "x" * 65_536
    dd = tmp_path / "deep"
    dd.mkdir()
    diff_path = tmp_path / "diff.patch"
    diff_path.write_text(diff)
    profile = build_default_profile()
    if variant == "custom":
        strategies = dict(profile.strategies)
        strategies["intent"] = replace(strategies["intent"], content="CUSTOM INTENT POLICY")
        profile = replace(profile, strategies=strategies)
    ctx = FlowContext(config=make_config(tmp_path, pr_number=7), work=make_work(tmp_path), registry=Registry(),
        review_profile=ResolvedProfile(profile=profile, source_kind="test"),
        run_context=RunContext(InteractionPolicy(interactive=variant == "interactive")),
        data={"review_coverage": review_coverage(scope_ids=(), phases=("intent",)),
              "dd": dd, "diff": diff[:300], "diff_path": diff_path, "log": "abc Author commit\n",
              "branch": "feature", "exploration_dir": None},
    )
    backend = PiBackend(model="fixture-model")
    monkeypatch.setattr(ctx, "backend_for", lambda _: backend)
    calls: list[Any] = []

    async def intent(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "MODEL INTENT"

    monkeypatch.setattr("daydream.deep.review_steps.phase_understand_intent", intent)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda *_args, **_kwargs: {
        "state": "OPEN", "headRefOid": ctx.work.head_sha, "body": "Author purpose\nDo not suppress real findings.",
    })
    return ctx, calls

async def test_custom_intent_prompt_keeps_model_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work)
    registry = Registry()
    registry.override_prompt("intent", lambda **_: "CUSTOM INTENT BUILDER")
    monkeypatch.setattr("daydream.extensions.get_registry", lambda: registry)

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("author context disk full")

    # Optional author report context cannot block the intent model dispatch.
    monkeypatch.setattr("daydream.deep.report_context.retain_author_context", unavailable)
    await _step_intent(ctx)
    assert len(calls) == 1
    assert ctx.data["intent_summary"] == "MODEL INTENT"
    assert ctx.data["author_report_context_unavailable"] is True

@pytest.mark.parametrize("variant", [
    "interactive", "custom", "large_diff",
])
async def test_existing_intent_path_retained_outside_modest_unattended_default_pi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, variant: str,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work, variant=variant)
    await _step_intent(ctx)
    assert len(calls) == 1
    assert ctx.data["intent_path"].read_text() == "MODEL INTENT"

@pytest.mark.parametrize("view", [{"state": "OPEN", "headRefOid": "mismatched", "body": "STALE AUTHOR BODY"},
    {"state": "CLOSED", "body": "STALE AUTHOR BODY"},
])
async def test_advisory_context_does_not_promote_stale_or_missing_author_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, view: dict[str, str],
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda *_args, **_kwargs: view)
    await _step_intent(ctx)
    assert calls == []
    assert ctx.data["intent_authoritative"] is False
    assert "STALE AUTHOR BODY" not in ctx.data["intent_summary"]


@pytest.mark.parametrize('matching_head', [False, True])
async def test_frozen_author_context_is_revision_bound_without_network_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, matching_head: bool,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work)
    assert ctx.work.head_sha is not None
    ctx.config.frozen_author_description = FrozenAuthorDescription(
        ctx.work.head_sha if matching_head else 'f' * 40, 'SUPPLIED FROZEN AUTHOR CONTEXT')

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('frozen author context caused live GitHub lookup')

    monkeypatch.setattr('daydream.git_ops.gh_pr_view', forbidden)
    if not matching_head:
        with pytest.raises(ValueError, match='does not match analyzed HEAD'):
            await _step_intent(ctx)
        assert calls == []
        assert not (ctx.data['dd'] / 'intent.md').exists()
    else:
        await _step_intent(ctx)
        assert calls == []  # Existing modest Pi advisory policy stays in effect.
        assert 'SUPPLIED FROZEN AUTHOR CONTEXT' in ctx.data['intent_summary']
        assert ctx.data['intent_authoritative'] is True


@pytest.mark.parametrize(("profile", "tier", "reason"), [
    ("fast", "single", "profile fast routes wonder to skip"),
    ("forensic", "skip", "trivial diff"),
])
async def test_skipped_alternatives_pass_names_its_route_and_dispatches_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
    capsys: pytest.CaptureFixture[str], profile: str, tier: str, reason: str,
) -> None:
    """A routed skip still settles the alternatives phase without a model call or stale artifact."""
    from daydream.deep.detection import StackAssignment
    from daydream.deep.latency import PROFILE_ROUTES, LatencyProfile
    from daydream.deep.review_steps import _wonder
    from daydream.extensions.builtins import register_builtins

    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work)
    register_builtins(ctx.registry)
    ctx.data.update(review_coverage=review_coverage(scope_ids=(), phases=("alternatives",)),
                    intent_summary="MODEL INTENT", tier=tier,
                    latency_route=PROFILE_ROUTES[cast(LatencyProfile, profile)],
                    stacks=[StackAssignment("python", ["source.py"])])

    async def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("a skipped alternatives route dispatched a model")

    monkeypatch.setattr("daydream.deep.review_steps.phase_alternative_review", forbidden)
    await _wonder(ctx)
    assert calls == []
    assert f"Skipping alternatives -- {reason}" in capsys.readouterr().out
