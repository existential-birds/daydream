"""Golden coverage for builders that copy the default profile strategy."""
from pathlib import Path

from daydream import review_profile as rp


def _default_strategy(stage: str) -> str:
    return rp.build_default_profile().strategies[stage].content


def test_golden_baseline_generic_fallback() -> None:
    from daydream.deep.prompts import build_generic_fallback_prompt
    p = build_generic_fallback_prompt(
        strategy=_default_strategy("discovery.generic_fallback"),
        files=["a.js"], diff_path=Path("/d"), intent_path=Path("/i"),
        alternatives_path=Path("/a"), output_path=Path("/o"), cwd=Path("/c"),
    )
    # Check a stable sentinel in the rendered text.
    assert "language-agnostic review practices" in p
    assert "a.js" in p and "/d" in p and "/c" in p   # host runtime data present
    assert "beagle" not in p.lower() and "/beagle-" not in p


def test_golden_baseline_arbiter_and_merge() -> None:
    from daydream.deep.prompts import build_arbiter_prompt, build_merge_prompt
    a = build_arbiter_prompt(strategy=_default_strategy("arbitration"),
                             arbiter_input_path=Path("/in"), diff_path=Path("/d"),
                             intent_path=Path("/i"), alternatives_path=Path("/a"),
                             cwd=Path("/c"))
    m = build_merge_prompt(strategy=_default_strategy("merge"),
                           per_stack_records_paths=[Path("/ps")], intent_path=Path("/i"),
                           alternatives_path=Path("/a"), dedup_candidates_path=Path("/dc"),
                           output_path=Path("/o"))
    assert "adjudicating their work" in a      # strategy content present
    assert "/ps" in m and "/dc" in m           # envelope runtime data present
    assert "/in" in a and "/i" in m

def test_authored_blocks_land_verbatim() -> None:
    p = rp.build_default_profile()
    per_stack = p.strategies["discovery.per_stack"].content
    structural = p.strategies["discovery.structural"].content
    vet = p.strategies["improve.vetting"].content
    assert per_stack.startswith("Review the changed behavior assigned to this stack")
    assert structural.startswith("Review the repository-wide interactions introduced or exposed by this diff")
    assert vet.startswith("Treat every audit candidate as an untrusted hypothesis, not as evidence")
    for text in (per_stack, structural, vet):
        assert "beagle" not in text.lower()
    assert p.strategies["discovery.per_stack"].source == "authored: #886 NATIVE_PER_STACK_DISCOVERY_STRATEGY"
    assert p.strategies["discovery.structural"].source == "authored: #886 NATIVE_STRUCTURAL_DISCOVERY_STRATEGY"
    assert p.strategies["improve.vetting"].source == "authored: #886 NATIVE_IMPROVE_VET_STRATEGY"


def test_prompt_builders_include_supplied_strategy_and_runtime_context() -> None:
    from daydream.deep.coverage import build_uncovered_sweep_prompt
    from daydream.deep.prompts import (
        build_supervise_prompt,
        build_suppression_prompt,
        build_verification_prompt,
    )
    from daydream.phases import build_alternative_review_prompt, build_intent_prompt
    sv = build_supervise_prompt(
        strategy="supervise-strategy-sentinel {supervise_input_path}",
        supervise_input_path=Path("/supervise-input.json"),
        diff_path=Path("/review.diff"),
        intent_path=Path("/intent.md"),
        alternatives_path=Path("/alternatives.json"),
        cwd=Path("/review-repo"),
    )
    sup = build_suppression_prompt(
        strategy="suppression-strategy-sentinel {suppression_input_path}",
        suppression_input_path=Path("/suppression-input.json"),
        diff_path=Path("/review.diff"),
        intent_path=Path("/intent.md"),
        alternatives_path=Path("/alternatives.json"),
        cwd=Path("/review-repo"),
    )
    ver = build_verification_prompt(
        strategy="verification-strategy-sentinel",
        items=[
            {
                "id": 1,
                "lens": "per-stack",
                "file": "changed.py",
                "line": 1,
                "description": "verification-candidate-sentinel",
            }
        ],
        cwd=Path("/review-repo"),
        output_path=Path("/verification.json"),
    )
    unc = build_uncovered_sweep_prompt(
        strategy="uncovered-strategy-sentinel {file}",
        file="uncovered.py",
        hunks="@@ -1 +1 @@\n-old\n+new",
        intent_path=Path("/intent.md"),
        cwd=Path("/review-repo"),
        output_path=Path("/uncovered-review.md"),
    )
    intent = build_intent_prompt(
        strategy="intent-strategy-sentinel {diff_path}",
        diff_path="/review.diff",
        inline_diff="+new behavior",
    )
    alt = build_alternative_review_prompt(
        strategy="alternative-strategy-sentinel {intent_summary} {diff_path}",
        intent_summary="intent-summary-sentinel",
        diff_path="/alternative.diff",
    )

    assert "supervise-strategy-sentinel" in sv
    assert "/supervise-input.json" in sv and "/review.diff" in sv
    assert "suppression-strategy-sentinel" in sup
    assert "/suppression-input.json" in sup and "/review.diff" in sup
    assert "verification-strategy-sentinel" in ver
    assert "changed.py" in ver and "verification-candidate-sentinel" in ver
    assert "uncovered-strategy-sentinel" in unc
    assert "uncovered.py" in unc and "@@ -1 +1 @@" in unc
    assert "intent-strategy-sentinel" in intent
    assert "/review.diff" in intent and "+new behavior" in intent
    assert "alternative-strategy-sentinel" in alt
    assert "intent-summary-sentinel" in alt and "/alternative.diff" in alt

    # Native builders must not reintroduce the removed skill-invocation framing.
    for text in (sv, sup, ver, unc, intent, alt):
        assert "/beagle-" not in text and "beagle" not in text.lower()
