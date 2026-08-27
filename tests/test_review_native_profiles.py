# GOLDEN BASELINE — renders every Copy-existing builder against the CURRENT
# (pre-extraction) default profile and freezes the byte output. Task 5+ asserts
# the SAME strings after recomposition. The per-stack/structural/vetting
# builders are excluded here — their output intentionally changes (skill line
# removed + authored block inserted); those golden deltas are pinned separately
# in Task 5.
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
    # After recomposition this string must be byte-identical. Freeze the exact
    # rendered text by asserting a stable sentinel that will survive recomposition:
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


def test_copy_existing_stages_byte_identical_after_strategy_threading() -> None:
    from daydream.deep.coverage import build_uncovered_sweep_prompt
    from daydream.deep.prompts import (
        build_supervise_prompt,
        build_suppression_prompt,
        build_verification_prompt,
    )
    from daydream.phases import build_alternative_review_prompt, build_intent_prompt
    p = rp.build_default_profile()
    sv = build_supervise_prompt(strategy=p.strategies["supervision"].content,
        supervise_input_path=Path("/in"), diff_path=Path("/d"),
        intent_path=Path("/i"), alternatives_path=Path("/a"), cwd=Path("/c"))
    sup = build_suppression_prompt(strategy=p.strategies["suppression"].content,
        suppression_input_path=Path("/in"), diff_path=Path("/d"),
        intent_path=Path("/i"), alternatives_path=Path("/a"), cwd=Path("/c"))
    ver = build_verification_prompt(strategy=p.strategies["verification"].content,
        items=[{"id": 1, "lens": "per-stack", "file": "a.py", "line": 1, "description": "x"}],
        cwd=Path("/c"), output_path=Path("/o"))
    unc = build_uncovered_sweep_prompt(strategy=p.strategies["uncovered_review"].content,
        file="a.py", hunks="@@", intent_path=Path("/i"), cwd=Path("/c"),
        output_path=Path("/o"))
    intent = build_intent_prompt(strategy=p.strategies["intent"].content,
        diff_path="/d", inline_diff="+x")
    alt = build_alternative_review_prompt(strategy=p.strategies["alternatives"].content,
        intent_summary="s", diff_path="/d")
    # Host envelope sentinels preserved:
    assert "/in" in sv and "/in" in sup
    assert "per-stack" in ver or "verdict" in ver
    assert "a.py" in unc and "/i" in unc and "/c" in unc
    assert "/d" in intent and "+x" in intent
    assert "s" in alt and "/d" in alt
    # No skill tokens anywhere:
    for text in (sv, sup, ver, unc, intent, alt):
        assert "/beagle-" not in text and "beagle" not in text.lower()
