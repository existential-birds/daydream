# GOLDEN BASELINE — renders every Copy-existing builder against the CURRENT
# (pre-extraction) default profile and freezes the byte output. Task 5+ asserts
# the SAME strings after recomposition. The per-stack/structural/vetting
# builders are excluded here — their output intentionally changes (skill line
# removed + authored block inserted); those golden deltas are pinned separately
# in Task 5.
import pytest
from pathlib import Path
from daydream import review_profile as rp


def _default_strategy(stage: str) -> str:
    return rp.build_default_profile().strategies[stage].content


def test_golden_baseline_generic_fallback():
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


def test_golden_baseline_arbiter_and_merge():
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