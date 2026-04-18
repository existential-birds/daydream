"""Deep-review mode package.

Exports run_deep once the orchestrator exists (plan 05-09).
"""

from daydream.deep.artifacts import (
    alternatives_path,
    check_deep_artifacts,
    dedup_candidates_path,
    deep_dir,
    intent_path,
    per_stack_records_path,
    per_stack_review_path,
)
from daydream.deep.dedup import CandidatePair, build_dedup_candidates
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.prompts import (
    DOC_REVIEW_NOTICE,
    build_generic_fallback_prompt,
    build_per_stack_prompt,
)

__all__ = [
    "DOC_REVIEW_NOTICE",
    "CandidatePair",
    "StackAssignment",
    "alternatives_path",
    "build_dedup_candidates",
    "build_generic_fallback_prompt",
    "build_per_stack_prompt",
    "check_deep_artifacts",
    "dedup_candidates_path",
    "deep_dir",
    "detect_stacks",
    "intent_path",
    "per_stack_records_path",
    "per_stack_review_path",
]
