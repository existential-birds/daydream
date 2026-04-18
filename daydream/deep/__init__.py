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
from daydream.deep.detection import StackAssignment, detect_stacks

__all__ = [
    "StackAssignment",
    "alternatives_path",
    "check_deep_artifacts",
    "dedup_candidates_path",
    "deep_dir",
    "detect_stacks",
    "intent_path",
    "per_stack_records_path",
    "per_stack_review_path",
]
