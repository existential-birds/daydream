"""Historical phase imports retain the canonical prompt and schema objects."""
from collections.abc import Sequence
from types import ModuleType

import pytest

from daydream import phases
from daydream.phases import handoff, review_prompts, schemas, testing


def _case_historical_phase_export_is_canonical_object(owner: ModuleType, name: str) -> None:
    """Fork imports must survive extraction without wrappers or copied schemas."""
    assert getattr(phases, name) is getattr(owner, name)


def test_historical_phase_export_is_canonical_object(subtests: pytest.Subtests) -> None:
    cases_0: Sequence[tuple[ModuleType, str]] = [
        (owner, name)
        for owner, names in (
            (
                review_prompts,
                (
                    "_confidence_and_convention_instructions",
                    "_dependency_impact_instructions",
                    "_exploration_pointer",
                    "_settled_decisions_block",
                    "build_alternative_review_prompt",
                    "build_intent_prompt",
                ),
            ),
            (
                schemas,
                (
                    "ALTERNATIVE_REVIEW_SCHEMA",
                    "ARBITER_SCHEMA",
                    "FEEDBACK_SCHEMA",
                    "FIX_VERIFY_ACTIONABLE_VERDICTS",
                    "FIX_VERIFY_RETARGETABLE_VERDICTS",
                    "FIX_VERIFY_VERDICTS",
                    "FIX_VERIFY_VERDICTS_SCHEMA",
                    "MERGED_ITEMS_SCHEMA",
                    "PER_STACK_RECORD_SCHEMA",
                    "RECOMMENDATION_VERDICTS_SCHEMA",
                    "SUPERVISE_SCHEMA",
                    "SUPPRESSION_SCHEMA",
                ),
            ),
            (handoff, ("FAILURE_SUMMARIZER_SCHEMA",)),
            (testing, ("SETUP_INVESTIGATOR_SCHEMA",)),
        )
        for name in names
    ]
    for owner, name in cases_0:
        with subtests.test(owner=owner, name=name):
            _case_historical_phase_export_is_canonical_object(owner=owner, name=name)
