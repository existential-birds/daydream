"""Historical phase imports retain the canonical prompt and schema objects."""

from types import ModuleType

import pytest

from daydream import phases
from daydream.phases import handoff, review_prompts, schemas, testing


@pytest.mark.parametrize(
    "owner,name",
    [
        (owner, name)
        for owner, names in (
            (review_prompts, (
                "_confidence_and_convention_instructions",
                "_dependency_impact_instructions",
                "_exploration_pointer",
                "_settled_decisions_block",
                "build_alternative_review_prompt",
                "build_intent_prompt",
            )),
            (schemas, (
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
            )),
            (handoff, ("FAILURE_SUMMARIZER_SCHEMA",)),
            (testing, ("SETUP_INVESTIGATOR_SCHEMA",)),
        )
        for name in names
    ],
    ids=lambda value: value.__name__ if isinstance(value, ModuleType) else value,
)
def test_historical_phase_export_is_canonical_object(owner: ModuleType, name: str) -> None:
    """Fork imports must survive extraction without wrappers or copied schemas."""
    assert getattr(phases, name) is getattr(owner, name)
