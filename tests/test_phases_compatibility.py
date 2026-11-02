"""Canonical phase exports cross each owning module boundary."""
from daydream import phases
from daydream.phases import handoff, review_prompts, schemas, testing


def test_historical_phase_exports_are_canonical_objects() -> None:
    """Historical ``daydream.phases`` import paths keep resolving to each owner's object."""
    for owner, name in ((review_prompts, "build_intent_prompt"), (schemas, "PER_STACK_RECORD_SCHEMA"),
                        (handoff, "FAILURE_SUMMARIZER_SCHEMA"), (testing, "SETUP_INVESTIGATOR_SCHEMA")):
        assert getattr(phases, name) is getattr(owner, name), name
