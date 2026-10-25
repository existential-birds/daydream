"""Canonical phase exports cross each owning module boundary."""
from types import ModuleType

import pytest

from daydream import phases
from daydream.phases import handoff, review_prompts, schemas, testing


@pytest.mark.parametrize("owner,name", [
    (review_prompts, "build_intent_prompt"),
    (schemas, "PER_STACK_RECORD_SCHEMA"),
    (handoff, "FAILURE_SUMMARIZER_SCHEMA"),
    (testing, "SETUP_INVESTIGATOR_SCHEMA"),
])
def test_historical_phase_export_is_canonical_object(owner: ModuleType, name: str) -> None:
    assert getattr(phases, name) is getattr(owner, name)
