"""Canonical phase exports cross each owning module boundary."""
from daydream import phases
from daydream.phases import handoff, testing


def test_historical_phase_exports_are_canonical_objects() -> None:
    """Historical ``daydream.phases`` import paths keep resolving to each owner's object."""
    for owner, name in ((handoff, "FAILURE_SUMMARIZER_SCHEMA"), (testing, "SETUP_INVESTIGATOR_SCHEMA")):
        assert getattr(phases, name) is getattr(owner, name), name
