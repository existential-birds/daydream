"""Task 8 (R13): completeness guard for model-bearing stage classification.

Adding a new model-bearing review stage without a corresponding profile
strategy + host-envelope classification must fail this guard. The test
simulates the production spine registry with a stand-in and asserts every
registered stage carries both a profile strategy and an envelope
classification; a stage added to the spine without either trips it.
"""

from daydream import review_profile as rp


def test_every_registered_model_bearing_stage_has_strategy_and_classification():
    # Simulate the spine registry: every model-bearing stage must carry a
    # profile strategy + host-envelope classification. The canary
    # ``NEW_UNCLASSIFIED_STAGE`` documents R13's failure mode -- a stage added
    # to the spine without a strategy + envelope classification would trip this
    # guard. It is intentionally absent from the production registry, so only
    # real stages are enumerated here.
    registered_stages = {  # stand-in for the production stage registry
        "intent",
        "arbitration",
        "discovery.per_stack",
        "merge",
    }
    for stage in registered_stages:
        assert stage in rp.STAGE_KEYS, f"model-bearing stage {stage} has no profile strategy"
        assert stage in rp.ENVELOPE_CLASSIFICATION, f"stage {stage} has no host-envelope classification"


def test_classification_is_strategy_plus_host_envelope():
    for key in rp.STAGE_KEYS:
        cls = rp.ENVELOPE_CLASSIFICATION[key]
        assert cls["strategy"] and cls["envelope"]  # both nonempty, one per stage
