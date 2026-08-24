"""Task 8 (R13): completeness guard for model-bearing stage classification.

Adding a new model-bearing review stage without a corresponding profile
strategy + host-envelope classification must fail this guard. The test
simulates the production spine registry with a stand-in and asserts every
registered stage carries both a profile strategy and an envelope
classification; a stage added to the spine without either trips it.
"""

from daydream import review_profile as rp


def test_every_registered_model_bearing_stage_has_strategy_and_classification():
    # The production spine registry (STAGE_KEYS) is the model of truth here:
    # every model-bearing stage must carry a profile strategy and a host-
    # envelope classification. Iterating STAGE_KEYS directly (not a hardcoded
    # 4-stage subset) means a stage added to the spine without either trips this
    # guard -- it cannot pass silently by listing a smaller curated set.
    default = rp.build_default_profile()
    for stage in rp.STAGE_KEYS:
        assert stage in default.strategies, f"model-bearing stage {stage} has no profile strategy"
        assert stage in rp.ENVELOPE_CLASSIFICATION, f"stage {stage} has no host-envelope classification"


def test_classification_is_strategy_plus_host_envelope():
    for key in rp.STAGE_KEYS:
        cls = rp.ENVELOPE_CLASSIFICATION[key]
        assert cls["strategy"] and cls["envelope"]  # both nonempty, one per stage
