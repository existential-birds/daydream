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


def test_audit_stages_track_production_playbook():
    # The guard must not be purely self-referential: every audit category in the
    # production playbook (daydream.improve.prompts) is itself a model-bearing
    # stage, so it must already be registered as a profile strategy + envelope
    # classification. A category added to production without a corresponding
    # STAGE_KEYS edit trips the guard instead of passing silently.
    from daydream.improve.prompts import AUDIT_PLAYBOOK_SECTIONS

    default = rp.build_default_profile()
    for category in AUDIT_PLAYBOOK_SECTIONS:
        stage = f"improve.audit.{category}"
        assert stage in rp.STAGE_KEYS, (
            f"audit category `{category}` is not a registered review stage"
        )
        assert stage in default.strategies, (
            f"model-bearing audit stage {stage} has no profile strategy"
        )
        assert stage in rp.ENVELOPE_CLASSIFICATION, (
            f"stage {stage} has no host-envelope classification"
        )
