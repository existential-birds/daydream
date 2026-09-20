"""Task 8 (R13): completeness guard for model-bearing stage classification.

Every model-bearing stage must have both a profile strategy and a host-envelope
classification.
"""

from daydream import review_profile as rp

# Every model-bearing review-spine and Improve-judgment stage (R2). This is
# the canonical stage registry the completeness guard (R13) and the #886
# migration manifest must match.
STAGE_KEYS: frozenset[str] = frozenset(
    {
        "exploration.repository_survey",
        "exploration.pattern_scan",
        "exploration.dependency_trace",
        "exploration.test_mapping",
        "intent",
        "alternatives",
        "discovery.per_stack",
        "discovery.structural",
        "discovery.generic_fallback",
        "uncovered_review",
        "arbitration",
        "suppression",
        "merge",
        "supervision",
        "verification",
        "improve.audit.correctness",
        "improve.audit.security",
        "improve.audit.performance",
        "improve.audit.tests",
        "improve.audit.tech-debt",
        "improve.audit.dependencies",
        "improve.audit.dx",
        "improve.audit.docs",
        "improve.vetting",
    }
)


def test_every_registered_model_bearing_stage_has_strategy_and_classification() -> None:
    # The stage registry (STAGE_KEYS) is the model of truth here:
    # every model-bearing stage must carry a profile strategy and a host-
    # envelope classification. Iterating STAGE_KEYS directly (not a hardcoded
    # 4-stage subset) means a stage added to the spine without either trips this
    # guard -- it cannot pass silently by listing a smaller curated set.
    default = rp.build_default_profile()
    for stage in STAGE_KEYS:
        assert stage in default.strategies, f"model-bearing stage {stage} has no profile strategy"
        assert stage in rp._ENVELOPE_BY_STAGE, f"stage {stage} has no host-envelope classification"


def test_audit_stages_track_production_playbook() -> None:
    # The guard must not be purely self-referential: every audit category in the
    # production playbook (daydream.improve.prompts) is itself a model-bearing
    # stage, so it must already be registered as a profile strategy + envelope
    # classification. A category added to production without a corresponding
    # STAGE_KEYS edit trips the guard instead of passing silently.
    from daydream.improve.prompts import AUDIT_PLAYBOOK_SECTIONS

    default = rp.build_default_profile()
    for category in AUDIT_PLAYBOOK_SECTIONS:
        stage = f"improve.audit.{category}"
        assert stage in STAGE_KEYS, (
            f"audit category `{category}` is not a registered review stage"
        )
        assert stage in default.strategies, (
            f"model-bearing audit stage {stage} has no profile strategy"
        )
        assert stage in rp._ENVELOPE_BY_STAGE, (
            f"stage {stage} has no host-envelope classification"
        )
