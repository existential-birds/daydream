"""Every model-bearing stage requires a profile strategy."""

from daydream import review_profile as rp
from daydream.improve.prompts import AUDIT_PLAYBOOK_SECTIONS

# Expected model stages span review and Improve judgments.
STAGE_KEYS: frozenset[str] = frozenset({
        "exploration.repository_survey", "exploration.pattern_scan", "exploration.dependency_trace",
        "exploration.test_mapping", "intent", "alternatives", "discovery.per_stack", "discovery.structural",
        "discovery.generic_fallback", "arbitration", "suppression", "merge", "supervision", "verification",
        "improve.audit.correctness", "improve.audit.security", "improve.audit.performance", "improve.audit.tests",
        "improve.audit.tech-debt", "improve.audit.dependencies", "improve.audit.dx", "improve.audit.docs",
        "improve.vetting",
    }
)

def test_every_registered_model_bearing_stage_has_strategy() -> None:
    # Keep the inventory independent from the packaged default so missing strategies fail.
    default = rp.build_default_profile()
    for stage in STAGE_KEYS:
        assert stage in default.strategies, f"model-bearing stage {stage} has no profile strategy"

def test_audit_stages_track_production_playbook() -> None:
    # Check playbook categories independently so omitted registry entries cannot pass self-referential checks.
    default = rp.build_default_profile()
    for category in AUDIT_PLAYBOOK_SECTIONS:
        stage = f"improve.audit.{category}"
        assert stage in STAGE_KEYS, (f"audit category `{category}` is not a registered review stage")
        assert stage in default.strategies, (f"model-bearing audit stage {stage} has no profile strategy")
