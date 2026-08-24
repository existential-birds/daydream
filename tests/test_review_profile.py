"""Task 1 (R2): strict review-profile model + stage schema.

Test-first per the implementation plan (tasks 1-13 of issue #885).
This task: strict model with stage schema. Digests (R3), parse (R4),
fail-closed validation (R3), host caps (R5), typed clone (R8), provenance (R9/R12),
and Harbor delivery (R10/R11) land in later tasks — not here.
"""
from daydream import review_profile as rp


def test_stage_keys_cover_every_model_bearing_stage():
    # Every named stage from spec R2 must be present (subset of the #886 manifest keys).
    assert {
        "exploration.repository_survey",
        "exploration.pattern_scan",
        "exploration.dependency_trace",
        "exploration.test_mapping",
        "intent",
        "alternatives",
        "discovery.per_stack",
        "discovery.structural",
        "discovery.generic_fallback",
        "parse",
        "uncovered_review",
        "arbitration",
        "suppression",
        "merge",
        "supervision",
        "verification",
    } <= set(rp.STAGE_KEYS)


def test_improve_audits_and_vetting_are_stages():
    assert {
        "improve.audit.correctness",
        "improve.audit.security",
        "improve.audit.performance",
        "improve.audit.tests",
        "improve.audit.tech-debt",
        "improve.audit.dependencies",
        "improve.audit.dx",
        "improve.audit.docs",
        "improve.vetting",
    } <= set(rp.STAGE_KEYS)
    for cat in ("correctness", "security", "performance", "tests",
                 "tech-debt", "dependencies", "dx", "docs"):
        assert f"improve.audit.{cat}" in rp.STAGE_KEYS


def test_default_profile_carries_schema_version_name_and_every_stage():
    p = rp.build_default_profile()
    assert p.schema_version == 1
    assert p.name  # human-readable, nonempty
    assert set(p.strategies) == set(rp.STAGE_KEYS)  # every stage present

    for key, strategy in p.strategies.items():
        assert strategy.content  # nonempty, real content (copied, not invented)
        assert strategy.source  # provenance string present ("copied:" / "authored:")


# Task 2 (R4): canonical serialization + deterministic digest.
import json

from daydream import review_profile as rp


def test_digest_is_order_whitespace_comment_path_independent(tmp_path):
    a = rp.parse_profile('''schema_version = 1
name = "p"
[strategies.intent]
content = "X"
source = "copied: a"''')
    b = rp.parse_profile('''schema_version=1
# a comment
name="p"
[strategies.intent]
source="copied: a"
content="X"''')
    assert a.digest == b.digest           # order/whitespace/comment independent


def test_digest_semantic_change_changes_digest():
    # A semantic change to a stage's strategy content changes the digest.
    base = rp.parse_profile('''schema_version = 1
name = "p"
[strategies.intent]
content = "X"
source = "copied: a"''')
    changed = rp.parse_profile('''schema_version = 1
name = "p"
[strategies.intent]
content = "DIFFERENT"
source = "copied: a"''')
    assert changed.digest != base.digest


def test_omitted_defaults_and_explicit_defaults_hash_identically():
    implicit = rp.parse_profile('''schema_version = 1
name = "p"
[strategies.intent]
content = "X"
source = "copied: a"''')
    explicit = rp.parse_profile('''schema_version = 1
name = "p"
[strategies.intent]
content = "X"
source = "copied: a"
[pipeline]
structural_enabled = true''')   # default value spelled out
    assert implicit.digest == explicit.digest