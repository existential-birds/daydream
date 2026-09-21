"""Tests for the strict review-profile model and stage schema."""
from pathlib import Path

import pytest

from daydream import review_profile as rp
from tests.test_review_profile_completeness import STAGE_KEYS


def test_review_deadline_is_profiled_and_validated() -> None:
    base = rp.parse_profile('schema_version = 1\nname = "p"')
    bounded = rp.parse_profile('schema_version = 1\nname = "p"\n[pipeline]\nreview_wall_budget_s = 1200')
    assert bounded.pipeline.review_wall_budget_s == 1200
    assert base.pipeline.review_wall_budget_s == 2700
    assert bounded.digest != base.digest
    with pytest.raises(rp.ProfileError):
        rp.parse_profile('schema_version = 1\nname = "p"\n[pipeline]\nreview_wall_budget_s = -1')


def test_pipeline_keeps_existing_positional_constructor_order() -> None:
    pipeline = rp.Pipeline(False)
    assert pipeline.structural_enabled is False
    assert pipeline.review_wall_budget_s == 2700


def test_stage_keys_cover_every_model_bearing_stage() -> None:
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
        "uncovered_review",
        "arbitration",
        "suppression",
        "merge",
        "supervision",
        "verification",
    } <= set(STAGE_KEYS)


def test_improve_audits_and_vetting_are_stages() -> None:
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
    } <= set(STAGE_KEYS)


def test_default_profile_carries_schema_version_name_and_every_stage() -> None:
    p = rp.build_default_profile()
    assert p.schema_version == 1
    assert p.name  # human-readable, nonempty
    assert set(p.strategies) == set(STAGE_KEYS)  # every stage present

    for _key, strategy in p.strategies.items():
        assert strategy.content  # nonempty, real content (copied, not invented)
        assert strategy.source  # provenance string present ("copied:" / "authored:")


# Task 2 (R4): canonical serialization + deterministic digest.
def test_digest_is_order_whitespace_comment_path_independent(tmp_path: Path) -> None:
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


def test_digest_semantic_change_changes_digest() -> None:
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


def test_omitted_defaults_and_explicit_defaults_hash_identically() -> None:
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

# Task 3 (R3): fail-closed validation.
def test_unknown_key_fails_closed_naming_source() -> None:
    with pytest.raises(rp.ProfileError) as e:
        rp.parse_profile(
            'schema_version = 1\nname = "p"\nbogus = 1',
            source="/tmp/profile.toml",
        )
    assert "/tmp/profile.toml" in str(e.value) and "bogus" in str(e.value)


def test_unsupported_schema_version_fails_closed() -> None:
    with pytest.raises(rp.ProfileError) as e:
        rp.parse_profile('schema_version = 99\nname = "p"', source="x")
    assert "schema_version" in str(e.value)


def test_negative_limit_fails_closed() -> None:
    with pytest.raises(rp.ProfileError):
        rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_max_files = -5''')


def test_invalid_enum_fails_closed() -> None:
    with pytest.raises(rp.ProfileError):
        rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
arbitration_min_severity = "CRITICAL"''')   # not in the allowed severity enum


# Task 4 (R5): host invariants unoverridable + host caps.
def test_forbidden_host_fields_rejected() -> None:
    for field in ("backend", "model", "effort", "trust_mode", "egress",
                  "harbor_judge_model", "skill_name", "findings_schema"):
        with pytest.raises(rp.ProfileError) as e:
            rp.parse_profile(f'schema_version = 1\nname = "p"\n{field} = "x"', source="y")
        assert "host-owned" in str(e.value).lower() or field in str(e.value)


def test_host_cap_clamps_lower_profile_value_up() -> None:
    # Host caps are the floor: a profile supplying LOWER than the host cap is clamped up.
    p = rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_min_hunk_lines = 2''')   # below host cap of 5
    assert p.pipeline.uncovered_sweep_min_hunk_lines == 5   # clamped up, never below


def test_uncovered_sweep_max_files_is_tunable() -> None:
    # The uncovered-sweep cap is a live profile knob, not a silent no-op locked
    # to the production default: a value inside the host band passes through.
    p = rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_max_files = 5''')   # within host band (1, 10)
    assert p.pipeline.uncovered_sweep_max_files == 5   # tunable, not forced to 10


def test_profile_cannot_raise_host_cap() -> None:
    p = rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_max_files = 999''')   # above host cap
    assert p.pipeline.uncovered_sweep_max_files == 10   # capped at host ceiling


def test_suppression_severity_classes_default_narrowed() -> None:
    assert rp.Suppression.severity_classes == ("low",)


def test_review_profile_severity_levels_derive_from_severity_module() -> None:
    from daydream import severity

    assert rp._SEVERITY_LEVELS == frozenset(severity.CANONICAL_LEVELS)
