"""Task 1 (R2): strict review-profile model + stage schema.

Test-first per the implementation plan (tasks 1-13 of issue #885).
This task: strict model with stage schema. Digests (R3), parse (R4),
fail-closed validation (R3), host caps (R5), typed clone (R8), provenance (R9/R12),
and Harbor delivery (R10/R11) land in later tasks — not here.
"""
import pytest

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

# Task 3 (R3): fail-closed validation.
def test_unknown_key_fails_closed_naming_source():
    with pytest.raises(rp.ProfileError) as e:
        rp.parse_profile(
            'schema_version = 1\nname = "p"\nbogus = 1',
            source="/tmp/profile.toml",
        )
    assert "/tmp/profile.toml" in str(e.value) and "bogus" in str(e.value)


def test_unsupported_schema_version_fails_closed():
    with pytest.raises(rp.ProfileError) as e:
        rp.parse_profile('schema_version = 99\nname = "p"', source="x")
    assert "schema_version" in str(e.value)


def test_negative_limit_fails_closed():
    with pytest.raises(rp.ProfileError):
        rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_max_files = -5''')


def test_invalid_enum_fails_closed():
    with pytest.raises(rp.ProfileError):
        rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
arbitration_min_severity = "CRITICAL"''')   # not in the allowed severity enum


def test_invalid_profile_never_falls_through_to_default():
    # A failed parse raises; it does not silently return build_default_profile().
    with pytest.raises(rp.ProfileError):
        rp.parse_profile('schema_version = 1\nname = "p"\nunknown = true')


# Task 4 (R5): host invariants unoverridable + host caps.
def test_forbidden_host_fields_rejected():
    for field in ("backend", "model", "effort", "trust_mode", "egress",
                  "harbor_judge_model", "skill_name", "findings_schema"):
        with pytest.raises(rp.ProfileError) as e:
            rp.parse_profile(f'schema_version = 1\nname = "p"\n{field} = "x"', source="y")
        assert "host-owned" in str(e.value).lower() or field in str(e.value)


def test_host_cap_clamps_lower_profile_value_up():
    # Host caps are the floor: a profile supplying LOWER than the host cap is clamped up.
    p = rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_min_hunk_lines = 2''')   # below host cap of 5
    assert p.pipeline.uncovered_sweep_min_hunk_lines == 5   # clamped up, never below


def test_uncovered_sweep_max_files_is_tunable():
    # The uncovered-sweep cap is a live profile knob, not a silent no-op locked
    # to the production default: a value inside the host band passes through.
    p = rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_max_files = 5''')   # within host band (1, 10)
    assert p.pipeline.uncovered_sweep_max_files == 5   # tunable, not forced to 10


def test_profile_cannot_raise_host_cap():
    p = rp.parse_profile('''schema_version = 1
name = "p"
[pipeline]
uncovered_sweep_max_files = 999''')   # above host cap
    assert p.pipeline.uncovered_sweep_max_files == 10   # capped at host ceiling

# Task 5 (R8): typed clone with overrides.
def test_clone_no_overrides_preserves_bytes_and_digest():
    base = rp.build_default_profile()
    clone = rp.clone_with_overrides(base, {})
    assert clone.digest == base.digest
    assert clone.to_canonical_dict() == base.to_canonical_dict()

def test_clone_one_override_changes_only_that_stage():
    base = rp.build_default_profile()
    clone = rp.clone_with_overrides(base, {"intent": {"content": "NEW INTENT STRATEGY"}})
    assert clone.digest != base.digest
    for key in rp.STAGE_KEYS:
        if key != "intent":
            assert clone.strategies[key].content == base.strategies[key].content  # byte-identical
    assert clone.strategies["intent"].content == "NEW INTENT STRATEGY"

def test_clone_override_revalidates_and_rejects_forbidden():
    import pytest
    base = rp.build_default_profile()
    with pytest.raises(rp.ProfileError):
        rp.clone_with_overrides(base, {"intent": {"backend": "claude"}})  # host-owned
