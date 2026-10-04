"""Tests for the strict review-profile model and stage schema."""
from pathlib import Path

import pytest

from daydream import review_profile as rp, severity
from tests.test_review_profile_completeness import STAGE_KEYS


def _profile(body: str = "", *, source: str = "<string>") -> rp.ReviewProfile:
    return rp.parse_profile(f'schema_version = 1\nname = "p"\n{body}', source=source)

def test_review_deadline_is_profiled_and_validated() -> None:
    base = _profile()
    bounded = _profile("[pipeline]\nreview_wall_budget_s = 1200")
    assert bounded.pipeline.review_wall_budget_s == 1200
    assert base.pipeline.review_wall_budget_s == 2700
    assert bounded.digest != base.digest
    with pytest.raises(rp.ProfileError):
        _profile("[pipeline]\nreview_wall_budget_s = -1")

def test_pipeline_keeps_existing_positional_constructor_order() -> None:
    pipeline = rp.Pipeline(False)
    assert pipeline.structural_enabled is False
    assert pipeline.review_wall_budget_s == 2700

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
    a = _profile('''[strategies.intent]
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
    base = _profile('''[strategies.intent]
content = "X"
source = "copied: a"''')
    changed = _profile('''[strategies.intent]
content = "DIFFERENT"
source = "copied: a"''')
    assert changed.digest != base.digest

def test_omitted_defaults_and_explicit_defaults_hash_identically() -> None:
    implicit = _profile('''[strategies.intent]
content = "X"
source = "copied: a"''')
    explicit = _profile('''[strategies.intent]
content = "X"
source = "copied: a"
[pipeline]
structural_enabled = true''')   # default value spelled out
    assert implicit.digest == explicit.digest

# Task 3 (R3): fail-closed validation.
def test_unknown_key_fails_closed_naming_source() -> None:
    with pytest.raises(rp.ProfileError) as e:
        _profile("bogus = 1", source="/tmp/profile.toml")
    assert "/tmp/profile.toml" in str(e.value) and "bogus" in str(e.value)

def test_unsupported_schema_version_fails_closed() -> None:
    with pytest.raises(rp.ProfileError) as e:
        rp.parse_profile('schema_version = 99\nname = "p"', source="x")
    assert "schema_version" in str(e.value)

def test_invalid_enum_fails_closed() -> None:
    with pytest.raises(rp.ProfileError):
        _profile('''\
[pipeline]
arbitration_min_severity = "CRITICAL"''')   # not in the allowed severity enum


# Task 4 (R5): host invariants unoverridable + host caps.
def test_forbidden_host_fields_rejected() -> None:
    for field in ("backend", "model", "effort", "trust_mode", "egress",
                  "harbor_judge_model", "skill_name", "findings_schema",
                  "verifier", "judge", "scoring", "gold"):
        with pytest.raises(rp.ProfileError) as e:
            _profile(f'{field} = "x"', source="y")
        assert "host-owned" in str(e.value).lower() or field in str(e.value)

def test_suppression_severity_classes_default_narrowed() -> None:
    assert rp.Suppression.severity_classes == ("low",)

def test_review_profile_severity_levels_derive_from_severity_module() -> None:
    assert rp._SEVERITY_LEVELS == frozenset(severity.CANONICAL_LEVELS)
