"""Tests for the strict review-profile model and stage schema."""
from pathlib import Path
from typing import Any

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

def test_stage_keys_cover_every_model_bearing_stage() -> None:
    # Every named stage from spec R2 must be present (subset of the #886 manifest keys).
    assert {"exploration.repository_survey", "exploration.pattern_scan", "exploration.dependency_trace",
        "exploration.test_mapping", "intent", "alternatives", "discovery.per_stack", "discovery.structural",
        "discovery.generic_fallback", "arbitration", "suppression", "merge", "supervision", "verification",
    } <= set(STAGE_KEYS)

def test_improve_audits_and_vetting_are_stages() -> None:
    assert {"improve.audit.correctness", "improve.audit.security", "improve.audit.performance", "improve.audit.tests",
        "improve.audit.tech-debt", "improve.audit.dependencies", "improve.audit.dx", "improve.audit.docs",
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
    for level in severity.CANONICAL_LEVELS:
        assert _profile(f'[pipeline]\narbitration_min_severity = "{level}"').pipeline.arbitration.min_severity == level
        suppression = _profile(f'[pipeline]\nsuppression_severity_classes = ["{level}"]').pipeline.suppression
        assert suppression.severity_classes == (level,)



@pytest.mark.parametrize(("owner", "values"), [
    (rp.Pipeline, {"structural_enabled": 1}),
    (rp.Pipeline, {"review_wall_budget_s": True}),
    (rp.Pipeline, {"review_wall_budget_s": -1}),
    (rp.Arbitration, {"min_severity": "HIGH"}),
    (rp.Suppression, {"enabled": True, "confidence_classes": ()}),
    (rp.Suppression, {"severity_classes": {"low"}}),
    (rp.Strategy, {"content": 1}),
    (rp.ReviewProfile, {"schema_version": True}),
    (rp.ReviewProfile, {"schema_version": 2}),
])
def test_domain_owner_rejects_invalid_policy_before_use(owner: Any, values: dict[str, Any]) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        owner(**values)


def test_admitted_policy_retains_dataclass_replacement_and_frozen_semantics() -> None:
    from dataclasses import FrozenInstanceError, asdict, replace

    arbitration_input: dict[str, Any] = {"min_severity": None}
    suppression_input: dict[str, Any] = {"severity_classes": ["medium", "low"], "confidence_classes": ["HIGH", "LOW"]}
    pipeline = rp.Pipeline(
        arbitration=rp.Arbitration(**arbitration_input), suppression=rp.Suppression(**suppression_input),
    )
    assert pipeline.arbitration.min_severity == "high"
    assert pipeline.suppression.severity_classes == ("medium", "low")
    assert pipeline.suppression.confidence_classes == ("HIGH", "LOW")
    changed = replace(pipeline, structural_enabled=False)
    assert changed.structural_enabled is False and pipeline.structural_enabled is True
    assert asdict(changed)["arbitration"] == {"enabled": True, "min_severity": "high", "contested_location": True}
    with pytest.raises(FrozenInstanceError):
        setattr(changed, "structural_enabled", True)


@pytest.mark.parametrize("scope", ["", "strategies.intent"])
def test_host_policy_denial_precedes_unknown_policy_fields(scope: str) -> None:
    text = (f"[{scope}]\n" if scope else "") + 'backend = "private"\nfuture = "x"'
    with pytest.raises(rp.ProfileError, match="host-owned.*backend.*profile") as error:
        rp.parse_profile(text, source="owner.toml")
    assert error.value.source == "owner.toml"
