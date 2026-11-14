"""Tests for the strict review-profile model and stage schema."""

from dataclasses import asdict

import pytest

from daydream import review_profile as rp


def _profile(body: str = "", *, source: str = "<string>") -> rp.ReviewProfile:
    return rp.parse_profile(f'schema_version = 1\nname = "p"\n{body}', source=source)


def test_review_deadline_is_profiled_and_validated() -> None:
    base = _profile("[pipeline]\nreview_wall_budget_s = 1800")
    bounded = _profile("[pipeline]\nreview_wall_budget_s = 1200")
    assert bounded.pipeline.review_wall_budget_s == 1200
    assert bounded.digest != base.digest
    with pytest.raises(rp.ProfileError):
        _profile("[pipeline]\nreview_wall_budget_s = -1")


def test_invalid_profiles_fail_closed_naming_field_and_source() -> None:
    source = "/tmp/reviewer-profile.toml"
    invalid = [(f"[pipeline]\n{field} = {value}", f"pipeline.{field}")
               for field in ("reviewer_investigation_budget_s", "reviewer_tool_call_budget")
               for value in ("-1", "true")]
    invalid += [("bogus = 1", "bogus"),
                ("[pipeline]\nreviewer_investigation_budget_seconds = 480", "reviewer_investigation_budget_seconds")]
    # The removed policy key is refused whatever its value.
    invalid += [(f"[pipeline]\nassignment_policy = {value}", "pipeline.assignment_policy: removed")
                for value in ('"broad"', '"adaptive"')]
    # So is every key of the replaced staged reviewer, each named, valid values included.
    removed = {"pipeline.assignment_policy": '"broad"', "pipeline.structural_enabled": "false",
               "pipeline.arbitration_enabled": "false", "pipeline.arbitration_min_severity": '"high"',
               "pipeline.arbitration_contested_location": "false", "pipeline.suppression_enabled": "false",
               "pipeline.suppression_severity_classes": '["low"]', "pipeline.suppression_confidence_classes": '["LOW"]'}
    strategies = ("intent", "alternatives", "discovery.per_stack", "discovery.structural", "suppression", "merge",
                  "supervision", "arbitration")
    every = "[pipeline]\n" + "".join(f"{key.split('.')[1]} = {value}\n" for key, value in removed.items()) + "".join(
        f'[strategies."{name}"]\ncontent = "x"\nsource = "authored: test"\n' for name in strategies)
    invalid += [(every, f"{key}: removed") for key in [*removed, *(f"strategies.{name}" for name in strategies)]]
    for body, named in invalid:
        with pytest.raises(rp.ProfileError) as error:
            _profile(body, source=source)
        assert error.value.source == source
        assert source in str(error.value) and named in str(error.value)
    # None of them is in the packaged default, so none counts in its digest.
    default = rp.build_default_profile()
    assert set(asdict(default.pipeline)) == {"review_wall_budget_s", "reviewer_investigation_budget_s",
                                             "reviewer_tool_call_budget"}
    assert not set(default.strategies) & set(strategies)
    # Host invariants are unoverridable.
    for field in ("backend", "model", "effort", "trust_mode", "egress",
                  "harbor_judge_model", "skill_name", "findings_schema",
                  "verifier", "judge", "scoring", "gold"):
        with pytest.raises(rp.ProfileError) as e:
            _profile(f'{field} = "x"', source="y")
        assert "host-owned" in str(e.value).lower() or field in str(e.value)
