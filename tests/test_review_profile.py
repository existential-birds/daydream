"""Tests for the strict review-profile model and stage schema."""


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
    invalid += [(f"[pipeline]\nassignment_policy = {value}", "pipeline.assignment_policy")
                for value in ('"hierarchical"', '["adaptive"]')]
    for body, named in invalid:
        with pytest.raises(rp.ProfileError) as error:
            _profile(body, source=source)
        assert error.value.source == source
        assert source in str(error.value) and named in str(error.value)
    with pytest.raises(rp.ProfileError):
        _profile('''\
[pipeline]
arbitration_min_severity = "CRITICAL"''')  # not in the allowed severity enum
    # Host invariants are unoverridable.
    for field in ("backend", "model", "effort", "trust_mode", "egress",
                  "harbor_judge_model", "skill_name", "findings_schema",
                  "verifier", "judge", "scoring", "gold"):
        with pytest.raises(rp.ProfileError) as e:
            _profile(f'{field} = "x"', source="y")
        assert "host-owned" in str(e.value).lower() or field in str(e.value)
