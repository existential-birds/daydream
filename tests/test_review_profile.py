"""Tests for the strict review-profile model and stage schema."""
from pathlib import Path

import pytest

from daydream import review_profile as rp
from tests.harness.table_cases import isolated_case


def _profile(body: str = "", *, source: str = "<string>") -> rp.ReviewProfile:
    return rp.parse_profile(f'schema_version = 1\nname = "p"\n{body}', source=source)

def test_review_deadline_is_profiled_and_validated() -> None:
    base = _profile("[pipeline]\nreview_wall_budget_s = 1800")
    bounded = _profile("[pipeline]\nreview_wall_budget_s = 1200")
    assert bounded.pipeline.review_wall_budget_s == 1200
    assert bounded.digest != base.digest
    with pytest.raises(rp.ProfileError):
        _profile("[pipeline]\nreview_wall_budget_s = -1")


def test_reviewer_investigation_budget_is_independent_and_changes_identity(
    subtests: pytest.Subtests,
) -> None:
    cases: list[int] = [0, 900]
    for seconds in cases:
        with subtests.test(seconds=seconds):
            base = _profile("[pipeline]\nreview_wall_budget_s = 1800\nreviewer_investigation_budget_s = 600")
            configured = _profile(
                f"[pipeline]\nreview_wall_budget_s = 1800\nreviewer_investigation_budget_s = {seconds}"
            )
            assert configured.pipeline.reviewer_investigation_budget_s == seconds
            assert configured.pipeline.review_wall_budget_s == base.pipeline.review_wall_budget_s
            assert configured.to_canonical_dict() != base.to_canonical_dict()
            assert configured.digest != base.digest


def test_invalid_reviewer_investigation_budget_names_field_and_source(
    subtests: pytest.Subtests,
) -> None:
    cases: list[str] = ["-1", "true", "false", "1.5", '"480"', "[]"]
    for value in cases:
        with subtests.test(value=value):
            source = "/tmp/reviewer-profile.toml"
            with pytest.raises(rp.ProfileError) as error:
                _profile(f"[pipeline]\nreviewer_investigation_budget_s = {value}", source=source)
            assert error.value.source == source
            assert source in str(error.value)
            assert "pipeline.reviewer_investigation_budget_s" in str(error.value)


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

def test_omitted_defaults_and_explicit_defaults_hash_identically() -> None:
    implicit = _profile('''[strategies.intent]
content = "X"
source = "copied: a"''')
    explicit = _profile(f'''[strategies.intent]
content = "X"
source = "copied: a"
[pipeline]
structural_enabled = true
reviewer_investigation_budget_s = {rp.Pipeline().reviewer_investigation_budget_s}''')
    assert implicit.to_canonical_dict() == explicit.to_canonical_dict()
    assert implicit.digest == explicit.digest

# Task 3 (R3): fail-closed validation.
def test_unknown_key_fails_closed_naming_source(
    subtests: pytest.Subtests,
) -> None:
    cases: list[tuple[str, str]] = [
        ("bogus = 1", "bogus"),
        ("[pipeline]\nreviewer_investigation_budget_seconds = 480", "reviewer_investigation_budget_seconds"),
    ]
    for body, unknown_key in cases:
        with subtests.test(body=body):
            with pytest.raises(rp.ProfileError) as e:
                _profile(body, source="/tmp/profile.toml")
            assert "/tmp/profile.toml" in str(e.value) and unknown_key in str(e.value)

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


def test_composed_review_search_guidance_is_independent_of_strategy(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    cases_1: list[bool] = [False, True]
    for custom in cases_1:
        cases_2: list[str] = ["per_stack", "structural", "generic_fallback"]
        for kind in cases_2:
            row_index += 1
            with subtests.test(custom=custom, kind=kind):
                with isolated_case(table_root, str(row_index)) as case:
                    tmp_path = case.tmp_path
                    from daydream.deep import prompts
                    from daydream.prompts.grounding import REPOSITORY_SEARCH_GUIDANCE
                    from tests.harness.review_profile import prompt_paths

                    strategy = (
                        "Custom judgment policy"
                        if custom
                        else rp.build_default_profile().strategies[f"discovery.{kind}"].content
                    )
                    kwargs = {"stack_name": "python"} if kind == "per_stack" else {}
                    builder = getattr(prompts, f"build_{kind}_prompt")
                    prompt = builder(strategy=strategy, files=["src/main.py"], **prompt_paths(tmp_path), **kwargs)
                    assert strategy in prompt
                    assert REPOSITORY_SEARCH_GUIDANCE in prompt
