"""Stage-0 rubric v2 tests: learned outcome term + CR-Bench FP penalty (M2, M5, M6, M7)."""

from typing import Any, cast

import pytest

from daydream.training.rubric_v2 import (
    REWARD_VERSION_RUBRIC,
    RubricV2Breakdown,
    RubricV2Weights,
    _rubric_fingerprint,
    score_review,
)


class _StubModel:
    """Tiny stub satisfying the OutcomeModel scoring protocol."""

    def score_comment(self, text: str) -> float:  # noqa: ARG002
        return 0.5


@pytest.fixture
def model() -> _StubModel:
    return _StubModel()


def _finding(text: str = "nit", fid: str | None = None, **extra: Any) -> dict[str, Any]:
    f: dict[str, Any] = {"id": fid or text, "text": text}
    f.update(extra)
    return f


def test_all_noise_scores_below_clean_with_same_recall(model: _StubModel) -> None:
    noise = cast(
        float,
        score_review(
            model,
            findings=[_finding("vague") for _ in range(5)],
            fp_count=5,
            total_findings=5,
            grounded=0,
        ),
    )
    clean = cast(
        float,
        score_review(
            model,
            findings=[_finding("real bug, line 3") for _ in range(5)],
            fp_count=0,
            total_findings=5,
            grounded=5,
        ),
    )
    assert noise < clean  # M5: strictly below, same recall, no noise


def test_fp_penalty_term_present_and_dominant_direction(model: _StubModel) -> None:
    w = RubricV2Weights()
    assert w.w_false_positive > 0
    b = cast(
        RubricV2Breakdown,
        score_review(
            model,
            findings=[_finding()],
            fp_count=3,
            total_findings=3,
            grounded=1,
            breakdown=True,
        ),
    )
    fp_term = b.terms["fp_penalty"]
    assert b.false_positive_penalty is not None and b.false_positive_penalty > 0
    assert fp_term is not None and fp_term < 0  # explicit subtractive term


def test_version_fingerprint_changes_on_weight_change() -> None:
    f1 = _rubric_fingerprint(RubricV2Weights())
    f2 = _rubric_fingerprint(RubricV2Weights(w_false_positive=0.5))
    assert f1 != f2  # M7 discipline: formula identity detectable


def test_breakdown_stamps_rubric_version(model: _StubModel) -> None:
    b = cast(
        RubricV2Breakdown,
        score_review(model, findings=[_finding()], fp_count=0, total_findings=1, grounded=1, breakdown=True),
    )
    assert b.reward_version.startswith(REWARD_VERSION_RUBRIC)


def test_missing_signal_is_none_not_zero(model: _StubModel) -> None:
    # No tool signals on any finding -> tool_grounded term absent (None), never 0.0.
    b = cast(
        RubricV2Breakdown,
        score_review(model, findings=[_finding()], fp_count=0, total_findings=1, grounded=1, breakdown=True),
    )
    assert b.terms["tool_grounded"] is None


def test_zero_total_findings_guards_fp_and_snr_terms(model: _StubModel) -> None:
    # Regression: fp/total and (total-fp)/total must not divide by zero when
    # total_findings == 0; the ratio terms are then absent (None), never 0.0.
    b = cast(
        RubricV2Breakdown,
        score_review(model, findings=[_finding()], fp_count=0, total_findings=0, grounded=0, breakdown=True),
    )
    assert b.false_positive_penalty is None
    assert b.terms["fp_penalty"] is None  # renormalized out, not imputed 0.0
    assert b.signal_to_noise is None
    assert b.composite == 0.5  # learned term alone renormalized


def test_malformed_finding_raises_with_id(model: _StubModel) -> None:
    with pytest.raises(ValueError, match="f-9"):
        score_review(model, findings=[{"id": "f-9"}], fp_count=0, total_findings=1, grounded=1)
