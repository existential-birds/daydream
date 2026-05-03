"""Tests for the OpenAI cost-synthesis price table."""

from dataclasses import fields

import pytest

from daydream.pricing import MODEL_PRICES, ModelPrice, compute_cost


def test_compute_cost_gpt55_baseline() -> None:
    """gpt-5.5: 10K input + 5K cached + 2K output = $0.05 + $0.0025 + $0.06 = $0.1125."""
    cost = compute_cost(
        model="gpt-5.5",
        input_tokens=10_000,
        cached_input_tokens=5_000,
        output_tokens=2_000,
    )
    assert cost is not None
    # 10_000 * 5.00 / 1e6 + 5_000 * 0.50 / 1e6 + 2_000 * 30.00 / 1e6
    expected = 0.05 + 0.0025 + 0.06
    assert cost == pytest.approx(expected)


def test_compute_cost_unknown_model_returns_none() -> None:
    """Unknown model identifiers must return None — never a guess."""
    assert (
        compute_cost(
            model="not-a-real-model",
            input_tokens=1_000,
            cached_input_tokens=0,
            output_tokens=1_000,
        )
        is None
    )


def test_compute_cost_zero_tokens_returns_zero() -> None:
    """Zero tokens across all categories must return exactly 0.0."""
    cost = compute_cost(
        model="gpt-5.5",
        input_tokens=0,
        cached_input_tokens=0,
        output_tokens=0,
    )
    assert cost == 0.0


def test_all_covered_models_have_complete_entries() -> None:
    """Every model in MODEL_PRICES must have all ModelPrice fields populated and non-negative."""
    required_models = {"gpt-5.5", "gpt-5.5-pro", "gpt-5-codex", "gpt-5.3-codex"}
    assert required_models.issubset(MODEL_PRICES.keys())
    field_names = {f.name for f in fields(ModelPrice)}
    assert field_names == {"input", "cached_input", "output"}
    for name, price in MODEL_PRICES.items():
        assert isinstance(price, ModelPrice), name
        assert price.input >= 0, name
        assert price.cached_input >= 0, name
        assert price.output >= 0, name


def test_cached_tokens_priced_separately_from_input() -> None:
    """For gpt-5.5 the cached input price ($0.50) is distinct from input ($5.00).

    Routing the same token count through cached_input must produce a strictly
    smaller cost than routing it through input — proves cached tokens use the
    cached_input rate, not the input rate.
    """
    input_only = compute_cost(
        model="gpt-5.5",
        input_tokens=100_000,
        cached_input_tokens=0,
        output_tokens=0,
    )
    cached_only = compute_cost(
        model="gpt-5.5",
        input_tokens=0,
        cached_input_tokens=100_000,
        output_tokens=0,
    )
    assert input_only is not None
    assert cached_only is not None
    assert cached_only < input_only
    # Concrete values: 100K * $5/1M = $0.50; 100K * $0.50/1M = $0.05
    assert input_only == pytest.approx(0.50)
    assert cached_only == pytest.approx(0.05)
