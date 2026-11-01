"""Tests for the OpenAI cost-synthesis price table."""

from datetime import date
from pathlib import Path

import pytest

from daydream.pricing import (
    MODEL_PRICES,
    ModelPrice,
    compute_cost,
    compute_cost_from_totals,
    load_user_prices,
    resolve_prices,
)


@pytest.mark.parametrize("model", [
        "gpt-5.6-sol",
        "openai/gpt-6.1-sol",
        "gpt-6.1-sol",
    ])
def test_openai_long_context_rates_apply_to_whole_request(model: str) -> None:
    """GPT-5.6 and GPT-6.1 Sol use long-context rates only above the 272K boundary."""
    price = MODEL_PRICES[model]
    assert compute_cost_from_totals(model, total_input_tokens=272_000, cached_input_tokens=72_000, output_tokens=10_000
    ) == pytest.approx((200_000 * price.input + 72_000 * price.cached_input + 10_000 * price.output) / 1_000_000)
    assert compute_cost_from_totals(model, total_input_tokens=272_001, cached_input_tokens=72_000, output_tokens=10_000
    ) == pytest.approx(
        (200_001 * price.input * 2 + 72_000 * price.cached_input * 2 + 10_000 * price.output * 1.5) / 1_000_000
    )

def test_claude_sonnet_5_uses_the_rate_in_effect_on_the_usage_date() -> None:
    """Introductory pricing lasts through 2026-08-31; an omitted date uses standard rates."""
    tokens = (1_000_000, 1_000_000, 1_000_000)
    assert compute_cost("claude-sonnet-5", *tokens, effective_date=date(2026, 8, 31)) == pytest.approx(12.20)
    assert compute_cost("claude-sonnet-5", *tokens, effective_date=date(2026, 9, 1)) == pytest.approx(18.30)
    assert compute_cost("claude-sonnet-5", *tokens) == pytest.approx(18.30)


@pytest.mark.parametrize(("model", "expected"),
    [
        ("gpt-5.6-terra", 1.775), ("gpt-5.6-luna", 0.71),
        ("gpt-5.6", 3.55),
    ],
)
def test_new_default_model_ids_price_to_published_rates(model: str, expected: float) -> None:
    """Distinct GPT-5.6 policies must use their own baseline rates."""
    cost = compute_cost(model=model, input_tokens=100_000, cached_input_tokens=100_000, output_tokens=100_000,)
    assert cost is not None, model
    assert cost > 0, model
    assert cost == pytest.approx(expected), model


# --- User-overridable pricing -------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_user_prices_missing_required_field_skips_entry(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    prices_file = _write(tmp_path / "prices.toml",
        '[prices."has-output"]\noutput = 5.0\n[prices."neg"]\ninput = -1.0\noutput = 2.0\n'
        '[prices."nan-model"]\ninput = nan\noutput = 2.0\n[prices."ok"]\ninput = 1.0\noutput = 2.0\n',
    )
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert "has-output" not in loaded
    assert "neg" not in loaded
    assert "nan-model" not in loaded
    assert loaded == {"ok": ModelPrice(input=1.0, cached_input=1.0, output=2.0)}
    assert any("'has-output' missing required field 'input'" in rec.message for rec in caplog.records)
    assert any("'neg' field 'input' is negative" in rec.message for rec in caplog.records)
    assert any("'nan-model' field 'input' is non-finite" in rec.message for rec in caplog.records)

def test_load_user_prices_unresolvable_home_returns_empty(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("DAYDREAM_PRICES_FILE", raising=False)

    def _raise() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(_raise))
    with caplog.at_level("WARNING"):
        loaded = load_user_prices()
    assert loaded == {}
    assert any("could not resolve home directory" in rec.message.lower() for rec in caplog.records)


def test_resolve_prices_none_returns_builtin_copy() -> None:
    """The returned table must be independently mutable."""
    resolved = resolve_prices()
    assert resolved == MODEL_PRICES
    assert resolved is not MODEL_PRICES

    resolved["gpt-5.5"] = ModelPrice(input=0.0, cached_input=0.0, output=0.0)
    assert MODEL_PRICES["gpt-5.5"] == ModelPrice(input=5.0, cached_input=0.5, output=30.0)
