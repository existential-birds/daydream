"""Tests for the OpenAI cost-synthesis price table."""

from dataclasses import fields
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


@pytest.mark.parametrize("model", ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6"))
def test_gpt56_long_context_rates_apply_to_whole_request(model: str) -> None:
    """All GPT-5.6 ids use long-context rates only above the 272K boundary."""
    price = MODEL_PRICES[model]
    assert compute_cost_from_totals(model, total_input_tokens=272_000, cached_input_tokens=72_000, output_tokens=10_000
    ) == pytest.approx((200_000 * price.input + 72_000 * price.cached_input + 10_000 * price.output) / 1_000_000)
    assert compute_cost_from_totals(model, total_input_tokens=272_001, cached_input_tokens=72_000, output_tokens=10_000
    ) == pytest.approx(
        (200_001 * price.input * 2 + 72_000 * price.cached_input * 2 + 10_000 * price.output * 1.5) / 1_000_000
    )

def test_claude_sonnet_5_uses_the_rate_in_effect_on_the_usage_date() -> None:
    """Introductory pricing lasts through 2026-08-31."""
    assert compute_cost(
        model="claude-sonnet-5", input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=1_000_000,
        effective_date=date(2026, 8, 31),
    ) == pytest.approx(12.20)
    assert compute_cost(
        model="claude-sonnet-5", input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=1_000_000,
        effective_date=date(2026, 9, 1),
    ) == pytest.approx(18.30)


def test_compute_cost_zero_tokens_returns_zero() -> None:
    cost = compute_cost(model="gpt-5.5", input_tokens=0, cached_input_tokens=0, output_tokens=0,)
    assert cost == 0.0

def test_all_covered_models_have_complete_entries() -> None:
    required_models = {
        "gpt-5.5", "gpt-5.5-pro", "gpt-5-codex", "gpt-5.3-codex", "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra",
        "gpt-5.6-luna", "claude-sonnet-5",
    }
    assert required_models.issubset(MODEL_PRICES.keys())
    field_names = {f.name for f in fields(ModelPrice)}
    assert field_names == {"input", "cached_input", "output"}
    for name, price in MODEL_PRICES.items():
        assert isinstance(price, ModelPrice), name
        assert price.input >= 0, name
        assert price.cached_input >= 0, name
        assert price.output >= 0, name


@pytest.mark.parametrize(("model", "expected"),
    [
        # 100K input + 100K output at the published per-1M rates.
        ("gpt-5.6-sol", (5.00 + 30.00) * 0.1), ("gpt-5.6-terra", (2.50 + 15.00) * 0.1),
        ("gpt-5.6-luna", (1.00 + 6.00) * 0.1), ("gpt-5.6", (5.00 + 30.00) * 0.1),
        ("claude-sonnet-5", (3.00 + 15.00) * 0.1),
    ],
)
def test_new_default_model_ids_price_to_published_rates(model: str, expected: float) -> None:
    """Exact-match lookup requires every default id and alias to have a price entry."""
    cost = compute_cost(model=model, input_tokens=100_000, cached_input_tokens=0, output_tokens=100_000,)
    assert cost is not None, model
    assert cost > 0, model
    assert cost == pytest.approx(expected), model

def test_gpt56_bare_alias_matches_sol() -> None:
    assert MODEL_PRICES["gpt-5.6"] == MODEL_PRICES["gpt-5.6-sol"]

def test_gpt56_tiers_are_strictly_ordered_by_cost() -> None:
    sol, terra, luna = (MODEL_PRICES[f"gpt-5.6-{t}"] for t in ("sol", "terra", "luna"))
    assert sol.input > terra.input > luna.input
    assert sol.output > terra.output > luna.output
    assert sol.cached_input > terra.cached_input > luna.cached_input

def test_superseded_model_ids_still_price() -> None:
    """Archived trajectories still reference superseded model ids."""
    for legacy in ("gpt-5.5", "gpt-5.5-pro", "gpt-5-codex", "gpt-5.3-codex", "glm-5.2"):
        cost = compute_cost(model=legacy, input_tokens=1_000_000, cached_input_tokens=0, output_tokens=0,)
        assert cost is not None, legacy
        assert cost > 0, legacy


# --- User-overridable pricing -------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_user_prices_explicit_path_arg(tmp_path: Path) -> None:
    prices_file = _write(tmp_path / "p.toml", '[prices."explicit"]\ninput = 1.0\noutput = 3.0\n',)
    loaded = load_user_prices(path=prices_file)
    assert loaded == {"explicit": ModelPrice(input=1.0, cached_input=1.0, output=3.0)}


def test_load_user_prices_missing_required_field_skips_entry(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    prices_file = _write(tmp_path / "prices.toml",
        '[prices."has-output"]\noutput = 5.0\n[prices."ok"]\ninput = 1.0\noutput = 2.0\n',
    )
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert "has-output" not in loaded
    assert "ok" in loaded
    assert any("missing required field" in rec.message for rec in caplog.records)

@pytest.mark.parametrize(("model", "input_value", "warning"),
    [pytest.param("neg", "-1.0", "negative", id="negative"),
        pytest.param("nan-model", "nan", "non-finite", id="nan"),
        pytest.param("inf-model", "inf", "non-finite", id="positive-infinity"),
        pytest.param("neginf-model", "-inf", "non-finite", id="negative-infinity"),
    ],
)
def test_load_user_prices_invalid_value_skips_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, model: str, input_value: str, warning: str,
) -> None:
    prices_file = _write(tmp_path / "prices.toml", f'[prices."{model}"]\ninput = {input_value}\noutput = 2.0\n',)
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert loaded == {}
    assert any(warning in rec.message for rec in caplog.records)

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
