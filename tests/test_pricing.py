"""Tests for the OpenAI cost-synthesis price table."""

from dataclasses import fields
from pathlib import Path

import pytest

from daydream.pricing import (
    MODEL_PRICES,
    ModelPrice,
    compute_cost,
    load_user_prices,
    resolve_prices,
)


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


# --- User-overridable pricing -------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_user_prices_valid_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid prices.toml yields the parsed ModelPrice via the env-var path."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."my-model"]\ninput = 2.0\ncached_input = 0.5\noutput = 8.0\n',
    )
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))
    loaded = load_user_prices()
    assert loaded == {"my-model": ModelPrice(input=2.0, cached_input=0.5, output=8.0)}


def test_load_user_prices_explicit_path_arg(tmp_path: Path) -> None:
    """The explicit path argument is honored without any env var set."""
    prices_file = _write(
        tmp_path / "p.toml",
        '[prices."explicit"]\ninput = 1.0\noutput = 3.0\n',
    )
    loaded = load_user_prices(path=prices_file)
    assert loaded == {"explicit": ModelPrice(input=1.0, cached_input=1.0, output=3.0)}


def test_load_user_prices_override_builtin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A user entry keyed to a built-in model name is loaded as an override."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."gpt-5.5"]\ninput = 1.0\ncached_input = 0.1\noutput = 2.0\n',
    )
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))
    loaded = load_user_prices()
    assert loaded["gpt-5.5"] == ModelPrice(input=1.0, cached_input=0.1, output=2.0)


def test_load_user_prices_add_new_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A user entry for an unknown model is loaded as a new entry."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."brand-new"]\ninput = 9.0\noutput = 90.0\n',
    )
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))
    loaded = load_user_prices()
    assert "brand-new" in loaded
    assert "brand-new" not in MODEL_PRICES


def test_load_user_prices_cached_input_defaults_to_input(tmp_path: Path) -> None:
    """Omitting cached_input defaults it to the input price."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."m"]\ninput = 4.0\noutput = 12.0\n',
    )
    loaded = load_user_prices(path=prices_file)
    assert loaded["m"].cached_input == 4.0


def test_load_user_prices_malformed_toml_returns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed TOML logs a warning and yields {} — never raises."""
    bad = _write(tmp_path / "prices.toml", "this is = = not toml")
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=bad)
    assert loaded == {}
    assert any("malformed" in rec.message.lower() for rec in caplog.records)


def test_load_user_prices_absent_file_returns_empty(tmp_path: Path) -> None:
    """An absent file yields {} without raising."""
    assert load_user_prices(path=tmp_path / "nope.toml") == {}


def test_load_user_prices_missing_required_field_skips_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An entry missing a required field is logged and skipped."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."has-output"]\noutput = 5.0\n[prices."ok"]\ninput = 1.0\noutput = 2.0\n',
    )
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert "has-output" not in loaded
    assert "ok" in loaded
    assert any("missing required field" in rec.message for rec in caplog.records)


def test_load_user_prices_negative_value_skips_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A negative price value is logged and skips the entry."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."neg"]\ninput = -1.0\noutput = 2.0\n',
    )
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert loaded == {}
    assert any("negative" in rec.message for rec in caplog.records)


def test_load_user_prices_nan_value_skips_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A nan price value is logged and skips the entry."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."nan-model"]\ninput = nan\noutput = 2.0\n',
    )
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert loaded == {}
    assert any("non-finite" in rec.message for rec in caplog.records)


def test_load_user_prices_inf_value_skips_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A +inf or -inf price value is logged and skips the entry."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."inf-model"]\ninput = inf\noutput = 2.0\n',
    )
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert loaded == {}
    assert any("non-finite" in rec.message for rec in caplog.records)


def test_load_user_prices_negative_inf_value_skips_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """-inf is rejected as non-finite (not as negative) and skips the entry."""
    prices_file = _write(
        tmp_path / "prices.toml",
        '[prices."neginf-model"]\ninput = -inf\noutput = 2.0\n',
    )
    with caplog.at_level("WARNING"):
        loaded = load_user_prices(path=prices_file)
    assert loaded == {}
    assert any("non-finite" in rec.message for rec in caplog.records)


def test_load_user_prices_unresolvable_home_returns_empty(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When the default path is used and Path.home() raises, yield {} — never raise."""
    monkeypatch.delenv("DAYDREAM_PRICES_FILE", raising=False)

    def _raise() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(_raise))
    with caplog.at_level("WARNING"):
        loaded = load_user_prices()
    assert loaded == {}
    assert any("could not resolve home directory" in rec.message.lower() for rec in caplog.records)


def test_resolve_prices_override_wins_per_model() -> None:
    """resolve_prices applies overrides per-model, leaving other built-ins intact."""
    override = ModelPrice(input=1.0, cached_input=0.1, output=2.0)
    merged = resolve_prices({"gpt-5.5": override})
    assert merged["gpt-5.5"] == override
    # A non-overridden built-in is preserved unchanged.
    assert merged["gpt-5-codex"] == MODEL_PRICES["gpt-5-codex"]


def test_resolve_prices_none_returns_builtin_copy() -> None:
    """resolve_prices(None) returns the built-in table contents."""
    assert resolve_prices() == MODEL_PRICES


def test_compute_cost_uses_override_prices() -> None:
    """compute_cost(..., prices=...) looks up in the override table, not MODEL_PRICES."""
    prices = resolve_prices({"gpt-5.5": ModelPrice(input=1.0, cached_input=0.1, output=2.0)})
    cost = compute_cost(
        model="gpt-5.5",
        input_tokens=1_000_000,
        cached_input_tokens=0,
        output_tokens=0,
        prices=prices,
    )
    assert cost == pytest.approx(1.0)
    # Built-in (no prices arg) still uses the $5.00 input rate.
    builtin = compute_cost(
        model="gpt-5.5",
        input_tokens=1_000_000,
        cached_input_tokens=0,
        output_tokens=0,
    )
    assert builtin == pytest.approx(5.0)


def test_resolve_prices_adds_unknown_model_then_compute_cost_succeeds() -> None:
    """Chaining resolve → compute for a genuinely-unknown model (#156 #1).

    An override for a model NOT in MODEL_PRICES must land in the merged table
    and make compute_cost return a non-None synthesized value. The individual
    building blocks (load, resolve-per-model, compute-with-prices) are tested
    above; this asserts the full chain that the Codex cost-synthesis renderer
    relies on (resolve_prices(load_user_prices()) → compute_cost).
    """
    overrides = {"private-finetune": ModelPrice(input=3.0, cached_input=1.0, output=12.0)}
    merged = resolve_prices(overrides)
    # The unknown model is now present; built-ins are preserved unchanged.
    assert "private-finetune" in merged
    assert "gpt-5.5" in merged
    assert merged["gpt-5.5"] == MODEL_PRICES["gpt-5.5"]
    cost = compute_cost(
        model="private-finetune",
        input_tokens=500_000,
        cached_input_tokens=100_000,
        output_tokens=1_000,
        prices=merged,
    )
    assert cost is not None
    # 500K*3/1M + 100K*1/1M + 1K*12/1M = 1.5 + 0.1 + 0.012 = 1.612
    assert cost == pytest.approx(1.612)
