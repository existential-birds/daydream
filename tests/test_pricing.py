"""Tests for the OpenAI cost-synthesis price table."""

from datetime import date
from pathlib import Path

import pytest

from daydream.pricing import (
    ModelPrice,
    compute_cost,
    load_user_prices,
)


def test_claude_sonnet_5_uses_the_rate_in_effect_on_the_usage_date() -> None:
    """Introductory pricing lasts through 2026-08-31; an omitted date uses standard rates."""
    tokens = (1_000_000, 1_000_000, 1_000_000)
    assert compute_cost("claude-sonnet-5", *tokens, effective_date=date(2026, 8, 31)) == pytest.approx(12.20)
    assert compute_cost("claude-sonnet-5", *tokens, effective_date=date(2026, 9, 1)) == pytest.approx(18.30)
    assert compute_cost("claude-sonnet-5", *tokens) == pytest.approx(18.30)


# --- User-overridable pricing -------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_user_prices_skips_invalid_entries_and_unresolvable_home(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.delenv("DAYDREAM_PRICES_FILE", raising=False)

    def _raise() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(_raise))
    with caplog.at_level("WARNING"):
        loaded = load_user_prices()
    assert loaded == {}
    assert any("could not resolve home directory" in rec.message.lower() for rec in caplog.records)
