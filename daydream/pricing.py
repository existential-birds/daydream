"""Hardcoded OpenAI price table for cost synthesis.

Provide a static, code-reviewed price table covering the OpenAI models daydream
runs through Codex and direct API backends. Used by the enriched PR comment
renderer to synthesize cost from token counts when a backend (notably Codex)
does not surface USD cost directly. Anthropic-backed runs use cost values
already supplied by the Claude SDK and do not pass through this module.

Reverses project decision D-16 ("no synthesis of cost from token prices").
Refs #65.

Pricing source: OpenAI pricing snapshot, May 2026 (per 1M tokens). Cached-input
prices for `gpt-5.5-pro`, `gpt-5-codex`, and `gpt-5.3-codex` were not
published in USD on https://openai.com/api/pricing/ or
https://developers.openai.com/codex/pricing at build time; those entries fall
back to the input-token price as a conservative upper bound (slight overcount,
transparent).

Exports:
    ModelPrice: dataclass holding input/cached_input/output USD per 1M tokens.
    MODEL_PRICES: dict[str, ModelPrice] - the built-in price table.
    load_user_prices: parse user-supplied price overrides from a TOML file.
    resolve_prices: merge user overrides over the built-in price table.
    compute_cost: function returning USD cost or None for unknown models.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """USD price per 1M tokens for a single model.

    Attributes:
        input: USD per 1M uncached input tokens.
        cached_input: USD per 1M cached input tokens.
        output: USD per 1M output tokens.
    """

    input: float
    cached_input: float
    output: float


# Per 1M tokens, USD. May 2026 snapshot.
# Users can override these per-model via ~/.daydream/prices.toml (see load_user_prices).
MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5.5": ModelPrice(input=5.00, cached_input=0.50, output=30.00),
    # cached_input fallback to input price — unpublished USD value at build time
    "gpt-5.5-pro": ModelPrice(input=30.00, cached_input=30.00, output=180.00),
    # cached_input fallback to input price — unpublished USD value at build time
    "gpt-5-codex": ModelPrice(input=1.25, cached_input=1.25, output=10.00),
    # cached_input fallback to input price — unpublished USD value at build time
    "gpt-5.3-codex": ModelPrice(input=1.75, cached_input=1.75, output=14.00),
}


def _load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file into a dict, returning {} when absent or malformed.

    Mirrors the resilient pattern in ``config_file.py`` but never raises: the
    price loader must never break cost synthesis over a bad user file.

    Args:
        path: Path to the TOML file.

    Returns:
        The parsed table, or an empty dict if the file is absent or malformed.

    Raises:
        Never. Malformed TOML is logged and yields ``{}``.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        logger.warning("daydream prices: malformed TOML in %s — ignoring (%s)", path, exc)
        return {}
    except OSError as exc:
        logger.warning("daydream prices: could not read %s — ignoring (%s)", path, exc)
        return {}


def _coerce_price(model: str, table: Any) -> ModelPrice | None:
    """Coerce one ``[prices."<model>"]`` table into a ModelPrice, or None.

    Requires ``input`` and ``output`` (numeric, >= 0). ``cached_input`` is
    optional and defaults to ``input``. Any missing, non-numeric, or negative
    required field is logged and yields None so the caller skips the model.

    Args:
        model: Model identifier the table belongs to (for log messages).
        table: Raw parsed value for the model's sub-table.

    Returns:
        A ``ModelPrice`` on success, or None when the entry is invalid.
    """
    if not isinstance(table, dict):
        logger.warning("daydream prices: entry %r is not a table — skipping", model)
        return None

    def _field(name: str) -> float | None:
        if name not in table:
            logger.warning("daydream prices: %r missing required field %r — skipping", model, name)
            return None
        value = table[name]
        # bool is an int subclass; reject it explicitly as a non-numeric price.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            logger.warning("daydream prices: %r field %r is non-numeric (%r) — skipping", model, name, value)
            return None
        coerced = float(value)
        if coerced < 0:
            logger.warning("daydream prices: %r field %r is negative (%r) — skipping", model, name, value)
            return None
        return coerced

    input_price = _field("input")
    output_price = _field("output")
    if input_price is None or output_price is None:
        return None

    if "cached_input" in table:
        cached_price = _field("cached_input")
        if cached_price is None:
            return None
    else:
        cached_price = input_price

    return ModelPrice(input=input_price, cached_input=cached_price, output=output_price)


def load_user_prices(path: Path | None = None) -> dict[str, ModelPrice]:
    """Load user-supplied price overrides from a TOML file.

    Resolves the file from, in order: the explicit ``path`` argument, the
    ``$DAYDREAM_PRICES_FILE`` environment variable, then
    ``~/.daydream/prices.toml``. Parses ``[prices."<model>"]`` tables. Each
    entry requires numeric, non-negative ``input`` and ``output``;
    ``cached_input`` is optional and defaults to ``input``. Invalid entries are
    logged and skipped; an absent file or malformed TOML yields ``{}``.

    Args:
        path: Explicit path to a prices TOML file. Overrides env/default.

    Returns:
        A mapping of model name to ``ModelPrice`` for every valid entry.

    Raises:
        Never. All error conditions are logged and yield ``{}`` or skip the
        offending entry.
    """
    if path is None:
        env_path = os.environ.get("DAYDREAM_PRICES_FILE")
        if env_path:
            path = Path(env_path)
        else:
            path = Path.home() / ".daydream" / "prices.toml"

    data = _load_toml(path)
    raw_prices = data.get("prices")
    if not isinstance(raw_prices, dict):
        if raw_prices is not None:
            logger.warning("daydream prices: [prices] is not a table in %s — ignoring", path)
        return {}

    result: dict[str, ModelPrice] = {}
    for model, table in raw_prices.items():
        price = _coerce_price(str(model), table)
        if price is not None:
            result[str(model)] = price
    return result


def resolve_prices(overrides: dict[str, ModelPrice] | None = None) -> dict[str, ModelPrice]:
    """Merge user overrides over the built-in price table.

    Args:
        overrides: Per-model overrides; each key replaces the built-in entry.

    Returns:
        A new dict of built-in prices with ``overrides`` applied per-model.
    """
    return {**MODEL_PRICES, **(overrides or {})}


def compute_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    *,
    prices: dict[str, ModelPrice] | None = None,
) -> float | None:
    """Compute USD cost for a model invocation, or None for unknown models.

    Args:
        model: Model identifier (e.g. "gpt-5.5"). Must match a key in the
            active price table.
        input_tokens: Count of uncached input tokens.
        cached_input_tokens: Count of cached input tokens (priced separately).
        output_tokens: Count of output tokens.
        prices: Optional price table to look up in. When None, the built-in
            ``MODEL_PRICES`` is used (back-compatible with existing callers).

    Returns:
        Total USD cost, or None when model is not in the active price table.
    """
    table = MODEL_PRICES if prices is None else prices
    price = table.get(model)
    if price is None:
        return None
    per_token = 1_000_000.0
    return (
        input_tokens * price.input / per_token
        + cached_input_tokens * price.cached_input / per_token
        + output_tokens * price.output / per_token
    )
