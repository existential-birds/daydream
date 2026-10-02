"""Static reviewed prices for synthesizing cost when backend USD usage is absent.

Lookup uses exact model ids and archived timestamps. Keep retired ids for old
trajectories. User overrides replace built-in policies per model; Claude SDK
costs bypass this table."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from daydream.config import DEFAULT_PI_MODEL
from daydream.config_file import load_toml_or_empty

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """USD per million uncached-input, cached-input, and output tokens."""

    input: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class PricingPolicy:
    """A model's dated and context-sensitive pricing policy."""

    price: ModelPrice
    long_context_threshold: int | None = None
    long_context_multiplier: ModelPrice | None = None
    effective_prices: tuple[tuple[date, ModelPrice], ...] = ()

    def resolve(self, *, total_input_tokens: int, effective_date: date) -> ModelPrice:
        """Return the price for the request's complete input context."""
        price = self.price
        for starts_on, candidate in self.effective_prices:
            if effective_date >= starts_on:
                price = candidate
        if self.long_context_threshold is not None and total_input_tokens > self.long_context_threshold:
            assert self.long_context_multiplier is not None
            return ModelPrice(
                input=price.input * self.long_context_multiplier.input,
                cached_input=price.cached_input * self.long_context_multiplier.cached_input,
                output=price.output * self.long_context_multiplier.output,
            )
        return price


class _ResolvedPrices(dict[str, ModelPrice]):
    """Resolved prices with policies for models not explicitly overridden."""

    def __init__(self, *args: Any, policies: dict[str, PricingPolicy], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policies = policies


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
    "gpt-5.6-sol": ModelPrice(input=5.00, cached_input=0.50, output=30.00),
    "gpt-5.6-terra": ModelPrice(input=2.50, cached_input=0.25, output=15.00),
    "gpt-5.6-luna": ModelPrice(input=1.00, cached_input=0.10, output=6.00),
    # Bare alias routes to gpt-5.6-sol; priced identically (lookup is exact-match, no prefix fallback).
    "gpt-5.6": ModelPrice(input=5.00, cached_input=0.50, output=30.00),
    # Post-introductory standard rate; the $2/$0.20/$10 intro rate lapses 2026-08-31.
    "claude-sonnet-5": ModelPrice(input=3.00, cached_input=0.30, output=15.00),
    "glm-5.2": ModelPrice(input=1.40, cached_input=0.26, output=4.40),
    # Nous-hosted DeepSeek V4 Flash — the Pi backend default (DEFAULT_PI_MODEL).
    # Keyed by the constant so the default model and its price entry cannot drift apart.
    DEFAULT_PI_MODEL: ModelPrice(input=0.28, cached_input=0.028, output=0.42),
}

_GPT56_LONG_CONTEXT_THRESHOLD = 272_000
_GPT56_POLICIES = {
    model: PricingPolicy(
        price,
        _GPT56_LONG_CONTEXT_THRESHOLD,
        ModelPrice(input=2.0, cached_input=2.0, output=1.5),
    )
    for model, price in MODEL_PRICES.items()
    if model.startswith("gpt-5.6")
}
_BUILTIN_POLICIES: dict[str, PricingPolicy] = {
    **_GPT56_POLICIES,
    "claude-sonnet-5": PricingPolicy(
        ModelPrice(input=2.00, cached_input=0.20, output=10.00),
        effective_prices=((date(2026, 9, 1), MODEL_PRICES["claude-sonnet-5"]),),
    ),
}

def _coerce_price(model: str, table: Any) -> ModelPrice | None:
    """Validate a model table; log and skip invalid entries.

    input/output are required finite nonnegative numbers; cached_input defaults to
    input. Booleans are not prices."""
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
        if not math.isfinite(coerced):
            logger.warning("daydream prices: %r field %r is non-finite (%r) — skipping", model, name, value)
            return None
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
    """Load model overrides from explicit path, environment, or ~/.daydream/prices.toml.

    Invalid entries are logged and skipped; absent/malformed TOML yields {}.
    Required input/output and optional cached_input use the same validation."""
    if path is None:
        env_path = os.environ.get("DAYDREAM_PRICES_FILE")
        if env_path:
            path = Path(env_path)
        else:
            try:
                path = Path.home() / ".daydream" / "prices.toml"
            except RuntimeError as exc:
                logger.warning("daydream prices: could not resolve home directory — ignoring (%s)", exc)
                return {}

    data = load_toml_or_empty(path)
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
    """Merge overrides into a fresh table, retaining policies only for non-overridden models."""
    override_models = set(overrides or {})
    return _ResolvedPrices(
        {**MODEL_PRICES, **(overrides or {})},
        policies={model: policy for model, policy in _BUILTIN_POLICIES.items() if model not in override_models},
    )


def compute_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    *,
    prices: dict[str, ModelPrice] | None = None,
    total_input_tokens: int | None = None,
    effective_date: date | None = None,
) -> float | None:
    """Price uncached input, cached input, and output; unknown models return None.

    Context policy uses total_input_tokens or their summed input counts. Omitted
    usage dates retain standard-rate behavior; an explicit table controls lookup."""
    table = MODEL_PRICES if prices is None else prices
    price = table.get(model)
    if price is None:
        return None
    policies = _BUILTIN_POLICIES if prices is None else getattr(prices, "policies", {})
    if policy := policies.get(model):
        price = policy.resolve(
            total_input_tokens=input_tokens + cached_input_tokens if total_input_tokens is None else total_input_tokens,
            effective_date=date.max if effective_date is None else effective_date,
        )
    per_token = 1_000_000.0
    return (
        input_tokens * price.input / per_token
        + cached_input_tokens * price.cached_input / per_token
        + output_tokens * price.output / per_token
    )


def compute_cost_from_totals(
    model: str,
    *,
    total_input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    prices: dict[str, ModelPrice] | None = None,
    effective_date: date | None = None,
) -> float | None:
    """Subtract cached tokens from total input, clamp uncached at zero, then price the usage."""
    uncached_input = max(total_input_tokens - cached_input_tokens, 0)
    return compute_cost(
        model,
        uncached_input,
        cached_input_tokens,
        output_tokens,
        prices=prices,
        total_input_tokens=total_input_tokens,
        effective_date=effective_date,
    )
