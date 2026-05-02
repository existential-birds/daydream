"""Hardcoded OpenAI price table for cost synthesis.

Provide a static, code-reviewed price table covering the OpenAI models daydream
runs through Codex and direct API backends. Used by the enriched PR comment
renderer to synthesize cost from token counts when a backend (notably Codex)
does not surface USD cost directly. Anthropic-backed runs use cost values
already supplied by the Claude SDK and do not pass through this module.

Repeals project decision D-16 ("no synthesis of cost from token prices") per
the enriched-pr-comment spec (.beagle/concepts/enriched-pr-comment/spec.md).
Refs #65.

Pricing source: OpenAI pricing snapshot, May 2026 (per 1M tokens). Cached-input
prices for `gpt-5.5-pro`, `gpt-5-codex`, and `gpt-5.3-codex` were not
published in USD on https://openai.com/api/pricing/ or
https://developers.openai.com/codex/pricing at build time; per spec OQ1
fallback, those entries use the input-token price as a conservative upper
bound (slight overcount, transparent).

Exports:
    ModelPrice: dataclass holding input/cached_input/output USD per 1M tokens.
    MODEL_PRICES: dict[str, ModelPrice] - the price table.
    compute_cost: function returning USD cost or None for unknown models.
"""

from dataclasses import dataclass


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
MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5.5": ModelPrice(input=5.00, cached_input=0.50, output=30.00),
    # cached_input fallback to input price — unpublished USD value at build time
    "gpt-5.5-pro": ModelPrice(input=30.00, cached_input=30.00, output=180.00),
    # cached_input fallback to input price — unpublished USD value at build time
    "gpt-5-codex": ModelPrice(input=1.25, cached_input=1.25, output=10.00),
    # cached_input fallback to input price — unpublished USD value at build time
    "gpt-5.3-codex": ModelPrice(input=1.75, cached_input=1.75, output=14.00),
}


def compute_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Compute USD cost for a model invocation, or None for unknown models.

    Args:
        model: Model identifier (e.g. "gpt-5.5"). Must match a key in MODEL_PRICES.
        input_tokens: Count of uncached input tokens.
        cached_input_tokens: Count of cached input tokens (priced separately).
        output_tokens: Count of output tokens.

    Returns:
        Total USD cost, or None when model is not in MODEL_PRICES.
    """
    price = MODEL_PRICES.get(model)
    if price is None:
        return None
    per_token = 1_000_000.0
    return (
        input_tokens * price.input / per_token
        + cached_input_tokens * price.cached_input / per_token
        + output_tokens * price.output / per_token
    )
