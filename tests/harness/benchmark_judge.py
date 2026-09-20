"""Shared judge client fake and env for benchmark verifier tests."""

from __future__ import annotations

from typing import Any


def judge_env() -> dict[str, Any]:
    """The canonical permissive judge environment used by verifier tests."""
    return {
        "DAYDREAM_JUDGE_PROVIDER": "anthropic",
        "DAYDREAM_JUDGE_MODEL": "m",
        "DAYDREAM_JUDGE_API_KEY": "k",
        "DAYDREAM_JUDGE_BASE_URL": None,
    }


class MatchClient:
    async def complete_json(self, *, user: Any, system: Any, max_tokens: Any) -> dict[str, Any]:
        return {"match": True, "confidence": 1.0, "reasoning": "identical"}
