"""Fake-HTTP tests for the isolated Harbor verifier entry (``templates/tests/score_review.py``).

Exercises the self-contained judge clients (Anthropic + OpenAI-compatible),
the bounded prompt renderer, strict verdict parsing, shared retry/redirect
policy, concurrency/pair-cap runner, fail-whole-task error path, provider
selection, oracle parity, and end-to-end ``run_verifier`` — all with an
injected fake HTTP client against ``tmp_path``.
"""

import asyncio
import json
from pathlib import Path

import pytest


def test_spike_template_loads_with_bare_import(sr_module) -> None:
    """The template loads via importlib and its bare import resolves to the sibling copy."""
    assert sr_module.__name__ == "score_review"
    assert sr_module.verifier_core is not None
    assert sr_module.verifier_core.CONFIDENCE_THRESHOLD == 0.7