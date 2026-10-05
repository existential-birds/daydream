"""Preserve reward inputs beside separately identified redacted review text."""

from __future__ import annotations

import hashlib
from typing import Any, cast

from daydream.dataset.privacy import sanitize_evidence
from daydream.training.reward import DEFAULT_WEIGHTS, REWARD_VERSION, ScoringInputs, score_trajectory


def capture_scoring(inputs: ScoringInputs, review_text: str | None) -> dict[str, Any]:
    """Keep the producer's exact scoring length and existing intrinsic reducer."""
    captured_text = sanitize_evidence(review_text)
    return cast(dict[str, Any], sanitize_evidence({
        "verifier_verdicts": inputs.verifier_verdicts,
        "format_valid": inputs.format_valid,
        "review_text": captured_text,
        "source_review_sha256": hashlib.sha256(review_text.encode()).hexdigest() if review_text is not None else None,
        "review_text_redaction": {
            "policy": "daydream.shared.v1",
            "applied": captured_text != review_text,
            "captured_sha256": (
                hashlib.sha256(captured_text.encode()).hexdigest() if captured_text is not None else None
            ),
        },
        "length": inputs.length,
        "reward_policy": {"version": REWARD_VERSION, "configuration": {
            "w_len": DEFAULT_WEIGHTS.w_len, "w_fp": DEFAULT_WEIGHTS.w_fp,
            "len_tau": DEFAULT_WEIGHTS.len_tau, "len_scale": DEFAULT_WEIGHTS.len_scale,
            "verdict_map": dict(DEFAULT_WEIGHTS.verdict_map),
            "fp_penalty_map": dict(DEFAULT_WEIGHTS.fp_penalty_map),
        }},
        "persisted_breakdown": score_trajectory(inputs).to_dict(),
        "posterior_cost": None,
    }))
