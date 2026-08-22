"""Self-contained isolated Harbor judge verifier for the compiled-grade verifier image.

Stdlib + httpx only. Never imports daydream: this file must run unchanged
inside a compiled-grade verifier image that has no daydream wheel. It wires a
bounded per-pair judge prompt, two isolated external judge clients (Anthropic
Messages + OpenAI-compatible) behind one ``complete_json`` seam, strict verdict
parsing, a shared retry/redirect/timeout policy, a concurrency-10 runner, and
``run_verifier`` which writes ``reward.json`` / ``reward-details.json``
atomically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import verifier_core


class _AsyncHttpClient(Protocol):
    """An ``httpx.AsyncClient``-shaped seam for the injected fake clients."""

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        """POST ``json`` to ``url`` and return an httpx-like response object."""


_VERIFIER_THRESHOLD = verifier_core.CONFIDENCE_THRESHOLD

VerifierError = verifier_core.VerifierError

JUDGE_PROMPT_TEMPLATE = (
    Path(__file__).with_name("judge_prompt.md").read_text(encoding="utf-8")
)

_PROMPT_CAP_BYTES = 24 * 1024


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Truncate ``value`` to at most ``max_bytes`` on a UTF-8 boundary."""
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    return value.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def _render_filled(
    template: str,
    gold: dict[str, Any],
    candidate: dict[str, Any],
    *,
    gold_body: str,
    candidate_body: str,
) -> str:
    return template.format(
        gold_title=gold.get("title", ""),
        gold_severity=str(gold.get("severity") or ""),
        gold_path=gold.get("path", ""),
        gold_start_line=gold.get("start_line", ""),
        gold_end_line=gold.get("end_line", ""),
        gold_body=gold_body,
        candidate_title=candidate.get("title", ""),
        candidate_severity=str(candidate.get("severity") or ""),
        candidate_path=candidate.get("path", ""),
        candidate_start_line=candidate.get("start_line", ""),
        candidate_end_line=candidate.get("end_line", ""),
        candidate_body=candidate_body,
    )


def render_pair_prompt(gold: dict, candidate: dict, *, template: str) -> str:
    """Render a bounded, untrusted-fenced prompt for one gold/candidate pair.

    If the filled template would exceed 24 KiB, the body fields (gold then
    candidate) are truncated on a UTF-8 boundary until the result fits — the
    only cap mechanism, and it never fails the pair.
    """
    gold_body = gold.get("body", "") or ""
    candidate_body = candidate.get("body", "") or ""
    filled = _render_filled(
        template, gold, candidate, gold_body=gold_body, candidate_body=candidate_body
    )
    while len(filled.encode("utf-8")) > _PROMPT_CAP_BYTES:
        if gold_body:
            gold_body = _truncate_utf8(
                gold_body, max(0, len(gold_body.encode("utf-8")) // 2)
            )
        elif candidate_body:
            candidate_body = _truncate_utf8(
                candidate_body, max(0, len(candidate_body.encode("utf-8")) // 2)
            )
        else:
            break
        filled = _render_filled(
            template, gold, candidate, gold_body=gold_body, candidate_body=candidate_body
        )
    return filled


def parse_verdict(raw: object) -> verifier_core.Verdict:
    """Validate a raw judge verdict dict and return a ``Nondeterministic.Verdict``.

    ``gold_id``/``candidate_id`` are placeholders the caller (``judge_pairs``)
    stamps onto the returned verdict. Any violation — wrong type, out-of-range
    confidence, missing key, non-dict input — raises ``VerifierError``; never
    silently coerces a fallback value.
    """
    if not isinstance(raw, dict):
        raise VerifierError("verdict must be a JSON object")
    if "match" not in raw:
        raise VerifierError("verdict missing required field 'match'")
    match = raw["match"]
    if not isinstance(match, bool):
        raise VerifierError(f"verdict 'match' must be a boolean, got {match!r}")
    if "confidence" not in raw:
        raise VerifierError("verdict missing required field 'confidence'")
    confidence = raw["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VerifierError(f"verdict 'confidence' must be a number in [0,1], got {confidence!r}")
    if not 0.0 <= confidence <= 1.0:
        raise VerifierError(f"verdict 'confidence' must be in [0,1], got {confidence!r}")
    if "reasoning" not in raw:
        raise VerifierError("verdict missing required field 'reasoning'")
    reasoning = raw["reasoning"]
    if not isinstance(reasoning, str):
        raise VerifierError(f"verdict 'reasoning' must be a string, got {reasoning!r}")
    return verifier_core.Verdict(
        gold_id="", candidate_id="", match=match, confidence=float(confidence), reasoning=reasoning
    )


def main() -> None:
    ...


if __name__ == "__main__":
    main()
