"""The single semantic-matching seam for benchmark scoring.

Every in-process "are these two findings the same issue?" decision in the
repo goes through :class:`FindingJudge`. Both scored corpora reach it by the
same route: the withmartian golden corpus and a harvested bot-review corpus
are the same data shape, scored by the same ``anthropic-direct`` path, so the
vendor-overlap comparison and the golden-corpus comparison share one judge
implementation and one prompt.

(The ``martian`` scoring route is deliberately outside this seam: it does not
judge in process at all — it shells out to the withmartian submission
protocol, which is out of scope to change.)

Attach points for issue #92 (design only — none of this is implemented here):

* **Blinded ordering** — a wrapping ``FindingJudge`` that calls the inner
  judge as ``same_issue(a, b)`` and ``same_issue(b, a)`` and combines the two
  verdicts, cancelling any position bias in the prompt.
* **Inter-judge agreement (κ)** — run two ``FindingJudge`` implementations
  over the same per-pair stream and compare their :class:`JudgeVerdict`\\ s.
  Per-pair granularity is why the judge is a per-pair call rather than one
  batch call per PR.
* **Human spot-check** — sample persisted verdicts: the evaluation grid
  already records ``reasoning`` and ``confidence`` per matched pair into the
  ``evaluations.json`` leaves.

All three wrap or consume the Protocol; none of them requires touching
``anthropic_score._evaluate_review`` again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from daydream.benchmark.score import BenchmarkStepError

_MAX_JUDGE_TOKENS = 1024

_JUDGE_SYSTEM = "You are a precise code review evaluator. Always respond with valid JSON."

_JUDGE_PROMPT = """You are evaluating AI code review tools.
Determine if the candidate issue matches the golden (expected) comment.

Golden Comment (the issue we're looking for):
{golden_comment}

Candidate Issue (from the tool's review):
{candidate}

Instructions:
- Determine if the candidate identifies the SAME underlying issue as the golden comment
- Accept semantic matches - different wording is fine if it's the same problem
- Focus on whether they point to the same bug, concern, or code issue

Respond with ONLY a JSON object:
{{"reasoning": "brief explanation", "match": true/false, "confidence": 0.0-1.0}}"""


class JudgeError(BenchmarkStepError):
    """A single same-issue judgement failed (call error or malformed verdict).

    Raised per pair; the evaluation grid records it against that pair and
    keeps scoring the rest.
    """


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's decision about a single (golden, candidate) pair."""

    match: bool
    confidence: float
    reasoning: str


class FindingJudge(Protocol):
    async def same_issue(self, golden: str, candidate: str) -> JudgeVerdict:
        """Decide whether two finding texts describe the same underlying issue.

        Raises:
            JudgeError: If the judgement could not be obtained or parsed.
        """
        ...


class _JsonCompleter(Protocol):
    async def complete_json(self, *, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        ...


@dataclass
class AnthropicFindingJudge:
    """`FindingJudge` backed by one Anthropic Messages call per pair."""

    client: _JsonCompleter

    async def same_issue(self, golden: str, candidate: str) -> JudgeVerdict:
        try:
            response = await self.client.complete_json(
                system=_JUDGE_SYSTEM,
                user=_JUDGE_PROMPT.format(golden_comment=golden, candidate=candidate),
                max_tokens=_MAX_JUDGE_TOKENS,
            )
        except Exception as exc:
            raise JudgeError(str(exc)) from exc
        return _parse_verdict(response)


def _parse_verdict(response: dict[str, Any]) -> JudgeVerdict:
    if "error" in response:
        raise JudgeError(str(response["error"]))
    if not isinstance(response.get("match"), bool):
        raise JudgeError("Anthropic judge response 'match' must be a boolean.")
    confidence = response.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise JudgeError("Anthropic judge response 'confidence' must be a number.")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise JudgeError("Anthropic judge response 'confidence' must be between 0.0 and 1.0.")
    reasoning = response.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise JudgeError("Anthropic judge response 'reasoning' must be a string.")
    return JudgeVerdict(match=response["match"], confidence=confidence, reasoning=reasoning)
