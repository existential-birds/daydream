"""Structured dedup pre-filter for deep-review mode (D-27).

Pure function. Takes parsed per-stack records + TTT alternative-review issues and
emits ``CandidatePair`` entries where each pair shares at least one file AND has
a normalized-title bigram Jaccard similarity >= 0.5.

The merge agent (plan 05-08) adjudicates candidate pairs. This pre-filter exists
to keep the merger's prompt small and keep quadratic-pair enumeration out of the
LLM.

Thresholds (per RESEARCH.md Open Question 3):

- Bigram Jaccard similarity >= 0.5 on normalized titles
- AND at least one shared file path

Both gates must hold — a loose pre-filter is safer than a tight one because the
merge agent still adjudicates.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_STOP_WORDS = frozenset(
    {"the", "a", "an", "is", "on", "in", "of", "to", "for", "and", "or", "with", "by"}
)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_SIM_THRESHOLD = 0.5


@dataclass(frozen=True)
class CandidatePair:
    """A same-concern candidate between a per-stack record and a TTT alt-review issue.

    Attributes:
        record_id: The parsed record's id (from FEEDBACK_SCHEMA).
        record_file: The record's file field.
        record_description: The record's description (kept verbatim).
        alt_title: The TTT alternative-review issue's title.
        alt_files: The TTT issue's files tuple.
        similarity: Jaccard bigram similarity between normalized titles.
    """

    record_id: str
    record_file: str
    record_description: str
    alt_title: str
    alt_files: tuple[str, ...]
    similarity: float


def _normalize_title(text: str) -> str:
    """Lowercase, strip punctuation, drop stop words, return whitespace-joined string."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    tokens = [tok for tok in cleaned.split() if tok and tok not in _STOP_WORDS]
    return " ".join(tokens)


def _bigrams(normalized: str) -> set[str]:
    """Return the set of 2-character bigrams from a normalized title string.

    Character-level bigrams are used because they are robust to token
    reordering (per RESEARCH.md Open Question 3 recommendation). Titles
    shorter than 2 characters return a sentinel single-element set so
    very short titles remain comparable under Jaccard.
    """
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Return Jaccard similarity, or 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _files_overlap(record_file: str, alt_files: Iterable[str]) -> bool:
    """Return True when the record's file appears in the alt-issue files."""
    return bool(record_file) and record_file in set(alt_files)


def build_dedup_candidates(
    records: list[dict[str, Any]],
    alt_issues: list[dict[str, Any]],
) -> list[CandidatePair]:
    """Return same-concern candidate pairs per D-27 thresholds.

    Args:
        records: Parsed per-stack records matching FEEDBACK_SCHEMA
            (``id``, ``file``, ``line``, ``description`` keys).
        alt_issues: TTT alternative-review issues matching ALTERNATIVE_REVIEW_SCHEMA
            (``title``, ``files`` keys).

    Returns:
        Deterministically-ordered list of ``CandidatePair`` instances for every
        record/alt-issue combination that shares a file path AND has normalized
        title bigram Jaccard similarity >= 0.5. Order is ``(record_id, alt_title)``.
    """
    pairs: list[CandidatePair] = []
    for r in records:
        r_file = str(r.get("file", ""))
        r_desc = str(r.get("description", ""))
        r_bigrams = _bigrams(_normalize_title(r_desc))
        if not r_file or not r_bigrams:
            continue
        for a in alt_issues:
            a_files = tuple(a.get("files") or [])
            a_title = str(a.get("title", ""))
            if not a_files or not a_title:
                continue
            if not _files_overlap(r_file, a_files):
                continue
            sim = _jaccard(r_bigrams, _bigrams(_normalize_title(a_title)))
            if sim >= _SIM_THRESHOLD:
                pairs.append(
                    CandidatePair(
                        record_id=str(r.get("id", "")),
                        record_file=r_file,
                        record_description=r_desc,
                        alt_title=a_title,
                        alt_files=a_files,
                        similarity=sim,
                    )
                )
    pairs.sort(key=lambda p: (p.record_id, p.alt_title))
    return pairs
