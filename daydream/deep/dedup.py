"""Deterministic same-concern candidates for merge and arbiter adjudication.

Record/alternative pairs require a shared file and normalized-title bigram
Jaccard similarity >= 0.5. Record/record pairs may span files. Candidates still
need adjudication; destructive host folding uses the separate, higher bar.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from daydream.deep.records import record_uid

_STOP_WORDS = frozenset(
    {"the", "a", "an", "is", "on", "in", "of", "to", "for", "and", "or", "with", "by"}
)
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_SIM_THRESHOLD = 0.5

# Host folding has no downstream adjudicator, so it requires a higher
# similarity than candidate generation.
FOLD_SIM_THRESHOLD = 0.8


@dataclass(frozen=True)
class CandidatePair:
    """Record/alternative candidate with verbatim descriptions and bigram similarity."""

    record_id: str
    record_file: str
    record_description: str
    alt_title: str
    alt_files: tuple[str, ...]
    similarity: float


def normalize_title(text: str) -> str:
    """Lowercase, strip punctuation, drop stop words, return whitespace-joined string."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    tokens = [tok for tok in cleaned.split() if tok and tok not in _STOP_WORDS]
    return " ".join(tokens)


def bigrams(normalized: str) -> set[str]:
    """Character bigrams tolerate token reordering; a one-character title is its own gram."""
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Return Jaccard similarity, or 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def descriptions_match(a: str, b: str, *, threshold: float = _SIM_THRESHOLD) -> bool:
    """Compare normalized-description bigrams, rejecting empty/stop-word-only input.

    The default threshold proposes candidates for review. Destructive host folding
    must use FOLD_SIM_THRESHOLD because no later adjudicator checks its decision.
    """
    a_bigrams = bigrams(normalize_title(a))
    b_bigrams = bigrams(normalize_title(b))
    if not a_bigrams or not b_bigrams:
        return False
    return jaccard(a_bigrams, b_bigrams) >= threshold


def _files_overlap(record_file: str, alt_files: Iterable[str]) -> bool:
    """Return True when the record's file appears in the alt-issue files."""
    return bool(record_file) and record_file in set(alt_files)


@dataclass(frozen=True)
class RecordDuplicatePair:
    """Candidate pair whose field names are the dedup-candidates.json wire format.

    Serialization preserves every field. Reviewer IDs may repeat across stacks;
    host UIDs disambiguate them, or are empty for post-merge inputs without that
    identity. Renaming or dropping fields changes the persisted schema.
    """

    record_a_id: str
    record_a_uid: str
    record_a_file: str
    record_a_description: str
    record_a_source: str
    record_b_id: str
    record_b_uid: str
    record_b_file: str
    record_b_description: str
    record_b_source: str
    similarity: float


def build_dedup_candidates(
    records: list[dict[str, Any]],
    alt_issues: list[dict[str, Any]],
) -> list[CandidatePair]:
    """Return record/alternative pairs sharing a file with similarity >= 0.5.

    Sort by (record_id, alt_title); descriptions and alternative files stay verbatim.
    """
    pairs: list[CandidatePair] = []
    for r in records:
        r_file = str(r.get("file", ""))
        r_desc = str(r.get("description", ""))
        r_bigrams = bigrams(normalize_title(r_desc))
        if not r_file or not r_bigrams:
            continue
        for a in alt_issues:
            a_files = tuple(a.get("files") or [])
            a_title = str(a.get("title", ""))
            if not a_files or not a_title:
                continue
            if not _files_overlap(r_file, a_files):
                continue
            sim = jaccard(r_bigrams, bigrams(normalize_title(a_title)))
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


def build_record_dedup_candidates(
    records: list[dict[str, Any]],
    sources: list[str],
    *,
    threshold: float = _SIM_THRESHOLD,
) -> list[RecordDuplicatePair]:
    """Compare record descriptions across files, ordered by IDs then UID tie-breakers.

    sources must parallel records or ValueError is raised. A zero threshold returns
    every comparable pair for distribution analysis; empty descriptions never pair.
    """
    if len(sources) != len(records):
        raise ValueError("sources must contain exactly one entry per record")
    pairs: list[RecordDuplicatePair] = []
    n = len(records)
    for i in range(n):
        r_a = records[i]
        a_id = str(r_a.get("id", ""))
        # Post-merge eval inputs may lack pre-merge UIDs; they must still participate.
        a_uid = record_uid(r_a)
        a_file = str(r_a.get("file", ""))
        a_desc = str(r_a.get("description", ""))
        a_source = sources[i]
        a_bigrams = bigrams(normalize_title(a_desc))
        if not a_desc or not a_bigrams:
            continue
        for j in range(i + 1, n):
            r_b = records[j]
            b_id = str(r_b.get("id", ""))
            b_desc = str(r_b.get("description", ""))
            b_bigrams = bigrams(normalize_title(b_desc))
            if not b_desc or not b_bigrams:
                continue
            sim = jaccard(a_bigrams, b_bigrams)
            if sim >= threshold:
                pairs.append(
                    # Keyword arguments throughout: the dataclass now has ten
                    # same-typed ``str`` fields in a/b order, so a positional
                    # construction would mis-assign silently (and type-check
                    # clean) the next time a field is inserted.
                    RecordDuplicatePair(
                        record_a_id=a_id,
                        record_a_uid=a_uid,
                        record_a_file=a_file,
                        record_a_description=a_desc,
                        record_a_source=a_source,
                        record_b_id=b_id,
                        record_b_uid=record_uid(r_b),
                        record_b_file=str(r_b.get("file", "")),
                        record_b_description=b_desc,
                        record_b_source=sources[j],
                        similarity=sim,
                    )
                )
    # Reviewer IDs repeat across stacks; UID tie-breakers make their order stable.
    pairs.sort(key=lambda p: (p.record_a_id, p.record_b_id, p.record_a_uid, p.record_b_uid))
    return pairs
