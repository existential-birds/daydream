"""Corpus-side dataset loaders for the training pipeline (M17, M22).

Thin loading layer over the exported JSONL corpus records. The C5 exclusion
gate and the C8 copyleft opt-in gate run **before any records are returned**:
the whole file is scanned and every offending slug is named on the first
violation class encountered. The loader never skip-and-warns — a corpus that
touches an excluded or unopted-copyleft repo fails closed.

The lists themselves are owned exclusively by :mod:`daydream.training.exclusion`;
this module re-implements no parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream.training.exclusion import load_copyleft_list, load_exclusion_list


def load_dataset(
    path: str | Path,
    *,
    allow_copyleft: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, object]]:
    """Load a JSONL corpus file, enforcing C5 and C8 fail-closed.

    Args:
        path: Path to the JSONL corpus (one JSON object per line).
        allow_copyleft: ``owner/repo`` slugs the caller has explicitly opted
            in. Empty by default, so copyleft repos are always refused unless
            explicitly admitted.

    Returns:
        The full list of corpus record dicts. Records without a ``pr_number``
        (pre-PR runs) load identically to PR records (M22).

    Raises:
        ValueError: When any record's repo is on the exclusion list (C5) or
            on the copyleft list without being in ``allow_copyleft`` (C8).
            All offending slugs are named; no records are returned.
    """
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))

    excluded = {slug.casefold() for slug in load_exclusion_list()}
    excluded_offenders = sorted(
        {slug for rec in records if (slug := str(rec.get("repo_slug", "")).casefold()) in excluded}
    )
    if excluded_offenders:
        raise ValueError(
            f"C5 violation: excluded repo(s) in corpus {path}: {', '.join(excluded_offenders)}. "
            "These repositories are the held-out benchmark and must never appear in a "
            "training dataset, regardless of any flag."
        )

    copyleft = {slug.casefold() for slug in load_copyleft_list()}
    allowed = {slug.casefold() for slug in allow_copyleft}
    copyleft_offenders = sorted(
        {
            slug
            for rec in records
            if (slug := str(rec.get("repo_slug", "")).casefold()) in copyleft
            and slug not in allowed
        }
    )
    if copyleft_offenders:
        raise ValueError(
            f"C8 violation: copyleft repo(s) in corpus {path} without explicit opt-in: "
            f"{', '.join(copyleft_offenders)}. Pass these slugs via allow_copyleft to admit them."
        )

    return records
