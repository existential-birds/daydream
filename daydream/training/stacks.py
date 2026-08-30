"""Corpus-side dataset loaders for the training pipeline (M17, M22).

Thin loading layer over the exported JSONL corpus records. The C5 exclusion
gate and the C8 copyleft opt-in gate run **before any records are returned**:
the whole file is scanned and every offending slug is named on the first
violation class encountered. The loader never skip-and-warns — a corpus that
touches an excluded or unopted-copyleft repo fails closed.

The lists themselves are owned exclusively by :mod:`daydream.training.exclusion`;
this module re-implements no parsing.

Corpus v2 adds :func:`load_dataset_v2`, an additive sibling that loads the
frozen per-split manifests of a ``run_build_corpus_v2`` projection directory
and refuses any record not stamped ``schema_version == "2"``. The v1 surface
(``load_dataset``, its ``legacy_policy`` stamping, and its error messages) is
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream.training.exclusion import load_copyleft_list, load_exclusion_list

__all__ = ["load_dataset", "load_dataset_v2"]


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
            record = json.loads(stripped)
            # M23 legacy tagging: a row whose labeler policy version is absent
            # (or null) was admitted under the legacy reply-count/merge-presence
            # gold policy. Tag it explicitly — current-policy SFT prefers
            # native-profile traces, and this flag is how downstream selection
            # tells the two classes apart. Tagging is metadata, never a drop:
            # the loader's only refusals are the C5/C8 fail-closed gates above.
            policy_version = record.get("labeler_policy_version")
            record["legacy_policy"] = not (isinstance(policy_version, str) and policy_version)
            records.append(record)

    _enforce_license_gates(records, path, allow_copyleft)
    return records


def _enforce_license_gates(
    records: list[dict[str, object]],
    path: str | Path,
    allow_copyleft: frozenset[str] | set[str],
) -> None:
    """Shared C5/C8 fail-closed gates (see module docstring). The lists are
    owned by :mod:`daydream.training.exclusion`; this module never parses them."""
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


def load_dataset_v2(path: str | Path) -> list[dict[str, object]]:
    """Load a projected corpus v2 directory (the frozen train/validation/
    holdout JSONL manifests from ``run_build_corpus_v2``), enforcing the same
    C5 and C8 fail-closed gates as :func:`load_dataset`.

    Args:
        path: The projection output directory containing ``train.jsonl``,
            ``validation.jsonl`` and ``holdout.jsonl``.

    Returns:
        The full list of v2 training-record dicts, in split-file order.

    Raises:
        ValueError: When any record's ``schema_version`` is not ``"2"`` (the
            offending record id is named), or a C5/C8 gate fires.
        json.JSONDecodeError: When a line is not valid JSON — never a silent
            skip.
    """
    projection_dir = Path(path)
    records: list[dict[str, object]] = []
    for filename in ("train.jsonl", "validation.jsonl", "holdout.jsonl"):
        with (projection_dir / filename).open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)  # malformed line propagates verbatim
                schema_version = record.get("schema_version")
                if schema_version != "2":
                    raise ValueError(
                        f"corpus v2 record {record.get('record_id')!r} in "
                        f"{projection_dir / filename}: schema_version {schema_version!r} "
                        "!= '2' — refusing a record not projected by the v2 schema"
                    )
                records.append(record)

    _enforce_license_gates(records, projection_dir, frozenset())
    return records
