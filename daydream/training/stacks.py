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
and refuses any record not stamped ``schema_version == "2"``. Every v2
record must also carry its repo identity and immutable license decision
under ``lineage`` (structurally required — an absent field is itself the
failure, never a bypass), and the C5/C8 gates are re-run fail-closed over
every loaded record: an excluded repo is refused unconditionally, and a
copyleft-class record is refused unless its exact slug was passed via
``allow_copyleft``. The v1 surface (``load_dataset``, its ``legacy_policy``
stamping, and its error messages) is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from daydream.archive.hydrate_rules import (
    REASON_CODE_C5_EXCLUDED_REPO,
    REASON_CODE_C8_COPYLEFT_UNOPTED,
)
from daydream.training.exclusion import (
    is_copyleft,
    load_copyleft_list,
    load_exclusion_list,
)

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


def load_dataset_v2(
    path: str | Path,
    *,
    allow_copyleft: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, object]]:
    """Load a projected corpus v2 directory (the frozen train/validation/
    holdout JSONL manifests from ``run_build_corpus_v2``), enforcing repo
    identity, the license-decision stamp, and C5/C8 fail-closed.

    ``holdout.jsonl``. A ``_SUCCESS`` completeness marker (written last by
    the projector, mirroring the bundle's own gate) is required: a partial
    projection left by a mid-write failure is refused, never consumed.

    Structural gate: every record must carry ``lineage.repo_slug`` (a
    non-empty string) and ``lineage.license_decision`` (a dict with
    ``status`` in ``{"admitted", "rejected"}`` and a non-empty ``repo_slug``).
    The field's absence is itself the failure — a stripped record can never
    slip through as "not applicable".

    Consumption gate (defense in depth over the recorded decisions): the
    C5/C8 lists are re-evaluated over every loaded record. A record whose
    ``repo_slug`` is on the exclusion list is refused unconditionally — no
    keyword can suppress C5. A copyleft-class record (on the copyleft list,
    or carrying a ``c8_copyleft_unopted`` decision reason) is refused unless
    its exact slug was passed via ``allow_copyleft``.

    Args:
        path: The projection output directory containing ``train.jsonl``,
            ``validation.jsonl`` and ``holdout.jsonl``.
        allow_copyleft: ``owner/repo`` slugs the caller has explicitly opted
            in. Empty by default, so copyleft repos are always refused unless
            explicitly admitted. Never overrides the C5 exclusion list.

    Returns:
        The full list of v2 training-record dicts, in split-file order.

    Raises:
        ValueError: When the projection lacks its ``_SUCCESS`` marker, when
            any record's ``schema_version`` is not ``"2"``, when a record is
            missing its repo identity or license decision (the offending
            record id and field are named), or when any record's repo is on
            the exclusion list (C5) or is copyleft without being in
            ``allow_copyleft`` (C8). All offending slugs are named; no
            records are returned.
        json.JSONDecodeError: When a line is not valid JSON — never a silent
            skip.
    """
    projection_dir = Path(path)
    if not (projection_dir / "_SUCCESS").is_file():
        raise ValueError(
            f"corpus v2 projection {projection_dir}: missing _SUCCESS marker — "
            "refusing a partial or incomplete projection"
        )
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

    _enforce_v2_identity_and_gates(records, projection_dir, allow_copyleft)
    return records


_ALLOWED_V2_LICENSE_STATUSES = frozenset({"admitted", "rejected"})


def _v2_lineages(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Validated ``lineage`` dicts (structural gate already passed)."""
    return [
        cast(dict[str, object], rec["lineage"])
        for rec in records
        if isinstance(rec.get("lineage"), dict)
    ]


def _enforce_v2_identity_and_gates(
    records: list[dict[str, object]],
    path: Path,
    allow_copyleft: frozenset[str] | set[str],
) -> None:
    """Structural repo-identity requirement plus the C5/C8 fail-closed gates
    re-run over loaded v2 records (see :func:`load_dataset_v2`)."""
    for record in records:
        record_id = record.get("record_id")
        lineage_obj = record.get("lineage")
        lineage = lineage_obj if isinstance(lineage_obj, dict) else None
        repo_slug = lineage.get("repo_slug") if lineage else None
        if not isinstance(repo_slug, str) or not repo_slug:
            raise ValueError(
                f"corpus v2 record {record_id!r} in {path}: lineage.repo_slug "
                "missing or empty — refusing a record without repo identity"
            )
        decision_obj = lineage.get("license_decision") if lineage else None
        decision = decision_obj if isinstance(decision_obj, dict) else None
        decision_slug = decision.get("repo_slug") if decision else None
        if (
            not isinstance(decision, dict)
            or decision.get("status") not in _ALLOWED_V2_LICENSE_STATUSES
            or not isinstance(decision_slug, str)
            or not decision_slug
        ):
            raise ValueError(
                f"corpus v2 record {record_id!r} in {path}: lineage.license_decision "
                "missing, malformed, or not a resolved admitted/rejected decision — "
                "refusing a record without an immutable license decision"
            )

    excluded = {slug.casefold() for slug in load_exclusion_list()}
    excluded_offenders = sorted(
        {
            slug
            for lineage in _v2_lineages(records)
            if (slug := str(lineage["repo_slug"]).casefold()) in excluded
        }
    )
    if excluded_offenders:
        raise ValueError(
            f"C5 violation ({REASON_CODE_C5_EXCLUDED_REPO}): excluded repo(s) in "
            f"corpus v2 projection {path}: {', '.join(excluded_offenders)}. These "
            "repositories are the held-out benchmark and must never appear in a "
            "training dataset, regardless of any flag."
        )

    allowed = frozenset(slug.casefold() for slug in allow_copyleft)
    copyleft_known = frozenset(slug.casefold() for slug in load_copyleft_list())
    copyleft_offenders = sorted(
        {
            slug
            for lineage in _v2_lineages(records)
            if (slug := str(lineage["repo_slug"]).casefold())
            and slug not in allowed
            and (
                is_copyleft(slug, allowed, copyleft_list=copyleft_known)
                or (
                    isinstance(lineage.get("license_decision"), dict)
                    and cast(dict[str, object], lineage["license_decision"]).get("reason_code")
                    == REASON_CODE_C8_COPYLEFT_UNOPTED
                )
            )
        }
    )
    if copyleft_offenders:
        raise ValueError(
            f"C8 violation ({REASON_CODE_C8_COPYLEFT_UNOPTED}): copyleft repo(s) "
            f"in corpus v2 projection {path} without explicit opt-in: "
            f"{', '.join(copyleft_offenders)}. Pass these slugs via "
            "allow_copyleft to admit them."
        )
