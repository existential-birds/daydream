"""Frozen-corpus projection loaders for the training pipeline.

:func:`load_v2_projection` loads a ``build_frozen_corpus`` projection
directory into a :class:`V2Projection` and refuses anything that is not a
complete, gate-clean projection:

- per-split record loading over the frozen ``train``/``validation``/
  ``holdout`` manifests (:func:`load_dataset_v2`), refusing any record not
  stamped ``schema_version == "2"``. Every record must also carry its repo
  identity and immutable license decision under ``lineage`` (structurally
  required — an absent field is itself the failure, never a bypass);
- the ``_SUCCESS`` completeness marker (written last by the projector,
  mirroring the bundle's own gate) is required: a partial projection left
  by a mid-write failure is refused, never consumed;
- the C5/C8 license gates re-run fail-closed over every loaded record: an
  excluded repo is refused unconditionally, and a copyleft-class record is
  refused unless its exact slug was passed via ``allow_copyleft``;
- per-split record access derived from each record's ``lineage.split``;
- the projection's ``lineage.json`` (salt, rates, provenance pins);
- per-file sha256 digests and a deterministic directory-level digest over
  the sorted ``(relpath, sha256(file_bytes))`` pairs — a pure function of
  the directory bytes, so the same projection always yields the same digest;
- a **split-drift gate**: the split is recomputed from every record's id via
  :func:`daydream.training.corpus_projection.splits.assign_split` under the lineage's
  pinned salt/rates, and any disagreement with the record's recorded
  ``lineage.split`` refuses the whole load (``ValueError`` naming the
  offending record id) — never a silent accept.

The lists themselves are owned exclusively by :mod:`daydream.training.exclusion`;
this module re-implements no parsing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from daydream.archive.hydrate_rules import (
    REASON_CODE_C5_EXCLUDED_REPO,
    REASON_CODE_C8_COPYLEFT_UNOPTED,
)
from daydream.training.corpus_projection.splits import Split, assign_split
from daydream.training.exclusion import (
    is_copyleft,
    load_copyleft_list,
    load_exclusion_list,
)

__all__ = [
    "V2Projection",
    "load_dataset_v2",
    "load_v2_projection",
    "recompute_split_from_record_id",
]

_SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "holdout": "holdout.jsonl",
}


@dataclass(frozen=True)
class V2Projection:
    """A loaded corpus-v2 projection directory.

    Attributes:
        records: All v2 records, in split-file order (train, validation,
            holdout), exactly as :func:`load_dataset_v2` returns them.
        by_split: Records grouped by their recorded ``lineage.split``; all
            three keys are always present.
        lineage: The parsed ``lineage.json`` dict.
        split_digests: sha256 of each split JSONL file's bytes, keyed by
            filename.
        digest: Deterministic directory-level digest: sha256 over the sorted
            ``(relpath, sha256(file_bytes))`` pairs of every file in the
            projection directory. A pure function of the directory bytes.
    """

    records: list[dict[str, object]]
    by_split: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    lineage: dict[str, object] = field(default_factory=dict)
    split_digests: dict[str, str] = field(default_factory=dict)
    digest: str = ""


def recompute_split_from_record_id(
    record_id: str,
    *,
    salt: str,
    holdout_rate: float,
    val_rate: float,
) -> Split:
    """Recompute the frozen content-derived split for one record id.

    Thin named wrapper over :func:`assign_split` so the drift gate and its
    tests share one call site for the recompute side of the comparison.
    """
    return assign_split(
        record_id, salt=salt, holdout_rate=holdout_rate, val_rate=val_rate
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(projection_dir: Path) -> str:
    """sha256 over sorted ``(relpath, sha256(file_bytes))`` pairs — the same
    directory always yields the same digest."""
    pairs: list[str] = []
    for path in sorted(projection_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(projection_dir).as_posix()
            pairs.append(f"{rel}\x1f{_sha256_file(path)}")
    return hashlib.sha256("\x1e".join(pairs).encode("utf-8")).hexdigest()


def _load_lineage(projection_dir: Path) -> dict[str, object]:
    lineage_path = projection_dir / "lineage.json"
    if not lineage_path.is_file():
        raise ValueError(
            f"corpus v2 projection {projection_dir}: missing lineage.json — "
            "refusing a projection without its pinned split parameters"
        )
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"corpus v2 projection {projection_dir}: lineage.json is not valid "
            f"JSON: {exc}"
        ) from exc
    if not isinstance(lineage, dict):
        raise ValueError(
            f"corpus v2 projection {projection_dir}: lineage.json is not a JSON object"
        )
    for key in ("salt", "holdout_rate", "val_rate"):
        if lineage.get(key) is None:
            raise ValueError(
                f"corpus v2 projection {projection_dir}: lineage.json missing "
                f"{key!r} — refusing to recompute splits without the pinned "
                "assignment parameters"
            )
    return lineage


def _enforce_split_consistency(
    records: list[dict[str, object]],
    lineage: dict[str, object],
    projection_dir: Path,
) -> None:
    """Split-drift gate: recompute each record's split from its record id and
    refuse the load when it disagrees with the recorded ``lineage.split``."""
    salt = str(lineage["salt"])
    holdout_rate_obj = lineage["holdout_rate"]
    val_rate_obj = lineage["val_rate"]
    if not isinstance(holdout_rate_obj, (int, float)) or not isinstance(
        val_rate_obj, (int, float)
    ):
        raise ValueError(
            f"corpus v2 projection {projection_dir}: lineage.json holdout_rate/"
            "val_rate must be numeric"
        )
    holdout_rate = float(holdout_rate_obj)
    val_rate = float(val_rate_obj)
    offenders: list[str] = []
    for record in records:
        record_id = str(record.get("record_id", ""))
        recorded = None
        lineage_obj = record.get("lineage")
        if isinstance(lineage_obj, dict):
            recorded = lineage_obj.get("split")
        expected = recompute_split_from_record_id(
            record_id, salt=salt, holdout_rate=holdout_rate, val_rate=val_rate
        )
        if recorded != expected:
            offenders.append(f"{record_id} (recorded {recorded!r}, recomputed {expected!r})")
    if offenders:
        raise ValueError(
            f"corpus v2 projection {projection_dir}: split drift detected for "
            f"{len(offenders)} record(s): {', '.join(offenders)}. The recorded "
            "lineage.split disagrees with the split recomputed from the record "
            "id under the pinned salt/rates — refusing a drifted projection."
        )


def load_dataset_v2(
    path: str | Path,
    *,
    allow_copyleft: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, object]]:
    """Load a projected corpus v2 directory (the frozen train/validation/
    holdout JSONL manifests from ``build_frozen_corpus``), enforcing repo
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


def load_v2_projection(
    path: str | Path,
    *,
    allow_copyleft: frozenset[str] | set[str] = frozenset(),
) -> V2Projection:
    """Load a corpus-v2 projection directory into a :class:`V2Projection`.

    Reuses :func:`load_dataset_v2` for the existing
    fail-closed gates (missing ``_SUCCESS``, non-``"2"`` ``schema_version``,
    repo identity, license decisions, C5/C8), then parses ``lineage.json``,
    recomputes every record's split from its id, and refuses any drift.

    Args:
        path: The projection output directory written by
            ``build_frozen_corpus``.
        allow_copyleft: Passed through to the underlying v2 loader.

    Returns:
        The :class:`V2Projection` for the directory.

    Raises:
        ValueError: On any existing-gate failure, on a missing split file
            or a missing/malformed ``lineage.json``, or on split drift (the
            offending record ids and both splits are named).
    """
    projection_dir = Path(path)
    try:
        records = load_dataset_v2(projection_dir, allow_copyleft=allow_copyleft)
    except FileNotFoundError as exc:
        raise ValueError(
            f"corpus v2 projection {projection_dir}: missing split file "
            f"{exc.filename!r} — refusing an incomplete projection"
        ) from exc
    lineage = _load_lineage(projection_dir)
    _enforce_split_consistency(records, lineage, projection_dir)

    by_split: dict[str, list[dict[str, object]]] = {name: [] for name in _SPLIT_FILENAMES}
    for record in records:
        lineage_obj = record.get("lineage")
        split = lineage_obj.get("split") if isinstance(lineage_obj, dict) else None
        if split not in by_split:
            raise ValueError(
                f"corpus v2 projection {projection_dir}: record "
                f"{record.get('record_id')!r} carries unknown split {split!r}"
            )
        by_split[cast(str, split)].append(record)

    split_digests: dict[str, str] = {}
    for filename in _SPLIT_FILENAMES.values():
        split_path = projection_dir / filename
        if not split_path.is_file():
            raise ValueError(
                f"corpus v2 projection {projection_dir}: missing split file "
                f"{filename!r} — refusing an incomplete projection"
            )
        split_digests[filename] = _sha256_file(split_path)
    return V2Projection(
        records=records,
        by_split=by_split,
        lineage=lineage,
        split_digests=split_digests,
        digest=_directory_digest(projection_dir),
    )
