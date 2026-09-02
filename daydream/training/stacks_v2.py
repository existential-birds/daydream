"""V2 projection directory loader: per-split records, lineage, and digests.

Wraps :func:`daydream.training.stacks.load_dataset_v2` (which enforces the
``_SUCCESS`` completeness marker, the ``schema_version == "2"`` structural
gate, the repo-identity/license-decision requirement, and the C5/C8
fail-closed gates) and adds the boundary the downstream pipeline needs:

- per-split record access derived from each record's ``lineage.split``;
- the projection's ``lineage.json`` (salt, rates, provenance pins);
- per-file sha256 digests and a deterministic directory-level digest over
  the sorted ``(relpath, sha256(file_bytes))`` pairs — a pure function of
  the directory bytes, so the same projection always yields the same digest;
- a **split-drift gate**: the split is recomputed from every record's id via
  :func:`daydream.training.corpus_v2.splits.assign_split` under the lineage's
  pinned salt/rates, and any disagreement with the record's recorded
  ``lineage.split`` refuses the whole load (``ValueError`` naming the
  offending record id) — never a silent accept.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from daydream.training.corpus_v2.splits import Split, assign_split
from daydream.training.stacks import load_dataset_v2

__all__ = ["V2Projection", "load_v2_projection", "recompute_split_from_record_id"]

_SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "holdout": "holdout.jsonl",
}


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


def load_v2_projection(
    path: str | Path,
    *,
    allow_copyleft: frozenset[str] | set[str] = frozenset(),
) -> V2Projection:
    """Load a corpus-v2 projection directory into a :class:`V2Projection`.

    Reuses :func:`daydream.training.stacks.load_dataset_v2` for the existing
    fail-closed gates (missing ``_SUCCESS``, non-``"2"`` ``schema_version``,
    repo identity, license decisions, C5/C8), then parses ``lineage.json``,
    recomputes every record's split from its id, and refuses any drift.

    Args:
        path: The projection output directory written by
            ``run_build_corpus_v2``.
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
