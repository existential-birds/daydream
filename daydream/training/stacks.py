"""Load complete frozen projections and recheck their consumption gates.

Require v2 records, repository/license identity, and _SUCCESS. C5 exclusions
always refuse consumption; C8 requires exact repository opt-ins. Full projection
loading also verifies pinned split assignments and computes a deterministic
file-tree digest. Exclusion-list parsing remains owned by training.exclusion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from daydream.training.admission import (
    REASON_CODE_C5_EXCLUDED_REPO,
    REASON_CODE_C8_COPYLEFT_UNOPTED,
)
from daydream.training.corpus_projection.splits import SPLIT_FILENAMES, assign_split
from daydream.training.exclusion import (
    is_copyleft,
    load_copyleft_list,
    load_exclusion_list,
)

__all__ = [
    "V2Projection",
    "load_dataset_v2",
    "load_v2_projection",
]


@dataclass(frozen=True)
class V2Projection:
    """Validated records in train/validation/holdout file order, grouped by split.

    lineage contains the pinned assignment parameters; digest covers sorted
    (relative path, file hash) pairs for every projection file."""

    records: list[dict[str, object]]
    by_split: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    lineage: dict[str, object] = field(default_factory=dict)
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
            f"projection {projection_dir}: missing lineage.json — "
            "refusing a projection without its pinned split parameters"
        )
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"projection {projection_dir}: lineage.json is not valid "
            f"JSON: {exc}"
        ) from exc
    if not isinstance(lineage, dict):
        raise ValueError(
            f"projection {projection_dir}: lineage.json is not a JSON object"
        )
    for key in ("salt", "holdout_rate", "val_rate"):
        if lineage.get(key) is None:
            raise ValueError(
                f"projection {projection_dir}: lineage.json missing "
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
            f"projection {projection_dir}: lineage.json holdout_rate/"
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
        expected = assign_split(
            record_id, salt=salt, holdout_rate=holdout_rate, val_rate=val_rate
        )
        if recorded != expected:
            offenders.append(f"{record_id} (recorded {recorded!r}, recomputed {expected!r})")
    if offenders:
        raise ValueError(
            f"projection {projection_dir}: split drift detected for "
            f"{len(offenders)} record(s): {', '.join(offenders)}. The recorded "
            "lineage.split disagrees with the split recomputed from the record "
            "id under the pinned salt/rates — refusing a drifted projection."
        )


def load_dataset_v2(
    path: str | Path,
    *,
    allow_copyleft: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, object]]:
    """Load split manifests in train/validation/holdout order after completeness checks.

    Every record must have schema_version="2", a nonempty lineage.repo_slug,
    and a resolved license decision with its own repository identity. Re-run
    C5/C8 checks: exclusions always win; copyleft requires the exact slug in
    allow_copyleft. Gate failures raise ValueError naming offending records or
    repositories; malformed JSON propagates unchanged. No partial list returns."""
    projection_dir = Path(path)
    if not (projection_dir / "_SUCCESS").is_file():
        raise ValueError(
            f"projection {projection_dir}: missing _SUCCESS marker — "
            "refusing a partial or incomplete projection"
        )
    records: list[dict[str, object]] = []
    for filename in SPLIT_FILENAMES.values():
        with (projection_dir / filename).open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)  # malformed line propagates verbatim
                schema_version = record.get("schema_version")
                if schema_version != "2":
                    raise ValueError(
                        f"projection record {record.get('record_id')!r} in "
                        f"{projection_dir / filename}: schema_version {schema_version!r} "
                        "!= '2' — refusing a record not projected by the v2 schema"
                    )
                records.append(record)

    _enforce_v2_identity_and_gates(records, projection_dir, allow_copyleft)
    return records


_ALLOWED_V2_LICENSE_STATUSES = frozenset({"admitted", "rejected"})


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
                f"projection record {record_id!r} in {path}: lineage.repo_slug "
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
                f"projection record {record_id!r} in {path}: lineage.license_decision "
                "missing, malformed, or not a resolved admitted/rejected decision — "
                "refusing a record without an immutable license decision"
            )

    lineages = [cast(dict[str, object], rec["lineage"]) for rec in records]

    excluded = {slug.casefold() for slug in load_exclusion_list()}
    excluded_offenders = sorted(
        {
            slug
            for lineage in lineages
            if (slug := str(lineage["repo_slug"]).casefold()) in excluded
        }
    )
    if excluded_offenders:
        raise ValueError(
            f"C5 violation ({REASON_CODE_C5_EXCLUDED_REPO}): excluded repo(s) in "
            f"projection {path}: {', '.join(excluded_offenders)}. These "
            "repositories are the held-out benchmark and must never appear in a "
            "training dataset, regardless of any flag."
        )

    allowed = frozenset(slug.casefold() for slug in allow_copyleft)
    copyleft_known = frozenset(slug.casefold() for slug in load_copyleft_list())
    copyleft_offenders = sorted(
        {
            slug
            for lineage in lineages
            if (slug := str(lineage["repo_slug"]).casefold())
            and slug not in allowed
            and (
                is_copyleft(slug, allowed, copyleft_list=copyleft_known)
                or (
                    cast(dict[str, object], lineage["license_decision"]).get("reason_code")
                    == REASON_CODE_C8_COPYLEFT_UNOPTED
                )
            )
        }
    )
    if copyleft_offenders:
        raise ValueError(
            f"C8 violation ({REASON_CODE_C8_COPYLEFT_UNOPTED}): copyleft repo(s) "
            f"in projection {path} without explicit opt-in: "
            f"{', '.join(copyleft_offenders)}. Pass these slugs via "
            "allow_copyleft to admit them."
        )


def load_v2_projection(
    path: str | Path,
    *,
    allow_copyleft: frozenset[str] | set[str] = frozenset(),
) -> V2Projection:
    """Load gate-clean records, verify pinned splits, and digest the directory.

    Missing split files become ValueError. Missing/invalid lineage and every
    recomputed split mismatch refuse the entire load before grouping."""
    projection_dir = Path(path)
    try:
        records = load_dataset_v2(projection_dir, allow_copyleft=allow_copyleft)
    except FileNotFoundError as exc:
        raise ValueError(
            f"projection {projection_dir}: missing split file "
            f"{exc.filename!r} — refusing an incomplete projection"
        ) from exc
    lineage = _load_lineage(projection_dir)
    _enforce_split_consistency(records, lineage, projection_dir)

    by_split: dict[str, list[dict[str, object]]] = {name: [] for name in SPLIT_FILENAMES}
    for record in records:
        # Identity and split consistency were validated before grouping.
        split = cast(dict[str, object], record["lineage"])["split"]
        by_split[cast(str, split)].append(record)

    return V2Projection(
        records=records,
        by_split=by_split,
        lineage=lineage,
        digest=_directory_digest(projection_dir),
    )
