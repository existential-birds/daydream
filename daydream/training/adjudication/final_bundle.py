"""Staging final-bundle constructor (issue #1078, M4 core).

Assembles the publish-ready annotation staging bundle — ``annotations.jsonl``,
``sessions.jsonl``, ``label-observations.jsonl``, ``coverage-report.json`` and
a **generated** ``lineage.json`` — from pipeline state alone (the
materialization dir, the hydrated index, and the archive). Pure construction:
no Hub I/O, no publishing; :func:`publish_final_annotation_bundle` consumes the
directory this function produces, so ``--dry-run`` and the real publish share
100% of the construction/validation code (M6).

Determinism: every file is written as canonical JSON (sorted keys, compact
separators), so identical pipeline state produces byte-identical bundles
across re-runs. Every missing or invalid input raises ``ValueError`` /
``FileNotFoundError`` naming the artifact — no fallback defaults, no silent
skips (the lineage file must never contain a fabricated field).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from daydream.archive.index import label_observation_history
from daydream.archive.sanitize import _derivative_digest
from daydream.training.adjudication.dispositions import (
    DECISIVE_DISPOSITIONS,
    NON_DECISIVE_DISPOSITIONS,
)
from daydream.training.adjudication.materialize import (
    _SESSIONS_OUT_FILENAME,
    _sessions_from_hydrated_stage,
)
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.report import build_report
from daydream.training.corpus_v2.tiers import classify_tier
from daydream.training.labeler_versions import ANNOTATION_SNAPSHOT_SCHEMA_VERSION

__all__ = ["build_final_bundle"]

_ANNOTATIONS_FILENAME = "annotations.jsonl"
_OBSERVATIONS_FILENAME = "label-observations.jsonl"
_REPORT_FILENAME = "coverage-report.json"
_LINEAGE_FILENAME = "lineage.json"
_MANIFEST_FILENAME = "preview-manifest.json"

_BUNDLE_FILES = (
    _ANNOTATIONS_FILENAME,
    _SESSIONS_OUT_FILENAME,
    _OBSERVATIONS_FILENAME,
    _REPORT_FILENAME,
    _LINEAGE_FILENAME,
)

# The lineage fields that must be present and non-empty in the pin/manifest —
# a missing field is a hard error naming the field, never a fallback default.
_LINEAGE_PIN_FIELDS = (
    "curation_id",
    "sanitized_hub_commit",
    "snapshot_id",
    "labeler_version",
    "rubric_version",
    "classifier_version",
)


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_materialized_records(materialize_dir: Path) -> list[dict[str, Any]]:
    annotations_path = materialize_dir / _ANNOTATIONS_FILENAME
    if not annotations_path.is_file():
        raise FileNotFoundError(
            f"materialized annotations not found (run `corpus adjudicate materialize` and "
            f"`corpus adjudicate harvest` first): {annotations_path}"
        )
    try:
        return [
            json.loads(line)
            for line in annotations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable materialized annotations at {annotations_path}: {exc}") from exc


def _load_sessions_output(materialize_dir: Path) -> list[dict[str, Any]]:
    sessions_path = materialize_dir / _SESSIONS_OUT_FILENAME
    if not sessions_path.is_file():
        raise FileNotFoundError(
            f"materialized preview snapshot not found (run `corpus adjudicate materialize` "
            f"first): {sessions_path}"
        )
    try:
        return [
            json.loads(line)
            for line in sessions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable materialized snapshot at {sessions_path}: {exc}") from exc


def _load_manifest(materialize_dir: Path) -> dict[str, Any]:
    manifest_path = materialize_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"preview manifest not found (run `corpus adjudicate materialize` first): "
            f"{manifest_path}"
        )
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable preview manifest at {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"preview manifest at {manifest_path} is not a JSON object")
    return manifest


def _index_sessions(index_root: Path) -> list[dict[str, Any]]:
    """Load the segmented sessions the fresh queue is built over, mirroring
    ``canonical.run_canonical_harvest``'s source selection."""
    if (index_root / _SESSIONS_OUT_FILENAME).is_file():
        sessions, _index_revision = _load_sessions(index_root)
    else:
        sessions, _index_revision = _sessions_from_hydrated_stage(index_root)
    return sessions


def _lineage_field(manifest: Mapping[str, Any], field: str, manifest_path: Path) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"preview manifest at {manifest_path} is missing required lineage field {field!r}"
        )
    return value


def build_final_bundle(
    *,
    index_root: Path,
    materialize_dir: Path,
    archive_dir: Path,
    out_dir: Path,
    curation_bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Construct the final annotation staging bundle into a fresh ``out_dir``.

    Writes exactly the five contract files (``_BUNDLE_FILES``) and never
    publishes: the caller feeds the directory to
    :func:`daydream.training.adjudication.publish.publish_final_annotation_bundle`.

    - ``annotations.jsonl`` / ``sessions.jsonl``: copied byte-for-byte from the
      materialization dir (missing file raises ``FileNotFoundError`` naming it).
    - ``label-observations.jsonl``: the archive's immutable per-session
      observation history (``archive.index.label_observation_history``) for
      every session in the snapshot, flattened and sorted chronologically by
      ``observed_at``. The projector's bundle verification treats unknown
      bundle files as part of the SHA256SUMS file set, so this extra file is
      additive-safe.
    - ``coverage-report.json``: ``report.build_report`` over a fresh complete
      queue (``build_queue(..., include_decisive=True)``) whose dispositions
      and as-of flags are reconciled against the materialized (merge-applied)
      records, with ``tier``/``posterior_eligible`` classified via the shared
      ``corpus_v2.tiers.classify_tier``.
    - ``lineage.json``: generated from the preview manifest's pin fields —
      never hand-authored, never defaulted. ``batch_fileset_digest`` is
      ``_derivative_digest`` over ``curation_bundle_dir`` (default: the index
      root when it *is* the curation bundle root), validated to exist.

    ``out_dir`` must not already exist or must be empty; a non-empty directory
    raises ``ValueError`` naming it (a stale staging dir must never be
    silently mixed with fresh content).

    Returns a summary dict with ``disposition_counts`` covering all five
    dispositions (``accepted``/``rejected``/``ambiguous``/``unanswered``/
    ``missing``) plus ``record_count`` and the written file names.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(
            f"final-bundle staging dir {out_dir} is not empty; remove it or pass a fresh path"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_materialized_records(materialize_dir)
    # Validate the materialized sessions output exists and parses (the bundle
    # copies it verbatim below).
    _load_sessions_output(materialize_dir)
    manifest_path = materialize_dir / _MANIFEST_FILENAME
    manifest = _load_manifest(materialize_dir)

    # 1. annotations.jsonl + sessions.jsonl: verbatim copies of the canonical
    #    materialized artifacts (already canonical JSONL — re-serializing
    #    would be a second code path for the same bytes).
    (out_dir / _ANNOTATIONS_FILENAME).write_bytes(
        (materialize_dir / _ANNOTATIONS_FILENAME).read_bytes()
    )
    (out_dir / _SESSIONS_OUT_FILENAME).write_bytes(
        (materialize_dir / _SESSIONS_OUT_FILENAME).read_bytes()
    )

    # 2. label-observations.jsonl: the archive's per-session observation
    #    history, chronological by ``observed_at`` (per-session rows are
    #    already ordered; the cross-session flatten re-sorts for determinism).
    snapshot_session_ids = sorted({str(r["session_id"]) for r in records})
    history_rows: list[dict[str, Any]] = []
    for session_id in snapshot_session_ids:
        history_rows.extend(label_observation_history(archive_dir, session_id))
    history_rows.sort(key=lambda row: (str(row.get("observed_at")), str(row.get("session_id"))))
    (out_dir / _OBSERVATIONS_FILENAME).write_text(
        "".join(_canonical(row) + "\n" for row in history_rows), encoding="utf-8"
    )

    # 3. coverage-report.json over the fresh complete queue, reconciled with
    #    the merge-applied materialized records so the report reflects the
    #    dispositions that will actually publish.
    merged_by_record_id = {str(r["record_id"]): r for r in records}
    report_items: list[dict[str, Any]] = []
    for item in build_queue(_index_sessions(index_root), include_decisive=True):
        record_id = str(item["record_id"])
        merged = merged_by_record_id.get(record_id)
        enriched = dict(item)
        if merged is not None:
            enriched["disposition"] = merged["disposition"]
            enriched["evidence_after_as_of"] = bool(merged.get("evidence_after_as_of"))
        try:
            tier = classify_tier(enriched)
        except Exception as exc:  # GoldGateError / TypeError: fail closed naming the record
            raise ValueError(
                f"final bundle: cannot classify tier for record_id {record_id!r}: {exc}"
            ) from exc
        enriched["tier"] = tier
        enriched["posterior_eligible"] = tier == "gold"
        report_items.append(enriched)
    report = build_report(report_items)
    # ``strata`` keys are (stack, profile) tuples in memory; the on-disk report
    # is JSON, so flatten them to ``"stack/profile"`` (deterministic order).
    report["strata"] = {
        f"{stack}/{profile}": count for (stack, profile), count in report["strata"].items()
    }
    (out_dir / _REPORT_FILENAME).write_text(_canonical(report) + "\n", encoding="utf-8")

    # 4. lineage.json: generated from the pin — every field must be present.
    lineage: dict[str, Any] = {
        field: _lineage_field(manifest, field, manifest_path) for field in _LINEAGE_PIN_FIELDS
    }
    lineage["schema_version"] = f"annotation-snapshot/{ANNOTATION_SNAPSHOT_SCHEMA_VERSION}"
    bundle_root = curation_bundle_dir if curation_bundle_dir is not None else index_root
    if not bundle_root.is_dir():
        raise FileNotFoundError(
            f"curation bundle dir for the batch fileset digest not found: {bundle_root}"
        )
    lineage["batch_fileset_digest"] = _derivative_digest(bundle_root)
    # ``as_of`` is pin-required but legitimately empty when unpinned (mirroring
    # ``snapshot.snapshot_id``'s as-of edge) — a *missing* key is an error.
    if "as_of" not in manifest:
        raise ValueError(
            f"preview manifest at {manifest_path} is missing required lineage field 'as_of'"
        )
    lineage["as_of"] = str(manifest["as_of"])
    (out_dir / _LINEAGE_FILENAME).write_text(_canonical(lineage) + "\n", encoding="utf-8")

    written = sorted(path.name for path in out_dir.iterdir())
    missing = [name for name in _BUNDLE_FILES if name not in written]
    if missing:
        raise ValueError(f"final bundle at {out_dir} is missing contract files: {missing}")

    counts = Counter(str(r["disposition"]) for r in records)
    disposition_counts = {
        **{d: counts[d] for d in sorted(DECISIVE_DISPOSITIONS | NON_DECISIVE_DISPOSITIONS)},
        **{k: v for k, v in sorted(counts.items())
           if k not in DECISIVE_DISPOSITIONS and k not in NON_DECISIVE_DISPOSITIONS},
    }
    return {
        "disposition_counts": disposition_counts,
        "record_count": len(records),
        "observation_history_rows": len(history_rows),
        "files": written,
        "out_dir": str(out_dir),
    }
