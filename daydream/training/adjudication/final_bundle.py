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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from daydream.archive.index import label_observation_history
from daydream.archive.sanitize import _derivative_digest
from daydream.training.adjudication.canonical import _evidence_after_as_of
from daydream.training.adjudication.materialize import (
    _SESSIONS_OUT_FILENAME,
    _sessions_from_hydrated_stage,
)
from daydream.training.adjudication.observations import load_observations
from daydream.training.adjudication.precedence import effective_adjudication
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.report import build_report
from daydream.training.corpus_v2.tiers import classify_tier
from daydream.training.dispositions import (
    DECISIVE_DISPOSITIONS,
    NON_DECISIVE_DISPOSITIONS,
)
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

# publish.py stages upload payloads into ``<bundle-dir>/.publish-stage`` and
# leaves the scratch dir behind after a real publish; it is publish-internal,
# never bundle content, so a re-construction over the same out_dir tolerates it
# (re-publish idempotence).
_PUBLISH_STAGE_DIRNAME = ".publish-stage"

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
            f"`corpus adjudicate harvest-snapshot` first): {annotations_path}"
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


def _enrich_report_items(
    items: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Enrich queue items into report items — the single shared implementation
    behind both the CLI report path (``cli._report_items``) and the final
    bundle's coverage report, so the published 80% admission gate sees exactly
    the same human-decision state as ``corpus adjudicate report``.

    Attaches each record's observations, applies three-tier effective
    adjudication (a human decisive judgment whose evidence digest matches the
    fresh item overrides the disposition), stamps the temporal axis first, and
    classifies ``tier``/``posterior_eligible`` with the corpus-v2 authority.
    Gold eligibility requires a human decision, so automatic decisive records
    without one are demoted to task-only and never count as adjudicated. An
    observation referencing a record_id absent from ``items`` raises
    ``ValueError`` naming it (fail-closed, mirroring the CLI twin).
    """
    queue_ids = {str(item["record_id"]) for item in items}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for obs in observations:
        record_id = str(obs["record_id"])
        if record_id not in queue_ids:
            raise ValueError(
                f"observation references record_id {record_id!r} which is not in the "
                f"adjudication queue over the hydrated index"
            )
        grouped.setdefault(record_id, []).append(obs)

    enriched: list[dict[str, Any]] = []
    for item in items:
        enriched_item = dict(item)
        record_obs = grouped.get(str(item["record_id"]), [])
        enriched_item["observations"] = record_obs
        gold_eligible = False
        if record_obs:
            resolved = effective_adjudication(record_obs)
            gold_eligible = resolved["gold_eligible"]
            if (
                resolved["role"] in ("rater", "adjudicator")
                and resolved["evidence_digest"] == str(item["evidence_digest"])
                and resolved["disposition"] in DECISIVE_DISPOSITIONS
            ):
                enriched_item["disposition"] = resolved["disposition"]
        # Stamp the temporal axis FIRST so classify_tier sees it, matching the
        # canonical serializer and the corpus projection (tiers.py C5/M9): an
        # evidence-after-as_of record must classify "silver" here exactly as it
        # does on the canonical record — never gold/posterior_eligible.
        enriched_item["evidence_after_as_of"] = _evidence_after_as_of(enriched_item, as_of)
        # The gold gate has one implementation (classify_tier); gold-eligibility
        # comes from the human-observation resolution (conflict/review-required
        # decisive judgments stay out of the gold tier). A classifier failure
        # fail-closes naming the record, never a silent skip.
        try:
            tier = classify_tier(enriched_item)
        except Exception as exc:
            raise ValueError(
                f"cannot classify tier for record_id {str(item['record_id'])!r}: {exc}"
            ) from exc
        if tier == "gold" and not gold_eligible:
            tier = "task-only"
        enriched_item["tier"] = tier
        enriched_item["posterior_eligible"] = tier == "gold" and str(
            enriched_item["profile"]
        ) == "pr_review"
        enriched.append(enriched_item)
    return enriched


def build_final_bundle(
    *,
    index_root: Path,
    materialize_dir: Path,
    archive_dir: Path,
    out_dir: Path,
    curation_bundle_dir: Path | None = None,
    observations_path: Path | None = None,
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
      queue (``build_queue(..., include_decisive=True)``) enriched by the
      shared :func:`_enrich_report_items` — the same observations/
      effective-adjudication/gold-eligibility enrichment the CLI report twin
      (``cli._report_items``) applies — so the published 80% admission gate
      counts human-adjudicated outcome-bearing records only, never every
      automatic decisive record. ``observations_path`` (the ``--state-dir``
      observations store; a missing file is an empty store) supplies the
      human observations; empty when not given.
    - ``lineage.json``: generated from the preview manifest's pin fields —
      never hand-authored, never defaulted. ``batch_fileset_digest`` is
      ``_derivative_digest`` over ``curation_bundle_dir`` (default: the index
      root when it *is* the curation bundle root), validated to exist.

    ``out_dir`` must not exist, must be empty, or may contain only the
    five contract files from a prior (deterministic) construction — e.g. a
    ``--dry-run`` validation immediately followed by a real publish over the
    same state. Construction is deterministic, so re-writing those files is
    byte-identical; any foreign file (a previous run's ``_SUCCESS``, editor
    droppings, a partial publish) raises ``ValueError`` naming the directory,
    because a stale or published staging dir must never be silently mixed
    with fresh content. The ``.publish-stage`` scratch dir a real publish
    leaves behind is publish-internal, not foreign content, so a re-publish
    over the same dir is tolerated.

    Returns a summary dict with ``disposition_counts`` covering all five
    dispositions (``accepted``/``rejected``/``ambiguous``/``unanswered``/
    ``missing``) plus ``record_count`` and the written file names.
    """
    if out_dir.exists():
        foreign = sorted(
            p.name for p in out_dir.iterdir()
            if p.name not in _BUNDLE_FILES and p.name != _PUBLISH_STAGE_DIRNAME
        )
        if foreign:
            raise ValueError(
                f"final-bundle staging dir {out_dir} contains foreign content "
                f"({', '.join(foreign)}); remove it or pass a fresh path"
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

    # 3. coverage-report.json over the fresh complete queue, enriched exactly
    #    like the CLI report twin (``cli._report_items`` -> shared
    #    ``_enrich_report_items``): observations attached per record, three-tier
    #    effective adjudication applied, and gold records without a human
    #    decision demoted to task-only, so the published 80% admission gate
    #    sees real human adjudication state instead of counting every automatic
    #    decisive record as adjudicated. ``as_of`` comes from the preview
    #    manifest (empty when unpinned).
    observations = load_observations(observations_path) if observations_path is not None else []
    report_items = _enrich_report_items(
        build_queue(_index_sessions(index_root), include_decisive=True),
        observations,
        as_of=manifest.get("as_of"),
    )
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
    # ``snapshot.snapshot_id``'s as-of edge) — a *missing* key is an error, and
    # a null/empty value is the unpinned edge, never the fabricated "None".
    if "as_of" not in manifest:
        raise ValueError(
            f"preview manifest at {manifest_path} is missing required lineage field 'as_of'"
        )
    as_of = manifest["as_of"]
    lineage["as_of"] = "" if as_of is None else str(as_of)
    (out_dir / _LINEAGE_FILENAME).write_text(_canonical(lineage) + "\n", encoding="utf-8")

    written = sorted(path.name for path in out_dir.iterdir() if path.is_file())
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
