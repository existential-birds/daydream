"""Preview materialization of per-finding annotation snapshots (issue #1055).

Collects fresh semantic evidence from production bronze through the harvester's
read-only services, or reads stored resolutions from legacy/index-only history,
then runs the resulting per-finding payload through the shared serializer
(``snapshot.build_canonical_record``) and emits a deterministic
``sessions.jsonl`` plus a digest-pinned ``preview-manifest.json``.

Preview mode guarantees (AC 4 / M2): never appends ``label_observations``,
never writes any resume-cache or harvest-complete marker. When the input is a
hydrated staging archive the SQLite index is only ever **read** (via
``archive.index.query_runs``) — never written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import HubUnavailableError
from daydream.archive.index import readonly_connection
from daydream.json_utils import atomic_write_bytes, canonical_json as _canonical, umask_derived_mode
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.snapshot import build_canonical_record, snapshot_id
from daydream.training.dispositions import DECISIVE_DISPOSITIONS
from daydream.training.labeler_signals import resolution_from_dict

__all__ = ["run_materialize"]

_SESSIONS_OUT_FILENAME = "sessions.jsonl"
_MANIFEST_FILENAME = "preview-manifest.json"
_ANNOTATIONS_FILENAME = "annotations.jsonl"

# Disposition written for a conflicted generation's materialized records
# (sessions.jsonl). The operator queue (``queue.build_queue``'s default
# non-decisive set) and the final bundle's sessions.jsonl must route the
# finding to task-only adjudication -- never gold -- and the archive
# `rubric_json` keeps the real decisive disposition for provenance (the
# canonical harvest restores it from the fresh queue). Corpus-v2's gold gate
# (``tiers.classify_tier``) keys solely on disposition/evidence and never
# reads the ``conflicting`` flag, so a decisive disposition here would still
# classify gold; a non-decisive disposition forces ``task-only``.
_CONFLICTED_DISPOSITION = "ambiguous"


def index_sessions(index_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Load sessions from ``sessions.jsonl`` when present, else the hydrated index."""
    if (index_root / _SESSIONS_OUT_FILENAME).is_file():
        return _load_sessions(index_root)
    return _sessions_from_hydrated_stage(index_root)


def _sessions_from_hydrated_stage(index_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Build queue-consumable session records from a hydrated staging archive.

    Production trajectories have no embedded resolutions. Their finding
    identities and fresh GitHub evidence feed the same annotation builder as
    ``corpus harvest``, entirely in memory. This happens even when historical
    annotations exist, so canonical drift checks cannot reuse stale replies.
    DB-only histories and legacy trajectories retain the stored-resolution
    adapter. Index reads never create SQLite sidecars or update bronze.

    Latest-observation selection (deterministic): a session's rows are
    grouped by the harvester dedup key ``(evidence_sha, labeler_policy_version,
    reply_evidence_digest, labels, has_posterior)``; within an identical key
    the latest ``observed_at`` wins; across distinct keys the winner follows
    the archive's ``_PRECEDENCE_ORDER`` (human-first ``source='human'``, then
    ``observed_at DESC``). The session is **conflicting** only when its
    generations disagree in disposition-relevant content — more than one
    distinct decision-bearing ``labels`` set across the dedup-key groups,
    scoped to non-human rows (``_winning_observation``): the archive appends
    fresh generations with identical labels on policy-version bumps,
    edited-reply digest changes, and label-preserving observation overlays
    (``index.append_label_observation``), so a dedup-key split alone never
    marks a session non-gold, and neither a human override row (authoritative
    under the archive's precedence) nor a non-decisive-only generation (a
    pre-adjudication evolution) does. Production bronze supplies fresh
    resolutions; the winner supplies them only for stored-history fallbacks.
    Every emitted record for a conflicted session
    carries ``"conflicting": true`` with the disposition neutralized to
    ``_CONFLICTED_DISPOSITION`` (surfaced non-gold downstream, never merged
    away). A session whose rows carry no materializable per-finding
    resolutions (``rubric_json`` NULL — a human ``daydream label`` row — or
    a legacy labels-only row) and no sanitized per-run trajectory contributes
    no records at all: such sessions are evidence-only (e.g. rows a runbook
    step-3b import admitted from a backup root outside the curation) and must
    not fail the whole curation.

    The index revision is the pinned source commit (the single revision
    directory under ``downloads/``) — a full 40-hex SHA, exactly what the
    publication machinery's pinned-revision resolver accepts.
    """
    if not (index_root / "index.db").is_file():
        raise HubUnavailableError(
            f"hydrated index sessions file not found: {index_root / 'sessions.jsonl'}"
        )
    rows = _query_runs_readonly(index_root / "index.db")
    if not rows:
        raise HubUnavailableError(f"hydrated index at {index_root} has no runs")
    sessions: list[dict[str, Any]] = []
    for row in rows:
        session_id = str(row["session_id"])
        observations = _label_observations_readonly(index_root / "index.db", session_id)
        conflicting = False
        resolutions = _semantic_resolutions_readonly(index_root, row)
        session: dict[str, Any]
        if observations:
            winner, conflicting = _winning_observation(observations)
            rubric_raw = winner.get("rubric_json")
            if rubric_raw is not None:
                try:
                    rubric = json.loads(rubric_raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise HubUnavailableError(
                        f"session {session_id!r}: unreadable winning rubric_json: {exc}"
                    ) from exc
                if not isinstance(rubric, dict):
                    raise HubUnavailableError(
                        f"session {session_id!r}: winning rubric_json is not an object"
                    )
                per_finding = rubric.get("per_finding_resolutions")
                if resolutions is None and isinstance(per_finding, list) and per_finding:
                    resolutions = per_finding
        # DB-only history and legacy embedded resolutions remain supported.
        # Production bronze was already acquired above, independently of the
        # winner's dispositions and without changing the pinned source tree.
        if resolutions is None:
            resolutions = _trajectory_resolutions_readonly(index_root, session_id)
            if resolutions is None:
                continue
        if not resolutions:
            continue  # An explicitly empty bronze finding inventory has no records.
        session = {
            "session_id": session_id,
            "trajectory_id": session_id,
            "segment_id": session_id,
            "resolutions": resolutions,
        }
        if conflicting:
            session["conflicting"] = True
        sessions.append(session)
    downloads = index_root / "downloads"
    if not downloads.is_dir():
        raise HubUnavailableError(f"hydrated index at {index_root} has no downloads/ revision pin")
    revisions = sorted(p.name for p in downloads.iterdir() if p.is_dir())
    if len(revisions) != 1:
        raise HubUnavailableError(
            f"hydrated index at {index_root} has {len(revisions)} downloaded revisions; "
            "expected exactly one pinned source commit"
        )
    return sessions, revisions[0]


def _semantic_resolutions_readonly(
    index_root: Path, row: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Acquire live evidence for production bronze, including on re-harvest.

    Legacy snapshots with embedded resolutions and DB-only imported histories
    retain their stored-evidence adapter. Production trajectories never need
    an annotation field or a prior canonical write.
    """
    from daydream.training.harvest import HarvestConfig, collect_annotation, make_harvest_services
    from daydream.training.harvest_types import HarvestRow
    from daydream.trajectory import run_directory, run_document_path
    from daydream.ui import create_console

    run_dir = run_directory(index_root, str(row["session_id"]))
    path = run_document_path(run_dir)
    if not path.is_file():
        return None
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be an object")
        if "resolutions" in trajectory:
            return None
        findings_path = run_dir / "findings.json"
        findings = json.loads(findings_path.read_text()).get("findings") if findings_path.is_file() else None
        if findings == [] or (findings is None and row.get("total_findings") == 0):
            return []
        # Hydration owns the bronze path; never follow an archived producer's
        # absolute archive_path into a different tree.
        harvest_row = HarvestRow.from_mapping(
            {**row, "archive_path": str(run_dir.resolve())}, row_number=1,
        )
        config = HarvestConfig(archive_dir=index_root, dry_run=True)
        _linked_row, payload = collect_annotation(
            harvest_row, services=make_harvest_services(config), readonly=True,
            console=create_console(),
        )
        rubric = json.loads(payload.rubric_json or "{}")
        resolutions = rubric.get("per_finding_resolutions")
        if not isinstance(resolutions, list) or not resolutions:
            raise ValueError("no per-finding resolutions; missing recorded finding identities")
        by_fingerprint = {str(finding["fingerprint"]): finding for finding in findings or []}
        provenance_keys = (
            "profile_schema_version", "profile_name", "profile_source_kind", "profile_digest", "stack",
        )
        return [
            {
                **{key: row.get(key) for key in provenance_keys},
                **{key: value for key, value in by_fingerprint.get(resolution["fingerprint"], {}).items()
                   if key in provenance_keys},
                **resolution,
            }
            for resolution in resolutions
        ]
    except Exception as exc:
        raise HubUnavailableError(
            f"semantic preview for session {row['session_id']!r} failed: {exc}"
        ) from exc


def _trajectory_resolutions_readonly(
    index_root: Path, session_id: str
) -> list[dict[str, Any]] | None:
    """Read a session's per-finding resolutions from the sanitized per-run
    trajectory (the pre-#1095 materialization source). Used when the hydrated
    staging archive has no materializable ``label_observations`` history for
    the session: no observation rows yet (freshly hydrated stage; canonical
    harvest appends the first rows), a NULL ``rubric_json`` (human-sourced
    row), or only legacy labels-only rows whose rubric_json carries no
    ``per_finding_resolutions`` (pre-#1095 ``Rubric.to_dict``; such rows are
    appended verbatim by the import path, runbook step 3b).

    Returns ``None`` when the trajectory is absent -- the session has no
    materializable content at all (e.g. a session a runbook step-3b import
    admitted from a backup root outside the curation: DB-only ``runs`` row,
    evidence-only observation rows, no ``runs/<sid>`` files) and contributes
    no records; the caller skips it. Anything *present* but unreadable,
    malformed, or empty still raises ``HubUnavailableError`` naming the
    session -- corrupt data is never silently skipped.
    """
    from daydream.trajectory import run_directory, run_document_path

    trajectory_path = run_document_path(run_directory(index_root, session_id))
    if not trajectory_path.is_file():
        return None
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HubUnavailableError(
            f"unreadable hydrated trajectory at {trajectory_path}: {exc}"
        ) from exc
    resolutions = trajectory.get("resolutions") if isinstance(trajectory, dict) else None
    if not isinstance(resolutions, list) or not resolutions:
        raise HubUnavailableError(
            f"hydrated trajectory for session {session_id!r} carries no "
            "per-finding resolutions to materialize"
        )
    return resolutions


def _readonly_query(
    db_path: Path, sql: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    """Run one SELECT over a **read-only** ``mode=ro&immutable=1`` URI —
    never ``_get_connection``, which opens read-write and runs WAL pragmas
    against the hydrated staging index; ``immutable=1`` also keeps a WAL-mode
    db from materializing ``-shm``/``-wal`` sidecars on read, while a surviving
    uncheckpointed ``index.db-wal`` is rejected by ``readonly_connection``.
    """
    try:
        conn = readonly_connection(db_path.parent)
    except ValueError as exc:
        raise HubUnavailableError(str(exc)) from exc
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _query_runs_readonly(db_path: Path) -> list[dict[str, Any]]:
    return _readonly_query(db_path, "SELECT * FROM runs")


def _label_observations_readonly(db_path: Path, session_id: str) -> list[dict[str, Any]]:
    return _readonly_query(
        db_path,
        "SELECT * FROM label_observations WHERE session_id = ?",
        (session_id,),
    )


def _winning_observation(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Deterministic latest-observation selection (see the module docstring for
    the rule). Returns the winning row and whether the session is conflicting
    (more than one distinct disposition-relevant ``labels`` set across the
    dedup-key groups — see ``index.append_label_observation``; a dedup-key
    split that preserves the labels — policy-version bump, edited-reply
    digest change, label-preserving overlay — is agreeing generations, not a
    conflict).
    """
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for obs in observations:
        key = (
            obs.get("evidence_sha"),
            obs.get("labeler_policy_version"),
            obs.get("reply_evidence_digest"),
            obs.get("labels"),
            obs.get("has_posterior"),
        )
        existing = groups.get(key)
        if existing is None or str(obs.get("observed_at", "")) > str(existing.get("observed_at", "")):
            groups[key] = obs
    winners = list(groups.values())
    winner = max(
        winners,
        key=lambda o: (
            1 if o.get("source") == "human" else 0,
            str(o.get("observed_at", "")),
        ),
    )
    # Conflict is decided from disposition-relevant content (the archived
    # labels projection), never from the full dedup tuple: two rows agreeing
    # on the disposition but split on evidence_sha / policy version / reply
    # digest / has_posterior are agreeing generations that stay gold-eligible.
    # The comparison is additionally scoped (issue #336): a human override row
    # is authoritative under the archive's precedence (``_PRECEDENCE_ORDER``,
    # ``index.update_labels``' human-wins contract) -- never a disagreeing
    # generation -- so it cannot contribute a distinct label set; and only
    # decision-bearing label sets (labels claiming a decisive
    # ``finding-<disposition>``) count, so a pre-adjudication generation
    # (``[]`` labels, an ``unanswered``-only snapshot) that a later decisive
    # generation resolves is an evolution -- resolved-unanswered -> accepted --
    # not a harvester disagreement, and stays gold-eligible after
    # re-materialization.
    decisive_sets = {
        o.get("labels")
        for o in winners
        if o.get("source") != "human" and _labels_claim_decisive(o.get("labels"))
    }
    return winner, len(decisive_sets) > 1


def _labels_claim_decisive(labels: Any) -> bool:
    """True when the archived labels projection claims at least one decisive
    ``finding-<disposition>`` label (e.g. ``finding-accepted``). Rows store the
    labels column as a JSON array string; non-string / non-list / unparsable
    values claim nothing (a session with no decisive claim cannot disagree
    about a gold disposition). Non-decisive finding labels (``finding-unanswered``)
    and non-finding labels (``posterior``) are not claims.
    """
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except ValueError:
            return False
    if not isinstance(labels, list):
        return False
    for label in labels:
        if not isinstance(label, str):
            continue
        disposition = label[len("finding-"):] if label.startswith("finding-") else None
        if disposition in DECISIVE_DISPOSITIONS:
            return True
    return False


def run_materialize(
    index_root: Path,
    out_dir: Path,
    *,
    pin: dict[str, str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Materialize the preview snapshot for one curation pin.

    Loads the hydrated index's sessions (via ``preview._load_sessions`` —
    raises the ``HydrationError`` family on a missing/unreadable index and
    ``MovingBranchError`` on a symbolic index revision), builds one canonical
    record per finding (every disposition — automatic decisive, human-decisive,
    and non-decisive), and emits:

    - ``out_dir/sessions.jsonl``: canonical-JSON records sorted by
      ``record_id``, written atomically. Identical index + pin ⇒
      byte-identical file (C4).
    - ``out_dir/preview-manifest.json``: all K2 pin components plus the
      content-addressed ``snapshot_id`` and ``index_revision``, canonical JSON.

    The ``snapshot_id`` is derived from the pin + the emitted records'
    evidence digests, so any evidence change yields a new id (AC 8) — a stale
    id is never reused. Missing/empty pin components raise ``ValueError``
    naming the field (propagated from ``snapshot.snapshot_id``).

    ``dry_run=True`` validates everything and returns the summary without
    writing any file.
    """
    # Validate the pin before touching its components in the loop body:
    # ``snapshot_id`` raises the documented ValueError naming the missing
    # component, never a KeyError from ``pin["evidence_observed_at"]``.
    pin_id = snapshot_id(pin)
    sessions, index_revision = index_sessions(index_root)

    records: list[dict[str, Any]] = []
    for session in sessions:
        resolutions = session.get("resolutions")
        if not isinstance(resolutions, list):
            continue
        for row in resolutions:
            if not isinstance(row, dict):
                raise ValueError(f"materialize: non-object resolution row in session data: {row!r}")
            resolution = resolution_from_dict(row)
            record = build_canonical_record(
                session,
                resolution,
                evidence_observed_at=pin["evidence_observed_at"],
                as_of=pin.get("as_of"),
            )
            if session.get("conflicting"):
                # The session-level conflict flag rides on every emitted
                # per-finding record so downstream consumers (canonical
                # harvest) can exclude the disposition from decisive labels
                # while the full record — flag included — lands in rubric_json.
                # The winner's decisive disposition is neutralized to
                # ``_CONFLICTED_DISPOSITION``: the operator queue (build_queue's
                # default non-decisive set) and the final bundle's sessions.jsonl
                # then route the finding to task-only adjudication — never gold,
                # one disposition in the bundle — while the canonical harvest
                # restores the real decisive disposition for the archive
                # rubric_json provenance from the freshly re-derived queue.
                record["conflicting"] = True
                record["disposition"] = _CONFLICTED_DISPOSITION
                # The record embeds the session-shape view (``resolutions``)
                # that ``project_findings``/``build_queue`` consume; neutralize
                # its disposition too, or the operator queue would still
                # classify the finding gold (``tiers.classify_tier`` reads the
                # resolution, never the record's top-level disposition).
                for nested in record.get("resolutions") or []:
                    if isinstance(nested, dict):
                        nested["disposition"] = _CONFLICTED_DISPOSITION
            records.append(record)
    records.sort(key=lambda r: str(r["record_id"]))

    id_digest = hashlib.sha256(
        (
            pin_id
            + ":"
            + hashlib.sha256(
                "".join(str(r["evidence_digest"]) for r in records).encode("utf-8")
            ).hexdigest()
        ).encode("utf-8")
    ).hexdigest()

    summary: dict[str, Any] = {
        "snapshot_id": id_digest,
        "index_revision": index_revision,
        "record_count": len(records),
    }
    if dry_run:
        return summary

    atomic_write_bytes(
        out_dir / _SESSIONS_OUT_FILENAME,
        "".join(_canonical(r) + "\n" for r in records).encode("utf-8"),
        fsync=False,
        dir_fsync=False,
        mode=umask_derived_mode(),
    )
    manifest: dict[str, Any] = dict(pin)
    manifest["snapshot_id"] = id_digest
    manifest["index_revision"] = index_revision
    atomic_write_bytes(
        out_dir / _MANIFEST_FILENAME,
        _canonical(manifest).encode("utf-8"),
        fsync=False,
        dir_fsync=False,
        mode=umask_derived_mode(),
    )
    return summary
