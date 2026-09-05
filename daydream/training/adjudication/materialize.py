"""Preview materialization of per-finding annotation snapshots (issue #1055).

Runs the #980 semantic resolutions stored in a hydrated index's
``label_observations.rubric_json`` through the shared serializer
(``snapshot.build_canonical_record``) and emits a deterministic
``sessions.jsonl`` plus a pin-pinned ``preview-manifest.json``.

Preview mode guarantees (AC 4 / M2): never appends ``label_observations``,
never writes any resume-cache or harvest-complete marker. When the input is a
hydrated staging archive the SQLite index is only ever **read** (via
``archive.index.query_runs``) — never written.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast, get_args

from daydream.archive.hydrate import HubUnavailableError
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.snapshot import build_canonical_record, snapshot_id
from daydream.training.labeler_signals import (
    PerFindingDisposition,
    PerFindingResolution,
    resolution_from_dict,
)

__all__ = ["run_materialize"]

_SESSIONS_OUT_FILENAME = "sessions.jsonl"
_MANIFEST_FILENAME = "preview-manifest.json"


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_atomic(out_path: Path, payload: str) -> None:
    """Temp-file + ``os.replace`` write, mirroring ``export.write_export_rows``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, out_path)


def _sessions_from_hydrated_stage(index_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Build queue-consumable session records from a hydrated staging archive.

    A hydrated staging archive (``archive.hydrate.run_hydrate_hub``) has no
    ``sessions.jsonl``: its per-finding resolutions live in the SQLite
    index's ``label_observations.rubric_json`` (the canonical dict shape
    emitted by ``Rubric.to_dict``). This adapter joins ``query_runs`` rows
    to their observations into the same session shape ``_load_sessions``
    returns — the observations connection is opened read-only, fail-closed
    on any missing or adjudication-empty run.

    Latest-observation selection (deterministic): a session's rows are
    grouped by the harvester dedup key ``(evidence_sha, labeler_policy_version,
    reply_evidence_digest, labels, has_posterior)``; within an identical key
    the latest ``observed_at`` wins; across distinct keys the winner follows
    the archive's ``_PRECEDENCE_ORDER`` (human-first ``source='human'``, then
    ``observed_at DESC``). More than one distinct dedup key means the session
    is **conflicting**: the winner still supplies the resolutions and every
    emitted record for that session carries ``"conflicting": true`` (surfaced
    non-gold downstream, never merged away).

    The index revision is the pinned source commit (the single revision
    directory under ``downloads/``) — a full 40-hex SHA, exactly what the
    publication machinery's pinned-revision resolver accepts.
    """
    if not (index_root / "index.db").is_file():
        raise HubUnavailableError(
            f"hydrated index sessions file not found: {index_root / 'sessions.jsonl'}"
        )
    sessions: list[dict[str, Any]] = []
    for row in _query_runs_readonly(index_root / "index.db"):
        session_id = str(row["session_id"])
        observations = _label_observations_readonly(index_root / "index.db", session_id)
        session: dict[str, Any]
        if observations:
            winner, conflicting = _winning_observation(observations)
            try:
                rubric = json.loads(winner["rubric_json"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HubUnavailableError(
                    f"session {session_id!r}: unreadable winning rubric_json: {exc}"
                ) from exc
            if not isinstance(rubric, dict):
                raise HubUnavailableError(
                    f"session {session_id!r}: winning rubric_json is not an object"
                )
            resolutions = rubric.get("per_finding_resolutions")
            if not isinstance(resolutions, list) or not resolutions:
                raise HubUnavailableError(
                    f"session {session_id!r}: winning rubric_json carries no "
                    "per_finding_resolutions to materialize (legacy labels-only rows "
                    "are not backfilled)"
                )
            session = {
                "session_id": session_id,
                "trajectory_id": session_id,
                "segment_id": session_id,
                "resolutions": resolutions,
            }
            if conflicting:
                session["conflicting"] = True
        else:
            # Freshly hydrated staging archive: no label_observations history
            # yet (canonical harvest appends the first rows, runbook step 5).
            # Fall back to the sanitized per-run trajectory the hydration gate
            # guarantees (_REQUIRED_SESSION_ARTIFACTS) -- the pre-#1095
            # materialization source -- so the first preview pass over a new
            # stage still works. Once any observation row exists for the
            # session it wins; the two sources are never mixed.
            resolutions = _trajectory_resolutions_readonly(index_root, session_id)
            session = {
                "session_id": session_id,
                "trajectory_id": session_id,
                "segment_id": session_id,
                "resolutions": resolutions,
            }
        sessions.append(session)
    if not sessions:
        raise HubUnavailableError(f"hydrated index at {index_root} has no runs")
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


def _trajectory_resolutions_readonly(index_root: Path, session_id: str) -> list[dict[str, Any]]:
    """Read a session's per-finding resolutions from the sanitized per-run
    trajectory (the pre-#1095 materialization source). Used only when the
    hydrated staging archive carries no ``label_observations`` history for
    the session yet (freshly hydrated stage; canonical harvest appends the
    first rows). Fail-closed: a missing/unreadable/empty trajectory raises
    ``HubUnavailableError`` naming the session -- never silently skipped.
    """
    trajectory_path = index_root / "runs" / session_id / "trajectory.json"
    if not trajectory_path.is_file():
        raise HubUnavailableError(
            f"hydrated index session {session_id!r} has no label_observations rows "
            f"and no trajectory at {trajectory_path}"
        )
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


def _query_runs_readonly(db_path: Path) -> list[dict[str, Any]]:
    """Read all ``runs`` rows over a **read-only** connection (``mode=ro``
    URI — ``query_runs`` goes through ``_get_connection``, which opens
    read-write, runs ``PRAGMA journal_mode=WAL`` against the hydrated
    staging index, and leaves ``-wal``/``-shm`` sidecars behind).
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM runs").fetchall()]
    finally:
        conn.close()


def _label_observations_readonly(db_path: Path, session_id: str) -> list[dict[str, Any]]:
    """Read one session's ``label_observations`` rows over a **read-only**
    connection (``mode=ro&immutable=1`` URI — never ``_get_connection``,
    which opens read-write and runs WAL pragmas against the hydrated
    staging index; ``immutable=1`` also keeps a WAL-mode db from
    materializing ``-shm``/``-wal`` sidecars on read).
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM label_observations WHERE session_id = ?", (session_id,)
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def _winning_observation(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Deterministic latest-observation selection (see the module docstring for
    the rule). Returns the winning row and whether the session is conflicting
    (more than one distinct dedup key).
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
    return winner, len(groups) > 1


def _resolution_from_row(row: dict[str, Any]) -> PerFindingResolution:
    """Rebuild the #980 semantic-resolution object from a stored resolution entry.

    Delegates to the canonical ``labeler_signals.resolution_from_dict``;
    fail-closed: a stored entry missing a required field raises ``ValueError``
    naming the field and fingerprint — never ``None``-coerced into a record.
    """
    fingerprint = row.get("fingerprint")
    disposition = row.get("disposition")
    evidence_digest = row.get("evidence_digest")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"materialize: stored resolution is missing 'fingerprint': {row!r}")
    if not isinstance(disposition, str) or not disposition:
        raise ValueError(
            f"materialize: resolution for fingerprint {fingerprint!r} is missing 'disposition'"
        )
    if not isinstance(evidence_digest, str) or not evidence_digest:
        raise ValueError(
            f"materialize: resolution for fingerprint {fingerprint!r} is missing 'evidence_digest'"
        )
    if disposition not in get_args(PerFindingDisposition):
        raise ValueError(
            f"materialize: resolution for fingerprint {fingerprint!r} has unknown "
            f"disposition {disposition!r}"
        )
    typed_disposition = cast(PerFindingDisposition, disposition)
    return resolution_from_dict(
        {
            "fingerprint": fingerprint,
            "comment_id": row.get("comment_id"),
            "disposition": typed_disposition,
            "evidence": list(row.get("evidence") or []),
            "evidence_digest": evidence_digest,
        }
    )


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
    if (index_root / _SESSIONS_OUT_FILENAME).is_file():
        sessions, index_revision = _load_sessions(index_root)
    else:
        # Hydrated staging archive: derive the sessions from the SQLite
        # index's label_observations (read-only).
        sessions, index_revision = _sessions_from_hydrated_stage(index_root)

    # Validate the pin before touching its components in the loop body:
    # ``snapshot_id`` raises the documented ValueError naming the missing
    # component, never a KeyError from ``pin["evidence_observed_at"]``.
    pin_id = snapshot_id(pin)

    records: list[dict[str, Any]] = []
    for session in sessions:
        resolutions = session.get("resolutions")
        if not isinstance(resolutions, list):
            continue
        for row in resolutions:
            if not isinstance(row, dict):
                raise ValueError(f"materialize: non-object resolution row in session data: {row!r}")
            resolution = _resolution_from_row(row)
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
                record["conflicting"] = True
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

    _write_atomic(
        out_dir / _SESSIONS_OUT_FILENAME,
        "".join(_canonical(r) + "\n" for r in records),
    )
    manifest: dict[str, Any] = dict(pin)
    manifest["snapshot_id"] = id_digest
    manifest["index_revision"] = index_revision
    _write_atomic(out_dir / _MANIFEST_FILENAME, _canonical(manifest))
    return summary
