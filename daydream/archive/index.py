"""SQLite archive index with immutable, bitemporal label observations.

Connections bootstrap the schema. Label projections prefer human observations,
then recency; appends refresh the run cache from that winner.

Time cutoffs compare TEXT lexically, so writers and cutoff boundaries use UTC
``datetime.isoformat()``: ``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``. Legacy ``Z``
rows remain untouched. At an equal second they sort after canonical rows, which
can shrink the reviewer-prior pool but cannot leak future outcomes. Re-harvest
appends canonical generations; the corpus leakage guard parses timestamps.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from daydream.archive._schema import (
    _CREATE_INDEXES,
    _CREATE_LABEL_OBSERVATIONS_TABLE,
    _CREATE_TABLE,
    _PRECEDENCE_ORDER,
    _UPSERT_SQL,
    LABEL_OBSERVATION_NAMES,
    RUNS_COLUMNS,
    SCHEMA_VERSION,
    _migrate_label_observations_schema,
    _migrate_schema,
    _recreate_label_observations_if_stale,
)
from daydream.archive.git_safe import normalize_remote_url
from daydream.training.labeler_versions import STALE_LEGACY
from daydream.training.reward import FP_PENALTY_MAP

# Observation body order follows the schema after session_id and observed_at.
_LABEL_OBSERVATION_ROW_BODY_NAMES = LABEL_OBSERVATION_NAMES[2:]
_INSERT_LABEL_OBSERVATION_SQL = (
    f"INSERT INTO label_observations ({', '.join(LABEL_OBSERVATION_NAMES)}) "
    f"VALUES ({', '.join('?' * len(LABEL_OBSERVATION_NAMES))})"
)
_SELECT_LABEL_OBSERVATION_ROW_SQL = (
    f"SELECT {', '.join(_LABEL_OBSERVATION_ROW_BODY_NAMES)} FROM label_observations "
    "WHERE session_id = ? AND observed_at = ?"
)

# Re-export for callers (including tests) that import these names from this module.
__all__ = [
    "SCHEMA_VERSION",
    "RUNS_COLUMNS",
    "LABEL_OBSERVATION_NAMES",
    "_CREATE_TABLE",
    "upsert_run",
    "manifest_index_fields",
    "update_labels",
    "query_runs",
    "append_label_observation",
    "latest_label_observation",
    "reviewer_set_penalty_prior",
    "label_observation_history",
    "set_run_pr_link",
    "canonical_utc_iso",
    "normalize_as_of",
]


def canonical_utc_iso(ts: str) -> str:
    """Convert an aware ISO-8601 timestamp to canonical UTC; reject malformed or naive input."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"naive timestamp {ts!r}: an explicit UTC offset is required")
    return dt.astimezone(timezone.utc).isoformat()


def normalize_as_of(value: str) -> str:
    """Canonicalize a reproducibility pin at its entry boundary."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"as_of {value!r} is not a valid ISO-8601 timestamp") from None
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"as_of {value!r} must be a UTC timestamp (ending in Z or +00:00)")
    return dt.astimezone(timezone.utc).isoformat()


def readonly_connection(archive_dir: Path) -> sqlite3.Connection:
    """Read a checkpointed index without schema changes or SQLite sidecars."""
    db_path = archive_dir / "index.db"
    if db_path.with_name(db_path.name + "-wal").exists():
        raise ValueError(f"index {db_path} has an uncheckpointed WAL; checkpoint before reading")
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _get_connection(archive_dir: Path) -> sqlite3.Connection:
    """Open a row-based connection, bootstrap the schema, and enable WAL with a busy timeout."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    db_path = archive_dir / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_LABEL_OBSERVATIONS_TABLE)
    _recreate_label_observations_if_stale(conn)
    _migrate_label_observations_schema(conn)
    _migrate_schema(conn)
    if version != SCHEMA_VERSION:
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


@contextmanager
def _connection(archive_dir: Path, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Yield an index connection, closing it on exit — read-only when *readonly*."""
    conn = readonly_connection(archive_dir) if readonly else _get_connection(archive_dir)
    try:
        yield conn
    finally:
        conn.close()


def _project_daydream(daydream: Any) -> dict[str, Any]:
    """Project executable provenance; absent provenance and non-bool dirty states become NULL."""
    values = {
        f"daydream_{name}": daydream.get(name) if daydream is not None else None
        for name in ("version", "install_source", "commit", "dirty", "container_digest")
    }
    dirty = values["daydream_dirty"]
    values["daydream_dirty"] = int(dirty) if isinstance(dirty, bool) else None
    return values


_RUN_DEFAULTS: dict[str, Any] = {
    "session_id": "",
    "archived_at": "",
    "status": "complete",
    "archive_status": "complete",
    "pipeline_status": "unknown",
    "run_flow": "",
    "backend": "claude",
    "review_only": False,
    "deep": False,
    "changed_files": [],
    "outcome_labels": "[]",
    "archive_path": "",
}


def manifest_index_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    """Decode portable manifest blocks and flat legacy fields in historical precedence.

    The index owns this persisted projection. Native executable identity stays in
    its separate block; hydration explicitly discards that untrusted provenance.
    """
    valid = {col.name for col in RUNS_COLUMNS if col.upserted and not col.name.startswith("daydream_")}
    values = {key: value for key, value in data.items() if key in valid}
    if "daydream" in data:
        values["daydream"] = data["daydream"]
    for block_name in ("run", "git", "code_context", "pr", "metrics", "outcome"):
        block = data.get(block_name)
        if not isinstance(block, dict):
            continue
        aliases = {
            "run": {"flow": "run_flow"},
            "pr": {"number": "pr_number", "repo": "pr_repo"},
            "outcome": {"labels": "outcome_labels"},
        }.get(block_name, {})
        for key, value in block.items():
            name = aliases.get(key, key)
            if name in valid:
                values[name] = json.dumps(value) if name == "outcome_labels" else value
    return values


def _run_upsert_values(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Project portable or legacy wire, then normalize index values at persistence."""
    fields = manifest_index_fields(fields)
    values: dict[str, Any] = {
        col.name: fields.get(col.name, _RUN_DEFAULTS.get(col.name)) for col in RUNS_COLUMNS if col.upserted
    }
    # Normalize again at persistence even if capture bypassed URL sanitization.
    slug, url = ((values["repo_slug"], None) if values["remote_url"] is None
                 else normalize_remote_url(values["remote_url"]))
    values.update(
        **_project_daydream(fields.get("daydream")),
        review_only=int(values["review_only"]),
        deep=int(values["deep"]),
        remote_url=url,
        repo_slug=slug,
        changed_files=json.dumps(values["changed_files"]),
        phase_states=json.dumps(values["phase_states"]) if values["phase_states"] is not None else None,
        fix_quality_gate=json.dumps(values["fix_quality_gate"]) if values["fix_quality_gate"] is not None else None,
        schema_version=SCHEMA_VERSION,
    )
    return values


def upsert_run(archive_dir: Path, fields: Mapping[str, Any]) -> None:
    """Persist portable or legacy metadata through the declared column/parameter contract."""
    with _connection(archive_dir) as conn:
        conn.execute(_UPSERT_SQL, _run_upsert_values(fields))
        conn.commit()


def append_label_observation(
    archive_dir: Path,
    session_id: str,
    *,
    labels: list[str],
    pr_state: str | None,
    labeler_version: str,
    evidence_sha: str | None,
    rubric_json: str | None = None,
    valid_at: str | None = None,
    reward_version: str | None = None,
    reward_json: str | None = None,
    composite_reward: float | None = None,
    reviewer_logins: list[str] | None = None,
    has_posterior: bool = False,
    source: str = "auto",
    reply_classifier_version: str | None = None,
    reply_evidence_digest: str | None = None,
    labeler_policy_version: str | None = None,
    legacy: str = "auto",
    observed_at: str | None = None,
) -> bool:
    """Append an immutable observation and refresh the run cache in one transaction."""
    if valid_at is not None:
        valid_at = canonical_utc_iso(valid_at)
    if observed_at is not None:
        # Validate imported transaction time before any write.
        try:
            parsed = datetime.fromisoformat(observed_at)
        except ValueError:
            raise ValueError(
                f"observed_at {observed_at!r} is not a parseable ISO-8601 timestamp"
            ) from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(
                f"observed_at {observed_at!r} is naive: an explicit UTC offset is required"
            )
        observed_dt = parsed.astimezone(timezone.utc)
    else:
        observed_dt = datetime.now(timezone.utc)
    # Legacy policy stays NULL so the gold gate cannot admit unknown provenance.
    if labeler_policy_version == STALE_LEGACY:
        labeler_policy_version = None
        legacy = "legacy"
    elif labeler_policy_version is None:
        labeler_policy_version = labeler_version
    labels_json = json.dumps(labels)
    reviewer_logins_json = json.dumps(reviewer_logins) if reviewer_logins is not None else None
    has_posterior_int = int(has_posterior)
    with _connection(archive_dir) as conn:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id = ?",
            (session_id,),
        )
        if cursor.fetchone() is None:
            msg = f"Unknown session {session_id!r}"
            raise ValueError(msg)
        # Compare with the latest auto row; intervening human overrides cannot mask a
        # re-score.
        if source == "auto":
            latest_auto = conn.execute(
                "SELECT evidence_sha, labeler_policy_version, reply_evidence_digest, labels, has_posterior, "
                "reward_version "
                "FROM label_observations "
                "WHERE session_id = ? AND source = 'auto' "
                "ORDER BY observed_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            # Population membership varies independently of the label (a
            # local_branch outcome is labeled but not posterior evidence).
            if latest_auto is not None and tuple(latest_auto) == (
                evidence_sha,
                labeler_policy_version,
                reply_evidence_digest,
                labels_json,
                has_posterior_int,
                reward_version,
            ):
                return False
        # Bump observed_at by a microsecond and retry on a same-microsecond
        # primary-key collision so the column stays a clean ISO 8601 timestamp.
        while True:
            observed_at = observed_dt.isoformat()
            valid_at_value = valid_at if valid_at is not None else observed_at
            row_body = (
                labels_json,
                pr_state,
                labeler_version,
                evidence_sha,
                rubric_json,
                valid_at_value,
                reward_version,
                reward_json,
                composite_reward,
                reviewer_logins_json,
                has_posterior_int,
                source,
                labeler_policy_version,
                reply_classifier_version,
                reply_evidence_digest,
                legacy,
            )
            try:
                conn.execute(
                    _INSERT_LABEL_OBSERVATION_SQL,
                    (session_id, observed_at, *row_body),
                )
                break
            except sqlite3.IntegrityError:
                # Identical re-imports retain their timestamp; only distinct generations
                # advance it.
                existing = conn.execute(
                    _SELECT_LABEL_OBSERVATION_ROW_SQL,
                    (session_id, observed_at),
                ).fetchone()
                if existing is not None and tuple(existing) == row_body:
                    return False
                observed_dt += timedelta(microseconds=1)
        # Refresh from the human-first winner, which may differ from the newly inserted
        # row.
        winner = conn.execute(
            f"SELECT labels, observed_at, rubric_json, composite_reward, has_posterior "
            f"FROM label_observations WHERE session_id = ? "
            f"ORDER BY {_PRECEDENCE_ORDER} LIMIT 1",
            (session_id,),
        ).fetchone()
        conn.execute(
            "UPDATE runs SET outcome_labels = ?, labeled_at = ?, rubric_json = ?, composite_reward = ?, "
            "has_posterior = ? "
            "WHERE session_id = ?",
            (
                winner["labels"],
                winner["observed_at"],
                winner["rubric_json"],
                winner["composite_reward"],
                winner["has_posterior"],
                session_id,
            ),
        )
        conn.commit()
        return True


def latest_label_observation(
    archive_dir: Path,
    session_id: str,
    *,
    as_of: str | None = None,
) -> dict[str, Any] | None:
    """Return the human-first, then newest observation within the optional cutoff."""
    cutoff = "AND observed_at <= ? " if as_of is not None else ""
    params: tuple[Any, ...] = (session_id,) if as_of is None else (session_id, as_of)
    with _connection(archive_dir) as conn:
        cursor = conn.execute(
            f"SELECT * FROM label_observations WHERE session_id = ? "
            f"{cutoff}ORDER BY {_PRECEDENCE_ORDER} LIMIT 1",
            params,
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None


def reviewer_set_penalty_prior(
    archive_dir: Path,
    logins: list[str],
    *,
    before_valid_at: str,
    exclude_session: str,
    repo_slug: str | None = None,
    readonly: bool = False,
) -> tuple[float | None, int]:
    """Return (mean penalty, count) over prior sessions sharing a reviewer."""
    if not logins:
        return None, 0
    # Canonical spelling makes the SQL time cutoff chronological.
    before_valid_at = canonical_utc_iso(before_valid_at)

    # Filter reviewer intersection in SQLite to avoid loading the full archive.
    placeholders = ",".join("?" * len(logins))

    p = "lo." if repo_slug is not None else ""
    alias = " lo" if repo_slug is not None else ""
    join = "\n                JOIN runs r ON r.session_id = lo.session_id" if repo_slug is not None else ""
    repo_filter = "\n                  AND r.repo_slug = ?" if repo_slug is not None else ""
    params: tuple[Any, ...] = (exclude_session, before_valid_at, *logins)
    if repo_slug is not None:
        params = (*params, repo_slug)
    sql = f"""
            SELECT reviewer_logins, labels
            FROM (
                SELECT {p}reviewer_logins, {p}labels,
                       ROW_NUMBER() OVER (
                           PARTITION BY {p}session_id
                           ORDER BY {p}observed_at DESC
                       ) AS _rn
                FROM label_observations{alias}{join}
                WHERE {p}session_id != ?
                  AND {p}valid_at < ?
                  AND {p}reviewer_logins IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM json_each({p}reviewer_logins)
                      WHERE value IN ({placeholders})
                  ){repo_filter}
            )
            WHERE _rn = 1
            """

    with _connection(archive_dir, readonly=readonly) as conn:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

    penalties: list[float] = []
    for row in rows:
        raw_logins = row["reviewer_logins"]
        # reviewer_logins IS NOT NULL is enforced in SQL; guard retained for
        # safety in case the column somehow carries an empty string.
        if not raw_logins:
            continue
        try:
            row_logins = json.loads(raw_logins)
        except (json.JSONDecodeError, TypeError) as exc:
            warnings.warn(f"Invalid reviewer_logins payload {raw_logins!r}: {exc}", stacklevel=2)
            continue
        if not isinstance(row_logins, list):
            continue
        try:
            row_labels = json.loads(row["labels"])
        except (json.JSONDecodeError, TypeError) as exc:
            warnings.warn(f"Invalid labels payload {row['labels']!r}: {exc}", stacklevel=2)
            continue
        if not isinstance(row_labels, list) or not row_labels:
            continue
        penalty = FP_PENALTY_MAP.get(str(row_labels[0]))
        if penalty is None:
            continue
        penalties.append(penalty)

    if not penalties:
        return None, 0
    return sum(penalties) / len(penalties), len(penalties)


def label_observation_history(archive_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """Return session label rows ordered by ``observed_at`` ascending."""
    with _connection(archive_dir) as conn:
        cursor = conn.execute(
            "SELECT * FROM label_observations WHERE session_id = ? ORDER BY observed_at ASC",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def update_labels(archive_dir: Path, session_id: str, labels: list[str]) -> bool:
    """Append a human override for an exact or uniquely prefixed session id."""
    with _connection(archive_dir) as conn:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id LIKE ? || '%'",
            (session_id,),
        )
        matches = cursor.fetchall()

    if not matches:
        return False

    if len(matches) > 1:
        matched_ids = [row["session_id"] for row in matches]
        msg = f"Prefix '{session_id}' matches {len(matches)} sessions: {matched_ids}. Provide a longer prefix."
        raise ValueError(msg)

    full_id = matches[0]["session_id"]
    append_label_observation(
        archive_dir,
        full_id,
        labels=labels,
        pr_state=None,
        labeler_version="human",
        evidence_sha=None,
        source="human",
    )
    return True


def set_run_pr_link(archive_dir: Path, session_id: str, pr_number: int, pr_repo: str) -> None:
    """Backfill only PR linkage, leaving observations and caches untouched."""
    with _connection(archive_dir) as conn:
        conn.execute(
            "UPDATE runs SET pr_number = ?, pr_repo = ? WHERE session_id = ?",
            (pr_number, pr_repo, session_id),
        )
        conn.commit()


def query_runs(archive_dir: Path, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Query runs using an optional SQL WHERE expression and its bound parameters."""
    with _connection(archive_dir) as conn:
        sql = "SELECT * FROM runs"
        if where:
            sql += f" WHERE {where}"  # noqa: S608 - caller-supplied SQL fragment with bound params
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


