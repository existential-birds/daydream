"""SQLite index for cross-project querying of archived daydream runs.

Manages a SQLite database at ``~/.daydream/archive/index.db`` that indexes
all archived runs by their manifest metadata. The schema is created
idempotently on every connection open, so the database is self-bootstrapping.

Exports:
    SCHEMA_VERSION: Current schema version integer.
    upsert_run: Insert or replace a run from a Manifest.
    update_labels: Update outcome labels for a session (supports prefix matching).
    query_runs: Query runs with optional WHERE clause.
    append_label_observation: Append a row to the immutable bitemporal
        label_observations history (``observed_at`` transaction time,
        ``valid_at`` valid time, reward columns, plus ``reviewer_logins`` and
        the ``has_posterior`` population discriminator) and refresh the
        denormalized runs cache (including the ``has_posterior`` mirror).
    latest_label_observation: Return the highest-precedence (human-first, then
        recency) label_observations row for a session, optionally constrained by
        an ``as_of`` cutoff timestamp.
    reviewer_set_penalty_prior: Pooled mean false-positive penalty over prior
        runs sharing a reviewer (strict ``valid_at`` cutoff), for the posterior
        outcome prior (C4).
    label_observation_history: Return the full label_observations history for
        a session in chronological order.
    canonical_utc_iso: Convert an ISO-8601 timestamp to the canonical UTC
        spelling this index stores and compares (``+00:00`` suffix).
    normalize_as_of: Validate and canonicalize a user-supplied ``as_of`` pin
        (strict: UTC-only input) for lexical comparison against ``observed_at``.

Timestamp canonicalization contract
-----------------------------------

The bitemporal columns are TEXT and every cutoff (``observed_at <= as_of``,
``valid_at < before_valid_at``) is a lexical string comparison, which matches
chronological order only when both sides share one spelling. The canonical
spelling is ``datetime.isoformat()`` in UTC — ``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``
(fractional seconds absent or exactly six digits, never a ``Z`` suffix).

- ``observed_at`` has a single writer (:func:`append_label_observation` stamps
  ``datetime.now(timezone.utc).isoformat()``), so the stored column is uniformly
  canonical and ``observed_at <= as_of`` / ``ORDER BY observed_at`` are safe once
  ``as_of`` is canonical (enforced at its entry boundary via
  :func:`normalize_as_of`).
- ``valid_at`` historically mixed spellings: caller-supplied values (GitHub
  merge timestamps, harvest fallbacks) arrived ``Z``-suffixed while the
  ``None``→``observed_at`` collapse stored ``+00:00``. All rows are now
  canonicalized at write time (:func:`canonical_utc_iso` in
  :func:`append_label_observation`), so the column converges on the canonical
  spelling and the lexical ``valid_at < before_valid_at`` cutoff compares
  chronologically. Rows written by pre-convergence versions may still carry a
  ``Z`` suffix; they are deliberately NOT rewritten or deleted (destructive
  bootstrap migrations are off the table). A stray legacy row sorts after any
  ``+00:00`` string sharing its second prefix, so the reviewer-prior cutoff
  can only over-exclude it (a smaller pool, never posterior leakage); the
  corpus leakage guard parses datetimes and is spelling-immune. A re-harvest
  appends canonical generations that supersede legacy rows in every winner
  projection.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from daydream.archive._schema import (
    _CREATE_INDEXES,
    _CREATE_LABEL_OBSERVATIONS_TABLE,
    _CREATE_TABLE,
    _PRECEDENCE_ORDER,
    _REVIEWER_PENALTY_MAP,
    _UPSERT_SQL,
    LABEL_OBSERVATION_NAMES,
    RUNS_COLUMNS,
    SCHEMA_VERSION,
    _migrate_label_observations_schema,
    _migrate_schema,
    _recreate_label_observations_if_stale,
)
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.known_versions import STALE_LEGACY
from daydream.archive.manifest import Manifest

# The 16 non-identity columns, in the canonical declaration order: exactly the
# values ``row_body`` supplies after the ``(session_id, observed_at)`` prefix.
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
    """Return *ts* in the canonical UTC spelling stored by this index.

    Parses any valid ISO-8601 timestamp (``Z`` or numeric offset, any
    sub-second precision) and re-emits ``datetime.isoformat()`` in UTC:
    ``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``. Aware non-UTC offsets are
    *converted* to UTC — a data timestamp in a foreign zone is an unambiguous
    instant, so conversion is always chronologically correct. Idempotent for
    already-canonical input.

    Raises:
        ValueError: When *ts* is not parseable ISO-8601, or is naive (no
            offset) — a naive timestamp names no single instant, so it cannot
            be canonicalized.
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"naive timestamp {ts!r}: an explicit UTC offset is required")
    return dt.astimezone(timezone.utc).isoformat()


def normalize_as_of(value: str) -> str:
    """Validate and canonicalize a user-supplied ``as_of`` pin.

    The single entry-boundary normalizer for ``as_of``: call it once where the
    pin enters the system (each projection's config boundary); downstream
    consumers — the
    ``observed_at <= as_of`` SQL cutoffs here and the valid-time leakage guard
    in ``daydream.training.corpus`` — receive the canonical spelling and never
    re-normalize.

    Stricter than :func:`canonical_utc_iso`: a non-UTC offset is *rejected*,
    not converted. An operator writing ``+05:00`` on a reproducibility pin is
    almost certainly thinking in local time; silently shifting the pin five
    hours invites irreproducible corpora, so the input must already be UTC
    (``Z`` or ``+00:00``, any sub-second precision).

    Raises:
        ValueError: When *value* is not parseable ISO-8601, is naive, or
            carries a non-UTC offset.
    """
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
    """Open the index database, creating schema if needed.

    Enables WAL mode for concurrent read access and sets a busy timeout
    to handle contention from parallel daydream runs.

    Returns:
        An open sqlite3.Connection with row_factory set to sqlite3.Row.
    """
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


def _project_daydream(daydream: Any) -> dict[str, Any]:
    """Project the ``manifest.daydream`` provenance onto per-column values.

    Collapses the guarded-ternary projection ladder into one place so the five
    ``daydream_*`` projections stay in sync with the ``RUNS_COLUMNS``
    declaration. ``None`` (no executable provenance captured) projects every
    field to ``None``; ``daydream_dirty`` is stored as an int only when it is a
    real bool so a sentinel (non-bool) dirty state persists as ``NULL``.
    """
    if daydream is None:
        return {
            "daydream_version": None,
            "daydream_install_source": None,
            "daydream_commit": None,
            "daydream_dirty": None,
            "daydream_container_digest": None,
        }
    dirty = daydream.dirty
    return {
        "daydream_version": daydream.version,
        "daydream_install_source": daydream.install_source,
        "daydream_commit": daydream.commit,
        "daydream_dirty": int(dirty) if isinstance(dirty, bool) else None,
        "daydream_container_digest": daydream.container_digest,
    }


def _run_upsert_values(manifest: Manifest) -> dict[str, Any]:
    """Project *manifest* onto the run upsert's per-column values.

    The name-keyed value expressions are the one piece of the upsert that stays
    explicit code: :func:`upsert_run` projects the declaration's ``upserted``
    column set through this mapping, so a hand-written parameter name can never
    reach SQLite. ``None`` identity is preserved (a missing source stores
    ``NULL``); the two booleans are stored as ints; JSON columns are serialised
    here.
    """
    daydream = manifest.daydream
    # Defense-in-depth: never persist a credential-bearing remote URL, even if
    # upstream capture bypassed the normalizer. None identity stores None.
    if manifest.remote_url is None:
        # No remote URL to normalize; keep the manifest's slug as-is.
        normalized_slug, normalized_url = manifest.repo_slug, None
    else:
        normalized_slug, normalized_url = normalize_remote_url(manifest.remote_url)
    return {
        "session_id": manifest.session_id,
        "archived_at": manifest.archived_at,
        "status": manifest.status,
        "archive_status": manifest.archive_status,
        "pipeline_status": manifest.pipeline_status,
        "phase_states": json.dumps(manifest.phase_states) if manifest.phase_states is not None else None,
        **_project_daydream(daydream),
        "run_flow": manifest.run_flow,
        "skill": manifest.skill,
        "model": manifest.model,
        "backend": manifest.backend,
        "review_backend": manifest.review_backend,
        "fix_backend": manifest.fix_backend,
        "test_backend": manifest.test_backend,
        "per_stack_review_backend": manifest.per_stack_review_backend,
        "per_stack_review_model": manifest.per_stack_review_model,
        "review_only": int(manifest.review_only),
        "deep": int(manifest.deep),
        "remote_url": normalized_url,
        "repo_slug": normalized_slug,
        "source_path": manifest.source_path,
        "branch": manifest.branch,
        "base_branch": manifest.base_branch,
        "head_sha": manifest.head_sha,
        "base_sha": manifest.base_sha,
        "changed_files": json.dumps(manifest.changed_files),
        "pr_number": manifest.pr_number,
        "pr_repo": manifest.pr_repo,
        "total_cost_usd": manifest.total_cost_usd,
        "total_findings": manifest.total_findings,
        "grounding_rate": manifest.grounding_rate,
        "coverage_ratio": manifest.coverage_ratio,
        "cost_per_finding_usd": manifest.cost_per_finding_usd,
        "wall_clock_seconds": manifest.wall_clock_seconds,
        "erosion": manifest.erosion,
        "verbosity": manifest.verbosity,
        "location_in_hunk_rate": manifest.location_in_hunk_rate,
        "shipped_duplicate_pairs": manifest.shipped_duplicate_pairs,
        "fix_quality_gate": json.dumps(manifest.fix_quality_gate)
        if manifest.fix_quality_gate is not None
        else None,
        "recommended_patch_capture": manifest.recommended_patch_capture,
        "total_prompt_tokens": manifest.total_prompt_tokens,
        "total_completion_tokens": manifest.total_completion_tokens,
        "total_cached_tokens": manifest.total_cached_tokens,
        "outcome_labels": manifest.outcome_labels,
        "labeled_at": manifest.labeled_at,
        "composite_reward": manifest.composite_reward,
        "archive_path": manifest.archive_path,
        "schema_version": SCHEMA_VERSION,
        "profile_schema_version": manifest.profile_schema_version,
        "profile_name": manifest.profile_name,
        "profile_source_kind": manifest.profile_source_kind,
        "profile_digest": manifest.profile_digest,
    }


def upsert_run(archive_dir: Path, manifest: Manifest) -> None:
    """Insert or replace a run entry from a Manifest.

    Bool fields (review_only, deep) are normalized to integers (0/1)
    for SQLite storage. The bound parameter set is projected from the
    ``upserted`` subset of the ``RUNS_COLUMNS`` declaration, so the statement,
    its placeholders and its parameters cannot drift apart.
    """
    values = _run_upsert_values(manifest)
    conn = _get_connection(archive_dir)
    try:
        conn.execute(
            _UPSERT_SQL,
            {col.name: values[col.name] for col in RUNS_COLUMNS if col.upserted},
        )
        conn.commit()
    finally:
        conn.close()


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
    """Append a row to the immutable ``label_observations`` history.

    Writes a single ``(session_id, observed_at)`` row capturing the current
    label decision plus the bitemporal valid time and reward breakdown, and in
    the same transaction refreshes the denormalized
    ``runs.outcome_labels`` / ``runs.labeled_at`` / ``runs.rubric_json`` /
    ``runs.composite_reward`` / ``runs.has_posterior`` cache.

    The cache is recomputed from the *winning* observation under the
    precedence projection (human-first, then most recent) — **not** necessarily
    the row just inserted. A newer automated append therefore cannot dethrone an
    existing human label in the denormalized cache.

    Args:
        archive_dir: Path to the archive root.
        session_id: Full session UUID — must already exist in ``runs``.
        labels: List of label strings; serialised as a JSON array.
        pr_state: One of ``open``/``merged``/``closed``/``reverted`` or
            ``None`` when not applicable (e.g. local-branch runs).
        labeler_version: Free-form version tag of the labeler that produced
            this observation (e.g. ``2026.05.22`` for an automated rubric, or
            ``human`` for a maintainer override).
        evidence_sha: Optional commit SHA / artifact hash that grounds the
            decision; ``None`` when no concrete evidence applies.
        rubric_json: Optional JSON-serialised rubric (``Rubric.to_dict()``).
        valid_at: ISO 8601 valid time — when the outcome the annotation
            describes became true (e.g. a PR merge timestamp). Canonicalized
            via :func:`canonical_utc_iso` before storage so the column
            converges on one spelling regardless of the caller's (GitHub emits
            ``Z``; the collapse path emits ``+00:00``). ``None`` for
            non-PR/local runs, in which case it collapses to ``observed_at``
            so an ``as_of``-pinned corpus never spuriously drops the run (Q2).
        reward_version: Version tag of the reward reducer that produced
            ``reward_json`` (``RewardBreakdown.reward_version``); ``None`` when
            no reward was scored.
        reward_json: Full ``RewardBreakdown.to_dict()`` serialised as JSON so a
            corpus re-projection has every axis; ``None`` when unscored.
        composite_reward: The cached composite reward scalar. Persisted on the
            ``label_observations`` row (so each annotation generation is
            self-describing) and mirrored onto ``runs.composite_reward`` for
            SQL thresholding; ``None`` when uncomputable.
        reviewer_logins: Human GitHub accounts whose review/reply outcomes
            seeded the posterior axis. Serialised as a JSON array on the
            ``label_observations`` row; ``None`` (stored as SQL ``NULL``) for
            non-PR/local runs with no reviewer set.
        has_posterior: Population discriminator. ``True`` when the row carries a
            ``PosteriorBreakdown`` (a mapped PR-outcome label was scored).
            Coerced to ``int`` and written to ``label_observations.has_posterior``
            and mirrored onto ``runs.has_posterior`` so SQL consumers can split
            labeled/unlabeled populations without parsing ``reward_json``.
        reply_classifier_version: Version of the reply classifier that produced
            the per-finding dispositions (version axis, M13); persisted on the
            row so each annotation generation is self-describing.
        reply_evidence_digest: Stable digest over the combined reply evidence
            (versioned dedup input, M14); persisted and compared in the auto
            dedup key so an edited reply appends a new generation.
        labeler_policy_version: Versioned policy axis (M13); persisted on the
            row and part of the auto dedup key so a policy bump appends a new
            generation. ``None`` (the default, keeping pre-policy callers
            unchanged) mirrors ``labeler_version`` into the policy column; the
            ``STALE_LEGACY`` sentinel — the inventory marker for legacy-schema
            source rows with no policy axis — is stored as SQL ``NULL`` with
            the ``legacy`` marker stamped, the representation the corpus gold
            gate (``labeler_policy_version IS NOT NULL``) already treats as
            non-gold.
        legacy: Legacy marker for the row — ``"auto"`` (the default) or
            ``"legacy"`` (a row whose policy axis predates
            ``label_observations`` versioning; never gold-eligible). Mirrors
            the schema's legacy stamping so imported legacy rows persist the
            same representation a migrated in-place row has.
        source: Provenance of this observation — ``"auto"`` (automated rubric
            labeler; the default that keeps existing harvest callers
            unchanged) or ``"human"`` (operator override). Human-sourced rows
            take precedence over automated ones in every projection regardless
            of timing, which is why the cache is written from the winning row
            rather than the inserted one.

    Returns:
        ``True`` when a new observation row was inserted; ``False`` when the
        append was a deduped no-op. Dedup is **auto-only**: an automated
        (``source="auto"``) append matching the latest existing *auto*
        observation for the session on the versioned evidence tuple
        ``(evidence_sha, labeler_policy_version, reply_evidence_digest,
        labels, has_posterior, reward_version)`` is
        skipped without inserting and without touching the cache. A
        policy-version bump, a reward-version bump, or an edited-reply digest
        change on otherwise-identical evidence appends
        a fresh generation. A ``None`` digest is not coerced to ``""`` — ``None``
        never equals a present digest, so digest-less legacy callers dedupe on
        the remaining tuple exactly as before (deliberate default). Human
        (``source != "auto"``) appends are never deduped by the evidence
        tuple; only a byte-identical row already occupying the same
        ``(session_id, observed_at)`` primary key is a no-op re-import (so a
        re-merged source never grows microsecond-shifted duplicates).

        An explicit ``observed_at`` (aware ISO-8601) is preserved bitemporally
        as the row's observation time instead of the wall clock; ``None``
        keeps the existing now() behavior.

    Raises:
        ValueError: When ``session_id`` is not present in the ``runs`` table,
            when ``valid_at`` is not a parseable aware ISO-8601 timestamp, or
            when ``observed_at`` is set and is not a parseable aware ISO-8601
            timestamp.
    """
    if valid_at is not None:
        valid_at = canonical_utc_iso(valid_at)
    if observed_at is not None:
        # Bitemporal preservation: an explicit data timestamp (e.g. imported
        # from a surviving local archive) is stored in place of the wall clock.
        # Fails closed on non-ISO-8601 or naive input before any write.
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
    # Policy axis resolution: versioned callers (the import merge) pass the
    # source's labeler_policy_version verbatim; callers that predate the axis
    # leave it None and the free-form labeler_version mirrors into the policy
    # column (inherited behavior). The STALE_LEGACY sentinel — the inventory-
    # time marker for a legacy-schema row with no policy axis — is stored as
    # the canonical legacy representation: NULL policy + legacy='legacy', so
    # the corpus gold gate (which rejects rows by labeler_policy_version IS
    # NULL) can never admit a row the importer's version gate excluded.
    if labeler_policy_version == STALE_LEGACY:
        labeler_policy_version = None
        legacy = "legacy"
    elif labeler_policy_version is None:
        labeler_policy_version = labeler_version
    labels_json = json.dumps(labels)
    reviewer_logins_json = json.dumps(reviewer_logins) if reviewer_logins is not None else None
    has_posterior_int = int(has_posterior)
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id = ?",
            (session_id,),
        )
        if cursor.fetchone() is None:
            msg = f"Unknown session {session_id!r}"
            raise ValueError(msg)
        # Idempotency: an automated re-score with identical evidence is a no-op.
        # Compare against the latest *auto* row specifically so a human override
        # appended in between cannot mask a genuine automated re-score.
        #
        # The dedup key is the versioned evidence tuple (M14):
        # ``(evidence_sha, labeler_policy_version, reply_evidence_digest, labels,
        # has_posterior, reward_version)``. A policy-version bump, a
        # reward-version bump, or an edited-reply digest change therefore
        # appends a new generation rather than deduping. A
        # ``None`` digest is deliberately NOT coerced to ``""``: ``None`` never
        # equals a present digest, so digest-less legacy callers dedupe on the
        # remaining tuple exactly as before.
        if source == "auto":
            latest_auto = conn.execute(
                "SELECT evidence_sha, labeler_policy_version, reply_evidence_digest, labels, has_posterior, "
                "reward_version "
                "FROM label_observations "
                "WHERE session_id = ? AND source = 'auto' "
                "ORDER BY observed_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if (
                latest_auto is not None
                and latest_auto["evidence_sha"] == evidence_sha
                and latest_auto["labeler_policy_version"] == labeler_policy_version
                and latest_auto["reply_evidence_digest"] == reply_evidence_digest
                and latest_auto["reward_version"] == reward_version
                and latest_auto["labels"] == labels_json
                # Population membership varies independently of the label (a
                # local_branch outcome is labeled but not posterior evidence).
                and latest_auto["has_posterior"] == has_posterior_int
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
                # Primary-key collision on (session_id, observed_at): a
                # byte-identical row already occupying this exact stamp is a
                # no-op re-import — the microsecond bump must never fabricate
                # a duplicate generation for it (idempotent re-merge of a
                # surviving source, human or auto). A genuinely distinct
                # generation at the same stamp keeps the pre-existing bump.
                existing = conn.execute(
                    _SELECT_LABEL_OBSERVATION_ROW_SQL,
                    (session_id, observed_at),
                ).fetchone()
                if existing is not None and tuple(existing) == row_body:
                    return False
                observed_dt += timedelta(microseconds=1)
        # Recompute the winning observation (human-first, then recency) so the
        # denormalized runs cache mirrors the precedence projection — not
        # necessarily the row just inserted (a newer auto must not dethrone a
        # human label).
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
    finally:
        conn.close()


def latest_label_observation(
    archive_dir: Path,
    session_id: str,
    *,
    as_of: str | None = None,
) -> dict[str, Any] | None:
    """Return the highest-precedence (human-first, then most recent) label observation for ``session_id``.

    Human-sourced observations win over automated ones regardless of timing;
    ties broken by recency. When ``as_of`` is provided, the result is the
    highest-precedence observation whose ``observed_at <= as_of`` — enabling
    reproducible corpus pinning.

    Args:
        as_of: Optional ISO 8601 cutoff timestamp in the canonical UTC
            spelling (see :func:`normalize_as_of` — the entry boundary
            normalizes once; this lexical cutoff assumes canonical input).
    """
    cutoff = "AND observed_at <= ? " if as_of is not None else ""
    params: tuple[Any, ...] = (session_id,) if as_of is None else (session_id, as_of)
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            f"SELECT * FROM label_observations WHERE session_id = ? "
            f"{cutoff}ORDER BY {_PRECEDENCE_ORDER} LIMIT 1",
            params,
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()



def reviewer_set_penalty_prior(
    archive_dir: Path,
    logins: list[str],
    *,
    before_valid_at: str,
    exclude_session: str,
    repo_slug: str | None = None,
    readonly: bool = False,
) -> tuple[float | None, int]:
    """Return the pooled mean penalty over prior runs sharing a reviewer (C4).

    Pools ``label_observations`` rows whose ``reviewer_logins`` JSON intersects
    *logins*, restricted to ``session_id != exclude_session`` and
    ``valid_at < before_valid_at`` (strict). When *repo_slug* is provided the
    pool is further restricted to rows whose parent ``runs.repo_slug`` matches —
    preventing cross-repo reviewer history from inflating or deflating the prior
    (C4 per-repo scoping). One outcome is taken per session (latest
    ``observed_at``); its first label is mapped to a false-positive penalty via
    ``_REVIEWER_PENALTY_MAP`` (``accepted→0.0``, ``contested→0.5``,
    ``rejected→1.0``). The raw pooled mean and count are returned — the ``>=10``
    sufficiency threshold and the ``0.5`` default fallback are the caller's
    responsibility.

    Rows with malformed ``reviewer_logins`` / ``labels`` JSON are skipped with a
    :func:`warnings.warn` so a single
    bad row never crashes the aggregate.

    Args:
        logins: The current run's reviewer set. Empty → no pool.
        before_valid_at: ISO 8601 strict upper bound on ``valid_at``.
            Canonicalized via :func:`canonical_utc_iso` so the lexical ``<``
            against the (uniformly canonical) stored column compares
            chronologically regardless of the caller's spelling.
        exclude_session: Session id to exclude (the current run).
        repo_slug: When provided, restrict the pool to observations whose
            parent run shares this ``repo_slug`` (joined via ``runs``).
            ``None`` disables per-repo filtering (backward-compatible).

    Returns:
        ``(mean_penalty, count)`` over the pooled sessions, or ``(None, 0)``
        when *logins* is empty or the pool is empty.
    """
    if not logins:
        return None, 0
    # Canonicalize the bound so the lexical < against the canonical stored
    # column stays chronological regardless of the caller's spelling.
    before_valid_at = canonical_utc_iso(before_valid_at)
    login_set = set(logins)
    penalty_map = _REVIEWER_PENALTY_MAP

    # Build an IN-list so SQLite's json_each() can filter reviewer intersection
    # inside the query, avoiding a full-table fetch followed by Python-side
    # isdisjoint() for every archived row.
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

    conn = readonly_connection(archive_dir) if readonly else _get_connection(archive_dir)
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

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
        if not isinstance(row_logins, list) or login_set.isdisjoint(row_logins):
            continue
        try:
            row_labels = json.loads(row["labels"])
        except (json.JSONDecodeError, TypeError) as exc:
            warnings.warn(f"Invalid labels payload {row['labels']!r}: {exc}", stacklevel=2)
            continue
        if not isinstance(row_labels, list) or not row_labels:
            continue
        penalty = penalty_map.get(str(row_labels[0]))
        if penalty is None:
            continue
        penalties.append(penalty)

    if not penalties:
        return None, 0
    return sum(penalties) / len(penalties), len(penalties)


def label_observation_history(archive_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """Return the full label history for ``session_id`` in chronological order.

    Returns:
        List of row dicts ordered by ``observed_at`` ascending.
    """
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT * FROM label_observations WHERE session_id = ? ORDER BY observed_at ASC",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_labels(archive_dir: Path, session_id: str, labels: list[str]) -> bool:
    """Update outcome labels for a session, supporting prefix matching.

    Thin wrapper around :func:`append_label_observation` that records a
    **human-sourced** observation (``source="human"``, ``labeler_version="human"``).
    Human labels win over automated ones in every precedence projection and are
    never deduped, so this is the authoritative override surface backing
    ``daydream label``. The session_id can be a prefix (e.g. first 8 chars of
    the UUID). If the prefix matches exactly one row, that row is updated. If it
    matches multiple rows, a ValueError is raised asking for a longer prefix.

    Returns:
        True if a row was updated, False if no matching session was found.

    Raises:
        ValueError: If the prefix matches more than one session.
    """
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id LIKE ? || '%'",
            (session_id,),
        )
        matches = cursor.fetchall()
    finally:
        conn.close()

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
    """Backfill the PR linkage columns on a run row.

    Used by harvest to durably record a PR resolved for an orphan run (a run
    launched before its PR existed, so ``pr_number`` was frozen as ``None``).
    Persisting the linkage keeps subsequent harvest passes from re-querying
    GitHub for the same row and makes the resolution auditable.

    This is a pure linkage backfill: it touches only the ``pr_number`` and
    ``pr_repo`` columns on the ``runs`` table and never writes to
    ``label_observations`` or any cache column. A zero-row match (no such
    ``session_id``) is a silent no-op; the caller guarantees the row exists.
    """
    conn = _get_connection(archive_dir)
    try:
        conn.execute(
            "UPDATE runs SET pr_number = ?, pr_repo = ? WHERE session_id = ?",
            (pr_number, pr_repo, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def query_runs(archive_dir: Path, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Query the runs index with an optional WHERE clause.

    Args:
        where: Optional SQL WHERE clause (without the ``WHERE`` keyword).
            Example: ``"repo_slug = ? AND status = ?"``.
        params: Parameter tuple to bind to the WHERE clause placeholders.
    """
    conn = _get_connection(archive_dir)
    try:
        sql = "SELECT * FROM runs"
        if where:
            sql += f" WHERE {where}"  # noqa: S608 - caller-supplied SQL fragment with bound params
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


