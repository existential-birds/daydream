"""Corpus snapshot writer for reproducible label pinning.

A *corpus snapshot* freezes the labeler state of the archive at a moment
in time so downstream training runs can be reproduced bit-for-bit even
as the immutable :mod:`daydream.archive.index` ``label_observations``
history accumulates new rows.

The snapshot JSON written by :func:`run_snapshot` has the following
schema (``schema_version = "1"``):

* ``schema_version`` — pin string, always ``"1"`` for this writer.
* ``archive_index_sha`` — SHA-256 of ``archive_dir/index.db`` at write
  time. Acts as a content-addressable handle to the SQLite snapshot.
* ``as_of_ts`` — ISO 8601 UTC cutoff timestamp passed through to
  :func:`daydream.archive.index.latest_label_observation`. Observations
  recorded *after* this timestamp are ignored.
* ``total_runs`` — count of rows in ``runs`` at write time.
* ``label_counts`` — mapping of label string → count. Runs with no
  observation at-or-before ``as_of_ts`` are tallied under
  ``"unlabeled"``. Multi-label observations contribute their first
  label only (single-element lists are the canonical shape — see
  ``/tmp/research-label-lifecycle.md``).
* ``generated_at`` — wall-clock UTC ISO timestamp when the snapshot was
  written. Distinct from ``as_of_ts``: the latter is the *replay
  cutoff*, the former is the *write moment*.

Failure mode: a missing ``index.db`` propagates a
:class:`FileNotFoundError` to the caller. The writer uses the standard
project atomic-write pattern (tempfile + ``os.replace``) so an
interrupted call leaves the previous snapshot — or nothing — never a
truncated read.

See ``/tmp/research-label-lifecycle.md`` for the rationale behind the
single-label-first convention and the immutable observation history.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daydream.archive.index import label_count_summary, query_runs


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for a single :func:`run_snapshot` invocation.

    Attributes:
        archive_dir: Path to the archive root (containing ``index.db``).
        out_path: Destination JSON file for the snapshot.
        as_of_ts: Optional ISO 8601 cutoff timestamp. When ``None``, the
            current UTC time is used so the snapshot reflects all
            observations recorded up to the write moment.
    """

    archive_dir: Path
    out_path: Path
    as_of_ts: str | None = None


def run_snapshot(config: SnapshotConfig) -> dict[str, Any]:
    """Write a corpus snapshot JSON file and return its in-memory dict.

    Args:
        config: Snapshot parameters; see :class:`SnapshotConfig`.

    Returns:
        The snapshot dict as written to ``config.out_path``.

    Raises:
        FileNotFoundError: When ``config.archive_dir / "index.db"`` does
            not exist. Bubbles up to the caller unmodified.
    """
    index_path = config.archive_dir / "index.db"
    with sqlite3.connect(index_path) as _wal_conn:
        _wal_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with open(index_path, "rb") as fp:
        archive_index_sha = hashlib.sha256(fp.read()).hexdigest()

    as_of_ts = config.as_of_ts or datetime.now(timezone.utc).isoformat()

    runs = query_runs(config.archive_dir)
    label_counts = label_count_summary(config.archive_dir, as_of=as_of_ts)

    snapshot: dict[str, Any] = {
        "schema_version": "1",
        "archive_index_sha": archive_index_sha,
        "as_of_ts": as_of_ts,
        "total_runs": len(runs),
        "label_counts": label_counts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _atomic_write_json(config.out_path, snapshot)
    return snapshot


def _extract_label(observation: dict[str, Any] | None) -> str:
    """Return the canonical label string for one observation row.

    Multi-label observations contribute their first label only. Rows
    with no observation (or empty label list) collapse to ``"unlabeled"``.
    """
    if observation is None:
        return "unlabeled"
    labels_raw = observation.get("labels")
    if not labels_raw:
        return "unlabeled"
    try:
        labels = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
    except json.JSONDecodeError:
        return "unlabeled"
    if isinstance(labels, list) and labels:
        first = labels[0]
        return str(first) if first else "unlabeled"
    return "unlabeled"


def _atomic_write_json(out_path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``out_path`` atomically (tempfile + replace)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
